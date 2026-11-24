"""
Bio-Steer Labelled Sensor Data Collector

Collects live FSR sensor readings from the ESP32 via backend/sensor_receiver.py
at approximately 5 samples per second and writes records to data/labelled_sensor_data.csv.

Supported collection modes:
- Legacy continuous collection for a selected grip label.
- Legacy single-type guided event trials.
- Guided sessions with repeated steady, firm, sudden-clench, and release trials.

Guided labels follow the instructed phase and its collector-side timing. They are
never inferred from pressure values. Sensor timestamps are local packet receipt
times because the current receiver does not provide ESP32 timestamps.
"""

import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Dict, List, Optional, Set
import uuid

# Ensure project root is available in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.sensor_receiver import get_sensor_data, SensorReceiverError

# Configuration constants
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_CSV = DATA_DIR / "labelled_sensor_data.csv"
DEFAULT_METADATA_CSV = DATA_DIR / "labelled_sensor_trials_metadata.csv"
CSV_HEADER = ["timestamp", "left_fsr", "right_fsr", "grip_label"]
TRIAL_METADATA_HEADER = [
    "session_id",
    "trial_id",
    "trial_type",
    "completion_status",
    "trial_started_at",
    "trial_ended_at",
    "event_start_at",
    "event_end_at",
    "sample_count",
    "phase_sample_counts_json",
    "phase_timings_json",
    "sensor_error_count",
    "invalid_reading_count",
    "unexpected_gap_count",
    "late_packet_count",
    "empty_phase_count",
    "runtime_error_count",
    "both_zero_sample_count",
    "saturated_sample_count",
    "max_sampling_gap_s",
    "warning_codes_json",
    "abort_reason",
]

SUSTAINED_LABELS = ["no_grip", "steady_grip", "firm_grip"]
EVENT_LABELS = ["sudden_clench", "release"]
VALID_GRIP_LABELS = SUSTAINED_LABELS + EVENT_LABELS

SAMPLE_RATE_HZ = 5.0
SAMPLE_INTERVAL = 1.0 / SAMPLE_RATE_HZ  # 0.2 seconds
RETRY_DELAY = 1.0  # seconds to wait after a connection failure

# Guided trial phase definitions: (phase_name, duration_seconds, label_to_assign, terminal_cue)
EVENT_TRIAL_CONFIG: Dict[str, List[Dict]] = {
    "sudden_clench": [
        {
            "phase": "Pre-Event Baseline",
            "duration": 3.0,
            "label": "steady_grip",
            "cue": "[PRE-EVENT] Maintain steady grip baseline...",
        },
        {
            "phase": "Event Window",
            "duration": 1.5,
            "label": "sudden_clench",
            "cue": "\a>>> [CLENCH NOW! SUDDEN STRONG PRESSURE] <<<",
        },
        {
            "phase": "Post-Event Stabilization",
            "duration": 2.5,
            "label": "firm_grip",
            "cue": "[POST-EVENT] Hold firm grip steady...",
        },
    ],
    "release": [
        {
            "phase": "Pre-Event Baseline",
            "duration": 3.0,
            "label": "firm_grip",
            "cue": "[PRE-EVENT] Maintain firm grip baseline...",
        },
        {
            "phase": "Event Window",
            "duration": 1.5,
            "label": "release",
            "cue": "\a>>> [RELEASE NOW! QUICKLY LET GO] <<<",
        },
        {
            "phase": "Post-Event Resting",
            "duration": 2.5,
            "label": "no_grip",
            "cue": "[POST-EVENT] Hands completely off sensors...",
        },
    ],
}

GUIDED_TRIAL_PHASES: Dict[str, List[Dict]] = {
    "steady_grip": [
        {
            "phase": "Steady Grip",
            "duration": 8.0,
            "label": "steady_grip",
            "cue": "\a[STEADY GRIP] Maintain comfortable, consistent pressure.",
        }
    ],
    "firm_grip": [
        {
            "phase": "Baseline",
            "duration": 3.0,
            "label": "steady_grip",
            "cue": "[BASELINE] Maintain a comfortable steady grip.",
        },
        {
            "phase": "Firm Grip",
            "duration": 4.0,
            "label": "firm_grip",
            "cue": "\a[FIRM GRIP] Increase to a firm, controlled pressure and hold.",
        },
        {
            "phase": "Recovery",
            "duration": 2.5,
            "label": "steady_grip",
            "cue": "\a[RECOVERY] Return to a comfortable steady grip.",
        },
    ],
    "sudden_clench": [
        {
            "phase": "Baseline",
            "duration": 3.0,
            "label": "steady_grip",
            "cue": "[BASELINE] Maintain a comfortable steady grip.",
        },
        {
            "phase": "Clench Event",
            "duration": 1.5,
            "label": "sudden_clench",
            "cue": "\a>>> EVENT START: Briefly increase grip pressure in a controlled clench. <<<",
            "event": True,
        },
        {
            "phase": "Recovery",
            "duration": 2.5,
            "label": "steady_grip",
            "cue": "[RECOVERY] Return to a comfortable steady grip.",
        },
    ],
    "release": [
        {
            "phase": "Baseline",
            "duration": 3.0,
            "label": "firm_grip",
            "cue": "[BASELINE] Maintain a firm, controlled grip.",
        },
        {
            "phase": "Release Event",
            "duration": 1.5,
            "label": "release",
            "cue": "\a>>> EVENT START: Release pressure from the sensors. <<<",
            "event": True,
        },
        {
            "phase": "Recovery",
            "duration": 2.5,
            "label": "no_grip",
            "cue": "[RECOVERY] Keep hands off the sensors.",
        },
    ],
}


def init_csv_file(filepath: Path) -> None:
    """
    Ensure the output directory and CSV file exist with the appropriate header.
    Never overwrites or deletes existing data.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    if not filepath.exists() or filepath.stat().st_size == 0:
        with open(filepath, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADER)
        return

    with open(filepath, mode="r", newline="", encoding="utf-8") as f:
        existing_header = next(csv.reader(f), [])
    if existing_header != CSV_HEADER:
        raise ValueError(
            f"Existing raw CSV header does not match expected schema: {existing_header}"
        )


def init_metadata_file(filepath: Path) -> None:
    """Create the separate trial metadata CSV or verify its append schema."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    if not filepath.exists() or filepath.stat().st_size == 0:
        with open(filepath, mode="w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(TRIAL_METADATA_HEADER)
        return

    with open(filepath, mode="r", newline="", encoding="utf-8") as f:
        existing_header = next(csv.reader(f), [])
    if existing_header != TRIAL_METADATA_HEADER:
        raise ValueError(
            f"Existing trial metadata header does not match expected schema: "
            f"{existing_header}"
        )


def _validate_guided_reading(data: Dict) -> Optional[str]:
    """Return a validation warning for readings that must not be appended."""
    if not isinstance(data, dict):
        return "sensor packet is not an object"
    for key in ("left_fsr", "right_fsr"):
        value = data.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return f"{key} is not numeric"
        if not math.isfinite(float(value)) or value < 0 or value > 4095:
            return f"{key} is outside the valid ADC range 0..4095"
    return None


def _trial_completion_status(
    aborted: bool,
    sensor_error_count: int,
    invalid_reading_count: int,
    unexpected_gap_count: int,
    late_packet_count: int,
    empty_phase_count: int,
    runtime_error_count: int,
    both_zero_sample_count: int,
    saturated_sample_count: int,
) -> str:
    """Derive explicit trial status without changing or inferring any labels."""
    if aborted:
        return "aborted"
    if (
        sensor_error_count
        or invalid_reading_count
        or unexpected_gap_count
        or late_packet_count
        or empty_phase_count
        or runtime_error_count
    ):
        return "incomplete"
    if both_zero_sample_count or saturated_sample_count:
        return "completed_with_warnings"
    return "completed"


def _append_trial_metadata(filepath: Path, metadata: Dict) -> None:
    with open(filepath, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TRIAL_METADATA_HEADER)
        writer.writerow(metadata)
        f.flush()


def collect_continuous_data(
    label: str,
    duration: Optional[float] = None,
    output_path: Path = DEFAULT_OUTPUT_CSV,
    sample_rate_hz: float = SAMPLE_RATE_HZ,
) -> int:
    """
    Continuous collection mode for sustained grip states: no_grip, steady_grip, firm_grip.
    """
    sample_interval = 1.0 / sample_rate_hz
    init_csv_file(output_path)

    print("=" * 65)
    print("Bio-Steer Continuous FSR Data Collector")
    print(f"Grip Label        : {label} (Sustained State)")
    print(f"Target Sample Rate: {sample_rate_hz} Hz (~{sample_interval}s interval)")
    print(f"Duration          : {f'{duration} seconds' if duration else 'Continuous (Ctrl+C to stop)'}")
    print(f"Output CSV        : {output_path}")
    print("=" * 65)

    session_samples = 0
    start_time = time.time()

    try:
        with open(output_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            while True:
                now = time.time()
                elapsed_total = now - start_time
                if duration is not None and elapsed_total >= duration:
                    print(f"\nTarget duration of {duration}s reached.")
                    break

                loop_start = time.time()

                try:
                    data = get_sensor_data(raise_on_error=True)
                except SensorReceiverError as err:
                    print(f"[{datetime.now().isoformat()}] ESP32 warning: {err}. Retrying in {RETRY_DELAY}s...")
                    time.sleep(RETRY_DELAY)
                    continue

                if data is not None:
                    timestamp = datetime.now().isoformat()
                    left_val = data["left_fsr"]
                    right_val = data["right_fsr"]

                    writer.writerow([timestamp, left_val, right_val, label])
                    f.flush()
                    session_samples += 1

                    time_info = (
                        f"Remaining: {max(0.0, duration - elapsed_total):4.1f}s"
                        if duration
                        else f"Elapsed: {elapsed_total:5.1f}s"
                    )
                    print(
                        f"[{timestamp}] #{session_samples:<4} ({time_info}) | "
                        f"Left: {left_val:<5} | Right: {right_val:<5} | Label: {label}"
                    )

                elapsed_loop = time.time() - loop_start
                sleep_duration = max(0.0, sample_interval - elapsed_loop)
                time.sleep(sleep_duration)

    except KeyboardInterrupt:
        print("\nContinuous collection interrupted by user (Ctrl+C).")

    print("=" * 65)
    print("Collection Session Finished")
    print(f"Grip label recorded   : {label}")
    print(f"Successfully collected: {session_samples} samples")
    print(f"Appended to           : {output_path}")
    print("=" * 65)

    return session_samples


def collect_event_trials(
    event_label: str,
    trials: int = 5,
    output_path: Path = DEFAULT_OUTPUT_CSV,
    sample_rate_hz: float = SAMPLE_RATE_HZ,
) -> int:
    """
    Guided event trial collector for transient events: sudden_clench, release.
    Each trial executes a pre-event baseline, an active event window, and a post-event stabilization.
    """
    phases = EVENT_TRIAL_CONFIG[event_label]
    sample_interval = 1.0 / sample_rate_hz
    init_csv_file(output_path)

    total_trial_duration = sum(p["duration"] for p in phases)

    print("=" * 70)
    print("Bio-Steer Guided Event Trial Collector")
    print(f"Target Event        : {event_label}")
    print(f"Total Trials        : {trials}")
    print(f"Duration per Trial  : {total_trial_duration:.1f}s")
    for idx, p in enumerate(phases, 1):
        print(f"  Phase {idx}: {p['phase']:<25} ({p['duration']}s) -> Label: '{p['label']}'")
    print(f"Target Sample Rate  : {sample_rate_hz} Hz (~{sample_interval}s interval)")
    print(f"Output CSV          : {output_path}")
    print("Note: Event windows are initial guided labeling intervals.")
    print("=" * 70)

    total_samples = 0
    samples_by_label: Dict[str, int] = {p["label"]: 0 for p in phases}

    try:
        with open(output_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            for trial_num in range(1, trials + 1):
                print(f"\n>>> PREPARING TRIAL {trial_num} OF {trials} <<<")
                # Inter-trial countdown
                for cd in [3, 2, 1]:
                    print(f"Starting in {cd}...", end="\r", flush=True)
                    time.sleep(1.0)
                print(" " * 30, end="\r")

                print(f"--- TRIAL {trial_num}/{trials} IN PROGRESS ---")

                for phase_info in phases:
                    phase_name = phase_info["phase"]
                    phase_duration = phase_info["duration"]
                    phase_label = phase_info["label"]
                    phase_cue = phase_info["cue"]

                    print(f"\n{phase_cue}")

                    phase_start = time.time()

                    while True:
                        now = time.time()
                        elapsed_phase = now - phase_start
                        if elapsed_phase >= phase_duration:
                            break

                        loop_start = time.time()

                        try:
                            data = get_sensor_data(raise_on_error=True)
                        except SensorReceiverError as err:
                            print(f"[{datetime.now().isoformat()}] ESP32 warning: {err}. Retrying in {RETRY_DELAY}s...")
                            time.sleep(RETRY_DELAY)
                            continue

                        if data is not None:
                            timestamp = datetime.now().isoformat()
                            left_val = data["left_fsr"]
                            right_val = data["right_fsr"]

                            writer.writerow([timestamp, left_val, right_val, phase_label])
                            f.flush()
                            total_samples += 1
                            samples_by_label[phase_label] = samples_by_label.get(phase_label, 0) + 1

                            rem = max(0.0, phase_duration - elapsed_phase)
                            print(
                                f"[{timestamp}] T{trial_num} {phase_name[:8]} ({rem:3.1f}s) | "
                                f"L: {left_val:<5} | R: {right_val:<5} | [{phase_label}]"
                            )

                        elapsed_loop = time.time() - loop_start
                        sleep_duration = max(0.0, sample_interval - elapsed_loop)
                        time.sleep(sleep_duration)

                print(f"\nTrial {trial_num} complete.")

    except KeyboardInterrupt:
        print("\n\nEvent trial collection interrupted by user (Ctrl+C).")

    print("\n" + "=" * 70)
    print(f"Event Trial Session Finished for: '{event_label}'")
    print(f"Total samples collected : {total_samples}")
    print("Breakdown by label:")
    for lbl, count in samples_by_label.items():
        print(f"  - {lbl:<15}: {count} samples")
    print(f"Appended to             : {output_path}")
    print("=" * 70)

    return total_samples


def collect_guided_session(
    trials_per_type: int = 5,
    output_path: Path = DEFAULT_OUTPUT_CSV,
    metadata_path: Path = DEFAULT_METADATA_CSV,
    sample_rate_hz: float = SAMPLE_RATE_HZ,
) -> int:
    """
    Collect repeated, phase-labeled trials for all four guided trial types.

    Sensor samples are appended only when received and validated. Trial status
    and phase timing are recorded separately from the fixed raw CSV schema.
    """
    if trials_per_type < 1:
        raise ValueError("trials_per_type must be at least 1.")
    if not math.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be a positive finite number.")
    if output_path.resolve() == metadata_path.resolve():
        raise ValueError("Raw sample output and trial metadata paths must be different.")

    sample_interval = 1.0 / sample_rate_hz
    gap_warning_threshold = 2.0 * sample_interval
    trial_plan = [
        trial_type
        for _ in range(trials_per_type)
        for trial_type in GUIDED_TRIAL_PHASES
    ]
    random.SystemRandom().shuffle(trial_plan)

    print("=" * 76)
    print("BIO-STEER GUIDED COLLECTION SESSION")
    print(f"Trial types          : {', '.join(GUIDED_TRIAL_PHASES)}")
    print(f"Trials per type      : {trials_per_type}")
    print(f"Total trials         : {len(trial_plan)}")
    print(f"Target sample rate   : {sample_rate_hz:g} Hz")
    print(f"Gap warning threshold: > {gap_warning_threshold:.3f}s")
    print(f"Raw samples          : {output_path}")
    print(f"Trial metadata       : {metadata_path}")
    print("Press Ctrl+C during a trial to abort it safely.")
    print("A bell cue depends on terminal bell/audio support.")
    print("=" * 76)

    try:
        consent = input("Press Enter to begin, or type q to cancel: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\nGuided session cancelled before collection began.")
        return 0
    if consent == "q":
        print("Guided session cancelled before collection began.")
        return 0

    init_csv_file(output_path)
    init_metadata_file(metadata_path)

    session_id = str(uuid.uuid4())
    total_samples = 0
    stop_session = False

    with open(output_path, mode="a", newline="", encoding="utf-8") as raw_file:
        raw_writer = csv.writer(raw_file)

        for trial_index, trial_type in enumerate(trial_plan, 1):
            try:
                choice = input(
                    f"\nTrial {trial_index}/{len(trial_plan)} ({trial_type}). "
                    "Press Enter to continue or q to stop before this trial: "
                ).strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\nSession stopped before the next trial; no trial record was started.")
                break
            if choice == "q":
                print("Session stopped before the next trial.")
                break

            trial_id = str(uuid.uuid4())
            trial_started_at = datetime.now().isoformat()
            record = {
                "session_id": session_id,
                "trial_id": trial_id,
                "trial_type": trial_type,
                "completion_status": "incomplete",
                "trial_started_at": trial_started_at,
                "trial_ended_at": "",
                "event_start_at": "",
                "event_end_at": "",
                "sample_count": 0,
                "phase_sample_counts_json": "{}",
                "phase_timings_json": "[]",
                "sensor_error_count": 0,
                "invalid_reading_count": 0,
                "unexpected_gap_count": 0,
                "late_packet_count": 0,
                "empty_phase_count": 0,
                "runtime_error_count": 0,
                "both_zero_sample_count": 0,
                "saturated_sample_count": 0,
                "max_sampling_gap_s": "",
                "warning_codes_json": "[]",
                "abort_reason": "",
            }
            phase_sample_counts: Dict[str, int] = {}
            phase_timings: List[Dict] = []
            warning_codes: Set[str] = set()
            saturation_warnings: Set[str] = set()
            previous_was_both_zero = False
            last_packet_monotonic: Optional[float] = None
            max_sampling_gap = 0.0
            trial_aborted = False
            trial_finished_normally = False

            print(f"\n>>> TRIAL {trial_index}/{len(trial_plan)}: {trial_type} <<<")
            try:
                for countdown in (3, 2, 1):
                    print(f"Starting in {countdown}...", end="\r", flush=True)
                    time.sleep(1.0)
                print(" " * 32, end="\r")

                for phase_info in GUIDED_TRIAL_PHASES[trial_type]:
                    phase_name = phase_info["phase"]
                    phase_label = phase_info["label"]
                    phase_duration = float(phase_info["duration"])
                    phase_samples = 0
                    phase_sample_counts[phase_name] = 0
                    print(f"\n{phase_info['cue']}")
                    phase_start_at = datetime.now().isoformat()
                    phase_start_monotonic = time.monotonic()
                    if phase_info.get("event"):
                        record["event_start_at"] = phase_start_at

                    while time.monotonic() - phase_start_monotonic < phase_duration:
                        loop_start = time.monotonic()
                        try:
                            data = get_sensor_data(raise_on_error=True)
                        except SensorReceiverError as err:
                            record["sensor_error_count"] += 1
                            warning_codes.add("missing_or_invalid_sensor_packet")
                            print(
                                f"[{datetime.now().isoformat()}] WARNING: Missing or "
                                f"invalid sensor packet: {err}. Trial will be incomplete."
                            )
                            remaining = max(
                                0.0,
                                phase_duration
                                - (time.monotonic() - phase_start_monotonic),
                            )
                            time.sleep(min(RETRY_DELAY, remaining))
                            continue

                        received_monotonic = time.monotonic()
                        timestamp = datetime.now().isoformat()
                        if data is None:
                            record["sensor_error_count"] += 1
                            warning_codes.add("missing_or_invalid_sensor_packet")
                            print(
                                f"[{timestamp}] WARNING: Sensor receiver returned no "
                                "packet. Trial will be incomplete."
                            )
                            continue

                        invalid_reason = _validate_guided_reading(data)
                        if invalid_reason:
                            record["invalid_reading_count"] += 1
                            warning_codes.add("invalid_sensor_reading")
                            print(
                                f"[{timestamp}] WARNING: Invalid sensor packet "
                                f"({invalid_reason}); packet not appended."
                            )
                            continue

                        packet_elapsed = received_monotonic - phase_start_monotonic
                        if packet_elapsed >= phase_duration:
                            record["late_packet_count"] += 1
                            warning_codes.add("packet_received_after_phase_window")
                            print(
                                f"[{timestamp}] WARNING: Packet arrived after the "
                                f"{phase_name} phase window; it was not assigned a "
                                "phase label or appended."
                            )
                            break

                        if last_packet_monotonic is not None:
                            sampling_gap = received_monotonic - last_packet_monotonic
                            max_sampling_gap = max(max_sampling_gap, sampling_gap)
                            if sampling_gap > gap_warning_threshold:
                                record["unexpected_gap_count"] += 1
                                warning_codes.add("unexpected_sampling_gap")
                                print(
                                    f"[{timestamp}] WARNING: Sampling gap "
                                    f"{sampling_gap:.3f}s exceeds "
                                    f"{gap_warning_threshold:.3f}s; trial will be "
                                    "incomplete."
                                )
                        last_packet_monotonic = received_monotonic

                        left_value = data["left_fsr"]
                        right_value = data["right_fsr"]
                        if left_value == 0 and right_value == 0:
                            record["both_zero_sample_count"] += 1
                            warning_codes.add("both_sensors_zero")
                            if not previous_was_both_zero:
                                print(
                                    f"[{timestamp}] WARNING: Both sensors read zero. "
                                    "This may be expected during no-grip recovery; "
                                    "check sensor contact if unexpected."
                                )
                            previous_was_both_zero = True
                        else:
                            previous_was_both_zero = False

                        saturated_sensors = [
                            sensor_name
                            for sensor_name, value in (
                                ("left_fsr", left_value),
                                ("right_fsr", right_value),
                            )
                            if value == 4095
                        ]
                        if saturated_sensors:
                            record["saturated_sample_count"] += 1
                            warning_codes.add("adc_saturation")
                            for sensor_name in saturated_sensors:
                                if sensor_name not in saturation_warnings:
                                    print(
                                        f"[{timestamp}] WARNING: {sensor_name} reached "
                                        "4095 (ADC saturation); reading is retained "
                                        "unchanged."
                                    )
                                    saturation_warnings.add(sensor_name)

                        raw_writer.writerow(
                            [timestamp, left_value, right_value, phase_label]
                        )
                        raw_file.flush()
                        total_samples += 1
                        record["sample_count"] += 1
                        phase_samples += 1
                        phase_sample_counts[phase_name] = phase_samples
                        print(
                            f"[{timestamp}] {phase_name:<16} "
                            f"L:{left_value!s:<6} R:{right_value!s:<6} "
                            f"Label:{phase_label}"
                        )

                        sleep_duration = max(
                            0.0,
                            sample_interval - (time.monotonic() - loop_start),
                        )
                        time.sleep(sleep_duration)

                    phase_end_at = datetime.now().isoformat()
                    phase_elapsed = time.monotonic() - phase_start_monotonic
                    phase_timings.append(
                        {
                            "phase": phase_name,
                            "label": phase_label,
                            "planned_duration_s": phase_duration,
                            "actual_duration_s": round(phase_elapsed, 6),
                            "start_at": phase_start_at,
                            "end_at": phase_end_at,
                            "sample_count": phase_samples,
                        }
                    )
                    if phase_samples == 0:
                        record["empty_phase_count"] += 1
                        warning_codes.add("phase_without_samples")
                        print(
                            f"WARNING: No valid samples were collected during "
                            f"{phase_name}; trial will be incomplete."
                        )
                    if phase_info.get("event"):
                        record["event_end_at"] = phase_end_at
                        print("\a[EVENT COMPLETE] Event window ended.")

                trial_finished_normally = True
            except KeyboardInterrupt:
                trial_aborted = True
                record["abort_reason"] = "keyboard_interrupt"
                warning_codes.add("user_aborted_trial")
                print("\nTrial aborted by user (Ctrl+C).")
            except Exception as err:
                record["runtime_error_count"] += 1
                record["abort_reason"] = f"{type(err).__name__}: {err}"
                warning_codes.add("collector_runtime_error")
                print(
                    f"\nERROR: Trial collection failed: {type(err).__name__}: {err}"
                )
                raise
            finally:
                record["trial_ended_at"] = datetime.now().isoformat()
                record["phase_sample_counts_json"] = json.dumps(
                    phase_sample_counts, sort_keys=True
                )
                record["phase_timings_json"] = json.dumps(
                    phase_timings, sort_keys=True
                )
                record["max_sampling_gap_s"] = (
                    round(max_sampling_gap, 6) if last_packet_monotonic is not None else ""
                )
                record["warning_codes_json"] = json.dumps(sorted(warning_codes))
                record["completion_status"] = _trial_completion_status(
                    aborted=trial_aborted,
                    sensor_error_count=record["sensor_error_count"],
                    invalid_reading_count=record["invalid_reading_count"],
                    unexpected_gap_count=record["unexpected_gap_count"],
                    late_packet_count=record["late_packet_count"],
                    empty_phase_count=record["empty_phase_count"],
                    runtime_error_count=record["runtime_error_count"],
                    both_zero_sample_count=record["both_zero_sample_count"],
                    saturated_sample_count=record["saturated_sample_count"],
                )
                _append_trial_metadata(metadata_path, record)

            print(
                f"Trial {trial_id} status: {record['completion_status']} "
                f"({record['sample_count']} samples)."
            )
            if not trial_finished_normally:
                stop_session = True
                break

    print("\n" + "=" * 76)
    print("GUIDED COLLECTION SESSION FINISHED")
    print(f"Session ID          : {session_id}")
    print(f"Samples appended    : {total_samples}")
    print(f"Raw CSV             : {output_path}")
    print(f"Trial metadata CSV  : {metadata_path}")
    if stop_session:
        print("Session stopped after an aborted or failed trial.")
    print("=" * 76)
    return total_samples


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bio-Steer Labelled FSR Data Collector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Permitted grip labels:\n"
            "  Sustained States (Continuous Mode):\n"
            "    no_grip        Neither sensor is being pressed\n"
            "    steady_grip    Consistent baseline pressure on sensors\n"
            "    firm_grip      Intentionally stronger pressure\n\n"
            "  Legacy Event Mode:\n"
            "    sudden_clench / release  Use --label with --trials\n\n"
            "  Guided Session Mode:\n"
            "    --guided-session collects repeated trials of steady_grip, firm_grip,\n"
            "    sudden_clench, and release with separate trial metadata.\n"
        ),
    )
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--label",
        type=str,
        choices=VALID_GRIP_LABELS,
        help="Legacy collection mode: grip pattern category to record.",
    )
    mode_group.add_argument(
        "--guided-session",
        action="store_true",
        help="Run repeated, prompted trials for all four guided trial types.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Duration in seconds for continuous mode (sustained labels). If omitted, runs until Ctrl+C.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=5,
        help="Number of guided trials for event mode (sudden_clench, release). Default is 5.",
    )
    parser.add_argument(
        "--trials-per-type",
        type=int,
        default=5,
        help="Guided-session repetitions for each of the four trial types (default: 5).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help=f"Path to output CSV file (default: {DEFAULT_OUTPUT_CSV}).",
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=DEFAULT_METADATA_CSV,
        help=f"Guided trial metadata CSV (default: {DEFAULT_METADATA_CSV}).",
    )

    args = parser.parse_args()

    if args.guided_session:
        if args.trials_per_type < 1:
            parser.error("--trials-per-type must be at least 1.")
        collect_guided_session(
            trials_per_type=args.trials_per_type,
            output_path=args.output,
            metadata_path=args.metadata_output,
        )
    elif args.label in EVENT_LABELS:
        if args.trials < 1:
            parser.error("--trials must be at least 1.")
        collect_event_trials(
            event_label=args.label,
            trials=args.trials,
            output_path=args.output,
        )
    else:
        if args.duration is not None and args.duration <= 0:
            parser.error("--duration must be positive.")
        collect_continuous_data(
            label=args.label,
            duration=args.duration,
            output_path=args.output,
        )


if __name__ == "__main__":
    main()
