"""
Bio-Steer Live Sensor Data Collector

Continuously polls real FSR sensor readings from the ESP32 via backend/sensor_receiver.py
at approximately 5 samples per second, timestamps them, and appends the records
to data/sensor_data.csv.
"""

import csv
from datetime import datetime
from pathlib import Path
import sys
import time

# Ensure project root is available in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.sensor_receiver import get_sensor_data, SensorReceiverError

# Configuration constants
DATA_DIR = PROJECT_ROOT / "data"
CSV_FILE = DATA_DIR / "sensor_data.csv"
CSV_HEADER = ["timestamp", "left_fsr", "right_fsr"]
SAMPLE_RATE_HZ = 5.0
SAMPLE_INTERVAL = 1.0 / SAMPLE_RATE_HZ  # 0.2 seconds
RETRY_DELAY = 1.0  # seconds to wait after a connection failure


def init_csv_file(filepath: Path) -> None:
    """
    Ensure the data directory and CSV file exist with the proper header.
    Never overwrites existing records.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    if not filepath.exists() or filepath.stat().st_size == 0:
        with open(filepath, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADER)


def collect_sensor_data(
    filepath: Path = CSV_FILE,
    sample_rate_hz: float = SAMPLE_RATE_HZ
) -> None:
    """
    Continuously requests real sensor readings, displays them, and appends
    valid samples to the CSV file. Stops cleanly on Ctrl+C.
    """
    sample_interval = 1.0 / sample_rate_hz
    init_csv_file(filepath)

    print("=" * 60)
    print("Bio-Steer Live FSR Data Collector")
    print(f"Target sample rate : {sample_rate_hz} samples/sec (~{sample_interval}s interval)")
    print(f"Output CSV file    : {filepath}")
    print("Press Ctrl+C to stop collection cleanly.")
    print("=" * 60)

    session_samples = 0

    try:
        with open(filepath, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            while True:
                loop_start = time.time()

                try:
                    data = get_sensor_data(raise_on_error=True)
                except SensorReceiverError as err:
                    # Handle temporary ESP32 connection failures without crashing
                    print(f"[{datetime.now().isoformat()}] ESP32 warning: {err}. Retrying in {RETRY_DELAY}s...")
                    time.sleep(RETRY_DELAY)
                    continue

                if data is not None:
                    timestamp = datetime.now().isoformat()
                    left_val = data["left_fsr"]
                    right_val = data["right_fsr"]

                    # Append reading to CSV and flush immediately to preserve data
                    writer.writerow([timestamp, left_val, right_val])
                    f.flush()
                    session_samples += 1

                    # Display current readings in terminal
                    print(
                        f"[{timestamp}] Sample #{session_samples:<5} | "
                        f"Left FSR: {left_val:<5} | Right FSR: {right_val:<5}"
                    )

                # Maintain approximately 5 samples per second
                elapsed = time.time() - loop_start
                sleep_duration = max(0.0, sample_interval - elapsed)
                time.sleep(sleep_duration)

    except KeyboardInterrupt:
        print("\n" + "=" * 60)
        print("Collection stopped cleanly by user (Ctrl+C).")
        print(f"Total samples collected this session: {session_samples}")
        print(f"Saved to: {filepath}")
        print("=" * 60)


if __name__ == "__main__":
    collect_sensor_data()
