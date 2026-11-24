"""
Bio-Steer Dataset Preparation & Temporal Feature Engineering Pipeline

Loads raw labelled FSR sensor telemetry from data/labelled_sensor_data.csv,
excludes the five verified corrupted/mislabeled record groups, detects recording segment and
collection session boundaries, computes localized rolling temporal features
without crossing segment boundaries, validates feature integrity (no NaNs, no Infinities),
and saves the prepared dataset to data/prepared_sensor_data.csv.

Dual-Grouping Architecture:
- segment_id : Assigned for timestamp gaps > 1.0s (controls localized feature resets).
- session_id : Assigned for timestamp gaps > 5.0s (controls macro collection session grouping
               and held-out cross-validation splits).
"""

from datetime import datetime
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_FILE = DATA_DIR / "labelled_sensor_data.csv"
PREPARED_DATA_FILE = DATA_DIR / "prepared_sensor_data.csv"

# Configuration constants
ROLLING_WINDOW_SAMPLES = 5  # ~1.0 second rolling window at ~5 Hz sampling rate
SEGMENT_GAP_THRESHOLD_S = 1.0  # seconds between samples to define segment boundary (feature resets)
SESSION_GAP_THRESHOLD_S = 5.0  # seconds between samples to define collection session boundary (validation splits)

# Exact exclusion ranges and conditions
EXCLUSION_A_START_TS = "2026-11-24T22:52:46.709699"
EXCLUSION_A_END_TS = "2026-11-24T22:52:48.738503"
EXCLUSION_B_START_TS = "2026-11-24T23:05:48.554831"
EXCLUSION_B_END_TS = "2026-11-24T23:09:15.716387"
EXCLUSION_C_START_TS = "2026-11-24T23:39:47.615875"
EXCLUSION_C_END_TS = "2026-11-24T23:39:53.575635"
EXCLUSION_D_START_TS = "2026-11-24T23:11:02.198042"
EXCLUSION_D_END_TS = "2026-11-24T23:12:11.832177"
EXCLUSION_E_START_TS = "2026-11-24T23:13:05.469594"
EXCLUSION_E_END_TS = "2026-11-24T23:13:12.874031"

VALID_CLASSES = ["no_grip", "steady_grip", "firm_grip", "sudden_clench", "release"]


def load_raw_dataset(filepath: Path = RAW_DATA_FILE) -> pd.DataFrame:
    """
    Load and perform initial schema validation on the raw labelled sensor dataset.
    Never modifies the underlying CSV file.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Raw dataset file not found at: {filepath}")

    df = pd.read_csv(filepath)
    required_columns = {"timestamp", "left_fsr", "right_fsr", "grip_label"}
    missing_cols = required_columns - set(df.columns)
    if missing_cols:
        raise ValueError(f"Raw CSV missing required columns: {missing_cols}")

    if df.empty:
        raise ValueError("Raw CSV file is empty.")

    return df


def exclude_corrupted_record_groups(
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Identifies and excludes the five verified record groups using exact
    timestamp ranges and no row-index or end-of-file assumptions.
    """
    exclusion_masks = {
        "A": (
            (df["timestamp"] >= EXCLUSION_A_START_TS)
            & (df["timestamp"] <= EXCLUSION_A_END_TS)
            & (df["left_fsr"] == 0)
            & (df["right_fsr"] == 0)
            & (df["grip_label"] == "steady_grip")
        ),
        "B": (
            (df["timestamp"] >= EXCLUSION_B_START_TS)
            & (df["timestamp"] <= EXCLUSION_B_END_TS)
        ),
        "C": (
            (df["timestamp"] >= EXCLUSION_C_START_TS)
            & (df["timestamp"] <= EXCLUSION_C_END_TS)
        ),
        "D": (
            (df["timestamp"] >= EXCLUSION_D_START_TS)
            & (df["timestamp"] <= EXCLUSION_D_END_TS)
        ),
        "E": (
            (df["timestamp"] >= EXCLUSION_E_START_TS)
            & (df["timestamp"] <= EXCLUSION_E_END_TS)
        ),
    }

    overlaps = np.logical_or.reduce(
        [exclusion_masks["A"].to_numpy(), exclusion_masks["B"].to_numpy(),
         exclusion_masks["C"].to_numpy(), exclusion_masks["D"].to_numpy(),
         exclusion_masks["E"].to_numpy()]
    )
    exclusion_counts = {name: int(mask.sum()) for name, mask in exclusion_masks.items()}

    pairwise_overlap = sum(
        int((exclusion_masks[left] & exclusion_masks[right]).sum())
        for index, left in enumerate(exclusion_masks)
        for right in list(exclusion_masks)[index + 1:]
    )
    if pairwise_overlap != 0:
        raise ValueError(f"Exclusion masks overlap in {pairwise_overlap} raw rows.")

    for name, expected in {"A": 8, "B": 538, "C": 21, "D": 208, "E": 29}.items():
        if exclusion_counts[name] != expected:
            raise ValueError(
                f"Exclusion group {name}: expected {expected} matching raw rows, "
                f"found {exclusion_counts[name]}."
            )

    excluded_df = df[overlaps].copy()
    cleaned_df = df[~np.asarray(overlaps)].copy()

    if len(excluded_df) != 804:
        raise ValueError(
            f"Expected exactly 804 excluded raw rows, found {len(excluded_df)}."
        )

    print("Exclusion counts:")
    for name in ["A", "B", "C", "D", "E"]:
        print(f"  Group {name}: {exclusion_counts[name]} rows")
    print(f"  Total excluded: {len(excluded_df)} rows")

    return cleaned_df, excluded_df


def identify_recording_segments(
    df: pd.DataFrame,
    gap_threshold_s: float = SEGMENT_GAP_THRESHOLD_S,
    session_gap_threshold_s: float = SESSION_GAP_THRESHOLD_S,
) -> pd.DataFrame:
    """
    Assigns:
    1. segment_id: Increments whenever time interval exceeds gap_threshold_s (> 1.0s).
       Used strictly to control temporal feature resets and avoid window leakage across pauses.
    2. session_id: Increments whenever time interval exceeds session_gap_threshold_s (> 5.0s).
       Calculated chronologically from actual timestamps.
       Keeps all trials of an original recording session under one session_id for held-out validation.
    """
    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True).copy()
    df["dt"] = pd.to_datetime(df["timestamp"])

    # Calculate delta t in seconds between adjacent records
    delta_t = df["dt"].diff().dt.total_seconds().fillna(0.0)

    # Segment increments when a gap exceeding 1.0s occurs (temporal feature reset)
    df["segment_id"] = (delta_t > gap_threshold_s).cumsum()

    # Session increments when a gap exceeding 5.0s occurs (macro collection session)
    df["session_id"] = (delta_t > session_gap_threshold_s).cumsum()

    return df


def compute_segment_features(
    segment_df: pd.DataFrame, rolling_window: int = ROLLING_WINDOW_SAMPLES
) -> pd.DataFrame:
    """
    Computes instantaneous spatial features, rates of change, and rolling temporal
    features strictly within the bounds of an individual segment.
    """
    seg = segment_df.copy()

    # 1. Instantaneous Spatial Grip Metrics
    seg["combined_pressure"] = seg["left_fsr"] + seg["right_fsr"]
    seg["pressure_diff"] = seg["left_fsr"] - seg["right_fsr"]

    # 2. Actual Timestamp Intervals and Differential Dynamics
    # First sample in a segment has sample_interval_s = 0.0 and zero change
    dt_series = seg["dt"].diff().dt.total_seconds().fillna(0.0)
    seg["sample_interval_s"] = dt_series

    delta_comb = seg["combined_pressure"].diff().fillna(0.0)
    delta_left = seg["left_fsr"].diff().fillna(0.0)
    delta_right = seg["right_fsr"].diff().fillna(0.0)

    seg["pressure_change"] = delta_comb

    # Rate of change: ADC units per second (safe division against zero dt)
    seg["pressure_rate_of_change"] = np.where(
        dt_series > 0, delta_comb / dt_series, 0.0
    )
    seg["left_rate_of_change"] = np.where(
        dt_series > 0, delta_left / dt_series, 0.0
    )
    seg["right_rate_of_change"] = np.where(
        dt_series > 0, delta_right / dt_series, 0.0
    )

    # 3. Localized Rolling Window Features (approximately 1 second = 5 samples)
    # min_periods=1 ensures short segments and initial samples are evaluated safely
    seg["left_fsr_mean"] = seg["left_fsr"].rolling(rolling_window, min_periods=1).mean()
    seg["left_fsr_std"] = (
        seg["left_fsr"].rolling(rolling_window, min_periods=1).std(ddof=1).fillna(0.0)
    )
    seg["left_fsr_min"] = seg["left_fsr"].rolling(rolling_window, min_periods=1).min()
    seg["left_fsr_max"] = seg["left_fsr"].rolling(rolling_window, min_periods=1).max()

    seg["right_fsr_mean"] = seg["right_fsr"].rolling(rolling_window, min_periods=1).mean()
    seg["right_fsr_std"] = (
        seg["right_fsr"].rolling(rolling_window, min_periods=1).std(ddof=1).fillna(0.0)
    )
    seg["right_fsr_min"] = seg["right_fsr"].rolling(rolling_window, min_periods=1).min()
    seg["right_fsr_max"] = seg["right_fsr"].rolling(rolling_window, min_periods=1).max()

    seg["combined_mean"] = (
        seg["combined_pressure"].rolling(rolling_window, min_periods=1).mean()
    )
    seg["combined_std"] = (
        seg["combined_pressure"].rolling(rolling_window, min_periods=1).std(ddof=1).fillna(0.0)
    )
    seg["combined_min"] = (
        seg["combined_pressure"].rolling(rolling_window, min_periods=1).min()
    )
    seg["combined_max"] = (
        seg["combined_pressure"].rolling(rolling_window, min_periods=1).max()
    )

    return seg


def engineer_features_by_segment(
    df: pd.DataFrame, rolling_window: int = ROLLING_WINDOW_SAMPLES
) -> pd.DataFrame:
    """
    Applies segment-isolated temporal feature extraction across all segments.
    Guarantees no window or delta crosses segment boundaries.
    """
    processed_segments: List[pd.DataFrame] = []

    for _, seg_df in df.groupby("segment_id", sort=False):
        processed_seg = compute_segment_features(seg_df, rolling_window=rolling_window)
        processed_segments.append(processed_seg)

    prepared_df = pd.concat(processed_segments, ignore_index=True)

    # Order columns cleanly: Metadata -> Raw Features -> Engineered Features -> Target
    # session_id is positioned immediately after timestamp
    ordered_columns = [
        "timestamp",
        "session_id",
        "segment_id",
        "sample_interval_s",
        # Raw readings
        "left_fsr",
        "right_fsr",
        # Spatial features
        "combined_pressure",
        "pressure_diff",
        # Dynamics & Rate of Change
        "pressure_change",
        "pressure_rate_of_change",
        "left_rate_of_change",
        "right_rate_of_change",
        # Rolling Temporal Left FSR (~1s)
        "left_fsr_mean",
        "left_fsr_std",
        "left_fsr_min",
        "left_fsr_max",
        # Rolling Temporal Right FSR (~1s)
        "right_fsr_mean",
        "right_fsr_std",
        "right_fsr_min",
        "right_fsr_max",
        # Rolling Temporal Combined Pressure (~1s)
        "combined_mean",
        "combined_std",
        "combined_min",
        "combined_max",
        # Target Label
        "grip_label",
    ]

    return prepared_df[ordered_columns]


def validate_prepared_dataset(df: pd.DataFrame) -> None:
    """
    Rigorous validation check ensuring zero NaNs, zero Infinities,
    valid class labels, exact column counts, and valid non-empty rows.
    """
    # 1. Null / NaN check
    null_counts = df.isnull().sum()
    if null_counts.any():
        cols_with_null = null_counts[null_counts > 0].to_dict()
        raise ValueError(f"Prepared dataset contains NaN values: {cols_with_null}")

    # 2. Infinite values check on numerical features
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_counts = np.isinf(df[numeric_cols]).sum()
    if inf_counts.any():
        cols_with_inf = inf_counts[inf_counts > 0].to_dict()
        raise ValueError(f"Prepared dataset contains Infinite values: {cols_with_inf}")

    # 3. Label validation
    unique_labels = set(df["grip_label"].unique())
    invalid_labels = unique_labels - set(VALID_CLASSES)
    if invalid_labels:
        raise ValueError(f"Prepared dataset contains invalid labels: {invalid_labels}")

    # 4. Confirm all 5 classes are present
    missing_classes = set(VALID_CLASSES) - unique_labels
    if missing_classes:
        raise ValueError(f"Prepared dataset is missing expected classes: {missing_classes}")

    # 5. Row and column count confirmation
    if len(df) != 1747:
        raise ValueError(f"Expected exactly 1747 rows, got {len(df)}")
    if len(df.columns) != 25:
        raise ValueError(f"Expected exactly 25 columns, got {len(df.columns)}")

    # 6. Session and segment range confirmation
    if df["session_id"].nunique() != 8:
        raise ValueError(f"Expected exactly 8 sessions, got {df['session_id'].nunique()}")
    if df["segment_id"].nunique() != 86:
        raise ValueError(f"Expected exactly 86 segments, got {df['segment_id'].nunique()}")


def run_pipeline(
    input_file: Path = RAW_DATA_FILE,
    output_file: Path = PREPARED_DATA_FILE,
    rolling_window: int = ROLLING_WINDOW_SAMPLES,
    gap_threshold: float = SEGMENT_GAP_THRESHOLD_S,
    session_gap_threshold: float = SESSION_GAP_THRESHOLD_S,
) -> Dict:
    """
    Executes the end-to-end dataset preparation and feature engineering pipeline.
    """
    print("=" * 75)
    print("Bio-Steer Dataset Preparation & Dual-Grouping Feature Pipeline")
    print(f"Input Raw Dataset       : {input_file}")
    print(f"Output Prepared File    : {output_file}")
    print(f"Rolling Window Size     : {rolling_window} samples (~1.0s)")
    print(f"Segment Gap Threshold   : {gap_threshold}s (for temporal resets)")
    print(f"Session Gap Threshold   : {session_gap_threshold}s (for macro session grouping)")
    print("=" * 75)

    # 1. Load raw data
    raw_df = load_raw_dataset(input_file)
    raw_count = len(raw_df)
    print(f"[1/5] Loaded raw dataset: {raw_count} rows.")

    # 2. Filter mislabeled records
    clean_df, excluded_df = exclude_corrupted_record_groups(raw_df)
    excluded_count = len(excluded_df)
    print(f"[2/5] Excluded {excluded_count} mislabeled verification test records.")

    # 3. Identify recording segments and macro sessions
    segmented_df = identify_recording_segments(
        clean_df,
        gap_threshold_s=gap_threshold,
        session_gap_threshold_s=session_gap_threshold,
    )
    num_segments = segmented_df["segment_id"].nunique()
    num_sessions = segmented_df["session_id"].nunique()
    print(
        f"[3/5] Identified {num_segments} segments (gaps > {gap_threshold}s) and "
        f"{num_sessions} macro sessions (gaps > {session_gap_threshold}s)."
    )

    # 4. Feature engineering per segment
    prepared_df = engineer_features_by_segment(segmented_df, rolling_window=rolling_window)
    prepared_count = len(prepared_df)
    print(f"[4/5] Successfully engineered {len(prepared_df.columns) - 4} features across all segments.")

    # 5. Validation check
    validate_prepared_dataset(prepared_df)
    print(
        f"[5/5] Data validation passed: exactly {prepared_count} rows, "
        f"exactly {len(prepared_df.columns)} columns, 0 NaNs, 0 Infinities."
    )

    # Save prepared dataset
    output_file.parent.mkdir(parents=True, exist_ok=True)
    prepared_df.to_csv(output_file, index=False)
    print(f"\nPrepared dataset saved to: {output_file}")
    print("=" * 75)

    # Compile report summary
    class_distribution = prepared_df["grip_label"].value_counts().to_dict()

    # Session-wise summary
    session_summaries = []
    for sid, s_df in prepared_df.groupby("session_id", sort=True):
        seg_min = int(s_df["segment_id"].min())
        seg_max = int(s_df["segment_id"].max())
        seg_str = f"{seg_min}" if seg_min == seg_max else f"{seg_min}-{seg_max}"
        session_summaries.append({
            "session_id": int(sid),
            "row_count": len(s_df),
            "segment_ids": seg_str,
            "start_time": s_df["timestamp"].iloc[0],
            "end_time": s_df["timestamp"].iloc[-1],
            "class_distribution": s_df["grip_label"].value_counts().to_dict(),
        })

    summary = {
        "raw_count": raw_count,
        "excluded_count": excluded_count,
        "prepared_count": prepared_count,
        "num_segments": num_segments,
        "num_sessions": num_sessions,
        "class_distribution": class_distribution,
        "session_summaries": session_summaries,
        "features": [
            c
            for c in prepared_df.columns
            if c not in ["timestamp", "session_id", "segment_id", "grip_label"]
        ],
    }

    return summary


def print_session_wise_summary(session_summaries: List[Dict]) -> None:
    """Prints a clear, formatted session-wise breakdown table."""
    print("\n" + "=" * 90)
    print("SESSION-WISE SUMMARY BREAKDOWN")
    print("=" * 90)
    for s in session_summaries:
        sid = s["session_id"]
        cnt = s["row_count"]
        segs = s["segment_ids"]
        t0 = s["start_time"]
        t1 = s["end_time"]
        dist = s["class_distribution"]
        dist_str = ", ".join(f"{k}: {v}" for k, v in dist.items())
        print(f"Session {sid:2d} | Rows: {cnt:4d} | Segments: {segs:<7} | {t0} -> {t1}")
        print(f"           Classes: {dist_str}")
    print("=" * 90)


if __name__ == "__main__":
    summary = run_pipeline()
    print_session_wise_summary(summary["session_summaries"])
    print("\nOverall Dataset Summary:")
    print(f"Raw records loaded        : {summary['raw_count']}")
    print(f"Excluded records          : {summary['excluded_count']}")
    print(f"Prepared records          : {summary['prepared_count']}")
    print(f"Total macro sessions (0-{summary['num_sessions'] - 1}): {summary['num_sessions']}")
    print(f"Total segments (0-67)     : {summary['num_segments']}")
    print("Final Class Distribution  :")
    for lbl, count in summary["class_distribution"].items():
        pct = (count / summary["prepared_count"]) * 100
        print(f"  - {lbl:<15}: {count:4d} ({pct:5.2f}%)")
