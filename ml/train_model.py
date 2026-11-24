"""
Bio-Steer session-aware Random Forest training pipeline.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedGroupKFold


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PREPARED_DATA_FILE = DATA_DIR / "prepared_sensor_data.csv"
MODEL_OUTPUT_FILE = PROJECT_ROOT / "ml" / "model.joblib"

FEATURE_COLS = [
    "sample_interval_s",
    "left_fsr",
    "right_fsr",
    "combined_pressure",
    "pressure_diff",
    "pressure_change",
    "pressure_rate_of_change",
    "left_rate_of_change",
    "right_rate_of_change",
    "left_fsr_mean",
    "left_fsr_std",
    "left_fsr_min",
    "left_fsr_max",
    "right_fsr_mean",
    "right_fsr_std",
    "right_fsr_min",
    "right_fsr_max",
    "combined_mean",
    "combined_std",
    "combined_min",
    "combined_max",
]

TARGET_COL = "grip_label"
GROUP_COL = "session_id"
METADATA_COLS = ["timestamp", "session_id", "segment_id", "grip_label"]
CLASS_ORDER = ["no_grip", "steady_grip", "firm_grip", "sudden_clench", "release"]


def load_and_validate_training_data(filepath: Path = PREPARED_DATA_FILE) -> pd.DataFrame:
    """Load and validate the prepared training schema without hardcoded row counts."""
    if not filepath.exists():
        raise FileNotFoundError(f"Prepared dataset not found: {filepath}")

    df = pd.read_csv(filepath)
    if df.empty:
        raise ValueError("Prepared dataset is empty.")

    missing_columns = sorted(
        set(FEATURE_COLS + [TARGET_COL, GROUP_COL]) - set(df.columns)
    )
    if missing_columns:
        raise ValueError(f"Prepared dataset missing required columns: {missing_columns}")

    for metadata_col in METADATA_COLS:
        if metadata_col in FEATURE_COLS:
            raise ValueError(
                f"CRITICAL: Metadata/target column '{metadata_col}' found in FEATURE_COLS."
            )

    null_count = int(df[FEATURE_COLS].isnull().sum().sum())
    if null_count:
        raise ValueError(f"Found {null_count} null values in feature columns.")

    inf_count = int(np.isinf(df[FEATURE_COLS].to_numpy(dtype=float)).sum())
    if inf_count:
        raise ValueError(f"Found {inf_count} infinite values in feature columns.")

    if df[GROUP_COL].isnull().any():
        raise ValueError("Session IDs contain null values.")
    if df[GROUP_COL].nunique() < 5:
        raise ValueError(
            f"Need at least 5 session groups for 5-fold validation; "
            f"found {df[GROUP_COL].nunique()}."
        )

    labels = set(df[TARGET_COL].dropna().unique())
    if labels != set(CLASS_ORDER):
        raise ValueError(
            f"Target labels mismatch. Found: {sorted(labels)}; "
            f"Expected: {CLASS_ORDER}"
        )
    if df[TARGET_COL].isnull().any():
        raise ValueError("Target labels contain null values.")

    return df


def class_distribution(labels: pd.Series) -> Dict[str, int]:
    counts = labels.value_counts()
    return {label: int(counts[label]) for label in CLASS_ORDER if label in counts}


def execute_session_aware_cross_validation(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """Run session-separated StratifiedGroupKFold and pool OOF predictions."""
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()
    groups = df[GROUP_COL].copy()

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof_predictions = np.empty(len(df), dtype=object)
    oof_probabilities = np.zeros((len(df), len(CLASS_ORDER)), dtype=float)
    fold_summaries: List[Dict] = []

    print("\n" + "=" * 80)
    print("5-FOLD STRATIFIED GROUP CROSS-VALIDATION (Grouped by session_id)")
    print("=" * 80)

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups), 1):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_val = y.iloc[val_idx]

        train_sessions = sorted(groups.iloc[train_idx].unique().tolist())
        val_sessions = sorted(groups.iloc[val_idx].unique().tolist())
        overlap = set(train_sessions) & set(val_sessions)
        if overlap:
            raise RuntimeError(
                f"Fold {fold}: Session leakage detected. Overlap: {sorted(overlap)}"
            )

        train_classes = set(y_train.unique())
        if train_classes != set(CLASS_ORDER):
            missing = sorted(set(CLASS_ORDER) - train_classes)
            raise RuntimeError(
                f"Fold {fold}: training partition missing classes: {missing}"
            )

        val_classes = set(y_val.unique())
        missing_validation_classes = sorted(set(CLASS_ORDER) - val_classes)

        rf_fold = RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        rf_fold.fit(X_train, y_train)

        val_predictions = rf_fold.predict(X_val)
        val_probabilities = rf_fold.predict_proba(X_val)
        oof_predictions[val_idx] = val_predictions

        for target_idx, cls in enumerate(CLASS_ORDER):
            if cls in rf_fold.classes_:
                model_idx = int(np.flatnonzero(rf_fold.classes_ == cls)[0])
                oof_probabilities[val_idx, target_idx] = (
                    val_probabilities[:, model_idx]
                )

        report = classification_report(
            y_val,
            val_predictions,
            labels=CLASS_ORDER,
            output_dict=True,
            zero_division=0,
        )
        per_class: Dict[str, Dict[str, float]] = {}
        for cls in CLASS_ORDER:
            per_class[cls] = {
                "precision": float(report[cls]["precision"]),
                "recall": float(report[cls]["recall"]),
                "f1": float(report[cls]["f1-score"]),
                "support": int(report[cls]["support"]),
            }

        fold_macro_f1 = float(
            f1_score(y_val, val_predictions, average="macro", zero_division=0)
        )
        fold_balanced_accuracy = float(
            balanced_accuracy_score(y_val, val_predictions)
        )
        fold_accuracy = float(accuracy_score(y_val, val_predictions))

        fold_summary = {
            "fold": fold,
            "train_sessions": train_sessions,
            "val_sessions": val_sessions,
            "train_size": len(train_idx),
            "val_size": len(val_idx),
            "train_class_distribution": class_distribution(y_train),
            "val_class_distribution": class_distribution(y_val),
            "val_classes_present": sorted(val_classes),
            "missing_validation_classes": missing_validation_classes,
            "accuracy": fold_accuracy,
            "macro_f1": fold_macro_f1,
            "balanced_accuracy": fold_balanced_accuracy,
            "per_class": per_class,
        }
        fold_summaries.append(fold_summary)

        print(f"\nFold {fold}:")
        print(f"  Train Sessions : {train_sessions} ({len(train_idx)} samples)")
        print(f"  Val Sessions   : {val_sessions} ({len(val_idx)} samples)")
        print(f"  Train Classes  : {class_distribution(y_train)}")
        print(f"  Val Classes    : {class_distribution(y_val)}")
        if missing_validation_classes:
            print(f"  Missing Val    : {missing_validation_classes}")
        print(
            f"  Fold Metrics   : Accuracy={fold_accuracy:.4f} "
            f"| Macro F1={fold_macro_f1:.4f} "
            f"| Balanced Acc={fold_balanced_accuracy:.4f}"
        )
        for cls in CLASS_ORDER:
            metrics = per_class[cls]
            print(
                f"    {cls:<15} Precision={metrics['precision']:.4f} "
                f"Recall={metrics['recall']:.4f} "
                f"F1={metrics['f1']:.4f} "
                f"Support={metrics['support']}"
            )

    return oof_predictions, oof_probabilities, fold_summaries


def evaluate_pooled_oof_predictions(
    y_true: pd.Series,
    oof_predictions: np.ndarray,
    fold_summaries: List[Dict],
) -> Dict:
    """Compute metrics from pooled out-of-fold predictions."""
    overall_accuracy = float(accuracy_score(y_true, oof_predictions))
    macro_f1 = float(
        f1_score(y_true, oof_predictions, average="macro", zero_division=0)
    )
    balanced_accuracy = float(
        balanced_accuracy_score(y_true, oof_predictions)
    )
    report = classification_report(
        y_true,
        oof_predictions,
        labels=CLASS_ORDER,
        output_dict=True,
        zero_division=0,
    )
    confusion_raw = confusion_matrix(
        y_true, oof_predictions, labels=CLASS_ORDER
    )
    confusion_normalized = confusion_matrix(
        y_true, oof_predictions, labels=CLASS_ORDER, normalize="true"
    )

    print("\n" + "=" * 80)
    print(f"POOLED OUT-OF-FOLD EVALUATION METRICS (Complete Dataset N = {len(y_true)})")
    print("=" * 80)
    print(
        f"Overall OOF Accuracy : {overall_accuracy:.4f} "
        f"({overall_accuracy * 100:.2f}%)"
    )
    print(f"Macro F1-Score       : {macro_f1:.4f}")
    print(
        f"Balanced Accuracy    : {balanced_accuracy:.4f} "
        f"({balanced_accuracy * 100:.2f}%)"
    )

    print("\nPer-Class Performance Breakdown:")
    print("-" * 70)
    print(
        f"{'Class':<16} {'Precision':<12} {'Recall':<12} "
        f"{'F1-Score':<12} {'Support':<8}"
    )
    print("-" * 70)
    for cls in CLASS_ORDER:
        metrics = report[cls]
        print(
            f"{cls:<16} {metrics['precision']:<12.4f} "
            f"{metrics['recall']:<12.4f} "
            f"{metrics['f1-score']:<12.4f} "
            f"{int(metrics['support']):<8d}"
        )
    print("-" * 70)

    print("\nConfusion Matrix (Raw Counts):")
    print(f"Order: {CLASS_ORDER}")
    print(confusion_raw)
    print("\nNormalized Confusion Matrix (Recall per class):")
    print(np.round(confusion_normalized, 4))

    return {
        "accuracy": overall_accuracy,
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_accuracy,
        "per_class": report,
        "confusion_matrix_raw": confusion_raw.tolist(),
        "confusion_matrix_norm": confusion_normalized.tolist(),
        "fold_summaries": fold_summaries,
    }


def train_final_model(
    df: pd.DataFrame,
    evaluation_metrics: Dict,
    output_path: Path = MODEL_OUTPUT_FILE,
) -> Path:
    """Fit the production Random Forest and save model plus metadata."""
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()

    final_model = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    final_model.fit(X, y)

    artifact = {
        "model": final_model,
        "feature_names": FEATURE_COLS,
        "label_classes": CLASS_ORDER,
        "model_config": {
            "classifier": "RandomForestClassifier",
            "n_estimators": 100,
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
        },
        "training_row_count": len(df),
        "training_session_count": int(df[GROUP_COL].nunique()),
        "training_segment_count": (
            int(df["segment_id"].nunique()) if "segment_id" in df.columns else None
        ),
        "training_timestamp": datetime.now().isoformat(),
        "validation_method": (
            "5-Fold StratifiedGroupKFold grouped by session_id "
            "(Pooled Out-of-Fold Evaluation)"
        ),
        "oof_metrics": {
            "accuracy": evaluation_metrics["accuracy"],
            "macro_f1": evaluation_metrics["macro_f1"],
            "balanced_accuracy": evaluation_metrics["balanced_accuracy"],
            "per_class": evaluation_metrics["per_class"],
            "confusion_matrix_raw": evaluation_metrics["confusion_matrix_raw"],
            "confusion_matrix_norm": evaluation_metrics["confusion_matrix_norm"],
            "fold_summaries": evaluation_metrics["fold_summaries"],
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path)
    print(f"\nFinal model saved to: {output_path}")
    return output_path


def verify_saved_model_artifact(artifact_path: Path = MODEL_OUTPUT_FILE) -> Dict:
    """Load the serialized artifact and validate metadata and model behavior."""
    if not artifact_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {artifact_path}")

    artifact = joblib.load(artifact_path)
    required_keys = {
        "model",
        "feature_names",
        "label_classes",
        "model_config",
        "training_row_count",
        "oof_metrics",
    }
    missing_keys = required_keys - set(artifact)
    if missing_keys:
        raise ValueError(f"Model artifact missing keys: {sorted(missing_keys)}")

    if artifact["feature_names"] != FEATURE_COLS:
        raise ValueError("Stored feature names or order do not match training.")
    if len(artifact["feature_names"]) != 21:
        raise ValueError(
            f"Expected 21 feature names, found {len(artifact['feature_names'])}"
        )

    model = artifact["model"]
    if not hasattr(model, "predict") or not hasattr(model, "predict_proba"):
        raise TypeError("Loaded object is not a probability-capable classifier.")

    return {
        "verified": True,
        "feature_count": len(artifact["feature_names"]),
        "training_row_count": artifact["training_row_count"],
        "model_classes": model.classes_.tolist(),
    }


def main() -> Dict:
    df = load_and_validate_training_data()
    oof_predictions, oof_probabilities, fold_summaries = (
        execute_session_aware_cross_validation(df)
    )
    metrics = evaluate_pooled_oof_predictions(
        df[TARGET_COL], oof_predictions, fold_summaries
    )
    artifact_path = train_final_model(df, metrics)
    verification = verify_saved_model_artifact(artifact_path)
    print("\nModel artifact load verification passed.")
    print(verification)
    return verification


if __name__ == "__main__":
    main()
