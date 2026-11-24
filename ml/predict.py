"""
Reusable live-inference module for BIO-STEER.

This module mirrors ml/prepare_dataset.py feature calculation. It does not
generate readings or connect predictions to vehicle actuation.
"""

from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Callable, Deque, Dict, Iterable, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "ml" / "model.joblib"

FEATURE_NAMES = [
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

ROLLING_WINDOW_SAMPLES = 5
SEGMENT_GAP_THRESHOLD_S = 1.0


class GripInferenceError(Exception):
    """Safe, inspectable live-inference error."""


class SensorCommunicationError(GripInferenceError):
    """Raised when a sensor-reading provider fails."""


def _parse_timestamp(timestamp: Union[str, datetime, pd.Timestamp]) -> pd.Timestamp:
    if isinstance(timestamp, pd.Timestamp):
        parsed = timestamp
    elif isinstance(timestamp, datetime):
        parsed = pd.Timestamp(timestamp)
    elif isinstance(timestamp, str):
        parsed = pd.Timestamp(timestamp)
    else:
        raise GripInferenceError(
            "Timestamp must be an ISO timestamp string, datetime, or pandas Timestamp."
        )

    if pd.isna(parsed):
        raise GripInferenceError("Timestamp is missing.")
    return parsed


def _validate_sensor_value(value: float, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise GripInferenceError(f"{name} must be a numeric reading.") from exc

    if not np.isfinite(result):
        raise GripInferenceError(f"{name} must be finite.")
    if result < 0:
        raise GripInferenceError(f"{name} cannot be negative.")
    return result


class LiveGripPredictor:
    """Stateful five-sample live feature calculator and grip classifier."""

    def __init__(
        self,
        model_path: Union[str, Path] = MODEL_PATH,
        rolling_window: int = ROLLING_WINDOW_SAMPLES,
        segment_gap_threshold_s: float = SEGMENT_GAP_THRESHOLD_S,
    ) -> None:
        self.model_path = Path(model_path)
        self.rolling_window = rolling_window
        self.segment_gap_threshold_s = segment_gap_threshold_s
        self.model = None
        self.feature_names = None
        self.label_classes = None
        self.reset()

        if not self.model_path.exists():
            raise GripInferenceError(
                f"Trained model file not found: {self.model_path}"
            )

        try:
            artifact = joblib.load(self.model_path)
        except Exception as exc:
            raise GripInferenceError(
                f"Unable to load trained model: {self.model_path}"
            ) from exc

        required = {"model", "feature_names", "label_classes"}
        missing = required - set(artifact)
        if missing:
            raise GripInferenceError(
                f"Model artifact missing required metadata: {sorted(missing)}"
            )

        if artifact["feature_names"] != FEATURE_NAMES:
            raise GripInferenceError(
                "Model feature metadata does not match the live feature contract."
            )
        if len(FEATURE_NAMES) != 21:
            raise GripInferenceError("Live feature contract must contain 21 features.")

        self.model = artifact["model"]
        self.feature_names = artifact["feature_names"]
        self.label_classes = artifact["label_classes"]

        if not hasattr(self.model, "predict") or not hasattr(self.model, "predict_proba"):
            raise GripInferenceError("Loaded model is not a probability classifier.")

    def reset(self) -> None:
        """Clear all temporal and rolling state."""
        self._previous_timestamp: Optional[pd.Timestamp] = None
        self._previous_left: Optional[float] = None
        self._previous_right: Optional[float] = None
        self._previous_combined: Optional[float] = None
        self._left: Deque[float] = deque(maxlen=self.rolling_window)
        self._right: Deque[float] = deque(maxlen=self.rolling_window)
        self._combined: Deque[float] = deque(maxlen=self.rolling_window)

    def calculate_features(
        self,
        left_fsr: float,
        right_fsr: float,
        timestamp: Union[str, datetime, pd.Timestamp],
    ) -> Dict[str, float]:
        """
        Calculate exactly one current-sample feature vector.

        A gap greater than 1 second resets the timestamp interval, deltas, rates,
        and rolling buffers. Future samples are never accessed.
        """
        left = _validate_sensor_value(left_fsr, "left_fsr")
        right = _validate_sensor_value(right_fsr, "right_fsr")
        current_timestamp = _parse_timestamp(timestamp)

        if self._previous_timestamp is not None:
            elapsed = (
                current_timestamp - self._previous_timestamp
            ).total_seconds()
            if not np.isfinite(elapsed):
                raise GripInferenceError("Timestamp interval is not finite.")
        else:
            elapsed = 0.0

        if elapsed < 0:
            raise GripInferenceError(
                "Timestamp is earlier than the previous received reading."
            )

        if elapsed > self.segment_gap_threshold_s:
            self.reset()
            elapsed = 0.0

        combined = left + right
        pressure_diff = left - right

        if self._previous_left is None:
            pressure_change = 0.0
            pressure_rate = 0.0
            left_rate = 0.0
            right_rate = 0.0
        else:
            left_change = left - self._previous_left
            right_change = right - self._previous_right
            pressure_change = combined - self._previous_combined
            if elapsed > 0:
                pressure_rate = pressure_change / elapsed
                left_rate = left_change / elapsed
                right_rate = right_change / elapsed
            else:
                pressure_rate = 0.0
                left_rate = 0.0
                right_rate = 0.0

        self._left.append(left)
        self._right.append(right)
        self._combined.append(combined)

        left_values = np.asarray(self._left, dtype=float)
        right_values = np.asarray(self._right, dtype=float)
        combined_values = np.asarray(self._combined, dtype=float)

        left_std = float(np.std(left_values, ddof=1)) if len(left_values) > 1 else 0.0
        right_std = (
            float(np.std(right_values, ddof=1)) if len(right_values) > 1 else 0.0
        )
        combined_std = (
            float(np.std(combined_values, ddof=1))
            if len(combined_values) > 1
            else 0.0
        )

        features = {
            "sample_interval_s": float(elapsed),
            "left_fsr": float(left),
            "right_fsr": float(right),
            "combined_pressure": float(combined),
            "pressure_diff": float(pressure_diff),
            "pressure_change": float(pressure_change),
            "pressure_rate_of_change": float(pressure_rate),
            "left_rate_of_change": float(left_rate),
            "right_rate_of_change": float(right_rate),
            "left_fsr_mean": float(np.mean(left_values)),
            "left_fsr_std": left_std,
            "left_fsr_min": float(np.min(left_values)),
            "left_fsr_max": float(np.max(left_values)),
            "right_fsr_mean": float(np.mean(right_values)),
            "right_fsr_std": right_std,
            "right_fsr_min": float(np.min(right_values)),
            "right_fsr_max": float(np.max(right_values)),
            "combined_mean": float(np.mean(combined_values)),
            "combined_std": combined_std,
            "combined_min": float(np.min(combined_values)),
            "combined_max": float(np.max(combined_values)),
        }

        self._previous_timestamp = current_timestamp
        self._previous_left = left
        self._previous_right = right
        self._previous_combined = combined

        return features

    def predict(
        self,
        left_fsr: float,
        right_fsr: float,
        timestamp: Union[str, datetime, pd.Timestamp],
    ) -> Dict[str, object]:
        """Calculate features and return a predicted class and probabilities."""
        features = self.calculate_features(left_fsr, right_fsr, timestamp)
        feature_row = pd.DataFrame([[features[name] for name in FEATURE_NAMES]],
                                   columns=FEATURE_NAMES)

        try:
            predicted = str(self.model.predict(feature_row)[0])
            raw_probabilities = self.model.predict_proba(feature_row)[0]
        except Exception as exc:
            raise GripInferenceError("Model prediction failed.") from exc

        probabilities: Dict[str, float] = {}
        for cls, probability in zip(self.model.classes_, raw_probabilities):
            probabilities[str(cls)] = float(probability)

        if predicted not in probabilities:
            raise GripInferenceError("Predicted class is absent from model probabilities.")

        probability_values = np.asarray(list(probabilities.values()), dtype=float)
        if (
            not np.all(np.isfinite(probability_values))
            or np.any(probability_values < 0)
            or abs(float(probability_values.sum()) - 1.0) > 1e-4
        ):
            raise GripInferenceError("Model returned invalid probability values.")

        return {
            "predicted_grip": predicted,
            "probabilities": probabilities,
            "features": features,
        }

    def predict_from_sensor(
        self,
        sensor_reader: Callable[[], object],
    ) -> Dict[str, object]:
        """
        Safely read one sensor record from a caller-provided communication function.

        The reader may return a tuple (left_fsr, right_fsr, timestamp) or dict
        containing left_fsr, right_fsr and timestamp.
        """
        try:
            reading = sensor_reader()
        except (OSError, TimeoutError, ConnectionError) as exc:
            raise SensorCommunicationError(
                "Sensor communication failed."
            ) from exc
        except Exception as exc:
            raise SensorCommunicationError(
                "Sensor reader raised an unexpected communication error."
            ) from exc

        if isinstance(reading, dict):
            try:
                left = reading["left_fsr"]
                right = reading["right_fsr"]
                timestamp = reading["timestamp"]
            except KeyError as exc:
                raise GripInferenceError(
                    f"Sensor reading missing key: {exc.args[0]}"
                ) from exc
        elif isinstance(reading, (tuple, list)) and len(reading) == 3:
            left, right, timestamp = reading
        else:
            raise GripInferenceError(
                "Sensor reading must be a 3-item tuple/list or a dict."
            )

        return self.predict(left, right, timestamp)
