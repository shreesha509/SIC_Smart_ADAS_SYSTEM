"""
Bio-Steer ESP32 Sensor Receiver Module

Fetches live driver grip pressure readings from an ESP32 microcontroller
over HTTP GET. The ESP32 reads two Force Sensitive Resistors (FSRs):
- Left FSR: GPIO 34
- Right FSR: GPIO 35
"""

import logging
from typing import Any, Dict, Optional
import requests

# Logger configuration
logger = logging.getLogger(__name__)

# Configuration constants
ESP32_IP = "10.185.113.250"
ESP32_URL = f"http://{ESP32_IP}/sensors"
DEFAULT_TIMEOUT = 3.0  # seconds


class SensorReceiverError(Exception):
    """Exception raised for ESP32 sensor communication and data validation errors."""
    pass


def get_sensor_data(
    url: str = ESP32_URL,
    timeout: float = DEFAULT_TIMEOUT,
    raise_on_error: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Fetch sensor readings from the ESP32 endpoint.

    Args:
        url: The HTTP GET endpoint of the ESP32.
        timeout: Maximum seconds to wait for a response.
        raise_on_error: If True, raises SensorReceiverError on failure.
                        If False, logs the error and returns None.

    Returns:
        A dictionary containing verified sensor readings:
            {"left_fsr": <numeric>, "right_fsr": <numeric>}
        or None if raise_on_error is False and an error occurred.

    Raises:
        SensorReceiverError: If the request fails, times out, returns invalid HTTP status,
                             returns non-JSON data, or sensor fields are missing/invalid.
    """
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()

        try:
            payload = response.json()
        except ValueError as json_err:
            raise SensorReceiverError(f"Invalid JSON response from ESP32: {json_err}") from json_err

        if not isinstance(payload, dict):
            raise SensorReceiverError(f"Expected JSON object from ESP32, got {type(payload).__name__}")

        # Validate required fields exist
        if "left_fsr" not in payload:
            raise SensorReceiverError("Missing 'left_fsr' field in ESP32 response.")
        if "right_fsr" not in payload:
            raise SensorReceiverError("Missing 'right_fsr' field in ESP32 response.")

        left_fsr = payload["left_fsr"]
        right_fsr = payload["right_fsr"]

        # Validate numeric types (ensure booleans are not accepted as numbers)
        if isinstance(left_fsr, bool) or not isinstance(left_fsr, (int, float)):
            try:
                left_fsr = float(left_fsr) if "." in str(left_fsr) else int(left_fsr)
            except (ValueError, TypeError):
                raise SensorReceiverError(f"Field 'left_fsr' must be numeric, got: {left_fsr!r}")

        if isinstance(right_fsr, bool) or not isinstance(right_fsr, (int, float)):
            try:
                right_fsr = float(right_fsr) if "." in str(right_fsr) else int(right_fsr)
            except (ValueError, TypeError):
                raise SensorReceiverError(f"Field 'right_fsr' must be numeric, got: {right_fsr!r}")

        return {
            "left_fsr": left_fsr,
            "right_fsr": right_fsr
        }

    except requests.exceptions.Timeout as err:
        msg = f"Connection timed out after {timeout}s while reaching ESP32 at {url}"
        logger.error(msg)
        if raise_on_error:
            raise SensorReceiverError(msg) from err
        return None

    except requests.exceptions.ConnectionError as err:
        msg = f"Failed to connect to ESP32 at {url}. Ensure ESP32 is powered on and connected to the hotspot."
        logger.error(msg)
        if raise_on_error:
            raise SensorReceiverError(msg) from err
        return None

    except requests.exceptions.HTTPError as err:
        msg = f"HTTP error {response.status_code} received from ESP32 at {url}"
        logger.error(msg)
        if raise_on_error:
            raise SensorReceiverError(msg) from err
        return None

    except requests.exceptions.RequestException as err:
        msg = f"Network error communicating with ESP32 at {url}: {err}"
        logger.error(msg)
        if raise_on_error:
            raise SensorReceiverError(msg) from err
        return None

    except SensorReceiverError as err:
        logger.error(str(err))
        if raise_on_error:
            raise
        return None


if __name__ == "__main__":
    import sys
    print(f"Connecting to ESP32 at {ESP32_URL} ...")
    try:
        data = get_sensor_data()
        print("ESP32 reachable: True")
        print(f"left_fsr: {data['left_fsr']}")
        print(f"right_fsr: {data['right_fsr']}")
    except SensorReceiverError as e:
        print("ESP32 reachable: False")
        print(f"Error: {e}")
        sys.exit(1)
