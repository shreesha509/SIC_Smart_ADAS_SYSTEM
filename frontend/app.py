"""
Bio-Steer: Driver Grip Monitoring and Intelligent Safety Prototype
Automotive ADAS-Inspired Monitoring Dashboard

Features:
- Live ESP32 FSR sensor acquisition via backend/sensor_receiver.py
- Dual-sensor real-time metrics and dynamic visual pressure gauges
- Interactive Plotly time-series rolling waveform
- Sensor connection state diagnostics and error reporting
- Clearly labelled placeholders for future physiological, ML, and control modules
"""

from datetime import datetime
from pathlib import Path
import sys
from typing import Dict, List, Optional

import plotly.graph_objects as go
import streamlit as st

# Ensure project root is available in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.sensor_receiver import get_sensor_data, SensorReceiverError, ESP32_URL, ESP32_IP

# Page configuration
st.set_page_config(
    page_title="Bio-Steer | ADAS Telemetry",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom ADAS Dark Cyberpunk / Automotive styling
CUSTOM_CSS = """
<style>
    /* Global App Background & Font */
    .stApp {
        background: #0a0e17;
        color: #f1f5f9;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }
    
    /* Hide Streamlit default decor */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* ADAS Container Styling */
    .adas-card {
        background: #111827;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 18px 20px;
        margin-bottom: 16px;
    }
    
    .adas-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid #1e293b;
        padding-bottom: 14px;
        margin-bottom: 20px;
    }
    
    .brand-title {
        font-size: 26px;
        font-weight: 800;
        letter-spacing: 2px;
        color: #00f0ff;
        margin: 0;
        text-transform: uppercase;
    }
    
    .brand-sub {
        font-size: 13px;
        color: #94a3b8;
        margin: 4px 0 0 0;
        letter-spacing: 0.5px;
    }
    
    .badge-status-online {
        background: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.4);
        padding: 5px 12px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 1px;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    
    .badge-status-offline {
        background: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.4);
        padding: 5px 12px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 1px;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    
    .badge-placeholder {
        background: rgba(148, 163, 184, 0.1);
        color: #94a3b8;
        border: 1px solid rgba(148, 163, 184, 0.25);
        padding: 3px 8px;
        border-radius: 3px;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    
    .metric-label {
        font-size: 12px;
        font-weight: 600;
        color: #94a3b8;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    
    .metric-value-cyan {
        font-size: 38px;
        font-weight: 800;
        color: #00f0ff;
        line-height: 1.1;
        font-variant-numeric: tabular-nums;
    }
    
    .metric-value-blue {
        font-size: 38px;
        font-weight: 800;
        color: #38bdf8;
        line-height: 1.1;
        font-variant-numeric: tabular-nums;
    }
    
    .metric-sub {
        font-size: 11px;
        color: #64748b;
        margin-top: 4px;
    }
    
    /* Custom Pressure Gauge Bars */
    .gauge-container {
        width: 100%;
        height: 10px;
        background: #1e293b;
        border-radius: 5px;
        overflow: hidden;
        margin-top: 10px;
    }
    
    .gauge-fill-cyan {
        height: 100%;
        background: linear-gradient(90deg, #0284c7, #00f0ff);
        transition: width 0.3s ease;
    }
    
    .gauge-fill-blue {
        height: 100%;
        background: linear-gradient(90deg, #1d4ed8, #38bdf8);
        transition: width 0.3s ease;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Configuration constants
MAX_HISTORY_POINTS = 50
ADC_MAX = 4095.0  # ESP32 12-bit ADC full scale

# Initialize session state for telemetry history
if "telemetry_history" not in st.session_state:
    st.session_state["telemetry_history"] = []
if "last_reading" not in st.session_state:
    st.session_state["last_reading"] = None
if "last_update_ts" not in st.session_state:
    st.session_state["last_update_ts"] = None
if "connection_active" not in st.session_state:
    st.session_state["connection_active"] = False
if "error_message" not in st.session_state:
    st.session_state["error_message"] = None


def render_header(is_connected: bool) -> None:
    """Renders the top branding and live status indicator."""
    status_badge = (
        f'<span class="badge-status-online">● ESP32 ONLINE ({ESP32_IP})</span>'
        if is_connected
        else f'<span class="badge-status-offline">● ESP32 OFFLINE</span>'
    )
    
    st.markdown(
        f"""
        <div class="adas-header">
            <div>
                <h1 class="brand-title">BIO-STEER</h1>
                <p class="brand-sub">Driver Grip Monitoring and Intelligent Safety Prototype</p>
            </div>
            <div>
                {status_badge}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_pressure_plot(history: List[Dict]) -> go.Figure:
    """Constructs the Plotly time-series graph with actual readings."""
    fig = go.Figure()

    if history:
        times = [entry["timestamp"] for entry in history]
        left_values = [entry["left_fsr"] for entry in history]
        right_values = [entry["right_fsr"] for entry in history]

        fig.add_trace(
            go.Scatter(
                x=times,
                y=left_values,
                name="Left FSR (GPIO 34)",
                mode="lines+markers",
                line=dict(color="#00f0ff", width=2.5),
                marker=dict(size=4, color="#00f0ff"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=times,
                y=right_values,
                name="Right FSR (GPIO 35)",
                mode="lines+markers",
                line=dict(color="#38bdf8", width=2.5),
                marker=dict(size=4, color="#38bdf8"),
            )
        )
    else:
        # Awaiting live readings placeholder trace
        fig.add_annotation(
            text="Awaiting live ESP32 telemetry...",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14, color="#64748b"),
        )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#111827",
        plot_bgcolor="#0a0e17",
        margin=dict(l=50, r=20, t=35, b=40),
        height=320,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            font=dict(size=11, color="#94a3b8"),
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor="#1e293b",
            tickfont=dict(size=10, color="#64748b"),
            title=dict(text="Packet Timestamp (HH:MM:SS)", font=dict(size=11, color="#64748b")),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#1e293b",
            tickfont=dict(size=10, color="#64748b"),
            title=dict(text="Raw Pressure (ADC 0 - 4095)", font=dict(size=11, color="#64748b")),
            range=[0, 4200],
        ),
    )
    return fig


@st.fragment(run_every="500ms")
def live_telemetry_fragment():
    """
    Auto-refreshing fragment that polls ESP32 sensor receiver every ~500ms
    and updates live metrics, pressure bars, chart, and status.
    """
    # 1. Fetch live sensor data without fallback / fabricated readings
    try:
        # Responsive timeout to keep UI fluid
        reading = get_sensor_data(timeout=0.8, raise_on_error=True)
        if reading is not None:
            ts_now = datetime.now()
            ts_short = ts_now.strftime("%H:%M:%S.%f")[:-3]
            ts_full = ts_now.strftime("%Y-%m-%d %H:%M:%S")

            # Append real reading to bounded history
            st.session_state["telemetry_history"].append(
                {
                    "timestamp": ts_short,
                    "left_fsr": reading["left_fsr"],
                    "right_fsr": reading["right_fsr"],
                }
            )
            if len(st.session_state["telemetry_history"]) > MAX_HISTORY_POINTS:
                st.session_state["telemetry_history"].pop(0)

            st.session_state["last_reading"] = reading
            st.session_state["last_update_ts"] = ts_full
            st.session_state["connection_active"] = True
            st.session_state["error_message"] = None

    except SensorReceiverError as err:
        st.session_state["connection_active"] = False
        st.session_state["error_message"] = str(err)
    except Exception as err:
        st.session_state["connection_active"] = False
        st.session_state["error_message"] = f"Unexpected failure: {err}"

    # Current state values
    is_connected = st.session_state["connection_active"]
    last_reading = st.session_state["last_reading"]
    last_ts = st.session_state["last_update_ts"] or "Never"
    last_err = st.session_state["error_message"]

    # 1. HEADER
    render_header(is_connected)

    # 2. LIVE SENSOR OVERVIEW
    st.markdown("### 1. Live Sensor Overview")
    col_left, col_right = st.columns(2)

    # Left FSR calculation
    if last_reading is not None and "left_fsr" in last_reading:
        left_val = last_reading["left_fsr"]
        left_display = str(left_val)
        left_pct = min(100.0, max(0.0, (left_val / ADC_MAX) * 100))
    else:
        left_display = "--"
        left_pct = 0.0

    # Right FSR calculation
    if last_reading is not None and "right_fsr" in last_reading:
        right_val = last_reading["right_fsr"]
        right_display = str(right_val)
        right_pct = min(100.0, max(0.0, (right_val / ADC_MAX) * 100))
    else:
        right_display = "--"
        right_pct = 0.0

    with col_left:
        st.markdown(
            f"""
            <div class="adas-card">
                <div class="metric-label">Left FSR Sensor &bull; GPIO 34 (ADC1_CH6)</div>
                <div class="metric-value-cyan">{left_display}</div>
                <div class="gauge-container">
                    <div class="gauge-fill-cyan" style="width: {left_pct:.1f}%;"></div>
                </div>
                <div class="metric-sub">Pressure Level: {left_pct:.1f}% &bull; Last update: {last_ts}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_right:
        st.markdown(
            f"""
            <div class="adas-card">
                <div class="metric-label">Right FSR Sensor &bull; GPIO 35 (ADC1_CH7)</div>
                <div class="metric-value-blue">{right_display}</div>
                <div class="gauge-container">
                    <div class="gauge-fill-blue" style="width: {right_pct:.1f}%;"></div>
                </div>
                <div class="metric-sub">Pressure Level: {right_pct:.1f}% &bull; Last update: {last_ts}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 3. LIVE PRESSURE MONITORING (Plotly time-series)
    st.markdown("### 2. Live Pressure Waveform (Rolling Window)")
    fig = build_pressure_plot(st.session_state["telemetry_history"])
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # 4. SENSOR STATUS
    st.markdown("### 3. Sensor Hardware Status")
    status_col1, status_col2, status_col3 = st.columns(3)

    with status_col1:
        st.markdown(
            f"""
            <div class="adas-card">
                <div class="metric-label">ESP32 Endpoint</div>
                <div style="font-size: 15px; font-weight: 700; color: #f1f5f9;">{ESP32_URL}</div>
                <div class="metric-sub">Hardware Node #1 &bull; Dual FSR Bridge</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with status_col2:
        state_text = '<span style="color: #10b981; font-weight: 700;">CONNECTED</span>' if is_connected else '<span style="color: #ef4444; font-weight: 700;">DISCONNECTED</span>'
        st.markdown(
            f"""
            <div class="adas-card">
                <div class="metric-label">Connection State</div>
                <div style="font-size: 15px;">{state_text}</div>
                <div class="metric-sub">Auto-polling rate: ~500ms</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with status_col3:
        st.markdown(
            f"""
            <div class="adas-card">
                <div class="metric-label">Last Successful Packet</div>
                <div style="font-size: 15px; font-weight: 700; color: #f1f5f9;">{last_ts}</div>
                <div class="metric-sub">Active buffer: {len(st.session_state['telemetry_history'])} samples</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if not is_connected and last_err:
        st.warning(f"⚠️ Sensor Link Offline: {last_err}")


# Main Dashboard Execution
def main():
    # Render the dynamic 500ms fragment
    live_telemetry_fragment()

    # 5. FUTURE MODULE PLACEHOLDERS (Static section)
    st.markdown("### 4. System Extension Modules")
    f_col1, f_col2, f_col3 = st.columns(3)

    with f_col1:
        st.markdown(
            """
            <div class="adas-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span class="metric-label" style="margin: 0;">Physiological Monitoring</span>
                    <span class="badge-placeholder">Not Connected</span>
                </div>
                <div style="font-size: 14px; font-weight: 700; color: #94a3b8; margin-bottom: 4px;">MAX30102 Optical Sensor</div>
                <div class="metric-sub">Driver pulse oximetry, heart rate, and fatigue telemetry integration.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with f_col2:
        st.markdown(
            """
            <div class="adas-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span class="metric-label" style="margin: 0;">ML Grip Classification</span>
                    <span class="badge-placeholder">Not Implemented</span>
                </div>
                <div style="font-size: 14px; font-weight: 700; color: #94a3b8; margin-bottom: 4px;">Inference Classifier</div>
                <div class="metric-sub">Real-time classification of driver grip anomalies, clenches, and releases.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with f_col3:
        st.markdown(
            """
            <div class="adas-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span class="metric-label" style="margin: 0;">Vehicle Control</span>
                    <span class="badge-placeholder">Not Connected</span>
                </div>
                <div style="font-size: 14px; font-weight: 700; color: #94a3b8; margin-bottom: 4px;">Miniature Actuator Interlock</div>
                <div class="metric-sub">Emergency speed mitigation and torque feedback assistance.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
