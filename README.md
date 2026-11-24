# BIO-STEER
### AI/ML-Based Driver Grip Monitoring and Intelligent Safety Prototype

BIO-STEER is an embedded AI/ML-based driver monitoring prototype designed to monitor steering-wheel grip patterns and demonstrate an early safety response through a miniature vehicle system.

The project combines **ESP32-based sensor acquisition, Python data processing, machine learning, and a real-time monitoring dashboard** to study driver grip behaviour.

> **Project status:** Academic prototype under development. This system is not a production-ready ADAS or vehicle braking system.

---

## 1. Project Overview

BIO-STEER uses Force Sensitive Resistors (FSRs) mounted on a steering-wheel prototype to measure grip pressure.

The collected sensor readings are transmitted wirelessly to a Python backend, where they can be processed, analysed and used for grip-pattern classification.

The system aims to demonstrate how abnormal grip-pattern events could trigger an early safety response in a controlled miniature vehicle setup.

### Key Objectives

- Monitor left-hand and right-hand steering grip pressure.
- Acquire real-time sensor readings using ESP32.
- Collect and label grip-pattern datasets.
- Develop machine learning models for grip classification.
- Visualise live sensor readings through a dashboard.
- Demonstrate a controlled safety response using a miniature vehicle.

---

## 2. System Architecture

```text
       STEERING WHEEL PROTOTYPE
        ┌─────────────────────┐
        │   Left FSR Sensor   │
        │   Right FSR Sensor  │
        └──────────┬──────────┘
                   │
                   ▼
             ┌───────────┐
             │  ESP32 #1 │
             │ Sensor    │
             │ Acquisition
             └─────┬─────┘
                   │
              Wi-Fi / HTTP
                   │
                   ▼
          ┌──────────────────┐
          │  Python Backend  │
          │ Sensor Receiver  │
          │ Data Collection  │
          └────────┬─────────┘
                   │
          ┌────────┴─────────┐
          ▼                  ▼
   ┌──────────────┐   ┌──────────────┐
   │ ML Pipeline  │   │  Streamlit   │
   │ Training &   │   │  Dashboard   │
   │ Prediction   │   │              │
   └──────────────┘   └──────────────┘
          │
          ▼
   Future Safety Response
          │
          ▼
     ESP32 #2
          │
          ▼
   Miniature Vehicle
```

---

## 3. Hardware Components

| Component | Purpose |
|---|---|
| ESP32 Development Board | Wireless sensor acquisition |
| FSR Sensors (2) | Left and right grip-pressure measurement |
| 10kΩ Resistors | FSR voltage-divider circuits |
| ESP32 Development Board #2 | Planned miniature vehicle control |
| Motor Driver | Planned motor control |
| DC Motors | Miniature vehicle movement |
| MAX30102 | Planned physiological sensing; currently not connected |

---

## 4. Software and Technologies

- **Programming:** Python, C/C++ (ESP32 firmware)
- **Embedded Platform:** ESP32
- **Communication:** Wi-Fi, HTTP, JSON
- **Backend:** Python
- **Frontend:** Streamlit
- **Visualisation:** Plotly
- **Data Processing:** NumPy, Pandas
- **Machine Learning:** Scikit-learn
- **Model Storage:** Joblib

---

## 5. Grip Classification

The dataset contains five grip-related labels:

| Class | Description |
|---|---|
| `no_grip` | No detected steering grip |
| `steady_grip` | Normal steady grip |
| `firm_grip` | Increased grip pressure |
| `sudden_clench` | Sudden increase in grip pressure |
| `release` | Release of steering grip |

The machine learning pipeline uses sensor-derived features to train and evaluate a classification model.

**Important:** FSR sensors measure pressure, not emotions, panic, stress or driver intent. A sudden clench is treated only as a measurable grip-pattern event.

---

## 6. Repository Structure

```text
SIC_AIML/
│
├── backend/
│   ├── __init__.py
│   ├── main.py
│   ├── sensor_receiver.py
│   └── labelled_collector.py
│
├── communication/
│   └── vehicle_control.py
│
├── data/
│   ├── labelled_sensor_data.csv
│   ├── prepared_sensor_data.csv
│   ├── sensor_data.csv
│   └── .gitkeep
│
├── frontend/
│   └── app.py
│
├── ml/
│   ├── prepare_dataset.py
│   ├── train_model.py
│   ├── predict.py
│   └── model.joblib
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## 7. Installation and Setup

### Prerequisites

- Python 3.11
- Git
- ESP32 development board
- Arduino IDE or compatible ESP32 firmware environment

### Clone the Repository

```bash
git clone https://github.com/shreesha509/SIC_Smart_ADAS_SYSTEM.git
cd SIC_Smart_ADAS_SYSTEM
```

### Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 8. Running the Dashboard

Start the Streamlit application:

```bash
streamlit run frontend/app.py --server.port 8501
```

Open the following URL in your browser:

```text
http://localhost:8501
```

The dashboard provides:

- ESP32 connection status
- Live left and right FSR readings
- Sensor value indicators
- Real-time grip-pressure graphs
- Network and packet status
- Hardware integration status

The ESP32 and computer must be connected to the same network for live sensor communication.

---

## 9. Machine Learning Pipeline

The machine learning workflow consists of three main stages.

**1. Dataset Preparation**

```bash
python ml/prepare_dataset.py
```

Processes labelled sensor data, extracts features and prepares the dataset for model training.

**2. Model Training**

```bash
python ml/train_model.py
```

Trains and evaluates the grip-classification model.

**3. Prediction**

```bash
python ml/predict.py
```

Runs the prediction module.

The current model is experimental. Its performance, particularly for the `sudden_clench` class, requires further improvement and independent validation before any safety-control integration.

---

## 10. Current Development Status

| Module | Status |
|---|---|
| ESP32 sensor acquisition | Implemented |
| Dual FSR integration | Implemented |
| Wi-Fi sensor communication | Implemented |
| Python sensor receiver | Implemented |
| Raw sensor data collection | Implemented |
| Labelled data collection | Implemented |
| Dataset preparation | Implemented |
| ML model training | Implemented |
| ML prediction pipeline | Implemented |
| Streamlit monitoring dashboard | Implemented |
| MAX30102 integration | Pending |
| Miniature vehicle integration | Pending |
| End-to-end safety response validation | Pending |

---

## 11. Safety and Limitations

BIO-STEER is an academic proof-of-concept.

- Grip pressure alone cannot establish driver fatigue, stress or panic.
- Machine learning predictions are experimental and must not independently control real vehicle braking.
- The miniature vehicle demonstration requires independent manual stopping and a fail-safe stop on communication loss.
- Hardware and model performance require further testing under different grip conditions and sensor placements.
- Physiological sensing and vehicle integration remain future development tasks.

All vehicle-response experiments should be conducted on a controlled miniature setup, not a real vehicle.

---

## 12. Future Enhancements

- Improve grip classification using additional independent sensor sessions.
- Improve model generalisation and sudden-clench detection.
- Integrate physiological sensors when hardware is available.
- Implement reliable ESP32-to-ESP32 communication.
- Complete miniature vehicle integration.
- Develop a unified real-time dashboard for sensor, ML and vehicle status.
- Validate the complete system under controlled experimental conditions.

---

## 13. Project Repository

**GitHub:** https://github.com/shreesha509/SIC_Smart_ADAS_SYSTEM

**Project:** BIO-STEER – AI/ML-Based Driver Grip Monitoring and Intelligent Safety Prototype

---

*Developed as an academic embedded systems and AI/ML project.*
