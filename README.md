# IoT Anomaly Detection System

This project implements a real-time anomaly detection system for IoT sensor data in a factory setting. The system monitors temperature, humidity, and sound levels to detect potential anomalies in the production cycle.

## Features

- Real-time sensor data simulation
- Anomaly detection using Isolation Forest algorithm
- RESTful API for predictions
- Stream processing capabilities
- Logging and monitoring

## Project Structure

```
.
├── README.md
├── requirements.txt
├── app.py              # Flask API server
├── model.py            # Anomaly detection model
└── data_simulator.py   # IoT sensor data simulator
```

## Installation

1. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the API server:
```bash
python app.py
```
This will train the model and start the Flask server on port 5000.

2. In a separate terminal, run the data simulator:
```bash
python data_simulator.py
```
This will start generating simulated sensor data and sending it to the API.

## API Endpoints

- `GET /health`: Health check endpoint
- `POST /predict`: Submit sensor data for anomaly detection

### Example API Request

```json
{
    "timestamp": "2024-01-20T10:30:00",
    "temperature": 25.5,
    "humidity": 48.2,
    "sound": 62.1
}
```

### Example API Response

```json
{
    "input_data": {
        "timestamp": "2024-01-20T10:30:00",
        "temperature": 25.5,
        "humidity": 48.2,
        "sound": 62.1
    },
    "prediction": {
        "anomaly_score": 0.85,
        "is_anomaly": false
    }
}
```

## Monitoring

The system includes logging functionality that tracks:
- Incoming sensor data
- Predictions
- System errors and exceptions

Logs are printed to the console and can be redirected to a file if needed.

## Customization

You can adjust the following parameters in the code:
- `data_simulator.py`: Baseline values and variation ranges for sensors
- `model.py`: Anomaly detection model parameters
- `app.py`: API configuration and logging settings 