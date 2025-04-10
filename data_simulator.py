import time
import random
import json
import requests
from datetime import datetime, timedelta
import numpy as np


class IoTSensorSimulator:
    """
    Simulates an IoT sensor in a factory environment.
    Generates realistic sensor readings and controlled anomalies.
    """
    def __init__(self):

        self.base_temperature = 22.0
        self.base_humidity = 45.0
        self.base_sound = 65.0

        # Safe operating ranges for normal conditions
        self.temp_range = (18.0, 26.0)
        self.humidity_range = (35.0, 55.0)
        self.sound_range = (60.0, 85.0)


        self.anomaly_types = {
            'overheating': {
                'temp_range': (27.0, 35.0),
                'humidity_range': (45.0, 60.0),
                'sound_range': (70.0, 90.0),
                'description': 'High temperature with moderate humidity and sound increase'
            },
            'ventilation': {
                'temp_range': (20.0, 25.0),
                'humidity_range': (56.0, 70.0),
                'sound_range': (65.0, 80.0),
                'description': 'High humidity with normal temperature and moderate sound'
            },
            'noise': {
                'temp_range': (20.0, 25.0),
                'humidity_range': (40.0, 55.0),
                'sound_range': (86.0, 100.0),
                'description': 'High sound levels with normal temperature and humidity'
            },
            'steam_leak': {
                'temp_range': (23.0, 25.0),
                'humidity_range': (51.0, 54.0),
                'sound_range': (76.0, 84.0),
                'description': 'Moderate increases in all sensors'
            }
        }

        # Initialize trending components for smooth transitions
        self.temp_trend = 0
        self.humidity_trend = 0
        self.sound_trend = 0

    def _update_trends(self):

        # Limit trend changes to prevent sudden spikes
        self.temp_trend = max(-2.0, min(2.0, self.temp_trend + random.uniform(-0.05, 0.05)))
        self.humidity_trend = max(-5.0, min(5.0, self.humidity_trend + random.uniform(-0.1, 0.1)))
        self.sound_trend = max(-3.0, min(3.0, self.sound_trend + random.uniform(-0.15, 0.15)))

    def _generate_anomaly(self, anomaly_type=None):

        if anomaly_type is None:
            anomaly_type = random.choice(list(self.anomaly_types.keys()))
        ranges = self.anomaly_types[anomaly_type]

        # Generate values within the anomaly ranges
        temp = random.uniform(ranges['temp_range'][0], ranges['temp_range'][1])
        humidity = random.uniform(ranges['humidity_range'][0], ranges['humidity_range'][1])
        sound = random.uniform(ranges['sound_range'][0], ranges['sound_range'][1])

        return (temp, humidity, sound), anomaly_type

    def generate_reading(self, force_anomaly_type=None):

        self._update_trends()

        # 5% chance of generating an anomaly
        if random.random() < 0.05 or force_anomaly_type:
            values, anomaly_type = self._generate_anomaly(force_anomaly_type)
            temp, humidity, sound = values
            is_anomaly = True
        else:

            temp = self.base_temperature + self.temp_trend + random.uniform(-0.5, 0.5)
            humidity = self.base_humidity + self.humidity_trend + random.uniform(-1.0, 1.0)
            sound = self.base_sound + self.sound_trend + random.uniform(-1.0, 1.0)
            anomaly_type = None
            is_anomaly = False


        temp = max(self.temp_range[0], min(self.temp_range[1], temp))
        humidity = max(self.humidity_range[0], min(self.humidity_range[1], humidity))
        sound = max(self.sound_range[0], min(self.sound_range[1], sound))

        return {
            'timestamp': datetime.utcnow().isoformat(),
            'temperature': round(temp, 2),
            'humidity': round(humidity, 2),
            'sound': round(sound, 2),
            'is_anomaly': is_anomaly,
            'anomaly_type': anomaly_type
        }


def generate_training_data(num_readings=1000, anomaly_ratio=0.1):

    simulator = IoTSensorSimulator()
    training_data = []

    # Calculate number of anomalies needed
    num_anomalies = int(num_readings * anomaly_ratio)
    anomalies_per_type = num_anomalies // len(simulator.anomaly_types)

    # Generate normal readings
    for _ in range(num_readings - num_anomalies):
        training_data.append(simulator.generate_reading())

    # Generate anomalies for each type
    for anomaly_type in simulator.anomaly_types.keys():
        for _ in range(anomalies_per_type):
            training_data.append(simulator.generate_reading(force_anomaly_type=anomaly_type))

    # Shuffle the data for better training
    random.shuffle(training_data)
    return training_data


def simulate_and_send_data(api_endpoint='http://localhost:5000/predict'):

    simulator = IoTSensorSimulator()

    print("Starting IoT sensor simulation...")
    print("Normal operating ranges:")
    print(f"Temperature: {simulator.temp_range[0]}-{simulator.temp_range[1]}°C")
    print(f"Humidity: {simulator.humidity_range[0]}-{simulator.humidity_range[1]}%")
    print(f"Sound: {simulator.sound_range[0]}-{simulator.sound_range[1]} dB")
    print("\nAnomaly types and their conditions:")
    for anomaly_type, ranges in simulator.anomaly_types.items():
        print(f"- {anomaly_type}:")
        print(f"  {ranges['description']}")
        print(f"  Temperature: {ranges['temp_range'][0]}-{ranges['temp_range'][1]}°C")
        print(f"  Humidity: {ranges['humidity_range'][0]}-{ranges['humidity_range'][1]}%")
        print(f"  Sound: {ranges['sound_range'][0]}-{ranges['sound_range'][1]} dB")

    while True:
        try:
            data = simulator.generate_reading()
            response = requests.post(api_endpoint, json=data)
            print("\nSensor Reading:", json.dumps(data, indent=2))
            print("Prediction Response:", json.dumps(response.json(), indent=2))

            time.sleep(random.uniform(0.5, 1.0))

        except requests.exceptions.RequestException as e:
            print(f"Error sending data to API: {e}")
            time.sleep(5)
        except KeyboardInterrupt:
            print("\nSimulation stopped by user")
            break


if __name__ == "__main__":
    # Start real-time simulation
    simulate_and_send_data()
