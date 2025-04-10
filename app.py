from flask import Flask, request, jsonify
from models import SensorReading, AnomalyDetector, Session
import logging
from datetime import datetime
import json

# Set up logging for better debugging and monitoring
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
detector = AnomalyDetector()


def generate_initial_training_data():

    logger.info("Generating initial training data...")
    readings = []

    # Try to load existing training data
    try:
        with open('training_data.json', 'r') as f:
            training_data = json.load(f)
            logger.info(f"Loaded {len(training_data)} training samples from file")
            readings = [
                SensorReading(
                    timestamp=datetime.fromisoformat(data['timestamp']),
                    temperature=data['temperature'],
                    humidity=data['humidity'],
                    sound=data['sound'],
                    is_anomaly=data['is_anomaly'],
                    issue_type=data['anomaly_type']
                )
                for data in training_data
            ]
            return readings
    except FileNotFoundError:
        logger.info("No existing training data found, generating new data...")

    from data_simulator import generate_training_data
    training_data = generate_training_data(num_readings=1000, anomaly_ratio=0.1)

    readings = [
        SensorReading(
            timestamp=datetime.fromisoformat(data['timestamp']),
            temperature=data['temperature'],
            humidity=data['humidity'],
            sound=data['sound'],
            is_anomaly=data['is_anomaly'],
            issue_type=data['anomaly_type']
        )
        for data in training_data
    ]

    with open('training_data.json', 'w') as f:
        json.dump(training_data, f, indent=2)

    anomaly_counts = {}
    for reading in readings:
        if reading.is_anomaly:
            anomaly_counts[reading.issue_type] = anomaly_counts.get(reading.issue_type, 0) + 1

    logger.info(f"Generated {len(readings)} training samples")
    for anomaly_type, count in anomaly_counts.items():
        logger.info(f"- {anomaly_type}: {count} samples")

    return readings


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        logger.info(f"Received data: {data}")

        # Create reading object from request data
        reading = SensorReading(
            timestamp=datetime.fromisoformat(data['timestamp']),
            temperature=float(data['temperature']),
            humidity=float(data['humidity']),
            sound=float(data['sound'])
        )

        # Get prediction from anomaly detector
        is_anomaly, score, issue_type = detector.predict(reading)

        # Store reading and prediction in database
        session = Session()
        reading.is_anomaly = is_anomaly
        reading.anomaly_score = score
        reading.issue_type = issue_type
        session.add(reading)
        session.commit()
        session.close()

        # Prepare response with prediction details
        response = {
            'prediction': {
                'anomaly_score': score,
                'is_anomaly': is_anomaly,
                'issue_type': issue_type
            }
        }

        logger.info(f"Prediction: {response['prediction']}")
        return jsonify(response)

    except Exception as e:
        logger.error(f"Error processing request: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/readings', methods=['GET'])
def get_readings():
    """
    Endpoint for retrieving recent sensor readings.
    Returns the most recent readings, limited by the 'limit' query parameter.
    """
    try:
        limit = request.args.get('limit', default=30, type=int)
        session = Session()
        readings = session.query(SensorReading).order_by(
            SensorReading.timestamp.desc()
        ).limit(limit).all()
        session.close()
        return jsonify([reading.to_dict() for reading in readings])
    except Exception as e:
        logger.error(f"Error fetching readings: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/anomalies', methods=['GET'])
def get_anomalies():
    """
    Endpoint for retrieving recent anomalies.
    Returns the 10 most recent anomalous readings.
    """
    try:
        session = Session()
        anomalies = session.query(SensorReading).filter(
            SensorReading.is_anomaly == True
        ).order_by(SensorReading.timestamp.desc()).limit(10).all()
        session.close()
        return jsonify([anomaly.to_dict() for anomaly in anomalies])
    except Exception as e:
        logger.error(f"Error fetching anomalies: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/statistics', methods=['GET'])
@app.route('/stats', methods=['GET'])
def get_stats():

    try:
        session = Session()
        total_readings = session.query(SensorReading).count()
        total_anomalies = session.query(SensorReading).filter(
            SensorReading.is_anomaly == True
        ).count()


        from sqlalchemy import func
        avg_temp = session.query(func.avg(SensorReading.temperature)).scalar()
        avg_humid = session.query(func.avg(SensorReading.humidity)).scalar()
        avg_sound = session.query(func.avg(SensorReading.sound)).scalar()

        session.close()

        return jsonify({
            'total_readings': total_readings,
            'total_anomalies': total_anomalies,
            'anomaly_rate': round((total_anomalies / total_readings * 100) if total_readings > 0 else 0, 2),
            'average_temperature': round(float(avg_temp), 2) if avg_temp else 0,
            'average_humidity': round(float(avg_humid), 2) if avg_humid else 0,
            'average_sound': round(float(avg_sound), 2) if avg_sound else 0
        })
    except Exception as e:
        logger.error(f"Error calculating statistics: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/latest', methods=['GET'])
def get_latest():

    try:
        session = Session()
        latest = session.query(SensorReading).order_by(
            SensorReading.timestamp.desc()
        ).first()
        session.close()

        if not latest:
            return jsonify({'error': 'No readings available'}), 404

        return jsonify({
            'input_data': {
                'timestamp': latest.timestamp.isoformat(),
                'temperature': latest.temperature,
                'humidity': latest.humidity,
                'sound': latest.sound,
                'is_anomaly': latest.is_anomaly,
                'issue_type': latest.issue_type
            },
            'prediction': {
                'anomaly_score': latest.anomaly_score,
                'is_anomaly': latest.is_anomaly,
                'issue_type': latest.issue_type
            }
        })
    except Exception as e:
        logger.error(f"Error fetching latest reading: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':

    logger.info("Starting server initialization...")
    training_data = generate_initial_training_data()
    if detector.train(training_data):
        logger.info("Model trained successfully with initial data")
    else:
        logger.error("Failed to train model with initial data")

    logger.info("Starting Flask server...")
    app.run(debug=True, port=5000)
