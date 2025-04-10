from datetime import datetime
import logging
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database setup
Base = declarative_base()
engine = create_engine('sqlite:///factory_sensors.db')
Session = sessionmaker(bind=engine)


class SensorReading(Base):

    __tablename__ = 'sensor_readings'

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    temperature = Column(Float)
    humidity = Column(Float)
    sound = Column(Float)
    is_anomaly = Column(Boolean, default=False)
    anomaly_score = Column(Float)
    issue_type = Column(String, nullable=True)

    def to_dict(self):
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'temperature': self.temperature,
            'humidity': self.humidity,
            'sound': self.sound,
            'is_anomaly': self.is_anomaly,
            'anomaly_score': self.anomaly_score,
            'issue_type': self.issue_type
        }

    @property
    def features(self):
        return np.array([[self.temperature, self.humidity, self.sound]])


class AnomalyDetector:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.model_path = 'isolation_forest.joblib'
        self.scaler_path = 'scaler.joblib'
        # 10% anomalies
        self.contamination = 0.1
        # Number of trees in the forest
        self.n_estimators = 100

    def load_model(self):

        try:
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            logger.info("Model and scaler loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

    def train(self, readings):
        """Train the anomaly detection model"""
        try:
            if not readings:
                logger.warning("No readings available for training")
                return False

            # Convert readings to features
            X = np.array([[r.temperature, r.humidity, r.sound] for r in readings])
            X_scaled = self.scaler.fit_transform(X)

            # Train model
            self.model = IsolationForest(
                contamination=self.contamination,
                n_estimators=self.n_estimators,
                random_state=42
            )
            self.model.fit(X_scaled)

            # Save model and scaler
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)

            logger.info(f"Model trained successfully with {len(readings)} samples")
            return True

        except Exception as e:
            logger.error(f"Error training model: {e}")
            return False

    def predict(self, reading):

        try:
            if not self.model:
                if not self.load_model():
                    logger.error("Could not load model for prediction")
                    return False, 0.0, None

            X_scaled = self.scaler.transform(reading.features)
            prediction = self.model.predict(X_scaled)[0]
            score = float(self.model.score_samples(X_scaled)[0])
            is_anomaly = bool(prediction == -1)

            issue_type = None
            if is_anomaly:

                if (27.0 <= reading.temperature <= 35.0 and
                    45.0 <= reading.humidity <= 60.0 and
                    70.0 <= reading.sound <= 90.0):
                    issue_type = 'overheating'


                elif (20.0 <= reading.temperature <= 25.0 and
                      56.0 <= reading.humidity <= 70.0 and
                      65.0 <= reading.sound <= 80.0):
                    issue_type = 'ventilation'


                elif (20.0 <= reading.temperature <= 25.0 and
                      40.0 <= reading.humidity <= 55.0 and
                      86.0 <= reading.sound <= 100.0):
                    issue_type = 'noise'


                elif (23.0 <= reading.temperature <= 25.0 and
                      51.0 <= reading.humidity <= 54.0 and
                      76.0 <= reading.sound <= 84.0):
                    issue_type = 'steam_leak'

                else:

                    temp_deviation = abs(reading.temperature - 22.5)
                    humid_deviation = abs(reading.humidity - 45.0)
                    sound_deviation = abs(reading.sound - 75.0)

                    max_deviation = max(temp_deviation, humid_deviation, sound_deviation)
                    
                    if max_deviation == temp_deviation:
                        if reading.temperature > 22.5:
                            issue_type = 'overheating'
                        else:
                            issue_type = 'ventilation'
                    elif max_deviation == humid_deviation:
                        if reading.humidity > 45.0:
                            issue_type = 'ventilation'
                        else:
                            issue_type = 'steam_leak'
                    else:  # sound_deviation is highest
                        if reading.sound > 75.0:
                            issue_type = 'noise'
                        else:
                            issue_type = 'steam_leak'

            return is_anomaly, score, issue_type

        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return False, 0.0, None


# Create database tables
Base.metadata.create_all(engine)
