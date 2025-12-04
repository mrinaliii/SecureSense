import hashlib
from collections import deque
from datetime import datetime

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class AnomalyDetector:
    def __init__(self, contamination=0.1):
        self.model = IsolationForest(
            contamination=contamination, random_state=42, n_estimators=100
        )
        self.scaler = StandardScaler()
        self.access_patterns = deque(maxlen=1000)
        self.user_hashes = {}
        self._trained = False

    def extract_features(self, request_data: Dict) -> np.ndarray:
        features = []

        timestamp = datetime.now()
        hour = timestamp.hour
        day_of_week = timestamp.weekday()

        features.extend([hour, day_of_week])

        user_id = request_data.get("user_id", "anonymous")
        if user_id not in self.user_hashes:
            self.user_hashes[user_id] = hashlib.md5(user_id.encode()).hexdigest()[:8]

        user_hash_int = int(self.user_hashes[user_id], 16)
        features.append(user_hash_int % 100)

        data_size = len(request_data.get("text", ""))
        features.append(min(data_size, 10000) / 10000)

        sensitive_count = request_data.get("sensitive_count", 0)
        features.append(min(sensitive_count, 100) / 100)

        return np.array(features).reshape(1, -1)

    def detect_anomaly(self, request_data: Dict) -> Dict:
        features = self.extract_features(request_data)

        if self._trained and len(self.access_patterns) > 100:
            features_scaled = self.scaler.transform(features)
            is_anomaly = self.model.predict(features_scaled)[0] == -1
            anomaly_score = float(self.model.score_samples(features_scaled)[0])
        else:
            is_anomaly = False
            anomaly_score = 0.0

        self.access_patterns.append(features.flatten())

        if len(self.access_patterns) % 50 == 0 and len(self.access_patterns) >= 100:
            self._train_model()

        return {
            "is_anomaly": bool(is_anomaly),
            "anomaly_score": anomaly_score,
            "features": features.flatten().tolist(),
            "timestamp": datetime.now().isoformat(),
        }

    def _train_model(self):
        if len(self.access_patterns) < 100:
            return

        data = np.array(self.access_patterns)
        self.scaler.fit(data)
        data_scaled = self.scaler.transform(data)
        self.model.fit(data_scaled)
        self._trained = True

    def get_metrics(self) -> Dict:
        total_requests = len(self.access_patterns)
        anomalies = sum(
            1
            for pattern in self.access_patterns
            if self.detect_anomaly({"text": ""})["is_anomaly"]
        )

        return {
            "total_requests": total_requests,
            "anomaly_count": anomalies,
            "anomaly_rate": anomalies / max(total_requests, 1),
            "is_trained": self._trained,
        }
