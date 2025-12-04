import json
import logging
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

from src.detector.classifier import TransformerClassifier
from src.detector.regex_patterns import RegexDetector
from src.utils.data_loader import DataLoader
from src.utils.helpers import anonymize_text

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(self, config_path="config/config.yaml"):
        self.config = self._load_config(config_path)
        self.regex_detector = RegexDetector()
        self.transformer = TransformerClassifier()
        self.data_loader = DataLoader()
        self.model = None
        self.scaler = StandardScaler()

    def _load_config(self, config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def create_training_data(self, n_samples=1000):
        """Create synthetic training data with sensitive and non-sensitive examples"""
        logger.info(f"Creating training data with {n_samples} samples...")

        sensitive_patterns = [
            ("My email is test@example.com", True),
            ("PAN: ABCDE1234F", True),
            ("Aadhaar: 1234 5678 9012", True),
            ("Phone: 9876543210", True),
            ("Credit card: 4111-1111-1111-1111", True),
            ("Bank account: 12345678901234", True),
            ("Patient ID: P-78901", True),
            ("SSN: 123-45-6789", True),
            ("Password: Secret123!", True),
            ("API Key: sk_test_abcdef123456", True),
        ]

        non_sensitive_patterns = [
            ("The weather is nice today", False),
            ("Meeting at 3 PM tomorrow", False),
            ("Please review this document", False),
            ("Project deadline is Friday", False),
            ("Team lunch at the new restaurant", False),
            ("Weekly status report attached", False),
            ("Can you help with this task?", False),
            ("The quick brown fox jumps", False),
            ("Sample text for testing", False),
            ("Hello world example", False),
        ]

        data = []

        for text, label in sensitive_patterns * (n_samples // 10):
            data.append(
                {
                    "text": text,
                    "label": label,
                    "label_name": "sensitive" if label else "non_sensitive",
                }
            )

        for text, label in non_sensitive_patterns * (n_samples // 10):
            data.append(
                {
                    "text": text,
                    "label": label,
                    "label_name": "sensitive" if label else "non_sensitive",
                }
            )

        import random

        domains = ["gmail.com", "yahoo.com", "outlook.com", "company.com"]
        names = ["john", "jane", "alex", "smith", "kumar", "sharma", "patel"]

        for _ in range(n_samples // 4):
            if random.random() > 0.5:
                email = f"{random.choice(names)}.{random.choice(names)}@{random.choice(domains)}"
                text = f"Contact me at {email} for details"
                data.append({"text": text, "label": True, "label_name": "sensitive"})
            else:
                text = f"Random text {random.randint(1000, 9999)} for training"
                data.append(
                    {"text": text, "label": False, "label_name": "non_sensitive"}
                )

        df = pd.DataFrame(data)
        logger.info(f"Created dataset: {len(df)} samples")
        logger.info(f"Class distribution:\n{df['label_name'].value_counts()}")

        return df

    def extract_features(self, texts):
        """Extract hybrid features: regex matches + transformer embeddings"""
        logger.info("Extracting features from texts...")

        all_features = []

        for text in texts:
            features = []

            regex_results = self.regex_detector.detect(text)

            pattern_counts = {}
            for finding in regex_results:
                pattern_type = finding["type"]
                pattern_counts[pattern_type] = pattern_counts.get(pattern_type, 0) + 1

            patterns = [
                "email",
                "pan_card",
                "aadhaar",
                "phone_india",
                "credit_card",
                "bank_account",
            ]
            for pattern in patterns:
                features.append(pattern_counts.get(pattern, 0))

            features.append(len(text))

            features.append(text.count("@"))

            features.append(text.count("."))

            features.append(sum(c.isdigit() for c in text) / max(len(text), 1))

            features.append(sum(c.isupper() for c in text) / max(len(text), 1))

            try:
                embeddings = self.transformer.extract_embeddings([text])

                features.extend(embeddings[0][:10].tolist())
            except Exception as e:
                logger.warning(f"Embedding extraction failed: {e}")
                features.extend([0] * 10)

            all_features.append(features)

        feature_names = (
            [f"regex_{p}" for p in patterns]
            + ["text_len", "at_count", "dot_count", "digit_ratio", "upper_ratio"]
            + [f"embed_{i}" for i in range(10)]
        )

        logger.info(
            f"Feature extraction complete. Total features: {len(feature_names)}"
        )

        return np.array(all_features), feature_names

    def train_model(self, X_train, y_train):
        """Train RandomForest classifier"""
        logger.info("Training RandomForest classifier...")

        X_train_scaled = self.scaler.fit_transform(X_train)

        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        )

        cv_scores = cross_val_score(
            self.model, X_train_scaled, y_train, cv=5, scoring="f1"
        )
        logger.info(f"Cross-validation F1 scores: {cv_scores}")
        logger.info(
            f"Mean CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})"
        )

        self.model.fit(X_train_scaled, y_train)

        logger.info("Model training complete")
        return self.model

    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance"""
        logger.info("Evaluating model...")

        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]

        report = classification_report(
            y_test, y_pred, target_names=["non_sensitive", "sensitive"]
        )
        logger.info("Classification Report:\n" + report)

        cm = confusion_matrix(y_test, y_pred)
        logger.info(f"Confusion Matrix:\n{cm}")

        roc_auc = roc_auc_score(y_test, y_pred_proba)
        logger.info(f"ROC-AUC Score: {roc_auc:.4f}")

        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "roc_auc": float(roc_auc),
            "confusion_matrix": cm.tolist(),
            "support": {
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp),
            },
        }

        return metrics

    def save_model(self, model_path="models/classifier.joblib"):
        """Save trained model and metadata"""
        logger.info(f"Saving model to {model_path}...")

        import os

        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        joblib.dump(self.model, model_path)

        scaler_path = model_path.replace(".joblib", "_scaler.joblib")
        joblib.dump(self.scaler, scaler_path)

        features_path = model_path.replace(".joblib", "_features.json")

        metadata = {
            "model_type": "RandomForestClassifier",
            "n_estimators": 100,
            "max_depth": 10,
            "trained_date": datetime.now().isoformat(),
            "feature_count": self.scaler.n_features_in_
            if hasattr(self.scaler, "n_features_in_")
            else "unknown",
            "classes": self.model.classes_.tolist() if self.model is not None else [],
            "version": "1.0.0",
        }

        metadata_path = model_path.replace(".joblib", "_metadata.yaml")
        with open(metadata_path, "w") as f:
            yaml.dump(metadata, f, default_flow_style=False)

        logger.info(f"Model saved: {model_path}")
        logger.info(f"Scaler saved: {scaler_path}")
        logger.info(f"Metadata saved: {metadata_path}")

        return model_path, scaler_path, metadata_path

    def load_model(self, model_path="models/classifier.joblib"):
        """Load trained model"""
        logger.info(f"Loading model from {model_path}...")

        self.model = joblib.load(model_path)

        scaler_path = model_path.replace(".joblib", "_scaler.joblib")
        self.scaler = joblib.load(scaler_path)

        logger.info("Model loaded successfully")
        return self.model

    def predict_single(self, text):
        """Make prediction for single text"""
        if self.model is None:
            raise ValueError(
                "Model not trained or loaded. Call train_model() or load_model() first."
            )

        features, _ = self.extract_features([text])
        features_scaled = self.scaler.transform(features)

        prediction = self.model.predict(features_scaled)[0]
        probability = self.model.predict_proba(features_scaled)[0]

        return {
            "text": anonymize_text(text, ["email", "pan_card", "aadhaar"]),
            "prediction": bool(prediction),
            "probability_sensitive": float(probability[1]),
            "probability_non_sensitive": float(probability[0]),
            "confidence": float(max(probability)),
        }

    def run_training_pipeline(self, test_size=0.2, save_model=True):
        """Complete training pipeline"""
        logger.info("=" * 60)
        logger.info("Starting Training Pipeline")
        logger.info("=" * 60)

        start_time = datetime.now()

        df = self.create_training_data(n_samples=1000)

        X, feature_names = self.extract_features(df["text"].tolist())
        y = df["label"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        logger.info(f"Training set: {X_train.shape[0]} samples")
        logger.info(f"Test set: {X_test.shape[0]} samples")

        self.train_model(X_train, y_train)

        metrics = self.evaluate_model(X_test, y_test)

        if save_model:
            model_path, scaler_path, metadata_path = self.save_model()

            with open(metadata_path, "r") as f:
                metadata = yaml.safe_load(f)

            metadata.update(
                {
                    "performance_metrics": metrics,
                    "training_time_minutes": (
                        datetime.now() - start_time
                    ).total_seconds()
                    / 60,
                    "feature_names": feature_names,
                }
            )

            with open(metadata_path, "w") as f:
                yaml.dump(metadata, f, default_flow_style=False)

        elapsed_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Training pipeline completed in {elapsed_time:.2f} seconds")
        logger.info("=" * 60)

        return metrics


def main():
    """Main function for command line execution"""
    import argparse

    parser = argparse.ArgumentParser(description="Train Sensitive Data Classifier")
    parser.add_argument(
        "--config", default="config/config.yaml", help="Config file path"
    )
    parser.add_argument(
        "--samples", type=int, default=1000, help="Number of training samples"
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2, help="Test set size ratio"
    )
    parser.add_argument(
        "--save-model", action="store_true", default=True, help="Save trained model"
    )
    parser.add_argument(
        "--output-dir", default="models", help="Output directory for models"
    )

    args = parser.parse_args()

    trainer = ModelTrainer(args.config)

    metrics = trainer.run_training_pipeline(
        test_size=args.test_size, save_model=args.save_model
    )

    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print("=" * 60)

    print("\nTEST PREDICTIONS:")
    test_examples = [
        "My email is john.doe@example.com and PAN is ABCDE1234F",
        "The weather is nice today for a walk",
        "Contact me at 9876543210 for urgent matters",
        "Meeting scheduled for tomorrow at 3 PM",
    ]

    for example in test_examples:
        result = trainer.predict_single(example)
        print(f"\nText: {result['text'][:50]}...")
        print(f"Prediction: {'SENSITIVE' if result['prediction'] else 'NON-SENSITIVE'}")
        print(f"Confidence: {result['confidence']:.2%}")


if __name__ == "__main__":
    main()
