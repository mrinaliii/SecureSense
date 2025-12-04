import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from src.detector.classifier import TransformerClassifier
from src.detector.regex_patterns import RegexDetector


def load_training_data():
    data = []

    samples = [
        ("john.doe@example.com sent his PAN ABCDE1234F to process payment", True),
        ("Contact me at 9876543210 for details", True),
        ("The weather is nice today", False),
        ("Aadhaar: 1234 5678 9012 needs verification", True),
        ("Meeting scheduled for tomorrow at 3 PM", False),
        ("Credit card 4111-1111-1111-1111 used for transaction", True),
        ("Regular project update meeting", False),
        ("Patient ID: P-78901, Diagnosis: ICD-10 code J45", True),
        ("Team lunch at the new restaurant", False),
        ("Bank account 12345678901234 for salary deposit", True),
    ]

    for text, is_sensitive in samples:
        data.append({"text": text, "is_sensitive": is_sensitive})

    return pd.DataFrame(data)


def evaluate_model(df, classifier):
    texts = df["text"].tolist()
    true_labels = df["is_sensitive"].astype(int).tolist()

    predictions = classifier.predict(texts, threshold=0.85)
    pred_labels = [1 if p["is_sensitive"] else 0 for p in predictions]

    print(
        classification_report(
            true_labels, pred_labels, target_names=["Non-sensitive", "Sensitive"]
        )
    )

    accuracy = np.mean(
        [1 if true == pred else 0 for true, pred in zip(true_labels, pred_labels)]
    )
    print(f"Accuracy: {accuracy:.2%}")

    return accuracy


def main():
    print("Training Sensitive Data Classifier...")

    df = load_training_data()
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    classifier = TransformerClassifier()

    print("\nEvaluating on test set:")
    accuracy = evaluate_model(test_df, classifier)

    joblib.dump(classifier, "models/classifier.joblib")
    print(f"\nModel saved to models/classifier.joblib")

    regex_detector = RegexDetector()

    config = {
        "model_accuracy": float(accuracy),
        "training_samples": len(df),
        "last_trained": pd.Timestamp.now().isoformat(),
    }

    with open("models/model_info.yaml", "w") as f:
        yaml.dump(config, f)

    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()
