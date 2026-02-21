import os
from transformers import pipeline


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "distilbert-pii")


class TransformerPredictor:

    def __init__(self, model_path=MODEL_PATH):

        if not os.path.exists(model_path):
            raise ValueError(f"Model path not found: {model_path}")

        self.ner = pipeline(
            "token-classification",
            model=model_path,
            tokenizer=model_path,
            aggregation_strategy="simple"
        )

    def predict(self, text: str):

        results = self.ner(text)

        formatted = []

        for r in results:
            formatted.append({
                "entity": r["word"],
                "label": r["entity_group"],
                "start": r["start"],
                "end": r["end"]
            })

        return formatted