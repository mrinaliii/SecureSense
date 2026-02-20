from transformers import pipeline


class TransformerPredictor:

    def __init__(self, model_path="../models/distilbert-pii"):
        self.ner = pipeline(
            "token-classification",
            model=model_path,
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