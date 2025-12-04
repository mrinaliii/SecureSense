from typing import Dict, List

import joblib
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer


class TransformerClassifier:
    def __init__(self, model_name="distilbert-base-uncased"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.scaler = StandardScaler()
        self._init_scaler()

    def _init_scaler(self):
        dummy_data = np.random.randn(100, 768)
        self.scaler.fit(dummy_data)

    def extract_embeddings(self, texts: List[str]) -> np.ndarray:
        embeddings = []

        for text in texts:
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True, padding=True, max_length=512
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                embeddings.append(embedding)

        embeddings = np.vstack(embeddings)
        return self.scaler.transform(embeddings)

    def predict(self, texts: List[str], threshold: float = 0.85) -> List[Dict]:
        embeddings = self.extract_embeddings(texts)
        results = []

        for text, embedding in zip(texts, embeddings):
            sensitivity_score = self._compute_sensitivity_score(embedding)
            is_sensitive = sensitivity_score > threshold

            results.append(
                {
                    "text": text,
                    "sensitivity_score": float(sensitivity_score),
                    "is_sensitive": bool(is_sensitive),
                    "embedding": embedding.tolist(),
                }
            )

        return results

    def _compute_sensitivity_score(self, embedding: np.ndarray) -> float:
        norms = np.linalg.norm(embedding)
        entropy = -np.sum(embedding * np.log(embedding + 1e-10))
        return float(0.6 * norms + 0.4 * entropy / 100)
