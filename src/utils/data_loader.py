import csv
import json
from pathlib import Path
from typing import Dict, List, Union

import chardet
import pandas as pd


class DataLoader:
    @staticmethod
    def load_csv(file_path: str) -> pd.DataFrame:
        try:
            with open(file_path, "rb") as f:
                result = chardet.detect(f.read())
                encoding = result["encoding"]

            df = pd.read_csv(file_path, encoding=encoding)
            return df.fillna("")
        except Exception as e:
            raise ValueError(f"Error loading CSV: {str(e)}")

    @staticmethod
    def load_json(file_path: str) -> List[Dict]:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, list) else [data]
        except Exception as e:
            raise ValueError(f"Error loading JSON: {str(e)}")

    @staticmethod
    def load_text(file_path: str) -> str:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            raise ValueError(f"Error loading text file: {str(e)}")

    @staticmethod
    def batch_texts(texts: List[str], batch_size: int = 16) -> List[List[str]]:
        return [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]

    @staticmethod
    def sample_data(df: pd.DataFrame, column: str, n_samples: int = 100) -> List[str]:
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")

        sampled = df[column].dropna().sample(min(n_samples, len(df)), random_state=42)
        return sampled.tolist()
