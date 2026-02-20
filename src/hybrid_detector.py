from typing import List, Dict
from regex_detector import RegexDetector
from predict import TransformerPredictor


class HybridDetector:

    def __init__(self):
        self.regex_detector = RegexDetector()
        self.transformer_detector = TransformerPredictor()

    def merge_results(self, regex_results: List[Dict], transformer_results: List[Dict]) -> List[Dict]:

        combined = regex_results.copy()

        for t in transformer_results:
            overlap = False

            for r in regex_results:
                if not (t["end"] <= r["start"] or t["start"] >= r["end"]):
                    overlap = True
                    break

            if not overlap:
                combined.append(t)

        return combined

    def detect(self, text: str) -> List[Dict]:

        regex_results = self.regex_detector.detect(text)
        transformer_results = self.transformer_detector.predict(text)

        final_results = self.merge_results(regex_results, transformer_results)

        return sorted(final_results, key=lambda x: x["start"])