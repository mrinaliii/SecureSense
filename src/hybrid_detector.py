import re
from src.regex_detector import RegexDetector
from src.predict import TransformerPredictor


class HybridDetector:

    def __init__(self):

        self.regex_detector = RegexDetector()
        self.transformer_detector = TransformerPredictor()

    def detect(self, text):

        regex_results = self.regex_detector.detect(text)
        transformer_results = self.transformer_detector.predict(text)

        heuristic_results = self.detect_capitalized_names(text, transformer_results)

        combined = regex_results + transformer_results + heuristic_results

        return self.remove_overlaps(combined)

    def detect_capitalized_names(self, text, transformer_results):

        detections = []

        existing_spans = {(r["start"], r["end"]) for r in transformer_results}

        words = re.finditer(r"\b[A-Z][a-z]+\b", text)

        for match in words:

            word = match.group()
            start, end = match.start(), match.end()

            # Skip if at start of sentence (after period)
            if start > 1 and text[start - 2] == ".":
                continue

            if (start, end) not in existing_spans:
                detections.append({
                    "entity": word,
                    "label": "PER",
                    "start": start,
                    "end": end
                })

        return detections

    def remove_overlaps(self, detections):

        detections = sorted(detections, key=lambda x: (x["start"], -x["end"]))

        final = []

        for det in detections:
            if not any(
                det["start"] < f["end"] and det["end"] > f["start"]
                for f in final
            ):
                final.append(det)

        return final