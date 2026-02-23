import re


class RegexDetector:

    def __init__(self):

        self.patterns = {
            "PHONE": re.compile(r"\b\d{10}\b"),
            "EMAIL": re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b"),
            "AADHAAR": re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b"),
            "PAN": re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"),
            "CREDIT_CARD": re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\b"),
        }

    def detect(self, text):

        detections = []

        for label, pattern in self.patterns.items():
            for match in pattern.finditer(text):
                detections.append({
                    "entity": match.group(),
                    "label": label,
                    "start": match.start(),
                    "end": match.end()
                })

        return detections