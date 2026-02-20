import re
from typing import List, Dict


class RegexDetector:

    def __init__(self):
        self.patterns = {
            "EMAIL": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
            "PHONE": re.compile(r"\b\d{10}\b"),
            "CREDIT_CARD": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
            "IP_ADDRESS": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
            "AADHAAR": re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b")
        }

    def detect(self, text: str) -> List[Dict]:

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