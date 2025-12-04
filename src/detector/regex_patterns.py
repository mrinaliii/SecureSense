import re
from typing import Dict, List, Tuple

import yaml


class RegexDetector:
    def __init__(self, patterns_file="config/patterns.yaml"):
        with open(patterns_file, "r") as f:
            config = yaml.safe_load(f)
        self.patterns = config["regex_patterns"]
        self.data_types = config["data_types"]
        self.compiled_patterns = {}
        self._compile_patterns()

    def _compile_patterns(self):
        for name, config in self.patterns.items():
            self.compiled_patterns[name] = {
                "pattern": re.compile(config["pattern"], re.IGNORECASE),
                "confidence": config["confidence"],
            }

    def detect(self, text: str) -> List[Dict]:
        findings = []
        text_lower = text.lower()

        for name, detector in self.compiled_patterns.items():
            matches = detector["pattern"].findall(text)
            for match in matches:
                if self._validate_match(name, match):
                    findings.append(
                        {
                            "type": name,
                            "value": match,
                            "confidence": detector["confidence"],
                            "method": "regex",
                            "position": text.find(match),
                        }
                    )

        return findings

    def _validate_match(self, pattern_name: str, match: str) -> bool:
        if pattern_name == "pan_card":
            return self._validate_pan(match)
        elif pattern_name == "aadhaar":
            return self._validate_aadhaar(match)
        return True

    def _validate_pan(self, pan: str) -> bool:
        if len(pan) != 10:
            return False
        fifth_char = pan[4]
        return fifth_char.isalpha()

    def _validate_aadhaar(self, aadhaar: str) -> bool:
        clean_aadhaar = re.sub(r"\s", "", aadhaar)
        if len(clean_aadhaar) != 12:
            return False
        if clean_aadhaar.startswith("0") or clean_ahaar.startswith("1"):
            return False
        return True

    def classify_data_type(self, findings: List[Dict]) -> Dict[str, List]:
        classification = {dtype: [] for dtype in self.data_types.keys()}

        for finding in findings:
            for dtype, patterns in self.data_types.items():
                if finding["type"] in patterns:
                    classification[dtype].append(finding)

        return classification
