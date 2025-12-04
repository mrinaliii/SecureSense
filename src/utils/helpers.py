import hashlib
import json
import uuid
from datetime import datetime
from typing import Any, Dict, List


def format_results(results: List[Dict], format_type: str = "pretty") -> str:
    if format_type == "json":
        return json.dumps(results, indent=2, ensure_ascii=False)

    elif format_type == "pretty":
        output = []
        for i, result in enumerate(results, 1):
            output.append(f"\n{'=' * 50}")
            output.append(f"Finding #{i}")
            output.append(f"{'=' * 50}")

            for key, value in result.items():
                if isinstance(value, dict):
                    output.append(f"{key}:")
                    for k, v in value.items():
                        output.append(f"  {k}: {v}")
                elif isinstance(value, list):
                    output.append(f"{key}:")
                    for item in value[:3]:
                        output.append(f"  - {item}")
                    if len(value) > 3:
                        output.append(f"  ... and {len(value) - 3} more")
                else:
                    output.append(f"{key}: {value}")

        return "\n".join(output)

    elif format_type == "csv":
        import csv
        import io

        output = io.StringIO()
        if results:
            writer = csv.DictWriter(output, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        return output.getvalue()

    return str(results)


def generate_report_id() -> str:
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    random_hash = hashlib.md5(str(uuid.uuid4()).encode()).hexdigest()[:8]
    return f"REPORT_{timestamp}_{random_hash}"


def calculate_risk_score(findings: List[Dict], anomaly_score: float) -> float:
    if not findings:
        return anomaly_score * 0.5

    severity_weights = {"CRITICAL": 1.0, "HIGH": 0.7, "MEDIUM": 0.4, "LOW": 0.2}

    base_score = 0.0
    for finding in findings:
        severity = finding.get("severity", "LOW")
        weight = severity_weights.get(severity, 0.2)
        base_score += weight * 0.1

    final_score = min(0.6 * base_score + 0.4 * anomaly_score, 1.0)
    return round(final_score, 3)


def anonymize_text(text: str, sensitive_patterns: List[str]) -> str:
    import re

    anonymized = text

    for pattern in sensitive_patterns:
        if pattern == "email":
            anonymized = re.sub(
                r"([a-zA-Z0-9._%+-]+)@([a-zA-Z0-9.-]+)\.([a-zA-Z]{2,})",
                "[EMAIL_REDACTED]",
                anonymized,
            )
        elif pattern == "pan":
            anonymized = re.sub(
                r"[A-Z]{5}[0-9]{4}[A-Z]{1}", "[PAN_REDACTED]", anonymized
            )
        elif pattern == "aadhaar":
            anonymized = re.sub(
                r"\d{4}\s\d{4}\s\d{4}|\d{12}", "[AADHAAR_REDACTED]", anonymized
            )
        elif pattern == "phone":
            anonymized = re.sub(
                r"(\+91[\-\s]?)?[6-9]\d{9}", "[PHONE_REDACTED]", anonymized
            )

    return anonymized


def get_timestamp() -> str:
    return datetime.now().isoformat()


def hash_sensitive_data(data: str) -> str:
    return hashlib.sha256(data.encode()).hexdigest()[:16]
