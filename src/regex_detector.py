import re

PATTERNS = {
    "EMAIL": r"[a-zA-Z0-9+_.-]+@[a-zA-Z0-9.-]+",
    "PHONE": r"\+?\d[\d -]{8,12}\d",
    "CREDIT_CARD": r"\b(?:\d[ -]*?){13,16}\b",
}


def regex_detect(text):

    results = []

    for label, pattern in PATTERNS.items():

        for match in re.finditer(pattern, text):

            results.append({
                "entity": label,
                "value": match.group()
            })

    return results
