from typing import List, Dict


class Masker:

    @staticmethod
    def mask_text(text: str, detections: List[Dict], mode: str = "asterisk") -> str:

        masked_text = text
        offset = 0

        for item in sorted(detections, key=lambda x: x["start"]):

            start = item["start"] + offset
            end = item["end"] + offset

            if mode == "asterisk":
                replacement = "*" * (end - start)
            elif mode == "label":
                replacement = f"[{item['label']}]"
            else:
                replacement = "[REDACTED]"

            masked_text = (
                masked_text[:start] +
                replacement +
                masked_text[end:]
            )

            offset += len(replacement) - (end - start)

        return masked_text