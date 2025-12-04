import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple


class InputValidator:
    @staticmethod
    def validate_text(text: str, max_length: int = 10000) -> Tuple[bool, str]:
        if not isinstance(text, str):
            return False, "Input must be a string"

        if len(text.strip()) == 0:
            return False, "Input cannot be empty"

        if len(text) > max_length:
            return False, f"Input exceeds maximum length of {max_length} characters"

        dangerous_patterns = [
            (r"<script.*?>.*?</script>", "Potential XSS attack"),
            (r"(union.*select|select.*union)", "Potential SQL injection"),
            (r"(\|\||&&)", "Command injection pattern"),
            (r"\.\./", "Path traversal attempt"),
            (r"eval\(", "Dangerous eval function"),
        ]

        for pattern, reason in dangerous_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return False, f"Security validation failed: {reason}"

        return True, "Valid"

    @staticmethod
    def validate_email(email: str) -> bool:
        pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        return bool(re.match(pattern, email))

    @staticmethod
    def validate_pan(pan: str) -> bool:
        if len(pan) != 10:
            return False

        if not re.match(r"[A-Z]{5}[0-9]{4}[A-Z]{1}", pan):
            return False

        fifth_char = pan[4]
        if not fifth_char.isalpha():
            return False

        return True

    @staticmethod
    def validate_aadhaar(aadhaar: str) -> bool:
        clean_aadhaar = re.sub(r"\s", "", aadhaar)

        if len(clean_aadhaar) != 12:
            return False

        if not clean_aadhaar.isdigit():
            return False

        if clean_aadhaar.startswith("0") or clean_aadhaar.startswith("1"):
            return False

        return True

    @staticmethod
    def validate_phone(phone: str) -> bool:
        clean_phone = re.sub(r"[\s\-+]", "", phone)

        if len(clean_phone) != 10:
            return False

        if not clean_phone.isdigit():
            return False

        if not clean_phone[0] in ["6", "7", "8", "9"]:
            return False

        return True

    @staticmethod
    def sanitize_input(text: str) -> str:
        import html

        sanitized = html.escape(text)

        sanitized = re.sub(
            r"<\s*script[^>]*>.*?<\s*/\s*script\s*>",
            "",
            sanitized,
            flags=re.IGNORECASE | re.DOTALL,
        )

        dangerous_chars = [";", "|", "&", "`", "$"]
        for char in dangerous_chars:
            sanitized = sanitized.replace(char, "")

        return sanitized.strip()

    @staticmethod
    def validate_api_request(request_data: Dict) -> Dict[str, bool]:
        validations = {}

        if "text" in request_data:
            is_valid, message = InputValidator.validate_text(request_data["text"])
            validations["text"] = is_valid
            validations["text_message"] = message

        if "user_id" in request_data:
            user_id = str(request_data["user_id"])
            validations["user_id"] = bool(re.match(r"^[a-zA-Z0-9_\-@.]+$", user_id))

        validations["timestamp"] = "timestamp" in request_data

        return validations
