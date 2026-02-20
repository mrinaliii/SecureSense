from regex_detector import RegexDetector

detector = RegexDetector()

text = "Contact me at 9876543210 or email test@example.com. My Aadhaar is 1234 5678 9123."

results = detector.detect(text)

for r in results:
    print(r)