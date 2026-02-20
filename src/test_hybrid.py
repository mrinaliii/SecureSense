from hybrid_detector import HybridDetector

detector = HybridDetector()

text = "John lives in Mumbai. Call him at 9876543210 or email john@example.com"

results = detector.detect(text)

for r in results:
    print(r)