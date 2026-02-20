from hybrid_detector import HybridDetector
from masker import Masker

detector = HybridDetector()
masker = Masker()

text = "John lives in Mumbai. Call him at 9876543210 or email john@example.com."

detections = detector.detect(text)

print("Detections:")
for d in detections:
    print(d)

masked = masker.mask_text(text, detections, mode="asterisk")

print("\nMasked Text:")
print(masked)