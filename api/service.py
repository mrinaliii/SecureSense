from src.hybrid_detector import HybridDetector
from src.masker import Masker


detector = HybridDetector()
masker = Masker()


def process_text(text: str):

    detections = detector.detect(text)

    masked_text = masker.mask_text(
        text,
        detections,
        mode="label"
    )

    return {
        "original_text": text,
        "masked_text": masked_text,
        "detections": detections
    }