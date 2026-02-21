from pydantic import BaseModel
from typing import List, Dict


class TextRequest(BaseModel):
    text: str


class DetectionResponse(BaseModel):
    original_text: str
    masked_text: str
    detections: List[Dict]