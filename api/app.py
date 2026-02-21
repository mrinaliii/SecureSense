from fastapi import FastAPI
from api.schemas import TextRequest, DetectionResponse
from api.service import process_text

app = FastAPI(
    title="SecureSense API",
    description="Hybrid Sensitive Data Detection System",
    version="1.0"
)
#

@app.get("/")
def root():
    return {"message": "SecureSense API is running."}


@app.post("/detect", response_model=DetectionResponse)
def detect(request: TextRequest):

    result = process_text(request.text)

    return result