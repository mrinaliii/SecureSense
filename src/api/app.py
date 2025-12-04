# src/api/app.py
import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, File, HTTPException, Security, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from src.api.audit_logger import AuditLogger
from src.api.auth import AuthHandler
from src.main import SensitiveDataClassifier
from src.utils.data_loader import DataLoader
from src.utils.helpers import anonymize_text, format_results
from src.utils.validator import InputValidator

app = FastAPI(
    title="Sensitive Data Classifier API",
    description="AI-powered sensitive data detection with anomaly detection",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

security = HTTPBearer()
auth_handler = AuthHandler()
classifier = SensitiveDataClassifier()
audit_logger = AuditLogger()
data_loader = DataLoader()
validator = InputValidator()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ClassificationRequest(BaseModel):
    text: str = Field(
        ..., min_length=1, max_length=10000, description="Text to analyze"
    )
    user_id: str = Field(
        default="anonymous", description="User identifier for audit logging"
    )
    encrypt_output: bool = Field(
        default=False, description="Encrypt sensitive findings"
    )
    session_id: Optional[str] = Field(None, description="Session identifier")


class BatchRequest(BaseModel):
    texts: List[str] = Field(
        ..., min_items=1, max_items=100, description="List of texts to analyze"
    )
    user_id: str = Field(default="batch_user", description="User identifier")
    encrypt_output: bool = Field(
        default=False, description="Encrypt sensitive findings"
    )


class TokenRequest(BaseModel):
    username: str
    password: str


@app.on_event("startup")
async def startup_event():
    logging.info("Sensitive Data Classifier API starting up...")
    # Warm up models
    try:
        classifier.process_text("test@example.com")
        logging.info("Models loaded successfully")
    except Exception as e:
        logging.error(f"Model warmup failed: {e}")


@app.post("/token")
async def get_token(request: TokenRequest):
    user = auth_handler.authenticate_user(request.username, request.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = auth_handler.create_access_token(user)

    audit_logger.log_access(
        user_id=user,
        action="token_generation",
        request_data={"username": request.username},
        response_data={"token_issued": True},
        risk_score=0.0,
    )

    return {"access_token": token, "token_type": "bearer"}


@app.post("/classify", response_model=Dict[str, Any])
async def classify_text(
    request: ClassificationRequest,
    credentials: HTTPAuthorizationCredentials = Security(security),
):
    if not auth_handler.verify_token(credentials.credentials):
        raise HTTPException(status_code=401, detail="Invalid token")

    user_id = auth_handler.get_user_from_token(credentials.credentials)

    is_valid, message = validator.validate_text(request.text)
    if not is_valid:
        audit_logger.log_access(
            user_id=user_id,
            action="classify",
            request_data={"text_preview": request.text[:100]},
            response_data={"error": message},
            risk_score=0.8,
        )
        raise HTTPException(status_code=400, detail=message)

    try:
        result = classifier.process_text(
            text=request.text, user_id=user_id, encrypt_output=request.encrypt_output
        )

        audit_logger.log_access(
            user_id=user_id,
            action="classify",
            request_data={
                "text_length": len(request.text),
                "encrypt_output": request.encrypt_output,
            },
            response_data={
                "request_id": result.get("request_id"),
                "risk_level": result.get("risk_assessment", {}).get("risk_level"),
            },
            risk_score=result.get("risk_assessment", {}).get("overall_risk_score", 0),
        )

        return result

    except Exception as e:
        logging.error(f"Classification error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/classify/batch", response_model=Dict[str, Any])
async def classify_batch(
    request: BatchRequest,
    credentials: HTTPAuthorizationCredentials = Security(security),
):
    if not auth_handler.verify_token(credentials.credentials):
        raise HTTPException(status_code=401, detail="Invalid token")

    user_id = auth_handler.get_user_from_token(credentials.credentials)

    if len(request.texts) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 texts per batch")

    invalid_texts = []
    for i, text in enumerate(request.texts):
        is_valid, _ = validator.validate_text(text)
        if not is_valid:
            invalid_texts.append(i)

    if invalid_texts:
        raise HTTPException(
            status_code=400, detail=f"Invalid texts at indices: {invalid_texts}"
        )

    try:
        result = classifier.batch_process(texts=request.texts, user_id=user_id)

        audit_logger.log_access(
            user_id=user_id,
            action="batch_classify",
            request_data={
                "batch_size": len(request.texts),
                "encrypt_output": request.encrypt_output,
            },
            response_data={
                "batch_id": result.get("batch_id"),
                "successful": result.get("successful"),
            },
            risk_score=result.get("summary", {}).get("average_risk_score", 0),
        )

        return result

    except Exception as e:
        logging.error(f"Batch classification error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/classify/file")
async def classify_file(
    file: UploadFile = File(...),
    encrypt_output: bool = False,
    credentials: HTTPAuthorizationCredentials = Security(security),
):
    if not auth_handler.verify_token(credentials.credentials):
        raise HTTPException(status_code=401, detail="Invalid token")

    user_id = auth_handler.get_user_from_token(credentials.credentials)

    if file.content_type not in ["text/plain", "application/json", "text/csv"]:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    try:
        contents = await file.read()

        if file.content_type == "application/json":
            texts = json.loads(contents)
            if not isinstance(texts, list):
                texts = [texts]
        elif file.content_type == "text/csv":
            import io

            import pandas as pd

            df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
            texts = df.iloc[:, 0].tolist()
        else:
            texts = [contents.decode("utf-8")]

        result = classifier.batch_process(texts, user_id)

        return result

    except Exception as e:
        logging.error(f"File processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    metrics = classifier.get_system_metrics()

    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "metrics": metrics,
        "uptime": "running",
        "version": "2.0.0",
    }


@app.get("/metrics")
async def get_metrics(credentials: HTTPAuthorizationCredentials = Security(security)):
    if not auth_handler.verify_token(credentials.credentials):
        raise HTTPException(status_code=401, detail="Invalid token")

    user_id = auth_handler.get_user_from_token(credentials.credentials)

    if user_id != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")

    system_metrics = classifier.get_system_metrics()
    audit_metrics = {
        "total_requests": audit_logger.get_total_requests(),
        "high_risk_requests": audit_logger.get_high_risk_count(),
    }

    return {
        "system_metrics": system_metrics,
        "audit_metrics": audit_metrics,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/formats/{request_id}")
async def get_formatted_results(
    request_id: str,
    format_type: str = "json",
    credentials: HTTPAuthorizationCredentials = Security(security),
):
    if not auth_handler.verify_token(credentials.credentials):
        raise HTTPException(status_code=401, detail="Invalid token")

    # In production, you'd fetch from database
    # For now, return mock or implement caching

    sample_results = [
        {
            "type": "email",
            "value": "test@example.com",
            "confidence": 0.95,
            "position": 15,
        }
    ]

    formatted = format_results(sample_results, format_type)

    return JSONResponse(
        content={"request_id": request_id, "formatted_results": formatted},
        media_type="application/json" if format_type == "json" else "text/plain",
    )


@app.get("/")
async def root():
    return {
        "message": "Sensitive Data Classifier API",
        "version": "2.0.0",
        "docs": "/docs",
        "endpoints": [
            "/classify - POST - Classify single text",
            "/classify/batch - POST - Classify multiple texts",
            "/classify/file - POST - Classify file upload",
            "/health - GET - Health check",
            "/metrics - GET - System metrics (admin)",
            "/token - POST - Get access token",
        ],
    }
