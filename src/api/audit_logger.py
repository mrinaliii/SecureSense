import hashlib
import json
import logging
import os
import uuid
from datetime import datetime
from typing import Dict

from elasticsearch import Elasticsearch


class AuditLogger:
    def __init__(self):
        self.es_host = os.getenv("ELASTICSEARCH_HOST", "localhost:9200")
        self.index_name = "sensitive-data-audit"

        try:
            self.es = Elasticsearch([self.es_host])
            if not self.es.ping():
                self.es = None
        except:
            self.es = None

        self.logger = logging.getLogger("audit")
        self.logger.setLevel(logging.INFO)

        file_handler = logging.FileHandler("logs/audit.log")
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def generate_id(self) -> str:
        return str(uuid.uuid4())

    def _hash_sensitive_data(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def log_access(
        self,
        user_id: str,
        action: str,
        request_data: Dict,
        response_data: Dict,
        risk_score: float,
    ):
        timestamp = datetime.now().isoformat()
        log_id = self.generate_id()

        if "text" in request_data:
            request_data["text_hash"] = self._hash_sensitive_data(request_data["text"])
            request_data["text"] = "[REDACTED]"

        log_entry = {
            "log_id": log_id,
            "timestamp": timestamp,
            "user_id": user_id,
            "action": action,
            "risk_score": risk_score,
            "risk_level": "HIGH"
            if risk_score > 0.7
            else "MEDIUM"
            if risk_score > 0.3
            else "LOW",
            "request": request_data,
            "response_summary": {
                "findings_count": len(response_data.get("regex_findings", [])),
                "is_sensitive": response_data.get("ml_prediction", {}).get(
                    "is_sensitive", False
                ),
            },
        }

        self.logger.info(json.dumps(log_entry))

        if self.es:
            try:
                self.es.index(index=self.index_name, id=log_id, body=log_entry)
            except:
                pass

    def get_total_requests(self) -> int:
        if self.es:
            try:
                result = self.es.count(index=self.index_name)
                return result["count"]
            except:
                pass
        return 0

    def get_high_risk_count(self) -> int:
        if self.es:
            try:
                query = {"query": {"range": {"risk_score": {"gte": 0.7}}}}
                result = self.es.count(index=self.index_name, body=query)
                return result["count"]
            except:
                pass
        return 0
