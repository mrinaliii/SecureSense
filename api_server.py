#!/usr/bin/env python3
import os

import uvicorn
from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    uvicorn.run(
        "src.api.app:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("DEBUG", "False").lower() == "true",
        workers=int(os.getenv("WORKERS", 4)),
        log_level=os.getenv("LOG_LEVEL", "info"),
    )
