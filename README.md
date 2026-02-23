🔐 SecureSense --- Hybrid Sensitive Data Detection System
=======================================================

SecureSense is a production-oriented hybrid NLP system designed to automatically detect and mask sensitive information (PII) in text using a fine-tuned Transformer model combined with rule-based intelligence.

This project demonstrates an end-to-end ML system --- from dataset preparation and GPU training to hybrid inference, masking, and REST API deployment.

* * * * *

🚀 Project Overview
-------------------

Sensitive data detection is critical for:

-   Privacy protection

-   Regulatory compliance (GDPR, HIPAA)

-   Secure data pipelines

-   Enterprise document processing

-   AI data sanitization

SecureSense combines:

-   Transformer-based NER (DistilBERT)

-   Regex-based structured PII detection

-   Heuristic recall enhancement

-   Overlap-safe hybrid merging

-   REST API deployment via FastAPI

This hybrid architecture mirrors real-world PII detection systems used in production.

* * * * *

🧠 Model Architecture
---------------------

-   **Base Model:** DistilBERT (`distilbert-base-uncased`)

-   **Task:** Token Classification (NER)

-   **Framework:** HuggingFace Transformers

-   **Training Device:** NVIDIA GTX 1650 GPU (FP16 enabled)

-   **Dataset:** WikiANN (English)

### Key Training Strategies:

-   Linear learning rate scheduler with warmup

-   Gradient accumulation for memory efficiency

-   FP16 mixed precision training

-   Automatic evaluation per epoch

-   Best-model checkpoint selection (based on F1)

-   Label mapping (id2label / label2id) for clean inference

* * * * *

📊 Performance
--------------

| Metric | Score |
| --- | --- |
| **F1 Score** | **0.819** |
| Precision | 0.809 |
| Recall | 0.829 |
| Eval Loss | 0.279 |
| Token Accuracy | 0.92 |

### ✔ Interpretation

-   Strong convergence

-   Balanced precision-recall

-   Slight recall prioritization (ideal for PII detection)

-   Stable generalization

Recall is intentionally favored since missing sensitive data is riskier than slightly over-masking.

* * * * *

🏗️ Hybrid Detection Architecture
---------------------------------

SecureSense is not purely model-based. It uses a layered hybrid system:

Input Text\
    ↓\
Transformer NER (PER / LOC / ORG)\
    ↓\
Regex Structured PII Detection\
    ↓\
Heuristic Enhancement Layer\
    ↓\
Overlap Resolution\
    ↓\
Masking Layer\
    ↓\
JSON API Response

* * * * *

🔍 Detection Capabilities
-------------------------

### 1️⃣ Transformer-Based Detection

Detects contextual entities:

-   PERSON (PER)

-   LOCATION (LOC)

-   ORGANIZATION (ORG)

### 2️⃣ Regex-Based Structured PII Detection

Deterministic detection for:

-   Phone numbers

-   Emails

-   Aadhaar numbers

-   PAN numbers

-   Credit card numbers

Regex ensures high precision for structured identifiers.

### 3️⃣ Heuristic Recall Enhancement

A lightweight contextual heuristic improves recall for:

-   Single capitalized names missed by the transformer

### 4️⃣ Overlap Resolution

Ensures:

-   No duplicate spans

-   No conflicting detections

-   Clean, non-overlapping entity outputs

* * * * *

🧪 Example Inference
--------------------

### Input

John lives in Mumbai. Call at 9876543210.

### Output

[PER] lives in [LOC]. Call at [PHONE].

* * * * *

### Input

Barack Obama visited Mumbai.

### Output

[PER] visited [LOC].

* * * * *

### Input

Contact me at john@example.com or 9876543210.

### Output

Contact me at [EMAIL] or [PHONE].

* * * * *

🏗️ Pipeline Architecture
-------------------------

### Training Pipeline

Dataset → Tokenization → Label Alignment → Transformer Fine-Tuning → Evaluation → Metrics Export

Includes:

-   Batched tokenization

-   Word-piece label alignment

-   GPU acceleration

-   Experiment logging

-   Best-model selection

-   CSV metric export

* * * * *

🌐 REST API Deployment
----------------------

SecureSense exposes a real-time inference API using FastAPI.

### Start the Server

uvicorn api.app:app --reload

### Swagger UI

http://127.0.0.1:8000/docs

### POST `/detect`

Request:

{\
  "text": "John lives in Mumbai. Call at 9876543210."\
}

Response:

{\
  "original_text": "...",\
  "masked_text": "...",\
  "detections": [...]\
}

* * * * *

📂 Project Structure
--------------------

SecureSense/\
│\
├── src/\
│   ├── train.py                # Full training pipeline\
│   ├── preprocess.py           # Tokenization + alignment\
│   ├── data_loader.py          # Dataset loading\
│   ├── metrics.py              # F1 / Precision / Recall\
│   ├── predict.py              # Transformer inference\
│   ├── hybrid_detector.py      # Hybrid detection engine\
│   ├── regex_detector.py       # Structured PII detection\
│   ├── masker.py               # Redaction logic\
│   └── analyse_results.py      # Metrics inspection\
│\
├── api/\
│   ├── app.py                  # FastAPI entry point\
│   ├── service.py              # Detection service layer\
│   └── schemas.py              # Request/response models\
│\
├── models/\
│   └── distilbert-pii/         # Fine-tuned model\
│\
├── results/                    # Training metrics CSV\
├── Dockerfile\
├── requirements.txt\
└── README.md

* * * * *

⚙️ Installation
---------------

git clone https://github.com/YOUR_USERNAME/SecureSense.git\
cd SecureSense

python -m venv venv\
venv\Scripts\activate   # Windows

pip install -r requirements.txt

* * * * *

🏋️ Training
------------

python src/train.py

Features:

-   Automatic evaluation

-   Metrics saved to `/results`

-   Best model automatically selected

-   GPU optimized training

* * * * *

🧪 Engineering Highlights
-------------------------

SecureSense emphasizes real ML engineering practices:

-   Modular architecture

-   Clean separation of training and inference

-   Hybrid detection strategy

-   Label-mapped transformer outputs

-   GPU-aware training

-   Structured metric logging

-   REST API deployment

-   Overlap-safe span merging

This is a system-oriented NLP implementation --- not a notebook experiment.

* * * * *

🔮 Future Enhancements
----------------------

-   Train on CoNLL-2003 for improved PERSON recall

-   Confidence-based filtering

-   Detection source tagging (regex / transformer / heuristic)

-   Docker deployment

-   Cloud hosting

-   Multilingual support

-   Expanded real-world PII datasets

* * * * *

👩‍💻 Author
------------

**Mrinali Charhate**\
Computer Science Engineer focused on Applied AI, NLP, and Secure Intelligent Systems.

* * * * *

⭐ Why This Project Matters
--------------------------

SecureSense demonstrates the ability to:

✔ Fine-tune transformer models effectively\
✔ Design hybrid ML + rule-based systems\
✔ Balance precision and recall for security use-cases\
✔ Build deployable ML services\
✔ Structure production-style NLP pipelines

This project reflects practical AI engineering with deployment readiness --- not just experimental modeling.
