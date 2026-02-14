# 🔐 SecureSense — Sensitive Data Detection using Transformers

SecureSense is a high-performance NLP system designed to automatically detect sensitive information in text using a fine-tuned Transformer model.

Built with a production-oriented mindset, this project demonstrates an end-to-end machine learning pipeline — from dataset preparation and token alignment to GPU training, evaluation, and inference readiness.

---

## 🚀 Project Overview

Sensitive data detection is critical for:

- Privacy protection
- Regulatory compliance (GDPR, HIPAA)
- Secure data pipelines
- Enterprise document processing

SecureSense leverages **DistilBERT** for token-level classification to identify personally identifiable information (PII) with strong precision-recall balance.

---

## 🧠 Model Architecture

- **Base Model:** DistilBERT (`distilbert-base-uncased`)
- **Task:** Named Entity Recognition (Token Classification)
- **Framework:** HuggingFace Transformers
- **Training Device:** NVIDIA GTX 1650 GPU (FP16 enabled)

### Key Training Strategies:

- Linear learning rate scheduler with warmup
- Gradient accumulation for memory efficiency
- Best-model checkpointing
- Automated evaluation per epoch

---

## 📊 Performance

| Metric       | Score     |
| ------------ | --------- |
| **F1 Score** | **0.818** |
| Precision    | 0.808     |
| Recall       | 0.828     |
| Eval Loss    | 0.278     |

### ✔ Interpretation:

- Strong generalization
- Stable convergence
- Balanced precision-recall
- Recall slightly prioritized — desirable for sensitive data detection systems

---

## 🏗️ Pipeline Architecture

```
Dataset → Tokenization → Label Alignment → Transformer Fine-Tuning → Evaluation → Metrics Export
```

### Training Pipeline Includes:

✅ Batched tokenization  
✅ Word-piece label alignment  
✅ GPU acceleration  
✅ Experiment logging  
✅ Automatic best-model selection

---

## 📂 Project Structure

```
SecureSense/
│
├── src/
│   ├── train.py              # End-to-end training pipeline
│   ├── preprocess.py        # Tokenization + label alignment
│   ├── data_loader.py       # Dataset loading
│   ├── metrics.py           # Precision / Recall / F1 computation
│   ├── predict.py           # Inference utilities
│   ├── config.py           # Central configuration
│   └── analyse_results.py   # Training visualization
│
├── api/
│   └── app.py              # FastAPI scaffold for deployment
│
├── Dockerfile              # Container-ready setup
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/YOUR_USERNAME/SecureSense.git
cd SecureSense

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

---

## 🏋️ Training

```bash
python src/train.py
```

Training features:

- Automatic evaluation
- Metrics saved to `/results`
- Best checkpoint retained
- GPU optimized

---

## 🔎 Example Use Case

Input:

```
"My name is John and I live in New York."
```

Output:

```
John → PERSON
New York → LOCATION
```

---

## 📈 Experiment Tracking

Training logs and metrics are exported as CSV files for analysis and reproducibility.

Supports visualization of:

- Loss curves
- F1 progression
- Precision vs Recall

---

## 🧪 Engineering Highlights

This project emphasizes **real-world ML engineering practices**, including:

- Modular code design
- Config-driven pipeline
- GPU-aware training
- Memory-efficient batching
- Structured evaluation
- Deployment-ready API scaffold

---

## 🔮 Future Enhancements

Planned improvements include:

- Hybrid detection (Transformer + rule-based patterns)
- Automatic PII masking/redaction
- Production API deployment
- Real-time inference
- Expanded dataset training

---

## ⭐ Why This Project Matters

SecureSense demonstrates the ability to:

✔ Train transformer models effectively  
✔ Optimize under hardware constraints  
✔ Evaluate using correct NLP metrics  
✔ Structure production-style ML codebases

This is not a notebook experiment — it is a system-oriented implementation of modern NLP practices.

---
