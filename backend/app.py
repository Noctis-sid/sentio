"""
Sentio backend — serves your trained DistilBERT symptom-to-disease model
over a simple REST API that the Sentio web UI (index.html) calls.

Run with:
    pip install fastapi uvicorn torch transformers --break-system-packages
    python app.py

Then open ../index.html in your browser (or serve it with any static
file server) — it's already pointed at http://localhost:8000.
"""

import json
import os
import re

import torch
import torch.nn.functional as F
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

# ============================================================
# CONFIG — update these paths to match your project structure
# ============================================================
PROJECT_ROOT = r"D:\deep_learning_v2"
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "distilbert", "best_distilbert")
LABEL2ID_PATH = os.path.join(PROJECT_ROOT, "dataset", "processed", "label2id.json")
MAX_LEN = 50

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# Load model, tokenizer, and label mapping once at startup
# ============================================================
print(f"Loading model from {MODEL_DIR} on {DEVICE} ...")

with open(LABEL2ID_PATH) as f:
    label2id = json.load(f)
id2label = {v: k for k, v in label2id.items()}

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE)
model.eval()

print(f"Loaded. {len(id2label)} disease classes available.")

# ============================================================
# Text cleaning — must match the preprocessing used in training
# ============================================================
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9,.\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# ============================================================
# API
# ============================================================
app = FastAPI(title="Sentio API")

# Allow the local HTML file (or any dev frontend) to call this API.
# Tighten this to specific origins before deploying anywhere public.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    text: str
    top_k: int = 4


class PredictionItem(BaseModel):
    label: str
    confidence: float


class PredictResponse(BaseModel):
    predictions: list[PredictionItem]
    model: str


@app.get("/health")
def health():
    return {"status": "ok", "num_classes": len(id2label), "device": str(DEVICE)}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    cleaned = clean_text(req.text)

    inputs = tokenizer(
        cleaned,
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=-1).squeeze(0)

    top_k = min(req.top_k, len(id2label))
    top_probs, top_idx = torch.topk(probs, top_k)

    predictions = [
        PredictionItem(label=id2label[idx.item()], confidence=round(prob.item(), 4))
        for prob, idx in zip(top_probs, top_idx)
    ]

    return PredictResponse(predictions=predictions, model="DistilBERT (fine-tuned)")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
