"""
Sentio backend — serves your trained DistilBERT symptom-to-disease model
over a simple REST API that the Sentio web UI (index.html) calls.

Uses ONNX Runtime instead of PyTorch for inference -- much lighter on memory,
which matters on Render's free tier (512MB RAM limit). The full PyTorch +
transformers + torch model was crashing with "Out of memory (used over 512Mi)".

Run locally with:
    pip install -r requirements.txt
    python app.py

Then open ../index.html in your browser -- it's pointed at http://localhost:8000
when running locally, and at your Render URL once deployed.
"""

import json
import os
import re

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import DistilBertTokenizerFast
from huggingface_hub import hf_hub_download

# ============================================================
# CONFIG
# ============================================================
HF_REPO_ID = "Zacksid1/sentio-symptom-distilbert"
ONNX_FILENAME = "distilbert_int8.onnx"  # the quantized model you uploaded

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LABEL2ID_PATH = os.path.join(BASE_DIR, "label2id.json")
MAX_LEN = 50

# ============================================================
# Load tokenizer, label mapping, and ONNX model once at startup
# ============================================================
print(f"Downloading ONNX model from {HF_REPO_ID}/{ONNX_FILENAME} ...")
onnx_path = hf_hub_download(repo_id=HF_REPO_ID, filename=ONNX_FILENAME)

print("Loading ONNX Runtime session ...")
session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

print(f"Loading tokenizer from {HF_REPO_ID} ...")
tokenizer = DistilBertTokenizerFast.from_pretrained(HF_REPO_ID)

with open(LABEL2ID_PATH) as f:
    label2id = json.load(f)
id2label = {v: k for k, v in label2id.items()}

print(f"Loaded. {len(id2label)} disease classes available.")

# input/output names inside the ONNX graph -- printed here so you can sanity
# check them against convert_to_onnx.py if predictions look wrong
INPUT_NAMES = [i.name for i in session.get_inputs()]
OUTPUT_NAME = session.get_outputs()[0].name
print(f"ONNX inputs: {INPUT_NAMES} | output: {OUTPUT_NAME}")

# ============================================================
# Text cleaning — must match the preprocessing used in training
# ============================================================
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9,.\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


# ============================================================
# API
# ============================================================
app = FastAPI(title="Sentio API")

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
    return {"status": "ok", "num_classes": len(id2label), "backend": "onnxruntime"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    cleaned = clean_text(req.text)

    encoded = tokenizer(
        cleaned,
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_tensors="np",
    )

    # Build the ONNX Runtime input dict from whatever inputs the graph actually
    # expects (some exports include token_type_ids, some don't).
    ort_inputs = {}
    for name in INPUT_NAMES:
        if name == "input_ids":
            ort_inputs[name] = encoded["input_ids"].astype(np.int64)
        elif name == "attention_mask":
            ort_inputs[name] = encoded["attention_mask"].astype(np.int64)
        elif name == "token_type_ids" and "token_type_ids" in encoded:
            ort_inputs[name] = encoded["token_type_ids"].astype(np.int64)

    logits = session.run([OUTPUT_NAME], ort_inputs)[0]
    probs = softmax(logits[0])

    top_k = min(req.top_k, len(id2label))
    top_idx = np.argsort(probs)[::-1][:top_k]

    predictions = [
        PredictionItem(label=id2label[int(idx)], confidence=round(float(probs[idx]), 4))
        for idx in top_idx
    ]

    return PredictResponse(predictions=predictions, model="DistilBERT (ONNX, int8 quantized)")


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))  # Render sets PORT automatically
    uvicorn.run(app, host="0.0.0.0", port=port)