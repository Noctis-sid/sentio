import os
import onnxruntime as ort

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "..",
    "..",
    "distilbert_int8.onnx"
)

session = ort.InferenceSession(MODEL_PATH)

print("Inputs:")
for x in session.get_inputs():
    print(x.name, x.shape, x.type)

print("\nOutputs:")
for x in session.get_outputs():
    print(x.name, x.shape, x.type)

import os
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "label2id.json")) as f:
    label2id = json.load(f)

print(len(label2id))