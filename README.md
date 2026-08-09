# Sentio — Symptom Intelligence

A simple chat interface for your symptom-to-disease DistilBERT model, styled like Claude/ChatGPT.

## Structure

```
sentio/
├── index.html              # the web app (open directly in a browser)
├── README.md
└── backend/
    ├── app.py               # FastAPI server that loads your trained DistilBERT model
    └── requirements.txt
```

## Running it

1. **Install backend dependencies** (inside your `.venv`):
   ```
   cd backend
   pip install -r requirements.txt --break-system-packages
   ```

2. **Check the paths in `backend/app.py`** match your project:
   ```python
   PROJECT_ROOT = r"D:\deep_learning_v2"
   MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "distilbert", "best_distilbert")
   LABEL2ID_PATH = os.path.join(PROJECT_ROOT, "dataset", "processed", "label2id.json")
   ```
   These should already match your notebook's folder structure, but double-check
   after training your final v4 model.

3. **Start the backend:**
   ```
   python app.py
   ```
   You should see `Loaded. 512 disease classes available.` and the server running on
   `http://localhost:8000`.

4. **Open `index.html`** directly in your browser (double-click it, or drag it into a
   browser window). The status pill in the top-right will show "model server connected"
   once it can reach the backend.

## Demo mode

If you open `index.html` without the backend running, it still works — it falls back to a
small set of hardcoded example predictions labeled "demo mode" so you can show off the UI
(e.g. in a presentation) without needing the model loaded. Once the real backend is running,
it automatically switches to live predictions.

## Notes

- The frontend cleans input text the same way your training pipeline did (lowercase,
  strip non-alphanumeric characters) before sending it to the model — this happens
  server-side in `app.py`, matching your notebook's `clean_text()` function exactly.
- CORS is wide open (`allow_origins=["*"]`) for local development. If you ever deploy
  this somewhere public, lock that down to your actual frontend's origin.
- This is explicitly framed as a research/demo tool in the UI copy — the disclaimer under
  the input box and under every prediction card is intentional, not decorative, given this
  is a student project rather than a validated clinical tool.
