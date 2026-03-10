# Medical Chatbot

A Retrieval-Augmented Generation (RAG) medical assistant with:
- a **FastAPI backend** for chat + prescription PDF generation,
- a modern **HTML/CSS/JS frontend**,
- an alternative **Streamlit frontend**,
- and **multi-provider LLM fallback** support.

The assistant answers using your local knowledge-base PDFs, adds safety guidance, and can generate a structured prescription-style PDF from a conversation.

---

## Features

- **RAG over local PDFs** using Chroma vector store.
- **Provider fallback chain** for reliability:
  1. Gemini
  2. OpenRouter
  3. GitHub Models
- **Local embeddings** via `sentence-transformers/all-MiniLM-L6-v2` by default.
- **Emergency red-flag detection** appended to responses.
- **Prescription PDF generation** from chat context.
- **Source references** returned with answers.

---

## Project Structure

```text
.
├── backend/                    # FastAPI API + RAG + PDF generation
├── html_css_js_frontend/       # Vanilla JS web UI (recommended)
├── streamlit_frontend/         # Streamlit UI
├── chroma_db/                  # Persisted vector store (runtime-generated)
├── OPENROUTER_SETUP.md         # Notes for OpenRouter free-model privacy settings
├── test_providers.py           # Provider/env validation script
└── README.md
```

> Note: The backend expects a `knowledge/` folder (relative to where the backend process runs) containing your source PDFs.

---

## Prerequisites

- Python 3.10+
- `pip`
- At least one provider credential:
  - `GEMINI_API_KEY` (or `GOOGLE_API_KEY`), or
  - `OPENROUTER_API_KEY`, or
  - `GITHUB_TOKEN`

---

## Backend Setup (FastAPI)

```bash
cd backend
python -m venv venv
source venv/bin/activate   # macOS/Linux
# .\venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

Create a `.env` file in `backend/` (or otherwise ensure env vars are available):

```env
# Optional embedding model
LOCAL_EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Gemini
GEMINI_API_KEY=your_key
GEMINI_MODEL=gemini-2.5-flash

# OpenRouter
OPENROUTER_API_KEY=your_key
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_MODEL=qwen/qwen-2-7b-instruct:free

# GitHub Models
GITHUB_TOKEN=your_token
GITHUB_OPENAI_BASE_URL=https://models.inference.ai.azure.com
GITHUB_MODEL=gpt-4o-mini
```

Create and populate knowledge PDFs (from `backend/`):

```bash
mkdir -p knowledge
# place your .pdf files in backend/knowledge/
```

Run backend:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
```

---

## Frontend Options

### 1) HTML/CSS/JS Frontend (Recommended)

```bash
cd html_css_js_frontend
python -m http.server 5500
```

Open: `http://localhost:5500`

The frontend calls backend at `http://localhost:8000` by default.

### 2) Streamlit Frontend

```bash
cd streamlit_frontend
python -m venv venv
source venv/bin/activate   # macOS/Linux
# .\venv\Scripts\activate  # Windows
pip install -r requirements.txt
streamlit run app.py
```

Open: `http://localhost:8501`

---

## API Endpoints

- `POST /chat` — asks a medical question and returns answer + references.
- `POST /generate_prescription_pdf_from_chat` — generates downloadable PDF from chat context.
- `GET /health` — service health status.

---

## Provider Validation

From repo root, you can validate configured providers:

```bash
python test_providers.py
```

For OpenRouter free models, review:
- `OPENROUTER_SETUP.md`

---

## Notes

- If no provider is configured successfully, backend startup will fail.
- Chroma data and processed tracker are persisted; add new PDFs to `backend/knowledge/` and restart to ingest.
- This project provides educational guidance, not a replacement for professional medical care.
