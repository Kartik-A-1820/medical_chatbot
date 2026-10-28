# Medical document Q&A

An educational retrieval-augmented generation prototype with a FastAPI backend, local PDF embeddings, optional graph context, and HTML or Streamlit interfaces.

The engineering focus is document ingestion, source retrieval, conversational context, and provider fallback. There is no clinical validation, medical accuracy benchmark, or evidence that the keyword-based emergency checks reliably detect emergencies. Do not use this application for diagnosis, treatment decisions, or prescribing.

## Implementation

- PDF loading and chunking, local sentence-transformer embeddings, and a persistent Chroma index.
- Optional entity/relation extraction into a NetworkX graph with persisted JSON.
- History-aware retrieval with source references.
- An ordered fallback chain: OmniRoute, Gemini, OpenRouter, then GitHub Models, using configured providers.
- A legacy prescription-style PDF export endpoint. It is unvalidated and is not part of the recommended document-Q&A demo.

Local embeddings do not make the application fully local: retrieved passages, questions, and graph-extraction inputs can be sent to the configured model provider.

## Local setup

Use a separate Python environment. Python 3.10+ is the intended baseline; requirements are not locked, so record the resolved versions and report installation failures.

```bash
git clone https://github.com/Kartik-A-1820/medical_chatbot.git
cd medical_chatbot
python -m venv .venv
```

Activate with `source .venv/bin/activate` on Linux/macOS or `.venv\Scripts\Activate.ps1` in Windows PowerShell:

```bash
python -m pip install -r backend/requirements.txt
cd backend
```

Copy `.env.example` to `.env` in `backend/`. Configure one provider you can access. Set `OMNIROUTE_ENABLED=false` when using only a hosted provider, and leave unused credentials empty. The example disables graph extraction for a simpler first run.

Create `backend/knowledge/` and add a small public, redistributable PDF with facts you can verify. Do not use patient records or private health information. From `backend/`:

```bash
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

The working directory matters: `knowledge/`, `chroma_db/`, and processed-file trackers are relative to it. The first launch downloads embedding weights, ingests PDFs, and initializes providers. Check `http://127.0.0.1:8000/health` and `/docs` after startup. Provider initialization or a health response alone does not establish that a live model request succeeds.

## Browser interface

In another activated terminal, from the repository root:

```bash
python -m http.server 5500 --bind 127.0.0.1 --directory html_css_js_frontend
```

Open `http://localhost:5500`. The default backend address is `http://localhost:8000`; the HTML meta setting controls the API address.

For Streamlit instead:

```bash
python -m pip install -r streamlit_frontend/requirements.txt
python -m streamlit run streamlit_frontend/app.py
```

Set `MEDCHAT_BACKEND_URL` in the launching environment if the backend address differs.

For a first check, ask a factual question answered by the supplied PDF and compare the answer and references with the source. Also ask a question the PDF cannot answer. Record unsupported statements and missing references rather than treating fluent text as successful retrieval.

## Configuration and state

[backend/.env.example](backend/.env.example) lists the supported provider settings. Model IDs are configuration choices; use one available to your account rather than assuming an old default remains available.

- New PDFs are ingested on restart. Filename trackers do not detect changed contents under the same name.
- Graph extraction can add model calls during startup. Enable it only after the vector-only path works.
- Runtime indexes and trackers should be regenerated from your chosen corpus, not shared as source artifacts.
- Chat history is process-local. The Streamlit client currently uses the default backend session ID; it does not provide independent multi-user sessions.
- The API has permissive CORS and no application authentication. Keep this demo local.

## Validation and open work

`test_providers.py` is a live connectivity diagnostic, not an offline test suite. From the repository root, supply credentials through the environment or a root `.env` and run `python test_providers.py`. It can send requests to multiple configured providers and does not assess clinical or retrieval quality.

No reproducible labeled retrieval/answer benchmark is included. Priority work includes fixture-based ingestion tests, citation/abstention evaluation, content-hash tracking, session isolation, dependency locking, and removal or redesign of the prescription workflow.

## Contribute

See [CONTRIBUTING.md](CONTRIBUTING.md) and [SECURITY.md](SECURITY.md). Use synthetic or public redistributable documents in reproductions and clearly separate provider connectivity, retrieval quality, and application safety.

No standalone LICENSE file is included. Repository and source-document terms must be checked separately.
