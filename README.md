# aivhuman

Sentence-level AI text detection. A React frontend (`frontend/`) talks to a FastAPI
backend (`backend/`), which scores text with the model in `backend/models/p5-adv/`.

## Setup

### Requirements

- [uv](https://docs.astral.sh/uv/getting-started/installation/) (installs Python 3.13
  for you if it's missing)
- [Node.js](https://nodejs.org/) 22.12 or later (or 20.19+), with npm
- Internet access for the first run (Python packages, spaCy model, GPT-2)

No `.env` file or Hugging Face token is needed. Every setting has a working default.

### 1. Start the backend

```bash
cd backend
uv sync
uv run fastapi dev src/aivhuman_api/main.py
```

`uv sync` installs the exact versions in `uv.lock`, including PyTorch, so the first
install is a large download and can take several minutes.

On first start the backend downloads GPT-2 and the ModernBERT tokenizer from Hugging
Face, then serves on <http://localhost:8000>. API docs are at
<http://localhost:8000/docs>. A warning suggesting you set `HF_TOKEN` can be ignored,
both models are public.

### 2. Start the frontend

In a second terminal, with the backend still running:

```bash
cd frontend
npm ci
npm run dev
```

Open <http://localhost:5173>. The dev server forwards `/api` to the backend on port 8000,
so the backend must be running on that port.

### Tests

```bash
cd backend
uv run pytest
```
