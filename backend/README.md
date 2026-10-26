# aivhuman-api

FastAPI backend for the aivhuman web app. Scores text with the `p5-adv` model in
`models/`, using the `aivhuman` package from `../ml` (installed editable).

```bash
uv sync
uv run fastapi dev src/aivhuman_api/main.py   # http://localhost:8000/docs
uv run pytest
```

First start downloads GPT-2 and the ModernBERT tokenizer into the Hugging Face cache.
Settings are listed in `.env.example`.

## API

| Method | Path | |
|---|---|---|
| GET | `/api/health` | liveness |
| POST | `/api/score` | score text |

## Security

- **Denial of service:** scoring runs on a worker thread behind `ScoreGate` (`gate.py`):
  1 job at a time, 4 waiting, anything beyond or waiting over 30 s gets a 503
  (`tests/test_gate.py`).
- Request bodies over 2 MB get a 413 before they're read (`limits.py`).
