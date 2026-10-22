# aivhuman-api

FastAPI backend for the aivhuman web app. Scores text with the `p5-adv` model in
`models/`, using the `aivhuman` package from `../ml` (installed editable).

```bash
uv sync
uv run fastapi dev src/aivhuman_api/main.py   # http://localhost:8000/docs
uv run pytest
```

First start downloads GPT-2 and the ModernBERT tokenizer into the Hugging Face cache, and
creates the SQLite database at `data/app.db`. Settings are listed in `.env.example`.

## API

| Method | Path | Auth | |
|---|---|---|---|
| POST | `/api/auth/register` | | create account, sets session cookie |
| POST | `/api/auth/login` | | sets session cookie |
| POST | `/api/auth/logout` | cookie | |
| GET | `/api/auth/me` | cookie | |
| POST | `/api/score` | cookie | score text, saved to history |
| GET | `/api/checks` | cookie | own history, newest 50 |
| GET | `/api/checks/{id}` | cookie | one of your own checks, 404 otherwise |

## Security

- **Access control:** every history query filters on the logged-in user's id, so another
  user's check id returns the same 404 as a missing one (`tests/test_access.py`).
- **Denial of service:** scoring runs on a worker thread behind `ScoreGate` (`gate.py`):
  1 job at a time, 4 waiting, anything beyond or waiting over 30 s gets a 503
  (`tests/test_gate.py`).
- Passwords are hashed with argon2. Unknown usernames are checked against a dummy hash so
  login timing doesn't reveal which accounts exist. Session cookies are HttpOnly and
  SameSite=Lax, and only their sha256 is stored.
- Request bodies over 2 MB get a 413 before they're read (`limits.py`).
