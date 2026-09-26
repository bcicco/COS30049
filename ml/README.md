# aivhuman

Detects AI-written text **sentence by sentence**, not just "this whole document is AI".

Right now the project is at **Phase 1**: downloading three public datasets of human and
AI text (RAID, MAGE, SeqXGPT), cleaning them, and saving them in one shared format. The
model itself comes in later phases. See [`PLAN.md`](PLAN.md) for the full roadmap.

---

## 1. What you need first

 **uv**  installs Python 3.13 and every dependency for you. You don't need to install Python yourself 
 **~20 GB free disk** RAID's raw CSV  is 11.8 GB. Only needed if you download the data (step 4) 

### Install uv

**macOS / Linux:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Close and reopen your terminal afterwards, then check that it worked:

```bash
uv --version
```

---

## 2. Set up the project

```bash
git clone <repo-url>
cd aivhuman/ml              # everything below is run from inside ml/
uv sync --extra dev         # creates .venv/ and installs everything from uv.lock
```

That's it. There is no need to activate the virtual environment: `uv run <command>`
always uses the project's environment.

Then copy the example settings file:

```bash
cp .env.example .env        # Windows cmd: copy .env.example .env
```

The defaults are fine. `HF_TOKEN` can stay empty, since all three datasets are public. If
Hugging Face starts rate-limiting you, create a free token at
<https://huggingface.co/settings/tokens> and paste it in.

---

## 3. Check everything works

```bash
uv run pytest
```

You should see a wall of dots and no `F`s. These tests use small built-in samples, so
they run in seconds and don't download anything. If this passes, your setup is good.

---

## 4. Build the dataset (optional, slow)

Only do this if you actually need the processed data. Run the steps **in order**:

```bash
uv run aivhuman-data acquire      # 1. download raw corpora         (~12 GB, depends on your internet)
uv run aivhuman-data derive       # 2. convert RAID's CSV to parquet (~2 min)
uv run aivhuman-data peek         # 3. print sample rows so you can eyeball the labels
uv run aivhuman-data ingest       # 4. split into sentences, write JSONL (~20-40 min on 8+ cores)
uv run aivhuman-data verify       # 5. re-check every document on disk
uv run aivhuman-data report       # 6. write summary CSVs to reports/phase1/
```

Useful options:

- `--source raid|mage|seqxgpt` on `acquire` and `ingest` does one corpus at a time.
  SeqXGPT is the smallest, so start with it if you just want to see things working.
- `--help` on any command lists its options, e.g. `uv run aivhuman-data ingest --help`.

### Where things end up

```
ml/data/raw/                 downloaded files, untouched
ml/data/interim/raid/        RAID converted to parquet
ml/data/processed/phase1/    final output: {raid,mage,seqxgpt}.jsonl + .stats.json
ml/reports/phase1/           summary report (this one IS committed to git)
```

`data/` is gitignored, so never try to commit it.

---

## 5. Before you push code

CI runs these four checks on Windows and Linux. Run them locally first:

```bash
uv run ruff format .     # auto-format
uv run ruff check .      # lint (add --fix to auto-fix)
uv run mypy              # type check (strict)
uv run pytest            # tests
```

---

## 6. Adding a dependency

```bash
uv add <package>                 # runtime dependency
uv add --optional dev <package>  # dev-only tool
```

This updates both `pyproject.toml` and `uv.lock`, so commit both. Don't `pip install`
into the environment by hand, because the next `uv sync` will remove it.

The `train` extra (torch and friends ~2.5 GB) is only for later. Install it
with `uv sync --extra dev --extra train` when you need it.

---

## Common problems

| Symptom | Fix |
| --- | --- |
| `uv: command not found` | Reopen your terminal after installing uv |
| Weird characters / `UnicodeEncodeError` when printing text on Windows | Make sure `.env` has `PYTHONUTF8=1` (it does if you copied `.env.example`) |
| `ingest` is slow or your machine becomes unresponsive | Lower the worker count: `--workers 4`, or set `AIVHUMAN_WORKERS` in `.env` |
| Running out of disk | Point the data somewhere else: set `AIVHUMAN_DATA_ROOT=D:/aivhuman-data` in `.env` |
| `peek` or `ingest` says a file is missing | You skipped a step. `acquire` must run before `derive`, and `derive` before RAID `ingest` |

---

## Where to read next
- `src/aivhuman/schema.py`: the `Doc` / `SentenceSpan` format every output line follows
