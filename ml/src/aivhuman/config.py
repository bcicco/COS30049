"""Paths, environment resolution, and console encoding."""

# *** IMPORTANT ***
# The stdout reconfig. may look like overkill, but it makes life much easier
# for debugging on Windows
# Especially when corpora contains em-dashes, which we all know AI loves to do.

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Final

from dotenv import load_dotenv

__all__ = [
    "DATA_ROOT",
    "INTERIM_DIR",
    "ML_ROOT",
    "PROCESSED_DIR",
    "RAW_DIR",
    "REPORTS_DIR",
    "configure_stdio",
    "ensure_dirs",
    "workers",
]

ML_ROOT: Final = Path(__file__).resolve().parents[2]

load_dotenv(ML_ROOT / ".env")

DATA_ROOT: Final = Path(os.environ.get("AIVHUMAN_DATA_ROOT") or ML_ROOT / "data").resolve()
RAW_DIR: Final = DATA_ROOT / "raw"
INTERIM_DIR: Final = DATA_ROOT / "interim"
PROCESSED_DIR: Final = DATA_ROOT / "processed" / "phase1"
HF_DIR: Final = DATA_ROOT / "hf"

# Committed deliverables. Deliberately outside DATA_ROOT, because the root
REPORTS_DIR: Final = ML_ROOT / "reports" / "phase1"
MANIFESTS_DIR: Final = ML_ROOT / "manifests"


def configure_stdio() -> None:
    """Force UTF-8 on stdout/stderr, replacing anything unencodable."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def ensure_dirs() -> None:
    """Create the data and report directories if they are missing."""
    for path in (
        RAW_DIR,
        INTERIM_DIR,
        PROCESSED_DIR,
        HF_DIR,
        REPORTS_DIR,
        MANIFESTS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def workers() -> int:
    """Segmentation worker count. Leaves two cores for the OS and the writer."""
    # NOTE: Reconfigure this as you want, this is what works best for me on my 8-core.
    override = os.environ.get("AIVHUMAN_WORKERS")
    if override:
        return max(1, int(override))
    return max(1, (os.cpu_count() or 4) - 2)
