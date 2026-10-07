# ******************** IMPORTANT ******************************
# The stdout reconfig. may look like overkill, but it makes life much easier
# for debugging on Windows w/ stdout
# Especially when corpora contains em-dashes, which we all know AI loves to do XD.

import os
import sys
from pathlib import Path
from typing import Final

from dotenv import load_dotenv

ML_ROOT: Final = Path(__file__).resolve().parents[2]  # two levels up from ml/src/aivhuman/config.py

load_dotenv(ML_ROOT / ".env")

DATA_ROOT: Final = Path(os.environ.get("AIVHUMAN_DATA_ROOT") or ML_ROOT / "data").resolve()
RAW_DIR: Final = DATA_ROOT / "raw"
INTERIM_DIR: Final = DATA_ROOT / "interim"
PROCESSED_DIR: Final = DATA_ROOT / "processed" / "phase1"
HF_DIR: Final = DATA_ROOT / "hf"
PREDICTIONS_DIR: Final = DATA_ROOT / "predictions"
CHECKPOINTS_DIR: Final = DATA_ROOT / "checkpoints"
FEATURES_DIR: Final = DATA_ROOT / "features"

REPORTS_DIR: Final = ML_ROOT / "reports" / "phase1"
SPLITS_REPORT: Final = ML_ROOT / "reports" / "phase2" / "phase2_splits.json"
MANIFESTS_DIR: Final = ML_ROOT / "manifests"
BASELINES_REPORT_DIR: Final = ML_ROOT / "reports" / "phase3"
MIL_REPORT_DIR: Final = ML_ROOT / "reports" / "phase4"
ROBUSTNESS_REPORT_DIR: Final = ML_ROOT / "reports" / "phase5"
CALIBRATION_REPORT_DIR: Final = ML_ROOT / "reports" / "phase6"
EVALUATION_REPORT_DIR: Final = ML_ROOT / "reports" / "phase7"


def configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def ensure_dirs() -> None:
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
    # leaves 2 cores free for the OS + writer
    # NOTE: Reconfigure this as you want, this is what works best for me on my 8-core but may
    # need adjusting on your machine.
    override = os.environ.get("AIVHUMAN_WORKERS")
    if override:
        return max(1, int(override))
    return max(1, (os.cpu_count() or 4) - 2)
