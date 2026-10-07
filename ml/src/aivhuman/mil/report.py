"""evaluation.json (doc + sentence + calibration)"""

from pathlib import Path
from typing import Final

import orjson

from aivhuman.evaluate import SplitMetrics
from aivhuman.mil.calibrate import ECE_TARGET, CellMetrics
from aivhuman.mil.sentences import SentenceReport

DOC_SPLITS: Final = ("raid-ood", "mage-x", "mage-para", "daigt")


def write(
    run: str,
    doc_metrics: list[SplitMetrics],
    commentary: int,
    sentences: SentenceReport,
    straddling: int,
    cells: list[CellMetrics],
    out_dir: Path,
) -> Path:
    """write the final evaluation json, returns its path"""
    payload = {
        "run": run,
        "documents": [m.model_dump() for m in doc_metrics],
        "commentary_excluded": commentary,
        "sentences": sentences.model_dump(),
        "straddling_excluded": straddling,
        "calibration": [c.model_dump() for c in cells],
        "ece_target": ECE_TARGET,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "evaluation.json"
    path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))
    return path
