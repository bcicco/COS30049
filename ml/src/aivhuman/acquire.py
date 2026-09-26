"""Fetch the raw corpus files using util func from base.py"""

from pathlib import Path

from aivhuman.config import RAW_DIR
from aivhuman.sources.base import fetch_github_raw, fetch_hf_file

MAGE_REPO = "yaful/MAGE"
# ----- Notes  ---------
# **********  IMPORTANT *********
# Read the CSVs directly DO NOT USE MAGE LOADING SCRIPT (DeepfakeTextDetect.py)
# The script is not compatible with datasets 4.x and the repo is not maintained any more.
# I wasted so much time trying to get it to work

MAGE_FILES: dict[str, str] = {
    "train": "train.csv",
    "valid": "valid.csv",
    "test": "test.csv",
    "ood_gpt": "test_ood_set_gpt.csv",
    "ood_gpt_para": "test_ood_set_gpt_para.csv",
}
# The CSVs start with a BOM, which plain utf-8 would leave glued to the first header name.
MAGE_ENCODING = "utf-8-sig"

RAID_REPO = "liamdugan/raid"
RAID_FILE = "train.csv"

SEQXGPT_REPO = "Jihuai-wpy/SeqXGPT"
SEQXGPT_COMMIT = "main"

_BENCH_DIR = "SeqXGPT/dataset/SeqXGPT-Bench"
# Note the spaces in the directory name ----> they are real and must be URL-quoted.
_OOD_DIR = "SeqXGPT/dataset/OOD sentence-level detection dataset"
_GENERATOR_STEMS = ("gpt2", "gpt3", "gptj", "gptneo", "llama", "human")

SEQXGPT_BENCH_FILES: tuple[str, ...] = tuple(
    f"{_BENCH_DIR}/en_{stem}_lines.jsonl" for stem in _GENERATOR_STEMS
)
SEQXGPT_OOD_FILES: tuple[str, ...] = tuple(
    f"{_OOD_DIR}/{stem}_lines.jsonl" for stem in _GENERATOR_STEMS
)


def fetch_mage(revision: str | None = None) -> list[Path]:
    """Download MAGE's five CSVs. Read them with :data:`MAGE_ENCODING`."""
    dest = RAW_DIR / "mage"
    return [
        fetch_hf_file(repo=MAGE_REPO, path=path, dest_dir=dest, revision=revision)
        for path in MAGE_FILES.values()
    ]


def fetch_seqxgpt(
    commit: str = SEQXGPT_COMMIT, *, include_ood: bool = True
) -> list[Path]:
    """Download SeqXGPT-Bench, and by default the OOD sentence-level set too."""
    dest = RAW_DIR / "seqxgpt"
    paths = list(SEQXGPT_BENCH_FILES)
    if include_ood:
        paths += list(SEQXGPT_OOD_FILES)

    return [
        fetch_github_raw(
            repo=SEQXGPT_REPO,
            commit=commit,
            path=path,
            dest_dir=dest / ("bench" if path.startswith(_BENCH_DIR) else "ood"),
        )
        for path in paths
    ]


def fetch_raid(revision: str | None = None) -> list[Path]:
    """Download RAID's labeled train split. ~11.8 GB, resumable, fetched once."""
    return [
        fetch_hf_file(
            repo=RAID_REPO,
            path=RAID_FILE,
            dest_dir=RAW_DIR / "raid",
            revision=revision,
        )
    ]
