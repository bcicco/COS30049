"""Fetching raw corpus files from Hugging Face and GitHub, just a downloader utility, not a data
loader.
"""

import time
import urllib.parse
from pathlib import Path

import requests
from huggingface_hub import hf_hub_download

_CHUNK = 1 << 20  # 2^20 = 1MB


def fetch_hf_file(
    *,
    repo: str,
    path: str,
    dest_dir: Path,
    revision: str | None = None,
    attempts: int = 8,
) -> Path:
    """Download one file from a HF dataset repo into `dest_dir`."""

    # ************* Note **************
    # Retries on transport errors because they occured in testing....
    #  Each retry resumes from the partial `.incomplete` file, so a drop
    #  doesn't cost the whole transfer

    dest_dir.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, attempts + 1):
        try:
            return Path(
                hf_hub_download(
                    repo_id=repo,
                    filename=path,
                    repo_type="dataset",
                    revision=revision,
                    local_dir=str(dest_dir),
                )
            )
        except (requests.RequestException, OSError) as exc:
            if attempt == attempts:
                raise
            backoff = min(2**attempt, 60)
            print(
                f"  {path}: {type(exc).__name__} on attempt {attempt}/{attempts}, "
                f"resuming in {backoff}s",
                flush=True,
            )
            time.sleep(backoff)
    raise AssertionError("unreachable")


def fetch_github_raw(
    *,
    repo: str,
    commit: str,
    path: str,
    dest_dir: Path,
    timeout: int = 300,
) -> Path:
    """Download one file from a GitHub repo at `commit`"""
    dest_dir.mkdir(parents=True, exist_ok=True)
    quoted = urllib.parse.quote(path)
    url = f"https://raw.githubusercontent.com/{repo}/{commit}/{quoted}"
    local = dest_dir / Path(path).name

    if not local.exists():
        with requests.get(url, stream=True, timeout=timeout) as resp:
            resp.raise_for_status()
            with local.open("wb") as fh:
                for chunk in resp.iter_content(chunk_size=_CHUNK):
                    fh.write(chunk)
    return local
