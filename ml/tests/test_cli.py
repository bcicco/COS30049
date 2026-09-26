"""The command line: subcommand wiring and exit codes."""

from pathlib import Path
from typing import Any

import pytest

from aivhuman import cli, config
from test_overlap_report import doc, sidecar, write


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the CLI's directories at a `tmp_path` instead of the real data root."""
    for name in ("PROCESSED_DIR", "REPORTS_DIR", "INTERIM_DIR", "RAW_DIR", "MANIFESTS_DIR"):
        monkeypatch.setattr(config, name, tmp_path / name.lower(), raising=False)
        (tmp_path / name.lower()).mkdir(parents=True, exist_ok=True)
    return tmp_path


def good_corpus(workspace: Path) -> Path:
    path = write(
        config.PROCESSED_DIR / "raid.jsonl",
        [doc("raid", "raid:a", "Some text here."), doc("raid", "raid:b", "More text here.")],
    )
    sidecar(
        path,
        {
            "docs": 2,
            "is_green": True,
            "segment_stats": {"spans": 2, "nonws_gap_chars": 0},
        },
    )
    return path


def test_no_subcommand_prints_help_and_fails(capsys: Any) -> None:
    assert cli.main([]) == 2
    assert "acquire" in capsys.readouterr().out


def test_verify_returns_zero_on_a_clean_corpus(workspace: Path, capsys: Any) -> None:
    good_corpus(workspace)

    assert cli.main(["verify"]) == 0
    assert "ok" in capsys.readouterr().out


def test_verify_returns_nonzero_on_a_problem(workspace: Path, capsys: Any) -> None:
    """Exit codes matter: this is what a CI step or a Makefile checks."""
    path = good_corpus(workspace)
    sidecar(path, {"docs": 99, "is_green": True})

    assert cli.main(["verify"]) == 1
    assert "sidecar says 99" in capsys.readouterr().err


def test_verify_on_an_empty_directory_is_an_error_not_a_pass(workspace: Path) -> None:
    """Zero files verifying clean would be the most dangerous possible green."""
    assert cli.main(["verify"]) == 2


def test_report_writes_the_deliverable(workspace: Path, capsys: Any) -> None:
    good_corpus(workspace)

    assert cli.main(["report"]) == 0

    out = config.REPORTS_DIR / "phase1_metrics.csv"
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "corpus," in text
    assert "verify," in text, "the report must state that the data was re-checked"


def test_report_can_skip_verification(workspace: Path) -> None:
    good_corpus(workspace)

    assert cli.main(["report", "--skip-verify"]) == 0

    text = (config.REPORTS_DIR / "phase1_metrics.csv").read_text(encoding="utf-8")
    assert "verify," not in text


def test_every_subcommand_is_reachable() -> None:
    """A subcommand with no handler would print help and exit 2 at runtime."""
    parser = cli._parser()
    actions = [a for a in parser._actions if hasattr(a, "choices") and a.choices]
    names = sorted(next(iter(actions)).choices)
    assert names == [
        "acquire",
        "derive",
        "ingest",
        "peek",
        "report",
        "split",
        "verify",
    ]


def test_clip_flattens_whitespace() -> None:
    """peek prints one row per line, and corpus text is full of newlines."""
    assert cli._clip("a\n\n  b   c", 80) == "a b c"
    assert cli._clip("x" * 100, 10) == "x" * 10 + "..."


def test_reservoir_samples_without_holding_the_corpus() -> None:
    """RAID and MAGE are sorted, so a head-slice shows one domain and one generator."""
    import random

    rng = random.Random(1)
    sample = cli._reservoir(iter(range(10_000)), 20, rng)

    assert len(sample) == 20
    assert len(set(sample)) == 20
    assert max(sample) > 1_000, "a head-slice would never reach here"
