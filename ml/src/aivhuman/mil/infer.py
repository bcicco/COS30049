"""Scoring one document end to end: segmentation, features, model, calibration, explanation."""

import threading
from pathlib import Path
from typing import Final

import numpy as np
import orjson
from pydantic import BaseModel, ConfigDict

from aivhuman.features import FEATURE_NAMES
from aivhuman.mil.calibrate import EDGES

SENTENCE_POINT: Final = "FPR 1%"
DOC_SPLIT: Final = "mage-x"
"""Unseen-corpus human documents set the document threshold."""
MAX_CHARS: Final = 100_000
TOP_CONTRIBUTIONS: Final = 3
SHORT_TOKENS: Final = EDGES[0]
"""Sentences below this length are capped by the calibrator and never flagged."""

NOTICE: Final = (
    "Machine-like patterns flagged for review. This is evidence for a person to weigh, not a "
    "finding that the text was AI-written."
)


class Thresholds(BaseModel):
    """Operating thresholds and the rates measured at them."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sentence: float
    sentence_recall: float
    false_highlight_rate: float
    """Share of wholly human test documents with at least one flagged sentence."""
    document: float
    document_tpr: float
    document_fpr: float = 0.01

    @classmethod
    def from_report(cls, path: Path, run: str) -> "Thresholds":
        payload = orjson.loads(path.read_bytes())
        point = next(
            p for p in payload["sentences"]["points"] if p["name"] == SENTENCE_POINT
        )
        doc = next(
            m
            for m in payload["documents"]
            if m["model"] == run and m["split"] == DOC_SPLIT
        )
        return cls(
            sentence=point["threshold"],
            sentence_recall=point["cells"][-1]["recall"],
            false_highlight_rate=point["false_highlight_rate"],
            document=doc["threshold_1pct"],
            document_tpr=doc["tpr_at_1pct_fpr"],
        )


class Contribution(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    feature: str
    value: float
    """Signed contribution to the sentence logit; positive pushes towards machine."""


class SentenceScore(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    start: int
    end: int
    """Character offsets into the returned (NFC) text, end-exclusive."""
    n_tokens: int
    score: float
    """Calibrated evidence for machine authorship at an even prior; use as display intensity."""
    flagged: bool
    too_short: bool
    """Too short to score confidently; render neutrally."""
    contributions: list[Contribution]


class Region(BaseModel):
    """A run of consecutive flagged sentences."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    start: int
    end: int
    sentences: list[int]
    max_score: float


class DocumentResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    flagged: bool
    """Worth a closer look: above the threshold that flags 1% of unseen-corpus human documents."""
    basis: str


class Coverage(BaseModel):
    """Mean sentence probability. Internal: not calibrated, not for display."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    value: float
    calibrated: bool = False


class ScoreResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str
    sentences: list[SentenceScore]
    regions: list[Region]
    document: DocumentResult
    coverage_internal: Coverage
    style: str
    notice: str
    caveats: list[str]


def regions(
    flagged: list[bool], spans: list[tuple[int, int]], probs: np.ndarray
) -> list[Region]:
    out: list[Region] = []
    run: list[int] = []
    for i, f in enumerate([*flagged, False]):
        if f:
            run.append(i)
        elif run:
            out.append(
                Region(
                    start=spans[run[0]][0],
                    end=spans[run[-1]][1],
                    sentences=run,
                    max_score=float(probs[run].max()),
                )
            )
            run = []
    return out


def build_response(
    text: str,
    spans: list[tuple[int, int]],
    n_tokens: list[int],
    probs: np.ndarray,
    doc_prob: float,
    coverage: float,
    contributions: list[list[tuple[str, float]]],
    style: str,
    thresholds: Thresholds,
) -> ScoreResponse:
    """Assemble the response from per-sentence scores; no model is involved."""
    too_short = [n < SHORT_TOKENS for n in n_tokens]
    flagged = [
        bool(p >= thresholds.sentence) and not s
        for p, s in zip(probs, too_short, strict=True)
    ]
    sentences = [
        SentenceScore(
            start=a,
            end=b,
            n_tokens=n,
            score=float(p),
            flagged=f,
            too_short=s,
            contributions=[Contribution(feature=k, value=v) for k, v in c],
        )
        for (a, b), n, p, f, s, c in zip(
            spans, n_tokens, probs, flagged, too_short, contributions, strict=True
        )
    ]
    caveats = [
        f"At this sentence threshold, {thresholds.false_highlight_rate:.0%} of wholly human test "
        f"documents still had at least one flagged sentence, and "
        f"{thresholds.sentence_recall:.0%} of machine sentences were flagged.",
        f"The document flag catches about {thresholds.document_tpr:.0%} of machine documents "
        f"from an unseen corpus while flagging {thresholds.document_fpr:.0%} of human ones.",
        "Paraphrased machine text is not detected.",
        "Writing by non-native English speakers was not evaluated and may be flagged more often.",
    ]
    if style != "natural":
        caveats.append(
            f"Detected text style '{style}': calibration was fitted mainly on natural-cased "
            "text, so scores here are less reliable."
        )
    if sum(not s for s in too_short) < 3:
        caveats.append("Very little scorable text; the result carries little evidence.")
    doc_flag = doc_prob >= thresholds.document
    return ScoreResponse(
        text=text,
        sentences=sentences,
        regions=regions(flagged, spans, probs),
        document=DocumentResult(
            flagged=doc_flag,
            basis=f"Threshold flags {thresholds.document_fpr:.0%} of human documents from a "
            "corpus the model was not trained on.",
        ),
        coverage_internal=Coverage(value=coverage),
        style=style,
        notice=NOTICE,
        caveats=caveats,
    )


class Scorer:
    """Loads the checkpoint, calibrator, thresholds and feature pipeline once.

    The reference LM runs on CPU in float32 by default, so a text scores the same on any
    machine. Extraction ran in float16 on GPU, whose batching alone moves LM features by up to
    0.05 nats; served scores differ from stored ones by that noise, not by a pipeline change.
    """

    def __init__(
        self, checkpoint: Path, report: Path, run: str, device: str = "cpu"
    ) -> None:
        import torch

        from aivhuman.features import extract, lm
        from aivhuman.mil import train
        from aivhuman.mil.calibrate import Calibrator
        from aivhuman.text.segment import Segmenter

        self.model, self.std = train.load(checkpoint)
        self.calibrator = Calibrator.load(checkpoint.parent / "calibrator.json")
        self.thresholds = Thresholds.from_report(report, run)
        self.ref = lm.ReferenceLM(torch.device(device))
        self.segmenter = Segmenter()
        extract.init_worker()
        self.columns = [FEATURE_NAMES.index(n) for n in self.std.names]
        self._lock = threading.Lock()

    def score(self, raw: str) -> ScoreResponse:
        from aivhuman.features import extract, lm
        from aivhuman.mil.data import Bags
        from aivhuman.mil.predict import explain
        from aivhuman.mil.train import score
        from aivhuman.text.normalize import nfc
        from aivhuman.text.style import detect_style
        from aivhuman.text.tokens import count_tokens

        if len(raw) > MAX_CHARS:
            raise ValueError(f"text longer than {MAX_CHARS:,} characters")
        text = nfc(raw)
        with self._lock:
            spans = self.segmenter.segment(text)
            if not spans:
                raise ValueError("no text to score")
            _, n_tokens = count_tokens(text, spans)
            ids, offsets = self.ref.encode([text])
            token_scores = self.ref.token_scores(ids)[0]
            x = np.hstack(
                [
                    lm.span_features(offsets[0], token_scores, spans),
                    extract.text_features((text, spans)),
                    np.asarray(n_tokens, dtype=np.float64)[:, None],
                ]
            )[:, self.columns]
            n = len(spans)
            bags = Bags(
                doc_ids=["request"],
                labels=np.zeros(1, dtype=np.float32),
                x=self.std.transform(x),
                offsets=np.array([0, n]),
                span_labels=np.full(n, -1.0),
                straddles=np.zeros(n, dtype=bool),
            )
            s = score(self.model, bags)
            contributions = explain(self.model, self.std, x, TOP_CONTRIBUTIONS)
        probs = self.calibrator.apply(
            s.sentence_logits.astype(np.float64), np.asarray(n_tokens)
        )
        doc_prob = float(1.0 / (1.0 + np.exp(-s.doc_logits[0])))
        return build_response(
            text,
            spans,
            n_tokens,
            probs,
            doc_prob,
            float(s.coverage[0]),
            contributions,
            detect_style(text).style,
            self.thresholds,
        )
