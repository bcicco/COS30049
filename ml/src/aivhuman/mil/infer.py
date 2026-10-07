"""score a single text end to end (used by `aivhuman-mil score`)"""

from pathlib import Path
from typing import Final

import numpy as np
import orjson
from pydantic import BaseModel, ConfigDict

from aivhuman.features import FEATURE_NAMES
from aivhuman.mil.calibrate import EDGES

SENTENCE_POINT: Final = "FPR 1%"
DOC_SPLIT: Final = "mage-x"  # doc threshold comes from the unseen corpus
MAX_CHARS: Final = 100_000
TOP_CONTRIBUTIONS: Final = 3
SHORT_TOKENS: Final = EDGES[0]  # below this its capped, never flagged


class Thresholds(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    sentence: float
    sentence_recall: float
    false_highlight_rate: float  # human docs w/ at least one flagged sentence
    document: float
    document_tpr: float
    document_fpr: float = 0.01

    @classmethod
    def from_report(cls, path: Path, run: str) -> "Thresholds":
        payload = orjson.loads(path.read_bytes())
        point = next(p for p in payload["sentences"]["points"] if p["name"] == SENTENCE_POINT)
        doc = next(m for m in payload["documents"] if m["model"] == run and m["split"] == DOC_SPLIT)
        return cls(
            sentence=point["threshold"],
            sentence_recall=point["cells"][-1]["recall"],
            false_highlight_rate=point["false_highlight_rate"],
            document=doc["threshold_1pct"],
            document_tpr=doc["tpr_at_1pct_fpr"],
        )


class SentenceScore(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    start: int
    end: int  # char offsets into the nfc text
    n_tokens: int
    score: float  # calibrated, even prior
    flagged: bool
    too_short: bool
    contributions: list[tuple[str, float]]  # top signed, + means more machine


class ScoreResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str
    sentences: list[SentenceScore]
    document_score: float
    document_flagged: bool
    caveats: list[str]


def build_result(
    text: str,
    spans: list[tuple[int, int]],
    n_tokens: list[int],
    probs: np.ndarray,
    doc_prob: float,
    contributions: list[list[tuple[str, float]]],
    thresholds: Thresholds,
) -> ScoreResult:
    # no model here, split out so its testable
    sentences = []
    for (a, b), n, p, c in zip(spans, n_tokens, probs, contributions, strict=True):
        too_short = n < SHORT_TOKENS
        sentences.append(
            SentenceScore(
                start=a,
                end=b,
                n_tokens=n,
                score=float(p),
                flagged=bool(p >= thresholds.sentence) and not too_short,
                too_short=too_short,
                contributions=c,
            )
        )
    caveats = [
        f"At this sentence threshold, {thresholds.false_highlight_rate:.0%} of wholly human test "
        f"documents still had at least one flagged sentence, and "
        f"{thresholds.sentence_recall:.0%} of machine sentences were flagged.",
        f"The document flag catches about {thresholds.document_tpr:.0%} of machine documents "
        f"from an unseen corpus while flagging {thresholds.document_fpr:.0%} of human ones.",
        "Paraphrased machine text is not detected.",
    ]
    if sum(not s.too_short for s in sentences) < 3:
        caveats.append("Very little scorable text; the result carries little evidence.")
    return ScoreResult(
        text=text,
        sentences=sentences,
        document_score=doc_prob,
        document_flagged=doc_prob >= thresholds.document,
        caveats=caveats,
    )


class Scorer:
    # cpu fp32 by default. extraction ran fp16 on gpu (~0.05 nats off on lm feats) so
    # scores wont exactly match the stored preds

    def __init__(self, checkpoint: Path, report: Path, run: str, device: str = "cpu") -> None:
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

    def score(self, raw: str) -> ScoreResult:
        from aivhuman.features import extract, lm
        from aivhuman.mil.data import Bags
        from aivhuman.mil.predict import explain
        from aivhuman.mil.train import score
        from aivhuman.text.normalize import nfc
        from aivhuman.text.tokens import count_tokens

        if len(raw) > MAX_CHARS:
            raise ValueError(f"text longer than {MAX_CHARS:,} characters")
        text = nfc(raw)
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
            doc_ids=["input"],
            labels=np.zeros(1, dtype=np.float32),
            x=self.std.transform(x),
            offsets=np.array([0, n]),
            span_labels=np.full(n, -1.0),
            straddles=np.zeros(n, dtype=bool),
        )
        s = score(self.model, bags)
        contributions = explain(self.model, self.std, x, TOP_CONTRIBUTIONS)
        probs = self.calibrator.apply(s.sentence_logits.astype(np.float64), np.asarray(n_tokens))
        doc_prob = float(1.0 / (1.0 + np.exp(-s.doc_logits[0])))
        return build_result(text, spans, n_tokens, probs, doc_prob, contributions, self.thresholds)
