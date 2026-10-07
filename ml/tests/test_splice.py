import pytest

pytest.importorskip("sklearn")

from aivhuman.features.load import SpanDoc
from aivhuman.features.splice import build, splice


def _doc(doc_id: str, label: int, sentences: list[str], group: str = "raid:g0") -> SpanDoc:
    text, spans, pos = "", [], 0
    for s in sentences:
        if text:
            text += " "
            pos += 1
        spans.append((pos, pos + len(s)))
        text += s
        pos += len(s)
    return SpanDoc(
        doc_id=doc_id,
        text=text,
        label=label,
        group_id=group,
        domain="news",
        breakdown="human" if label == 0 else "gpt2",
        spans=spans,
        span_tokens=[len(s.split()) for s in sentences],
        span_labels=None,
        straddles=[False] * len(spans),
    )


H = _doc("raid:h", 0, ["Human one.", "Human two.", "Human three."])
M = _doc("raid:m", 1, ["Machine one.", "Machine two.", "Machine three.", "Machine four."])


def test_splice_offsets_recover_each_sentence() -> None:
    d = splice(H, M, 2, 2, "raid:g0:splice0")
    assert [d.text[s:e] for s, e in d.spans] == [
        "Human one.",
        "Human two.",
        "Machine three.",
        "Machine four.",
    ]
    assert d.span_labels == [0, 0, 1, 1]
    assert d.label == 1
    assert d.span_tokens == [2, 2, 2, 2]


def test_splice_machine_first() -> None:
    d = splice(M, H, 1, 1, "raid:g0:splice1")
    assert [d.text[s:e] for s, e in d.spans] == ["Machine one.", "Human two.", "Human three."]
    assert d.span_labels == [1, 0, 0]


def test_build_mixes_both_classes_in_every_splice() -> None:
    docs = [H, M, _doc("raid:m2", 1, ["Other one.", "Other two."])]
    out = build(docs, per_group=20, seed=0)
    assert len(out) == 20
    for d in out:
        assert d.span_labels is not None
        assert set(d.span_labels) == {0, 1}
        assert [d.text[s:e] for s, e in d.spans] == [d.text[s:e].strip() for s, e in d.spans]
    orders = {d.span_labels[0] for d in out if d.span_labels}
    assert orders == {0, 1}


def test_splices_never_mix_attacks() -> None:
    docs = []
    for attack in ("synonym", "number"):
        for label, name in ((0, "human"), (1, "gpt2")):
            d = _doc(f"raid:{name}-{attack}", label, [f"S{i} {attack}." for i in range(4)])
            docs.append(d.model_copy(update={"breakdown": f"{name}@{attack}"}))
    out = build(docs, per_group=3)
    assert {d.breakdown for d in out} == {"spliced@synonym", "spliced@number"}
    for d in out:
        attack = d.breakdown.partition("@")[2]
        assert all(attack in d.text[s:e] for s, e in d.spans)
