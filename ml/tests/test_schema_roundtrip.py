import unicodedata
from typing import Any

import orjson
import pytest
from pydantic import ValidationError

from aivhuman.schema import (
    LABEL_HUMAN,
    LABEL_MACHINE,
    Doc,
    SchemaError,
    SentenceSpan,
    doc_from_json,
    doc_to_json,
    validate_doc,
)

DOC_KEYS = [
    "doc_id",
    "text",
    "label",
    "source",
    "domain",
    "generator",
    "group_id",
    "split_role",
    "sentences",
    "label_raw",
    "meta",
]

RAID_TEXT = "One sentence. Two sentences."
SEQXGPT_TEXT = "Human wrote this. Machine wrote that."


def raid_doc(**overrides: Any) -> Doc:
    kwargs: dict[str, Any] = {
        "doc_id": "raid:abc-123",
        "text": RAID_TEXT,
        "label": LABEL_MACHINE,
        "source": "raid",
        "domain": "news",
        "generator": "gpt4",
        "group_id": "raid:src-999",
        "split_role": "train_pool",
        "sentences": [
            SentenceSpan(start=0, end=13, n_tokens=3, n_words=2),
            SentenceSpan(start=14, end=28, n_tokens=4, n_words=2),
        ],
        "label_raw": "gpt4",
        "meta": {"attack": "none", "decoding": "greedy"},
    }
    kwargs.update(overrides)
    return Doc(**kwargs)


def seqxgpt_doc(**overrides: Any) -> Doc:
    kwargs: dict[str, Any] = {
        "doc_id": "seqxgpt:en_gpt2_lines:000001",
        "text": SEQXGPT_TEXT,
        "label": LABEL_MACHINE,
        "source": "seqxgpt",
        "domain": None,
        "generator": "gpt2",
        "group_id": "seqxgpt:base:deadbeefdeadbeef",
        "split_role": "calib_pool",
        "sentences": [
            SentenceSpan(
                start=0, end=17, n_tokens=4, n_words=3, label=LABEL_HUMAN, machine_char_frac=0.0
            ),
            SentenceSpan(
                start=18, end=37, n_tokens=4, n_words=3, label=LABEL_MACHINE, machine_char_frac=1.0
            ),
        ],
        "label_raw": "gpt2",
        "meta": {"prompt_len_orig": 18},
    }
    kwargs.update(overrides)
    return Doc(**kwargs)


@pytest.mark.parametrize("factory", [raid_doc, seqxgpt_doc], ids=["raid", "seqxgpt"])
def test_roundtrip_is_lossless(factory) -> None:  # type: ignore[no-untyped-def]
    doc = factory()
    assert doc_from_json(doc_to_json(doc)) == doc


def test_key_order_is_fixed() -> None:
    assert list(orjson.loads(doc_to_json(raid_doc()))) == DOC_KEYS


def test_optional_span_keys_are_omitted_at_defaults() -> None:
    payload = orjson.loads(doc_to_json(raid_doc()))
    assert list(payload["sentences"][0]) == ["start", "end", "n_tokens", "n_words"]


def test_seqxgpt_span_keys_include_labels() -> None:
    span = orjson.loads(doc_to_json(seqxgpt_doc()))["sentences"][0]
    assert "label" in span
    assert "machine_char_frac" in span


def test_straddle_flag_survives_roundtrip() -> None:
    doc = seqxgpt_doc(
        sentences=[
            SentenceSpan(
                start=0,
                end=17,
                n_tokens=4,
                n_words=3,
                label=LABEL_MACHINE,
                machine_char_frac=0.6,
                straddles_boundary=True,
            ),
            SentenceSpan(
                start=18, end=37, n_tokens=4, n_words=3, label=LABEL_MACHINE, machine_char_frac=1.0
            ),
        ]
    )
    back = doc_from_json(doc_to_json(doc))
    assert back.sentences[0].straddles_boundary is True
    assert back.sentences[0].machine_char_frac == pytest.approx(0.6)


def test_missing_key_is_a_schema_error() -> None:
    payload = orjson.loads(doc_to_json(raid_doc()))
    del payload["label_raw"]
    with pytest.raises(SchemaError):
        doc_from_json(orjson.dumps(payload))


def test_unknown_key_is_rejected() -> None:
    payload = orjson.loads(doc_to_json(raid_doc()))
    payload["labl_raw"] = "typo"
    with pytest.raises(SchemaError):
        doc_from_json(orjson.dumps(payload))


def test_malformed_json_is_a_schema_error() -> None:
    with pytest.raises(SchemaError, match="malformed JSON"):
        doc_from_json(b"{not json")


def test_docs_are_frozen() -> None:
    doc = raid_doc()
    with pytest.raises(ValidationError):
        doc.label = LABEL_HUMAN  # type: ignore[misc]


def test_spans_are_frozen() -> None:
    span = SentenceSpan(start=0, end=5, n_tokens=1, n_words=1)
    with pytest.raises(ValidationError):
        span.start = 3  # type: ignore[misc]


def test_validate_doc_accepts_valid_documents() -> None:
    validate_doc(raid_doc())
    validate_doc(seqxgpt_doc())


def test_validate_doc_catches_a_model_construct_bypass() -> None:
    bad = Doc.model_construct(
        doc_id="raid:x",
        text="hello",
        label=7,
        source="raid",
        domain=None,
        generator=None,
        group_id="raid:g",
        split_role="train_pool",
        sentences=[],
        label_raw="human",
        meta={},
    )
    with pytest.raises(SchemaError):
        validate_doc(bad)


@pytest.mark.parametrize(("start", "end"), [(5, 5), (10, 5)])
def test_empty_or_inverted_spans_are_rejected(start: int, end: int) -> None:
    with pytest.raises(ValidationError, match="empty or inverted"):
        SentenceSpan(start=start, end=end, n_tokens=1, n_words=1)


def test_negative_offsets_are_rejected() -> None:
    with pytest.raises(ValidationError):
        SentenceSpan(start=-1, end=5, n_tokens=1, n_words=1)


def test_span_past_end_of_text_is_rejected() -> None:
    with pytest.raises(ValidationError, match="past text length"):
        raid_doc(sentences=[SentenceSpan(start=0, end=999, n_tokens=1, n_words=1)])


def test_overlapping_spans_are_rejected() -> None:
    with pytest.raises(ValidationError, match="before previous end"):
        raid_doc(
            sentences=[
                SentenceSpan(start=0, end=20, n_tokens=4, n_words=3),
                SentenceSpan(start=10, end=28, n_tokens=4, n_words=3),
            ]
        )


def test_non_whitespace_gap_is_rejected() -> None:
    # pysbd actually dropped 413 chars from the middle of an arxiv abstract once
    with pytest.raises(ValidationError, match="non-whitespace gap"):
        raid_doc(
            sentences=[
                SentenceSpan(start=0, end=13, n_tokens=3, n_words=2),
                SentenceSpan(start=20, end=28, n_tokens=2, n_words=1),
            ]
        )


def test_uncovered_tail_is_rejected() -> None:
    with pytest.raises(ValidationError, match="tail"):
        raid_doc(sentences=[SentenceSpan(start=0, end=13, n_tokens=3, n_words=2)])


def test_whitespace_only_gaps_are_fine() -> None:
    validate_doc(raid_doc())


@pytest.mark.parametrize("label", [2, -1, 99])
def test_non_binary_label_is_rejected(label: int) -> None:
    with pytest.raises(ValidationError):
        raid_doc(label=label)


def test_unknown_source_is_rejected() -> None:
    with pytest.raises(ValidationError, match="unknown source"):
        raid_doc(source="kaggle")


def test_unknown_split_role_is_rejected() -> None:
    with pytest.raises(ValidationError, match="unknown split_role"):
        raid_doc(split_role="train")


def test_missing_source_prefix_is_rejected() -> None:
    with pytest.raises(ValidationError, match="lacks the"):
        raid_doc(doc_id="abc-123")
    with pytest.raises(ValidationError, match="lacks the source prefix"):
        raid_doc(group_id="src-999")


def test_empty_label_raw_is_rejected() -> None:
    with pytest.raises(ValidationError):
        raid_doc(label_raw="")


def test_non_nfc_text_is_rejected() -> None:
    # use NFD here, editors precompose the literal on save
    text = unicodedata.normalize("NFD", "café is decomposed.")
    assert not unicodedata.is_normalized("NFC", text), "fixture must be decomposed"
    with pytest.raises(ValidationError, match="NFC"):
        raid_doc(
            text=text,
            sentences=[SentenceSpan(start=0, end=len(text), n_tokens=5, n_words=3)],
        )


def test_seqxgpt_spans_must_all_be_labelled() -> None:
    with pytest.raises(ValidationError, match="without a label"):
        seqxgpt_doc(
            sentences=[
                SentenceSpan(
                    start=0, end=17, n_tokens=4, n_words=3, label=LABEL_HUMAN, machine_char_frac=0.0
                ),
                SentenceSpan(start=18, end=37, n_tokens=4, n_words=3),
            ]
        )


def test_other_sources_must_not_carry_sentence_labels() -> None:
    with pytest.raises(ValidationError, match="must not carry sentence labels"):
        raid_doc(
            sentences=[
                SentenceSpan(start=0, end=13, n_tokens=3, n_words=2, label=LABEL_MACHINE),
                SentenceSpan(start=14, end=28, n_tokens=4, n_words=2),
            ]
        )


def test_human_doc_must_not_name_a_generator() -> None:
    with pytest.raises(ValidationError, match="human doc names generator"):
        raid_doc(label=LABEL_HUMAN)


def test_mage_may_disagree_because_paraphrased_human_is_labelled_machine() -> None:
    # mage labels paraphrased human as machine on purpose, so generator/label can disagree
    doc = Doc(
        doc_id="mage:ood_gpt_para:0000001",
        text=RAID_TEXT,
        label=LABEL_HUMAN,
        source="mage",
        domain="cnn",
        generator="human",
        group_id="mage:abcdef0123456789",
        split_role="xcorpus_para_test",
        sentences=[
            SentenceSpan(start=0, end=13, n_tokens=3, n_words=2),
            SentenceSpan(start=14, end=28, n_tokens=4, n_words=2),
        ],
        label_raw="cnn_human_para",
    )
    assert doc.generator == "human"
