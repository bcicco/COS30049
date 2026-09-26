"""Label polarity and src parsing — the regression suite for Phase 1's worst bug."""

import pytest

from aivhuman.labels import (
    MAGE_DOMAINS,
    MAGE_MODELS,
    MAGE_STRATEGIES,
    RAID_MODELS,
    SEQXGPT_GENERATORS,
    UnknownLabelError,
    mage_label,
    parse_src,
    raid_label,
    seqxgpt_doc_label,
)
from aivhuman.schema import LABEL_HUMAN, LABEL_MACHINE

# --------------------------------------------------------------------------- #
# RAID
# --------------------------------------------------------------------------- #


def test_raid_human_is_human() -> None:
    assert raid_label("human") == LABEL_HUMAN


@pytest.mark.parametrize("model", sorted(RAID_MODELS - {"human"}))
def test_raid_every_generator_is_machine(model: str) -> None:
    assert raid_label(model) == LABEL_MACHINE


@pytest.mark.parametrize("model", ["Human", "HUMAN", "humans", "", "gpt5", "none"])
def test_raid_unknown_model_raises(model: str) -> None:
    """A permissive rule would classify every one of these as machine.

    `"Human"` is the dangerous one: a casing change upstream would relabel
    every human document as machine, corpus-wide, with no error.
    """
    with pytest.raises(UnknownLabelError):
        raid_label(model)


# --------------------------------------------------------------------------- #
# MAGE polarity
# --------------------------------------------------------------------------- #


def test_mage_polarity_is_inverted() -> None:
    """MAGE's "1" means HUMAN. Confirmed across all 338 distinct src values, and
    corroborated by upstream prepare_testbeds.py, which asserts res[1] == "0"
    for machine-generated rows and "1" for human-written ones.
    """
    assert mage_label("1") == LABEL_HUMAN
    assert mage_label("0") == LABEL_MACHINE


def test_mage_label_tolerates_surrounding_whitespace() -> None:
    assert mage_label(" 1 ") == LABEL_HUMAN


@pytest.mark.parametrize("raw", ["2", "", "human", "-1", "1.0", "True"])
def test_mage_unknown_label_raises(raw: str) -> None:
    with pytest.raises(UnknownLabelError):
        mage_label(raw)


# --------------------------------------------------------------------------- #
# MAGE src parsing
# --------------------------------------------------------------------------- #


def test_human_src_parses() -> None:
    parsed = parse_src("cmv_human")
    assert parsed == ("cmv", "human", None, False, True)


def test_paraphrase_suffix_is_stripped_before_human_matching() -> None:
    """`cnn_human_para` must not parse as domain `cnn_human`.

    Strip `_para` first or the domain vocabulary never matches and the whole
    paraphrase testbed reports as unparsed.
    """
    parsed = parse_src("cnn_human_para")
    assert parsed.domain == "cnn"
    assert parsed.generator == "human"
    assert parsed.is_paraphrased is True
    assert parsed.ok is True


def test_underscored_domain_is_not_split_naively() -> None:
    """The anti-`split("_", 1)` test.

    Naive splitting yields domain `sci`, generator `gen_machine_...`. Closed
    vocabulary matching is the only thing that gets `sci_gen` right, and both
    domains and models here contain underscores.
    """
    parsed = parse_src("sci_gen_machine_continuation_flan_t5_xl")
    assert parsed.domain == "sci_gen"
    assert parsed.generator == "flan_t5_xl"
    assert parsed.strategy == "continuation"
    assert parsed.ok is True


def test_underscored_model_survives() -> None:
    parsed = parse_src("eli5_machine_continuation_opt_iml_max_1.3b")
    assert parsed.domain == "eli5"
    assert parsed.generator == "opt_iml_max_1.3b"
    assert parsed.ok is True


@pytest.mark.parametrize("strategy", sorted(MAGE_STRATEGIES))
def test_every_strategy_parses(strategy: str) -> None:
    parsed = parse_src(f"cmv_machine_{strategy}_gpt_j")
    assert parsed.strategy == strategy
    assert parsed.ok is True


def test_ood_domain_model_grammar() -> None:
    """The third grammar: the GPT-4 OOD testbeds use `{domain}_{model}`."""
    parsed = parse_src("pubmed_gpt4")
    assert parsed == ("pubmed", "gpt4", None, False, True)
    assert parse_src("imdb_gpt4_para").is_paraphrased is True


@pytest.mark.parametrize("domain", sorted(MAGE_DOMAINS))
def test_every_domain_parses_in_all_grammars(domain: str) -> None:
    assert parse_src(f"{domain}_human").domain == domain
    assert parse_src(f"{domain}_machine_continuation_gpt_j").domain == domain
    assert parse_src(f"{domain}_gpt4").domain == domain


@pytest.mark.parametrize("model", sorted(MAGE_MODELS - {"human", "gpt4"}))
def test_every_model_parses(model: str) -> None:
    parsed = parse_src(f"xsum_machine_continuation_{model}")
    assert parsed.generator == model
    assert parsed.ok is True


@pytest.mark.parametrize("src", ["", "nonsense", "notadomain_human", "xsum_notamodel"])
def test_unparseable_src_reports_not_ok_rather_than_guessing(src: str) -> None:
    """Returns `ok=False` instead of raising: the count is reported and gated
    on (`unparsed_src == 0`), which locates a vocabulary drift precisely. A
    raise here would abort a 400k-row pass on its last row.
    """
    assert parse_src(src).ok is False


# --------------------------------------------------------------------------- #
# SeqXGPT
# --------------------------------------------------------------------------- #


def test_seqxgpt_pure_human_document() -> None:
    assert seqxgpt_doc_label("human", boundary=500, length=500) == LABEL_HUMAN


def test_seqxgpt_mixed_document_is_machine() -> None:
    assert seqxgpt_doc_label("gpt2", boundary=100, length=500) == LABEL_MACHINE


def test_seqxgpt_boundary_at_end_is_human() -> None:
    """A generator label with the prefix covering everything leaves no machine text."""
    assert seqxgpt_doc_label("gpt2", boundary=500, length=500) == LABEL_HUMAN


def test_seqxgpt_boundary_at_zero_is_machine() -> None:
    assert seqxgpt_doc_label("llama", boundary=0, length=500) == LABEL_MACHINE


@pytest.mark.parametrize("gen", sorted(SEQXGPT_GENERATORS))
def test_every_seqxgpt_generator_is_known(gen: str) -> None:
    seqxgpt_doc_label(gen, boundary=10, length=100)


@pytest.mark.parametrize("gen", ["gpt3", "gpt5", "", "Human"])
def test_seqxgpt_unknown_generator_raises(gen: str) -> None:
    """`gpt3` is the trap: upstream spells it `gpt3re`."""
    with pytest.raises(UnknownLabelError):
        seqxgpt_doc_label(gen, boundary=10, length=100)
