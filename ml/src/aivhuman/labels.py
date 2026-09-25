"""Every label derivation in the project, in one module."""

from __future__ import annotations

import inspect
import re
import sys
from typing import Final, NamedTuple

from aivhuman.schema import LABEL_HUMAN, LABEL_MACHINE
from aivhuman.text.normalize import stable_hash

__all__ = [
    "MAGE_DOMAINS",
    "MAGE_MODELS",
    "MAGE_STRATEGIES",
    "RAID_DOMAINS",
    "RAID_MODELS",
    "SEQXGPT_GENERATORS",
    "ParsedSrc",
    "UnknownLabelError",
    "mage_label",
    "parse_src",
    "polarity_rules_fingerprint",
    "raid_label",
    "seqxgpt_doc_label",
]


class UnknownLabelError(ValueError):
    """An upstream label value we have never seen. Never silently defaulted."""


# --------------------------------------------------------------------------- #
# RAID
# --------------------------------------------------------------------------- #

#: The ``model`` column's full range. ``"human"`` is a value here, which is the
#: only reason RAID can be labelled at all -- there is no label column.
RAID_MODELS: Final = frozenset(
    {
        "human",
        "chatgpt",
        "gpt4",
        "gpt3",
        "gpt2",
        "llama-chat",
        "mistral",
        "mistral-chat",
        "mpt",
        "mpt-chat",
        "cohere",
        "cohere-chat",
    }
)

#: Domains present in ``train.csv``. ``extra.csv`` adds code/Czech/German, which
#: are out of scope, so an unexpected domain here means the wrong file.
RAID_DOMAINS: Final = frozenset(
    {"abstracts", "books", "news", "poetry", "recipes", "reddit", "reviews", "wiki"}
)


def raid_label(model: str) -> int:
    """Derive RAID's document label from its ``model`` column.

    >>> raid_label("human"), raid_label("gpt4")
    (0, 1)

    Raises:
        UnknownLabelError: on any value outside :data:`RAID_MODELS`, including
            near-misses like ``"Human"`` that a permissive rule would silently
            classify as machine.
    """
    if model not in RAID_MODELS:
        raise UnknownLabelError(f"RAID model {model!r} not in the known set")
    return LABEL_HUMAN if model == "human" else LABEL_MACHINE


# --------------------------------------------------------------------------- #
# MAGE
# --------------------------------------------------------------------------- #

#: 14 domains, enumerated from the data. Note PLAN.md and the MAGE paper both
#: describe 10 -- ``cnn``, ``dialogsum``, ``imdb`` and ``pubmed`` appear in the
#: OOD testbeds and are absent from the documented list.
MAGE_DOMAINS: Final = frozenset(
    {
        "cmv",
        "cnn",
        "dialogsum",
        "eli5",
        "hswag",
        "imdb",
        "pubmed",
        "roct",
        "sci_gen",
        "squad",
        "tldr",
        "wp",
        "xsum",
        "yelp",
    }
)

#: Prompting strategy. Not mentioned in PLAN.md, and worth keeping: it is a
#: generalisation axis (does a detector trained on continuations transfer to
#: topically-prompted text?) that costs nothing to carry through.
MAGE_STRATEGIES: Final = frozenset({"continuation", "specified", "topical"})

MAGE_MODELS: Final = frozenset(
    {
        "human",
        "gpt4",
        "7B",
        "13B",
        "30B",
        "65B",
        "GLM130B",
        "bloom_7b",
        "flan_t5_small",
        "flan_t5_base",
        "flan_t5_large",
        "flan_t5_xl",
        "flan_t5_xxl",
        "gpt-3.5-trubo",  # misspelled upstream, kept verbatim
        "gpt_j",
        "gpt_neox",
        "opt_125m",
        "opt_350m",
        "opt_1.3b",
        "opt_2.7b",
        "opt_6.7b",
        "opt_13b",
        "opt_30b",
        "opt_iml_30b",
        "opt_iml_max_1.3b",
        "t0_3b",
        "t0_11b",
        "text-davinci-002",
        "text-davinci-003",
    }
)

_MAGE_LABELS: Final = {"1": LABEL_HUMAN, "0": LABEL_MACHINE}

_MACHINE_SRC_RE = re.compile(
    r"^(?P<domain>.+?)_machine_(?P<strategy>continuation|specified|topical)_(?P<model>.+)$"
)


def mage_label(raw: str) -> int:
    """Collapse MAGE's ``label`` column to canonical polarity"""
    try:
        return _MAGE_LABELS[raw.strip()]
    except KeyError:
        raise UnknownLabelError(f"MAGE label {raw!r} is neither '0' nor '1'") from None


class ParsedSrc(NamedTuple):
    """Decomposition of a MAGE ``src`` string."""

    domain: str | None
    generator: str | None
    strategy: str | None
    is_paraphrased: bool
    ok: bool


def parse_src(src: str) -> ParsedSrc:
    """Decompose a MAGE ``src`` into domain, generator, strategy and paraphrase.

    MAGE has **three** ``src`` grammars, not the one its documentation implies:

    ``{domain}_human``
        ``cmv_human`` -- human-written.
    ``{domain}_machine_{strategy}_{model}``
        ``eli5_machine_continuation_flan_t5_xl`` -- the bulk of the corpus.
    ``{domain}_{model}``
        ``cnn_gpt4`` -- the GPT-4 OOD testbeds only.

    Any of them may carry a trailing ``_para``."""
    core = src
    is_para = False
    if core.endswith("_para"):
        core = core[: -len("_para")]
        is_para = True

    match = _MACHINE_SRC_RE.match(core)
    if match:
        domain = match.group("domain")
        model = match.group("model")
        ok = domain in MAGE_DOMAINS and model in MAGE_MODELS
        return ParsedSrc(domain, model, match.group("strategy"), is_para, ok)

    if core.endswith("_human"):
        domain = core[: -len("_human")]
        return ParsedSrc(domain, "human", None, is_para, domain in MAGE_DOMAINS)

    # {domain}_{model}: longest-first over the closed domain vocabulary.
    for domain in sorted(MAGE_DOMAINS, key=len, reverse=True):
        prefix = f"{domain}_"
        if core.startswith(prefix):
            model = core[len(prefix) :]
            return ParsedSrc(domain, model, None, is_para, model in MAGE_MODELS)

    return ParsedSrc(None, None, None, is_para, False)


# --------------------------------------------------------------------------- #
# SeqXGPT
# --------------------------------------------------------------------------- #

#: ``gpt3re`` is the upstream spelling for the GPT-3 re-generation variant.
SEQXGPT_GENERATORS: Final = frozenset({"gpt2", "gptneo", "gptj", "llama", "gpt3re", "human"})


def seqxgpt_doc_label(raw: str, boundary: int, length: int) -> int:
    """Document label for a SeqXGPT record."""

    # boundary is where the human prompt ends.

    if raw not in SEQXGPT_GENERATORS:
        raise UnknownLabelError(f"SeqXGPT label {raw!r} not in the known set")
    if raw == "human":
        return LABEL_HUMAN
    return LABEL_HUMAN if boundary >= length else LABEL_MACHINE


def polarity_rules_fingerprint() -> str:
    return stable_hash(inspect.getsource(sys.modules[__name__]))
