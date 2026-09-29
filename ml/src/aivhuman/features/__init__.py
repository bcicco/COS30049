"""Per-sentence interpretable features for the MIL detector."""

from typing import Final

LM_FEATURES: Final = ("lm_logprob", "lm_logrank", "lm_top10", "lm_entropy")
CONTEXT_FEATURES: Final = ("ctx_logprob_delta", "ctx_burstiness")
LEXICAL_FEATURES: Final = (
    "lex_mattr",
    "lex_word_len",
    "lex_function_rate",
    "lex_rare_rate",
    "rep_max_cos",
)
POS_TAGS: Final = (
    "NOUN",
    "VERB",
    "ADJ",
    "ADV",
    "PRON",
    "DET",
    "ADP",
    "AUX",
    "CCONJ",
    "PROPN",
    "NUM",
    "PUNCT",
)
SYNTAX_FEATURES: Final = (*(f"pos_{t.lower()}" for t in POS_TAGS), "syn_dep_depth")
LENGTH_FEATURES: Final = ("len_tokens",)

FEATURE_NAMES: Final = (
    *LM_FEATURES,
    *CONTEXT_FEATURES,
    *LEXICAL_FEATURES,
    *SYNTAX_FEATURES,
    *LENGTH_FEATURES,
)
"""Column order of every feature file. NaN marks a feature undefined for that span."""

EXCLUDED: Final[frozenset[str]] = frozenset(
    {
        "ctx_burstiness",  # length rho -0.54; dev AUROC 0.42 -> mage-x 0.50
        "lex_mattr",  # length rho -0.51
        "pos_punct",  # direction reverses across corpora: dev 0.52 -> mage-x 0.41
        # With lm_logprob it spans the document mean log-prob, which MIL learned as a
        # per-document shortcut -> sentence AUROC on seqxgpt-calib fell to 0.61 from 0.76.
        "ctx_logprob_delta",
    }
)
"""Features dropped before modelling, after vetting for corpus and length artefacts."""
