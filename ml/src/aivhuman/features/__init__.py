"""per sentence features for the MIL model"""

from typing import Final

# predictability under GPT-2 small, mean over the span tokens (lm.py)
LM_FEATURES: Final = ("lm_logprob", "lm_logrank", "lm_top10", "lm_entropy")
# span vs the rest of its doc (lm.py)
CONTEXT_FEATURES: Final = ("ctx_logprob_delta", "ctx_burstiness")
# word choice + local repetition (lexical.py)
LEXICAL_FEATURES: Final = (
    "lex_mattr",
    "lex_word_len",
    "lex_function_rate",
    "lex_rare_rate",
    "rep_max_cos",
)
# spacy universal POS tags, each becomes a share of the span tokens (syntax.py)
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
)  # column order in the feature files, NaN = undefined for that span

# extracted but dropped before training, see vet.py for the checks
EXCLUDED: Final[frozenset[str]] = frozenset(
    {
        "ctx_burstiness",  # length rho -0.54; dev AUROC 0.42 -> mage-x 0.50
        "lex_mattr",  # length rho -0.51
        "pos_punct",  # direction reverses across corpora: dev 0.52 -> mage-x 0.41
        # together w/ lm_logprob it gives the doc mean logprob, MIL used it as a per doc
        # shortcut. sentence AUROC on seqxgpt-calib went 0.76 -> 0.61
        "ctx_logprob_delta",
    }
)
