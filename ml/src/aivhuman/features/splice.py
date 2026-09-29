"""Mixed human/machine documents spliced from RAID groups, for within-document contrast."""

# TLDR;
# RAID is 100% human or machine. Document labels solely reward any document level cue
# Joining a human doc. sentence to a machine from the same group gives positive bag that
# that shares its topic! These splices are for validation only.

import random
from collections import defaultdict
from collections.abc import Sequence
from typing import Final

from aivhuman.features.load import SpanDoc
from aivhuman.schema import LABEL_HUMAN, LABEL_MACHINE

SPLICES_PER_GROUP: Final = 2
SEED: Final = 20240502


def _attack(d: SpanDoc) -> str:
    """The `@attack` suffix of an adversarial document's breakdown, or `""` for a clean one."""
    return d.breakdown.partition("@")[2]


def splice(
    first: SpanDoc, second: SpanDoc, n_first: int, start_second: int, doc_id: str
) -> SpanDoc:
    """`first`'s first `n_first` sentences, a space, then `second`'s from `start_second` on."""
    a_end = first.spans[n_first - 1][1]
    b_start = second.spans[start_second][0]
    shift = a_end + 1 - b_start
    text = first.text[:a_end] + " " + second.text[b_start:]
    spans = [
        *first.spans[:n_first],
        *((s + shift, e + shift) for s, e in second.spans[start_second:]),
    ]
    sentence_labels = [first.label] * n_first + [second.label] * (len(second.spans) - start_second)
    return SpanDoc(
        doc_id=doc_id,
        text=text,
        label=LABEL_MACHINE,
        group_id=first.group_id,
        domain=first.domain,
        breakdown=f"spliced@{_attack(first)}" if _attack(first) else "spliced",
        spans=spans,
        span_tokens=[*first.span_tokens[:n_first], *second.span_tokens[start_second:]],
        span_labels=sentence_labels,
        straddles=[False] * len(spans),
        detok_style=first.detok_style,
    )


def build(
    docs: Sequence[SpanDoc], per_group: int = SPLICES_PER_GROUP, seed: int = SEED
) -> list[SpanDoc]:
    """Spliced documents for every group with a human doc of 2+ sentences and a machine doc."""
    rng = random.Random(seed)
    human: dict[tuple[str, str], SpanDoc] = {}
    machine: dict[tuple[str, str], list[SpanDoc]] = defaultdict(list)
    for d in docs:
        key = (d.group_id, _attack(d))
        if d.label == LABEL_HUMAN:
            human[key] = d
        else:
            machine[key].append(d)

    out = []
    for key in sorted(human):
        h = human[key]
        candidates = [m for m in machine.get(key, []) if len(m.spans) >= 2]
        if len(h.spans) < 2 or not candidates:
            continue
        group, attack = key
        stem = f"{group}@{attack}" if attack else group
        for k in range(per_group):
            m = rng.choice(candidates)
            n = min(len(h.spans), len(m.spans))
            n_human = rng.randint(1, n - 1)
            doc_id = f"{stem}:splice{k}"
            if rng.random() < 0.5:
                out.append(splice(h, m, n_human, n_human, doc_id))
            else:
                out.append(splice(m, h, n - n_human, n - n_human, doc_id))
    return out
