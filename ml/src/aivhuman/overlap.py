"""Exact-duplicate detection within and across the three corpora."""

# ---------------------------- REASONING -------------------------------
# RAID is the only corpus trained on, so a RAID training document that reappears
# in MAGE turns cross-corpus into a memorisation measurement.
# That number has to be known before the result is quoted, not after.

from itertools import combinations
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from aivhuman.schema import iter_jsonl
from aivhuman.text.normalize import stable_hash, text_key

MAX_EXAMPLES = 10


class OverlapStats(BaseModel):
    """Duplicate content counts, within and across corpora."""

    model_config = ConfigDict(validate_assignment=False, extra="forbid")

    docs: dict[str, int] = Field(default_factory=dict)
    unique_keys: dict[str, int] = Field(default_factory=dict)
    internal_duplicate_docs: dict[str, int] = Field(default_factory=dict)
    """Documents in a corpus whose text already appeared in that same corpus."""

    shared_keys: dict[str, int] = Field(default_factory=dict)
    """Distinct texts appearing in both corpora of a pair, keyed "a|b"."""

    examples: dict[str, list[tuple[str, str]]] = Field(default_factory=dict)

    @property
    def is_clean(self) -> bool:
        """True when no text is shared between two different corpora."""
        return not any(self.shared_keys.values())

    def as_dict(self) -> dict[str, Any]:
        return {
            "docs": dict(sorted(self.docs.items())),
            "unique_keys": dict(sorted(self.unique_keys.items())),
            "internal_duplicate_docs": dict(sorted(self.internal_duplicate_docs.items())),
            "shared_keys": dict(sorted(self.shared_keys.items())),
            "is_clean": self.is_clean,
            "examples": {k: [list(p) for p in v] for k, v in sorted(self.examples.items())},
        }


def overlap_report(directory: Path) -> OverlapStats:
    """Hash every document in every JSONL and count the collisions."""
    stats = OverlapStats()
    keys: dict[str, dict[str, str]] = {}

    for path in sorted(directory.glob("*.jsonl")):
        source = path.stem
        seen: dict[str, str] = {}
        internal = 0
        n = 0
        for doc in iter_jsonl(path):
            n += 1
            key = stable_hash(text_key(doc.text))
            if key in seen:
                internal += 1
            else:
                seen[key] = doc.doc_id
        keys[source] = seen
        stats.docs[source] = n
        stats.unique_keys[source] = len(seen)
        stats.internal_duplicate_docs[source] = internal

    for left, right in combinations(sorted(keys), 2):
        shared = keys[left].keys() & keys[right].keys()
        pair = f"{left}|{right}"
        stats.shared_keys[pair] = len(shared)
        if shared:
            stats.examples[pair] = [
                (keys[left][key], keys[right][key]) for key in sorted(shared)[:MAX_EXAMPLES]
            ]
    return stats
