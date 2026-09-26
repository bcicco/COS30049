"""Group ids that keep related documents on the same side of a split.

- RAID: `source_id` identifies the human original.
- MAGE: no source identifier exists, so each document is its own group.
- SeqXGPT: no base-document field exists; groups are recovered from shared
  human prefixes (prompts)
"""

from collections import Counter, defaultdict
from collections.abc import Sequence
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from aivhuman.text.normalize import content_key, stable_hash, text_key

# Maximum prefix length used as a grouping key. Fixed-length keys either
# fuse distinct documents that share  short keys or run
# past long keys, so each key uses as much of the
# record's human prefix as exists, up to this cap.
PREFIX_CHARS: Final = 200

# Measured with fixed-length keys. 40 chars gives 99.3% recovery but 224
# same-file collisions; 120 chars gives 20 collisions but 93% recovery.

# Minimum key length... shorter keys are too generic to identify a document.
MIN_PREFIX_CHARS: Final = 16

# Minimum shared prefix length required to link two keys.
MIN_LINK_CHARS: Final = 24


def raid_group_id(source_id: str) -> str:
    """Group RAID rows by their human source document."""
    return f"raid:{source_id}"


def mage_group_id(text: str) -> str:
    """Return a per-document group id for MAGE, derived from a content hash."""
    return f"mage:{stable_hash(text_key(text))}"


class GroupStats(BaseModel):
    """Quality metrics for SeqXGPT group recovery."""

    model_config = ConfigDict(extra="forbid")

    n_records: int = 0
    n_groups: int = 0
    n_keys: int = 0
    prefix_merges: int = 0
    same_file_collisions: int = 0
    multi_file_frac: float = 0.0
    singleton_frac: float = 0.0
    prefix_chars: int = PREFIX_CHARS
    size_hist: dict[int, int] = Field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "n_records": self.n_records,
            "n_groups": self.n_groups,
            "n_keys": self.n_keys,
            "prefix_merges": self.prefix_merges,
            "same_file_collisions": self.same_file_collisions,
            "collision_rate": round(self.collision_rate, 6),
            "multi_file_frac": round(self.multi_file_frac, 4),
            "singleton_frac": round(self.singleton_frac, 4),
            "prefix_chars": self.prefix_chars,
            "size_hist": {str(k): v for k, v in sorted(self.size_hist.items())},
        }

    @property
    def collision_rate(self) -> float:
        return self.same_file_collisions / self.n_records if self.n_records else 0.0

    @property
    def is_green(self) -> bool:
        """Whether recovery meets the acceptance bar.

        Requires >=95% multi-file recovery and a same-file collision rate of at
        most 0.1%. The collision bar is a rate rather than zero because SeqXGPT
        contains a small number of genuine duplicate base documents based on exploration.
        """
        return self.multi_file_frac >= 0.95 and self.collision_rate <= 0.001


class GroupAssignment(BaseModel):
    """Per-record group ids in input order, with recovery stats."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    group_ids: list[str]
    stats: GroupStats


def recover_seqxgpt_groups(
    records: Sequence[tuple[str, str, int | None]],
    prefix_chars: int = PREFIX_CHARS,
    min_link_chars: int = MIN_LINK_CHARS,
) -> GroupAssignment:
    """Recover the base document each SeqXGPT record derives from."""
    # ************* IMPORTANT *************************************
    # `records` is `[(file_stem, text, prompt_len), ...]`; `prompt_len` is
    # `None` for fully human records.

    # SeqXGPT records carry no identifier --> row indices are not aligned across
    # files AND `prompt_len` varies by generator for the same base.

    n = len(records)
    stats = GroupStats(n_records=n, prefix_chars=prefix_chars)
    if n == 0:
        return GroupAssignment(group_ids=[], stats=stats)

    # Key each record on as much of its human prefix as it has.
    keys: list[str] = []
    for _stem, text, prompt_len in records:
        budget = len(text) if prompt_len is None else prompt_len
        budget = max(MIN_PREFIX_CHARS, min(prefix_chars, budget))
        keys.append(content_key(text, budget))

    unique = sorted(set(keys))
    parent = {k: k for k in unique}

    def find(k: str) -> str:
        root = k
        while parent[root] != root:
            root = parent[root]
        while parent[k] != root:  # path compression
            parent[k], k = root, parent[k]
        return root

    def union(a: str, b: str) -> bool:
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        parent[rb] = ra
        return True

    # Sorting places each key directly before its extensions the stack tracks
    # the current chain of prefixes so one pass links them all.
    stack: list[str] = []
    for key in unique:
        while stack and not key.startswith(stack[-1]):
            stack.pop()
        if stack and len(stack[-1]) >= min_link_chars and union(stack[-1], key):
            stats.prefix_merges += 1
        stack.append(key)

    stats.n_keys = len(unique)

    roots = [find(k) for k in keys]
    final: dict[str, list[int]] = defaultdict(list)
    for idx, root in enumerate(roots):
        final[root].append(idx)

    multi_file = 0
    for idxs in final.values():
        stems = Counter(records[i][0] for i in idxs)
        stats.same_file_collisions += sum(c - 1 for c in stems.values() if c > 1)
        if len(stems) > 1:
            multi_file += len(idxs)

    stats.n_groups = len(final)
    stats.size_hist = dict(Counter(len(v) for v in final.values()))
    stats.multi_file_frac = multi_file / n
    stats.singleton_frac = sum(1 for v in final.values() if len(v) == 1) / n

    group_ids = [f"seqxgpt:base:{stable_hash(root)}" for root in roots]
    return GroupAssignment(group_ids=group_ids, stats=stats)
