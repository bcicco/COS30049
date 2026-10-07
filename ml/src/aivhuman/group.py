"""group ids so related docs land on the same side of a split

RAID has source_id, MAGE has nothing so every doc is its own group, SeqXGPT
groups get recovered from the shared human prefix (prompt)
"""

from collections import Counter, defaultdict
from collections.abc import Sequence
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from aivhuman.text.normalize import content_key, stable_hash, text_key

# max prefix len for the grouping key. fixed len keys either merge diff docs w/
# short keys or run past the prefix, so use as much human prefix as there is up to this
PREFIX_CHARS: Final = 200

# w/ fixed len keys: 40 chars -> 99.3% recovery but 224 same-file collisions,
# 120 chars -> 20 collisions but only 93% recovery

# shorter keys are too generic to id a doc
MIN_PREFIX_CHARS: Final = 16

# a key must be at least this long before longer keys can be merged into it
MIN_LINK_CHARS: Final = 24


def raid_group_id(source_id: str) -> str:
    """RAID generations share the source_id of the human doc they came from"""
    return f"raid:{source_id}"


def mage_group_id(text: str) -> str:
    """no link in MAGE, each doc is its own group keyed on its text hash"""
    return f"mage:{stable_hash(text_key(text))}"


class GroupStats(BaseModel):
    """how well the seqxgpt group recovery went"""

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
        # collision bar isnt 0 bc SeqXGPT has a few real duplicate base docs (saw in exploration)
        return self.multi_file_frac >= 0.95 and self.collision_rate <= 0.001


class GroupAssignment(BaseModel):
    """one group id per input record, same order"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    group_ids: list[str]
    stats: GroupStats


def recover_seqxgpt_groups(
    records: Sequence[tuple[str, str, int | None]],
    prefix_chars: int = PREFIX_CHARS,
    min_link_chars: int = MIN_LINK_CHARS,
) -> GroupAssignment:
    # ************* IMPORTANT *************************************
    # records is [(file_stem, text, prompt_len), ...], prompt_len is None for fully human ones

    # SeqXGPT records have no id --> row indices dont line up across
    # files AND prompt_len changes by generator for the same base.

    # so: key each record on a normalised slice of its human prefix, then union keys where
    # one is a prefix of another (short prompt in one file, longer prompt in another)

    n = len(records)
    stats = GroupStats(n_records=n, prefix_chars=prefix_chars)
    if n == 0:
        return GroupAssignment(group_ids=[], stats=stats)

    # never read past the human prefix, the machine text differs per generator
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

    # after sorting each key sits right before its extensions, stack holds the
    # current prefix chain so one pass links everything
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

    # sanity stats: a real group spans several generator files, two records from the same
    # file in one group means two different base docs got merged
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
