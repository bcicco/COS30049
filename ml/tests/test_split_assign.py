"""Split assignment on synthetic rows: held-out routing, merging, overlap drops."""

import pytest

from aivhuman.schema import LABEL_HUMAN, LABEL_MACHINE
from aivhuman.splits import (
    Row,
    SplitError,
    SplitStats,
    assign,
    check_disjoint,
    merge_groups,
    split_mage,
    split_raid,
)


def row(
    doc_id: str,
    group_id: str,
    *,
    key: str | None = None,
    label: int = LABEL_MACHINE,
    domain: str | None = "news",
    generator: str | None = "gpt4",
    split_role: str = "train_pool",
) -> Row:
    return Row(
        doc_id=doc_id,
        group_id=group_id,
        label=label,
        domain=domain,
        generator=generator,
        split_role=split_role,
        key=key or doc_id,
    )


def raid_corpus(n_groups: int = 400) -> list[Row]:
    rows = []
    for g in range(n_groups):
        domain = ("news", "books", "reviews", "wiki")[g % 4]
        gid = f"raid:g{g}"
        rows.append(row(f"raid:h{g}", gid, label=LABEL_HUMAN, domain=domain, generator=None))
        for gen in ("gpt4", "cohere", "cohere-chat", "mpt"):
            rows.append(row(f"raid:{gen}{g}", gid, domain=domain, generator=gen))
    return rows


def test_merge_groups_is_order_independent() -> None:
    rows = [row("a", "s:3", key="k"), row("b", "s:1", key="k"), row("c", "s:2", key="j")]
    assert merge_groups(rows) == merge_groups(rows[::-1])
    assert merge_groups(rows) == {"s:3": "s:1", "s:1": "s:1", "s:2": "s:2"}


def test_merge_groups_is_transitive() -> None:
    rows = [row("a", "s:c", key="x"), row("b", "s:b", key="x"), row("c", "s:b", key="y")]
    rows.append(row("d", "s:a", key="y"))
    assert set(merge_groups(rows).values()) == {"s:a"}


def test_raid_training_never_sees_held_out_generators_or_domains() -> None:
    out = split_raid(raid_corpus(), SplitStats())
    for name in ("train", "dev"):
        assert out[name]
        assert not {r.generator for r in out[name]} & {"cohere", "cohere-chat"}
        assert not {r.domain for r in out[name]} & {"reviews", "wiki"}


def test_raid_ood_covers_all_three_cells() -> None:
    ood = split_raid(raid_corpus(), SplitStats())["raid-ood"]
    cells = {(r.domain in {"reviews", "wiki"}, r.generator == "cohere") for r in ood}
    assert {(True, False), (False, True), (True, True)} <= cells


def test_raid_groups_stay_whole_except_dropped_generators() -> None:
    stats = SplitStats()
    out = split_raid(raid_corpus(), stats)
    check_disjoint(out)
    kept = sum(len(v) for v in out.values())
    assert kept + stats.dropped["raid_held_out_generator"] == len(raid_corpus())


def test_duplicate_text_pulls_groups_into_one_split() -> None:
    rows = raid_corpus()
    # Every news group shares a text with the next one, chaining them together.
    rows += [row(f"raid:dup{g}", f"raid:g{g}", key=f"raid:h{g + 4}") for g in range(0, 396, 4)]
    out = split_raid(rows, SplitStats())
    news_splits = {n for n, rs in out.items() for r in rs if r.domain == "news"}
    assert len(news_splits) == 1


def test_mage_drops_foreign_and_para_overlap() -> None:
    rows = [
        row("mage:a", "mage:1", key="shared", split_role="xcorpus_test"),
        row("mage:b", "mage:2", split_role="xcorpus_test"),
        row("mage:c", "mage:3", split_role="xcorpus_test"),
        row("mage:d", "mage:3", split_role="xcorpus_para_test"),
        row("mage:e", "mage:4", split_role="xcorpus_ood_test"),
    ]
    stats = SplitStats()
    out = split_mage(rows, {"shared"}, stats)
    assert [r.doc_id for r in out["mage-x"]] == ["mage:b", "mage:e"]
    assert [r.doc_id for r in out["mage-para"]] == ["mage:d"]
    assert stats.dropped == {"mage_shared_with_other_corpus": 1, "mage_x_shared_with_para": 1}


def test_assign_is_deterministic_and_splits_seqxgpt_by_group() -> None:
    seq = [row(f"seqxgpt:f:{i}", f"seqxgpt:b{i // 3}", split_role="calib_pool") for i in range(300)]
    splits, stats = assign(raid_corpus(), [], seq)
    again, _ = assign(raid_corpus(), [], seq[::-1])
    assert {k: sorted(r.doc_id for r in v) for k, v in splits.items()} == {
        k: sorted(r.doc_id for r in v) for k, v in again.items()
    }
    assert 30 < stats.groups["seqxgpt-calib"] < 70
    assert stats.docs["seqxgpt-calib"] + stats.docs["seqxgpt-test"] == 300


def test_check_disjoint_catches_a_straddling_group() -> None:
    with pytest.raises(SplitError):
        check_disjoint({"train": [row("a", "g")], "dev": [row("b", "g")]})
