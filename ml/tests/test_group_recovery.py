# seqxgpt has no base doc id so groups get rebuilt from the shared human prefix

import pytest

from aivhuman.group import (
    MIN_LINK_CHARS,
    MIN_PREFIX_CHARS,
    PREFIX_CHARS,
    mage_group_id,
    raid_group_id,
    recover_seqxgpt_groups,
)

Record = tuple[str, str, int | None]

TAG_A, TAG_B = "alfa", "brav"


def base_text(tag: str, n_words: int = 40) -> str:
    # 8 char words so any multiple of 8 is a word edge
    assert len(tag) == 4, "tag must be 4 chars to keep words 8 chars wide"
    return "".join(f"{tag}{i:03d} " for i in range(n_words))


def rec(stem: str, base: str, prompt_len: int | None) -> Record:
    # prompt_len None = en_human_lines, all human
    if prompt_len is None:
        return (stem, base.strip(), None)
    return (stem, base[:prompt_len] + f"then {stem} generated the rest of it.", prompt_len)


def test_differing_prompt_lens_collapse_to_one_group() -> None:
    # prompt_len differs per generator for the same base (saw 541/541/1012 in the data)
    base = base_text(TAG_A)
    assignment = recover_seqxgpt_groups(
        [
            rec("gpt2_lines", base, 24),
            rec("gptj_lines", base, 80),
            rec("human_lines", base, None),
        ]
    )

    assert len(set(assignment.group_ids)) == 1
    assert assignment.stats.n_keys == 3, "three distinct keys, merged by prefix linking"
    assert assignment.stats.prefix_merges == 2
    assert assignment.stats.same_file_collisions == 0
    assert assignment.stats.multi_file_frac == 1.0
    assert assignment.stats.is_green


def test_prefix_linking_is_transitive_through_a_chain() -> None:
    base = base_text(TAG_A)
    stems = ["gpt2_lines", "gpt3_lines", "gptj_lines", "gptneo_lines", "llama_lines"]
    records = [rec(stem, base, 24 + 8 * i) for i, stem in enumerate(stems)]

    assignment = recover_seqxgpt_groups(records)

    assert len(set(assignment.group_ids)) == 1
    assert assignment.stats.prefix_merges == len(stems) - 1


def test_unrelated_bases_stay_separate() -> None:
    base_a, base_b = base_text(TAG_A), base_text(TAG_B)
    assignment = recover_seqxgpt_groups(
        [
            rec("gpt2_lines", base_a, 80),
            rec("gptj_lines", base_a, 80),
            rec("gpt2_lines", base_b, 80),
            rec("gptj_lines", base_b, 80),
        ]
    )

    assert assignment.stats.n_groups == 2
    assert assignment.stats.size_hist == {2: 2}
    assert assignment.stats.same_file_collisions == 0
    assert assignment.group_ids[0] == assignment.group_ids[1]
    assert assignment.group_ids[2] == assignment.group_ids[3]
    assert assignment.group_ids[0] != assignment.group_ids[2]


def test_a_lone_record_is_its_own_group() -> None:
    assignment = recover_seqxgpt_groups([rec("gpt2_lines", base_text(TAG_A), 80)])

    assert assignment.stats.n_groups == 1
    assert assignment.stats.singleton_frac == 1.0
    assert assignment.stats.multi_file_frac == 0.0


def test_a_key_shorter_than_the_link_threshold_does_not_merge() -> None:
    # 16 chars is too short to link, lots of unrelated docs start the same
    assert MIN_PREFIX_CHARS < MIN_LINK_CHARS

    base = base_text(TAG_A)
    assignment = recover_seqxgpt_groups(
        [
            rec("gpt2_lines", base, MIN_PREFIX_CHARS),
            rec("gptj_lines", base, MIN_LINK_CHARS),
            rec("gptneo_lines", base, 80),
        ]
    )

    assert assignment.stats.n_keys == 3
    assert len(set(assignment.group_ids)) == 2
    assert assignment.group_ids[1] == assignment.group_ids[2]
    assert assignment.group_ids[0] not in assignment.group_ids[1:]


def test_prompt_len_zero_does_not_raise() -> None:
    assignment = recover_seqxgpt_groups([("gpt2_lines", "all of this text is machine.", 0)])

    assert assignment.stats.n_keys == 1
    assert len(assignment.group_ids) == 1


def test_a_short_prefix_cap_fuses_documents_sharing_a_boilerplate_opening() -> None:
    # 40 chars got better recovery but merged docs with the same stock opening,
    # hence the cap of 200
    opening = "in this paper we investigate the effects of "
    doc_a = opening + "high salt on inflammatory mediators in cells."
    doc_b = opening + "low temperature on the yield of the reaction."
    records: list[Record] = [
        ("gpt2_lines", doc_a + " machine one.", len(doc_a)),
        ("gptj_lines", doc_b + " machine two.", len(doc_b)),
    ]

    tight = recover_seqxgpt_groups(records, prefix_chars=40)
    assert len(set(tight.group_ids)) == 1, "40 chars cannot see past the boilerplate"

    wide = recover_seqxgpt_groups(records)
    assert wide.stats.prefix_chars == PREFIX_CHARS
    assert len(set(wide.group_ids)) == 2


def test_two_records_from_one_file_sharing_a_prefix_are_a_collision() -> None:
    # only one record per base per generator, so a repeat in one file = bad merge
    base = base_text(TAG_A)
    assignment = recover_seqxgpt_groups(
        [
            ("gpt2_lines", base[:80] + "first continuation.", 80),
            ("gpt2_lines", base[:80] + "second continuation.", 80),
        ]
    )

    assert len(set(assignment.group_ids)) == 1
    assert assignment.stats.same_file_collisions == 1
    assert assignment.stats.collision_rate == 0.5
    assert assignment.stats.multi_file_frac == 0.0
    assert not assignment.stats.is_green


def test_no_records_yields_no_groups() -> None:
    assignment = recover_seqxgpt_groups([])

    assert assignment.group_ids == []
    assert assignment.stats.n_records == 0
    assert assignment.stats.n_groups == 0
    assert assignment.stats.collision_rate == 0.0
    assert assignment.stats.as_dict()["size_hist"] == {}


def test_group_ids_are_independent_of_input_order() -> None:
    base_a, base_b = base_text(TAG_A), base_text(TAG_B)
    records = [
        rec("gpt2_lines", base_a, 24),
        rec("gptj_lines", base_a, 80),
        rec("gpt2_lines", base_b, 40),
        rec("llama_lines", base_b, 96),
        rec("human_lines", base_a, None),
    ]
    order = [3, 0, 4, 2, 1]

    forward = recover_seqxgpt_groups(records)
    shuffled = recover_seqxgpt_groups([records[i] for i in order])

    by_text = dict(zip((r[1] for r in records), forward.group_ids, strict=True))
    for i, group_id in zip(order, shuffled.group_ids, strict=True):
        assert by_text[records[i][1]] == group_id


def test_group_id_carries_the_source_prefix_the_schema_requires() -> None:
    group_id = recover_seqxgpt_groups([rec("gpt2_lines", base_text(TAG_A), 80)]).group_ids[0]

    source, local_id = group_id.split(":", 1)
    assert source == "seqxgpt"
    assert local_id.startswith("base:")
    assert len(local_id.removeprefix("base:")) == 16


def test_size_hist_and_fractions() -> None:
    base_a, base_b = base_text(TAG_A), base_text(TAG_B)
    stats = recover_seqxgpt_groups(
        [
            rec("gpt2_lines", base_a, 80),
            rec("gptj_lines", base_a, 80),
            rec("gpt2_lines", base_b, 80),
        ]
    ).stats

    assert stats.n_groups == 2
    assert stats.size_hist == {2: 1, 1: 1}
    assert stats.singleton_frac == pytest.approx(1 / 3)
    assert stats.multi_file_frac == pytest.approx(2 / 3)


def test_as_dict_is_json_shaped_for_the_report() -> None:
    stats = recover_seqxgpt_groups([rec("gpt2_lines", base_text(TAG_A), 80)]).stats
    payload = stats.as_dict()

    assert set(payload) == {
        "n_records",
        "n_groups",
        "n_keys",
        "prefix_merges",
        "same_file_collisions",
        "collision_rate",
        "multi_file_frac",
        "singleton_frac",
        "prefix_chars",
        "size_hist",
    }
    assert payload["size_hist"] == {"1": 1}, "int keys would not survive JSON"


def test_is_green_needs_both_recovery_and_a_low_collision_rate() -> None:
    # 95% recovery, 0.1% collisions
    stats = recover_seqxgpt_groups([rec("gpt2_lines", base_text(TAG_A), 80)]).stats

    stats.multi_file_frac, stats.same_file_collisions = 0.99, 0
    assert stats.is_green
    stats.multi_file_frac = 0.94
    assert not stats.is_green
    stats.multi_file_frac, stats.n_records, stats.same_file_collisions = 0.99, 1000, 2
    assert not stats.is_green


def test_raid_group_id_is_the_human_source_document() -> None:
    # group on source_id, adv_source_id would split machine rows from their human text
    assert raid_group_id("abcd-1234") == "raid:abcd-1234"


def test_mage_group_ids_ignore_detokenisation_style() -> None:
    assert mage_group_id("Disease . Next  one") == mage_group_id("disease.  next one")
    assert mage_group_id("one text") != mage_group_id("another text")
    assert mage_group_id("one text").startswith("mage:")
