from __future__ import annotations

import itertools

from testbed.simverify.next_condition_support import (
    SECTORS,
    derive_next_condition_support,
)


def _anchor(
    *,
    split: str,
    episode_id: int,
    cycle_id: int,
    base: str,
    target: str,
) -> dict:
    return {
        "split": split,
        "episode_id": episode_id,
        "cycle_id": cycle_id,
        "changed_factors": ["next_sector"],
        "supported": True,
        "base_condition": {"next_sector": base},
        "target_condition": {"next_sector": target},
    }


def _episode_rows(split: str, episode_id: int, repeats: int) -> list[dict]:
    pairs = list(itertools.permutations(SECTORS, 2))
    return [
        _anchor(
            split=split,
            episode_id=episode_id,
            cycle_id=index,
            base=pairs[index % len(pairs)][0],
            target=pairs[index % len(pairs)][1],
        )
        for index in range(repeats)
    ]


def test_support_gate_excludes_single_anchor_episode_before_bootstrap() -> None:
    anchors = [
        *_episode_rows("train", 3, 3),
        *_episode_rows("train", 4, 4),
        *_episode_rows("train", 6, 5),
        *_episode_rows("train", 7, 3),
        *_episode_rows("validation", 12, 6),
        *_episode_rows("validation", 20, 1),
        *_episode_rows("validation", 34, 6),
    ]
    result = derive_next_condition_support(
        anchors,
        sector_centers={"left": 0.1, "center": 0.2, "right": 0.3},
    )

    assert result["thresholds"][
        "minimum_supported_anchors_per_source_episode"
    ] == 3
    assert result["gate"]["eligible_validation_source_episode_ids"] == [12, 34]
    assert result["gate"]["excluded_validation_source_episode_ids"] == [20]
    assert result["gate"]["passed"] is True
    assert all(
        row["validation"]["informative_source_episode_count"] >= 2
        for row in result["permutation_support"]["permutations"]
    )


def test_support_gate_fails_when_a_permutation_has_one_source_episode() -> None:
    anchors = [
        *_episode_rows("train", 3, 3),
        *_episode_rows("train", 4, 4),
        *_episode_rows("train", 6, 5),
        *_episode_rows("train", 7, 3),
        *_episode_rows("validation", 12, 3),
        # Only center<->right pairs in the second eligible episode leave
        # left<->center semantic swaps without a second informative source.
        _anchor(
            split="validation",
            episode_id=34,
            cycle_id=0,
            base="center",
            target="right",
        ),
        _anchor(
            split="validation",
            episode_id=34,
            cycle_id=1,
            base="right",
            target="center",
        ),
        _anchor(
            split="validation",
            episode_id=34,
            cycle_id=2,
            base="center",
            target="right",
        ),
    ]
    result = derive_next_condition_support(
        anchors,
        sector_centers={"left": 0.1, "center": 0.2, "right": 0.3},
    )

    assert result["gate"]["passed"] is False
    assert result["gate"]["criteria"][
        "all_five_permutations_have_informative_validation_support"
    ]["passed"] is False
