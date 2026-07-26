from __future__ import annotations

import itertools

import numpy as np

from testbed.simverify.next_condition_causal import _evaluate
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


def _support_anchors() -> list[dict]:
    pairs = list(itertools.permutations(SECTORS, 2))
    rows = []
    for episode_id in (3, 4, 6, 7):
        for index, (base, target) in enumerate(pairs):
            rows.append(
                _anchor(
                    split="train",
                    episode_id=episode_id,
                    cycle_id=index,
                    base=base,
                    target=target,
                )
            )
    for episode_id in (12, 34):
        for index, (base, target) in enumerate(pairs):
            rows.append(
                _anchor(
                    split="validation",
                    episode_id=episode_id,
                    cycle_id=index,
                    base=base,
                    target=target,
                )
            )
    return rows


def _replay_row(anchor: dict, *, scale: float) -> dict:
    centers = {"left": 0.1, "center": 0.2, "right": 0.3}
    base = anchor["base_condition"]["next_sector"]
    target = anchor["target_condition"]["next_sector"]
    direction = int(np.sign(centers[target] - centers[base]))
    return {
        "episode_id": anchor["episode_id"],
        "cycle_id": anchor["cycle_id"],
        "changed_factor": "next_sector",
        "base_condition": {"next_sector": base},
        "target_condition": {"next_sector": target},
        "metrics": {
            "token_swap_action_effect": abs(scale),
            "swing_action_delta_mean": direction * scale,
            "per_tick_effect_l1": [0.0, abs(scale)],
            "relevant_window_local": [1, 2],
            "event_coverage_delta": 0.0,
            "target_event_order_valid": True,
        },
    }


def test_next_condition_gate_uses_only_predeclared_informative_anchors() -> None:
    support_contract = derive_next_condition_support(
        _support_anchors(),
        sector_centers={"left": 0.1, "center": 0.2, "right": 0.3},
    )
    validation = [
        row for row in _support_anchors() if row["split"] == "validation"
    ]
    candidate_rows = {
        index: _replay_row(row, scale=0.2)
        for index, row in enumerate(validation)
    }
    shuffled_rows = {
        index: _replay_row(row, scale=0.02)
        for index, row in enumerate(validation)
    }
    masked_rows = {
        index: _replay_row(row, scale=0.0)
        for index, row in enumerate(validation)
    }

    def package(rows: dict, repeat_id: int = 0) -> dict:
        return {
            "supported_rows": rows,
            "manifest": {"repeat_id": repeat_id},
        }

    source_rows, criteria, noise = _evaluate(
        reference=package(candidate_rows),
        repeats=[package(candidate_rows, 1), package(candidate_rows, 2)],
        shuffled=package(shuffled_rows),
        masked=package(masked_rows),
        support={
            "eligible_validation_source_episode_ids": [12, 34],
            "permutation_support": support_contract["permutation_support"],
        },
        sector_centers={"left": 0.1, "center": 0.2, "right": 0.3},
        action_direction_sign=1,
        repetitions=10_000,
        seed=17,
    )

    assert len(source_rows) == 2
    assert all(item["passed"] for item in criteria.values())
    permutation_results = criteria["semantic_identifiability"][
        "permutation_results"
    ]
    assert len(permutation_results) == 5
    assert all(
        result["informative_source_episode_count"] == 2
        for result in permutation_results.values()
    )
    assert noise["signed_semantic_margin_q97_5"] == 0.0
