from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from testbed.simverify.annotations import SECTORS, condition_vector
from testbed.simverify.gates import (
    assign_episode_splits,
    build_condition_support_index,
    cycle_condition_schema,
    gate_thresholds_contract,
    transition_inventory,
    validate_condition_materialization,
    validate_gate_contract,
)


def test_cycle_condition_is_exactly_6d_and_changes_only_at_cycle_boundary() -> None:
    schema = cycle_condition_schema()
    left_to_center = np.asarray(
        condition_vector("left", "center"),
        dtype=np.float32,
    )
    center_to_right = np.asarray(
        condition_vector("center", "right"),
        dtype=np.float32,
    )
    condition = np.stack(
        [
            np.zeros(6, dtype=np.float32),
            left_to_center,
            left_to_center,
            center_to_right,
            center_to_right,
        ]
    )
    cycle_id = np.asarray([-1, 0, 0, 1, 1], dtype=np.int64)
    valid_mask = np.asarray([False, True, True, True, True])

    validate_condition_materialization(condition, cycle_id, valid_mask)

    assert schema["dtype"] == "float32"
    assert schema["shape"] == [6]
    assert schema["normalization"] == "identity"
    assert schema["low_dim_injection"]["field_order"] == [
        "qpos",
        "qvel",
        "cycle_condition_v1",
    ]
    assert schema["low_dim_injection"]["independent_transformer_token"] is False
    np.testing.assert_array_equal(left_to_center, [1, 0, 0, 0, 1, 0])
    np.testing.assert_array_equal(center_to_right, [0, 1, 0, 0, 0, 1])

    changes_inside_cycle = condition.copy()
    changes_inside_cycle[2] = center_to_right
    with pytest.raises(ValueError, match="changes inside cycle"):
        validate_condition_materialization(
            changes_inside_cycle,
            cycle_id,
            valid_mask,
        )


@pytest.mark.parametrize(
    ("condition", "match"),
    [
        (np.zeros((2, 5), dtype=np.float32), "shape must be"),
        (np.zeros((2, 6), dtype=np.float64), "dtype must be"),
        (
            np.asarray(
                [
                    [1, 1, 0, 0, 1, 0],
                    [1, 0, 0, 0, 1, 0],
                ],
                dtype=np.float32,
            ),
            "current-sector condition is not one-hot",
        ),
    ],
)
def test_cycle_condition_validation_fails_closed(
    condition: np.ndarray,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        validate_condition_materialization(
            condition,
            np.zeros(2, dtype=np.int64),
            np.ones(2, dtype=bool),
        )


def test_episode_split_is_deterministic_disjoint_and_source_episode_level() -> None:
    episode_epochs = {
        **{episode_id: "epoch_a" for episode_id in range(1, 7)},
        **{episode_id: "epoch_b" for episode_id in range(7, 13)},
    }

    split = assign_episode_splits(
        episode_epochs,
        seed="simverify-m0",
        validation_per_stratum=1,
        test_per_stratum=1,
    )
    repeated = assign_episode_splits(
        episode_epochs,
        seed="simverify-m0",
        validation_per_stratum=1,
        test_per_stratum=1,
    )

    assert split == repeated
    split_sets = {
        name: set(episode_ids)
        for name, episode_ids in split["splits"].items()
    }
    assert split_sets["train"].isdisjoint(split_sets["validation"])
    assert split_sets["train"].isdisjoint(split_sets["held_out_test"])
    assert split_sets["validation"].isdisjoint(split_sets["held_out_test"])
    assert set().union(*split_sets.values()) == set(episode_epochs)
    for details in split["strata"].values():
        assert len(details["validation"]) == 1
        assert len(details["held_out_test"]) == 1
        assert len(details["train"]) == 4
    assert split["unit"] == "source_episode"
    assert (
        split["held_out_test_policy"]["threshold_generation_allowed"] is False
    )


def _accepted_cycle(
    episode_id: int,
    cycle_id: int,
    current: str,
    next_sector: str,
) -> dict[str, object]:
    return {
        "episode_id": episode_id,
        "cycle_id": cycle_id,
        "quality": {"status": "accepted"},
        "policy_condition": {
            "current_sector": current,
            "next_ready_sector": next_sector,
            "vector": condition_vector(current, next_sector),
        },
    }


def test_transition_inventory_counts_only_adjacent_accepted_cycles() -> None:
    records = [
        _accepted_cycle(1, 0, "left", "center"),
        _accepted_cycle(1, 1, "center", "right"),
        _accepted_cycle(1, 2, "right", "left"),
        _accepted_cycle(2, 0, "right", "right"),
        _accepted_cycle(2, 1, "left", "center"),
        _accepted_cycle(3, 0, "center", "left"),
        {
            **_accepted_cycle(1, 3, "left", "left"),
            "quality": {"status": "ambiguous"},
        },
    ]
    split = {
        "splits": {
            "train": [1],
            "validation": [2],
            "held_out_test": [3],
        }
    }

    inventory = transition_inventory(records, split)

    train = inventory["splits"]["train"]
    assert train["accepted_cycle_count"] == 3
    assert train["adjacent_two_cycle_pair_count"] == 2
    assert train["transition_matrix"]["left"]["center"] == 1
    assert train["transition_matrix"]["center"]["right"] == 1
    assert train["nonzero_transition_count"] == 2
    assert train["three_cycle_inventory"]["left->center->right"] == 1
    assert train["continuity_errors"] == []

    validation = inventory["splits"]["validation"]
    assert validation["transition_matrix"]["right"]["right"] == 1
    assert validation["continuity_errors"] == [
        {
            "episode_id": 2,
            "left_cycle_id": 0,
            "right_cycle_id": 1,
            "left_next": "right",
            "right_current": "left",
        }
    ]
    assert inventory["condition_source"] == "hindsight_outcome"


def _support_entries() -> list[dict[str, object]]:
    bases = {
        "left": np.asarray([1.0, 0.0, 0.0]),
        "center": np.asarray([0.0, 1.0, 0.0]),
        "right": np.asarray([0.0, 0.0, 1.0]),
    }
    entries: list[dict[str, object]] = []
    for episode_id, split_name, perturbation in (
        (1, "train", 0.01),
        (2, "train", 0.02),
        (3, "validation", 0.03),
        (90, "held_out_test", 0.00),
        (91, "held_out_test", 0.00),
    ):
        for cycle_id, sector in enumerate(SECTORS):
            feature = bases[sector] + perturbation
            entries.append(
                {
                    "episode_id": episode_id,
                    "cycle_id": cycle_id,
                    "split": split_name,
                    "current_sector": sector,
                    "next_sector": sector,
                    "current_feature": feature.copy(),
                    "next_feature": feature.copy(),
                }
            )
    return entries


def test_condition_support_never_uses_held_out_episodes() -> None:
    split = {
        "splits": {
            "train": [1, 2],
            "validation": [3],
            "held_out_test": [90, 91],
        }
    }
    entries = _support_entries()

    index = build_condition_support_index(entries, split=split)

    assert index["support_splits"] == ["train", "validation"]
    assert index["held_out_test_used_as_support"] is False
    for indexed in index["entries"]:
        for role in ("current", "next"):
            for sector in SECTORS:
                neighbors = indexed["counterfactuals"][role][sector][
                    "nearest_neighbors"
                ]
                assert all(
                    neighbor["episode_id"] not in {90, 91}
                    for neighbor in neighbors
                )
                assert all(
                    neighbor["evidence_split"] in {"train", "validation"}
                    for neighbor in neighbors
                )

    changed = deepcopy(entries)
    for entry in changed:
        if entry["episode_id"] in {90, 91}:
            entry["current_feature"] = np.asarray([50.0, -7.0, 3.0])
            entry["next_feature"] = np.asarray([-9.0, 2.0, 40.0])
    changed_index = build_condition_support_index(changed, split=split)
    assert changed_index["distance_thresholds"] == index["distance_thresholds"]
    assert changed_index["entries"][:9] == index["entries"][:9]


def test_m0_gate_contract_keeps_all_values_deferred_and_test_locked() -> None:
    contract = gate_thresholds_contract(
        split_manifest_sha256="a" * 64,
        annotation_manifest_sha256="b" * 64,
        bootstrap_replicates=2000,
        bootstrap_seed=20260724,
    )

    validate_gate_contract(contract)

    assert contract["evidence_scope"] == "recorded-observation/offline"
    assert contract["held_out_test"] == {
        "authorized": False,
        "allowed_inputs_for_threshold_generation": ["train", "validation"],
        "forbidden_inputs": ["held_out_test"],
    }
    assert contract["training_authorized"] is False
    assert contract["control_candidate"] is False
    for family in contract["threshold_families"].values():
        for metric in family:
            assert metric["status"] == "deferred"
            assert metric["value"] is None
            assert metric["required_artifact_sha256"] == []

    g4 = {
        metric["metric"]: metric
        for metric in contract["threshold_families"]["G4"]
    }
    latency = g4["token_response_latency_ticks"]
    assert "expert_train_validation_distribution" in latency["required_sources"]
    assert (
        "B1_repeated_same_checkpoint_validation_replay"
        in latency["required_sources"]
    )
    assert "B1_same_checkpoint_repeat_latency_jitter" in latency[
        "threshold_formula"
    ]
    repeat = g4["same_token_repeat_consistency"]
    assert repeat["required_sources"] == [
        "B1_repeated_same_checkpoint_validation_replay",
        "condition_support_index",
    ]
    assert "B2" not in repeat["threshold_formula"]
    assert "repeat_consistency_validation_q02_5" in repeat["threshold_formula"]

    for metric in contract["threshold_families"]["G5"][:2]:
        assert (
            "B0_repeated_same_checkpoint_validation_replay"
            in metric["required_sources"]
        )

    unlocked = deepcopy(contract)
    unlocked["held_out_test"]["authorized"] = True
    with pytest.raises(ValueError, match="held-out test locked"):
        validate_gate_contract(unlocked)

    fabricated = deepcopy(contract)
    fabricated["threshold_families"]["G4"][0]["status"] = "generated"
    fabricated["threshold_families"]["G4"][0]["value"] = 0.1
    with pytest.raises(ValueError, match="deferred/null"):
        validate_gate_contract(fabricated)
