"""Split, transition, support, and threshold contracts for SimVerify."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from testbed.simverify.annotations import SECTORS, unit_normalize

SPLIT_SCHEMA = "simverify_episode_split_v1"
CONDITION_SCHEMA_ID = "cycle_condition_v1"
GATE_CONTRACT_SCHEMA = "gate_thresholds_contract_v1"
SPLIT_NAMES = ("train", "validation", "held_out_test")


def cycle_condition_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": CONDITION_SCHEMA_ID,
        "title": "Observable-only conditioned-cycle ACT condition",
        "schema_version": CONDITION_SCHEMA_ID,
        "dtype": "float32",
        "shape": [6],
        "normalization": "identity",
        "encoding": "current_sector_onehot3_plus_next_ready_sector_onehot3",
        "indices": {
            "0": "current_left",
            "1": "current_center",
            "2": "current_right",
            "3": "next_left",
            "4": "next_center",
            "5": "next_right",
        },
        "valid_values": [0.0, 1.0],
        "valid_row_invariants": [
            "sum(vector[0:3]) == 1",
            "sum(vector[3:6]) == 1",
        ],
        "invalid_row": {
            "vector": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "must_be_excluded_by_valid_mask": True,
        },
        "condition_source": "hindsight_outcome",
        "historical_command": "unknown_not_recorded",
        "cycle_update": "atomic_at_observable_ready_boundary",
        "low_dim_injection": {
            "field_order": ["qpos", "qvel", CONDITION_SCHEMA_ID],
            "condition_normalization": "identity",
            "projection": "existing_input_proj_robot_state",
            "independent_transformer_token": False,
        },
        "deployment": {
            "state_domain": "sim_source_representation",
            "real_control_arming_allowed": False,
        },
    }


def _stable_rank(seed: str, stratum: str, episode_id: int) -> str:
    payload = f"{seed}:{stratum}:{int(episode_id)}".encode()
    return hashlib.sha256(payload).hexdigest()


def assign_episode_splits(
    episode_epochs: Mapping[int, str],
    *,
    seed: str,
    validation_per_stratum: int = 2,
    test_per_stratum: int = 2,
) -> dict[str, Any]:
    """Deterministically split whole episodes, stratified by source epoch."""

    if not episode_epochs:
        raise ValueError("episode_epochs is empty")
    groups: dict[str, list[int]] = defaultdict(list)
    for episode_id, epoch in episode_epochs.items():
        groups[str(epoch)].append(int(episode_id))

    result = {name: [] for name in SPLIT_NAMES}
    stratum_details: dict[str, Any] = {}
    for stratum in sorted(groups):
        ordered = sorted(
            groups[stratum],
            key=lambda episode_id: _stable_rank(seed, stratum, episode_id),
        )
        required = int(validation_per_stratum) + int(test_per_stratum) + 1
        if len(ordered) < required:
            raise ValueError(
                f"stratum {stratum!r} has {len(ordered)} episodes, need {required}"
            )
        validation = ordered[:validation_per_stratum]
        test = ordered[
            validation_per_stratum :
            validation_per_stratum + test_per_stratum
        ]
        train = ordered[validation_per_stratum + test_per_stratum :]
        result["train"].extend(train)
        result["validation"].extend(validation)
        result["held_out_test"].extend(test)
        stratum_details[stratum] = {
            "ordered_episode_ids": ordered,
            "train": sorted(train),
            "validation": sorted(validation),
            "held_out_test": sorted(test),
        }

    for name in SPLIT_NAMES:
        result[name] = sorted(result[name])
    sets = {name: set(result[name]) for name in SPLIT_NAMES}
    if any(
        sets[left] & sets[right]
        for index, left in enumerate(SPLIT_NAMES)
        for right in SPLIT_NAMES[index + 1 :]
    ):
        raise AssertionError("episode split overlap")
    if set().union(*sets.values()) != set(map(int, episode_epochs)):
        raise AssertionError("episode split does not cover the source inventory")
    return {
        "schema": SPLIT_SCHEMA,
        "unit": "source_episode",
        "seed": str(seed),
        "stratification_field": "controller_epoch",
        "assignment_method": "sha256_rank_within_stratum",
        "splits": result,
        "strata": stratum_details,
        "held_out_test_policy": {
            "threshold_generation_allowed": False,
            "parameter_free_export_and_integrity_qc_allowed": True,
            "annotation_application_before_gate_threshold_freeze": False,
            "model_eval_before_gate_threshold_freeze": False,
        },
    }


def _accepted_cycles(
    records: Sequence[Mapping[str, Any]],
    episode_ids: set[int],
) -> dict[int, list[Mapping[str, Any]]]:
    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        episode_id = int(record["episode_id"])
        if (
            episode_id in episode_ids
            and record["quality"]["status"] == "accepted"
            and record["policy_condition"]["vector"] is not None
        ):
            grouped[episode_id].append(record)
    for episode_id in grouped:
        grouped[episode_id].sort(key=lambda record: int(record["cycle_id"]))
    return grouped


def transition_inventory(
    records: Sequence[Mapping[str, Any]],
    split: Mapping[str, Any],
    *,
    locked_splits: Sequence[str] = (),
) -> dict[str, Any]:
    locked = set(map(str, locked_splits))
    unknown = locked - set(SPLIT_NAMES)
    if unknown:
        raise ValueError(f"unknown locked split names: {sorted(unknown)}")
    inventories: dict[str, Any] = {}
    for split_name in SPLIT_NAMES:
        episode_ids = set(map(int, split["splits"][split_name]))
        if split_name in locked:
            inventories[split_name] = {
                "status": "locked_unread",
                "episode_ids": sorted(episode_ids),
                "accepted_cycle_count": None,
                "sector_counts": None,
                "adjacent_two_cycle_pair_count": None,
                "transition_matrix": None,
                "nonzero_transition_count": None,
                "three_cycle_inventory": None,
                "nonzero_three_cycle_count": None,
                "continuity_errors": None,
                "included_in_success_denominator": False,
            }
            continue
        grouped = _accepted_cycles(records, episode_ids)
        sector_counts = Counter()
        transitions = Counter()
        triples = Counter()
        adjacent_pairs = 0
        continuity_errors: list[dict[str, Any]] = []
        for episode_id, cycles in grouped.items():
            for cycle in cycles:
                sector_counts[
                    str(cycle["policy_condition"]["current_sector"])
                ] += 1
            for left, right in zip(cycles[:-1], cycles[1:]):
                if int(right["cycle_id"]) != int(left["cycle_id"]) + 1:
                    continue
                current = str(left["policy_condition"]["current_sector"])
                expected_next = str(
                    left["policy_condition"]["next_ready_sector"]
                )
                actual_next = str(
                    right["policy_condition"]["current_sector"]
                )
                adjacent_pairs += 1
                transitions[(current, expected_next)] += 1
                if expected_next != actual_next:
                    continuity_errors.append(
                        {
                            "episode_id": episode_id,
                            "left_cycle_id": int(left["cycle_id"]),
                            "right_cycle_id": int(right["cycle_id"]),
                            "left_next": expected_next,
                            "right_current": actual_next,
                        }
                    )
            for first, second, third in zip(cycles[:-2], cycles[1:-1], cycles[2:]):
                if not (
                    int(second["cycle_id"]) == int(first["cycle_id"]) + 1
                    and int(third["cycle_id"]) == int(second["cycle_id"]) + 1
                ):
                    continue
                triples[
                    (
                        str(first["policy_condition"]["current_sector"]),
                        str(second["policy_condition"]["current_sector"]),
                        str(third["policy_condition"]["current_sector"]),
                    )
                ] += 1
        matrix = {
            current: {
                next_sector: int(transitions[(current, next_sector)])
                for next_sector in SECTORS
            }
            for current in SECTORS
        }
        triple_matrix = {
            f"{first}->{second}->{third}": int(
                triples[(first, second, third)]
            )
            for first in SECTORS
            for second in SECTORS
            for third in SECTORS
        }
        inventories[split_name] = {
            "status": "computed",
            "episode_ids": sorted(episode_ids),
            "accepted_cycle_count": int(sum(sector_counts.values())),
            "sector_counts": {
                sector: int(sector_counts[sector]) for sector in SECTORS
            },
            "adjacent_two_cycle_pair_count": int(adjacent_pairs),
            "transition_matrix": matrix,
            "nonzero_transition_count": int(
                sum(value > 0 for row in matrix.values() for value in row.values())
            ),
            "three_cycle_inventory": triple_matrix,
            "nonzero_three_cycle_count": int(
                sum(value > 0 for value in triple_matrix.values())
            ),
            "continuity_errors": continuity_errors,
            "included_in_success_denominator": True,
        }
    return {
        "schema": "split_transition_inventory_v1",
        "accepted_only": True,
        "command_source": "unknown_not_recorded",
        "condition_source": "hindsight_outcome",
        "locked_splits": sorted(locked),
        "splits": inventories,
    }


def validate_condition_materialization(
    condition: np.ndarray,
    cycle_id: np.ndarray,
    valid_mask: np.ndarray,
) -> None:
    values = np.asarray(condition)
    ids = np.asarray(cycle_id)
    valid = np.asarray(valid_mask, dtype=bool)
    if values.dtype != np.float32:
        raise ValueError(f"condition dtype must be float32, got {values.dtype}")
    if values.ndim != 2 or values.shape[1] != 6:
        raise ValueError(f"condition shape must be (T, 6), got {values.shape}")
    if ids.shape != (values.shape[0],) or valid.shape != (values.shape[0],):
        raise ValueError("cycle_id and valid_mask must have shape (T,)")
    if not np.isfinite(values).all():
        raise ValueError("condition contains non-finite values")
    if not np.isin(values, (0.0, 1.0)).all():
        raise ValueError("condition must be binary one-hot")
    if np.any(np.sum(values[valid, :3], axis=1) != 1.0):
        raise ValueError("valid current-sector condition is not one-hot")
    if np.any(np.sum(values[valid, 3:], axis=1) != 1.0):
        raise ValueError("valid next-sector condition is not one-hot")
    if np.any(values[~valid] != 0.0):
        raise ValueError("invalid rows must carry an all-zero condition")
    for identifier in np.unique(ids[valid]):
        rows = values[valid & (ids == identifier)]
        if rows.size and not np.all(rows == rows[0]):
            raise ValueError(f"condition changes inside cycle {int(identifier)}")
    if values.shape[0] > 1:
        token_change = np.any(values[1:] != values[:-1], axis=1)
        cycle_change = ids[1:] != ids[:-1]
        if np.any(token_change & ~cycle_change):
            raise ValueError("condition changed without an atomic cycle boundary")


def _same_sector_leave_episode_out_distances(
    entries: Sequence[Mapping[str, Any]],
    *,
    role: str,
) -> np.ndarray:
    distances: list[float] = []
    for entry in entries:
        candidates = [
            other
            for other in entries
            if int(other["episode_id"]) != int(entry["episode_id"])
            and str(other[f"{role}_sector"]) == str(entry[f"{role}_sector"])
        ]
        if not candidates:
            continue
        query = unit_normalize(np.asarray(entry[f"{role}_feature"], dtype=np.float64))
        candidate_features = unit_normalize(
            np.stack(
                [
                    np.asarray(other[f"{role}_feature"], dtype=np.float64)
                    for other in candidates
                ],
                axis=0,
            )
        )
        distances.append(float(np.min(1.0 - candidate_features @ query)))
    return np.asarray(distances, dtype=np.float64)


def build_condition_support_index(
    entries: Sequence[Mapping[str, Any]],
    *,
    split: Mapping[str, Any],
    distance_quantile: float = 0.95,
) -> dict[str, Any]:
    """Build nearest-neighbor evidence without using held-out episodes as support."""

    train_validation = set(
        map(int, split["splits"]["train"])
    ) | set(map(int, split["splits"]["validation"]))
    support_entries = [
        entry
        for entry in entries
        if int(entry["episode_id"]) in train_validation
    ]
    thresholds: dict[str, float] = {}
    for role in ("current", "next"):
        distances = _same_sector_leave_episode_out_distances(
            support_entries,
            role=role,
        )
        if distances.size == 0:
            raise ValueError(f"no leave-episode-out support distances for {role}")
        thresholds[role] = float(np.quantile(distances, distance_quantile))

    indexed: list[dict[str, Any]] = []
    for entry in entries:
        result: dict[str, Any] = {
            "episode_id": int(entry["episode_id"]),
            "cycle_id": int(entry["cycle_id"]),
            "candidate_condition_schema": CONDITION_SCHEMA_ID,
            "counterfactuals": {},
        }
        for role in ("current", "next"):
            query = unit_normalize(
                np.asarray(entry[f"{role}_feature"], dtype=np.float64)
            )
            role_result: dict[str, Any] = {}
            for sector in SECTORS:
                candidates = [
                    other
                    for other in support_entries
                    if str(other[f"{role}_sector"]) == sector
                    and int(other["episode_id"]) != int(entry["episode_id"])
                ]
                neighbors: list[dict[str, Any]] = []
                for other in candidates:
                    feature = unit_normalize(
                        np.asarray(other[f"{role}_feature"], dtype=np.float64)
                    )
                    neighbors.append(
                        {
                            "episode_id": int(other["episode_id"]),
                            "cycle_id": int(other["cycle_id"]),
                            "evidence_split": str(other["split"]),
                            "distance": float(1.0 - np.dot(query, feature)),
                        }
                    )
                neighbors.sort(key=lambda item: item["distance"])
                nearest = neighbors[:5]
                role_result[sector] = {
                    "supported": bool(
                        nearest
                        and nearest[0]["distance"] <= thresholds[role]
                    ),
                    "distance_threshold": thresholds[role],
                    "nearest_neighbors": nearest,
                }
            result["counterfactuals"][role] = role_result
        indexed.append(result)
    return {
        "schema": "condition_support_index_v1",
        "feature_schema": "observable_qpos_plus_frozen_eye_pair_resnet18_v1",
        "support_splits": ["train", "validation"],
        "held_out_test_used_as_support": False,
        "leave_source_episode_out": True,
        "distance": "cosine",
        "distance_quantile": float(distance_quantile),
        "distance_thresholds": thresholds,
        "entries": indexed,
    }


def gate_thresholds_contract(
    *,
    split_manifest_sha256: str,
    annotation_manifest_sha256: str,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Freeze the method while keeping unavailable model thresholds null."""

    expert_sources = [
        "expert_train_validation_distribution",
        "annotation_manifest",
        "transition_inventory",
    ]
    b0_sources = [
        *expert_sources,
        "B0_repeated_same_checkpoint_validation_replay",
    ]
    condition_sources = [
        "B0_validation_replay",
        "B1_conditioned_validation_replay",
        "B2_shuffled_condition_null",
        "condition_support_index",
    ]
    robustness_sources = [
        "B1_conditioned_validation_replay",
        "B1_perturbation_validation_replay",
        "B1_repeated_same_checkpoint_validation_replay",
    ]

    def deferred(
        metric: str,
        *,
        direction: str,
        required_sources: Sequence[str],
        definition: str,
        aggregation: str,
        threshold_formula: str,
    ) -> dict[str, Any]:
        return {
            "metric": metric,
            "status": "deferred",
            "value": None,
            "unit": "data_generated",
            "direction": direction,
            "required_sources": list(required_sources),
            "definition": definition,
            "aggregation": aggregation,
            "threshold_formula": threshold_formula,
            "estimator": "source_episode_paired_bootstrap",
            "comparison_unit": "same_source_episode_same_anchor",
            "minimum_support": {
                "distinct_source_episodes": 2,
                "unsupported_transition_policy": (
                    "mark_unsupported_and_exclude_from_success_denominator"
                ),
                "imputation_allowed": False,
            },
            "required_artifact_sha256": [],
            "future_sha_binding": (
                "every required source must provide path, schema, sha256, "
                "checkpoint_sha256, dataset_manifest_sha256, and split_sha256"
            ),
            "deferred_reason": "required model replay artifacts do not exist in M0",
        }

    expert_lower = (
        "lower_bound = expert_validation_q02_5 - "
        "B0_same_checkpoint_repeat_abs_delta_q97_5"
    )
    expert_upper = (
        "upper_bound = expert_validation_q97_5 + "
        "B0_same_checkpoint_repeat_abs_delta_q97_5"
    )
    condition_lower = (
        "lower_bound = B2_episode_null_delta_q97_5 + "
        "B1_same_checkpoint_repeat_abs_delta_q97_5; pass only when "
        "paired_bootstrap_CI95(B1_minus_B2).lower > lower_bound"
    )
    condition_upper = (
        "upper_bound = B2_episode_null_q02_5 - "
        "B1_same_checkpoint_repeat_abs_delta_q97_5; pass only when "
        "paired_bootstrap_CI95(B1_minus_B2).upper < upper_bound"
    )
    retention_lower = (
        "lower_bound = unperturbed_B1_validation_q02_5 - "
        "B1_same_checkpoint_repeat_abs_delta_q97_5"
    )
    perturbation_upper = (
        "upper_bound = unperturbed_B1_validation_q97_5 + "
        "B1_same_checkpoint_repeat_abs_delta_q97_5"
    )

    families = {
        "G3": [
            deferred(
                "required_event_coverage",
                direction="lower",
                required_sources=b0_sources,
                definition=(
                    "matched required observable events divided by required "
                    "events for each complete cycle"
                ),
                aggregation="per_cycle_then_source_episode_mean",
                threshold_formula=expert_lower,
            ),
            deferred(
                "event_order_violation_rate",
                direction="upper",
                required_sources=b0_sources,
                definition=(
                    "cycles whose matched event order differs from the frozen "
                    "ready-dig-carry-dump-ready order divided by complete cycles"
                ),
                aggregation="per_source_episode_rate",
                threshold_formula=expert_upper,
            ),
            deferred(
                "missing_phase_rate",
                direction="upper",
                required_sources=b0_sources,
                definition="required task phases absent from one replayed cycle",
                aggregation="per_source_episode_rate",
                threshold_formula=expert_upper,
            ),
            deferred(
                "opposite_direction_rate",
                direction="upper",
                required_sources=b0_sources,
                definition=(
                    "effective action signs opposite to the expert-compatible "
                    "sign at matched observable phases"
                ),
                aggregation="per_axis_per_phase_then_source_episode_rate",
                threshold_formula=expert_upper,
            ),
            deferred(
                "unexpected_effective_axis_rate",
                direction="upper",
                required_sources=b0_sources,
                definition=(
                    "deadzone-effective axes absent from the expert-compatible "
                    "axis set at the matched phase"
                ),
                aggregation="per_phase_then_source_episode_rate",
                threshold_formula=expert_upper,
            ),
            deferred(
                "deadzone_effective_recall",
                direction="lower",
                required_sources=b0_sources,
                definition=(
                    "expert-required deadzone-effective axis/sign events "
                    "recovered by replay output"
                ),
                aggregation="per_axis_per_phase_then_source_episode_mean",
                threshold_formula=expert_lower,
            ),
        ],
        "G4": [
            deferred(
                "token_swap_action_effect",
                direction="lower",
                required_sources=[
                    *condition_sources,
                ],
                definition=(
                    "L1 effective-action delta after a supported condition "
                    "swap on an otherwise identical observation history"
                ),
                aggregation="per_supported_anchor_then_source_episode_mean",
                threshold_formula=condition_lower,
            ),
            deferred(
                "token_swap_direction_accuracy",
                direction="lower",
                required_sources=condition_sources,
                definition=(
                    "supported token swaps whose swing response direction "
                    "matches the target-sector relation"
                ),
                aggregation="per_supported_anchor_then_source_episode_rate",
                threshold_formula=condition_lower,
            ),
            deferred(
                "token_response_latency_ticks",
                direction="upper",
                required_sources=condition_sources,
                definition=(
                    "20 Hz ticks from token replacement to the first "
                    "repeat-noise-exceeding effective action response"
                ),
                aggregation="per_supported_anchor_then_source_episode_q95",
                threshold_formula=expert_upper,
            ),
            deferred(
                "same_token_repeat_consistency",
                direction="lower",
                required_sources=condition_sources,
                definition=(
                    "one minus normalized action difference across repeated "
                    "replays of the same checkpoint, state, and condition"
                ),
                aggregation="per_anchor_then_source_episode_mean",
                threshold_formula=condition_lower,
            ),
            deferred(
                "current_sector_sensitivity",
                direction="lower",
                required_sources=condition_sources,
                definition=(
                    "supported current-sector swap response with next-sector "
                    "half held fixed"
                ),
                aggregation="per_supported_anchor_then_source_episode_mean",
                threshold_formula=condition_lower,
            ),
            deferred(
                "next_sector_sensitivity",
                direction="lower",
                required_sources=condition_sources,
                definition=(
                    "supported next-sector swap response with current-sector "
                    "half held fixed"
                ),
                aggregation="per_supported_anchor_then_source_episode_mean",
                threshold_formula=condition_lower,
            ),
            deferred(
                "condition_ignored_rate",
                direction="upper",
                required_sources=condition_sources,
                definition=(
                    "supported token swaps whose response does not exceed the "
                    "same-checkpoint repeat-noise floor"
                ),
                aggregation="per_source_episode_rate",
                threshold_formula=condition_upper,
            ),
        ],
        "G5": [
            deferred(
                "two_cycle_phase_coverage",
                direction="lower",
                required_sources=[
                    *expert_sources,
                    "B1_two_cycle_validation_replay",
                ],
                definition=(
                    "required phases recovered across two adjacent complete "
                    "cycles with the shared ready boundary counted once"
                ),
                aggregation="per_adjacent_pair_then_source_episode_mean",
                threshold_formula=expert_lower,
            ),
            deferred(
                "ready_boundary_discontinuity",
                direction="upper",
                required_sources=[
                    *expert_sources,
                    "B1_two_cycle_validation_replay",
                ],
                definition=(
                    "effective-action jump across the shared observable ready "
                    "boundary after temporal aggregation"
                ),
                aggregation="per_adjacent_pair_then_source_episode_q95",
                threshold_formula=expert_upper,
            ),
            deferred(
                "eye_only_retention",
                direction="lower",
                required_sources=robustness_sources,
                definition="G3/G4 metric retention with stick cameras masked",
                aggregation="paired_per_anchor_then_source_episode_mean",
                threshold_formula=retention_lower,
            ),
            deferred(
                "stick_only_retention",
                direction="lower",
                required_sources=robustness_sources,
                definition="G3/G4 metric retention with eye cameras masked",
                aggregation="paired_per_anchor_then_source_episode_mean",
                threshold_formula=retention_lower,
            ),
            deferred(
                "single_camera_dropout_retention",
                direction="lower",
                required_sources=robustness_sources,
                definition="worst-camera G3/G4 retention under one-camera masking",
                aggregation="worst_camera_per_anchor_then_source_episode_mean",
                threshold_formula=retention_lower,
            ),
            deferred(
                "pair_swap_failure_rate",
                direction="upper",
                required_sources=robustness_sources,
                definition="task or condition-response failures after within-pair swap",
                aggregation="per_source_episode_rate",
                threshold_formula=perturbation_upper,
            ),
            deferred(
                "cross_role_swap_failure_rate",
                direction="upper",
                required_sources=robustness_sources,
                definition="task or condition-response failures after eye/stick swap",
                aggregation="per_source_episode_rate",
                threshold_formula=perturbation_upper,
            ),
            deferred(
                "state_hold_direction_flip_rate",
                direction="upper",
                required_sources=robustness_sources,
                definition=(
                    "effective-axis sign flips during recorded-observation "
                    "state-hold replay"
                ),
                aggregation="per_anchor_then_source_episode_rate",
                threshold_formula=perturbation_upper,
            ),
            deferred(
                "delay_event_order_violation_rate",
                direction="upper",
                required_sources=robustness_sources,
                definition=(
                    "event-order violations under frozen 20 Hz skip, delay, "
                    "latest-wins, repeat-action, and timeout semantics"
                ),
                aggregation="per_delay_case_then_source_episode_rate",
                threshold_formula=perturbation_upper,
            ),
        ],
    }
    return {
        "schema_version": GATE_CONTRACT_SCHEMA,
        "contract_status": "method_frozen_values_deferred",
        "evidence_scope": "recorded-observation/offline",
        "split_manifest_sha256": str(split_manifest_sha256),
        "annotation_manifest_sha256": str(annotation_manifest_sha256),
        "held_out_test": {
            "authorized": False,
            "allowed_inputs_for_threshold_generation": ["train", "validation"],
            "forbidden_inputs": ["held_out_test"],
        },
        "bootstrap": {
            "unit": "source_episode",
            "paired": True,
            "replicates": int(bootstrap_replicates),
            "seed": int(bootstrap_seed),
            "confidence_level": 0.95,
            "interval": "percentile_two_sided",
            "resampling": (
                "resample source episode IDs with replacement; keep all "
                "anchors and paired B0/B1/B2 results from each drawn episode"
            ),
        },
        "expert_envelope_quantiles": [0.025, 0.5, 0.975],
        "noise_and_null_quantile": 0.975,
        "missing_transition_policy": (
            "report unsupported_counterfactual, exclude from the metric "
            "denominator, and never impute"
        ),
        "future_input_manifest_schema": {
            "required_fields": [
                "path",
                "schema",
                "sha256",
                "checkpoint_sha256",
                "dataset_manifest_sha256",
                "split_sha256",
                "episode_ids",
                "metric_version",
            ],
            "sha256_pattern": "^[0-9a-f]{64}$",
            "held_out_episode_ids_allowed": False,
        },
        "threshold_families": families,
        "auxiliary_not_gate_metrics": [
            "action_mae",
            "direction_agreement",
            "event_time_difference",
            "active_duration_difference",
            "latency_p50_ms",
            "latency_p95_ms",
            "latency_p99_ms",
            "raw_chunk_max_error",
            "executed_action_max_error",
        ],
        "numeric_threshold_artifact": {
            "status": "not_generated",
            "path": None,
            "sha256": None,
        },
        "training_authorized": False,
        "control_candidate": False,
    }


def validate_gate_contract(contract: Mapping[str, Any]) -> None:
    if contract.get("schema_version") != GATE_CONTRACT_SCHEMA:
        raise ValueError("unexpected gate threshold contract schema")
    if contract["held_out_test"]["authorized"] is not False:
        raise ValueError("M0 gate contract must keep held-out test locked")
    if contract["numeric_threshold_artifact"]["status"] != "not_generated":
        raise ValueError("M0 must not fabricate a numeric threshold artifact")
    if contract["numeric_threshold_artifact"]["sha256"] is not None:
        raise ValueError("deferred numeric threshold artifact cannot have a SHA")
    for family in contract["threshold_families"].values():
        for metric in family:
            if metric["status"] != "deferred" or metric["value"] is not None:
                raise ValueError("M0 model gate values must remain deferred/null")
            if not metric["required_sources"] or not metric["deferred_reason"]:
                raise ValueError("deferred metric lacks an auditable generation contract")
            for field in (
                "definition",
                "aggregation",
                "threshold_formula",
                "comparison_unit",
                "minimum_support",
                "future_sha_binding",
            ):
                if not metric.get(field):
                    raise ValueError(
                        f"deferred metric {metric['metric']} lacks {field}"
                    )
