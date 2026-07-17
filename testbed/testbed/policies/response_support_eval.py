"""Evaluate unsent model commands against a historical response envelope."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from testbed.data.deadzone_intent_labels import AXIS_NAMES
from testbed.data.execution_response_envelope import query_response_envelope

SCHEMA_VERSION = "policy_response_support_eval_v3"

RESPONSE_EVIDENCE_NAMES = {
    "supported": "sufficient_similar_condition_evidence",
    "weak_support": "weak_similar_condition_evidence",
    "out_of_support": "insufficient_similar_condition_data",
}


def evaluate_policy_response_support(
    *,
    model: str,
    events: Sequence[Mapping[str, Any]],
    policy_actions: Mapping[int, np.ndarray],
    positive_threshold: Sequence[float],
    negative_threshold: Sequence[float],
    envelope: Mapping[str, Any],
    training_supported_directions: Sequence[str],
) -> dict[str, Any]:
    """Score episode-action relation and response evidence independently."""

    expected_ids = {int(event["episode_id"]) for event in events}
    if set(policy_actions) != expected_ids:
        raise ValueError("policy action episode IDs must exactly match event IDs")
    positive = np.asarray(positive_threshold, dtype=np.float32)
    negative = np.asarray(negative_threshold, dtype=np.float32)
    if positive.shape != (4,) or negative.shape != (4,):
        raise ValueError("deadzone thresholds must have four entries")
    training_supported = set(training_supported_directions)
    episode_supported: dict[int, set[str]] = {}
    for event in events:
        episode_supported.setdefault(int(event["episode_id"]), set()).update(
            event["single_demo_event_support_directions"]
        )

    event_rows: list[dict[str, Any]] = []
    command_rows: list[dict[str, Any]] = []
    for event in events:
        episode_id = int(event["episode_id"])
        timestep = int(event["onset_step"])
        actions = np.asarray(policy_actions[episode_id], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != 4 or timestep >= len(actions):
            raise ValueError(f"invalid policy action array for episode {episode_id}")
        action = actions[timestep]
        anchor = set(event["anchor_intent"])
        immediate = set(event["immediate_intent_0_1"])
        near_2_5 = set(event["near_intent_2_5"])
        near_6_10 = set(event["near_intent_6_10"])
        local_supported = set(event["single_demo_event_support_directions"])
        current_episode_supported = episode_supported[episode_id]
        predicted: list[str] = []
        response_evidence: list[str] = []
        for axis_index, axis in enumerate(AXIS_NAMES):
            value = float(action[axis_index])
            if value >= positive[axis_index]:
                direction = "pos"
                label = f"{axis}+"
            elif value <= -negative[axis_index]:
                direction = "neg"
                label = f"{axis}-"
            else:
                continue
            predicted.append(label)
            query = query_response_envelope(
                axis=axis,
                direction=direction,
                command=value,
                qpos=float(event["qpos_at_onset"][axis]),
                positive_threshold=positive,
                negative_threshold=negative,
                envelope=envelope,
            )
            raw_response_status = str(query["support_status"])
            evidence = RESPONSE_EVIDENCE_NAMES[raw_response_status]
            response_evidence.append(evidence)
            opposite_label = f"{axis}{'-' if direction == 'pos' else '+'}"
            relation = _single_demo_action_relation(
                label=label,
                anchor=anchor,
                immediate=immediate,
                near_2_5=near_2_5,
                near_6_10=near_6_10,
                local_supported=local_supported,
                current_episode_supported=current_episode_supported,
                training_supported=training_supported,
            )
            command_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "model": model,
                    "event_id": str(event["event_id"]),
                    "episode_id": episode_id,
                    "event_index": int(event["event_index"]),
                    "onset_step": timestep,
                    "axis": axis,
                    "direction": direction,
                    "direction_label": label,
                    "command": value,
                    "qpos": float(event["qpos_at_onset"][axis]),
                    "single_demo_action_relation": relation,
                    "opposite_to_single_demo_anchor": opposite_label in anchor,
                    "historical_response_evidence": evidence,
                    "response_envelope_cell_status": raw_response_status,
                    "response_envelope_train_event_count": int(
                        query["train_event_count"]
                    ),
                    "command_magnitude_ratio": float(query["magnitude_ratio"]),
                    "response_envelope_magnitude_bin": int(query["magnitude_bin"]),
                    "response_envelope_qpos_bin": int(query["qpos_bin"]),
                    "historical_response_probability_by_horizon": query[
                        "predicted_response_probability_by_horizon"
                    ],
                    "response_claim_boundary": (
                        "this command was not sent; insufficient response data is "
                        "not an action-accuracy failure"
                    ),
                }
            )
        predicted_set = set(predicted)
        event_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "model": model,
                "event_id": str(event["event_id"]),
                "episode_id": episode_id,
                "event_index": int(event["event_index"]),
                "onset_step": timestep,
                "predicted_directions": predicted,
                "has_effective_command": bool(predicted),
                "single_demo_exact_anchor": predicted_set == anchor,
                "within_single_demo_event_actions": (
                    bool(predicted_set) and predicted_set <= local_supported
                ),
                "within_single_demo_episode_actions": (
                    bool(predicted_set) and predicted_set <= current_episode_supported
                ),
                "within_training_event_directions": (
                    bool(predicted_set) and predicted_set <= training_supported
                ),
                "outside_single_demo_event_directions": sorted(
                    predicted_set - local_supported
                ),
                "outside_single_demo_episode_directions": sorted(
                    predicted_set - current_episode_supported
                ),
                "training_unseen_directions": sorted(
                    predicted_set - training_supported
                ),
                "historical_response_evidence": response_evidence,
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "model": model,
        "capability_boundaries": {
            "directly_measures": (
                "single-demo action relation and train-derived historical "
                "response evidence as two independent axes"
            ),
            "does_not_measure": [
                "response to the unsent model command",
                "self-generated closed-loop state",
                "task success",
                "task correctness of single-demo differences",
            ],
            "single_demo_relation_is_correctness": False,
            "single_demo_action_policy": (
                "current-frame, near-future, local-event, other-phase episode, "
                "training-only, and training-unseen relations remain distinct"
            ),
            "insufficient_response_policy": (
                "insufficient similar-condition response data is unknown coverage; "
                "it is excluded from model action-accuracy failure counts"
            ),
        },
        "event_count": len(event_rows),
        "episode_count": len(expected_ids),
        "first_event": _aggregate(
            [row for row in event_rows if int(row["event_index"]) == 0],
            [row for row in command_rows if int(row["event_index"]) == 0],
        ),
        "all_events": _aggregate(event_rows, command_rows),
        "event_rows": event_rows,
        "command_rows": command_rows,
    }


def _aggregate(
    event_rows: Sequence[Mapping[str, Any]],
    command_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    probabilities: dict[str, list[float]] = {}
    for row in command_rows:
        for horizon, value in row["historical_response_probability_by_horizon"].items():
            if value is not None:
                probabilities.setdefault(str(horizon), []).append(float(value))
    return {
        "event_count": len(event_rows),
        "events_with_effective_command": int(
            sum(bool(row["has_effective_command"]) for row in event_rows)
        ),
        "single_demo_exact_anchor_events": int(
            sum(bool(row["single_demo_exact_anchor"]) for row in event_rows)
        ),
        "within_single_demo_event_actions": int(
            sum(bool(row["within_single_demo_event_actions"]) for row in event_rows)
        ),
        "within_single_demo_episode_actions": int(
            sum(bool(row["within_single_demo_episode_actions"]) for row in event_rows)
        ),
        "within_training_event_directions": int(
            sum(bool(row["within_training_event_directions"]) for row in event_rows)
        ),
        "command_count": len(command_rows),
        "outside_single_demo_episode_command_count": int(
            sum(
                row["single_demo_action_relation"]
                in {"training_dataset_only_match", "not_observed_in_training_events"}
                for row in command_rows
            )
        ),
        "single_demo_action_relation_counts": dict(
            sorted(
                Counter(
                    str(row["single_demo_action_relation"]) for row in command_rows
                ).items()
            )
        ),
        "historical_response_evidence_counts": dict(
            sorted(
                Counter(
                    str(row["historical_response_evidence"]) for row in command_rows
                ).items()
            )
        ),
        "mean_historical_response_probability_by_horizon": {
            horizon: float(np.mean(values)) for horizon, values in probabilities.items()
        },
        "scoring_rule": {
            "single_demo_relation_axis": "single_demo_action_relation",
            "response_coverage_axis": "historical_response_evidence",
            "insufficient_response_data_counts_as_action_error": False,
            "single_combined_score": False,
        },
    }


def _single_demo_action_relation(
    *,
    label: str,
    anchor: set[str],
    immediate: set[str],
    near_2_5: set[str],
    near_6_10: set[str],
    local_supported: set[str],
    current_episode_supported: set[str],
    training_supported: set[str],
) -> str:
    if label in anchor:
        return "current_frame_match"
    if label in immediate:
        return "immediate_0_1_match"
    if label in near_2_5:
        return "near_2_5_match"
    if label in near_6_10:
        return "near_6_10_match"
    if label in local_supported:
        return "later_within_40t_match"
    if label in current_episode_supported:
        return "other_phase_same_episode_match"
    if label in training_supported:
        return "training_dataset_only_match"
    return "not_observed_in_training_events"
