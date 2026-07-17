"""Compare whole activation sequences with a train-expert reference cohort.

This evaluator deliberately ignores leading recording idle time, exact action
amplitudes, and repeated same-direction deadzone chatter.  It asks whether the
order of semantic per-axis direction changes in a saved trajectory resembles
any task execution demonstrated by the training cohort.  It is an empirical
compatibility diagnostic, not a task-success label.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from testbed.policies.deadzone_eval import AXIS_NAMES, effective_direction_mask

SCHEMA_VERSION = "train_reference_task_sequence_compatibility_v1"
CORE_DIRECTION_MIN_EPISODE_RATE = 0.90

ActivationToken = tuple[str, ...]
ActivationMotif = tuple[ActivationToken, ...]


def evaluate_task_sequence_compatibility(
    *,
    model: str,
    training_expert_actions: Mapping[int, np.ndarray],
    validation_expert_actions: Mapping[int, np.ndarray],
    policy_actions: Mapping[int, np.ndarray],
    thresholds: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    """Evaluate one model against train trajectories and a held-out expert band."""

    if not training_expert_actions:
        raise ValueError("training_expert_actions must not be empty")
    expected_ids = set(validation_expert_actions)
    if not expected_ids or set(policy_actions) != expected_ids:
        raise ValueError(
            "policy action episode IDs must exactly match validation expert IDs"
        )

    train_motifs = {
        int(episode_id): activation_motif(actions, thresholds=thresholds)
        for episode_id, actions in training_expert_actions.items()
    }
    expert_motifs = {
        int(episode_id): activation_motif(actions, thresholds=thresholds)
        for episode_id, actions in validation_expert_actions.items()
    }
    model_motifs = {
        int(episode_id): activation_motif(actions, thresholds=thresholds)
        for episode_id, actions in policy_actions.items()
    }
    if any(not motif for motif in train_motifs.values()):
        empty_ids = sorted(key for key, value in train_motifs.items() if not value)
        raise ValueError(
            f"training reference contains empty activation motifs: {empty_ids}"
        )

    training_reference = _reference_summary(train_motifs)
    core_directions = tuple(training_reference["core_directions"])

    training_bigrams = {
        (motif[index], motif[index + 1])
        for motif in train_motifs.values()
        for index in range(len(motif) - 1)
    }
    expert_scores = _score_motifs(
        expert_motifs,
        train_motifs=train_motifs,
        training_bigrams=training_bigrams,
        core_directions=core_directions,
    )
    model_scores = _score_motifs(
        model_motifs,
        train_motifs=train_motifs,
        training_bigrams=training_bigrams,
        core_directions=core_directions,
    )
    reversed_scores = _score_motifs(
        {
            episode_id: tuple(reversed(motif))
            for episode_id, motif in expert_motifs.items()
        },
        train_motifs=train_motifs,
        training_bigrams=training_bigrams,
        core_directions=core_directions,
    )
    collapsed_scores = _score_motifs(
        {episode_id: motif[:1] for episode_id, motif in expert_motifs.items()},
        train_motifs=train_motifs,
        training_bigrams=training_bigrams,
        core_directions=core_directions,
    )

    expert_by_id = {int(row["episode_id"]): row for row in expert_scores["rows"]}
    rows = []
    for row in model_scores["rows"]:
        expert = expert_by_id[int(row["episode_id"])]
        rows.append(
            {
                **row,
                "validation_expert_nearest_train_similarity": expert[
                    "nearest_train_similarity"
                ],
                "similarity_delta_from_validation_expert": float(
                    row["nearest_train_similarity"] - expert["nearest_train_similarity"]
                ),
                "validation_expert_onset_event_count": expert["onset_event_count"],
            }
        )

    expert_summary = expert_scores["summary"]
    model_summary = model_scores["summary"]
    expert_q25 = float(expert_summary["nearest_train_similarity"]["q25"])
    model_similarities = [
        float(row["nearest_train_similarity"]) for row in model_scores["rows"]
    ]
    cohort_comparison = {
        "model_minus_validation_expert_mean_similarity": float(
            model_summary["nearest_train_similarity"]["mean"]
            - expert_summary["nearest_train_similarity"]["mean"]
        ),
        "model_minus_validation_expert_median_similarity": float(
            model_summary["nearest_train_similarity"]["median"]
            - expert_summary["nearest_train_similarity"]["median"]
        ),
        "model_episode_count_at_or_above_validation_expert_q25": int(
            sum(value >= expert_q25 for value in model_similarities)
        ),
        "model_episode_rate_at_or_above_validation_expert_q25": float(
            np.mean(np.asarray(model_similarities) >= expert_q25)
        ),
        "validation_expert_q25_reference": expert_q25,
        "interpretation": (
            "cohort-level descriptive comparison only; a few episodes below the "
            "reference band are not an automatic task failure"
        ),
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "model": str(model),
        "capability_boundaries": {
            "directly_measures": (
                "order and coverage of deadzone-effective per-axis direction "
                "changes in "
                "saved teacher-forced trajectories relative to training expert "
                "task sequences"
            ),
            "reference_semantics": (
                "set-valued 120-episode training expert cohort, calibrated by "
                "independent validation expert trajectories"
            ),
            "leading_idle_semantics": (
                "ignored; the first effective direction onset is sequence token 0"
            ),
            "amplitude_semantics": "ignored after asymmetric deadzone thresholding",
            "same_direction_reactivation_semantics": (
                "collapsed across idle gaps until that axis changes sign"
            ),
            "core_direction_semantics": (
                "directions present in at least 90% of training expert episodes; "
                "coverage is descriptive and is not a safety gate"
            ),
            "does_not_measure": [
                "self-generated closed-loop observations",
                "physical command response",
                "task completion or safety",
                "terrain-held-out generalization",
                "correctness beyond behavior represented in the training cohort",
            ],
            "decision_policy": (
                "use episode-cohort distributions and controls; do not veto a model "
                "because one held-out demonstration differs"
            ),
        },
        "training_reference": training_reference,
        "validation_expert_calibration": expert_summary,
        "counterfactual_controls": {
            "reversed_validation_expert": reversed_scores["summary"],
            "single_event_collapse": collapsed_scores["summary"],
        },
        "model_summary": model_summary,
        "cohort_comparison": cohort_comparison,
        "rows": rows,
    }


def activation_motif(
    actions: np.ndarray,
    *,
    thresholds: Mapping[str, Mapping[str, float]],
) -> ActivationMotif:
    """Return grouped semantic direction changes, without timing or chatter."""

    array = np.asarray(actions, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != len(AXIS_NAMES):
        raise ValueError(f"actions must have shape (T, {len(AXIS_NAMES)})")
    effective = effective_direction_mask(array, dict(thresholds))
    last_direction: list[int | None] = [None] * len(AXIS_NAMES)
    result: list[ActivationToken] = []
    for timestep in range(effective.shape[0]):
        labels = []
        for axis_index, axis in enumerate(AXIS_NAMES):
            direction = (
                0
                if bool(effective[timestep, axis_index, 0])
                else 1
                if bool(effective[timestep, axis_index, 1])
                else None
            )
            if direction is not None and direction != last_direction[axis_index]:
                labels.append(f"{axis}{'+' if direction == 0 else '-'}")
                last_direction[axis_index] = direction
        if labels:
            result.append(tuple(labels))
    return tuple(result)


def activation_sequence_similarity(
    candidate: Sequence[ActivationToken],
    reference: Sequence[ActivationToken],
) -> float:
    """Generalized edit similarity with Jaccard substitution cost."""

    candidate_tokens = tuple(tuple(token) for token in candidate)
    reference_tokens = tuple(tuple(token) for token in reference)
    candidate_size = len(candidate_tokens)
    reference_size = len(reference_tokens)
    denominator = max(candidate_size, reference_size, 1)
    distance = np.zeros((candidate_size + 1, reference_size + 1), dtype=np.float64)
    distance[:, 0] = np.arange(candidate_size + 1, dtype=np.float64)
    distance[0, :] = np.arange(reference_size + 1, dtype=np.float64)
    for candidate_index in range(1, candidate_size + 1):
        for reference_index in range(1, reference_size + 1):
            candidate_set = set(candidate_tokens[candidate_index - 1])
            reference_set = set(reference_tokens[reference_index - 1])
            union = candidate_set | reference_set
            substitution_cost = (
                0.0
                if not union
                else 1.0 - (len(candidate_set & reference_set) / len(union))
            )
            distance[candidate_index, reference_index] = min(
                distance[candidate_index - 1, reference_index] + 1.0,
                distance[candidate_index, reference_index - 1] + 1.0,
                distance[candidate_index - 1, reference_index - 1] + substitution_cost,
            )
    return float(1.0 - distance[candidate_size, reference_size] / denominator)


def _score_motifs(
    motifs: Mapping[int, ActivationMotif],
    *,
    train_motifs: Mapping[int, ActivationMotif],
    training_bigrams: set[tuple[ActivationToken, ActivationToken]],
    core_directions: Sequence[str],
) -> dict[str, Any]:
    rows = []
    for episode_id, motif in sorted(motifs.items()):
        nearest_episode_id = None
        nearest_similarity = -1.0
        for train_episode_id, train_motif in train_motifs.items():
            similarity = activation_sequence_similarity(motif, train_motif)
            if similarity > nearest_similarity:
                nearest_episode_id = int(train_episode_id)
                nearest_similarity = similarity
        bigrams = tuple(zip(motif, motif[1:]))
        observed_directions = {direction for token in motif for direction in token}
        missing_core_directions = sorted(set(core_directions) - observed_directions)
        rows.append(
            {
                "episode_id": int(episode_id),
                "onset_event_count": len(motif),
                "direction_onset_count": sum(len(token) for token in motif),
                "activation_motif": [list(token) for token in motif],
                "nearest_train_episode_id": nearest_episode_id,
                "nearest_train_similarity": float(nearest_similarity),
                "exact_training_bigram_count": int(
                    sum(bigram in training_bigrams for bigram in bigrams)
                ),
                "bigram_count": len(bigrams),
                "exact_training_bigram_support_rate": (
                    float(np.mean([bigram in training_bigrams for bigram in bigrams]))
                    if bigrams
                    else None
                ),
                "core_direction_coverage_rate": (
                    float(
                        (len(core_directions) - len(missing_core_directions))
                        / len(core_directions)
                    )
                    if core_directions
                    else None
                ),
                "missing_core_directions": missing_core_directions,
            }
        )
    return {
        "summary": _aggregate_rows(rows, motifs=motifs, train_motifs=train_motifs),
        "rows": rows,
    }


def _aggregate_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    motifs: Mapping[int, ActivationMotif],
    train_motifs: Mapping[int, ActivationMotif],
) -> dict[str, Any]:
    similarities = [float(row["nearest_train_similarity"]) for row in rows]
    event_counts = [int(row["onset_event_count"]) for row in rows]
    bigram_rates = [
        float(row["exact_training_bigram_support_rate"])
        for row in rows
        if row["exact_training_bigram_support_rate"] is not None
    ]
    core_coverage_rates = [
        float(row["core_direction_coverage_rate"])
        for row in rows
        if row["core_direction_coverage_rate"] is not None
    ]
    full_core_coverage_count = int(sum(value == 1.0 for value in core_coverage_rates))
    return {
        "episode_count": len(rows),
        "nearest_train_similarity": _distribution(similarities),
        "onset_event_count": _distribution(event_counts),
        "exact_training_bigram_support_rate": _distribution(bigram_rates),
        "core_direction_coverage_rate": _distribution(core_coverage_rates),
        "full_core_direction_episode_count": full_core_coverage_count,
        "full_core_direction_episode_rate": (
            float(full_core_coverage_count / len(core_coverage_rates))
            if core_coverage_rates
            else None
        ),
        "direction_onset_counts": _direction_counts(motifs),
        "direction_onset_episode_rates": _direction_episode_rates(motifs),
        "direction_histogram_similarity_to_training": _histogram_similarity(
            motifs, train_motifs
        ),
    }


def _reference_summary(motifs: Mapping[int, ActivationMotif]) -> dict[str, Any]:
    counts = [len(motif) for motif in motifs.values()]
    direction_episode_rates = _direction_episode_rates(motifs)
    return {
        "episode_count": len(motifs),
        "onset_event_count": _distribution(counts),
        "direction_onset_counts": _direction_counts(motifs),
        "direction_onset_episode_rates": direction_episode_rates,
        "core_direction_min_episode_rate": CORE_DIRECTION_MIN_EPISODE_RATE,
        "core_directions": [
            direction
            for direction in _direction_labels()
            if direction_episode_rates[direction] >= CORE_DIRECTION_MIN_EPISODE_RATE
        ],
    }


def _direction_counts(motifs: Mapping[int, ActivationMotif]) -> dict[str, int]:
    counts = Counter(
        direction for motif in motifs.values() for token in motif for direction in token
    )
    return {
        direction: int(counts.get(direction, 0)) for direction in _direction_labels()
    }


def _direction_episode_rates(
    motifs: Mapping[int, ActivationMotif],
) -> dict[str, float]:
    episode_count = len(motifs)
    return {
        direction: float(
            np.mean(
                [
                    any(direction in token for token in motif)
                    for motif in motifs.values()
                ]
            )
        )
        if episode_count
        else 0.0
        for direction in _direction_labels()
    }


def _histogram_similarity(
    motifs: Mapping[int, ActivationMotif],
    reference_motifs: Mapping[int, ActivationMotif],
) -> float:
    labels = _direction_labels()
    first = np.asarray(
        [_direction_counts(motifs)[label] for label in labels], dtype=np.float64
    )
    second = np.asarray(
        [_direction_counts(reference_motifs)[label] for label in labels],
        dtype=np.float64,
    )
    if first.sum() == 0.0 or second.sum() == 0.0:
        return 0.0
    first /= first.sum()
    second /= second.sum()
    midpoint = 0.5 * (first + second)

    def divergence(distribution: np.ndarray) -> float:
        present = distribution > 0.0
        return float(
            np.sum(
                distribution[present]
                * np.log(distribution[present] / midpoint[present])
            )
        )

    jensen_shannon = 0.5 * (divergence(first) + divergence(second))
    return float(1.0 - np.sqrt(jensen_shannon / np.log(2.0)))


def _distribution(values: Sequence[float | int]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "min": None,
            "q10": None,
            "q25": None,
            "median": None,
            "q75": None,
            "q90": None,
            "max": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "min": float(np.min(array)),
        "q10": float(np.quantile(array, 0.10)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "q75": float(np.quantile(array, 0.75)),
        "q90": float(np.quantile(array, 0.90)),
        "max": float(np.max(array)),
    }


def _direction_labels() -> tuple[str, ...]:
    return tuple(f"{axis}{sign}" for axis in AXIS_NAMES for sign in ("+", "-"))


__all__ = [
    "SCHEMA_VERSION",
    "activation_motif",
    "activation_sequence_similarity",
    "evaluate_task_sequence_compatibility",
]
