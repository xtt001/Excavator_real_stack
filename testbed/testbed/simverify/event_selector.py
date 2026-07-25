"""Observable-only interval event selection for SimVerify M0.

Numeric qpos/qvel/action signals own the candidate event type and half-open
interval.  Frozen eye/stick ResNet features only confirm support and select a
representative row.  This module deliberately has no HDF5, model, filesystem,
held-out-test, or privileged-state dependency so the complete selector can be
refit inside a source-episode bootstrap.
"""

from __future__ import annotations

import copy
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from testbed.simverify.annotations import (
    EpisodeSignals,
    fit_sector_thresholds,
    unit_normalize,
)

EVENT_NAMES = (
    "ready_start",
    "dig_entry_proxy",
    "carry_transition_proxy",
    "dump_start_proxy",
    "dump_end_proxy",
    "ready_end",
)
EVENT_PHASE = {
    "ready_start": "ready",
    "ready_end": "ready",
    "dig_entry_proxy": "dig_entry_proxy",
    "carry_transition_proxy": "carry_transition_proxy",
    "dump_start_proxy": "dump_start_proxy",
    "dump_end_proxy": "dump_end_proxy",
}
EVENT_PHASES = (
    "ready",
    "dig_entry_proxy",
    "carry_transition_proxy",
    "dump_start_proxy",
    "dump_end_proxy",
)
EVENT_REQUIRED_ROLES = {
    "ready": ("stick",),
    "dig_entry_proxy": ("eye", "stick"),
    "carry_transition_proxy": ("stick",),
    "dump_start_proxy": ("eye", "stick"),
    "dump_end_proxy": ("eye", "stick"),
}
EVENT_CHANGE_KIND = {
    "ready": "stable_two_sided",
    "dig_entry_proxy": "entering",
    "carry_transition_proxy": "centered",
    "dump_start_proxy": "entering",
    "dump_end_proxy": "exiting",
}
FEATURE_ROLES = ("eye", "stick")
ROLE_CLASSIFICATION_PHASES = {
    role: tuple(phase for phase in EVENT_PHASES if role in EVENT_REQUIRED_ROLES[phase])
    for role in FEATURE_ROLES
}
SUPPORT_QUANTILE = 0.01
TRANSITION_CHANGE_QUANTILE = 0.01
READY_MOTION_QUANTILE = 0.99
OFFSET_LOW_QUANTILE = 0.025
OFFSET_HIGH_QUANTILE = 0.975
MAXIMUM_BOOTSTRAP_FAILURE_RATE = 0.01
MAXIMUM_PARAMETER_CI_FRACTION = 0.25


def event_rows(
    cycles: Mapping[int, Sequence[Mapping[str, Any]]],
    *,
    episode_ids: Sequence[int],
) -> list[dict[str, Any]]:
    """Return immutable, deduplicated numeric event references.

    A ready gap shared by ``ready_end(i)`` and ``ready_start(i+1)`` has one
    event key and therefore contributes one calibration/selection row.
    """

    by_key: dict[str, dict[str, Any]] = {}
    for episode_id in map(int, episode_ids):
        for cycle in cycles[episode_id]:
            cycle_id = int(cycle["cycle_id"])
            for event_name in EVENT_NAMES:
                event = cycle["observable_events"].get(event_name)
                if event is None:
                    continue
                interval = tuple(map(int, event["interval"]))
                numeric_step = int(
                    event.get(
                        "numeric_representative_step",
                        event["representative_step"],
                    )
                )
                phase = EVENT_PHASE[event_name]
                if phase == "ready":
                    key = (
                        f"episode_{episode_id}:ready:"
                        f"{interval[0]}:{interval[1]}:{numeric_step}"
                    )
                else:
                    key = f"episode_{episode_id}:cycle_{cycle_id}:{event_name}"
                if key not in by_key:
                    by_key[key] = {
                        "event_key": key,
                        "episode_id": episode_id,
                        "phase": phase,
                        "interval": list(interval),
                        "numeric_representative_step": numeric_step,
                        "references": [],
                    }
                row = by_key[key]
                if (
                    row["phase"] != phase
                    or row["interval"] != list(interval)
                    or int(row["numeric_representative_step"]) != numeric_step
                ):
                    raise ValueError(f"inconsistent shared event identity: {key}")
                reference = {
                    "cycle_id": cycle_id,
                    "event_name": event_name,
                }
                if reference not in row["references"]:
                    row["references"].append(reference)
    return [by_key[key] for key in sorted(by_key)]


def fit_event_selector(
    cycles: Mapping[int, Sequence[Mapping[str, Any]]],
    features: Mapping[tuple[int, int], Mapping[str, np.ndarray]],
    *,
    train_draw: Sequence[int],
    validation_draw: Sequence[int],
) -> dict[str, Any]:
    """Fit one event selector from train/validation episode block draws.

    The chronology is fixed:

    1. train numeric representatives fit episode-balanced role prototypes;
    2. validation numeric representatives fit own-support/change envelopes;
    3. interval matching without an offset bound fits signed offset bounds;
    4. those complete frozen bounds are applied to validation intervals.
    """

    train_ids = list(map(int, train_draw))
    validation_ids = list(map(int, validation_draw))
    if not train_ids or not validation_ids:
        raise ValueError("event selector requires train and validation episodes")
    available = set(map(int, cycles))
    unknown = (set(train_ids) | set(validation_ids)) - available
    if unknown:
        raise ValueError(
            f"event selector received unavailable episodes: {sorted(unknown)}"
        )
    train_rows = event_rows(cycles, episode_ids=sorted(set(train_ids)))
    validation_rows = event_rows(
        cycles,
        episode_ids=sorted(set(validation_ids)),
    )
    train_by_episode = _rows_by_episode(train_rows)
    validation_by_episode = _rows_by_episode(validation_rows)
    prototypes, prototype_counts = _fit_role_prototypes(
        train_by_episode,
        features,
        draw=train_ids,
    )
    anchor = _calibrate_anchor_envelopes(
        validation_by_episode,
        features,
        prototypes,
        draw=validation_ids,
    )
    unbounded = _select_rows_for_draw(
        validation_by_episode,
        features,
        prototypes=prototypes,
        support_thresholds=anchor["support_thresholds"],
        change_thresholds=anchor["change_thresholds"],
        offset_bounds=None,
        draw=validation_ids,
    )
    offset_bounds, offset_samples = _fit_signed_offset_bounds(
        unbounded,
        validation_by_episode,
        draw=validation_ids,
    )
    bounded = _select_rows_for_draw(
        validation_by_episode,
        features,
        prototypes=prototypes,
        support_thresholds=anchor["support_thresholds"],
        change_thresholds=anchor["change_thresholds"],
        offset_bounds=offset_bounds,
        draw=validation_ids,
    )
    validation = _selection_coverage(
        bounded,
        validation_by_episode,
        draw=validation_ids,
    )
    return {
        "schema": "observable_event_interval_selector_v2",
        "method": (
            "numeric_candidate_plus_independent_eye_stick_"
            "own_support_and_event_specific_change"
        ),
        "fit_split": "train",
        "calibration_split": "validation",
        "candidate_interval": "numeric_half_open_interval",
        "feature_halo_rows": 1,
        "halo_may_be_representative": False,
        "top1_policy": "diagnostic_only",
        "prototype_weighting": ("equal_source_episode_then_equal_bootstrap_draw_slot"),
        "support_quantile": SUPPORT_QUANTILE,
        "transition_change_quantile": TRANSITION_CHANGE_QUANTILE,
        "ready_motion_quantile": READY_MOTION_QUANTILE,
        "signed_offset_quantiles": [
            OFFSET_LOW_QUANTILE,
            OFFSET_HIGH_QUANTILE,
        ],
        "required_roles": {
            phase: list(EVENT_REQUIRED_ROLES[phase]) for phase in EVENT_PHASES
        },
        "change_rules": dict(EVENT_CHANGE_KIND),
        "selection_rules": {
            "ready": "minimum_required_role_two_sided_motion",
            "dig_entry_proxy": "earliest_supported_local_entering_peak",
            "carry_transition_proxy": "maximum_supported_centered_change",
            "dump_start_proxy": "earliest_supported_local_entering_peak",
            "dump_end_proxy": "latest_supported_local_exiting_peak",
            "tie_break": (
                "stronger_minimum_own_support_then_smaller_numeric_offset_"
                "then_earlier_source_step"
            ),
            "offset_filter_applied_before_ranking": True,
        },
        "prototypes": prototypes,
        "prototype_counts": prototype_counts,
        "support_thresholds": anchor["support_thresholds"],
        "change_thresholds": anchor["change_thresholds"],
        "offset_bounds": offset_bounds,
        "validation": {
            "classification": anchor["classification"],
            "coverage": validation,
            "unbounded_coverage": _selection_coverage(
                unbounded,
                validation_by_episode,
                draw=validation_ids,
            ),
            "interval_length_steps": _interval_length_summary(
                validation_by_episode,
                draw=validation_ids,
            ),
        },
        "_calibration_samples": {
            "support": anchor["support_samples"],
            "change": anchor["change_samples"],
            "offset": offset_samples,
        },
    }


def fit_event_null_control(
    selector: Mapping[str, Any],
    cycles: Mapping[int, Sequence[Mapping[str, Any]]],
    features: Mapping[tuple[int, int], Mapping[str, np.ndarray]],
    *,
    validation_ids: Sequence[int],
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    """Episode-mapping permutation null for diagnostics and support coverage."""

    ids = list(map(int, validation_ids))
    rows = event_rows(cycles, episode_ids=ids)
    by_episode = _rows_by_episode(rows)
    rng = np.random.default_rng(int(seed))
    accuracies: dict[str, list[float]] = defaultdict(list)
    balanced_accuracies: dict[str, list[float]] = defaultdict(list)
    coverages: dict[str, list[float]] = defaultdict(list)
    for _ in range(int(replicates)):
        coverage_mappings = {
            episode_id: dict(
                zip(
                    EVENT_PHASES,
                    _non_identity_permutation(EVENT_PHASES, rng),
                )
            )
            for episode_id in ids
        }
        accuracy_mappings = {
            role: {
                episode_id: dict(
                    zip(
                        ROLE_CLASSIFICATION_PHASES[role],
                        _non_identity_permutation(
                            ROLE_CLASSIFICATION_PHASES[role],
                            rng,
                        ),
                    )
                )
                for episode_id in ids
            }
            for role in FEATURE_ROLES
        }
        role_correct: dict[str, list[bool]] = defaultdict(list)
        role_outcomes: dict[str, list[tuple[str, bool]]] = defaultdict(list)
        phase_confirmed: dict[str, list[bool]] = defaultdict(list)
        for episode_id in ids:
            for row in by_episode[episode_id]:
                expected_phase = str(row["phase"])
                for role in FEATURE_ROLES:
                    if expected_phase not in ROLE_CLASSIFICATION_PHASES[role]:
                        continue
                    feature = _feature(
                        features,
                        episode_id,
                        int(row["numeric_representative_step"]),
                        role,
                    )
                    if feature is None:
                        continue
                    scores = _prototype_scores(
                        feature,
                        selector["prototypes"],
                        role=role,
                    )
                    prediction = max(scores, key=scores.get)
                    correct = (
                        prediction
                        == accuracy_mappings[role][episode_id][expected_phase]
                    )
                    role_correct[role].append(correct)
                    role_outcomes[role].append((expected_phase, correct))
                selection = match_event_interval(
                    row,
                    features,
                    selector=selector,
                    support_phase=coverage_mappings[episode_id][expected_phase],
                )
                phase_confirmed[expected_phase].append(
                    selection["status"] == "confirmed"
                )
        for role in FEATURE_ROLES:
            values = role_correct.get(role, [])
            accuracies[role].append(float(np.mean(values)) if values else 0.0)
            outcomes = role_outcomes.get(role, [])
            labels = sorted({label for label, _correct in outcomes})
            balanced_accuracies[role].append(
                float(
                    np.mean(
                        [
                            np.mean(
                                [
                                    correct
                                    for observed_label, correct in outcomes
                                    if observed_label == label
                                ]
                            )
                            for label in labels
                        ]
                    )
                )
                if labels
                else 0.0
            )
        for phase in EVENT_PHASES:
            values = phase_confirmed.get(phase, [])
            coverages[phase].append(float(np.mean(values)) if values else 0.0)
    return {
        "unit": "source_episode_event_mapping",
        "seed": int(seed),
        "replicates": int(replicates),
        "accuracy_p95": {
            role: float(np.quantile(values, 0.95))
            for role, values in sorted(accuracies.items())
        },
        "balanced_accuracy_p95": {
            role: float(np.quantile(values, 0.95))
            for role, values in sorted(balanced_accuracies.items())
        },
        "coverage_p95": {
            phase: float(np.quantile(values, 0.95))
            for phase, values in sorted(coverages.items())
        },
    }


def match_event_interval(
    row: Mapping[str, Any],
    features: Mapping[tuple[int, int], Mapping[str, np.ndarray]],
    *,
    selector: Mapping[str, Any],
    support_phase: str | None = None,
) -> dict[str, Any]:
    """Apply one frozen event-specific interval selector without mutation."""

    episode_id = int(row["episode_id"])
    phase = str(row["phase"])
    prototype_phase = phase if support_phase is None else str(support_phase)
    numeric_step = int(row["numeric_representative_step"])
    start, end = map(int, row["interval"])
    bounds = selector.get("offset_bounds")
    bound = None if bounds is None else bounds[phase]
    candidates: list[dict[str, Any]] = []
    for step in range(start, end):
        signed_offset = int(step - numeric_step)
        offset_allowed = bool(
            bound is None
            or (
                int(bound["minimum_signed_offset_steps"])
                <= signed_offset
                <= int(bound["maximum_signed_offset_steps"])
            )
        )
        role_metrics: dict[str, Any] = {}
        missing_roles: list[str] = []
        for role in FEATURE_ROLES:
            feature = _feature(features, episode_id, step, role)
            if feature is None:
                missing_roles.append(role)
                continue
            scores = _prototype_scores(
                feature,
                selector["prototypes"],
                role=role,
            )
            ranked = sorted(
                scores.items(),
                key=lambda item: (-item[1], item[0]),
            )
            similarity = float(scores[prototype_phase])
            role_metrics[role] = {
                "expected_prototype": prototype_phase,
                "expected_similarity": similarity,
                "prediction": ranked[0][0],
                "margin": float(similarity - ranked[1][1]),
                "scores": scores,
            }
        required_roles = EVENT_REQUIRED_ROLES[phase]
        support_pass = True
        support_excess: list[float] = []
        change_pass = True
        changes: dict[str, float | None] = {}
        for role in required_roles:
            metric = role_metrics.get(role)
            if metric is None:
                support_pass = False
                change_pass = False
                changes[role] = None
                continue
            threshold = float(selector["support_thresholds"][prototype_phase][role])
            excess = float(metric["expected_similarity"]) - threshold
            support_excess.append(excess)
            if excess < 0.0:
                support_pass = False
            change = _event_change(
                features,
                episode_id=episode_id,
                step=step,
                role=role,
                kind=EVENT_CHANGE_KIND[phase],
            )
            changes[role] = change
            if change is None:
                change_pass = False
                continue
            change_threshold = float(selector["change_thresholds"][phase][role])
            if phase == "ready":
                change_pass = change_pass and change <= change_threshold
            else:
                change_pass = change_pass and change >= change_threshold
        aggregate_change = (
            None
            if any(changes.get(role) is None for role in required_roles)
            else (
                max(float(changes[role]) for role in required_roles)
                if phase == "ready"
                else min(float(changes[role]) for role in required_roles)
            )
        )
        eligible = bool(
            offset_allowed
            and support_pass
            and change_pass
            and aggregate_change is not None
        )
        candidates.append(
            {
                "step": int(step),
                "signed_offset_steps": signed_offset,
                "absolute_offset_steps": abs(signed_offset),
                "offset_allowed": offset_allowed,
                "required_support_pass": support_pass,
                "required_change_pass": change_pass,
                "eligible": eligible,
                "minimum_support_excess": (
                    min(support_excess) if support_excess else None
                ),
                "aggregate_change": aggregate_change,
                "role_metrics": role_metrics,
                "role_change": changes,
                "missing_roles": missing_roles,
            }
        )
    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    selected, peak_diagnostic = _select_candidate(phase, candidates, eligible)
    reasons: list[str] = []
    if not candidates:
        reasons.append("numeric_interval_has_no_feature_candidates")
    if candidates and not any(
        not candidate["missing_roles"] for candidate in candidates
    ):
        reasons.append("required_feature_missing")
    if candidates and not any(
        candidate["required_support_pass"] for candidate in candidates
    ):
        reasons.append("own_prototype_support_below_frozen_envelope")
    if candidates and not any(
        candidate["required_change_pass"] for candidate in candidates
    ):
        reasons.append(
            "ready_stability_outside_frozen_envelope"
            if phase == "ready"
            else "event_change_below_frozen_envelope"
        )
    if candidates and not any(candidate["offset_allowed"] for candidate in candidates):
        reasons.append("no_candidate_inside_frozen_signed_offset")
    if selected is None and not reasons:
        reasons.append("no_jointly_eligible_visual_candidate")
    if peak_diagnostic["equal_separated_peak_ambiguity"]:
        selected = None
        reasons.append("separated_equal_visual_peaks")
    return {
        "event_key": str(row["event_key"]),
        "phase": phase,
        "status": "confirmed" if selected is not None else "ambiguous",
        "reason_codes": sorted(set(reasons)),
        "interval": [start, end],
        "numeric_representative_step": numeric_step,
        "representative_step": (None if selected is None else int(selected["step"])),
        "signed_offset_steps": (
            None if selected is None else int(selected["signed_offset_steps"])
        ),
        "absolute_offset_steps": (
            None if selected is None else int(selected["absolute_offset_steps"])
        ),
        "signed_offset_bounds": (None if bound is None else copy.deepcopy(dict(bound))),
        "candidate_count": len(candidates),
        "eligible_candidate_count": len(eligible),
        "selected": selected,
        "peak_diagnostic": peak_diagnostic,
        "acceptance_rule": (
            "numeric_event_type_unchanged;required_role_own_prototype_support;"
            "event_specific_change_or_stability;signed_offset_filter_before_rank;"
            "top1_and_margin_diagnostic_only"
        ),
    }


def select_event_corpus(
    selector: Mapping[str, Any],
    cycles: Mapping[int, Sequence[Mapping[str, Any]]],
    features: Mapping[tuple[int, int], Mapping[str, np.ndarray]],
    *,
    episode_ids: Sequence[int],
) -> dict[str, Any]:
    """Select every unique event and validate shared boundaries/event order."""

    ids = list(map(int, episode_ids))
    rows = event_rows(cycles, episode_ids=ids)
    selections = {
        str(row["event_key"]): match_event_interval(
            row,
            features,
            selector=selector,
        )
        for row in rows
    }
    reference_to_key: dict[tuple[int, int, str], str] = {}
    for row in rows:
        for reference in row["references"]:
            reference_to_key[
                (
                    int(row["episode_id"]),
                    int(reference["cycle_id"]),
                    str(reference["event_name"]),
                )
            ] = str(row["event_key"])
    cycle_results: dict[str, dict[str, Any]] = {}
    for episode_id in ids:
        for cycle in cycles[episode_id]:
            cycle_id = int(cycle["cycle_id"])
            result_key = f"episode_{episode_id}:cycle_{cycle_id}"
            event_keys: dict[str, str | None] = {}
            steps: dict[str, int | None] = {}
            reasons: list[str] = []
            for event_name in EVENT_NAMES:
                key = reference_to_key.get((episode_id, cycle_id, event_name))
                event_keys[event_name] = key
                if key is None:
                    steps[event_name] = None
                    reasons.append(f"{event_name}_not_identifiable")
                    continue
                selection = selections[key]
                step = selection["representative_step"]
                steps[event_name] = None if step is None else int(step)
                if selection["status"] != "confirmed":
                    reasons.append(f"{event_name}_visual_interval_not_confirmed")
            order_values = [steps[event_name] for event_name in EVENT_NAMES]
            order_complete = all(step is not None for step in order_values)
            order_valid = bool(
                order_complete
                and all(
                    int(order_values[index]) <= int(order_values[index + 1])
                    for index in range(len(order_values) - 1)
                )
            )
            if order_complete and not order_valid:
                reasons.append("observable_event_order_invalid_after_visual_selection")
            current_sector_order_valid = _current_sector_order_valid(steps)
            if steps["dig_entry_proxy"] is not None and not current_sector_order_valid:
                reasons.append(
                    "current_sector_event_order_invalid_after_visual_selection"
                )
            source_steps = (
                [int(steps["ready_start"]), int(steps["ready_end"])]
                if order_valid
                else None
            )
            cycle_results[result_key] = {
                "episode_id": episode_id,
                "cycle_id": cycle_id,
                "event_keys": event_keys,
                "event_steps": steps,
                "all_required_events_confirmed": order_complete,
                "event_order_valid": order_valid,
                "current_sector_order_valid": current_sector_order_valid,
                "confirmed_source_steps": source_steps,
                "reason_codes": sorted(set(reasons)),
            }
    _require_shared_ready_invariant(cycle_results)
    return {
        "schema": "observable_event_selections_v2",
        "events": selections,
        "cycles": cycle_results,
        "summary": _selection_summary(selections, cycle_results),
    }


def selected_sector_records(
    selection_result: Mapping[str, Any],
    cycles: Mapping[int, Sequence[Mapping[str, Any]]],
    signals: Mapping[int, EpisodeSignals],
    *,
    episode_draw: Sequence[int],
) -> list[dict[str, Any]]:
    """Build sector-fit rows from visually selected, order-valid dig entries."""

    records: list[dict[str, Any]] = []
    for episode_id in map(int, episode_draw):
        for cycle in cycles[episode_id]:
            cycle_id = int(cycle["cycle_id"])
            cycle_result = selection_result["cycles"][
                f"episode_{episode_id}:cycle_{cycle_id}"
            ]
            dig_key = cycle_result["event_keys"]["dig_entry_proxy"]
            dig_selection = (
                None if dig_key is None else selection_result["events"][dig_key]
            )
            valid = bool(
                cycle_result["current_sector_order_valid"]
                and dig_selection is not None
                and dig_selection["status"] == "confirmed"
            )
            swing_qpos = (
                None
                if not valid
                else float(
                    signals[episode_id].qpos[
                        int(dig_selection["representative_step"]),
                        0,
                    ]
                )
            )
            records.append(
                {
                    "numeric_sector_evidence": {
                        "current_swing_qpos": swing_qpos,
                    },
                    "sector_validity": {
                        "current": {
                            "valid": valid,
                        }
                    },
                }
            )
    return records


def _sector_candidates(
    selection_result: Mapping[str, Any],
    cycles: Mapping[int, Sequence[Mapping[str, Any]]],
    signals: Mapping[int, EpisodeSignals],
    *,
    episode_draw: Sequence[int],
) -> list[dict[str, Any]]:
    """Retain enough replicate state to apply the later point-stability mask."""

    candidates: list[dict[str, Any]] = []
    for episode_id in map(int, episode_draw):
        for cycle in cycles[episode_id]:
            cycle_result = selection_result["cycles"][
                f"episode_{episode_id}:cycle_{int(cycle['cycle_id'])}"
            ]
            dig_key = cycle_result["event_keys"]["dig_entry_proxy"]
            dig_selection = (
                None if dig_key is None else selection_result["events"][dig_key]
            )
            dig_step = (
                None
                if dig_selection is None or dig_selection["status"] != "confirmed"
                else int(dig_selection["representative_step"])
            )
            candidates.append(
                {
                    "event_keys": [
                        cycle_result["event_keys"][event_name]
                        for event_name in EVENT_NAMES
                    ],
                    "event_steps": [
                        cycle_result["event_steps"][event_name]
                        for event_name in EVENT_NAMES
                    ],
                    "swing_qpos": (
                        None
                        if dig_step is None
                        else float(signals[episode_id].qpos[dig_step, 0])
                    ),
                }
            )
    return candidates


def _sector_samples_from_candidates(
    candidates: Sequence[Mapping[str, Any]],
    *,
    stability_assessment: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Collect order-valid dig rows, optionally after masking every event."""

    samples: list[dict[str, Any]] = []
    assessments = (
        None if stability_assessment is None else stability_assessment.get("events", {})
    )
    for candidate in candidates:
        keys = list(candidate["event_keys"])
        values = list(candidate["event_steps"])
        if len(keys) != len(EVENT_NAMES) or len(values) != len(EVENT_NAMES):
            raise ValueError("sector candidate event vectors have invalid length")
        if assessments is not None:
            values = [
                (
                    step
                    if key is not None
                    and bool(assessments.get(str(key), {}).get("passed", False))
                    else None
                )
                for key, step in zip(keys, values, strict=True)
            ]
        steps = dict(zip(EVENT_NAMES, values, strict=True))
        dig_key = keys[EVENT_NAMES.index("dig_entry_proxy")]
        swing_qpos = candidate.get("swing_qpos")
        if (
            dig_key is None
            or swing_qpos is None
            or not _current_sector_order_valid(steps)
        ):
            continue
        samples.append(
            {
                "event_key": str(dig_key),
                "swing_qpos": float(swing_qpos),
            }
        )
    return samples


def bootstrap_event_selected_sector(
    cycles: Mapping[int, Sequence[Mapping[str, Any]]],
    signals: Mapping[int, EpisodeSignals],
    features: Mapping[tuple[int, int], Mapping[str, np.ndarray]],
    *,
    train_ids: Sequence[int],
    validation_ids: Sequence[int],
    point_selector: Mapping[str, Any],
    point_selections: Mapping[str, Any],
    samples: int,
    seed: int,
) -> dict[str, Any]:
    """Full selector-to-sector source-episode outer bootstrap.

    Each replicate refits role prototypes and all validation envelopes,
    reselects shared event boundaries, revalidates order, and only then fits
    sector clusters.  No already-selected representative is reused.
    """

    train = list(map(int, train_ids))
    validation = list(map(int, validation_ids))
    allowed = set(train) | set(validation)
    if set(map(int, cycles)) != allowed or set(map(int, signals)) != allowed:
        raise ValueError(
            "outer bootstrap inputs must contain calibration episodes only"
        )
    rng = np.random.default_rng(int(seed) + 31)
    centers: list[list[float]] = []
    boundaries: list[list[float]] = []
    coverage_values: dict[str, list[float]] = defaultdict(list)
    accuracy_values: dict[str, list[float]] = defaultdict(list)
    balanced_accuracy_values: dict[str, list[float]] = defaultdict(list)
    offset_low_values: dict[str, list[float]] = defaultdict(list)
    offset_high_values: dict[str, list[float]] = defaultdict(list)
    selected_cycle_counts: list[float] = []
    selector_failure_reasons: Counter[str] = Counter()
    sector_failure_reasons: Counter[str] = Counter()
    sector_candidates_by_replicate: list[list[dict[str, Any]]] = []
    stability: dict[str, dict[str, Any]] = {
        key: {
            "attempted_successful_replicates": 0,
            "confirmed_replicates": 0,
            "selected_steps": [],
        }
        for key in point_selections["events"]
    }
    calibration_ids = sorted(allowed)
    for _ in range(int(samples)):
        train_draw = rng.choice(train, size=len(train), replace=True).tolist()
        validation_draw = rng.choice(
            validation,
            size=len(validation),
            replace=True,
        ).tolist()
        try:
            fitted = fit_event_selector(
                cycles,
                features,
                train_draw=train_draw,
                validation_draw=validation_draw,
            )
            selections = select_event_corpus(
                fitted,
                cycles,
                features,
                episode_ids=calibration_ids,
            )
        except (KeyError, RuntimeError, ValueError) as exc:
            selector_failure_reasons[f"{type(exc).__name__}:{exc}"] += 1
            continue
        for phase, row in fitted["validation"]["coverage"].items():
            coverage_values[phase].append(float(row["confirmed_fraction"]))
        for role, row in fitted["validation"]["classification"].items():
            accuracy_values[role].append(float(row["accuracy"]))
            balanced_accuracy_values[role].append(
                float(row["balanced_accuracy"])
            )
        for phase, row in fitted["offset_bounds"].items():
            offset_low_values[phase].append(float(row["minimum_signed_offset_steps"]))
            offset_high_values[phase].append(float(row["maximum_signed_offset_steps"]))
        for key, stability_row in stability.items():
            stability_row["attempted_successful_replicates"] += 1
            selection = selections["events"].get(key)
            if selection is None or selection["status"] != "confirmed":
                continue
            stability_row["confirmed_replicates"] += 1
            stability_row["selected_steps"].append(
                int(selection["representative_step"])
            )
        sector_candidates = _sector_candidates(
            selections,
            cycles,
            signals,
            episode_draw=train_draw,
        )
        sector_candidates_by_replicate.append(sector_candidates)
        sector_samples = _sector_samples_from_candidates(sector_candidates)
        try:
            sector_records = [
                {
                    "numeric_sector_evidence": {
                        "current_swing_qpos": row["swing_qpos"],
                    },
                    "sector_validity": {"current": {"valid": True}},
                }
                for row in sector_samples
            ]
            sector = fit_sector_thresholds(sector_records)
        except (KeyError, RuntimeError, ValueError) as exc:
            sector_failure_reasons[f"{type(exc).__name__}:{exc}"] += 1
            continue
        centers.append(list(sector["cluster_centers_low_to_high"]))
        boundaries.append(list(sector["boundaries_low_to_high"]))
        selected_cycle_counts.append(
            float(
                sum(
                    row["sector_validity"]["current"]["valid"] for row in sector_records
                )
            )
        )
    selector_successful = int(samples) - sum(selector_failure_reasons.values())
    sector_successful = len(boundaries)
    if selector_successful == 0:
        return {
            "schema": "observable_event_selected_sector_outer_bootstrap_v1",
            "unit": "source_episode",
            "seed": int(seed) + 31,
            "requested_samples": int(samples),
            "successful_samples": 0,
            "failed_samples": int(samples),
            "failure_reasons": dict(selector_failure_reasons),
            "event_selector": {},
            "selection_stability": {},
            "sector": None,
        }
    point_bounds = point_selector["offset_bounds"]
    stability_public: dict[str, Any] = {}
    for key, row in stability.items():
        steps = np.asarray(row.pop("selected_steps"), dtype=np.float64)
        point = point_selections["events"][key]
        point_step = point["representative_step"]
        phase = str(point["phase"])
        tolerance = max(
            abs(int(point_bounds[phase]["minimum_signed_offset_steps"])),
            abs(int(point_bounds[phase]["maximum_signed_offset_steps"])),
        )
        attempted = int(row["attempted_successful_replicates"])
        confirmed = int(row["confirmed_replicates"])
        within = (
            0
            if point_step is None
            else int(np.sum(np.abs(steps - int(point_step)) <= tolerance))
        )
        stability_public[key] = {
            **row,
            "reselection_within_point_tolerance_replicates": within,
            "confirmation_frequency": confirmed / attempted,
            "reselection_within_point_tolerance_frequency": within / attempted,
            "selected_step": (None if steps.size == 0 else _summary(steps.tolist())),
        }
    return {
        "schema": "observable_event_selected_sector_outer_bootstrap_v1",
        "unit": "source_episode",
        "seed": int(seed) + 31,
        "requested_samples": int(samples),
        "successful_samples": selector_successful,
        "failed_samples": int(samples) - selector_successful,
        "failure_reasons": dict(sorted(selector_failure_reasons.items())),
        "event_selector": {
            "validation_accuracy": {
                role: _summary(values)
                for role, values in sorted(accuracy_values.items())
            },
            "validation_balanced_accuracy": {
                role: _summary(values)
                for role, values in sorted(balanced_accuracy_values.items())
            },
            "validation_coverage": {
                phase: _summary(values)
                for phase, values in sorted(coverage_values.items())
            },
            "offset_bounds": {
                phase: {
                    "minimum_signed_offset_steps": _summary(offset_low_values[phase]),
                    "maximum_signed_offset_steps": _summary(offset_high_values[phase]),
                }
                for phase in EVENT_PHASES
            },
        },
        "selection_stability": stability_public,
        "_sector_candidates_by_selector_successful_replicate": (
            sector_candidates_by_replicate
        ),
        "sector": (
            None
            if sector_successful == 0
            else {
                "unit": "source_episode_full_selector_refit",
                "seed": int(seed) + 31,
                "requested_samples": int(samples),
                "successful_samples": sector_successful,
                "failed_samples": int(samples) - sector_successful,
                "failure_reasons": dict(sorted(sector_failure_reasons.items())),
                "selected_train_cycle_count": _summary(selected_cycle_counts),
                "cluster_centers": _vector_summary(centers),
                "boundaries": _vector_summary(boundaries),
            }
        ),
    }


def refit_outer_sector_with_stability_mask(
    outer_bootstrap: dict[str, Any],
    stability_assessment: Mapping[str, Any],
    *,
    mask_name: str = "point_stability",
) -> None:
    """Refit outer sector distributions using the frozen point-stability mask."""

    replicate_candidates = outer_bootstrap.pop(
        "_sector_candidates_by_selector_successful_replicate",
        [],
    )
    requested = int(outer_bootstrap["requested_samples"])
    selector_failures = int(outer_bootstrap["failed_samples"])
    centers: list[list[float]] = []
    boundaries: list[list[float]] = []
    selected_counts: list[float] = []
    failures: Counter[str] = Counter()
    for candidates in replicate_candidates:
        filtered = _sector_samples_from_candidates(
            candidates,
            stability_assessment=stability_assessment,
        )
        records = [
            {
                "numeric_sector_evidence": {
                    "current_swing_qpos": row["swing_qpos"],
                },
                "sector_validity": {"current": {"valid": True}},
            }
            for row in filtered
        ]
        try:
            sector = fit_sector_thresholds(records)
        except (KeyError, RuntimeError, ValueError) as exc:
            failures[f"{type(exc).__name__}:{exc}"] += 1
            continue
        centers.append(list(sector["cluster_centers_low_to_high"]))
        boundaries.append(list(sector["boundaries_low_to_high"]))
        selected_counts.append(float(len(filtered)))
    successful = len(boundaries)
    total_failed = requested - successful
    outer_bootstrap["sector_pre_point_stability_mask"] = outer_bootstrap.get("sector")
    outer_bootstrap["sector"] = (
        None
        if successful == 0
        else {
            "unit": (
                "source_episode_full_selector_refit_with_frozen_"
                f"{mask_name}_mask"
            ),
            "seed": int(outer_bootstrap["seed"]),
            "requested_samples": requested,
            "successful_samples": successful,
            "failed_samples": total_failed,
            "selector_failed_samples": selector_failures,
            "stability_mask_or_sector_fit_failure_reasons": dict(
                sorted(failures.items())
            ),
            "selected_train_cycle_count": _summary(selected_counts),
            "cluster_centers": _vector_summary(centers),
            "boundaries": _vector_summary(boundaries),
        }
    )


def event_selector_gate_report(
    selector: Mapping[str, Any],
    null_control: Mapping[str, Any],
    outer_bootstrap: Mapping[str, Any],
    point_selections: Mapping[str, Any],
    stability_assessment: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate predeclared event-selector identifiability/stability gates."""

    failures: list[str] = []
    requested = int(outer_bootstrap["requested_samples"])
    failed = int(outer_bootstrap["failed_samples"])
    failure_rate = 1.0 if requested <= 0 else failed / requested
    if requested <= 0 or failure_rate > MAXIMUM_BOOTSTRAP_FAILURE_RATE:
        failures.append("event_selector_outer_bootstrap_failure_rate")
    bootstrap_selector = outer_bootstrap.get("event_selector", {})
    for phase in EVENT_PHASES:
        summary = bootstrap_selector.get("validation_coverage", {}).get(phase)
        if summary is None:
            failures.append(f"{phase}_selector_coverage_bootstrap_missing")
        elif float(summary["p02_5"]) <= float(null_control["coverage_p95"][phase]):
            failures.append(f"{phase}_selector_coverage_not_above_null")
        interval_median = float(
            selector["validation"]["interval_length_steps"][phase]["p50"]
        )
        denominator = max(interval_median, 1.0)
        offset = bootstrap_selector.get("offset_bounds", {}).get(phase)
        if offset is None:
            failures.append(f"{phase}_offset_bootstrap_missing")
            continue
        for side in (
            "minimum_signed_offset_steps",
            "maximum_signed_offset_steps",
        ):
            width = float(offset[side]["p97_5"]) - float(offset[side]["p02_5"])
            if width >= MAXIMUM_PARAMETER_CI_FRACTION * denominator:
                failures.append(f"{phase}_{side}_bootstrap_unstable")
        stable_fraction = float(
            stability_assessment["summary"]["by_phase"][phase]["stable_fraction"]
        )
        if summary is not None and stable_fraction < float(summary["p02_5"]):
            failures.append(f"{phase}_stable_point_coverage_below_bootstrap")
    return {
        "schema": "observable_event_selector_gate_report_v1",
        "stage": "M0",
        "evidence_scope": "recorded-observation/offline",
        "passed": not failures,
        "failure_reasons": sorted(set(failures)),
        "m1_import_smoke_authorized": False,
        "training_authorized": False,
        "criteria": {
            "bootstrap_unit": "source_episode",
            "maximum_failed_sample_rate": MAXIMUM_BOOTSTRAP_FAILURE_RATE,
            "cross_event_top1_accuracy": "diagnostic_only_not_a_gate",
            "event_coverage_lower_bound_above_permutation_null_p95": True,
            "minimum_point_confirmation_frequency": (
                1.0 - MAXIMUM_BOOTSTRAP_FAILURE_RATE
            ),
            "point_selected_step_ci95_width_within_signed_offset_span": True,
            "stable_point_coverage_not_below_validation_bootstrap_p02_5": True,
            "maximum_offset_endpoint_ci95_width_fraction_of_validation_"
            "median_interval": MAXIMUM_PARAMETER_CI_FRACTION,
            "posthoc_threshold_change_allowed": False,
        },
        "null_control": copy.deepcopy(null_control),
        "point_selection_stability": copy.deepcopy(stability_assessment["summary"]),
        "point_selection_count": len(point_selections["events"]),
        "outer_bootstrap": {
            "requested_samples": requested,
            "successful_samples": int(outer_bootstrap["successful_samples"]),
            "failed_samples": failed,
            "failure_reasons": copy.deepcopy(outer_bootstrap["failure_reasons"]),
            "event_selector": copy.deepcopy(outer_bootstrap.get("event_selector", {})),
        },
    }


def assess_interval_confirmation_stability(
    point_selections: Mapping[str, Any],
    outer_bootstrap: Mapping[str, Any],
    *,
    minimum_confirmation_frequency: float,
) -> dict[str, Any]:
    """Freeze an interval-confirmation mask without gating exact visual points.

    Numeric signals continue to own the candidate interval, event type, and
    representative source row.  Visual selection confirms that an eligible
    row exists inside the interval.  The exact visual argmin/argmax row and its
    reselection spread remain diagnostics because a broad observable envelope
    may contain several equally valid rows.
    """

    threshold = float(minimum_confirmation_frequency)
    if not 0.0 < threshold <= 1.0:
        raise ValueError("minimum_confirmation_frequency must be in (0, 1]")
    outer_rows = outer_bootstrap.get("selection_stability", {})
    assessments: dict[str, Any] = {}
    total: Counter[str] = Counter()
    confirmed: Counter[str] = Counter()
    retained: Counter[str] = Counter()
    reasons: dict[str, Counter[str]] = defaultdict(Counter)
    for key, selection in point_selections["events"].items():
        phase = str(selection["phase"])
        total[phase] += 1
        row_reasons: list[str] = []
        outer = outer_rows.get(key)
        if selection["status"] != "confirmed":
            row_reasons.append("point_selection_not_confirmed")
        elif outer is None:
            confirmed[phase] += 1
            row_reasons.append("outer_interval_confirmation_missing")
        else:
            confirmed[phase] += 1
            frequency = float(outer["confirmation_frequency"])
            if frequency < threshold:
                row_reasons.append(
                    "interval_confirmation_frequency_below_calibrated_threshold"
                )
        passed = not row_reasons
        if passed:
            retained[phase] += 1
        else:
            reasons[phase].update(row_reasons)
        assessments[key] = {
            "passed": passed,
            "reason_codes": sorted(set(row_reasons)),
            "minimum_interval_confirmation_frequency": threshold,
            "outer": copy.deepcopy(outer),
            "exact_visual_point_reselection": (
                "diagnostic_only_numeric_anchor_owns_representative"
            ),
        }
    return {
        "schema": "observable_event_interval_confirmation_stability_v2",
        "events": assessments,
        "summary": {
            "minimum_interval_confirmation_frequency": threshold,
            "representative_ownership": "numeric_observable_anchor",
            "exact_visual_point_reselection": "diagnostic_only",
            "by_phase": {
                phase: {
                    "event_count": int(total[phase]),
                    "point_confirmed_count": int(confirmed[phase]),
                    "retained_count": int(retained[phase]),
                    "retained_fraction": (
                        float(retained[phase] / total[phase])
                        if total[phase]
                        else 0.0
                    ),
                    "reason_counts": dict(sorted(reasons[phase].items())),
                }
                for phase in EVENT_PHASES
            },
        },
    }


def event_selector_gate_report_v2(
    null_control: Mapping[str, Any],
    outer_bootstrap: Mapping[str, Any],
    interval_stability: Mapping[str, Any],
    reliability_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate the audited M0 visual-event Gate.

    The Gate tests only properties that map directly to the annotation
    contract: every requested source-episode refit must be computable, the
    frozen visual roles must identify their numeric-anchor phases above an
    episode-mapping null, and the most conservative interval-confirmation
    threshold that preserves the declared train/validation transition support
    must retain majority bootstrap support.  Exact visual point localization,
    offset-endpoint width, and own-support coverage under a wrong-prototype
    permutation remain diagnostics rather than promotion operands.
    """

    failures: list[str] = []
    requested = int(outer_bootstrap["requested_samples"])
    failed = int(outer_bootstrap["failed_samples"])
    if requested <= 0 or failed != 0:
        failures.append("event_selector_outer_bootstrap_not_fully_computable")

    selector_summary = outer_bootstrap.get("event_selector", {})
    observed_balanced = selector_summary.get(
        "validation_balanced_accuracy",
        {},
    )
    null_balanced = null_control.get("balanced_accuracy_p95", {})
    identifiability: dict[str, Any] = {}
    for role in FEATURE_ROLES:
        observed = observed_balanced.get(role)
        null = null_balanced.get(role)
        if observed is None or null is None:
            failures.append(f"{role}_balanced_accuracy_null_operand_missing")
            identifiability[role] = None
            continue
        lower = float(observed["p02_5"])
        null_p95 = float(null)
        passed = lower > null_p95
        if not passed:
            failures.append(
                f"{role}_balanced_accuracy_lower_bound_not_above_null"
            )
        identifiability[role] = {
            "source_episode_bootstrap_p02_5": lower,
            "episode_mapping_permutation_null_p95": null_p95,
            "advantage": lower - null_p95,
            "passed": passed,
        }

    reliability_passed = bool(reliability_contract.get("passed", False))
    if not reliability_passed:
        failures.extend(
            str(reason)
            for reason in reliability_contract.get(
                "failure_reasons",
                ["interval_confirmation_reliability_contract_failed"],
            )
        )

    return {
        "schema": "observable_event_selector_gate_report_v2",
        "stage": "M0",
        "evidence_scope": "recorded-observation/offline",
        "passed": not failures,
        "failure_reasons": sorted(set(failures)),
        "m1_import_smoke_authorized": False,
        "training_authorized": False,
        "criteria": {
            "bootstrap_unit": "source_episode",
            "all_requested_refits_must_be_computable": True,
            "role_balanced_accuracy_lower_bound_above_episode_mapping_null_p95": (
                True
            ),
            "interval_confirmation_threshold": (
                "maximum_cycle_minimum_frequency_preserving_all_3x3_"
                "transitions_in_train_and_validation"
            ),
            "minimum_reliability_wilson_lower_bound": (
                "strictly_above_majority_0.5"
            ),
            "representative_ownership": "numeric_observable_anchor",
            "posthoc_heldout_threshold_change_allowed": False,
        },
        "visual_role_identifiability": identifiability,
        "interval_confirmation_reliability": copy.deepcopy(
            reliability_contract
        ),
        "interval_stability_summary": copy.deepcopy(
            interval_stability["summary"]
        ),
        "diagnostic_only_not_gate": {
            "wrong_prototype_own_support_coverage": copy.deepcopy(
                null_control.get("coverage_p95", {})
            ),
            "visual_argmin_argmax_point_reselection": True,
            "signed_offset_endpoint_ci_width": copy.deepcopy(
                selector_summary.get("offset_bounds", {})
            ),
            "v1_stable_point_fraction": (
                "retired_population_average_vs_per_item_extreme_threshold_"
                "comparison"
            ),
        },
        "outer_bootstrap": {
            "requested_samples": requested,
            "successful_samples": int(outer_bootstrap["successful_samples"]),
            "failed_samples": failed,
            "failure_reasons": copy.deepcopy(
                outer_bootstrap.get("failure_reasons", {})
            ),
        },
    }


def assess_point_selection_stability(
    selector: Mapping[str, Any],
    point_selections: Mapping[str, Any],
    null_control: Mapping[str, Any],
    outer_bootstrap: Mapping[str, Any],
) -> dict[str, Any]:
    """Assess each point representative against outer-bootstrap reselection."""

    assessments: dict[str, Any] = {}
    total: Counter[str] = Counter()
    stable: Counter[str] = Counter()
    point_confirmed: Counter[str] = Counter()
    stable_confirmed: Counter[str] = Counter()
    reasons: dict[str, Counter[str]] = defaultdict(Counter)
    outer_rows = outer_bootstrap.get("selection_stability", {})
    for key, selection in point_selections["events"].items():
        phase = str(selection["phase"])
        total[phase] += 1
        row_reasons: list[str] = []
        outer = outer_rows.get(key)
        if selection["status"] != "confirmed":
            row_reasons.append("point_selection_not_confirmed")
        elif outer is None:
            point_confirmed[phase] += 1
            row_reasons.append("outer_reselection_missing")
        else:
            point_confirmed[phase] += 1
            frequency = float(outer["confirmation_frequency"])
            minimum_frequency = 1.0 - MAXIMUM_BOOTSTRAP_FAILURE_RATE
            if frequency < minimum_frequency:
                row_reasons.append("confirmation_failure_rate_above_maximum")
            within_frequency = float(
                outer["reselection_within_point_tolerance_frequency"]
            )
            if within_frequency < minimum_frequency:
                row_reasons.append(
                    "within_tolerance_reselection_failure_rate_above_maximum"
                )
            selected_step = outer.get("selected_step")
            if selected_step is None:
                row_reasons.append("outer_selected_step_distribution_missing")
                step_width = None
            else:
                step_width = float(selected_step["p97_5"]) - float(
                    selected_step["p02_5"]
                )
                bounds = selector["offset_bounds"][phase]
                tolerance_span = float(
                    int(bounds["maximum_signed_offset_steps"])
                    - int(bounds["minimum_signed_offset_steps"])
                )
                if step_width > tolerance_span:
                    row_reasons.append(
                        "selected_step_ci95_wider_than_signed_offset_span"
                    )
        passed = not row_reasons
        if passed:
            stable[phase] += 1
            if selection["status"] == "confirmed":
                stable_confirmed[phase] += 1
        else:
            reasons[phase].update(row_reasons)
        assessments[key] = {
            "passed": passed,
            "reason_codes": sorted(set(row_reasons)),
            "phase_null_p95_coverage": float(null_control["coverage_p95"][phase]),
            "minimum_confirmation_frequency": (1.0 - MAXIMUM_BOOTSTRAP_FAILURE_RATE),
            "outer": copy.deepcopy(outer),
        }
    return {
        "schema": "observable_event_point_reselection_stability_v1",
        "events": assessments,
        "summary": {
            "by_phase": {
                phase: {
                    "point_event_count": int(total[phase]),
                    "stable_count": int(stable[phase]),
                    "point_confirmed_count": int(point_confirmed[phase]),
                    "stable_point_confirmed_count": int(stable_confirmed[phase]),
                    "all_point_confirmed_stable": bool(
                        stable_confirmed[phase] == point_confirmed[phase]
                    ),
                    "stable_fraction": (
                        float(stable[phase] / total[phase]) if total[phase] else 0.0
                    ),
                    "reason_counts": dict(sorted(reasons[phase].items())),
                }
                for phase in EVENT_PHASES
            }
        },
    }


def apply_event_selections(
    cycles: Mapping[int, Sequence[dict[str, Any]]],
    selection_result: dict[str, Any],
    *,
    stability: Mapping[str, Any],
    stability_assessment: Mapping[str, Any],
    selector: Mapping[str, Any],
    selector_sha256: str,
    episode_ids: Sequence[int],
    representative_ownership: str = "visual_selected_point",
) -> None:
    """Attach point selections/confidence and rebuild confirmed cycle ranges."""

    if representative_ownership not in (
        "visual_selected_point",
        "numeric_observable_anchor",
    ):
        raise ValueError(
            f"unsupported representative ownership: {representative_ownership!r}"
        )
    rows = event_rows(cycles, episode_ids=episode_ids)
    reference_to_row: dict[tuple[int, int, str], Mapping[str, Any]] = {}
    for row in rows:
        for reference in row["references"]:
            reference_to_row[
                (
                    int(row["episode_id"]),
                    int(reference["cycle_id"]),
                    str(reference["event_name"]),
                )
            ] = row
    samples = selector["_calibration_samples"]
    for key, selection in selection_result["events"].items():
        assessment = copy.deepcopy(stability_assessment["events"][key])
        selection["bootstrap_stability"] = assessment
        if selection["status"] == "confirmed" and not assessment["passed"]:
            selection["status"] = "ambiguous"
            selection["reason_codes"] = sorted(
                set(selection["reason_codes"] + assessment["reason_codes"])
            )
            selection["representative_step"] = None
            selection["signed_offset_steps"] = None
            selection["absolute_offset_steps"] = None
            selection["selected"] = None
        if selection["status"] == "confirmed":
            confidence = _selection_confidence(
                selection,
                selector=selector,
                calibration_samples=samples,
                stability=stability[key],
            )
            if confidence["joint"] <= 0.0:
                selection["status"] = "ambiguous"
                selection["reason_codes"] = sorted(
                    set(
                        selection["reason_codes"] + ["nonpositive_empirical_confidence"]
                    )
                )
                selection["representative_step"] = None
                selection["signed_offset_steps"] = None
                selection["absolute_offset_steps"] = None
                selection["selected"] = None
        else:
            confidence = {
                "kind": "empirical_support_score_not_probability",
                "joint": 0.0,
            }
        visual_representative = selection.get("representative_step")
        selection["representative_ownership"] = representative_ownership
        selection["visual_selected_representative_step"] = (
            None
            if visual_representative is None
            else int(visual_representative)
        )
        if (
            selection["status"] == "confirmed"
            and representative_ownership == "numeric_observable_anchor"
        ):
            selection["representative_step"] = int(
                selection["numeric_representative_step"]
            )
            confidence["bootstrap_interval_confirmation_frequency"] = float(
                stability[key]["confirmation_frequency"]
            )
            confidence["exact_visual_point_reselection"] = "diagnostic_only"
        selection["confidence"] = confidence
        selection["selector_sha256"] = str(selector_sha256)
    for cycle_result in selection_result["cycles"].values():
        steps: dict[str, int | None] = {}
        reasons = list(cycle_result["reason_codes"])
        for event_name in EVENT_NAMES:
            key = cycle_result["event_keys"][event_name]
            selection = None if key is None else selection_result["events"][key]
            step = (
                None
                if selection is None or selection["status"] != "confirmed"
                else int(selection["representative_step"])
            )
            steps[event_name] = step
            if key is not None and step is None:
                reasons.append(f"{event_name}_visual_interval_not_confirmed")
        order_values = [steps[event_name] for event_name in EVENT_NAMES]
        order_complete = all(step is not None for step in order_values)
        order_valid = bool(
            order_complete
            and all(
                int(order_values[index]) <= int(order_values[index + 1])
                for index in range(len(order_values) - 1)
            )
        )
        if order_complete and not order_valid:
            reasons.append("observable_event_order_invalid_after_visual_selection")
        current_sector_order_valid = _current_sector_order_valid(steps)
        if steps["dig_entry_proxy"] is not None and not current_sector_order_valid:
            reasons.append("current_sector_event_order_invalid_after_visual_selection")
        cycle_result["event_steps"] = steps
        cycle_result["all_required_events_confirmed"] = order_complete
        cycle_result["event_order_valid"] = order_valid
        cycle_result["current_sector_order_valid"] = current_sector_order_valid
        cycle_result["confirmed_source_steps"] = (
            [int(steps["ready_start"]), int(steps["ready_end"])]
            if order_valid
            else None
        )
        cycle_result["reason_codes"] = sorted(set(reasons))
    _require_shared_ready_invariant(selection_result["cycles"])
    for episode_id in map(int, episode_ids):
        for cycle in cycles[episode_id]:
            cycle_id = int(cycle["cycle_id"])
            cycle_result = selection_result["cycles"][
                f"episode_{episode_id}:cycle_{cycle_id}"
            ]
            cycle.setdefault(
                "numeric_source_steps",
                list(map(int, cycle["source_steps"])),
            )
            event_confidences: list[float] = []
            for event_name, event in cycle["observable_events"].items():
                if event is None:
                    continue
                row = reference_to_row[(episode_id, cycle_id, event_name)]
                key = str(row["event_key"])
                selection = copy.deepcopy(selection_result["events"][key])
                event.setdefault(
                    "numeric_representative_step",
                    int(event["representative_step"]),
                )
                event["ready_boundary_id"] = (
                    key if EVENT_PHASE[event_name] == "ready" else None
                )
                if selection["status"] == "confirmed":
                    event["representative_step"] = int(selection["representative_step"])
                    event_confidences.append(float(selection["confidence"]["joint"]))
                event["visual_interval_selection"] = selection
            if cycle_result["event_order_valid"]:
                cycle["source_steps"] = list(
                    map(int, cycle_result["confirmed_source_steps"])
                )
            reasons = list(cycle_result["reason_codes"])
            if reasons:
                cycle["quality"]["status"] = "ambiguous"
                cycle["quality"]["review_required"] = True
                cycle["quality"]["reason_codes"] = sorted(
                    set(cycle["quality"]["reason_codes"] + reasons)
                )
            cycle["quality"]["event_visual_confidence"] = (
                min(event_confidences) if event_confidences else 0.0
            )
            cycle["verification"]["visual_event_order_valid"] = bool(
                cycle_result["event_order_valid"]
            )
            cycle["verification"]["visual_current_sector_order_valid"] = bool(
                cycle_result["current_sector_order_valid"]
            )
    selection_result["summary"] = _selection_summary(
        selection_result["events"],
        selection_result["cycles"],
    )


def public_selector(selector: Mapping[str, Any]) -> dict[str, Any]:
    """Return the JSON-safe selector contract without calibration arrays."""

    result = copy.deepcopy(dict(selector))
    result.pop("_calibration_samples", None)
    result["prototypes"] = {
        phase: {
            role: {
                "npz_key": f"event_{role}_{phase}",
                "dimension": int(np.asarray(vector).size),
            }
            for role, vector in role_rows.items()
        }
        for phase, role_rows in result["prototypes"].items()
    }
    return result


def prototype_arrays(selector: Mapping[str, Any]) -> dict[str, np.ndarray]:
    return {
        f"event_{role}_{phase}": np.asarray(vector, dtype=np.float32)
        for phase, role_rows in selector["prototypes"].items()
        for role, vector in role_rows.items()
    }


def _rows_by_episode(
    rows: Sequence[Mapping[str, Any]],
) -> dict[int, list[Mapping[str, Any]]]:
    result: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        result[int(row["episode_id"])].append(row)
    return result


def _fit_role_prototypes(
    rows_by_episode: Mapping[int, Sequence[Mapping[str, Any]]],
    features: Mapping[tuple[int, int], Mapping[str, np.ndarray]],
    *,
    draw: Sequence[int],
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    episode_centroids: dict[tuple[int, str, str], np.ndarray] = {}
    raw_counts: Counter[tuple[str, str]] = Counter()
    episode_counts: Counter[tuple[str, str]] = Counter()
    for episode_id in sorted(set(map(int, draw))):
        grouped: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
        for row in rows_by_episode[episode_id]:
            phase = str(row["phase"])
            step = int(row["numeric_representative_step"])
            for role in FEATURE_ROLES:
                feature = _feature(features, episode_id, step, role)
                if feature is not None:
                    grouped[(phase, role)].append(feature)
        for (phase, role), values in grouped.items():
            episode_centroids[(episode_id, phase, role)] = unit_normalize(
                np.mean(np.stack(values, axis=0), axis=0).reshape(1, -1)
            )[0].astype(np.float32)
            raw_counts[(phase, role)] += len(values)
            episode_counts[(phase, role)] += 1
    prototypes: dict[str, dict[str, np.ndarray]] = {}
    counts: dict[str, Any] = {}
    for phase in EVENT_PHASES:
        prototypes[phase] = {}
        counts[phase] = {}
        for role in FEATURE_ROLES:
            vectors = [
                episode_centroids[(int(episode_id), phase, role)]
                for episode_id in map(int, draw)
                if (int(episode_id), phase, role) in episode_centroids
            ]
            if not vectors:
                raise ValueError(f"no train prototype rows for {phase}/{role}")
            prototypes[phase][role] = unit_normalize(
                np.mean(np.stack(vectors, axis=0), axis=0).reshape(1, -1)
            )[0].astype(np.float32)
            counts[phase][role] = {
                "raw_unique_row_count": int(raw_counts[(phase, role)]),
                "unique_episode_count": int(episode_counts[(phase, role)]),
                "bootstrap_draw_slot_count": len(vectors),
            }
    return prototypes, counts


def _calibrate_anchor_envelopes(
    rows_by_episode: Mapping[int, Sequence[Mapping[str, Any]]],
    features: Mapping[tuple[int, int], Mapping[str, np.ndarray]],
    prototypes: Mapping[str, Mapping[str, np.ndarray]],
    *,
    draw: Sequence[int],
) -> dict[str, Any]:
    support: dict[str, dict[str, list[float]]] = {
        phase: {role: [] for role in FEATURE_ROLES} for phase in EVENT_PHASES
    }
    changes: dict[str, dict[str, list[float]]] = {
        phase: {role: [] for role in EVENT_REQUIRED_ROLES[phase]}
        for phase in EVENT_PHASES
    }
    classification_rows: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for episode_id in map(int, draw):
        for row in rows_by_episode[episode_id]:
            phase = str(row["phase"])
            step = int(row["numeric_representative_step"])
            for role in FEATURE_ROLES:
                feature = _feature(features, episode_id, step, role)
                if feature is None:
                    continue
                scores = _prototype_scores(feature, prototypes, role=role)
                support[phase][role].append(float(scores[phase]))
                if phase in ROLE_CLASSIFICATION_PHASES[role]:
                    classification_rows[role].append(
                        (phase, max(scores, key=scores.get))
                    )
            for role in EVENT_REQUIRED_ROLES[phase]:
                change = _event_change(
                    features,
                    episode_id=episode_id,
                    step=step,
                    role=role,
                    kind=EVENT_CHANGE_KIND[phase],
                )
                if change is not None:
                    changes[phase][role].append(float(change))
    support_thresholds: dict[str, dict[str, float]] = {}
    change_thresholds: dict[str, dict[str, float]] = {}
    for phase in EVENT_PHASES:
        support_thresholds[phase] = {}
        for role in FEATURE_ROLES:
            values = support[phase][role]
            if not values:
                raise ValueError(f"no validation support samples for {phase}/{role}")
            support_thresholds[phase][role] = float(
                np.quantile(values, SUPPORT_QUANTILE)
            )
        change_thresholds[phase] = {}
        for role in EVENT_REQUIRED_ROLES[phase]:
            values = changes[phase][role]
            if not values:
                raise ValueError(f"no validation change samples for {phase}/{role}")
            quantile = (
                READY_MOTION_QUANTILE
                if phase == "ready"
                else TRANSITION_CHANGE_QUANTILE
            )
            change_thresholds[phase][role] = float(np.quantile(values, quantile))
    classification: dict[str, Any] = {}
    for role in FEATURE_ROLES:
        rows = classification_rows[role]
        if not rows:
            raise ValueError(f"no validation classification rows for {role}")
        labels = sorted(set(label for label, _prediction in rows))
        classification[role] = {
            "count": len(rows),
            "accuracy": float(
                np.mean([label == prediction for label, prediction in rows])
            ),
            "balanced_accuracy": float(
                np.mean(
                    [
                        np.mean(
                            [
                                prediction == label
                                for true_label, prediction in rows
                                if true_label == label
                            ]
                        )
                        for label in labels
                    ]
                )
            ),
        }
    return {
        "support_thresholds": support_thresholds,
        "change_thresholds": change_thresholds,
        "support_samples": support,
        "change_samples": changes,
        "classification": classification,
    }


def _select_rows_for_draw(
    rows_by_episode: Mapping[int, Sequence[Mapping[str, Any]]],
    features: Mapping[tuple[int, int], Mapping[str, np.ndarray]],
    *,
    prototypes: Mapping[str, Mapping[str, np.ndarray]],
    support_thresholds: Mapping[str, Mapping[str, float]],
    change_thresholds: Mapping[str, Mapping[str, float]],
    offset_bounds: Mapping[str, Any] | None,
    draw: Sequence[int],
) -> list[dict[str, Any]]:
    selector = {
        "prototypes": prototypes,
        "support_thresholds": support_thresholds,
        "change_thresholds": change_thresholds,
        "offset_bounds": offset_bounds,
    }
    return [
        {
            "episode_id": int(episode_id),
            "phase": str(row["phase"]),
            "interval_length_steps": int(row["interval"][1]) - int(row["interval"][0]),
            "selection": match_event_interval(
                row,
                features,
                selector=selector,
            ),
        }
        for episode_id in map(int, draw)
        for row in rows_by_episode[episode_id]
    ]


def _fit_signed_offset_bounds(
    selections: Sequence[Mapping[str, Any]],
    rows_by_episode: Mapping[int, Sequence[Mapping[str, Any]]],
    *,
    draw: Sequence[int],
) -> tuple[dict[str, Any], dict[str, list[int]]]:
    del rows_by_episode, draw
    offsets: dict[str, list[int]] = defaultdict(list)
    for row in selections:
        selection = row["selection"]
        if selection["status"] != "confirmed":
            continue
        offsets[str(row["phase"])].append(
            int(selection["selected"]["signed_offset_steps"])
        )
    result: dict[str, Any] = {}
    for phase in EVENT_PHASES:
        values = offsets.get(phase, [])
        if not values:
            raise ValueError(f"no validation interval match for {phase}")
        array = np.asarray(values, dtype=np.float64)
        result[phase] = {
            "minimum_signed_offset_steps": int(
                np.floor(np.quantile(array, OFFSET_LOW_QUANTILE))
            ),
            "maximum_signed_offset_steps": int(
                np.ceil(np.quantile(array, OFFSET_HIGH_QUANTILE))
            ),
            "fit_match_count": len(values),
            "observed_signed_offset_steps": _summary(values),
        }
    return result, {phase: list(offsets[phase]) for phase in EVENT_PHASES}


def _selection_coverage(
    selections: Sequence[Mapping[str, Any]],
    rows_by_episode: Mapping[int, Sequence[Mapping[str, Any]]],
    *,
    draw: Sequence[int],
) -> dict[str, Any]:
    del rows_by_episode, draw
    total: Counter[str] = Counter()
    confirmed: Counter[str] = Counter()
    reasons: dict[str, Counter[str]] = defaultdict(Counter)
    for row in selections:
        phase = str(row["phase"])
        total[phase] += 1
        selection = row["selection"]
        if selection["status"] == "confirmed":
            confirmed[phase] += 1
        else:
            reasons[phase].update(selection["reason_codes"])
    return {
        phase: {
            "eligible_numeric_candidate_count": int(total[phase]),
            "confirmed_count": int(confirmed[phase]),
            "confirmed_fraction": (
                float(confirmed[phase] / total[phase]) if total[phase] else 0.0
            ),
            "reason_counts": dict(sorted(reasons[phase].items())),
        }
        for phase in EVENT_PHASES
    }


def _interval_length_summary(
    rows_by_episode: Mapping[int, Sequence[Mapping[str, Any]]],
    *,
    draw: Sequence[int],
) -> dict[str, Any]:
    values: dict[str, list[int]] = defaultdict(list)
    for episode_id in map(int, draw):
        for row in rows_by_episode[episode_id]:
            values[str(row["phase"])].append(
                int(row["interval"][1]) - int(row["interval"][0])
            )
    return {phase: _summary(values[phase]) for phase in EVENT_PHASES}


def _select_candidate(
    phase: str,
    candidates: Sequence[Mapping[str, Any]],
    eligible: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any] | None, dict[str, Any]]:
    if not eligible:
        return None, {
            "local_peak_count": 0,
            "equal_separated_peak_ambiguity": False,
            "competing_peak_distance_steps": None,
            "competing_peak_score_gap": None,
        }
    # Offset/support/change eligibility is frozen before any ranking or
    # local-peak comparison.  An ineligible neighbor must not suppress an
    # otherwise valid peak.
    by_step = {int(candidate["step"]): candidate for candidate in eligible}
    local: list[Mapping[str, Any]] = []
    if phase == "ready":
        local = list(eligible)
    else:
        for candidate in eligible:
            step = int(candidate["step"])
            value = float(candidate["aggregate_change"])
            neighbor_values = [
                float(by_step[neighbor]["aggregate_change"])
                for neighbor in (step - 1, step + 1)
                if neighbor in by_step
                and by_step[neighbor]["aggregate_change"] is not None
            ]
            if all(value >= neighbor for neighbor in neighbor_values):
                local.append(candidate)
        if not local:
            local = list(eligible)

    def support(candidate: Mapping[str, Any]) -> float:
        value = candidate["minimum_support_excess"]
        return float("-inf") if value is None else float(value)

    if phase == "ready":
        selected = min(
            local,
            key=lambda candidate: (
                float(candidate["aggregate_change"]),
                -support(candidate),
                int(candidate["absolute_offset_steps"]),
                int(candidate["step"]),
            ),
        )
    elif phase in ("dig_entry_proxy", "dump_start_proxy"):
        selected = min(
            local,
            key=lambda candidate: (
                int(candidate["step"]),
                -float(candidate["aggregate_change"]),
                -support(candidate),
                int(candidate["absolute_offset_steps"]),
            ),
        )
    elif phase == "dump_end_proxy":
        selected = max(
            local,
            key=lambda candidate: (
                int(candidate["step"]),
                float(candidate["aggregate_change"]),
                support(candidate),
                -int(candidate["absolute_offset_steps"]),
            ),
        )
    else:
        selected = max(
            local,
            key=lambda candidate: (
                float(candidate["aggregate_change"]),
                support(candidate),
                -int(candidate["absolute_offset_steps"]),
                -int(candidate["step"]),
            ),
        )
    competing = [
        candidate
        for candidate in local
        if int(candidate["step"]) != int(selected["step"])
    ]
    if competing:
        runner_up = min(
            competing,
            key=lambda candidate: (
                abs(
                    float(candidate["aggregate_change"])
                    - float(selected["aggregate_change"])
                ),
                -abs(int(candidate["step"]) - int(selected["step"])),
            ),
        )
        score_gap = abs(
            float(selected["aggregate_change"]) - float(runner_up["aggregate_change"])
        )
        distance = abs(int(selected["step"]) - int(runner_up["step"]))
    else:
        score_gap = None
        distance = None
    ambiguity = bool(
        score_gap is not None
        and score_gap <= np.finfo(np.float32).eps
        and int(distance) > 1
    )
    return selected, {
        "local_peak_count": len(local),
        "equal_separated_peak_ambiguity": ambiguity,
        "competing_peak_distance_steps": distance,
        "competing_peak_score_gap": score_gap,
    }


def _event_change(
    features: Mapping[tuple[int, int], Mapping[str, np.ndarray]],
    *,
    episode_id: int,
    step: int,
    role: str,
    kind: str,
) -> float | None:
    current = _feature(features, episode_id, step, role)
    if current is None:
        return None
    previous = _feature(features, episode_id, step - 1, role)
    following = _feature(features, episode_id, step + 1, role)
    if kind == "stable_two_sided":
        if previous is None or following is None:
            return None
        return max(
            _cosine_change(previous, current), _cosine_change(current, following)
        )
    if kind == "entering":
        return None if previous is None else _cosine_change(previous, current)
    if kind == "centered":
        return (
            None
            if previous is None or following is None
            else _cosine_change(previous, following)
        )
    if kind == "exiting":
        return None if following is None else _cosine_change(current, following)
    raise ValueError(f"unknown event change kind: {kind}")


def _feature(
    features: Mapping[tuple[int, int], Mapping[str, np.ndarray]],
    episode_id: int,
    step: int,
    role: str,
) -> np.ndarray | None:
    row = features.get((int(episode_id), int(step)))
    if row is None or role not in row:
        return None
    return np.asarray(row[role], dtype=np.float32)


def _prototype_scores(
    feature: np.ndarray,
    prototypes: Mapping[str, Mapping[str, np.ndarray]],
    *,
    role: str,
) -> dict[str, float]:
    return {
        phase: float(np.dot(feature, np.asarray(prototypes[phase][role])))
        for phase in EVENT_PHASES
    }


def _cosine_change(first: np.ndarray, second: np.ndarray) -> float:
    return max(0.0, float(1.0 - np.dot(first, second)))


def _require_shared_ready_invariant(
    cycle_results: Mapping[str, Mapping[str, Any]],
) -> None:
    by_episode: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in cycle_results.values():
        by_episode[int(row["episode_id"])].append(row)
    for rows in by_episode.values():
        ordered = sorted(rows, key=lambda row: int(row["cycle_id"]))
        for current, following in zip(ordered[:-1], ordered[1:]):
            current_key = current["event_keys"]["ready_end"]
            following_key = following["event_keys"]["ready_start"]
            if current_key is not None and following_key is not None:
                if current_key != following_key:
                    raise ValueError(
                        "adjacent ready_end/ready_start do not share one boundary"
                    )
                current_step = current["event_steps"]["ready_end"]
                following_step = following["event_steps"]["ready_start"]
                if current_step != following_step:
                    raise ValueError(
                        "shared ready boundary has inconsistent representative"
                    )


def _current_sector_order_valid(
    steps: Mapping[str, int | None],
) -> bool:
    """Validate only order relations that can contaminate current dig sector.

    Missing unrelated ready/dump evidence remains a cycle-level ambiguity but
    must not erase an otherwise observable current dig sector.  Any available
    confirmed event that places dig after carry/dump or before ready_start
    still invalidates the sector row.
    """

    dig = steps.get("dig_entry_proxy")
    if dig is None:
        return False
    before = steps.get("ready_start")
    if before is not None and int(before) > int(dig):
        return False
    for event_name in (
        "carry_transition_proxy",
        "dump_start_proxy",
        "dump_end_proxy",
        "ready_end",
    ):
        after = steps.get(event_name)
        if after is not None and int(dig) > int(after):
            return False
    return True


def _selection_summary(
    selections: Mapping[str, Mapping[str, Any]],
    cycle_results: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    phase_total: Counter[str] = Counter()
    phase_confirmed: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    for selection in selections.values():
        phase = str(selection["phase"])
        phase_total[phase] += 1
        if selection["status"] == "confirmed":
            phase_confirmed[phase] += 1
        else:
            reason_counts.update(selection["reason_codes"])
    return {
        "unique_event_count": len(selections),
        "cycle_count": len(cycle_results),
        "event_order_valid_cycle_count": sum(
            bool(row["event_order_valid"]) for row in cycle_results.values()
        ),
        "current_sector_order_valid_cycle_count": sum(
            bool(row["current_sector_order_valid"]) for row in cycle_results.values()
        ),
        "by_phase": {
            phase: {
                "count": int(phase_total[phase]),
                "confirmed_count": int(phase_confirmed[phase]),
                "confirmed_fraction": (
                    float(phase_confirmed[phase] / phase_total[phase])
                    if phase_total[phase]
                    else 0.0
                ),
            }
            for phase in EVENT_PHASES
        },
        "reason_counts": dict(sorted(reason_counts.items())),
    }


def _selection_confidence(
    selection: Mapping[str, Any],
    *,
    selector: Mapping[str, Any],
    calibration_samples: Mapping[str, Any],
    stability: Mapping[str, Any],
) -> dict[str, Any]:
    selected = selection["selected"]
    phase = str(selection["phase"])
    support_components: dict[str, float] = {}
    change_components: dict[str, float] = {}
    for role in EVENT_REQUIRED_ROLES[phase]:
        similarity = float(selected["role_metrics"][role]["expected_similarity"])
        support_components[role] = _positive_percentile(
            calibration_samples["support"][phase][role],
            similarity,
            higher_is_better=True,
        )
        change_components[role] = _positive_percentile(
            calibration_samples["change"][phase][role],
            float(selected["role_change"][role]),
            higher_is_better=phase != "ready",
        )
    bound = selector["offset_bounds"][phase]
    low = int(bound["minimum_signed_offset_steps"])
    high = int(bound["maximum_signed_offset_steps"])
    offset = int(selected["signed_offset_steps"])
    span = max(high - low, 0)
    offset_centrality = (
        1.0
        if span == 0
        else float((min(offset - low, high - offset) + 1) / (span / 2.0 + 1))
    )
    offset_centrality = float(np.clip(offset_centrality, 0.0, 1.0))
    raw_reselection = float(stability["reselection_within_point_tolerance_frequency"])
    reselection = float(
        (int(stability["reselection_within_point_tolerance_replicates"]) + 1)
        / (int(stability["attempted_successful_replicates"]) + 2)
    )
    joint = min(
        list(support_components.values())
        + list(change_components.values())
        + [offset_centrality, reselection]
    )
    return {
        "kind": "empirical_support_score_not_probability",
        "required_role_support_percentile": support_components,
        "required_role_change_percentile": change_components,
        "offset_centrality": offset_centrality,
        "bootstrap_reselection_within_tolerance_frequency": raw_reselection,
        "bootstrap_reselection_add_one_support": reselection,
        "joint": float(joint),
    }


def _positive_percentile(
    samples: Sequence[float],
    value: float,
    *,
    higher_is_better: bool,
) -> float:
    array = np.sort(np.asarray(samples, dtype=np.float64))
    if array.size == 0:
        return 0.0
    if higher_is_better:
        rank = int(np.searchsorted(array, value, side="right"))
    else:
        rank = int(array.size - np.searchsorted(array, value, side="left"))
    return float((rank + 1) / (array.size + 2))


def _non_identity_permutation(
    values: Sequence[str],
    rng: np.random.Generator,
) -> tuple[str, ...]:
    original = tuple(map(str, values))
    while True:
        candidate = tuple(rng.permutation(original).tolist())
        if candidate != original:
            return candidate


def _summary(values: Sequence[float | int]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("cannot summarize an empty sample")
    return {
        "count": int(array.size),
        "median": float(np.median(array)),
        "p02_5": float(np.quantile(array, 0.025)),
        "p50": float(np.quantile(array, 0.5)),
        "p97_5": float(np.quantile(array, 0.975)),
        "std": float(np.std(array)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def _vector_summary(values: Sequence[Sequence[float]]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "median": np.median(array, axis=0).tolist(),
        "p02_5": np.quantile(array, 0.025, axis=0).tolist(),
        "p97_5": np.quantile(array, 0.975, axis=0).tolist(),
        "std": np.std(array, axis=0).tolist(),
    }
