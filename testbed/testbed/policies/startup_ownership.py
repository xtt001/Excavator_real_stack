"""Post-hoc ownership scoring for expert prestart and policy early motion.

The first expert effective command is used only as an offline label.  It is
never exposed as a policy input and this module has no runtime or HDF5 writes.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Any

import numpy as np

from testbed.policies.action_start_distribution import FORBIDDEN_HELDOUT
from testbed.policies.deadzone_eval import AXIS_NAMES, effective_direction_mask

DIRECTION_NAMES = ("pos", "neg")


def audit_startup_ownership(
    *,
    expert_actions: Mapping[str, np.ndarray],
    policy_collections: Mapping[str, Mapping[str, np.ndarray]],
    thresholds: Mapping[str, Mapping[str, float]],
    sample_hz: float,
) -> dict[str, Any]:
    """Compare stepwise imitation and autonomy-aligned prestart semantics."""

    rate_hz = float(sample_hz)
    if not np.isfinite(rate_hz) or rate_hz <= 0.0:
        raise ValueError("sample_hz must be finite and positive")
    if not expert_actions:
        raise ValueError("expert_actions must not be empty")
    if not policy_collections:
        raise ValueError("policy_collections must not be empty")

    episode_ids = sorted(
        (str(value) for value in expert_actions), key=_episode_sort_key
    )
    forbidden = [
        episode_id
        for episode_id in episode_ids
        if _episode_number(episode_id) in FORBIDDEN_HELDOUT
    ]
    if forbidden:
        raise ValueError(f"held-out episodes are forbidden: {forbidden}")

    expert = {
        episode_id: _validate_action(
            expert_actions[episode_id], name=f"expert_actions[{episode_id!r}]"
        )
        for episode_id in episode_ids
    }
    policies: dict[str, dict[str, np.ndarray]] = {}
    for model, collection in policy_collections.items():
        label = str(model).strip()
        if not label:
            raise ValueError("policy model labels must not be empty")
        collection_ids = {str(value) for value in collection}
        if collection_ids != set(episode_ids):
            raise ValueError(
                f"policy collection {label!r} episode ids differ: "
                f"missing={sorted(set(episode_ids) - collection_ids)} "
                f"extra={sorted(collection_ids - set(episode_ids))}"
            )
        policies[label] = {}
        for episode_id in episode_ids:
            action = _validate_action(
                collection[episode_id],
                name=f"policy_collections[{label!r}][{episode_id!r}]",
            )
            if action.shape != expert[episode_id].shape:
                raise ValueError(
                    f"policy/expert shapes differ for {label}/{episode_id}: "
                    f"{action.shape} != {expert[episode_id].shape}"
                )
            policies[label][episode_id] = action

    normalized_thresholds = {
        axis: {
            direction: float(thresholds[axis][direction])
            for direction in DIRECTION_NAMES
        }
        for axis in AXIS_NAMES
    }
    expert_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    expert_effective_by_episode: dict[str, np.ndarray] = {}
    for episode_id in episode_ids:
        action = expert[episode_id]
        effective = effective_direction_mask(action, normalized_thresholds)
        expert_effective_by_episode[episode_id] = effective
        any_effective = effective.any(axis=(1, 2))
        starts = np.flatnonzero(any_effective)
        onset = int(starts[0]) if starts.size else None
        prestart_end = int(onset if onset is not None else action.shape[0])
        first_directions = (
            _active_direction_rows(effective[onset]) if onset is not None else []
        )
        prestart_neutral = int(np.count_nonzero(~any_effective[:prestart_end]))
        elsewhere_neutral = int(np.count_nonzero(~any_effective[prestart_end:]))
        expert_rows.append(
            {
                "episode_id": episode_id,
                "steps": int(action.shape[0]),
                "expert_start_observed": onset is not None,
                "expert_first_onset_step": onset,
                "expert_first_onset_seconds": (
                    float(onset / rate_hz) if onset is not None else None
                ),
                "prestart_end_step_exclusive": prestart_end,
                "prestart_frames": prestart_end,
                "prestart_seconds": float(prestart_end / rate_hz),
                "expert_first_start_directions": first_directions,
                "expert_first_start_axis": (
                    first_directions[0]["axis"]
                    if len(first_directions) == 1
                    else None
                ),
                "expert_first_start_direction": (
                    first_directions[0]["direction"]
                    if len(first_directions) == 1
                    else None
                ),
                "expert_prestart_all_axis_neutral_frames": prestart_neutral,
                "expert_prestart_all_axis_neutral_pct": _pct(
                    prestart_neutral, prestart_end
                ),
                "expert_elsewhere_frames": int(action.shape[0] - prestart_end),
                "expert_elsewhere_all_axis_neutral_frames": elsewhere_neutral,
                "expert_elsewhere_all_axis_neutral_pct": _pct(
                    elsewhere_neutral, int(action.shape[0] - prestart_end)
                ),
            }
        )

        allowed = np.zeros((len(AXIS_NAMES), len(DIRECTION_NAMES)), dtype=bool)
        for direction in first_directions:
            allowed[int(direction["axis_index"]), int(direction["direction_index"])] = (
                True
            )
        startup_axis_indices = {
            int(direction["axis_index"]) for direction in first_directions
        }
        for model, collection in policies.items():
            policy_effective = effective_direction_mask(
                collection[episode_id], normalized_thresholds
            )
            model_rows.append(
                _score_model_prestart(
                    model=model,
                    episode_id=episode_id,
                    policy_effective=policy_effective,
                    expert_effective=effective,
                    allowed_start_directions=allowed,
                    startup_axis_indices=startup_axis_indices,
                    prestart_end=prestart_end,
                    expert_onset=onset,
                    sample_hz=rate_hz,
                )
            )

    return {
        "schema_version": 1,
        "contract": "startup_ownership_audit_v1",
        "action_domain": "direct_policy_output",
        "sample_hz": rate_hz,
        "episode_ids": episode_ids,
        "forbidden_heldout_episode_ids": sorted(FORBIDDEN_HELDOUT),
        "heldout_evaluated": False,
        "thresholds": normalized_thresholds,
        "semantics": {
            "prestart_operator_wait": (
                "episode start through the tick before the first expert "
                "deadzone-effective command"
            ),
            "imitation_aligned_early_extra": (
                "existing stepwise rule: policy effective while no current "
                "expert axis/direction matches"
            ),
            "autonomy_aligned_early_start": (
                "during prestart, only the first expert startup axis/direction "
                "may be early; any opposite or unsupported direction keeps the "
                "frame wrong/extra"
            ),
            "causality": (
                "first expert onset is a post-hoc ownership label, not a "
                "deployment input; no policy actions or images are generated"
            ),
        },
        "expert_episode_rows": expert_rows,
        "model_episode_rows": model_rows,
        "aggregate": {
            "expert": _aggregate_expert_rows(expert_rows, sample_hz=rate_hz),
            "models": {
                model: _aggregate_model_rows(
                    [row for row in model_rows if row["model"] == model],
                    sample_hz=rate_hz,
                )
                for model in policies
            },
        },
    }


def _score_model_prestart(
    *,
    model: str,
    episode_id: str,
    policy_effective: np.ndarray,
    expert_effective: np.ndarray,
    allowed_start_directions: np.ndarray,
    startup_axis_indices: set[int],
    prestart_end: int,
    expert_onset: int | None,
    sample_hz: float,
) -> dict[str, Any]:
    policy_prestart = policy_effective[:prestart_end]
    expert_prestart = expert_effective[:prestart_end]
    policy_any = policy_prestart.any(axis=(1, 2))
    same_current_direction = (policy_prestart & expert_prestart).any(axis=(1, 2))
    imitation_extra = policy_any & ~same_current_direction

    allowed_active = (policy_prestart & allowed_start_directions).any(axis=(1, 2))
    unsupported_direction_mask = policy_prestart & ~allowed_start_directions
    unsupported_active = unsupported_direction_mask.any(axis=(1, 2))
    clean_aligned = allowed_active & ~unsupported_active
    mixed_aligned_unsupported = allowed_active & unsupported_active

    opposite_mask = np.zeros_like(allowed_start_directions)
    for axis_index, direction_index in np.argwhere(allowed_start_directions):
        opposite_mask[int(axis_index), 1 - int(direction_index)] = True
    opposite_active = (policy_prestart & opposite_mask).any(axis=(1, 2))
    other_axis_mask = np.ones_like(allowed_start_directions)
    for axis_index in startup_axis_indices:
        other_axis_mask[axis_index, :] = False
    other_axis_active = (policy_prestart & other_axis_mask).any(axis=(1, 2))

    clean_steps = np.flatnonzero(clean_aligned)
    first_clean_step = int(clean_steps[0]) if clean_steps.size else None
    lead_ticks = (
        int(expert_onset - first_clean_step)
        if expert_onset is not None and first_clean_step is not None
        else None
    )
    return {
        "model": model,
        "episode_id": episode_id,
        "prestart_frames": int(prestart_end),
        "policy_prestart_all_axis_neutral_frames": int(
            np.count_nonzero(~policy_any)
        ),
        "imitation_aligned_early_extra_frames": int(
            np.count_nonzero(imitation_extra)
        ),
        "imitation_aligned_early_extra_pct": _pct(
            int(np.count_nonzero(imitation_extra)), prestart_end
        ),
        "autonomy_aligned_early_start_frames": int(
            np.count_nonzero(clean_aligned)
        ),
        "autonomy_aligned_early_start_pct": _pct(
            int(np.count_nonzero(clean_aligned)), prestart_end
        ),
        "autonomy_supported_direction_active_frames": int(
            np.count_nonzero(allowed_active)
        ),
        "autonomy_wrong_or_extra_frames": int(
            np.count_nonzero(unsupported_active)
        ),
        "autonomy_wrong_or_extra_pct": _pct(
            int(np.count_nonzero(unsupported_active)), prestart_end
        ),
        "autonomy_reclassified_early_start_frames": int(
            np.count_nonzero(imitation_extra & clean_aligned)
        ),
        "autonomy_mixed_supported_and_unsupported_frames": int(
            np.count_nonzero(mixed_aligned_unsupported)
        ),
        "autonomy_opposite_start_axis_frames": int(
            np.count_nonzero(opposite_active)
        ),
        "autonomy_other_axis_frames": int(np.count_nonzero(other_axis_active)),
        "autonomy_unsupported_direction_activations": int(
            np.count_nonzero(unsupported_direction_mask)
        ),
        "first_autonomy_aligned_early_start_step": first_clean_step,
        "first_autonomy_aligned_early_start_seconds": (
            float(first_clean_step / sample_hz)
            if first_clean_step is not None
            else None
        ),
        "autonomy_aligned_lead_ticks": lead_ticks,
        "autonomy_aligned_lead_seconds": (
            float(lead_ticks / sample_hz) if lead_ticks is not None else None
        ),
    }


def _aggregate_expert_rows(
    rows: list[dict[str, Any]], *, sample_hz: float
) -> dict[str, Any]:
    onset_ticks = [
        int(row["expert_first_onset_step"])
        for row in rows
        if row["expert_first_onset_step"] is not None
    ]
    first_direction_counts: Counter[str] = Counter()
    for row in rows:
        for direction in row["expert_first_start_directions"]:
            first_direction_counts[f"{direction['axis']}_{direction['direction']}"] += 1
    prestart_frames = sum(int(row["prestart_frames"]) for row in rows)
    prestart_neutral = sum(
        int(row["expert_prestart_all_axis_neutral_frames"]) for row in rows
    )
    elsewhere_frames = sum(int(row["expert_elsewhere_frames"]) for row in rows)
    elsewhere_neutral = sum(
        int(row["expert_elsewhere_all_axis_neutral_frames"]) for row in rows
    )
    return {
        "episodes": len(rows),
        "steps": sum(int(row["steps"]) for row in rows),
        "episodes_with_expert_start": len(onset_ticks),
        "episodes_without_expert_start": len(rows) - len(onset_ticks),
        "first_start_direction_episode_counts": dict(first_direction_counts),
        "first_onset_ticks": _distribution(onset_ticks),
        "first_onset_seconds": _distribution(
            [value / sample_hz for value in onset_ticks]
        ),
        "prestart_frames": prestart_frames,
        "prestart_seconds": float(prestart_frames / sample_hz),
        "prestart_all_axis_neutral_frames": prestart_neutral,
        "prestart_all_axis_neutral_pct": _pct(prestart_neutral, prestart_frames),
        "elsewhere_frames": elsewhere_frames,
        "elsewhere_all_axis_neutral_frames": elsewhere_neutral,
        "elsewhere_all_axis_neutral_pct": _pct(
            elsewhere_neutral, elsewhere_frames
        ),
    }


def _aggregate_model_rows(
    rows: list[dict[str, Any]], *, sample_hz: float
) -> dict[str, Any]:
    prestart_frames = sum(int(row["prestart_frames"]) for row in rows)
    count_fields = (
        "policy_prestart_all_axis_neutral_frames",
        "imitation_aligned_early_extra_frames",
        "autonomy_aligned_early_start_frames",
        "autonomy_supported_direction_active_frames",
        "autonomy_wrong_or_extra_frames",
        "autonomy_reclassified_early_start_frames",
        "autonomy_mixed_supported_and_unsupported_frames",
        "autonomy_opposite_start_axis_frames",
        "autonomy_other_axis_frames",
        "autonomy_unsupported_direction_activations",
    )
    aggregate = {
        "episodes": len(rows),
        "prestart_frames": prestart_frames,
        **{
            field: sum(int(row[field]) for row in rows)
            for field in count_fields
        },
    }
    for field in (
        "policy_prestart_all_axis_neutral_frames",
        "imitation_aligned_early_extra_frames",
        "autonomy_aligned_early_start_frames",
        "autonomy_wrong_or_extra_frames",
        "autonomy_reclassified_early_start_frames",
    ):
        aggregate[field.removesuffix("_frames") + "_pct"] = _pct(
            int(aggregate[field]), prestart_frames
        )
    first_steps = [
        int(row["first_autonomy_aligned_early_start_step"])
        for row in rows
        if row["first_autonomy_aligned_early_start_step"] is not None
    ]
    leads = [
        int(row["autonomy_aligned_lead_ticks"])
        for row in rows
        if row["autonomy_aligned_lead_ticks"] is not None
    ]
    aggregate.update(
        {
            "episodes_with_autonomy_aligned_early_start": len(first_steps),
            "first_autonomy_aligned_early_start_ticks": _distribution(first_steps),
            "first_autonomy_aligned_early_start_seconds": _distribution(
                [value / sample_hz for value in first_steps]
            ),
            "autonomy_aligned_lead_ticks": _distribution(leads),
            "autonomy_aligned_lead_seconds": _distribution(
                [value / sample_hz for value in leads]
            ),
        }
    )
    return aggregate


def _active_direction_rows(mask: np.ndarray) -> list[dict[str, Any]]:
    return [
        {
            "axis_index": int(axis_index),
            "axis": AXIS_NAMES[int(axis_index)],
            "direction_index": int(direction_index),
            "direction": DIRECTION_NAMES[int(direction_index)],
        }
        for axis_index, direction_index in np.argwhere(mask)
    ]


def _distribution(values: list[int] | list[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "p10": None,
            "median": None,
            "mean": None,
            "p90": None,
            "max": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "p10": float(np.percentile(array, 10)),
        "median": float(np.percentile(array, 50)),
        "mean": float(np.mean(array)),
        "p90": float(np.percentile(array, 90)),
        "max": float(np.max(array)),
    }


def _pct(count: int, total: int) -> float | None:
    return 100.0 * float(count) / float(total) if total else None


def _validate_action(value: np.ndarray, *, name: str) -> np.ndarray:
    action = np.asarray(value, dtype=np.float32)
    expected_tail = (len(AXIS_NAMES),)
    if action.ndim != 2 or action.shape[1:] != expected_tail:
        raise ValueError(f"{name} must have shape (T, {len(AXIS_NAMES)})")
    if action.shape[0] <= 0 or not np.isfinite(action).all():
        raise ValueError(f"{name} must be non-empty and finite")
    return action


def _episode_number(value: str) -> int:
    try:
        return int(str(value).split("_")[-1])
    except ValueError as exc:
        raise ValueError(f"invalid episode id: {value!r}") from exc


def _episode_sort_key(value: str) -> tuple[int, str]:
    return (_episode_number(value), str(value))


__all__ = ["audit_startup_ownership"]
