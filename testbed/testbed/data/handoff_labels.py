"""Handoff eligibility labels for policy-to-gohome training datasets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GohomeEligibilityLabels:
    t_go: int | None
    t_stop: int | None
    eligible_start: int | None
    gohome_eligible_label: np.ndarray
    gohome_loss_mask: np.ndarray
    tail_idle_mask: np.ndarray
    action_loss_mask: np.ndarray
    owner_automation: np.ndarray


def compute_gohome_eligibility_labels(
    *,
    actions: np.ndarray,
    go_home_requested: np.ndarray | None,
    go_home_start_accepted: np.ndarray | None,
    go_home_running: np.ndarray | None,
    idle_action_threshold: float,
    dwell_min_steps: int,
) -> GohomeEligibilityLabels:
    action_arr = _validate_actions(actions)
    n_steps = int(action_arr.shape[0])
    t_go = _first_positive_index(
        n_steps,
        go_home_requested,
        go_home_start_accepted,
        go_home_running,
    )
    if t_go is None:
        return GohomeEligibilityLabels(
            t_go=None,
            t_stop=None,
            eligible_start=None,
            gohome_eligible_label=np.zeros(n_steps, dtype=bool),
            gohome_loss_mask=np.zeros(n_steps, dtype=bool),
            tail_idle_mask=np.zeros(n_steps, dtype=bool),
            action_loss_mask=np.ones(n_steps, dtype=bool),
            owner_automation=np.zeros(n_steps, dtype=bool),
        )

    max_abs = np.max(np.abs(action_arr[: t_go + 1]), axis=1)
    active = np.flatnonzero(max_abs > float(idle_action_threshold))
    t_stop = int(active[-1]) + 1 if active.size else 0
    t_stop = min(max(t_stop, 0), t_go)
    eligible_start = int(t_stop) + max(0, int(dwell_min_steps))

    step = np.arange(n_steps, dtype=np.int64)
    eligible = (step >= eligible_start) & (step <= t_go)
    gohome_loss_mask = step <= t_go
    tail_idle_mask = (step >= t_stop) & (step <= t_go)
    owner_automation = step > t_go
    action_loss_mask = step < t_stop

    return GohomeEligibilityLabels(
        t_go=int(t_go),
        t_stop=int(t_stop),
        eligible_start=int(eligible_start),
        gohome_eligible_label=eligible.astype(bool),
        gohome_loss_mask=gohome_loss_mask.astype(bool),
        tail_idle_mask=tail_idle_mask.astype(bool),
        action_loss_mask=action_loss_mask.astype(bool),
        owner_automation=owner_automation.astype(bool),
    )


def _validate_actions(actions: np.ndarray) -> np.ndarray:
    arr = np.asarray(actions, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"actions must have shape (T, A), got {arr.shape}")
    if arr.shape[0] <= 0:
        raise ValueError("actions must contain at least one step")
    return arr


def _first_positive_index(n_steps: int, *arrays: np.ndarray | None) -> int | None:
    candidates: list[int] = []
    for raw in arrays:
        if raw is None:
            continue
        arr = np.asarray(raw).reshape(-1)
        if arr.size < n_steps:
            padded = np.zeros(n_steps, dtype=arr.dtype)
            padded[: arr.size] = arr
            arr = padded
        idx = np.flatnonzero(arr[:n_steps] > 0)
        if idx.size:
            candidates.append(int(idx[0]))
    return min(candidates) if candidates else None
