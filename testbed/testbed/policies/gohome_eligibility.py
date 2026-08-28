"""Offline helpers for conservative gohome eligibility probes."""

from __future__ import annotations

from typing import Any

import numpy as np


def consecutive_active_mask(
    probability: np.ndarray,
    *,
    threshold: float,
    consecutive_steps: int = 1,
) -> np.ndarray:
    """Return a runtime-causal mask that fires after N consecutive active frames."""

    probs = np.asarray(probability, dtype=np.float32).reshape(-1)
    required = max(1, int(consecutive_steps))
    active = probs >= float(threshold)
    out = np.zeros(active.shape[0], dtype=bool)
    run = 0
    for idx, value in enumerate(active):
        run = run + 1 if bool(value) else 0
        out[idx] = run >= required
    return out


def gated_active_mask(
    *,
    candidate_active: np.ndarray,
    eligibility_probability: np.ndarray,
    eligibility_threshold: float,
    eligibility_consecutive_steps: int = 1,
) -> np.ndarray:
    """Return eligibility activations allowed only inside a candidate phase."""

    candidate = np.asarray(candidate_active, dtype=bool).reshape(-1)
    eligibility = consecutive_active_mask(
        eligibility_probability,
        threshold=float(eligibility_threshold),
        consecutive_steps=int(eligibility_consecutive_steps),
    )
    n = int(min(candidate.shape[0], eligibility.shape[0]))
    if n <= 0:
        return np.zeros(0, dtype=bool)
    return candidate[:n] & eligibility[:n]


def gohome_event_metrics(
    *,
    episode_id: str,
    probability: np.ndarray,
    eligible_label: np.ndarray,
    loss_mask: np.ndarray,
    tail_idle_mask: np.ndarray | None = None,
    threshold: float,
    consecutive_steps: int = 1,
) -> dict[str, Any]:
    """Compute event-level gohome eligibility metrics for one episode."""

    label = np.asarray(eligible_label, dtype=bool).reshape(-1)
    mask = np.asarray(loss_mask, dtype=bool).reshape(-1)
    prob = np.asarray(probability, dtype=np.float32).reshape(-1)
    tail = np.asarray(tail_idle_mask, dtype=bool).reshape(-1) if tail_idle_mask is not None else None
    lengths = [label.shape[0], mask.shape[0], prob.shape[0]]
    if tail is not None:
        lengths.append(tail.shape[0])
    n = int(min(lengths))
    if n <= 0:
        raise ValueError(f"empty gohome eligibility episode: {episode_id}")
    label = label[:n]
    mask = mask[:n]
    prob = prob[:n]
    tail = tail[:n] if tail is not None else label

    positive_idx = np.flatnonzero(label & mask)
    if positive_idx.size == 0:
        return {
            "episode_id": str(episode_id),
            "steps": n,
            "eligible_start": "",
            "eligible_end": "",
            "threshold": float(threshold),
            "consecutive_steps": int(consecutive_steps),
            "first_active_step": "",
            "first_eligible_active_step": "",
            "first_early_active_step": "",
            "detected": 0,
            "early_false_positive": 0,
            "pre_tail_false_positive": 0,
            "early_active_frames": 0,
            "pre_tail_active_frames": 0,
            "dwell_early_active_frames": 0,
            "eligible_active_frames": 0,
            "detection_delay_steps": "",
            "steps_before_t_go": "",
        }

    active = consecutive_active_mask(prob, threshold=threshold, consecutive_steps=consecutive_steps)
    return gohome_event_metrics_from_active_mask(
        episode_id=episode_id,
        active_mask=active,
        eligible_label=label,
        loss_mask=mask,
        tail_idle_mask=tail,
        gate=f"thr_{float(threshold):.2f}_c{int(consecutive_steps)}",
        threshold=threshold,
        consecutive_steps=consecutive_steps,
    )


def gohome_event_metrics_from_active_mask(
    *,
    episode_id: str,
    active_mask: np.ndarray,
    eligible_label: np.ndarray,
    loss_mask: np.ndarray,
    tail_idle_mask: np.ndarray | None = None,
    gate: str | None = None,
    threshold: float | None = None,
    consecutive_steps: int | None = None,
) -> dict[str, Any]:
    """Compute event-level gohome eligibility metrics from an already gated mask."""

    label = np.asarray(eligible_label, dtype=bool).reshape(-1)
    mask = np.asarray(loss_mask, dtype=bool).reshape(-1)
    active_raw = np.asarray(active_mask, dtype=bool).reshape(-1)
    tail = np.asarray(tail_idle_mask, dtype=bool).reshape(-1) if tail_idle_mask is not None else None
    lengths = [label.shape[0], mask.shape[0], active_raw.shape[0]]
    if tail is not None:
        lengths.append(tail.shape[0])
    n = int(min(lengths))
    if n <= 0:
        raise ValueError(f"empty gohome eligibility episode: {episode_id}")
    label = label[:n]
    mask = mask[:n]
    active = active_raw[:n] & mask[:n]
    tail = tail[:n] if tail is not None else label

    positive_idx = np.flatnonzero(label & mask)
    if positive_idx.size == 0:
        row = {
            "episode_id": str(episode_id),
            "steps": n,
            "eligible_start": "",
            "eligible_end": "",
            "first_active_step": "",
            "first_eligible_active_step": "",
            "first_early_active_step": "",
            "detected": 0,
            "early_false_positive": 0,
            "pre_tail_false_positive": 0,
            "early_active_frames": 0,
            "pre_tail_active_frames": 0,
            "dwell_early_active_frames": 0,
            "eligible_active_frames": 0,
            "detection_delay_steps": "",
            "steps_before_t_go": "",
        }
        if gate is not None:
            row["gate"] = str(gate)
        if threshold is not None:
            row["threshold"] = float(threshold)
        if consecutive_steps is not None:
            row["consecutive_steps"] = int(consecutive_steps)
        return row

    eligible_start = int(positive_idx[0])
    eligible_end = int(positive_idx[-1])
    early_region = np.arange(n) < eligible_start
    pre_tail_region = early_region & ~tail
    dwell_early_region = early_region & tail
    eligible_region = label & mask
    early_idx = np.flatnonzero(active & early_region)
    pre_tail_idx = np.flatnonzero(active & pre_tail_region)
    dwell_early_idx = np.flatnonzero(active & dwell_early_region)
    eligible_active_idx = np.flatnonzero(active & eligible_region)
    all_active_idx = np.flatnonzero(active)

    first_active = int(all_active_idx[0]) if all_active_idx.size else ""
    first_eligible = int(eligible_active_idx[0]) if eligible_active_idx.size else ""
    first_early = int(early_idx[0]) if early_idx.size else ""
    detected = int(eligible_active_idx.size > 0)
    early = int(early_idx.size > 0)
    pre_tail_early = int(pre_tail_idx.size > 0)
    delay: int | str = int(first_eligible) - eligible_start if detected else ""
    before_t_go: int | str = eligible_end - int(first_eligible) if detected else ""

    row = {
        "episode_id": str(episode_id),
        "steps": n,
        "eligible_start": eligible_start,
        "eligible_end": eligible_end,
        "first_active_step": first_active,
        "first_eligible_active_step": first_eligible,
        "first_early_active_step": first_early,
        "detected": detected,
        "early_false_positive": early,
        "pre_tail_false_positive": pre_tail_early,
        "early_active_frames": int(early_idx.size),
        "pre_tail_active_frames": int(pre_tail_idx.size),
        "dwell_early_active_frames": int(dwell_early_idx.size),
        "eligible_active_frames": int(eligible_active_idx.size),
        "detection_delay_steps": delay,
        "steps_before_t_go": before_t_go,
    }
    if gate is not None:
        row["gate"] = str(gate)
    if threshold is not None:
        row["threshold"] = float(threshold)
    if consecutive_steps is not None:
        row["consecutive_steps"] = int(consecutive_steps)
    return row


def aggregate_gohome_event_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-episode gohome eligibility event rows."""

    episodes = len(rows)
    detected = sum(int(row.get("detected", 0)) for row in rows)
    early = sum(int(row.get("early_false_positive", 0)) for row in rows)
    pre_tail_early = sum(int(row.get("pre_tail_false_positive", 0)) for row in rows)
    early_frames = sum(int(row.get("early_active_frames", 0)) for row in rows)
    pre_tail_frames = sum(int(row.get("pre_tail_active_frames", 0)) for row in rows)
    dwell_early_frames = sum(int(row.get("dwell_early_active_frames", 0)) for row in rows)
    eligible_active_frames = sum(int(row.get("eligible_active_frames", 0)) for row in rows)
    delays = [_numeric(row.get("detection_delay_steps")) for row in rows]
    margins = [_numeric(row.get("steps_before_t_go")) for row in rows]
    delays = [value for value in delays if value is not None]
    margins = [value for value in margins if value is not None]
    return {
        "episodes": int(episodes),
        "detected_episodes": int(detected),
        "early_false_positive_episodes": int(early),
        "pre_tail_false_positive_episodes": int(pre_tail_early),
        "early_active_frames": int(early_frames),
        "pre_tail_active_frames": int(pre_tail_frames),
        "dwell_early_active_frames": int(dwell_early_frames),
        "eligible_active_frames": int(eligible_active_frames),
        "event_recall": _rate(detected, episodes),
        "early_false_positive_episode_rate": _rate(early, episodes),
        "pre_tail_false_positive_episode_rate": _rate(pre_tail_early, episodes),
        "mean_detection_delay_steps": _mean(delays),
        "median_detection_delay_steps": _median(delays),
        "mean_steps_before_t_go": _mean(margins),
        "median_steps_before_t_go": _median(margins),
    }


def _numeric(value: Any) -> float | None:
    if value == "" or value is None:
        return None
    return float(value)


def _rate(num: int, den: int) -> float:
    return float(num) / float(den) if den else 0.0


def _mean(values: list[float]) -> float | str:
    return float(np.mean(values)) if values else ""


def _median(values: list[float]) -> float | str:
    return float(np.median(values)) if values else ""
