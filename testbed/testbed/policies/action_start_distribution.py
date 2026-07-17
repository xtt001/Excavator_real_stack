"""Evidence-first analysis of when expert axes actually start moving.

This module does not infer visual intent or physical limits.  It describes the
observed action/qpos/qvel distribution so sampling and loss design can be
chosen from data rather than from a global deadzone threshold guess.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from testbed.policies.deadzone_eval import AXIS_NAMES, effective_direction_mask

DIRECTION_NAMES = ("pos", "neg")
STATE_NAMES = ("idle", "pos", "neg")
FORBIDDEN_HELDOUT = frozenset({105, 106, 107, 108, 109})
BOUNDARY_BANDS = (
    "lt_0",
    "0_to_0.8",
    "0.8_to_0.9",
    "0.9_to_0.95",
    "0.95_to_0.99",
    "0.99_to_1.0",
    "1.0_to_1.01",
    "1.01_to_1.05",
    "1.05_to_1.1",
    "1.1_to_1.25",
    "1.25_to_1.5",
    "ge_1.5",
)


def analyze_action_start_distribution(
    *,
    dataset_dir: str | Path,
    episode_ids: Sequence[int],
    train_episode_ids: Sequence[int],
    thresholds: Mapping[str, Mapping[str, float]],
    persistence_horizon: int = 4,
    pre_window: int = 5,
    boundary_fraction: float = 0.10,
    ambiguity_bins: int = 5,
) -> dict[str, Any]:
    """Return transition, persistence, boundary, and low-dim ambiguity stats."""

    if int(persistence_horizon) < 1 or int(pre_window) < 1:
        raise ValueError("persistence_horizon and pre_window must be positive")
    if not 0.0 < float(boundary_fraction) < 1.0:
        raise ValueError("boundary_fraction must be in (0, 1)")
    if int(ambiguity_bins) < 2:
        raise ValueError("ambiguity_bins must be >= 2")
    dataset_path = Path(dataset_dir).expanduser().resolve()
    selected_ids = sorted({int(value) for value in episode_ids})
    train_ids = sorted({int(value) for value in train_episode_ids})
    forbidden = sorted(set(selected_ids) & FORBIDDEN_HELDOUT)
    if forbidden:
        raise ValueError(f"held-out episodes are forbidden: {forbidden}")
    if not set(train_ids).issubset(selected_ids):
        raise ValueError("train_episode_ids must be a subset of episode_ids")

    episodes: list[dict[str, Any]] = []
    all_train_qpos: list[np.ndarray] = []
    all_train_qvel: list[np.ndarray] = []
    loaded: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for episode_id in selected_ids:
        path = dataset_path / f"episode_{episode_id}.hdf5"
        if not path.exists():
            raise FileNotFoundError(path)
        with h5py.File(path, "r") as handle:
            action = np.asarray(handle["/action"][()], dtype=np.float32)
            qpos = np.asarray(handle["observations/qpos"][()], dtype=np.float32)
            qvel = np.asarray(handle["observations/qvel"][()], dtype=np.float32)
        if action.shape != qpos.shape or action.shape != qvel.shape:
            raise ValueError(
                f"episode {episode_id} action/qpos/qvel shapes differ: "
                f"{action.shape}, {qpos.shape}, {qvel.shape}"
            )
        loaded[episode_id] = (action, qpos, qvel)
        if episode_id in train_ids:
            all_train_qpos.append(qpos)
            all_train_qvel.append(qvel)
        episodes.append(
            {
                "episode_id": episode_id,
                "split": "train" if episode_id in train_ids else "validation",
                "steps": int(action.shape[0]),
                "sha256": sha256_file(path),
            }
        )

    train_qpos = np.concatenate(all_train_qpos, axis=0)
    train_qvel = np.concatenate(all_train_qvel, axis=0)
    qpos_edges = _quantile_edges(train_qpos, bins=int(ambiguity_bins))
    qvel_edges = _quantile_edges(train_qvel, bins=int(ambiguity_bins))

    state_counts = {
        axis: {state: 0 for state in STATE_NAMES} for axis in AXIS_NAMES
    }
    transition_summary: dict[str, dict[str, Any]] = {
        f"{axis}_{direction}": {
            "axis": axis,
            "direction": direction,
            "episode_count": 0,
            "transition_count": 0,
            "persistent_4_count": 0,
            "persistent_10_count": 0,
            "run_lengths": [],
            "onset_action_ratio": [],
            "pre_window_peak_ratio": [],
            "onset_qpos": [],
            "onset_qvel": [],
            "onset_qpos_delta_4": [],
            "pre_idle_run_length": [],
        }
        for axis in AXIS_NAMES
        for direction in DIRECTION_NAMES
    }
    transition_rows: list[dict[str, Any]] = []
    combo_counts: Counter[str] = Counter()
    combo_counts_by_split: dict[str, Counter[str]] = {
        "train": Counter(),
        "validation": Counter(),
    }
    boundary_counts: dict[str, dict[str, Any]] = {
        f"{axis}_{direction}": {
            "axis": axis,
            "direction": direction,
            "total_steps": 0,
            "positive_side_steps": 0,
            "effective_steps": 0,
            "near_idle_steps_0.9_to_1.0": 0,
            "near_effective_steps_1.0_to_1.1": 0,
            "bands": {band: 0 for band in BOUNDARY_BANDS},
            "ratios": [],
        }
        for axis in AXIS_NAMES
        for direction in DIRECTION_NAMES
    }
    first_transition_rows: list[dict[str, Any]] = []
    ambiguity_counts: dict[str, dict[tuple[int, int], Counter[str]]] = {
        axis: defaultdict(Counter) for axis in AXIS_NAMES
    }

    for episode_id in selected_ids:
        action, qpos, qvel = loaded[episode_id]
        effective = effective_direction_mask(action, dict(thresholds))
        for axis_index, axis in enumerate(AXIS_NAMES):
            positive = effective[:, axis_index, 0]
            negative = effective[:, axis_index, 1]
            state = np.full(action.shape[0], "idle", dtype=object)
            state[positive] = "pos"
            state[negative] = "neg"
            for state_name in STATE_NAMES:
                state_counts[axis][state_name] += int(np.count_nonzero(state == state_name))
            qpos_bin = np.digitize(qpos[:, axis_index], qpos_edges[:, axis_index], right=False)
            qvel_bin = np.digitize(qvel[:, axis_index], qvel_edges[:, axis_index], right=False)
            for direction_index, direction in enumerate(DIRECTION_NAMES):
                key = f"{axis}_{direction}"
                threshold = float(thresholds[axis][direction])
                sign = 1.0 if direction_index == 0 else -1.0
                ratios = sign * action[:, axis_index] / threshold
                boundary = boundary_counts[key]
                boundary["total_steps"] += int(ratios.size)
                boundary["positive_side_steps"] += int(np.count_nonzero(ratios > 0.0))
                boundary["effective_steps"] += int(np.count_nonzero(ratios >= 1.0))
                boundary["near_idle_steps_0.9_to_1.0"] += int(
                    np.count_nonzero((ratios >= 0.9) & (ratios < 1.0))
                )
                boundary["near_effective_steps_1.0_to_1.1"] += int(
                    np.count_nonzero((ratios >= 1.0) & (ratios < 1.1))
                )
                boundary["ratios"].extend(float(value) for value in ratios)
                for value in ratios:
                    boundary["bands"][_boundary_band(float(value))] += 1
            for step in range(action.shape[0]):
                ambiguity_counts[axis][(int(qpos_bin[step]), int(qvel_bin[step]))][
                    str(state[step])
                ] += 1

            for direction_index, direction in enumerate(DIRECTION_NAMES):
                active = effective[:, axis_index, direction_index]
                starts = np.flatnonzero(active & ~np.r_[False, active[:-1]])
                key = f"{axis}_{direction}"
                summary = transition_summary[key]
                summary["episode_count"] += int(starts.size > 0)
                summary["transition_count"] += int(starts.size)
                for start in starts.tolist():
                    end = start
                    while end < action.shape[0] and active[end]:
                        end += 1
                    run_length = end - start
                    horizon_end = min(action.shape[0], start + int(persistence_horizon))
                    persistent_horizon = bool(np.all(active[start:horizon_end])) and (
                        horizon_end - start >= int(persistence_horizon)
                    )
                    persistent_10 = bool(
                        np.all(active[start : min(action.shape[0], start + 10)])
                        and action.shape[0] - start >= 10
                    )
                    sign = 1.0 if direction_index == 0 else -1.0
                    threshold = float(
                        thresholds[axis]["pos" if direction_index == 0 else "neg"]
                    )
                    onset_ratio = float(sign * action[start, axis_index] / threshold)
                    prior = action[max(0, start - int(pre_window)) : start, axis_index]
                    prior_ratio = (
                        float(np.max(sign * prior) / threshold)
                        if prior.size
                        else 0.0
                    )
                    delta_end = min(action.shape[0] - 1, start + 4)
                    qpos_delta_4 = float(qpos[delta_end, axis_index] - qpos[start, axis_index])
                    axis_active = positive | negative
                    pre_idle_start = int(start)
                    while pre_idle_start > 0 and not bool(axis_active[pre_idle_start - 1]):
                        pre_idle_start -= 1
                    pre_idle_run_length = int(start - pre_idle_start)
                    summary["run_lengths"].append(int(run_length))
                    summary["onset_action_ratio"].append(onset_ratio)
                    summary["pre_window_peak_ratio"].append(prior_ratio)
                    summary["onset_qpos"].append(float(qpos[start, axis_index]))
                    summary["onset_qvel"].append(float(qvel[start, axis_index]))
                    summary["onset_qpos_delta_4"].append(qpos_delta_4)
                    summary["pre_idle_run_length"].append(pre_idle_run_length)
                    transition_rows.append(
                        {
                            "episode_id": episode_id,
                            "split": "train" if episode_id in train_ids else "validation",
                            "axis": axis,
                            "direction": direction,
                            "step": int(start),
                            "run_length": int(run_length),
                            "persistent_horizon": int(persistent_horizon),
                            "persistent_10": int(persistent_10),
                            "onset_action_ratio": onset_ratio,
                            "pre_window_peak_ratio": prior_ratio,
                            "onset_qpos": float(qpos[start, axis_index]),
                            "onset_qvel": float(qvel[start, axis_index]),
                            "qpos_delta_4": qpos_delta_4,
                            "pre_idle_run_length": pre_idle_run_length,
                        }
                    )

        episode_split = "train" if episode_id in train_ids else "validation"
        for step in range(action.shape[0]):
            directions = _directions(effective[step])
            combo = directions or "idle"
            combo_counts[combo] += 1
            combo_counts_by_split[episode_split][combo] += 1
        episode_starts = [
            row for row in transition_rows if int(row["episode_id"]) == episode_id
        ]
        if episode_starts:
            first = min(episode_starts, key=lambda row: int(row["step"]))
            first_transition_rows.append(
                {
                    "episode_id": episode_id,
                    "split": episode_split,
                    "step": int(first["step"]),
                    "axis": str(first["axis"]),
                    "direction": str(first["direction"]),
                    "onset_action_ratio": float(first["onset_action_ratio"]),
                    "run_length": int(first["run_length"]),
                    "onset_qpos": float(first["onset_qpos"]),
                    "onset_qvel": float(first["onset_qvel"]),
                }
            )

    transition_summary = _summarize_transition_rows(transition_summary)
    ambiguity_summary = _summarize_ambiguity(ambiguity_counts)
    boundary_summary = _summarize_boundary_counts(boundary_counts)
    first_transition_summary = _summarize_first_transitions(first_transition_rows)
    return {
        "contract": "action_start_distribution_v1",
        "dataset_dir": str(dataset_path),
        "episode_ids": selected_ids,
        "train_episode_ids": train_ids,
        "heldout_episode_ids": sorted(FORBIDDEN_HELDOUT),
        "heldout_evaluated": False,
        "thresholds": {
            axis: {direction: float(thresholds[axis][direction]) for direction in ("pos", "neg")}
            for axis in AXIS_NAMES
        },
        "persistence_horizon": int(persistence_horizon),
        "pre_window": int(pre_window),
        "boundary_fraction": float(boundary_fraction),
        "ambiguity_bins": int(ambiguity_bins),
        "episodes": episodes,
        "state_counts": state_counts,
        "transition_summary": transition_summary,
        "combo_counts": dict(combo_counts),
        "combo_counts_by_split": {
            split: dict(counts) for split, counts in combo_counts_by_split.items()
        },
        "boundary_distribution": boundary_summary,
        "first_transition_summary": first_transition_summary,
        "first_transition_rows": first_transition_rows,
        "transition_rows": transition_rows,
        "low_dim_qpos_qvel_ambiguity": ambiguity_summary,
        "image_ambiguity_measured": False,
    }


def write_action_start_distribution_report(
    *,
    output_dir: str | Path,
    report: Mapping[str, Any],
    source_paths: Mapping[str, str | Path],
) -> Path:
    """Write JSON and reviewable transition/summary CSV artifacts."""

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    payload = dict(report)
    payload["source_paths"] = {
        key: str(Path(value).expanduser().resolve()) for key, value in source_paths.items()
    }
    payload["source_sha256"] = {key: sha256_file(value) for key, value in source_paths.items()}
    json_path = output / "action_start_distribution_report.json"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    fields = [
        "episode_id",
        "split",
        "axis",
        "direction",
        "step",
        "run_length",
        "persistent_horizon",
        "persistent_10",
        "onset_action_ratio",
        "pre_window_peak_ratio",
        "onset_qpos",
        "onset_qvel",
        "qpos_delta_4",
        "pre_idle_run_length",
    ]
    with (output / "transition_rows.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(report["transition_rows"])
    summary_fields = [
        "axis",
        "direction",
        "transition_count",
        "persistent_horizon_rate",
        "persistent_10_rate",
        "run_length_median",
        "run_length_p90",
        "onset_action_ratio_median",
        "onset_action_ratio_p10",
        "pre_window_peak_ratio_median",
        "qpos_delta_4_median",
        "pre_idle_run_length_median",
    ]
    with (output / "transition_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary_fields)
        writer.writeheader()
        for row in report["transition_summary"].values():
            writer.writerow({key: row.get(key) for key in summary_fields})
    first_fields = [
        "episode_id",
        "split",
        "step",
        "axis",
        "direction",
        "onset_action_ratio",
        "run_length",
        "onset_qpos",
        "onset_qvel",
    ]
    with (output / "first_transition_rows.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=first_fields)
        writer.writeheader()
        writer.writerows(report["first_transition_rows"])
    return json_path


def _quantile_edges(values: np.ndarray, *, bins: int) -> np.ndarray:
    quantiles = np.linspace(0.0, 1.0, int(bins) + 1)[1:-1]
    edges = np.quantile(values, quantiles, axis=0)
    return np.asarray(edges, dtype=np.float32)


def _summarize_transition_rows(summary: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for key, raw in summary.items():
        run_lengths = np.asarray(raw["run_lengths"], dtype=np.float64)
        onset_ratio = np.asarray(raw["onset_action_ratio"], dtype=np.float64)
        prior_ratio = np.asarray(raw["pre_window_peak_ratio"], dtype=np.float64)
        qpos_delta = np.asarray(raw["onset_qpos_delta_4"], dtype=np.float64)
        pre_idle = np.asarray(raw["pre_idle_run_length"], dtype=np.float64)
        count = int(raw["transition_count"])
        result[key] = {
            "axis": raw["axis"],
            "direction": raw["direction"],
            "episode_count": int(raw["episode_count"]),
            "transition_count": count,
            "persistent_horizon_count": int(
                sum(1 for value in raw["run_lengths"] if value >= 4)
            ),
            "persistent_10_count": int(
                sum(1 for value in raw["run_lengths"] if value >= 10)
            ),
            "persistent_horizon_rate": float(np.mean(run_lengths >= 4)) if count else 0.0,
            "persistent_10_rate": float(np.mean(run_lengths >= 10)) if count else 0.0,
            "run_length_median": _percentile(run_lengths, 50),
            "run_length_p90": _percentile(run_lengths, 90),
            "onset_action_ratio_median": _percentile(onset_ratio, 50),
            "onset_action_ratio_p10": _percentile(onset_ratio, 10),
            "pre_window_peak_ratio_median": _percentile(prior_ratio, 50),
            "qpos_delta_4_median": _percentile(qpos_delta, 50),
            "pre_idle_run_length_median": _percentile(pre_idle, 50),
        }
    return result


def _summarize_boundary_counts(
    counts: Mapping[str, Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for key, raw in counts.items():
        ratios = np.asarray(raw["ratios"], dtype=np.float64)
        total = int(raw["total_steps"])
        result[key] = {
            "axis": raw["axis"],
            "direction": raw["direction"],
            "total_steps": total,
            "positive_side_steps": int(raw["positive_side_steps"]),
            "effective_steps": int(raw["effective_steps"]),
            "positive_side_rate": (
                float(raw["positive_side_steps"] / total) if total else 0.0
            ),
            "effective_rate": float(raw["effective_steps"] / total) if total else 0.0,
            "near_idle_steps_0.9_to_1.0": int(raw["near_idle_steps_0.9_to_1.0"]),
            "near_effective_steps_1.0_to_1.1": int(
                raw["near_effective_steps_1.0_to_1.1"]
            ),
            "bands": dict(raw["bands"]),
            "ratio_quantiles": {
                str(percentile): _percentile(ratios, percentile)
                for percentile in (1, 10, 25, 50, 75, 90, 99)
            },
        }
    return result


def _summarize_first_transitions(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "episode_count": 0,
            "label_counts": {},
            "step_median": 0.0,
            "step_p10": 0.0,
            "step_p90": 0.0,
            "onset_action_ratio_median": 0.0,
            "onset_action_ratio_p10": 0.0,
            "run_length_median": 0.0,
            "by_split": {},
        }
    steps = np.asarray([float(row["step"]) for row in rows], dtype=np.float64)
    ratios = np.asarray(
        [float(row["onset_action_ratio"]) for row in rows], dtype=np.float64
    )
    runs = np.asarray([float(row["run_length"]) for row in rows], dtype=np.float64)
    labels = Counter(f"{row['axis']}_{row['direction']}" for row in rows)
    by_split: dict[str, Any] = {}
    for split in ("train", "validation"):
        selected = [row for row in rows if row["split"] == split]
        if not selected:
            continue
        split_steps = np.asarray([float(row["step"]) for row in selected], dtype=np.float64)
        split_ratios = np.asarray(
            [float(row["onset_action_ratio"]) for row in selected], dtype=np.float64
        )
        by_split[split] = {
            "episode_count": len(selected),
            "label_counts": dict(
                Counter(f"{row['axis']}_{row['direction']}" for row in selected)
            ),
            "step_median": _percentile(split_steps, 50),
            "step_p10": _percentile(split_steps, 10),
            "step_p90": _percentile(split_steps, 90),
            "onset_action_ratio_median": _percentile(split_ratios, 50),
            "onset_action_ratio_p10": _percentile(split_ratios, 10),
        }
    return {
        "episode_count": len(rows),
        "label_counts": dict(labels),
        "step_median": _percentile(steps, 50),
        "step_p10": _percentile(steps, 10),
        "step_p90": _percentile(steps, 90),
        "onset_action_ratio_median": _percentile(ratios, 50),
        "onset_action_ratio_p10": _percentile(ratios, 10),
        "run_length_median": _percentile(runs, 50),
        "by_split": by_split,
    }


def _boundary_band(value: float) -> str:
    if value < 0.0:
        return "lt_0"
    if value < 0.8:
        return "0_to_0.8"
    if value < 0.9:
        return "0.8_to_0.9"
    if value < 0.95:
        return "0.9_to_0.95"
    if value < 0.99:
        return "0.95_to_0.99"
    if value < 1.0:
        return "0.99_to_1.0"
    if value < 1.01:
        return "1.0_to_1.01"
    if value < 1.05:
        return "1.01_to_1.05"
    if value < 1.1:
        return "1.05_to_1.1"
    if value < 1.25:
        return "1.1_to_1.25"
    if value < 1.5:
        return "1.25_to_1.5"
    return "ge_1.5"


def _summarize_ambiguity(
    counts: Mapping[str, Mapping[tuple[int, int], Mapping[str, int]]]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for axis, bins in counts.items():
        rows = []
        total = 0
        weighted_entropy = 0.0
        for (qpos_bin, qvel_bin), raw_counts in sorted(bins.items()):
            values = np.asarray([raw_counts.get(state, 0) for state in STATE_NAMES], dtype=np.float64)
            count = int(values.sum())
            if count == 0:
                continue
            probabilities = values / count
            entropy = float(
                -sum(float(value) * np.log2(float(value)) for value in probabilities if value > 0.0)
            )
            majority = float(np.max(probabilities))
            total += count
            weighted_entropy += entropy * count
            rows.append(
                {
                    "qpos_bin": int(qpos_bin),
                    "qvel_bin": int(qvel_bin),
                    "count": count,
                    "entropy_bits": entropy,
                    "majority_share": majority,
                    "state_counts": {
                        state: int(raw_counts.get(state, 0)) for state in STATE_NAMES
                    },
                }
            )
        result[axis] = {
            "rows": rows,
            "weighted_entropy_bits": weighted_entropy / total if total else 0.0,
            "high_entropy_bin_count": int(sum(row["entropy_bits"] >= 1.0 for row in rows)),
        }
    return result


def _percentile(values: np.ndarray, percentile: float) -> float:
    return float(np.percentile(values, percentile)) if values.size else 0.0


def _directions(mask: np.ndarray) -> str:
    return ",".join(
        f"{AXIS_NAMES[axis]}{DIRECTION_NAMES[direction][0]}"
        for axis, direction in np.argwhere(mask)
    )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "analyze_action_start_distribution",
    "sha256_file",
    "write_action_start_distribution_report",
]
