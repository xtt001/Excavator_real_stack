#!/usr/bin/env python3
"""Build a teacher-forced Level 1 effective-intent integral report."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from testbed.policies.offline_eval import AXIS_NAMES
from testbed.policies.trajectory_support_eval import (
    DIRECTION_NAMES,
    compute_intent_horizon_rows,
    cumulative_intent,
    effective_action_channels,
)


CONTROL_NAMES = (
    "expert",
    "zero",
    "expert_delay1",
    "expert_delay5",
    "expert_sign_flipped",
    "expert_axis_shuffled",
    "expert_scale_0.5",
    "expert_scale_1.5",
)

EPISODE_METRICS = (
    "channel_l1_error",
    "cumulative_path_mean_channel_l1",
    "cumulative_path_max_channel_l1",
    "missing_expert_impulse",
    "extra_policy_impulse",
    "net_axis_l1_error",
    "direction_cosine",
    "magnitude_ratio",
    "policy_cancellation_ratio",
    "policy_stick_impulse",
)

PAIRED_METRIC_PREFERENCE = {
    "channel_l1_error": "lower",
    "cumulative_path_mean_channel_l1": "lower",
    "missing_expert_impulse": "lower",
    "extra_policy_impulse": "lower",
    "net_axis_l1_error": "lower",
    "direction_cosine": "higher",
    "policy_cancellation_ratio": "lower",
    "policy_stick_impulse": "lower",
}


@dataclass(frozen=True)
class EvalSpec:
    model: str
    eval_dir: Path


@dataclass(frozen=True)
class EpisodeActions:
    time_s: np.ndarray
    expert_action: np.ndarray
    policy_action: np.ndarray
    dt: float


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval",
        dest="eval_specs",
        action="append",
        required=True,
        help="Replay action directory in MODEL=DIR form. Repeat for each candidate stage.",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--deadzone-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--horizons", default="5,10,20,40")
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260710)
    args = parser.parse_args()

    paths = run_level1_report(
        eval_specs=[parse_eval_spec(value) for value in args.eval_specs],
        manifest_path=args.manifest,
        deadzone_path=args.deadzone_json,
        output_dir=args.output_dir,
        horizons=_parse_positive_ints(args.horizons, name="horizons"),
        stride=int(args.stride),
        bootstrap_samples=int(args.bootstrap_samples),
        bootstrap_seed=int(args.bootstrap_seed),
        argv=list(sys.argv),
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


def parse_eval_spec(value: str) -> EvalSpec:
    if "=" not in value:
        raise ValueError(f"--eval must use MODEL=DIR form, got {value!r}")
    model, raw_path = value.split("=", 1)
    model = model.strip()
    raw_path = raw_path.strip()
    if not model or not raw_path:
        raise ValueError(f"--eval must use non-empty MODEL=DIR values, got {value!r}")
    return EvalSpec(model=model, eval_dir=Path(raw_path).expanduser())


def build_control_actions(expert_action: np.ndarray) -> dict[str, np.ndarray]:
    expert = _validate_actions(expert_action, name="expert_action")
    return {
        "expert": expert.copy(),
        "zero": np.zeros_like(expert),
        "expert_delay1": _delay_actions(expert, 1),
        "expert_delay5": _delay_actions(expert, 5),
        "expert_sign_flipped": -expert,
        "expert_axis_shuffled": expert[:, [1, 2, 3, 0]].copy(),
        "expert_scale_0.5": expert * 0.5,
        "expert_scale_1.5": expert * 1.5,
    }


def run_level1_report(
    *,
    eval_specs: list[EvalSpec],
    manifest_path: Path,
    deadzone_path: Path,
    output_dir: Path,
    horizons: tuple[int, ...],
    stride: int,
    bootstrap_samples: int,
    bootstrap_seed: int,
    argv: list[str],
) -> dict[str, Path]:
    if not eval_specs:
        raise ValueError("at least one eval spec is required")
    candidate_names = [spec.model for spec in eval_specs]
    if len(set(candidate_names)) != len(candidate_names):
        raise ValueError(f"candidate model names must be unique, got {candidate_names}")
    collisions = sorted(set(candidate_names) & set(CONTROL_NAMES))
    if collisions:
        raise ValueError(f"candidate model names collide with required controls: {collisions}")
    if not horizons or any(int(value) <= 0 for value in horizons):
        raise ValueError("horizons must contain positive integers")
    if stride <= 0:
        raise ValueError("stride must be positive")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")

    episode_ids = _load_episode_ids(manifest_path)
    thresholds = _load_deadzone_thresholds(deadzone_path)
    canonical, candidate_actions = _load_and_validate_candidates(eval_specs, episode_ids)
    model_order = [*CONTROL_NAMES, *candidate_names]

    episode_rows: list[dict[str, Any]] = []
    full_episode_rows: list[dict[str, Any]] = []
    observed_dts: list[float] = []
    total_steps = 0
    for episode_id in episode_ids:
        source = canonical[episode_id]
        observed_dts.append(source.dt)
        total_steps += int(source.expert_action.shape[0])
        actions_by_model = {
            **build_control_actions(source.expert_action),
            **{model: candidate_actions[model][episode_id] for model in candidate_names},
        }
        for model in model_order:
            policy = actions_by_model[model]
            horizon_rows = compute_intent_horizon_rows(
                source.expert_action,
                policy,
                thresholds,
                dt=source.dt,
                horizons=horizons,
                stride=stride,
            )
            episode_rows.extend(
                _summarize_episode_horizons(
                    model=model,
                    episode_id=episode_id,
                    dt=source.dt,
                    rows=horizon_rows,
                )
            )
            full_episode_rows.append(
                _full_episode_intent_row(
                    model=model,
                    episode_id=episode_id,
                    expert_action=source.expert_action,
                    policy_action=policy,
                    thresholds=thresholds,
                    dt=source.dt,
                )
            )

    aggregate_rows = aggregate_episode_rows(
        episode_rows,
        model_order=model_order,
        horizons=horizons,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    baseline_model = candidate_names[0]
    paired_rows = build_paired_comparisons(
        episode_rows,
        baseline_model=baseline_model,
        candidate_models=candidate_names[1:],
        horizons=horizons,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    plot_dir = output / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    generated_at = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    git_commit = _git_commit()

    paths = {
        "run_manifest": output / "run_manifest.json",
        "data_split": output / "data_split.json",
        "intent_integral_by_episode": output / "intent_integral_by_episode.csv",
        "intent_integral_aggregate": output / "intent_integral_aggregate.csv",
        "candidate_comparison_to_baseline": output / "candidate_comparison_to_baseline.csv",
        "full_episode_intent_by_episode": output / "full_episode_intent_by_episode.csv",
        "summary": output / "summary.json",
        "cumulative_intent_by_axis_plot": plot_dir / "cumulative_intent_by_axis.png",
        "intent_error_by_horizon_plot": plot_dir / "intent_error_by_horizon.png",
    }
    _write_json(
        paths["run_manifest"],
        {
            "schema_version": 1,
            "generated_at": generated_at,
            "git_commit": git_commit,
            "argv": list(argv),
            "manifest": str(Path(manifest_path).resolve()),
            "deadzone_json": str(Path(deadzone_path).resolve()),
            "evals": [
                {"model": spec.model, "eval_dir": str(spec.eval_dir.resolve())}
                for spec in eval_specs
            ],
            "controls": list(CONTROL_NAMES),
            "horizons_steps": [int(value) for value in horizons],
            "stride_steps": int(stride),
            "bootstrap_samples": int(bootstrap_samples),
            "bootstrap_seed": int(bootstrap_seed),
            "output_dir": str(output.resolve()),
        },
    )
    _write_json(
        paths["data_split"],
        {
            "schema_version": 1,
            "selection": "manifest_train_ready_episode_ids",
            "episode_ids": episode_ids,
            "episode_count": len(episode_ids),
            "role": "teacher_forced_level1_evaluation",
            "policy_training_overlap": "not_inferred_by_this_evaluator",
            "threshold_tuning": "none_in_this_report",
            "final_test_claim": False,
        },
    )
    _write_csv(paths["intent_integral_by_episode"], episode_rows)
    _write_csv(paths["intent_integral_aggregate"], aggregate_rows)
    _write_csv(paths["candidate_comparison_to_baseline"], paired_rows)
    _write_csv(paths["full_episode_intent_by_episode"], full_episode_rows)
    _plot_intent_error_by_horizon(aggregate_rows, model_order, paths["intent_error_by_horizon_plot"])
    _plot_cumulative_intent_by_axis(full_episode_rows, model_order, paths["cumulative_intent_by_axis_plot"])
    _write_json(
        paths["summary"],
        {
            "schema_version": 1,
            "generated_at": generated_at,
            "claim_boundary": "teacher_forced_level1_only",
            "git_commit": git_commit,
            "episodes": len(episode_ids),
            "steps": total_steps,
            "models": model_order,
            "dt_values": sorted({float(value) for value in observed_dts}),
            "horizons_steps": [int(value) for value in horizons],
            "stride_steps": int(stride),
            "stick_contract": "expected_zero_action; policy stick impulse is leakage",
            "aggregate": aggregate_rows,
            "paired_comparison_to_baseline": paired_rows,
            "artifacts": {name: str(path.resolve()) for name, path in paths.items() if name != "summary"},
        },
    )
    return paths


def aggregate_episode_rows(
    rows: list[dict[str, Any]],
    *,
    model_order: list[str],
    horizons: tuple[int, ...],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(int(bootstrap_seed))
    output: list[dict[str, Any]] = []
    for model in model_order:
        for horizon in horizons:
            group = [
                row
                for row in rows
                if row["model"] == model and int(row["horizon_steps"]) == int(horizon)
            ]
            if not group:
                continue
            aggregate: dict[str, Any] = {
                "model": model,
                "horizon_steps": int(horizon),
                "horizon_seconds": float(group[0]["horizon_seconds"]),
                "episodes": len(group),
                "windows": int(sum(int(row["windows"]) for row in group)),
            }
            for metric in EPISODE_METRICS:
                values = np.asarray(
                    [float(row[metric]) for row in group if row.get(metric) is not None],
                    dtype=np.float64,
                )
                if values.size == 0:
                    aggregate[f"{metric}_mean"] = None
                    aggregate[f"{metric}_ci95_low"] = None
                    aggregate[f"{metric}_ci95_high"] = None
                    continue
                bootstrap_means = values[
                    rng.integers(0, values.size, size=(int(bootstrap_samples), values.size))
                ].mean(axis=1)
                aggregate[f"{metric}_mean"] = float(values.mean())
                aggregate[f"{metric}_ci95_low"] = float(np.percentile(bootstrap_means, 2.5))
                aggregate[f"{metric}_ci95_high"] = float(np.percentile(bootstrap_means, 97.5))
            output.append(aggregate)
    return output


def build_paired_comparisons(
    rows: list[dict[str, Any]],
    *,
    baseline_model: str,
    candidate_models: list[str],
    horizons: tuple[int, ...],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(int(bootstrap_seed) + 1)
    output: list[dict[str, Any]] = []
    for candidate_model in candidate_models:
        for horizon in horizons:
            baseline_by_episode = {
                str(row["episode_id"]): row
                for row in rows
                if row["model"] == baseline_model and int(row["horizon_steps"]) == int(horizon)
            }
            candidate_by_episode = {
                str(row["episode_id"]): row
                for row in rows
                if row["model"] == candidate_model and int(row["horizon_steps"]) == int(horizon)
            }
            if set(baseline_by_episode) != set(candidate_by_episode):
                raise ValueError(
                    f"paired episode mismatch for {baseline_model} vs {candidate_model} at horizon {horizon}"
                )
            episode_ids = sorted(baseline_by_episode)
            for metric, preference in PAIRED_METRIC_PREFERENCE.items():
                pairs = [
                    (
                        baseline_by_episode[episode_id].get(metric),
                        candidate_by_episode[episode_id].get(metric),
                    )
                    for episode_id in episode_ids
                ]
                valid_pairs = [
                    (float(baseline), float(candidate))
                    for baseline, candidate in pairs
                    if baseline is not None and candidate is not None
                ]
                if not valid_pairs:
                    output.append(
                        {
                            "baseline_model": baseline_model,
                            "candidate_model": candidate_model,
                            "horizon_steps": int(horizon),
                            "horizon_seconds": float(horizon * float(rows[0]["dt"])),
                            "metric": metric,
                            "preference": preference,
                            "episodes": 0,
                            "baseline_mean": None,
                            "candidate_mean": None,
                            "delta_mean": None,
                            "delta_ci95_low": None,
                            "delta_ci95_high": None,
                            "relative_delta_pct": None,
                            "episodes_improved_rate": None,
                        }
                    )
                    continue
                values = np.asarray(valid_pairs, dtype=np.float64)
                baseline_values = values[:, 0]
                candidate_values = values[:, 1]
                deltas = candidate_values - baseline_values
                bootstrap_means = deltas[
                    rng.integers(0, deltas.size, size=(int(bootstrap_samples), deltas.size))
                ].mean(axis=1)
                baseline_mean = float(baseline_values.mean())
                improved = deltas < 0.0 if preference == "lower" else deltas > 0.0
                output.append(
                    {
                        "baseline_model": baseline_model,
                        "candidate_model": candidate_model,
                        "horizon_steps": int(horizon),
                        "horizon_seconds": float(horizon * float(rows[0]["dt"])),
                        "metric": metric,
                        "preference": preference,
                        "episodes": int(deltas.size),
                        "baseline_mean": baseline_mean,
                        "candidate_mean": float(candidate_values.mean()),
                        "delta_mean": float(deltas.mean()),
                        "delta_ci95_low": float(np.percentile(bootstrap_means, 2.5)),
                        "delta_ci95_high": float(np.percentile(bootstrap_means, 97.5)),
                        "relative_delta_pct": (
                            float(100.0 * deltas.mean() / abs(baseline_mean))
                            if abs(baseline_mean) > 1.0e-12
                            else None
                        ),
                        "episodes_improved_rate": float(np.mean(improved)),
                    }
                )
    return output


def _load_and_validate_candidates(
    specs: list[EvalSpec],
    episode_ids: list[str],
) -> tuple[dict[str, EpisodeActions], dict[str, dict[str, np.ndarray]]]:
    canonical: dict[str, EpisodeActions] = {}
    candidates: dict[str, dict[str, np.ndarray]] = {}
    for spec_index, spec in enumerate(specs):
        model_actions: dict[str, np.ndarray] = {}
        for episode_id in episode_ids:
            loaded = _load_episode_actions(spec.eval_dir, episode_id)
            if spec_index == 0:
                canonical[episode_id] = loaded
            else:
                expected = canonical[episode_id]
                if not np.array_equal(loaded.expert_action, expected.expert_action):
                    raise ValueError(f"expert_action mismatch for {spec.model}/{episode_id}")
                if not np.array_equal(loaded.time_s, expected.time_s):
                    raise ValueError(f"time_s mismatch for {spec.model}/{episode_id}")
            model_actions[episode_id] = loaded.policy_action
        candidates[spec.model] = model_actions
    return canonical, candidates


def _load_episode_actions(eval_dir: Path, episode_id: str) -> EpisodeActions:
    path = Path(eval_dir) / "episodes" / episode_id / "actions.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path) as data:
        required = {"time_s", "expert_action", "policy_action"}
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(f"{path} is missing arrays: {missing}")
        time_s = np.asarray(data["time_s"], dtype=np.float64)
        expert = _validate_actions(data["expert_action"], name=f"{path}:expert_action")
        policy = _validate_actions(data["policy_action"], name=f"{path}:policy_action")
    if expert.shape != policy.shape or time_s.shape != (expert.shape[0],):
        raise ValueError(
            f"aligned shapes required in {path}, got time={time_s.shape}, expert={expert.shape}, policy={policy.shape}"
        )
    diffs = np.diff(time_s)
    if diffs.size == 0 or not np.all(np.isfinite(diffs)) or np.any(diffs <= 0.0):
        raise ValueError(f"{path} must contain increasing timestamps with at least two steps")
    dt_value = float(np.median(diffs))
    if not np.allclose(diffs, dt_value, rtol=0.0, atol=1.0e-9):
        raise ValueError(f"{path} has non-uniform timestamps")
    return EpisodeActions(time_s=time_s, expert_action=expert, policy_action=policy, dt=dt_value)


def _summarize_episode_horizons(
    *,
    model: str,
    episode_id: str,
    dt: float,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    horizons = sorted({int(row["horizon_steps"]) for row in rows})
    output = []
    for horizon in horizons:
        group = [row for row in rows if int(row["horizon_steps"]) == horizon]
        if not group:
            continue
        summarized: dict[str, Any] = {
            "model": model,
            "episode_id": episode_id,
            "dt": float(dt),
            "horizon_steps": horizon,
            "horizon_seconds": float(horizon * dt),
            "windows": len(group),
            "channel_l1_error": _mean_required(group, "channel_l1_error"),
            "cumulative_path_mean_channel_l1": _mean_required(
                group, "cumulative_path_mean_channel_l1"
            ),
            "cumulative_path_max_channel_l1": _mean_required(
                group, "cumulative_path_max_channel_l1"
            ),
            "missing_expert_impulse": _mean_required(group, "missing_expert_impulse"),
            "extra_policy_impulse": _mean_required(group, "extra_policy_impulse"),
            "net_axis_l1_error": _mean_required(group, "net_axis_l1_error"),
            "direction_cosine": _mean_optional(group, "direction_cosine"),
            "magnitude_ratio": _mean_optional(group, "magnitude_ratio"),
            "policy_cancellation_ratio": _mean_required(group, "policy_cancellation_ratio"),
            "policy_stick_impulse": float(
                np.mean(
                    [
                        float(row["policy_stick_pos_impulse"])
                        + float(row["policy_stick_neg_impulse"])
                        for row in group
                    ]
                )
            ),
        }
        output.append(summarized)
    return output


def _full_episode_intent_row(
    *,
    model: str,
    episode_id: str,
    expert_action: np.ndarray,
    policy_action: np.ndarray,
    thresholds: dict[str, dict[str, float]],
    dt: float,
) -> dict[str, Any]:
    expert = cumulative_intent(effective_action_channels(expert_action, thresholds), dt=dt)[-1]
    policy = cumulative_intent(effective_action_channels(policy_action, thresholds), dt=dt)[-1]
    row: dict[str, Any] = {
        "model": model,
        "episode_id": episode_id,
        "steps": int(expert_action.shape[0]),
        "duration_seconds": float(expert_action.shape[0] * dt),
    }
    for axis_idx, axis in enumerate(AXIS_NAMES):
        for direction_idx, direction in enumerate(DIRECTION_NAMES):
            row[f"expert_{axis}_{direction}_impulse"] = float(expert[axis_idx, direction_idx])
            row[f"policy_{axis}_{direction}_impulse"] = float(policy[axis_idx, direction_idx])
    return row


def _plot_intent_error_by_horizon(
    rows: list[dict[str, Any]],
    model_order: list[str],
    path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(17, 7), sharex=True)
    candidate_models = [model for model in model_order if model not in CONTROL_NAMES]
    for ax, selected, title in (
        (axes[0], model_order, "All controls and candidates"),
        (axes[1], candidate_models, "Candidate stages (zoomed)"),
    ):
        for model in selected:
            group = [row for row in rows if row["model"] == model]
            if not group:
                continue
            x = np.asarray([row["horizon_seconds"] for row in group], dtype=np.float64)
            y = np.asarray([row["channel_l1_error_mean"] for row in group], dtype=np.float64)
            low = np.asarray([row["channel_l1_error_ci95_low"] for row in group], dtype=np.float64)
            high = np.asarray([row["channel_l1_error_ci95_high"] for row in group], dtype=np.float64)
            line = ax.plot(x, y, marker="o", linewidth=1.2, label=model)[0]
            ax.fill_between(x, low, high, color=line.get_color(), alpha=0.10)
        ax.set_xlabel("Horizon (s)")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)
    axes[0].set_ylabel("Mean effective-intent channel L1")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_cumulative_intent_by_axis(
    rows: list[dict[str, Any]],
    model_order: list[str],
    path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    channel_labels = [f"{axis}_{direction}" for axis in AXIS_NAMES for direction in DIRECTION_NAMES]
    matrix = np.zeros((len(model_order), len(channel_labels)), dtype=np.float64)
    for model_idx, model in enumerate(model_order):
        group = [row for row in rows if row["model"] == model]
        for channel_idx, label in enumerate(channel_labels):
            matrix[model_idx, channel_idx] = float(
                np.mean([float(row[f"policy_{label}_impulse"]) for row in group])
            )
    fig, ax = plt.subplots(figsize=(12, max(6, 0.42 * len(model_order))))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(channel_labels)))
    ax.set_xticklabels(channel_labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(model_order)))
    ax.set_yticklabels(model_order)
    ax.set_title("Mean full-episode effective intent impulse")
    fig.colorbar(image, ax=ax, label="Impulse (effective command * s)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _load_episode_ids(path: Path) -> list[str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    values = payload.get("train_ready_episode_ids")
    if not isinstance(values, list) or not values:
        raise ValueError(f"manifest must contain non-empty train_ready_episode_ids: {path}")
    episode_ids = [str(value) for value in values]
    if len(set(episode_ids)) != len(episode_ids):
        raise ValueError(f"manifest contains duplicate episode ids: {path}")
    return episode_ids


def _load_deadzone_thresholds(path: Path) -> dict[str, dict[str, float]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    raw = payload.get("deadzone_action", payload)
    if not isinstance(raw, dict):
        raise ValueError(f"deadzone artifact must contain a mapping: {path}")
    return raw


def _validate_actions(values: np.ndarray, *, name: str) -> np.ndarray:
    actions = np.asarray(values, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != len(AXIS_NAMES):
        raise ValueError(f"{name} must have shape (T, {len(AXIS_NAMES)}), got {actions.shape}")
    if not np.all(np.isfinite(actions)):
        raise ValueError(f"{name} must contain finite values")
    return actions


def _delay_actions(actions: np.ndarray, steps: int) -> np.ndarray:
    delayed = np.zeros_like(actions)
    if steps < actions.shape[0]:
        delayed[steps:] = actions[:-steps]
    return delayed


def _mean_required(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def _mean_optional(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(np.mean(values)) if values else None


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _parse_positive_ints(value: str, *, name: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in str(value).split(",") if item.strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be comma-separated positive integers") from exc
    if not values or any(item <= 0 for item in values):
        raise ValueError(f"{name} must contain positive integers")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicates")
    return values


def _git_commit() -> str | None:
    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


if __name__ == "__main__":
    main()
