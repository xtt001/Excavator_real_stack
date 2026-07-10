#!/usr/bin/env python3
"""Evaluate held-out short-horizon qpos transition baselines."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from testbed.policies.offline_eval import AXIS_NAMES
from testbed.policies.trajectory_transition_eval import TransitionSamples, build_transition_samples


MODEL_ORDER = ("constant_state", "initial_qvel", "action_linear", "qvel_action_linear")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--deadzone-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--horizons", default="5,10,20,40")
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--qvel-to-qpos-signs", required=True)
    parser.add_argument("--action-to-qpos-signs", required=True)
    parser.add_argument("--inactive-axes", default="stick")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260710)
    args = parser.parse_args()

    paths = run_probe(
        dataset_dir=args.dataset_dir,
        manifest_path=args.manifest,
        deadzone_path=args.deadzone_json,
        output_dir=args.output_dir,
        horizons=_parse_positive_ints(args.horizons),
        stride=int(args.stride),
        qvel_to_qpos_sign=_parse_signs(args.qvel_to_qpos_signs),
        action_to_qpos_sign=_parse_signs(args.action_to_qpos_signs),
        inactive_axes=_parse_axes(args.inactive_axes),
        bootstrap_samples=int(args.bootstrap_samples),
        bootstrap_seed=int(args.bootstrap_seed),
        argv=list(sys.argv),
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


def run_probe(
    *,
    dataset_dir: Path,
    manifest_path: Path,
    deadzone_path: Path,
    output_dir: Path,
    horizons: tuple[int, ...],
    stride: int,
    qvel_to_qpos_sign: np.ndarray,
    action_to_qpos_sign: np.ndarray,
    inactive_axes: tuple[str, ...],
    bootstrap_samples: int,
    bootstrap_seed: int,
    argv: list[str],
) -> dict[str, Path]:
    if stride <= 0:
        raise ValueError("stride must be positive")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    episode_ids = _load_episode_ids(manifest_path)
    thresholds = _load_thresholds(deadzone_path)
    episodes, dt_values = _load_episodes(dataset_dir, episode_ids)
    if len(dt_values) != 1:
        raise ValueError(f"all episodes must share one dt, got {sorted(dt_values)}")
    sample_period = next(iter(dt_values))
    samples = {
        (episode_id, horizon): build_transition_samples(
            qpos=episodes[episode_id]["qpos"],
            qvel=episodes[episode_id]["qvel"],
            action=episodes[episode_id]["action"],
            thresholds=thresholds,
            dt=sample_period,
            horizon_steps=horizon,
            stride=stride,
            qvel_to_qpos_sign=qvel_to_qpos_sign,
            action_to_qpos_sign=action_to_qpos_sign,
        )
        for episode_id in episode_ids
        for horizon in horizons
    }
    correlation_rows = _qvel_qpos_correlation_rows(
        episodes,
        qvel_to_qpos_sign=qvel_to_qpos_sign,
        dt=sample_period,
    )
    episode_rows = _loeo_rows(
        samples,
        episode_ids=episode_ids,
        horizons=horizons,
        inactive_axes=set(inactive_axes),
        sample_period=sample_period,
    )
    aggregate_rows = _aggregate_rows(
        episode_rows,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )

    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    plot_dir = output / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "run_manifest": output / "run_manifest.json",
        "state_contract": output / "state_contract.json",
        "transition_baseline_by_episode": output / "transition_baseline_by_episode.csv",
        "transition_baseline_aggregate": output / "transition_baseline_aggregate.csv",
        "summary": output / "summary.json",
        "transition_mae_plot": plot_dir / "transition_mae_by_horizon.png",
    }
    generated_at = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    git_commit = _git_commit()
    _write_json(
        paths["run_manifest"],
        {
            "schema_version": 1,
            "generated_at": generated_at,
            "git_commit": git_commit,
            "argv": argv,
            "dataset_dir": str(Path(dataset_dir).resolve()),
            "manifest": str(Path(manifest_path).resolve()),
            "deadzone_json": str(Path(deadzone_path).resolve()),
            "horizons_steps": list(horizons),
            "stride_steps": stride,
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": bootstrap_seed,
        },
    )
    _write_json(
        paths["state_contract"],
        {
            "schema_version": 1,
            "target": "future_qpos_delta",
            "swing_delta": "shortest_angle",
            "qvel_to_qpos_sign": dict(zip(AXIS_NAMES, qvel_to_qpos_sign.tolist())),
            "action_to_qpos_sign": dict(zip(AXIS_NAMES, action_to_qpos_sign.tolist())),
            "inactive_axes": list(inactive_axes),
            "inactive_axis_rule": "constant-state is the required baseline; no action model is fitted",
            "qvel_source_boundary": "gyro-derived qvel is mapped explicitly; it is not assumed to equal qpos finite difference",
            "correlation": correlation_rows,
        },
    )
    _write_csv(paths["transition_baseline_by_episode"], episode_rows)
    _write_csv(paths["transition_baseline_aggregate"], aggregate_rows)
    _plot_aggregate(aggregate_rows, paths["transition_mae_plot"])
    _write_json(
        paths["summary"],
        {
            "schema_version": 1,
            "generated_at": generated_at,
            "claim_boundary": "held_out_expert_transition_probe_only",
            "git_commit": git_commit,
            "episodes": len(episode_ids),
            "sample_period_s": sample_period,
            "inactive_axes": list(inactive_axes),
            "correlation": correlation_rows,
            "aggregate": aggregate_rows,
            "artifacts": {name: str(path.resolve()) for name, path in paths.items() if name != "summary"},
        },
    )
    return paths


def _loeo_rows(
    samples: dict[tuple[str, int], TransitionSamples],
    *,
    episode_ids: list[str],
    horizons: tuple[int, ...],
    inactive_axes: set[str],
    sample_period: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for horizon in horizons:
        for heldout in episode_ids:
            train = [samples[(episode_id, horizon)] for episode_id in episode_ids if episode_id != heldout]
            test = samples[(heldout, horizon)]
            train_target = np.concatenate([item.target_qpos_delta for item in train], axis=0)
            train_qvel = np.concatenate([item.initial_qvel_displacement for item in train], axis=0)
            train_action = np.concatenate([item.action_impulse for item in train], axis=0)
            for axis_idx, axis in enumerate(AXIS_NAMES):
                predictions = {
                    "constant_state": np.zeros(test.target_qpos_delta.shape[0], dtype=np.float64),
                    "initial_qvel": test.initial_qvel_displacement[:, axis_idx],
                }
                if axis not in inactive_axes:
                    predictions["action_linear"] = _fit_predict(
                        train_action[:, [axis_idx]],
                        train_target[:, axis_idx],
                        test.action_impulse[:, [axis_idx]],
                    )
                    predictions["qvel_action_linear"] = _fit_predict(
                        np.column_stack([train_qvel[:, axis_idx], train_action[:, axis_idx]]),
                        train_target[:, axis_idx],
                        np.column_stack(
                            [test.initial_qvel_displacement[:, axis_idx], test.action_impulse[:, axis_idx]]
                        ),
                    )
                target = test.target_qpos_delta[:, axis_idx]
                for model in MODEL_ORDER:
                    if model not in predictions:
                        continue
                    error = predictions[model] - target
                    rows.append(
                        {
                            "episode_id": heldout,
                            "dt": float(sample_period),
                            "horizon_steps": horizon,
                            "axis": axis,
                            "axis_status": "inactive_invariant" if axis in inactive_axes else "task_active",
                            "model": model,
                            "samples": int(target.size),
                            "mae": float(np.mean(np.abs(error))),
                            "rmse": float(np.sqrt(np.mean(np.square(error)))),
                            "target_mean_abs": float(np.mean(np.abs(target))),
                            "target_p95_abs": float(np.percentile(np.abs(target), 95)),
                        }
                    )
    return rows


def _fit_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(train_x.shape[0]), np.asarray(train_x, dtype=np.float64)])
    coefficients = np.linalg.lstsq(design, np.asarray(train_y, dtype=np.float64), rcond=None)[0]
    test_design = np.column_stack([np.ones(test_x.shape[0]), np.asarray(test_x, dtype=np.float64)])
    return test_design @ coefficients


def _aggregate_rows(
    rows: list[dict[str, Any]],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(bootstrap_seed)
    output = []
    keys = sorted({(int(row["horizon_steps"]), str(row["axis"]), str(row["model"])) for row in rows})
    for horizon, axis, model in keys:
        group = [
            row
            for row in rows
            if int(row["horizon_steps"]) == horizon and row["axis"] == axis and row["model"] == model
        ]
        values = np.asarray([float(row["mae"]) for row in group], dtype=np.float64)
        bootstrap = values[
            rng.integers(0, values.size, size=(bootstrap_samples, values.size))
        ].mean(axis=1)
        zero_by_episode = {
            row["episode_id"]: float(row["mae"])
            for row in rows
            if int(row["horizon_steps"]) == horizon
            and row["axis"] == axis
            and row["model"] == "constant_state"
        }
        zero = np.asarray([zero_by_episode[row["episode_id"]] for row in group], dtype=np.float64)
        zero_mean = float(zero.mean())
        output.append(
            {
                "horizon_steps": horizon,
                "horizon_seconds": float(horizon * float(group[0].get("dt", 0.05))),
                "axis": axis,
                "axis_status": group[0]["axis_status"],
                "model": model,
                "episodes": len(group),
                "samples": int(sum(int(row["samples"]) for row in group)),
                "mae_mean": float(values.mean()),
                "mae_ci95_low": float(np.percentile(bootstrap, 2.5)),
                "mae_ci95_high": float(np.percentile(bootstrap, 97.5)),
                "relative_vs_constant_state_pct": (
                    float(100.0 * (values.mean() - zero_mean) / zero_mean) if zero_mean > 1.0e-12 else None
                ),
                "episodes_better_than_constant_state": int(np.count_nonzero(values < zero)),
            }
        )
    return output


def _qvel_qpos_correlation_rows(
    episodes: dict[str, dict[str, Any]],
    *,
    qvel_to_qpos_sign: np.ndarray,
    dt: float,
) -> list[dict[str, Any]]:
    rows = []
    for axis_idx, axis in enumerate(AXIS_NAMES):
        episode_values = []
        all_delta = []
        all_qvel = []
        for episode in episodes.values():
            qpos = episode["qpos"]
            qvel = episode["qvel"]
            delta = np.diff(qpos[:, axis_idx])
            if axis_idx == 0:
                delta = (delta + np.pi) % (2.0 * np.pi) - np.pi
            delta = delta / dt
            mapped = qvel[:-1, axis_idx] * qvel_to_qpos_sign[axis_idx]
            if np.std(delta) > 1.0e-12 and np.std(mapped) > 1.0e-12:
                episode_values.append(float(np.corrcoef(delta, mapped)[0, 1]))
            all_delta.append(delta)
            all_qvel.append(mapped)
        delta_all = np.concatenate(all_delta)
        qvel_all = np.concatenate(all_qvel)
        global_correlation = (
            float(np.corrcoef(delta_all, qvel_all)[0, 1])
            if np.std(delta_all) > 1.0e-12 and np.std(qvel_all) > 1.0e-12
            else None
        )
        rows.append(
            {
                "axis": axis,
                "mapped_sign": float(qvel_to_qpos_sign[axis_idx]),
                "global_correlation": global_correlation,
                "episode_correlation_median": float(np.median(episode_values)) if episode_values else None,
                "episode_correlation_p10": float(np.percentile(episode_values, 10)) if episode_values else None,
                "episode_correlation_p90": float(np.percentile(episode_values, 90)) if episode_values else None,
                "positive_episode_correlations": int(np.count_nonzero(np.asarray(episode_values) > 0.0)),
                "episodes": len(episode_values),
            }
        )
    return rows


def _load_episodes(
    dataset_dir: Path,
    episode_ids: list[str],
) -> tuple[dict[str, dict[str, Any]], set[float]]:
    episodes = {}
    dts = set()
    for episode_id in episode_ids:
        path = Path(dataset_dir) / f"{episode_id}.hdf5"
        if not path.exists():
            raise FileNotFoundError(path)
        with h5py.File(path, "r") as handle:
            action = np.asarray(handle["action"], dtype=np.float64)
            qpos = np.asarray(handle["observations/qpos"], dtype=np.float64)
            qvel = np.asarray(handle["observations/qvel"], dtype=np.float64)
            metadata = handle.get("metadata")
            sample_dt = float(metadata.attrs["dt"]) if metadata is not None and "dt" in metadata.attrs else 0.05
        if action.shape != qpos.shape or action.shape != qvel.shape:
            raise ValueError(f"aligned (T, 4) arrays required in {path}")
        episodes[episode_id] = {"action": action, "qpos": qpos, "qvel": qvel}
        dts.add(sample_dt)
    return episodes, dts


def _plot_aggregate(rows: list[dict[str, Any]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for axis_name, ax in zip(AXIS_NAMES, axes.reshape(-1)):
        for model in MODEL_ORDER:
            group = [row for row in rows if row["axis"] == axis_name and row["model"] == model]
            if not group:
                continue
            x = [row["horizon_seconds"] for row in group]
            y = [row["mae_mean"] for row in group]
            ax.plot(x, y, marker="o", label=model)
        ax.set_title(axis_name)
        ax.set_xlabel("Horizon (s)")
        ax.set_ylabel("Held-out qpos delta MAE (rad)")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
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


def _load_thresholds(path: Path) -> dict[str, dict[str, float]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload.get("deadzone_action", payload)


def _parse_positive_ints(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values or any(item <= 0 for item in values) or len(set(values)) != len(values):
        raise ValueError("horizons must contain unique positive integers")
    return values


def _parse_signs(value: str) -> np.ndarray:
    values = np.asarray([float(item.strip()) for item in value.split(",") if item.strip()], dtype=np.float64)
    if values.shape != (len(AXIS_NAMES),) or not np.all(np.isin(values, (-1.0, 1.0))):
        raise ValueError("sign mappings must contain four comma-separated values chosen from -1 and 1")
    return values


def _parse_axes(value: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    invalid = sorted(set(values) - set(AXIS_NAMES))
    if invalid:
        raise ValueError(f"unknown inactive axes: {invalid}")
    return values


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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
