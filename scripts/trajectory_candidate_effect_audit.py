#!/usr/bin/env python3
"""Audit whether a held-out transition probe can resolve candidate action effects."""

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
from testbed.policies.trajectory_transition_eval import (
    TransitionSamples,
    build_transition_samples,
    fit_feature_support_model,
    fit_linear_transition_model,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--deadzone-json", type=Path, required=True)
    parser.add_argument("--raw-eval-dir", type=Path, required=True)
    parser.add_argument("--candidate-eval-dir", type=Path, required=True)
    parser.add_argument("--phase-prob-dir", type=Path, required=True)
    parser.add_argument("--candidate-name", default="candidate")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--horizons", default="5,10,20,40")
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--qvel-to-qpos-signs", required=True)
    parser.add_argument("--action-to-qpos-signs", required=True)
    parser.add_argument("--support-quantile", type=float, default=0.99)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260710)
    args = parser.parse_args()
    paths = run_audit(
        dataset_dir=args.dataset_dir,
        manifest_path=args.manifest,
        deadzone_path=args.deadzone_json,
        raw_eval_dir=args.raw_eval_dir,
        candidate_eval_dir=args.candidate_eval_dir,
        phase_prob_dir=args.phase_prob_dir,
        candidate_name=str(args.candidate_name),
        output_dir=args.output_dir,
        horizons=_parse_ints(args.horizons),
        stride=int(args.stride),
        qvel_to_qpos_sign=_parse_signs(args.qvel_to_qpos_signs),
        action_to_qpos_sign=_parse_signs(args.action_to_qpos_signs),
        support_quantile=float(args.support_quantile),
        bootstrap_samples=int(args.bootstrap_samples),
        bootstrap_seed=int(args.bootstrap_seed),
        argv=list(sys.argv),
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


def run_audit(
    *,
    dataset_dir: Path,
    manifest_path: Path,
    deadzone_path: Path,
    raw_eval_dir: Path,
    candidate_eval_dir: Path,
    phase_prob_dir: Path,
    candidate_name: str,
    output_dir: Path,
    horizons: tuple[int, ...],
    stride: int,
    qvel_to_qpos_sign: np.ndarray,
    action_to_qpos_sign: np.ndarray,
    support_quantile: float,
    bootstrap_samples: int,
    bootstrap_seed: int,
    argv: list[str],
) -> dict[str, Path]:
    if not candidate_name:
        raise ValueError("candidate_name must be non-empty")
    if stride <= 0 or bootstrap_samples <= 0:
        raise ValueError("stride and bootstrap_samples must be positive")
    if not 0.0 < support_quantile < 1.0:
        raise ValueError("support_quantile must be between zero and one")
    episode_ids = _episode_ids(manifest_path)
    if len(episode_ids) < 3:
        raise ValueError("candidate-effect audit requires at least three episodes")
    thresholds = _thresholds(deadzone_path)
    episodes, sample_period = _load_inputs(
        dataset_dir, raw_eval_dir, candidate_eval_dir, phase_prob_dir, episode_ids
    )
    sample_sets: dict[tuple[str, str, int], TransitionSamples] = {}
    phase_means: dict[tuple[str, int], np.ndarray] = {}
    for episode_id in episode_ids:
        episode = episodes[episode_id]
        for horizon in horizons:
            for action_name in ("expert", "raw", "candidate", "zero"):
                sample_sets[(episode_id, action_name, horizon)] = build_transition_samples(
                    qpos=episode["qpos"],
                    qvel=episode["qvel"],
                    action=episode[action_name],
                    thresholds=thresholds,
                    dt=sample_period,
                    horizon_steps=horizon,
                    stride=stride,
                    qvel_to_qpos_sign=qvel_to_qpos_sign,
                    action_to_qpos_sign=action_to_qpos_sign,
                )
            starts = sample_sets[(episode_id, "expert", horizon)].start_steps
            phase_means[(episode_id, horizon)] = np.asarray(
                [episode["phase_prob"][start : start + horizon].mean() for start in starts]
            )

    episode_rows: list[dict[str, Any]] = []
    stratum_rows: list[dict[str, Any]] = []
    for horizon in horizons:
        for heldout in episode_ids:
            train_ids = [episode_id for episode_id in episode_ids if episode_id != heldout]
            train = _concat_samples(sample_sets, train_ids, "expert", horizon)
            test = sample_sets[(heldout, "expert", horizon)]
            raw = sample_sets[(heldout, "raw", horizon)]
            candidate = sample_sets[(heldout, "candidate", horizon)]
            zero = sample_sets[(heldout, "zero", horizon)]
            train_phase = np.concatenate([phase_means[(episode_id, horizon)] for episode_id in train_ids])
            phase_edges = np.quantile(train_phase, [1.0 / 3.0, 2.0 / 3.0])
            concurrency = np.count_nonzero(np.abs(test.action_impulse) > 1.0e-12, axis=1)
            for axis_idx, axis in enumerate(AXIS_NAMES):
                if axis == "stick":
                    continue
                model = fit_linear_transition_model(
                    train.initial_qvel_displacement[:, axis_idx],
                    train.action_impulse[:, axis_idx],
                    train.target_qpos_delta[:, axis_idx],
                )
                train_features = np.column_stack(
                    [train.initial_qvel_displacement[:, axis_idx], train.action_impulse[:, axis_idx]]
                )
                support = fit_feature_support_model(train_features, quantile=support_quantile)
                target = test.target_qpos_delta[:, axis_idx]
                predictions = {
                    name: model.predict(
                        samples.initial_qvel_displacement[:, axis_idx], samples.action_impulse[:, axis_idx]
                    )
                    for name, samples in (("expert", test), ("raw", raw), ("candidate", candidate), ("zero", zero))
                }
                distances = {
                    name: support.distances(
                        np.column_stack(
                            [samples.initial_qvel_displacement[:, axis_idx], samples.action_impulse[:, axis_idx]]
                        )
                    )
                    for name, samples in (("expert", test), ("raw", raw), ("candidate", candidate), ("zero", zero))
                }
                base = _metric_row(target, predictions, distances, support.distance_threshold)
                episode_rows.append(
                    {
                        "episode_id": heldout,
                        "horizon_steps": horizon,
                        "horizon_seconds": horizon * sample_period,
                        "axis": axis,
                        **base,
                    }
                )
                magnitude_edges = np.quantile(np.abs(train.action_impulse[:, axis_idx]), [1 / 3, 2 / 3])
                labels = {
                    "phase_probability": _three_bins(phase_means[(heldout, horizon)], phase_edges),
                    "expert_action_magnitude": _three_bins(
                        np.abs(test.action_impulse[:, axis_idx]), magnitude_edges
                    ),
                    "expert_direction": np.where(
                        test.action_impulse[:, axis_idx] > 1.0e-12,
                        "positive",
                        np.where(test.action_impulse[:, axis_idx] < -1.0e-12, "negative", "inactive"),
                    ),
                    "active_axis_concurrency": np.where(
                        concurrency == 0, "zero", np.where(concurrency == 1, "one", "two_or_more")
                    ),
                }
                for dimension, values in labels.items():
                    for label in sorted(set(values.tolist())):
                        mask = values == label
                        if not np.any(mask):
                            continue
                        subset = _metric_row(
                            target[mask],
                            {name: value[mask] for name, value in predictions.items()},
                            {name: value[mask] for name, value in distances.items()},
                            support.distance_threshold,
                        )
                        stratum_rows.append(
                            {
                                "episode_id": heldout,
                                "horizon_steps": horizon,
                                "horizon_seconds": horizon * sample_period,
                                "axis": axis,
                                "dimension": dimension,
                                "stratum": label,
                                **subset,
                            }
                        )

    aggregate = _aggregate(episode_rows, bootstrap_samples, bootstrap_seed)
    stratum_aggregate = _aggregate_strata(stratum_rows)
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "run_manifest": output / "run_manifest.json",
        "candidate_effect_by_episode": output / "candidate_effect_by_episode.csv",
        "candidate_effect_aggregate": output / "candidate_effect_aggregate.csv",
        "candidate_effect_by_stratum": output / "candidate_effect_by_stratum.csv",
        "candidate_effect_stratum_aggregate": output / "candidate_effect_stratum_aggregate.csv",
        "summary": output / "summary.json",
    }
    generated_at = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    commit = _git_commit()
    _json(
        paths["run_manifest"],
        {
            "schema_version": 1,
            "generated_at": generated_at,
            "git_commit": commit,
            "argv": argv,
            "dataset_dir": str(Path(dataset_dir).resolve()),
            "manifest": str(Path(manifest_path).resolve()),
            "deadzone_json": str(Path(deadzone_path).resolve()),
            "raw_eval_dir": str(Path(raw_eval_dir).resolve()),
            "candidate_eval_dir": str(Path(candidate_eval_dir).resolve()),
            "phase_prob_dir": str(Path(phase_prob_dir).resolve()),
            "candidate_name": candidate_name,
            "horizons_steps": list(horizons),
            "stride_steps": stride,
            "support_quantile": support_quantile,
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": bootstrap_seed,
        },
    )
    _csv(paths["candidate_effect_by_episode"], episode_rows)
    _csv(paths["candidate_effect_aggregate"], aggregate)
    _csv(paths["candidate_effect_by_stratum"], stratum_rows)
    _csv(paths["candidate_effect_stratum_aggregate"], stratum_aggregate)
    _json(
        paths["summary"],
        {
            "schema_version": 1,
            "generated_at": generated_at,
            "git_commit": commit,
            "claim_boundary": "held_out_candidate_effect_resolvability_audit_only",
            "episodes": len(episode_ids),
            "sample_period_s": sample_period,
            "candidate_name": candidate_name,
            "phase_semantics": "probability quantiles only; no semantic phase names assigned",
            "support_rule": f"training-fold Mahalanobis distance <= training q{support_quantile:g}",
            "resolvability_rule": "report effect/model-error ratio; no pass threshold is imposed",
            "aggregate": aggregate,
            "artifacts": {name: str(path.resolve()) for name, path in paths.items() if name != "summary"},
        },
    )
    return paths


def _metric_row(
    target: np.ndarray,
    predictions: dict[str, np.ndarray],
    distances: dict[str, np.ndarray],
    threshold: float,
) -> dict[str, Any]:
    errors = {name: np.abs(value - target) for name, value in predictions.items()}
    model_error = float(errors["expert"].mean())
    effect = float(np.mean(np.abs(predictions["candidate"] - predictions["raw"])))
    return {
        "windows": int(target.size),
        "expert_model_mae": model_error,
        "zero_target_mae": float(errors["zero"].mean()),
        "raw_target_mae": float(errors["raw"].mean()),
        "candidate_target_mae": float(errors["candidate"].mean()),
        "candidate_minus_raw_target_mae": float(errors["candidate"].mean() - errors["raw"].mean()),
        "candidate_raw_predicted_effect_mae": effect,
        "effect_to_model_error_ratio": effect / model_error if model_error > 1.0e-12 else None,
        "expert_support_coverage": float(np.mean(distances["expert"] <= threshold)),
        "raw_support_coverage": float(np.mean(distances["raw"] <= threshold)),
        "candidate_support_coverage": float(np.mean(distances["candidate"] <= threshold)),
        "raw_support_distance_p95": float(np.percentile(distances["raw"], 95)),
        "candidate_support_distance_p95": float(np.percentile(distances["candidate"], 95)),
        "support_distance_threshold": float(threshold),
    }


def _aggregate(rows: list[dict[str, Any]], samples: int, seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    output = []
    for horizon, axis in sorted({(row["horizon_steps"], row["axis"]) for row in rows}):
        group = [row for row in rows if row["horizon_steps"] == horizon and row["axis"] == axis]
        delta = np.asarray([row["candidate_minus_raw_target_mae"] for row in group])
        boot = delta[rng.integers(0, delta.size, size=(samples, delta.size))].mean(axis=1)
        mean = lambda key: float(np.mean([row[key] for row in group]))
        model_error = mean("expert_model_mae")
        effect = mean("candidate_raw_predicted_effect_mae")
        output.append(
            {
                "horizon_steps": horizon,
                "horizon_seconds": group[0]["horizon_seconds"],
                "axis": axis,
                "episodes": len(group),
                "expert_model_mae": model_error,
                "zero_target_mae": mean("zero_target_mae"),
                "raw_target_mae": mean("raw_target_mae"),
                "candidate_target_mae": mean("candidate_target_mae"),
                "candidate_minus_raw_target_mae": float(delta.mean()),
                "candidate_minus_raw_ci95_low": float(np.percentile(boot, 2.5)),
                "candidate_minus_raw_ci95_high": float(np.percentile(boot, 97.5)),
                "episodes_candidate_better": int(np.count_nonzero(delta < 0.0)),
                "candidate_raw_predicted_effect_mae": effect,
                "effect_to_model_error_ratio": effect / model_error if model_error > 1.0e-12 else None,
                "expert_support_coverage": mean("expert_support_coverage"),
                "raw_support_coverage": mean("raw_support_coverage"),
                "candidate_support_coverage": mean("candidate_support_coverage"),
            }
        )
    return output


def _aggregate_strata(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = sorted({(r["horizon_steps"], r["axis"], r["dimension"], r["stratum"]) for r in rows})
    output = []
    for horizon, axis, dimension, stratum in keys:
        group = [r for r in rows if (r["horizon_steps"], r["axis"], r["dimension"], r["stratum"]) == (horizon, axis, dimension, stratum)]
        output.append(
            {
                "horizon_steps": horizon,
                "horizon_seconds": group[0]["horizon_seconds"],
                "axis": axis,
                "dimension": dimension,
                "stratum": stratum,
                "episodes": len(group),
                "windows": sum(r["windows"] for r in group),
                **{
                    key: float(np.average([r[key] for r in group], weights=[r["windows"] for r in group]))
                    for key in (
                        "expert_model_mae",
                        "raw_target_mae",
                        "candidate_target_mae",
                        "candidate_minus_raw_target_mae",
                        "candidate_raw_predicted_effect_mae",
                        "raw_support_coverage",
                        "candidate_support_coverage",
                    )
                },
            }
        )
    return output


def _concat_samples(data: dict[tuple[str, str, int], TransitionSamples], ids: list[str], name: str, horizon: int) -> TransitionSamples:
    values = [data[(episode_id, name, horizon)] for episode_id in ids]
    return TransitionSamples(
        start_steps=np.concatenate([v.start_steps for v in values]),
        target_qpos_delta=np.concatenate([v.target_qpos_delta for v in values]),
        initial_qvel_displacement=np.concatenate([v.initial_qvel_displacement for v in values]),
        action_impulse=np.concatenate([v.action_impulse for v in values]),
    )


def _three_bins(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.asarray(["low", "mid", "high"])[np.digitize(values, edges, right=True)]


def _load_inputs(dataset_dir: Path, raw_dir: Path, candidate_dir: Path, phase_dir: Path, episode_ids: list[str]) -> tuple[dict[str, dict[str, np.ndarray]], float]:
    episodes = {}
    dts = set()
    for episode_id in episode_ids:
        with h5py.File(Path(dataset_dir) / f"{episode_id}.hdf5", "r") as handle:
            expert = np.asarray(handle["action"], dtype=np.float64)
            qpos = np.asarray(handle["observations/qpos"], dtype=np.float64)
            qvel = np.asarray(handle["observations/qvel"], dtype=np.float64)
            metadata = handle.get("metadata")
            sample_dt = float(metadata.attrs["dt"]) if metadata is not None and "dt" in metadata.attrs else 0.05
        raw_time, raw_expert, raw = _actions(Path(raw_dir) / "episodes" / episode_id / "actions.npz")
        candidate_time, candidate_expert, candidate = _actions(Path(candidate_dir) / "episodes" / episode_id / "actions.npz")
        with np.load(Path(phase_dir) / f"{episode_id}.npz") as payload:
            phase = np.asarray(payload["phase_prob"], dtype=np.float64)
        if not np.array_equal(expert, raw_expert) or not np.array_equal(raw_expert, candidate_expert):
            raise ValueError(f"expert action mismatch for {episode_id}")
        if not np.array_equal(raw_time, candidate_time) or phase.shape != (expert.shape[0],):
            raise ValueError(f"replay alignment mismatch for {episode_id}")
        if any(value.shape != expert.shape for value in (qpos, qvel, raw, candidate)):
            raise ValueError(f"aligned (T, 4) arrays required for {episode_id}")
        episodes[episode_id] = {"expert": expert, "raw": raw, "candidate": candidate, "zero": np.zeros_like(expert), "qpos": qpos, "qvel": qvel, "phase_prob": phase}
        dts.add(sample_dt)
    if len(dts) != 1:
        raise ValueError(f"episodes must share one sample period, got {sorted(dts)}")
    return episodes, next(iter(dts))


def _actions(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path) as payload:
        return tuple(np.asarray(payload[key]) for key in ("time_s", "expert_action", "policy_action"))  # type: ignore[return-value]


def _episode_ids(path: Path) -> list[str]:
    values = json.loads(Path(path).read_text(encoding="utf-8")).get("train_ready_episode_ids")
    if not isinstance(values, list) or not values:
        raise ValueError("manifest must contain train_ready_episode_ids")
    return [str(value) for value in values]


def _thresholds(path: Path) -> dict[str, dict[str, float]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload.get("deadzone_action", payload)


def _parse_ints(value: str) -> tuple[int, ...]:
    values = tuple(int(item) for item in value.split(","))
    if not values or any(item <= 0 for item in values):
        raise ValueError("horizons must be positive")
    return values


def _parse_signs(value: str) -> np.ndarray:
    values = np.asarray([float(item) for item in value.split(",")])
    if values.shape != (4,) or not np.all(np.isin(values, (-1.0, 1.0))):
        raise ValueError("sign mappings require four values chosen from -1 and 1")
    return values


def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _git_commit() -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


if __name__ == "__main__":
    main()
