#!/usr/bin/env python3
"""Smoke final lightweight gate models for E34 runtime packaging."""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from scripts.e32_gohome_eligibility_probe import FEATURE_NAMES as GATE_FEATURE_NAMES
from testbed.policies.gohome_eligibility import (
    aggregate_gohome_event_rows,
    consecutive_active_mask,
    gated_active_mask,
    gohome_event_metrics_from_active_mask,
)
from testbed.policies.offline_eval import (
    aggregate_episode_results,
    compute_action_metrics,
    load_train_ready_episode_ids,
    write_collection_report,
)
from testbed.policies.phase_gate import apply_phase_gate_to_actions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-eval-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--handoff-dataset-dir", type=Path, required=True)
    parser.add_argument("--intent-prob-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--phase-model", type=Path, required=True)
    parser.add_argument("--phase-gate-name", required=True)
    parser.add_argument("--tail-candidate-model", type=Path, required=True)
    parser.add_argument("--gohome-eligibility-model", type=Path, required=True)
    parser.add_argument("--gohome-gate-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--latency-samples", type=int, default=2000)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    phase_gate = parse_phase_gate_name(args.phase_gate_name)
    gohome_gate = parse_gohome_gate_name(args.gohome_gate_name)

    phase_bundle = load_gate_model(args.phase_model)
    tail_bundle = load_gate_model(args.tail_candidate_model)
    eligibility_bundle = load_gate_model(args.gohome_eligibility_model)
    _validate_feature_contract(phase_bundle, tail_bundle, eligibility_bundle)

    episode_ids = load_train_ready_episode_ids(args.manifest)
    phase_dir = output_dir / "final_phase_action_replay"
    phase_probs: dict[str, np.ndarray] = {}
    action_results = []
    feature_batches: list[np.ndarray] = []
    for episode_id in episode_ids:
        phase_ep = _load_phase_episode(
            episode_id=episode_id,
            dataset_dir=args.dataset_dir,
            base_eval_dir=args.base_eval_dir,
            intent_prob_dir=args.intent_prob_dir,
        )
        phase_prob = predict_gate_probabilities(phase_bundle, phase_ep["features"])
        phase_probs[episode_id] = phase_prob
        active = phase_prob >= float(phase_gate["threshold"])
        gated_action = apply_phase_gate_to_actions(
            phase_ep["policy_action"],
            active,
            inactive_scale=float(phase_gate["inactive_scale"]),
        )
        episode_dir = phase_dir / "episodes" / episode_id
        episode_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            episode_dir / "actions.npz",
            time_s=phase_ep["time_s"],
            expert_action=phase_ep["expert_action"],
            policy_action=gated_action,
        )
        action_results.append(
            {
                "episode_id": episode_id,
                "episode_path": episode_id,
                "n_steps": int(gated_action.shape[0]),
                "dt": 0.05,
                "expert_action": phase_ep["expert_action"],
                "policy_action": gated_action,
                "metrics": compute_action_metrics(phase_ep["expert_action"], gated_action),
            }
        )
        feature_batches.append(phase_ep["features"])
    action_aggregate = aggregate_episode_results(action_results)
    write_collection_report(
        aggregate=action_aggregate,
        output_dir=phase_dir,
        metadata={
            "selection_mode": "final_phase_gate_model_runtime_smoke",
            "gate_name": str(args.phase_gate_name),
            "source_model": str(args.phase_model),
        },
    )
    _write_phase_probs(output_dir / "final_phase_probs", phase_probs)

    gohome_events, gohome_probs = _compute_final_gohome_events(
        episode_ids=episode_ids,
        handoff_dataset_dir=args.handoff_dataset_dir,
        intent_prob_dir=args.intent_prob_dir,
        tail_bundle=tail_bundle,
        eligibility_bundle=eligibility_bundle,
        gohome_gate=gohome_gate,
        gohome_gate_name=str(args.gohome_gate_name),
    )
    _write_csv(output_dir / "final_gohome_events.csv", gohome_events)
    _write_npz_probs(output_dir / "final_tail_candidate_probs", gohome_probs["tail_candidate"], "candidate_prob")
    _write_npz_probs(output_dir / "final_gohome_eligibility_probs", gohome_probs["eligibility"], "eligibility_prob")
    gohome_summary = aggregate_gohome_event_rows(gohome_events)

    latency_features = np.concatenate(feature_batches, axis=0)
    latency_summary = measure_gate_latency_ms(
        [phase_bundle, tail_bundle, eligibility_bundle],
        latency_features,
        samples=int(args.latency_samples),
    )
    summary = {
        "phase_gate_name": str(args.phase_gate_name),
        "gohome_gate_name": str(args.gohome_gate_name),
        "episodes": len(episode_ids),
        "action_replay_dir": str(phase_dir),
        "action_aggregate": action_aggregate,
        "gohome_event_summary": gohome_summary,
        "latency_summary": latency_summary,
        "artifact_paths": {
            "phase_model": str(args.phase_model),
            "tail_candidate_model": str(args.tail_candidate_model),
            "gohome_eligibility_model": str(args.gohome_eligibility_model),
            "base_eval_dir": str(args.base_eval_dir),
            "dataset_dir": str(args.dataset_dir),
            "handoff_dataset_dir": str(args.handoff_dataset_dir),
            "intent_prob_dir": str(args.intent_prob_dir),
            "manifest": str(args.manifest),
        },
    }
    _write_json(output_dir / "runtime_gate_smoke_summary.json", json_safe(summary))
    print(f"Runtime gate smoke summary: {output_dir / 'runtime_gate_smoke_summary.json'}")
    print(f"Final phase replay: {phase_dir}")
    print(f"Final gohome events: {output_dir / 'final_gohome_events.csv'}")


def parse_phase_gate_name(gate_name: str) -> dict[str, Any]:
    if "_s" not in gate_name:
        raise ValueError(f"phase gate must include inactive-scale suffix: {gate_name}")
    base, scale_text = gate_name.rsplit("_s", 1)
    if not base.startswith("simple_"):
        raise ValueError(f"only simple phase gates are supported for runtime smoke: {gate_name}")
    return {
        "mode": "simple",
        "threshold": float(base.split("_", 1)[1]),
        "inactive_scale": float(scale_text),
    }


def parse_gohome_gate_name(gate_name: str) -> dict[str, Any]:
    prefix = "learned_tail_t"
    if not gate_name.startswith(prefix):
        raise ValueError(f"unsupported gohome gate: {gate_name}")
    tail_text, elig_text = gate_name.removeprefix(prefix).split("_e", 1)
    candidate_threshold_text, candidate_consecutive_text = tail_text.split("_tc", 1)
    eligibility_threshold_text, eligibility_consecutive_text = elig_text.split("_ec", 1)
    return {
        "candidate_threshold": float(candidate_threshold_text),
        "candidate_consecutive_steps": int(candidate_consecutive_text),
        "eligibility_threshold": float(eligibility_threshold_text),
        "eligibility_consecutive_steps": int(eligibility_consecutive_text),
    }


def active_mask_from_gohome_gate(
    candidate_probability: np.ndarray,
    eligibility_probability: np.ndarray,
    gate: dict[str, Any],
) -> np.ndarray:
    candidate_active = consecutive_active_mask(
        candidate_probability,
        threshold=float(gate["candidate_threshold"]),
        consecutive_steps=int(gate["candidate_consecutive_steps"]),
    )
    return gated_active_mask(
        candidate_active=candidate_active,
        eligibility_probability=eligibility_probability,
        eligibility_threshold=float(gate["eligibility_threshold"]),
        eligibility_consecutive_steps=int(gate["eligibility_consecutive_steps"]),
    )


@dataclass(frozen=True)
class GateModelBundle:
    model: torch.nn.Module
    mean: np.ndarray
    std: np.ndarray
    feature_names: list[str]


def load_gate_model(path: Path) -> GateModelBundle:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    feature_names = list(payload["feature_names"])
    hidden_dim = int(payload["hidden_dim"])
    model = _GateMlp(input_dim=len(feature_names), hidden_dim=hidden_dim)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return GateModelBundle(
        model=model,
        mean=np.asarray(payload["feature_mean"], dtype=np.float32),
        std=np.asarray(payload["feature_std"], dtype=np.float32),
        feature_names=feature_names,
    )


def predict_gate_probabilities(bundle: GateModelBundle, features: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != len(bundle.feature_names):
        raise ValueError(f"feature shape mismatch: got {x.shape}, expected (*, {len(bundle.feature_names)})")
    with torch.no_grad():
        logits = bundle.model(torch.as_tensor((x - bundle.mean) / bundle.std, dtype=torch.float32)).reshape(-1)
        return torch.sigmoid(logits).cpu().numpy().astype(np.float32)


def measure_gate_latency_ms(
    bundles: list[GateModelBundle],
    features: np.ndarray,
    *,
    samples: int,
) -> dict[str, Any]:
    x = np.asarray(features, dtype=np.float32)
    if x.shape[0] == 0:
        raise ValueError("latency features are empty")
    count = min(max(1, int(samples)), int(x.shape[0]))
    sample = x[:count]
    timings: list[float] = []
    with torch.no_grad():
        for row in sample:
            start = time.perf_counter()
            for bundle in bundles:
                tensor = torch.as_tensor((row.reshape(1, -1) - bundle.mean) / bundle.std, dtype=torch.float32)
                _ = torch.sigmoid(bundle.model(tensor))
            timings.append((time.perf_counter() - start) * 1000.0)
    return {
        "samples": int(count),
        "models_per_step": len(bundles),
        "mean_ms": float(np.mean(timings)),
        "p50_ms": float(np.percentile(timings, 50)),
        "p95_ms": float(np.percentile(timings, 95)),
        "max_ms": float(np.max(timings)),
    }


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _validate_feature_contract(*bundles: GateModelBundle) -> None:
    expected = list(GATE_FEATURE_NAMES)
    for bundle in bundles:
        if list(bundle.feature_names) != expected:
            raise ValueError(f"gate feature names mismatch: {bundle.feature_names}")


def _load_phase_episode(
    *,
    episode_id: str,
    dataset_dir: Path,
    base_eval_dir: Path,
    intent_prob_dir: Path,
) -> dict[str, Any]:
    action_path = base_eval_dir / "episodes" / episode_id / "actions.npz"
    episode_path = dataset_dir / f"{episode_id}.hdf5"
    intent_path = intent_prob_dir / f"{episode_id}.npz"
    if not action_path.exists():
        raise FileNotFoundError(action_path)
    if not episode_path.exists():
        raise FileNotFoundError(episode_path)
    if not intent_path.exists():
        raise FileNotFoundError(intent_path)
    with np.load(action_path) as data:
        time_s = np.asarray(data["time_s"], dtype=np.float64)
        expert = np.asarray(data["expert_action"], dtype=np.float32)
        policy = np.asarray(data["policy_action"], dtype=np.float32)
    with np.load(intent_path) as data:
        intent_prob = np.asarray(data["intent_prob"], dtype=np.float32)
    with h5py.File(episode_path, "r") as f:
        qpos = np.asarray(f["observations/qpos"][()], dtype=np.float32)
        qvel = np.asarray(f["observations/qvel"][()], dtype=np.float32)
    n = int(min(time_s.shape[0], expert.shape[0], policy.shape[0], intent_prob.shape[0], qpos.shape[0], qvel.shape[0]))
    features = np.concatenate([intent_prob[:n], qpos[:n], qvel[:n]], axis=1).astype(np.float32)
    return {
        "time_s": time_s[:n],
        "expert_action": expert[:n],
        "policy_action": policy[:n],
        "features": features,
    }


def _compute_final_gohome_events(
    *,
    episode_ids: list[str],
    handoff_dataset_dir: Path,
    intent_prob_dir: Path,
    tail_bundle: GateModelBundle,
    eligibility_bundle: GateModelBundle,
    gohome_gate: dict[str, Any],
    gohome_gate_name: str,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, np.ndarray]]]:
    rows: list[dict[str, Any]] = []
    candidate_probs: dict[str, np.ndarray] = {}
    eligibility_probs: dict[str, np.ndarray] = {}
    for episode_id in episode_ids:
        episode_path = handoff_dataset_dir / f"{episode_id}.hdf5"
        intent_path = intent_prob_dir / f"{episode_id}.npz"
        if not episode_path.exists():
            raise FileNotFoundError(episode_path)
        if not intent_path.exists():
            raise FileNotFoundError(intent_path)
        with np.load(intent_path) as data:
            intent_prob = np.asarray(data["intent_prob"], dtype=np.float32)
        with h5py.File(episode_path, "r") as f:
            qpos = np.asarray(f["observations/qpos"][()], dtype=np.float32)
            qvel = np.asarray(f["observations/qvel"][()], dtype=np.float32)
            label = np.asarray(f["handoff/gohome_eligible_label"][()], dtype=bool)
            loss_mask = np.asarray(f["handoff/gohome_loss_mask"][()], dtype=bool)
            tail_idle = np.asarray(f["handoff/tail_idle_mask"][()], dtype=bool)
        n = int(min(intent_prob.shape[0], qpos.shape[0], qvel.shape[0], label.shape[0], loss_mask.shape[0], tail_idle.shape[0]))
        features = np.concatenate([intent_prob[:n], qpos[:n], qvel[:n]], axis=1).astype(np.float32)
        candidate = predict_gate_probabilities(tail_bundle, features)
        eligibility = predict_gate_probabilities(eligibility_bundle, features)
        active = active_mask_from_gohome_gate(candidate, eligibility, gohome_gate)
        candidate_probs[episode_id] = candidate
        eligibility_probs[episode_id] = eligibility
        rows.append(
            gohome_event_metrics_from_active_mask(
                episode_id=episode_id,
                active_mask=active,
                eligible_label=label[:n],
                loss_mask=loss_mask[:n],
                tail_idle_mask=tail_idle[:n],
                gate=gohome_gate_name,
            )
        )
    return rows, {"tail_candidate": candidate_probs, "eligibility": eligibility_probs}


def _write_phase_probs(output_dir: Path, phase_probs: dict[str, np.ndarray]) -> None:
    _write_npz_probs(output_dir, phase_probs, "phase_prob")


def _write_npz_probs(output_dir: Path, probs_by_episode: dict[str, np.ndarray], key: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for episode_id, probs in sorted(probs_by_episode.items()):
        np.savez_compressed(output_dir / f"{episode_id}.npz", **{key: np.asarray(probs, dtype=np.float32)})


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


class _GateMlp(torch.nn.Module):
    def __init__(self, *, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


if __name__ == "__main__":
    main()
