#!/usr/bin/env python3
"""Train and materialize an axis-direction startup/release gate probe."""

from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from scripts.e37_full_act_gate_smoke import read_train_ready_episode_ids_compatible
from scripts.e40_deadzone_snap_probe import _load_actions, _write_actions_npz
from testbed.policies.deadzone_eval import load_deadzone_thresholds
from testbed.policies.offline_eval import aggregate_episode_results, compute_action_metrics, write_collection_report
from testbed.policies.phase_gate import apply_direction_gate_to_actions, direction_effective_labels


FEATURE_NAMES = [
    "intent_swing_pos",
    "intent_swing_neg",
    "intent_boom_pos",
    "intent_boom_neg",
    "intent_stick_pos",
    "intent_stick_neg",
    "intent_bucket_pos",
    "intent_bucket_neg",
    "qpos_swing",
    "qpos_boom",
    "qpos_stick",
    "qpos_bucket",
    "qvel_swing",
    "qvel_boom",
    "qvel_stick",
    "qvel_bucket",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-eval-dir", type=Path, required=True)
    parser.add_argument("--gate-prob-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--deadzone-json", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--thresholds", default="0.30,0.50,0.70,0.80,0.90")
    parser.add_argument("--inactive-scales", default="0.0,0.25,0.50")
    args = parser.parse_args()

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    thresholds = load_deadzone_thresholds(args.deadzone_json)
    episode_ids = read_train_ready_episode_ids_compatible(args.manifest)
    episodes = [
        _load_episode(
            episode_id=episode_id,
            source_eval_dir=args.source_eval_dir,
            gate_prob_dir=args.gate_prob_dir,
            dataset_dir=args.dataset_dir,
            thresholds=thresholds,
        )
        for episode_id in episode_ids
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fold_rows, direction_probs = _train_episode_heldout(
        episodes=episodes,
        folds=int(args.folds),
        epochs=int(args.epochs),
        hidden_dim=int(args.hidden_dim),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        seed=int(args.seed),
    )
    _write_json(args.output_dir / "fold_summary.json", fold_rows)
    _write_direction_probs(args.output_dir / "direction_probs", direction_probs)

    final_payload = _train_final_model(
        episodes=episodes,
        epochs=int(args.epochs),
        hidden_dim=int(args.hidden_dim),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        seed=int(args.seed),
    )
    torch.save(final_payload, args.output_dir / "direction_gate_model.pt")
    _write_json(
        args.output_dir / "direction_gate_model_metadata.json",
        {
            "feature_names": FEATURE_NAMES,
            "output_order": [
                "swing_pos",
                "swing_neg",
                "boom_pos",
                "boom_neg",
                "stick_pos",
                "stick_neg",
                "bucket_pos",
                "bucket_neg",
            ],
            "label": "expert_action_crosses_runtime_scaled_deadzone_per_axis_direction",
            "source_eval_dir": str(args.source_eval_dir),
            "gate_prob_dir": str(args.gate_prob_dir),
            "dataset_dir": str(args.dataset_dir),
            "deadzone_json": str(args.deadzone_json),
            "manifest": str(args.manifest),
        },
    )

    scan_rows = []
    for threshold in _parse_float_list(str(args.thresholds)):
        for inactive_scale in _parse_float_list(str(args.inactive_scales)):
            label = _label(threshold, inactive_scale)
            replay_dir = args.output_dir / label
            aggregate, scaled_frames = materialize_direction_gate(
                episodes=episodes,
                direction_probs=direction_probs,
                threshold=threshold,
                inactive_scale=inactive_scale,
                output_dir=replay_dir,
            )
            scan_rows.append(
                {
                    "label": label,
                    "threshold": threshold,
                    "inactive_scale": inactive_scale,
                    "scaled_frames": scaled_frames,
                    "mae": float(aggregate["global_metrics"]["overall"]["mae"]),
                    "rmse": float(aggregate["global_metrics"]["overall"]["rmse"]),
                    "replay_dir": str(replay_dir),
                }
            )
    _write_csv(args.output_dir / "direction_gate_scan.csv", scan_rows)
    _write_json(
        args.output_dir / "direction_gate_manifest.json",
        {
            "source_eval_dir": str(args.source_eval_dir),
            "gate_prob_dir": str(args.gate_prob_dir),
            "dataset_dir": str(args.dataset_dir),
            "deadzone_json": str(args.deadzone_json),
            "manifest": str(args.manifest),
            "thresholds": _parse_float_list(str(args.thresholds)),
            "inactive_scales": _parse_float_list(str(args.inactive_scales)),
            "scan_csv": str(args.output_dir / "direction_gate_scan.csv"),
        },
    )
    print(f"Direction gate scan: {args.output_dir / 'direction_gate_scan.csv'}")


def apply_direction_probability_gate(
    policy_action: np.ndarray,
    direction_prob: np.ndarray,
    *,
    threshold: float,
    inactive_scale: float,
) -> np.ndarray:
    active = np.asarray(direction_prob, dtype=np.float32) >= float(threshold)
    return apply_direction_gate_to_actions(policy_action, active, inactive_scale=float(inactive_scale))


def materialize_direction_gate(
    *,
    episodes: list[dict[str, Any]],
    direction_probs: dict[str, np.ndarray],
    threshold: float,
    inactive_scale: float,
    output_dir: Path,
) -> tuple[dict[str, Any], int]:
    results = []
    scaled_frames = 0
    for ep in episodes:
        episode_id = str(ep["episode_id"])
        policy = apply_direction_probability_gate(
            ep["policy_action"],
            direction_probs[episode_id],
            threshold=float(threshold),
            inactive_scale=float(inactive_scale),
        )
        _write_actions_npz(output_dir / "episodes" / episode_id / "actions.npz", expert=ep["expert_action"], policy=policy)
        scaled_frames += int(np.count_nonzero(np.any(policy != ep["policy_action"], axis=1)))
        results.append(
            {
                "episode_id": episode_id,
                "episode_path": episode_id,
                "n_steps": int(policy.shape[0]),
                "dt": 0.05,
                "expert_action": ep["expert_action"],
                "policy_action": policy,
                "metrics": compute_action_metrics(ep["expert_action"], policy),
            }
        )
    aggregate = aggregate_episode_results(results)
    write_collection_report(
        aggregate=aggregate,
        output_dir=output_dir,
        metadata={
            "selection_mode": "e43_direction_gate_probe",
            "threshold": float(threshold),
            "inactive_scale": float(inactive_scale),
            "scaled_frames": int(scaled_frames),
        },
    )
    return aggregate, int(scaled_frames)


def _load_episode(
    *,
    episode_id: str,
    source_eval_dir: Path,
    gate_prob_dir: Path,
    dataset_dir: Path,
    thresholds: dict[str, dict[str, float]],
) -> dict[str, Any]:
    actions = _load_actions(source_eval_dir, episode_id)
    gate_path = gate_prob_dir / f"{episode_id}.npz"
    episode_path = dataset_dir / f"{episode_id}.hdf5"
    if not gate_path.exists():
        raise FileNotFoundError(gate_path)
    if not episode_path.exists():
        raise FileNotFoundError(episode_path)
    with np.load(gate_path) as data:
        intent_prob = np.asarray(data["intent_prob"], dtype=np.float32)
    with h5py.File(episode_path, "r") as f:
        qpos = np.asarray(f["observations/qpos"][()], dtype=np.float32)
        qvel = np.asarray(f["observations/qvel"][()], dtype=np.float32)
    n = min(
        actions["expert_action"].shape[0],
        actions["policy_action"].shape[0],
        intent_prob.shape[0],
        qpos.shape[0],
        qvel.shape[0],
    )
    features = np.concatenate([intent_prob[:n], qpos[:n], qvel[:n]], axis=1).astype(np.float32)
    if features.shape[1] != len(FEATURE_NAMES):
        raise ValueError(f"feature shape mismatch for {episode_id}: {features.shape}")
    return {
        "episode_id": episode_id,
        "expert_action": actions["expert_action"][:n],
        "policy_action": actions["policy_action"][:n],
        "features": features,
        "label": direction_effective_labels(actions["expert_action"][:n], thresholds).astype(np.float32),
    }


def _train_episode_heldout(
    *,
    episodes: list[dict[str, Any]],
    folds: int,
    epochs: int,
    hidden_dim: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    rows = []
    probs_by_episode: dict[str, np.ndarray] = {}
    for fold in range(int(folds)):
        val_episodes = [ep for idx, ep in enumerate(episodes) if idx % int(folds) == fold]
        train_episodes = [ep for idx, ep in enumerate(episodes) if idx % int(folds) != fold]
        model, mean, std, best_val_loss = _fit_model(
            train_episodes=train_episodes,
            val_episodes=val_episodes,
            epochs=int(epochs),
            hidden_dim=int(hidden_dim),
            learning_rate=float(learning_rate),
            weight_decay=float(weight_decay),
            seed=int(seed) + fold,
        )
        val_probs = _predict_episodes(model, val_episodes, mean=mean, std=std)
        for episode_id, prob in val_probs.items():
            probs_by_episode[episode_id] = prob
        labels = np.concatenate([ep["label"] for ep in val_episodes], axis=0)
        probs = np.concatenate([val_probs[str(ep["episode_id"])] for ep in val_episodes], axis=0)
        rows.append({"fold": fold, "best_val_loss": best_val_loss, **_classification_metrics(labels, probs, 0.5)})
    return rows, probs_by_episode


def _train_final_model(
    *,
    episodes: list[dict[str, Any]],
    epochs: int,
    hidden_dim: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> dict[str, Any]:
    model, mean, std, best_loss = _fit_model(
        train_episodes=episodes,
        val_episodes=episodes,
        epochs=int(epochs),
        hidden_dim=int(hidden_dim),
        learning_rate=float(learning_rate),
        weight_decay=float(weight_decay),
        seed=int(seed) + 1000,
    )
    return {
        "model_state_dict": model.state_dict(),
        "feature_mean": mean.astype(np.float32),
        "feature_std": std.astype(np.float32),
        "feature_names": FEATURE_NAMES,
        "hidden_dim": int(hidden_dim),
        "best_train_eval_loss": float(best_loss),
    }


def _fit_model(
    *,
    train_episodes: list[dict[str, Any]],
    val_episodes: list[dict[str, Any]],
    epochs: int,
    hidden_dim: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> tuple[torch.nn.Module, np.ndarray, np.ndarray, float]:
    torch.manual_seed(int(seed))
    x_train = np.concatenate([ep["features"] for ep in train_episodes], axis=0)
    y_train = np.concatenate([ep["label"] for ep in train_episodes], axis=0)
    x_val = np.concatenate([ep["features"] for ep in val_episodes], axis=0)
    y_val = np.concatenate([ep["label"] for ep in val_episodes], axis=0)
    mean = x_train.mean(axis=0).astype(np.float32)
    std = x_train.std(axis=0)
    std = np.where(std > 1e-6, std, 1.0).astype(np.float32)
    x_train_t = torch.as_tensor((x_train - mean) / std, dtype=torch.float32)
    y_train_t = torch.as_tensor(y_train, dtype=torch.float32)
    x_val_t = torch.as_tensor((x_val - mean) / std, dtype=torch.float32)
    y_val_t = torch.as_tensor(y_val, dtype=torch.float32)

    model = _DirectionGateMlp(input_dim=x_train.shape[1], hidden_dim=int(hidden_dim), output_dim=y_train.shape[1])
    positives = y_train.sum(axis=0)
    negatives = y_train.shape[0] - positives
    pos_weight = np.where(positives > 0.0, negatives / np.maximum(positives, 1.0), 1.0).astype(np.float32)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.as_tensor(pos_weight, dtype=torch.float32))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))
    best_state = deepcopy(model.state_dict())
    best_val_loss = float("inf")
    for _ in range(int(epochs)):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x_train_t), y_train_t)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(criterion(model(x_val_t), y_val_t).item())
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    return model, mean, std, best_val_loss


def _predict_episodes(
    model: torch.nn.Module,
    episodes: list[dict[str, Any]],
    *,
    mean: np.ndarray,
    std: np.ndarray,
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for ep in episodes:
            x = np.asarray((ep["features"] - mean) / std, dtype=np.float32)
            logits = model(torch.as_tensor(x, dtype=torch.float32))
            out[str(ep["episode_id"])] = torch.sigmoid(logits).cpu().numpy().astype(np.float32)
    return out


def _classification_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float) -> dict[str, float]:
    y = np.asarray(labels, dtype=bool)
    pred = np.asarray(probs, dtype=np.float32) >= float(threshold)
    tp = int(np.count_nonzero(pred & y))
    fp = int(np.count_nonzero(pred & ~y))
    tn = int(np.count_nonzero(~pred & ~y))
    fn = int(np.count_nonzero(~pred & y))
    return {
        "micro_recall": _rate(tp, tp + fn),
        "micro_precision": _rate(tp, tp + fp),
        "micro_false_positive_rate": _rate(fp, fp + tn),
        "micro_accuracy": _rate(tp + tn, y.size),
    }


def _rate(num: int, den: int) -> float:
    return float(num) / float(den) if den else 0.0


def _parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _label(threshold: float, inactive_scale: float) -> str:
    return f"dir_t{int(round(float(threshold) * 100)):02d}_s{int(round(float(inactive_scale) * 100)):02d}"


def _write_direction_probs(output_dir: Path, probs_by_episode: dict[str, np.ndarray]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for episode_id, probs in sorted(probs_by_episode.items()):
        np.savez_compressed(output_dir / f"{episode_id}.npz", direction_prob=np.asarray(probs, dtype=np.float32))


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


class _DirectionGateMlp(torch.nn.Module):
    def __init__(self, *, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


if __name__ == "__main__":
    main()
