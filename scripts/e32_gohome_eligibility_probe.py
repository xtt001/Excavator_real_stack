#!/usr/bin/env python3
"""Train and evaluate a lightweight gohome eligibility probe for E32."""

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

from testbed.policies.gohome_eligibility import (
    aggregate_gohome_event_rows,
    gohome_event_metrics,
)
from testbed.policies.phase_gate import phase_gate_metadata
from testbed.policies.offline_eval import load_train_ready_episode_ids


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
    parser.add_argument("--handoff-dataset-dir", type=Path, required=True)
    parser.add_argument("--intent-prob-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--thresholds",
        default="0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.60,0.70,0.80,0.90",
    )
    parser.add_argument("--consecutive-steps", default="1,2,3,4,5")
    parser.add_argument("--materialize", default="auto")
    args = parser.parse_args()

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    episode_ids = load_train_ready_episode_ids(args.manifest)
    episodes = [
        _load_episode(
            episode_id=episode_id,
            handoff_dataset_dir=args.handoff_dataset_dir,
            intent_prob_dir=args.intent_prob_dir,
        )
        for episode_id in episode_ids
    ]

    fold_rows, probs_by_episode = _train_episode_heldout(
        episodes=episodes,
        folds=int(args.folds),
        epochs=int(args.epochs),
        hidden_dim=int(args.hidden_dim),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        seed=int(args.seed),
    )
    _write_json(output_dir / "fold_summary.json", fold_rows)
    _write_probs(output_dir / "eligibility_probs", probs_by_episode)

    final_payload = _train_final_model(
        episodes=episodes,
        epochs=int(args.epochs),
        hidden_dim=int(args.hidden_dim),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        seed=int(args.seed),
    )
    torch.save(final_payload, output_dir / "gohome_eligibility_model.pt")
    _write_json(
        output_dir / "gohome_eligibility_model_metadata.json",
        phase_gate_metadata(
            feature_names=FEATURE_NAMES,
            open_threshold=0.0,
            close_threshold=0.0,
            extra={
                "model": "mlp_16_32_1",
                "label": "handoff/gohome_eligible_label",
                "training": "all train-ready episodes; use OOF probabilities for reported gates",
                "handoff_dataset_dir": str(args.handoff_dataset_dir),
                "intent_prob_dir": str(args.intent_prob_dir),
                "manifest": str(args.manifest),
            },
        ),
    )

    scan_rows, event_rows_by_gate = _scan_thresholds(
        episodes=episodes,
        probs_by_episode=probs_by_episode,
        thresholds=_parse_float_list(args.thresholds),
        consecutive_steps=_parse_int_list(args.consecutive_steps),
    )
    _write_csv(output_dir / "threshold_scan.csv", scan_rows)

    materialize = str(args.materialize)
    if materialize == "auto":
        materialize = _choose_gate(scan_rows)
    selected_rows = event_rows_by_gate[materialize]
    _write_csv(output_dir / f"{materialize}_events.csv", selected_rows)
    _write_json(
        output_dir / "gate_summary.json",
        {
            "gate": materialize,
            "scan_row": next(row for row in scan_rows if str(row["gate"]) == materialize),
        },
    )
    print(f"Fold summary: {output_dir / 'fold_summary.json'}")
    print(f"Threshold scan: {output_dir / 'threshold_scan.csv'}")
    print(f"Selected events: {output_dir / f'{materialize}_events.csv'}")


def _load_episode(
    *,
    episode_id: str,
    handoff_dataset_dir: Path,
    intent_prob_dir: Path,
) -> dict[str, Any]:
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
    if n <= 0:
        raise ValueError(f"empty episode after alignment: {episode_id}")
    features = np.concatenate([intent_prob[:n], qpos[:n], qvel[:n]], axis=1).astype(np.float32)
    if features.shape[1] != len(FEATURE_NAMES):
        raise ValueError(f"feature shape mismatch for {episode_id}: {features.shape}")
    return {
        "episode_id": str(episode_id),
        "features": features,
        "label": label[:n].astype(np.float32),
        "label_bool": label[:n].astype(bool),
        "loss_mask": loss_mask[:n].astype(bool),
        "tail_idle_mask": tail_idle[:n].astype(bool),
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
    if folds < 2:
        raise ValueError("folds must be >= 2")
    rows: list[dict[str, Any]] = []
    probs_by_episode: dict[str, np.ndarray] = {}
    for fold in range(folds):
        val_episodes = [ep for idx, ep in enumerate(episodes) if idx % folds == fold]
        train_episodes = [ep for idx, ep in enumerate(episodes) if idx % folds != fold]
        model, mean, std, best_val_loss = _fit_model(
            train_episodes=train_episodes,
            val_episodes=val_episodes,
            epochs=epochs,
            hidden_dim=hidden_dim,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            seed=seed + fold,
        )
        val_probs = _predict_episodes(model, val_episodes, mean=mean, std=std)
        probs_by_episode.update(val_probs)
        labels, probs = _flatten_labeled_probs(val_episodes, val_probs)
        rows.append(
            {
                "fold": fold,
                "val_episodes": [str(ep["episode_id"]) for ep in val_episodes],
                "best_val_loss": float(best_val_loss),
                **_classification_metrics(labels, probs, threshold=0.5),
            }
        )
    missing = [str(ep["episode_id"]) for ep in episodes if str(ep["episode_id"]) not in probs_by_episode]
    if missing:
        raise RuntimeError(f"missing OOF probabilities for episodes: {missing}")
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
        epochs=epochs,
        hidden_dim=hidden_dim,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        seed=seed + 1000,
    )
    return {
        "state_dict": model.state_dict(),
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
    x_train, y_train = _flatten_labeled_features(train_episodes)
    x_val, y_val = _flatten_labeled_features(val_episodes)
    mean = x_train.mean(axis=0).astype(np.float32)
    std = x_train.std(axis=0)
    std = np.where(std > 1e-6, std, 1.0).astype(np.float32)
    x_train_t = torch.as_tensor((x_train - mean) / std, dtype=torch.float32)
    y_train_t = torch.as_tensor(y_train.reshape(-1, 1), dtype=torch.float32)
    x_val_t = torch.as_tensor((x_val - mean) / std, dtype=torch.float32)
    y_val_t = torch.as_tensor(y_val.reshape(-1, 1), dtype=torch.float32)

    model = _EligibilityMlp(input_dim=x_train.shape[1], hidden_dim=hidden_dim)
    positives = float(y_train.sum())
    negatives = float(y_train.shape[0] - y_train.sum())
    pos_weight = torch.tensor([negatives / max(positives, 1.0)], dtype=torch.float32)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
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


def _flatten_labeled_features(episodes: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for ep in episodes:
        mask = np.asarray(ep["loss_mask"], dtype=bool)
        xs.append(np.asarray(ep["features"], dtype=np.float32)[mask])
        ys.append(np.asarray(ep["label"], dtype=np.float32)[mask])
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def _flatten_labeled_probs(
    episodes: list[dict[str, Any]],
    probs_by_episode: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    labels: list[np.ndarray] = []
    probs: list[np.ndarray] = []
    for ep in episodes:
        mask = np.asarray(ep["loss_mask"], dtype=bool)
        episode_id = str(ep["episode_id"])
        labels.append(np.asarray(ep["label"], dtype=np.float32)[mask])
        probs.append(np.asarray(probs_by_episode[episode_id], dtype=np.float32)[mask])
    return np.concatenate(labels, axis=0), np.concatenate(probs, axis=0)


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
            logits = model(torch.as_tensor(x, dtype=torch.float32)).reshape(-1)
            out[str(ep["episode_id"])] = torch.sigmoid(logits).cpu().numpy().astype(np.float32)
    return out


def _scan_thresholds(
    *,
    episodes: list[dict[str, Any]],
    probs_by_episode: dict[str, np.ndarray],
    thresholds: list[float],
    consecutive_steps: list[int],
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    rows: list[dict[str, Any]] = []
    events_by_gate: dict[str, list[dict[str, Any]]] = {}
    for threshold in thresholds:
        for consecutive in consecutive_steps:
            gate = f"thr_{threshold:.2f}_c{int(consecutive)}"
            event_rows = [
                gohome_event_metrics(
                    episode_id=str(ep["episode_id"]),
                    probability=probs_by_episode[str(ep["episode_id"])],
                    eligible_label=np.asarray(ep["label_bool"], dtype=bool),
                    loss_mask=np.asarray(ep["loss_mask"], dtype=bool),
                    tail_idle_mask=np.asarray(ep["tail_idle_mask"], dtype=bool),
                    threshold=threshold,
                    consecutive_steps=consecutive,
                )
                for ep in episodes
            ]
            agg = aggregate_gohome_event_rows(event_rows)
            labels, probs = _flatten_labeled_probs(episodes, probs_by_episode)
            frame_metrics = _classification_metrics(labels, probs, threshold=threshold)
            row = {
                "gate": gate,
                "threshold": float(threshold),
                "consecutive_steps": int(consecutive),
                **agg,
                **{f"frame_{key}": value for key, value in frame_metrics.items()},
            }
            row["score"] = _selection_score(row)
            rows.append(row)
            events_by_gate[gate] = event_rows
    return rows, events_by_gate


def _selection_score(row: dict[str, Any]) -> float:
    mean_delay = row.get("mean_detection_delay_steps")
    delay_penalty = float(mean_delay) / 20.0 if mean_delay != "" else 1.0
    return (
        2.0 * float(row["event_recall"])
        - 5.0 * float(row["early_false_positive_episode_rate"])
        - 10.0 * float(row["pre_tail_false_positive_episode_rate"])
        - delay_penalty
        + 0.1 * float(row["frame_precision"])
    )


def _choose_gate(rows: list[dict[str, Any]]) -> str:
    viable = [
        row
        for row in rows
        if float(row["pre_tail_false_positive_episode_rate"]) == 0.0
        and float(row["early_false_positive_episode_rate"]) <= 0.25
        and float(row["event_recall"]) >= 0.8
    ]
    candidates = viable or rows
    best = max(candidates, key=lambda row: float(row["score"]))
    return str(best["gate"])


def _classification_metrics(labels: np.ndarray, probs: np.ndarray, *, threshold: float) -> dict[str, float]:
    y = np.asarray(labels, dtype=bool)
    pred = np.asarray(probs, dtype=np.float32) >= float(threshold)
    tp = int(np.count_nonzero(pred & y))
    fp = int(np.count_nonzero(pred & ~y))
    tn = int(np.count_nonzero(~pred & ~y))
    fn = int(np.count_nonzero(~pred & y))
    return {
        "recall": _rate(tp, tp + fn),
        "precision": _rate(tp, tp + fp),
        "false_positive_rate": _rate(fp, fp + tn),
        "accuracy": _rate(tp + tn, y.shape[0]),
    }


def _rate(num: int, den: int) -> float:
    return float(num) / float(den) if den else 0.0


def _parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _write_probs(output_dir: Path, probs_by_episode: dict[str, np.ndarray]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for episode_id, probs in sorted(probs_by_episode.items()):
        np.savez_compressed(output_dir / f"{episode_id}.npz", eligibility_prob=np.asarray(probs, dtype=np.float32))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


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


class _EligibilityMlp(torch.nn.Module):
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
