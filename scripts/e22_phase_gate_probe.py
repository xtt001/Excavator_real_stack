#!/usr/bin/env python3
"""Train and evaluate a lightweight should-move phase gate for E22."""

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

from testbed.policies.deadzone_eval import (
    aggregate_window_rows,
    compute_deadzone_window_rows,
    load_deadzone_thresholds,
)
from testbed.policies.offline_eval import (
    aggregate_episode_results,
    compute_action_metrics,
    load_train_ready_episode_ids,
    write_collection_report,
)
from testbed.policies.phase_gate import (
    apply_phase_gate_to_actions,
    build_hysteresis_mask,
    phase_gate_metadata,
    should_move_labels,
)


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
    parser.add_argument("--base-eval-dir", type=Path, required=True)
    parser.add_argument("--intent-prob-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--deadzone-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--simple-thresholds",
        default="0.15,0.20,0.25,0.30,0.35,0.40,0.50",
        help="Comma-separated per-frame probability thresholds to scan.",
    )
    parser.add_argument(
        "--hysteresis-pairs",
        default="0.25:0.10,0.30:0.10,0.30:0.15,0.35:0.15,0.40:0.20,0.50:0.20",
        help="Comma-separated open:close threshold pairs to scan.",
    )
    parser.add_argument(
        "--inactive-scales",
        default="0.0",
        help="Comma-separated action scales for inactive gate steps. 0.0 is hard zeroing.",
    )
    parser.add_argument(
        "--materialize",
        default="auto",
        help="Gate name to materialize, or auto to select by scan score.",
    )
    args = parser.parse_args()

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    thresholds = load_deadzone_thresholds(args.deadzone_json)
    episode_ids = load_train_ready_episode_ids(args.manifest)
    episodes = [
        _load_episode(
            episode_id=episode_id,
            dataset_dir=args.dataset_dir,
            base_eval_dir=args.base_eval_dir,
            intent_prob_dir=args.intent_prob_dir,
            thresholds=thresholds,
        )
        for episode_id in episode_ids
    ]

    fold_rows, phase_probs = _train_episode_heldout(
        episodes=episodes,
        folds=int(args.folds),
        epochs=int(args.epochs),
        hidden_dim=int(args.hidden_dim),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        seed=int(args.seed),
    )
    _write_json(output_dir / "fold_summary.json", fold_rows)
    _write_phase_probs(output_dir / "phase_probs", phase_probs)

    final_payload = _train_final_model(
        episodes=episodes,
        epochs=int(args.epochs),
        hidden_dim=int(args.hidden_dim),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        seed=int(args.seed),
    )
    torch.save(final_payload, output_dir / "phase_gate_model.pt")
    _write_json(
        output_dir / "phase_gate_model_metadata.json",
        phase_gate_metadata(
            feature_names=FEATURE_NAMES,
            open_threshold=0.0,
            close_threshold=0.0,
            extra={
                "model": "mlp_16_32_1",
                "label": "expert_action_crosses_any_runtime_scaled_deadzone",
                "training": "all train-ready episodes; use OOF probabilities for reported gates",
                "base_eval_dir": str(args.base_eval_dir),
                "intent_prob_dir": str(args.intent_prob_dir),
                "dataset_dir": str(args.dataset_dir),
                "manifest": str(args.manifest),
                "deadzone_json": str(args.deadzone_json),
            },
        ),
    )

    scan_rows = _scan_gates(
        episodes=episodes,
        phase_probs=phase_probs,
        thresholds=thresholds,
        simple_thresholds=_parse_float_list(args.simple_thresholds),
        hysteresis_pairs=_parse_pairs(args.hysteresis_pairs),
        inactive_scales=_parse_float_list(args.inactive_scales),
    )
    _write_csv(output_dir / "threshold_scan.csv", scan_rows)

    materialize_name = str(args.materialize)
    if materialize_name == "auto":
        materialize_name = _choose_gate(scan_rows)
    gate_dir = output_dir / materialize_name
    _materialize_gate(
        output_dir=gate_dir,
        gate_name=materialize_name,
        episodes=episodes,
        phase_probs=phase_probs,
        scan_rows=scan_rows,
    )
    print(f"Fold summary: {output_dir / 'fold_summary.json'}")
    print(f"Threshold scan: {output_dir / 'threshold_scan.csv'}")
    print(f"Materialized gate: {gate_dir}")


def _load_episode(
    *,
    episode_id: str,
    dataset_dir: Path,
    base_eval_dir: Path,
    intent_prob_dir: Path,
    thresholds: dict[str, dict[str, float]],
) -> dict[str, Any]:
    action_path = base_eval_dir / "episodes" / episode_id / "actions.npz"
    intent_path = intent_prob_dir / f"{episode_id}.npz"
    episode_path = dataset_dir / f"{episode_id}.hdf5"
    if not action_path.exists():
        raise FileNotFoundError(action_path)
    if not intent_path.exists():
        raise FileNotFoundError(intent_path)
    if not episode_path.exists():
        raise FileNotFoundError(episode_path)

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
    if n <= 0:
        raise ValueError(f"empty episode after alignment: {episode_id}")
    features = np.concatenate([intent_prob[:n], qpos[:n], qvel[:n]], axis=1).astype(np.float32)
    if features.shape[1] != len(FEATURE_NAMES):
        raise ValueError(f"feature shape mismatch for {episode_id}: {features.shape}")
    labels = should_move_labels(expert[:n], thresholds).astype(np.float32)
    return {
        "episode_id": episode_id,
        "time_s": time_s[:n],
        "expert_action": expert[:n],
        "policy_action": policy[:n],
        "features": features,
        "label": labels,
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
        for episode_id, prob in val_probs.items():
            probs_by_episode[episode_id] = prob
        labels = np.concatenate([ep["label"] for ep in val_episodes])
        probs = np.concatenate([val_probs[str(ep["episode_id"])] for ep in val_episodes])
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
    x_train = np.concatenate([ep["features"] for ep in train_episodes], axis=0)
    y_train = np.concatenate([ep["label"] for ep in train_episodes], axis=0)
    x_val = np.concatenate([ep["features"] for ep in val_episodes], axis=0)
    y_val = np.concatenate([ep["label"] for ep in val_episodes], axis=0)
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std = np.where(std > 1e-6, std, 1.0).astype(np.float32)
    mean = mean.astype(np.float32)
    x_train_t = torch.as_tensor((x_train - mean) / std, dtype=torch.float32)
    y_train_t = torch.as_tensor(y_train.reshape(-1, 1), dtype=torch.float32)
    x_val_t = torch.as_tensor((x_val - mean) / std, dtype=torch.float32)
    y_val_t = torch.as_tensor(y_val.reshape(-1, 1), dtype=torch.float32)

    model = _PhaseGateMlp(input_dim=x_train.shape[1], hidden_dim=hidden_dim)
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


def _scan_gates(
    *,
    episodes: list[dict[str, Any]],
    phase_probs: dict[str, np.ndarray],
    thresholds: dict[str, dict[str, float]],
    simple_thresholds: list[float],
    hysteresis_pairs: list[tuple[float, float]],
    inactive_scales: list[float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for threshold in simple_thresholds:
        for inactive_scale in inactive_scales:
            rows.append(
                _gate_metrics(
                    name=_simple_gate_name(threshold, inactive_scale),
                    episodes=episodes,
                    thresholds=thresholds,
                    inactive_scale=inactive_scale,
                    active_by_episode={
                        str(ep["episode_id"]): phase_probs[str(ep["episode_id"])] >= threshold
                        for ep in episodes
                    },
                )
            )
    for open_threshold, close_threshold in hysteresis_pairs:
        for inactive_scale in inactive_scales:
            rows.append(
                _gate_metrics(
                    name=_hysteresis_gate_name(open_threshold, close_threshold, inactive_scale),
                    episodes=episodes,
                    thresholds=thresholds,
                    inactive_scale=inactive_scale,
                    active_by_episode={
                        str(ep["episode_id"]): build_hysteresis_mask(
                            phase_probs[str(ep["episode_id"])],
                            open_threshold=open_threshold,
                            close_threshold=close_threshold,
                        )
                        for ep in episodes
                    },
                )
            )
    return rows


def _gate_metrics(
    *,
    name: str,
    episodes: list[dict[str, Any]],
    thresholds: dict[str, dict[str, float]],
    active_by_episode: dict[str, np.ndarray],
    inactive_scale: float,
) -> dict[str, Any]:
    results = []
    window_rows = []
    startup_rows = []
    for ep in episodes:
        episode_id = str(ep["episode_id"])
        active = active_by_episode[episode_id]
        gated = apply_phase_gate_to_actions(ep["policy_action"], active, inactive_scale=inactive_scale)
        results.append(
            {
                "episode_id": episode_id,
                "episode_path": episode_id,
                "n_steps": int(gated.shape[0]),
                "dt": 0.05,
                "expert_action": ep["expert_action"],
                "policy_action": gated,
                "metrics": compute_action_metrics(ep["expert_action"], gated),
            }
        )
        window_rows.extend(
            compute_deadzone_window_rows(
                model=name,
                episode_id=episode_id,
                expert_action=ep["expert_action"],
                policy_action=gated,
                thresholds=thresholds,
            )
        )
        startup_rows.append(_startup_metrics(ep["expert_action"], gated, thresholds=thresholds))
    aggregate = aggregate_episode_results(results)
    window_agg = {
        str(row["window"]): row for row in aggregate_window_rows(window_rows) if str(row["model"]) == name
    }
    start40 = window_agg.get("start40", {})
    main = window_agg.get("longest_expert_effective_segment_gap5", {})
    full = window_agg.get("full_available", {})
    startup = _mean_startup(startup_rows)
    metrics = aggregate["global_metrics"]["overall"]
    return {
        "gate": name,
        "inactive_scale": float(inactive_scale),
        "mae": float(metrics["mae"]),
        "rmse": float(metrics["rmse"]),
        "policy_p95_abs": float(metrics["policy_p95_abs"]),
        "policy_max_abs": float(metrics["policy_max_abs"]),
        "start40_policy_any_effective_pct": float(start40.get("mean_policy_any_effective_pct", 0.0)),
        "start40_same_dir_pct": float(start40.get("mean_same_axis_dir_effective_pct_of_expert_effective", 0.0)),
        "start40_extra_or_wrong_pct": float(start40.get("mean_policy_extra_or_wrong_effective_pct", 0.0)),
        "startup_policy_any_effective_pct": startup["policy_any_effective_pct"],
        "startup_same_dir_pct": startup["same_axis_dir_pct"],
        "startup_extra_or_wrong_pct": startup["extra_or_wrong_pct"],
        "main_policy_any_effective_pct": float(main.get("mean_policy_any_effective_pct", 0.0)),
        "main_same_dir_pct": float(main.get("mean_same_axis_dir_effective_pct_of_expert_effective", 0.0)),
        "main_extra_or_wrong_pct": float(main.get("mean_policy_extra_or_wrong_effective_pct", 0.0)),
        "full_policy_any_effective_pct": float(full.get("mean_policy_any_effective_pct", 0.0)),
        "full_same_dir_pct": float(full.get("mean_same_axis_dir_effective_pct_of_expert_effective", 0.0)),
        "score": _selection_score(startup, start40, main, metrics),
    }


def _startup_metrics(
    expert_action: np.ndarray,
    policy_action: np.ndarray,
    *,
    thresholds: dict[str, dict[str, float]],
    window_steps: int = 40,
) -> dict[str, float]:
    from testbed.policies.deadzone_eval import effective_direction_mask

    expert_eff = effective_direction_mask(expert_action, thresholds)
    policy_eff = effective_direction_mask(policy_action, thresholds)
    expert_any = expert_eff.any(axis=(1, 2))
    start = int(np.flatnonzero(expert_any)[0]) if np.any(expert_any) else 0
    end = min(start + int(window_steps), expert_action.shape[0])
    expert_slice = expert_eff[start:end]
    policy_slice = policy_eff[start:end]
    steps = max(0, end - start)
    expert_any_slice = expert_slice.any(axis=(1, 2)) if steps else np.zeros(0, dtype=bool)
    policy_any_slice = policy_slice.any(axis=(1, 2)) if steps else np.zeros(0, dtype=bool)
    same = (expert_slice & policy_slice).any(axis=(1, 2)) if steps else np.zeros(0, dtype=bool)
    extra = policy_any_slice & ~same if steps else np.zeros(0, dtype=bool)
    expert_count = max(int(expert_any_slice.sum()), 1)
    policy_count = max(int(policy_any_slice.sum()), 1)
    return {
        "policy_any_effective_pct": 100.0 * float(policy_any_slice.sum()) / max(steps, 1),
        "same_axis_dir_pct": 100.0 * float(same.sum()) / expert_count,
        "extra_or_wrong_pct": 100.0 * float(extra.sum()) / policy_count,
    }


def _mean_startup(rows: list[dict[str, float]]) -> dict[str, float]:
    return {
        "policy_any_effective_pct": float(np.mean([row["policy_any_effective_pct"] for row in rows])),
        "same_axis_dir_pct": float(np.mean([row["same_axis_dir_pct"] for row in rows])),
        "extra_or_wrong_pct": float(np.mean([row["extra_or_wrong_pct"] for row in rows])),
    }


def _selection_score(
    startup: dict[str, float],
    start40: dict[str, Any],
    main: dict[str, Any],
    metrics: dict[str, Any],
) -> float:
    return (
        1.0 * float(startup["policy_any_effective_pct"])
        + 0.8 * float(startup["same_axis_dir_pct"])
        + 0.4 * float(main.get("mean_same_axis_dir_effective_pct_of_expert_effective", 0.0))
        - 2.0 * float(start40.get("mean_policy_extra_or_wrong_effective_pct", 0.0))
        - 50.0 * float(metrics["mae"])
    )


def _choose_gate(rows: list[dict[str, Any]]) -> str:
    viable = [
        row
        for row in rows
        if float(row["start40_policy_any_effective_pct"]) <= 5.0
        and float(row["main_same_dir_pct"]) >= 90.0
        and float(row["startup_policy_any_effective_pct"]) >= 55.0
    ]
    candidates = viable or rows
    best = max(candidates, key=lambda row: float(row["score"]))
    return str(best["gate"])


def _materialize_gate(
    *,
    output_dir: Path,
    gate_name: str,
    episodes: list[dict[str, Any]],
    phase_probs: dict[str, np.ndarray],
    scan_rows: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    active_by_episode, inactive_scale = _active_masks_for_gate(
        gate_name,
        episodes=episodes,
        phase_probs=phase_probs,
    )
    results = []
    for ep in episodes:
        episode_id = str(ep["episode_id"])
        gated = apply_phase_gate_to_actions(
            ep["policy_action"],
            active_by_episode[episode_id],
            inactive_scale=inactive_scale,
        )
        episode_dir = output_dir / "episodes" / episode_id
        episode_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            episode_dir / "actions.npz",
            time_s=ep["time_s"],
            expert_action=ep["expert_action"],
            policy_action=gated,
        )
        results.append(
            {
                "episode_id": episode_id,
                "episode_path": episode_id,
                "n_steps": int(gated.shape[0]),
                "dt": 0.05,
                "expert_action": ep["expert_action"],
                "policy_action": gated,
                "metrics": compute_action_metrics(ep["expert_action"], gated),
            }
        )
    aggregate = aggregate_episode_results(results)
    matched_rows = [row for row in scan_rows if str(row["gate"]) == gate_name]
    metadata = {
        "selection_mode": "all_train_ready_oof_phase_gate",
        "gate_name": gate_name,
        "scan_row": matched_rows[0] if matched_rows else None,
    }
    write_collection_report(aggregate=aggregate, output_dir=output_dir, metadata=metadata)
    _write_json(output_dir / "gate_summary.json", metadata)


def _active_masks_for_gate(
    gate_name: str,
    *,
    episodes: list[dict[str, Any]],
    phase_probs: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], float]:
    base_name, inactive_scale = _split_inactive_scale(gate_name)
    if base_name.startswith("simple_"):
        threshold = float(base_name.split("_", 1)[1])
        return {
            str(ep["episode_id"]): phase_probs[str(ep["episode_id"])] >= threshold
            for ep in episodes
        }, inactive_scale
    if base_name.startswith("hyst_o"):
        open_text, close_text = base_name.removeprefix("hyst_o").split("_c", 1)
        open_threshold = float(open_text)
        close_threshold = float(close_text)
        return {
            str(ep["episode_id"]): build_hysteresis_mask(
                phase_probs[str(ep["episode_id"])],
                open_threshold=open_threshold,
                close_threshold=close_threshold,
            )
            for ep in episodes
        }, inactive_scale
    raise ValueError(f"unsupported gate name: {gate_name}")


def _simple_gate_name(threshold: float, inactive_scale: float) -> str:
    return _with_scale_suffix(f"simple_{threshold:.2f}", inactive_scale)


def _hysteresis_gate_name(open_threshold: float, close_threshold: float, inactive_scale: float) -> str:
    return _with_scale_suffix(f"hyst_o{open_threshold:.2f}_c{close_threshold:.2f}", inactive_scale)


def _with_scale_suffix(base_name: str, inactive_scale: float) -> str:
    scale = float(inactive_scale)
    if scale == 0.0:
        return base_name
    return f"{base_name}_s{scale:.2f}"


def _split_inactive_scale(gate_name: str) -> tuple[str, float]:
    if "_s" not in gate_name:
        return gate_name, 0.0
    base_name, scale_text = gate_name.rsplit("_s", 1)
    return base_name, float(scale_text)


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


def _parse_pairs(value: str) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for item in value.split(","):
        text = item.strip()
        if not text:
            continue
        open_text, close_text = text.split(":", 1)
        out.append((float(open_text), float(close_text)))
    return out


def _write_phase_probs(output_dir: Path, phase_probs: dict[str, np.ndarray]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for episode_id, probs in sorted(phase_probs.items()):
        np.savez_compressed(output_dir / f"{episode_id}.npz", phase_prob=np.asarray(probs, dtype=np.float32))


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


class _PhaseGateMlp(torch.nn.Module):
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
