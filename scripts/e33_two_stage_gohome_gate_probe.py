#!/usr/bin/env python3
"""Evaluate two-stage gohome gates: tail candidate first, eligibility second."""

from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts.e32_gohome_eligibility_probe import (
    FEATURE_NAMES,
    _classification_metrics,
    _load_episode,
    _parse_float_list,
    _parse_int_list,
    _train_episode_heldout,
    _train_final_model,
    _write_csv,
    _write_json,
)
from testbed.policies.gohome_eligibility import (
    aggregate_gohome_event_rows,
    consecutive_active_mask,
    gated_active_mask,
    gohome_event_metrics_from_active_mask,
)
from testbed.policies.offline_eval import load_train_ready_episode_ids
from testbed.policies.phase_gate import phase_gate_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff-dataset-dir", type=Path, required=True)
    parser.add_argument("--intent-prob-dir", type=Path, required=True)
    parser.add_argument("--eligibility-prob-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--candidate-thresholds", default="0.90,0.95,0.97,0.99")
    parser.add_argument("--eligibility-thresholds", default="0.80,0.90,0.95")
    parser.add_argument("--candidate-consecutive-steps", default="5,8,10")
    parser.add_argument("--eligibility-consecutive-steps", default="3,4,5")
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
    eligibility_probs = _load_eligibility_probs(args.eligibility_prob_dir, episode_ids)
    candidate_episodes = [_as_candidate_episode(ep) for ep in episodes]

    fold_rows, candidate_probs = _train_episode_heldout(
        episodes=candidate_episodes,
        folds=int(args.folds),
        epochs=int(args.epochs),
        hidden_dim=int(args.hidden_dim),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        seed=int(args.seed),
    )
    _write_json(output_dir / "candidate_fold_summary.json", fold_rows)
    _write_candidate_probs(output_dir / "candidate_probs", candidate_probs)

    final_payload = _train_final_model(
        episodes=candidate_episodes,
        epochs=int(args.epochs),
        hidden_dim=int(args.hidden_dim),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        seed=int(args.seed),
    )
    torch.save(final_payload, output_dir / "tail_candidate_model.pt")
    _write_json(
        output_dir / "tail_candidate_model_metadata.json",
        phase_gate_metadata(
            feature_names=FEATURE_NAMES,
            open_threshold=0.0,
            close_threshold=0.0,
            extra={
                "model": "mlp_16_32_1",
                "label": "handoff/tail_idle_mask",
                "training": "all train-ready episodes; use OOF probabilities for reported gates",
                "handoff_dataset_dir": str(args.handoff_dataset_dir),
                "intent_prob_dir": str(args.intent_prob_dir),
                "eligibility_prob_dir": str(args.eligibility_prob_dir),
                "manifest": str(args.manifest),
            },
        ),
    )

    scan_rows, events_by_gate = _scan_two_stage_gates(
        episodes=episodes,
        eligibility_probs=eligibility_probs,
        candidate_probs=candidate_probs,
        candidate_thresholds=_parse_float_list(args.candidate_thresholds),
        eligibility_thresholds=_parse_float_list(args.eligibility_thresholds),
        candidate_consecutive_steps=_parse_int_list(args.candidate_consecutive_steps),
        eligibility_consecutive_steps=_parse_int_list(args.eligibility_consecutive_steps),
    )
    _write_csv(output_dir / "threshold_scan.csv", scan_rows)

    materialize = str(args.materialize)
    if materialize == "auto":
        materialize = _choose_gate(scan_rows)
    selected_rows = events_by_gate[materialize]
    _write_csv(output_dir / f"{materialize}_events.csv", selected_rows)
    _write_json(
        output_dir / "gate_summary.json",
        {
            "gate": materialize,
            "scan_row": next(row for row in scan_rows if str(row["gate"]) == materialize),
        },
    )
    print(f"Candidate fold summary: {output_dir / 'candidate_fold_summary.json'}")
    print(f"Threshold scan: {output_dir / 'threshold_scan.csv'}")
    print(f"Selected events: {output_dir / f'{materialize}_events.csv'}")


def _as_candidate_episode(episode: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(episode)
    candidate_label = np.asarray(episode["tail_idle_mask"], dtype=bool)
    out["label"] = candidate_label.astype(np.float32)
    out["label_bool"] = candidate_label
    return out


def _load_eligibility_probs(prob_dir: Path, episode_ids: list[str]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for episode_id in episode_ids:
        path = Path(prob_dir) / f"{episode_id}.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        with np.load(path) as data:
            out[str(episode_id)] = np.asarray(data["eligibility_prob"], dtype=np.float32)
    return out


def _write_candidate_probs(output_dir: Path, probs_by_episode: dict[str, np.ndarray]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for episode_id, probs in sorted(probs_by_episode.items()):
        np.savez_compressed(output_dir / f"{episode_id}.npz", candidate_prob=np.asarray(probs, dtype=np.float32))


def _scan_two_stage_gates(
    *,
    episodes: list[dict[str, Any]],
    eligibility_probs: dict[str, np.ndarray],
    candidate_probs: dict[str, np.ndarray],
    candidate_thresholds: list[float],
    eligibility_thresholds: list[float],
    candidate_consecutive_steps: list[int],
    eligibility_consecutive_steps: list[int],
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    rows: list[dict[str, Any]] = []
    events_by_gate: dict[str, list[dict[str, Any]]] = {}
    for eligibility_threshold in eligibility_thresholds:
        for eligibility_consecutive in eligibility_consecutive_steps:
            oracle_gate = f"oracle_tail_e{eligibility_threshold:.2f}_ec{int(eligibility_consecutive)}"
            oracle_rows = _event_rows_for_gate(
                gate=oracle_gate,
                episodes=episodes,
                eligibility_probs=eligibility_probs,
                candidate_active_by_episode={
                    str(ep["episode_id"]): np.asarray(ep["tail_idle_mask"], dtype=bool) for ep in episodes
                },
                eligibility_threshold=eligibility_threshold,
                eligibility_consecutive_steps=eligibility_consecutive,
            )
            row = {
                "gate": oracle_gate,
                "candidate_source": "oracle_tail",
                "candidate_threshold": "",
                "candidate_consecutive_steps": "",
                "eligibility_threshold": float(eligibility_threshold),
                "eligibility_consecutive_steps": int(eligibility_consecutive),
                **aggregate_gohome_event_rows(oracle_rows),
            }
            row["score"] = _selection_score(row)
            rows.append(row)
            events_by_gate[oracle_gate] = oracle_rows

            for candidate_threshold in candidate_thresholds:
                for candidate_consecutive in candidate_consecutive_steps:
                    gate = (
                        f"learned_tail_t{candidate_threshold:.2f}_tc{int(candidate_consecutive)}"
                        f"_e{eligibility_threshold:.2f}_ec{int(eligibility_consecutive)}"
                    )
                    candidate_active_by_episode = {
                        str(ep["episode_id"]): consecutive_active_mask(
                            candidate_probs[str(ep["episode_id"])],
                            threshold=candidate_threshold,
                            consecutive_steps=candidate_consecutive,
                        )
                        for ep in episodes
                    }
                    event_rows = _event_rows_for_gate(
                        gate=gate,
                        episodes=episodes,
                        eligibility_probs=eligibility_probs,
                        candidate_active_by_episode=candidate_active_by_episode,
                        eligibility_threshold=eligibility_threshold,
                        eligibility_consecutive_steps=eligibility_consecutive,
                    )
                    candidate_labels, candidate_prob = _flatten_candidate_probs(episodes, candidate_probs)
                    candidate_metrics = _classification_metrics(
                        candidate_labels,
                        candidate_prob,
                        threshold=candidate_threshold,
                    )
                    row = {
                        "gate": gate,
                        "candidate_source": "learned_tail",
                        "candidate_threshold": float(candidate_threshold),
                        "candidate_consecutive_steps": int(candidate_consecutive),
                        "eligibility_threshold": float(eligibility_threshold),
                        "eligibility_consecutive_steps": int(eligibility_consecutive),
                        **aggregate_gohome_event_rows(event_rows),
                        **{f"candidate_frame_{key}": value for key, value in candidate_metrics.items()},
                    }
                    row["score"] = _selection_score(row)
                    rows.append(row)
                    events_by_gate[gate] = event_rows
    return rows, events_by_gate


def _event_rows_for_gate(
    *,
    gate: str,
    episodes: list[dict[str, Any]],
    eligibility_probs: dict[str, np.ndarray],
    candidate_active_by_episode: dict[str, np.ndarray],
    eligibility_threshold: float,
    eligibility_consecutive_steps: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ep in episodes:
        episode_id = str(ep["episode_id"])
        active = gated_active_mask(
            candidate_active=candidate_active_by_episode[episode_id],
            eligibility_probability=eligibility_probs[episode_id],
            eligibility_threshold=eligibility_threshold,
            eligibility_consecutive_steps=eligibility_consecutive_steps,
        )
        rows.append(
            gohome_event_metrics_from_active_mask(
                episode_id=episode_id,
                active_mask=active,
                eligible_label=np.asarray(ep["label_bool"], dtype=bool),
                loss_mask=np.asarray(ep["loss_mask"], dtype=bool),
                tail_idle_mask=np.asarray(ep["tail_idle_mask"], dtype=bool),
                gate=gate,
            )
        )
    return rows


def _flatten_candidate_probs(
    episodes: list[dict[str, Any]],
    probs_by_episode: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    labels: list[np.ndarray] = []
    probs: list[np.ndarray] = []
    for ep in episodes:
        mask = np.asarray(ep["loss_mask"], dtype=bool)
        episode_id = str(ep["episode_id"])
        labels.append(np.asarray(ep["tail_idle_mask"], dtype=np.float32)[mask])
        probs.append(np.asarray(probs_by_episode[episode_id], dtype=np.float32)[mask])
    return np.concatenate(labels, axis=0), np.concatenate(probs, axis=0)


def _selection_score(row: dict[str, Any]) -> float:
    mean_delay = row.get("mean_detection_delay_steps")
    delay_penalty = float(mean_delay) / 20.0 if mean_delay != "" else 1.0
    pre_tail_penalty = 20.0 * float(row["pre_tail_false_positive_episode_rate"])
    early_penalty = 2.0 * float(row["early_false_positive_episode_rate"])
    return 4.0 * float(row["event_recall"]) - pre_tail_penalty - early_penalty - delay_penalty


def _choose_gate(rows: list[dict[str, Any]]) -> str:
    learned = [row for row in rows if row["candidate_source"] == "learned_tail"]
    viable = [
        row
        for row in learned
        if float(row["pre_tail_false_positive_episode_rate"]) == 0.0
        and float(row["event_recall"]) >= 0.8
    ]
    candidates = viable or learned or rows
    best = max(candidates, key=lambda row: float(row["score"]))
    return str(best["gate"])


if __name__ == "__main__":
    main()
