"""Build the train-only A/B target-release supervision contract."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from testbed.data.dataset import _read_train_exclude_mask, _valid_start_indices
from testbed.data.open_loop_experiment import OpenLoopCalibration
from testbed.policies.act.target_release import (
    CONDITION_KEY,
    CONTRACT_SCHEMA,
    target_release_candidate_indices,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="tb-build-target-release-contract",
        description=(
            "Freeze the train-fold swing region where target A continues "
            "negative return and target B releases to zero."
        ),
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--train-ready-manifest", type=Path, required=True)
    parser.add_argument("--ready-contract", type=Path, required=True)
    parser.add_argument("--deadzone-thresholds", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-steps", type=int, default=20)
    parser.add_argument("--action-window-steps", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.chunk_steps <= 0 or args.action_window_steps <= 0:
        parser.error("window sizes must be positive")
    if args.action_window_steps > args.chunk_steps:
        parser.error("--action-window-steps cannot exceed --chunk-steps")
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"output already exists: {args.output}")

    contract = build_contract(
        dataset_dir=args.dataset_dir,
        split_manifest=args.split_manifest,
        train_ready_manifest=args.train_ready_manifest,
        ready_contract=args.ready_contract,
        deadzone_thresholds=args.deadzone_thresholds,
        chunk_steps=int(args.chunk_steps),
        action_window_steps=int(args.action_window_steps),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(contract, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "sha256": _sha256(args.output),
                "train_episode_count": len(contract["train_episode_ids"]),
                "candidate_count_by_side": contract["support_audit"][
                    "candidate_count_by_side"
                ],
            },
            ensure_ascii=False,
        )
    )


def build_contract(
    *,
    dataset_dir: Path,
    split_manifest: Path,
    train_ready_manifest: Path,
    ready_contract: Path,
    deadzone_thresholds: Path,
    chunk_steps: int,
    action_window_steps: int,
) -> dict[str, Any]:
    rows = json.loads(split_manifest.read_text(encoding="utf-8"))["episodes"]
    ready_ids = {
        int(value)
        for value in json.loads(
            train_ready_manifest.read_text(encoding="utf-8")
        )["train_ready_episode_ids"]
    }
    train_ids = sorted(
        int(row["episode_id"])
        for row in rows
        if row["split"] == "train" and int(row["episode_id"]) in ready_ids
    )
    if not train_ids:
        raise ValueError("target-release contract requires train-ready train episodes")
    calibration = OpenLoopCalibration.from_dataset(
        dataset_dir=dataset_dir,
        train_episode_ids=train_ids,
        ready_contract_path=ready_contract,
        deadzone_threshold_path=deadzone_thresholds,
    )
    decision_range = calibration.target_ranges["B"]
    decision = {
        "swing_axis_index": 0,
        "continue_target_side": "A",
        "stop_target_side": "B",
        "swing_qpos_range_rad": [float(value) for value in decision_range],
        "range_derivation": "train_B_endpoint_observed_min_max_plus_margin",
        "train_B_typical_q05_q95_plus_margin_rad": [
            float(value) for value in calibration.target_quantile_ranges["B"]
        ],
        "train_A_endpoint_range_rad": [
            float(value) for value in calibration.target_ranges["A"]
        ],
    }
    candidate_rule = {
        "after_swing_apex": True,
        "action_window_steps": int(action_window_steps),
        "stable_window_steps": int(calibration.stable_steps),
        "stable_qvel_abs_max_rad_s": float(calibration.stable_qvel_abs),
        "continue_rule": "all_window_swing_action_at_or_below_negative_deadzone",
        "stop_rule": "all_window_swing_action_sub_deadzone_and_stable_qvel_window",
    }
    candidate_config = {
        "enabled": True,
        "axis_index": 0,
        "decision_qpos_range": tuple(decision_range),
        "action_window_steps": int(action_window_steps),
        "stable_window_steps": int(calibration.stable_steps),
        "stable_qvel_abs": float(calibration.stable_qvel_abs),
        "positive_deadzone": float(calibration.deadzone_positive[0]),
        "negative_deadzone": float(calibration.deadzone_negative[0]),
    }
    counts: dict[int, int] = {}
    sides: dict[int, str] = {}
    endpoint_by_side: dict[str, list[float]] = {"A": [], "B": []}
    candidate_qpos_by_side: dict[str, list[float]] = {"A": [], "B": []}
    continue_action_abs_values: list[float] = []
    for episode_id in train_ids:
        path = dataset_dir / f"episode_{episode_id}.hdf5"
        with h5py.File(path, "r") as handle:
            qpos = np.asarray(handle["observations/qpos"][()], dtype=np.float32)
            qvel = np.asarray(handle["observations/qvel"][()], dtype=np.float32)
            action = np.asarray(handle["action"][()], dtype=np.float32)
            condition = np.asarray(
                handle[f"conditions/{CONDITION_KEY}"][()], dtype=np.float32
            )
            valid_starts = _valid_start_indices(
                total_steps=len(qpos),
                train_exclude_mask=_read_train_exclude_mask(handle, len(qpos)),
                action_chunk_size=int(chunk_steps),
            )
            candidates = target_release_candidate_indices(
                qpos=qpos,
                qvel=qvel,
                actions=action,
                condition=condition,
                valid_starts=valid_starts,
                condition_valid_mask=np.asarray(
                    handle["conditions/valid_mask"][()], dtype=bool
                ),
                config=candidate_config,
            )
        side = "A" if float(condition[0, 0]) < 0.0 else "B"
        counts[episode_id] = int(candidates.size)
        sides[episode_id] = side
        endpoint_by_side[side].append(float(qpos[-1, 0]))
        candidate_qpos_by_side[side].extend(
            float(value) for value in qpos[candidates, 0]
        )
        if side == "A":
            # Overlapping windows must not give central frames extra weight.
            # The selected target is the median of unique demonstrated
            # effective return frames, so it stays inside ordinary expert
            # behaviour while retaining real margin above the deadzone.
            unique_action_frames = np.unique(
                np.concatenate(
                    [
                        np.arange(
                            int(start), int(start) + int(action_window_steps)
                        )
                        for start in candidates
                    ]
                )
            )
            continue_action_abs_values.extend(
                float(value) for value in np.abs(action[unique_action_frames, 0])
            )
    missing = [episode_id for episode_id in train_ids if counts[episode_id] <= 0]
    if missing:
        raise ValueError(
            "train fold lacks target-release support for episode ids: "
            + ", ".join(str(value) for value in missing)
        )

    count_by_side = {
        side: int(sum(counts[eid] for eid in train_ids if sides[eid] == side))
        for side in ("A", "B")
    }
    episode_count_by_side = {
        side: int(sum(sides[eid] == side for eid in train_ids))
        for side in ("A", "B")
    }
    continue_quantile_levels = np.asarray(
        [0.0, 0.05, 0.10, 0.50, 0.90, 0.95, 1.0], dtype=np.float64
    )
    continue_quantiles = np.quantile(
        np.asarray(continue_action_abs_values, dtype=np.float64),
        continue_quantile_levels,
    )
    decision["continue_action_target_abs"] = float(continue_quantiles[3])
    decision["continue_action_target_derivation"] = (
        "train_A_unique_candidate_frame_abs_action_q50"
    )
    return {
        "schema": CONTRACT_SCHEMA,
        "schema_version": 1,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "condition_schema": CONDITION_KEY,
        "dataset_dir": str(dataset_dir.resolve()),
        "train_episode_ids": train_ids,
        "provenance": {
            "split_manifest_path": str(split_manifest.resolve()),
            "split_manifest_sha256": _sha256(split_manifest),
            "train_ready_manifest_path": str(train_ready_manifest.resolve()),
            "train_ready_manifest_sha256": _sha256(train_ready_manifest),
            "ready_contract_path": str(ready_contract.resolve()),
            "ready_contract_sha256": _sha256(ready_contract),
            "deadzone_threshold_path": str(deadzone_thresholds.resolve()),
            "deadzone_threshold_sha256": _sha256(deadzone_thresholds),
        },
        "decision_region": decision,
        "candidate_rule": candidate_rule,
        "mechanical_deadzone": {
            "action_domain": "direct_policy_output",
            "swing": {
                "pos": float(calibration.deadzone_positive[0]),
                "neg": float(calibration.deadzone_negative[0]),
            },
        },
        "support_audit": {
            "all_train_episodes_supported": True,
            "episode_count_by_side": episode_count_by_side,
            "candidate_count_by_side": count_by_side,
            "candidate_count_by_episode": {
                str(episode_id): counts[episode_id] for episode_id in train_ids
            },
            "endpoint_min_max_by_side_rad": {
                side: [
                    float(np.min(endpoint_by_side[side])),
                    float(np.max(endpoint_by_side[side])),
                ]
                for side in ("A", "B")
            },
            "candidate_qpos_min_max_by_side_rad": {
                side: [
                    float(np.min(candidate_qpos_by_side[side])),
                    float(np.max(candidate_qpos_by_side[side])),
                ]
                for side in ("A", "B")
            },
            "continue_action_abs_quantiles": {
                f"q{int(round(level * 100)):02d}": float(value)
                for level, value in zip(
                    continue_quantile_levels, continue_quantiles, strict=True
                )
            },
        },
        "evidence_boundary": (
            "The contract is derived from train-ready train episodes only. "
            "It supervises the supported B-side return-release decision and "
            "does not invent a positive correction from the A endpoint."
        ),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
