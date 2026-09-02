"""Held-observation deadzone liveness replay for goal-conditioned ACT.

At every recorded inactive-to-effective transition, the policy history is
rebuilt from the recorded prefix.  The branch then repeats the anchor image and
qpos, forces qvel to zero, and asks whether the policy can reproduce the same
axis/direction without borrowing the expert's next observation.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml

from testbed.actions.policy import (
    _policy_obs_from_real_obs,
    load_act_policy_from_bundle,
)
from testbed.data.dataset import (
    REAL_TRANSITION_CONDITION_KEY,
    _read_camera_image,
    _read_train_exclude_mask,
    _state_hold_transition_mask,
    _valid_start_indices,
)
from testbed.data.state_hold_transition import intersect_transition_starts
from testbed.policies.deadzone_eval import (
    effective_direction_mask,
    load_deadzone_thresholds,
)
from testbed.tasks.real_transition_phase import CYCLE_PHASE_KEY
from testbed.tasks.real_transition_return_commit import RETURN_COMMIT_KEY

CAMERAS = ("video4", "video5", "video6", "video7")
SPLITS = ("validation", "locked_test")
AXES = ("swing", "boom", "stick", "bucket")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="tb-state-hold-transition-replay",
        description=(
            "Freeze held-out transition observations and test natural "
            "same-direction deadzone crossing without expert state advance."
        ),
    )
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--train-ready-manifest", type=Path, default=None)
    parser.add_argument("--split-manifest", type=Path, default=None)
    parser.add_argument("--deadzone-thresholds", type=Path, default=None)
    parser.add_argument("--split", choices=("validation", "locked_test", "both"), default="both")
    parser.add_argument("--hold-horizon-steps", type=int, default=20)
    parser.add_argument("--short-window-steps", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint-name", default="policy_best.ckpt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.hold_horizon_steps <= 0 or args.short_window_steps <= 0:
        parser.error("state-hold windows must be positive")
    if args.short_window_steps > args.hold_horizon_steps:
        parser.error("--short-window-steps cannot exceed --hold-horizon-steps")
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"output already exists: {args.output}")

    report = evaluate_bundle(
        bundle_dir=args.bundle_dir,
        dataset_dir=args.dataset_dir,
        train_ready_manifest=args.train_ready_manifest,
        split_manifest=args.split_manifest,
        deadzone_thresholds=args.deadzone_thresholds,
        split=str(args.split),
        hold_horizon_steps=int(args.hold_horizon_steps),
        short_window_steps=int(args.short_window_steps),
        device=str(args.device),
        checkpoint_name=str(args.checkpoint_name),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "summary": report["summary"]}))


def evaluate_bundle(
    *,
    bundle_dir: Path,
    dataset_dir: Path | None,
    train_ready_manifest: Path | None,
    split_manifest: Path | None,
    deadzone_thresholds: Path | None,
    split: str,
    hold_horizon_steps: int,
    short_window_steps: int,
    device: str,
    checkpoint_name: str,
) -> dict[str, Any]:
    bundle = bundle_dir.resolve()
    resolved = yaml.safe_load(
        (bundle / "resolved_config.yaml").read_text(encoding="utf-8")
    ) or {}
    task = dict(resolved.get("task", {}) or {})
    episodes = (dataset_dir or Path(str(task["dataset_dir"]))).resolve()
    ready_path = train_ready_manifest or Path(
        str(task["train_ready_manifest_path"])
    )
    split_path = split_manifest or Path(str(task["split_manifest_path"]))
    threshold_path = deadzone_thresholds or _bundle_deadzone_path(bundle, resolved)
    checkpoint_path = bundle / checkpoint_name
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint_path}")
    thresholds = load_deadzone_thresholds(threshold_path)
    ready_ids = {
        int(value)
        for value in json.loads(ready_path.read_text(encoding="utf-8"))[
            "train_ready_episode_ids"
        ]
    }
    selected_splits = SPLITS if split == "both" else (split,)
    rows = json.loads(split_path.read_text(encoding="utf-8"))["episodes"]
    selected = [
        (int(row["episode_id"]), str(row["split"]))
        for row in rows
        if int(row["episode_id"]) in ready_ids
        and str(row["split"]) in selected_splits
    ]
    policy = load_act_policy_from_bundle(
        bundle_dir=bundle,
        ckpt_path=checkpoint_path,
        device=device,
        temporal_agg=True,
    )
    result_rows: list[dict[str, Any]] = []
    try:
        for episode_id, episode_split in selected:
            result_rows.extend(
                _evaluate_episode(
                    policy=policy,
                    episode_path=episodes / f"episode_{episode_id}.hdf5",
                    episode_id=episode_id,
                    episode_split=episode_split,
                    thresholds=thresholds,
                    hold_horizon_steps=hold_horizon_steps,
                    short_window_steps=short_window_steps,
                )
            )
    finally:
        close = getattr(policy, "close", None)
        if callable(close):
            close()
    return {
        "schema": "state_hold_transition_replay_v1",
        "bundle_dir": str(bundle),
        "checkpoint_path": str(checkpoint_path),
        "dataset_dir": str(episodes),
        "train_ready_manifest": str(ready_path.resolve()),
        "split_manifest": str(split_path.resolve()),
        "deadzone_thresholds": str(threshold_path.resolve()),
        "hold_horizon_steps": int(hold_horizon_steps),
        "short_window_steps": int(short_window_steps),
        "summary": _summaries(result_rows),
        "rows": result_rows,
        "evidence_boundary": (
            "Recorded prefix observations rebuild ACT temporal state. At each "
            "anchor the image/qpos are held and qvel is zero; model actions do "
            "not advance the state. A pass falsifies an absorbing software "
            "deadlock but does not prove hydraulic response."
        ),
    }


def _evaluate_episode(
    *,
    policy: Any,
    episode_path: Path,
    episode_id: int,
    episode_split: str,
    thresholds: dict[str, dict[str, float]],
    hold_horizon_steps: int,
    short_window_steps: int,
) -> list[dict[str, Any]]:
    with h5py.File(episode_path, "r") as handle:
        qpos = np.asarray(handle["observations/qpos"][()], dtype=np.float32)
        qvel = np.asarray(handle["observations/qvel"][()], dtype=np.float32)
        action = np.asarray(handle["action"][()], dtype=np.float32)
        condition = np.asarray(
            handle[f"conditions/{REAL_TRANSITION_CONDITION_KEY}"][()],
            dtype=np.float32,
        )
        cycle_phase = (
            np.asarray(
                handle[f"conditions/{CYCLE_PHASE_KEY}"][()], dtype=np.float32
            )
            if f"conditions/{CYCLE_PHASE_KEY}" in handle
            else np.zeros((len(action), 1), dtype=np.float32)
        )
        if cycle_phase.shape != (len(action), 1):
            raise ValueError(
                f"episode {episode_id} cycle phase must have shape (T, 1)"
            )
        return_commit = (
            np.asarray(
                handle[f"conditions/{RETURN_COMMIT_KEY}"][()], dtype=np.float32
            )
            if f"conditions/{RETURN_COMMIT_KEY}" in handle
            else np.zeros((len(action), 1), dtype=np.float32)
        )
        if return_commit.shape != (len(action), 1):
            raise ValueError(
                f"episode {episode_id} return commit must have shape (T, 1)"
            )
        transition_mask = _state_hold_transition_mask(
            handle,
            actions=action,
            config={"thresholds": thresholds},
        )
        valid_starts = _valid_start_indices(
            total_steps=len(action),
            train_exclude_mask=_read_train_exclude_mask(handle, len(action)),
            action_chunk_size=1,
        )
        starts = intersect_transition_starts(
            transition_mask,
            valid_starts,
            total_steps=len(action),
            hold_horizon_steps=1,
        )
        anchors_by_step: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for start in starts:
            for axis, direction in np.argwhere(transition_mask[int(start)]):
                anchors_by_step[int(start)].append((int(axis), int(direction)))
        first_anchor = int(starts[0]) if starts.size else None
        policy.reset()
        phase_enabled = CYCLE_PHASE_KEY in list(
            getattr(policy, "low_dim_keys", ()) or ()
        )
        active_phase = 0.0
        return_commit_enabled = RETURN_COMMIT_KEY in list(
            getattr(policy, "low_dim_keys", ()) or ()
        )
        active_return_commit = 0.0
        output: list[dict[str, Any]] = []
        for step in range(len(action)):
            next_phase = float(cycle_phase[step, 0])
            if phase_enabled and next_phase != active_phase:
                policy.reset()
                active_phase = next_phase
            next_return_commit = float(return_commit[step, 0])
            if (
                return_commit_enabled
                and next_return_commit != active_return_commit
            ):
                policy.reset()
                active_return_commit = next_return_commit
            observation = {
                "qpos": qpos[step],
                "qvel": qvel[step],
                "images": {
                    camera: _read_camera_image(handle, camera, step)
                    for camera in CAMERAS
                },
                REAL_TRANSITION_CONDITION_KEY: condition[step],
                CYCLE_PHASE_KEY: cycle_phase[step],
                RETURN_COMMIT_KEY: return_commit[step],
            }
            policy_observation = _policy_obs_from_real_obs(
                observation, camera_names=CAMERAS
            )
            if step in anchors_by_step:
                prefix_state = policy.snapshot_state()
                for axis, direction in anchors_by_step[step]:
                    policy.restore_state(prefix_state)
                    held = dict(policy_observation)
                    held["qvel"] = np.zeros_like(qvel[step])
                    trace = np.stack(
                        [
                            np.asarray(policy.predict(held), dtype=np.float32)
                            for _ in range(hold_horizon_steps)
                        ],
                        axis=0,
                    )
                    metrics = _trace_metrics(
                        trace=trace,
                        thresholds=thresholds,
                        target_axis=axis,
                        target_direction=direction,
                        short_window_steps=short_window_steps,
                        allowed_effective=effective_direction_mask(
                            action[step : step + 1], thresholds
                        )[0],
                    )
                    output.append(
                        {
                            "episode_id": int(episode_id),
                            "split": episode_split,
                            "anchor_step": int(step),
                            "anchor_group": (
                                "startup" if step == first_anchor else "mid_cycle"
                            ),
                            "axis_index": int(axis),
                            "axis": AXES[axis],
                            "direction": "pos" if direction == 0 else "neg",
                            "expert_action": float(action[step, axis]),
                            "condition": condition[step].astype(float).tolist(),
                            "cycle_phase": float(cycle_phase[step, 0]),
                            "return_commit": float(return_commit[step, 0]),
                            "held_qpos": qpos[step].astype(float).tolist(),
                            "held_qvel_zero": True,
                            "action_trace": trace.astype(float).tolist(),
                            **metrics,
                        }
                    )
                policy.restore_state(prefix_state)
            # Advance only the teacher-forced prefix after every held branch
            # has restored the pre-anchor cache snapshot.
            policy.predict(policy_observation)
    return output


def _trace_metrics(
    *,
    trace: np.ndarray,
    thresholds: dict[str, dict[str, float]],
    target_axis: int,
    target_direction: int,
    short_window_steps: int,
    allowed_effective: np.ndarray | None = None,
) -> dict[str, Any]:
    effective = effective_direction_mask(trace, thresholds)
    target = effective[:, target_axis, target_direction]
    target_indices = np.flatnonzero(target)
    allowed = np.zeros(effective.shape[1:], dtype=bool)
    allowed[target_axis, target_direction] = True
    if allowed_effective is not None:
        supplied = np.asarray(allowed_effective, dtype=bool)
        if supplied.shape != allowed.shape:
            raise ValueError(
                f"allowed_effective must have shape {allowed.shape}, got {supplied.shape}"
            )
        allowed |= supplied
    other = effective & ~allowed.reshape(1, *allowed.shape)
    opposite = effective[:, target_axis, 1 - target_direction]
    delay = int(target_indices[0]) if target_indices.size else None
    short = min(int(short_window_steps), len(trace))
    return {
        "target_reproduced_within_short_window": bool(target[:short].any()),
        "target_reproduced_within_horizon": bool(target.any()),
        "target_reproduction_delay_ticks": delay,
        "query0_target_effective": bool(target[0]),
        "query0_opposite_effective": bool(opposite[0]),
        "query0_other_effective": bool(other[0].any()),
        "horizon_opposite_effective_ticks": int(opposite.sum()),
        "horizon_other_effective_ticks": int(other.any(axis=(1, 2)).sum()),
    }


def _summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for split in SPLITS:
        split_rows = [row for row in rows if row["split"] == split]
        for group in ("overall", "startup", "mid_cycle"):
            group_rows = (
                split_rows
                if group == "overall"
                else [row for row in split_rows if row["anchor_group"] == group]
            )
            count = len(group_rows)
            result.append(
                {
                    "split": split,
                    "group": group,
                    "anchor_count": count,
                    "same_direction_within_5_rate": _mean_bool(
                        group_rows, "target_reproduced_within_short_window"
                    ),
                    "same_direction_within_20_rate": _mean_bool(
                        group_rows, "target_reproduced_within_horizon"
                    ),
                    "query0_wrong_effective_count": sum(
                        bool(row["query0_opposite_effective"])
                        or bool(row["query0_other_effective"])
                        for row in group_rows
                    ),
                    "episode_count": len(
                        {int(row["episode_id"]) for row in group_rows}
                    ),
                }
            )
    return result


def _mean_bool(rows: list[dict[str, Any]], key: str) -> float:
    return 0.0 if not rows else float(np.mean([bool(row[key]) for row in rows]))


def _bundle_deadzone_path(bundle: Path, resolved: dict[str, Any]) -> Path:
    raw = dict(resolved.get("train", {}) or {}).get("deadzone_loss", {})
    path = Path(str(raw["threshold_json"]))
    if path.is_file():
        return path
    bundled = bundle / path.name
    if bundled.is_file():
        return bundled
    raise FileNotFoundError(f"deadzone threshold file does not exist: {path}")


if __name__ == "__main__":
    main()
