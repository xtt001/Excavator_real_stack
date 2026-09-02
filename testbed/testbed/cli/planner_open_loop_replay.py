"""Planner-driven open-loop replay for real-transition ACT checkpoints.

The replay intentionally does not invent plant dynamics.  It feeds the policy
the recorded qpos/qvel and camera observations at 20 Hz, while the planner
commits each target from the real source-run sequence.  This is the safest
offline answer to the question "would the policy issue the right commands in
the observed cycle?" after the fitted linear plant failed an expert-action
reproduction check.  It exercises the real lifecycle, action guard, deadzone
labels, target geometry, and continuous planner composition; it does not claim
that the recorded state would remain unchanged after replacing the operator.
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
    PolicyActionSource,
    _policy_obs_from_real_obs,
    load_act_policy_from_bundle,
)
from testbed.data.dataset import _read_camera_image
from testbed.data.open_loop_experiment import OpenLoopCalibration
from testbed.runtime.guard import ActionGuard
from testbed.tasks.act_cycle_planner import ScriptCyclePlanner
from testbed.tasks.real_transition_excursion import (
    EXCURSION_OBSERVED_KEY,
    derive_excursion_observed,
)
from testbed.tasks.real_transition_phase import CYCLE_PHASE_KEY
from testbed.tasks.real_transition_return_commit import RETURN_COMMIT_KEY

CAMERAS = ("video4", "video5", "video6", "video7")
MODES = ("continuous", "per_goal_reset")
SPLITS = ("validation", "locked_test")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="tb-planner-open-loop-replay",
        description=(
            "Replay planner-conditioned ACT on recorded observations and "
            "score dig/return/deadzone command phases."
        ),
    )
    parser.add_argument("--bundle-dir", type=Path, action="append", required=True)
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--train-ready-manifest", type=Path, default=None)
    parser.add_argument("--cycle-manifest", type=Path, default=None)
    parser.add_argument("--ready-contract", type=Path, required=True)
    parser.add_argument("--deadzone-thresholds", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint-name", default="policy_best.ckpt")
    parser.add_argument(
        "--split", choices=("validation", "locked_test", "both"), default="both"
    )
    parser.add_argument("--mode", choices=MODES + ("both",), default="continuous")
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.max_runs is not None and args.max_runs <= 0:
        parser.error("--max-runs must be positive")

    reports = [
        evaluate_bundle(
            bundle_dir=path,
            dataset_dir=args.dataset_dir,
            train_ready_manifest=args.train_ready_manifest,
            cycle_manifest=args.cycle_manifest,
            ready_contract=args.ready_contract,
            deadzone_thresholds=args.deadzone_thresholds,
            device=str(args.device),
            checkpoint_name=str(args.checkpoint_name),
            split=str(args.split),
            mode=str(args.mode),
            max_runs=args.max_runs,
        )
        for path in args.bundle_dir
    ]
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"output already exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "schema": "planner_open_loop_replay_v1",
                "reports": reports,
                "boundary": (
                    "The policy receives recorded held-out observations, and "
                    "its action does not update qpos/qvel. Planner lifecycle and "
                    "action-layer diagnostics are exercised, but this is not "
                    "a physical closed-loop or hydraulic-effect proof."
                ),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "reports": reports}, ensure_ascii=False))


def evaluate_bundle(
    *,
    bundle_dir: Path,
    dataset_dir: Path | None,
    train_ready_manifest: Path | None,
    cycle_manifest: Path | None,
    ready_contract: Path,
    deadzone_thresholds: Path | None,
    device: str,
    checkpoint_name: str,
    split: str,
    mode: str,
    max_runs: int | None,
) -> dict[str, Any]:
    bundle_dir = bundle_dir.resolve()
    checkpoint_path = bundle_dir / checkpoint_name
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint_path}")
    resolved = yaml.safe_load(
        (bundle_dir / "resolved_config.yaml").read_text(encoding="utf-8")
    ) or {}
    task_cfg = dict(resolved.get("task", {}) or {})
    episodes_dir = dataset_dir or Path(str(task_cfg["dataset_dir"]))
    ready_path = train_ready_manifest or Path(
        str(task_cfg["train_ready_manifest_path"])
    )
    cycle_path = cycle_manifest or episodes_dir.parent / "cycle_manifest.jsonl"
    ready_ids = _read_ready_ids(ready_path)
    rows = _read_cycle_rows(cycle_path, ready_ids)
    train_rows = [row for row in rows if row["split"] == "train"]
    if not train_rows:
        raise ValueError("open-loop replay requires train rows for calibration")
    calibration = OpenLoopCalibration.from_dataset(
        dataset_dir=episodes_dir,
        train_episode_ids=[row["episode_id"] for row in train_rows],
        ready_contract_path=ready_contract,
        deadzone_threshold_path=(
            deadzone_thresholds
            if deadzone_thresholds is not None
            else _resolve_bundle_deadzone_thresholds(bundle_dir)
        ),
    )
    target_release_contract = _resolve_bundle_target_release_contract(
        bundle_dir=bundle_dir,
        resolved=resolved,
    )
    target_release_range = (
        tuple(
            float(value)
            for value in target_release_contract["decision_region"][
                "swing_qpos_range_rad"
            ]
        )
        if target_release_contract is not None
        else tuple(calibration.target_ranges["B"])
    )
    qvel_input = "qvel" in list((resolved.get("policy", {}) or {}).get("low_dim_keys", ()))
    selected_splits = SPLITS if split == "both" else (split,)
    selected_modes = MODES if mode == "both" else (mode,)
    reports = []
    for selected_mode in selected_modes:
        run_reports: list[dict[str, Any]] = []
        policy = load_act_policy_from_bundle(
            bundle_dir=bundle_dir,
            ckpt_path=checkpoint_path,
            device=device,
            temporal_agg=True,
        )
        counterfactual_policy = load_act_policy_from_bundle(
            bundle_dir=bundle_dir,
            ckpt_path=checkpoint_path,
            device=device,
            temporal_agg=False,
        )
        try:
            for selected_split in selected_splits:
                groups = _complete_run_groups(rows, selected_split)
                if max_runs is not None:
                    groups = groups[: int(max_runs)]
                for group in groups:
                    run_reports.append(
                        _evaluate_run(
                            policy=policy,
                            counterfactual_policy=counterfactual_policy,
                            bundle_dir=bundle_dir,
                            episodes_dir=episodes_dir,
                            rows=group,
                            calibration=calibration,
                            target_release_range=target_release_range,
                            qvel_input=qvel_input,
                            reset_policy_on_goal=selected_mode == "per_goal_reset",
                        )
                    )
        finally:
            for loaded_policy in (policy, counterfactual_policy):
                close = getattr(loaded_policy, "close", None)
                if callable(close):
                    close()
        reports.append(
            _summarize_mode(
                bundle_dir=bundle_dir,
                mode=selected_mode,
                qvel_input=qvel_input,
                calibration=calibration,
                run_reports=run_reports,
            )
        )
    return {
        "bundle_dir": str(bundle_dir),
        "checkpoint_path": str(checkpoint_path),
        "condition_action_weight": float(
            (resolved.get("train", {}).get("condition_action_loss", {}) or {}).get(
                "weight", 0.0
            )
        ),
        "target_release_weight": float(
            (resolved.get("train", {}).get("target_release_loss", {}) or {}).get(
                "weight", 0.0
            )
        ),
        "qvel_input": qvel_input,
        "train_episode_count": len(train_rows),
        "target_release_decision_range": list(target_release_range),
        "target_release_contract": target_release_contract,
        "calibration": _jsonable(calibration),
        "modes": reports,
    }


def _evaluate_run(
    *,
    policy: Any,
    counterfactual_policy: Any,
    bundle_dir: Path,
    episodes_dir: Path,
    rows: list[dict[str, Any]],
    calibration: OpenLoopCalibration,
    target_release_range: tuple[float, float],
    qvel_input: bool,
    reset_policy_on_goal: bool,
) -> dict[str, Any]:
    planner = ScriptCyclePlanner(
        initial_side=str(rows[0]["current_ready_side"]),
        steps=[
            {"target_side": str(row["scripted_target_side"])} for row in rows
        ],
        loop=False,
    )
    source = PolicyActionSource(
        policy=policy,
        source_id=f"planner-open-loop-replay:{bundle_dir.name}:{rows[0]['source_run_id']}",
        camera_name=CAMERAS[0],
        camera_names=list(CAMERAS),
        action_scale=[1.0] * 4,
        clip=1.0,
        output_mode="control",
        qvel_mode="raw",
        cycle_planner=planner,
        reset_policy_on_goal=reset_policy_on_goal,
    )
    guard = ActionGuard(action_clip=1.0, max_delta=1.0, sensor_timeout_s=1.0)
    source.reset()
    counterfactual_policy.reset()
    run_result: dict[str, Any] = {
        "source_run_id": str(rows[0]["source_run_id"]),
        "split": str(rows[0]["split"]),
        "reset_policy_on_goal": bool(reset_policy_on_goal),
        "planned_cycle_count": len(rows),
        "cycles": [],
        "status": "COMPLETED_REFERENCE_REPLAY",
        "failure_reason": None,
    }
    try:
        for row in rows:
            goal = source.commit_cycle_goal()
            cycle = _replay_cycle(
                source=source,
                counterfactual_policy=counterfactual_policy,
                guard=guard,
                episodes_dir=episodes_dir,
                row=row,
                goal=goal,
                        calibration=calibration,
                        target_release_range=target_release_range,
                        qvel_input=qvel_input,
            )
            run_result["cycles"].append(cycle)
            if not cycle["planner_transition_matches_reference"]:
                run_result["status"] = "PLANNER_ERROR"
                run_result["failure_reason"] = "planner_transition_mismatch"
                break
            if not cycle["condition_from_hdf5_matches_planner"]:
                run_result["status"] = "PLANNER_ERROR"
                run_result["failure_reason"] = "planner_condition_mismatch"
                break
            if cycle["status"] != "REFERENCE_CYCLE_COMPLETE":
                run_result["status"] = cycle["status"]
                run_result["failure_reason"] = cycle["failure_reason"]
                break
            try:
                source.mark_cycle_target_ready(str(cycle["target_side"]))
            except Exception as exc:
                run_result["status"] = "PLANNER_ERROR"
                run_result["failure_reason"] = f"{type(exc).__name__}: {exc}"
                break
    finally:
        source.close()
    run_result["completed_cycle_count"] = sum(
        cycle["status"] == "REFERENCE_CYCLE_COMPLETE"
        for cycle in run_result["cycles"]
    )
    return run_result


def _replay_cycle(
    *,
    source: PolicyActionSource,
    counterfactual_policy: Any,
    guard: ActionGuard,
    episodes_dir: Path,
    row: dict[str, Any],
    goal: Any,
    calibration: OpenLoopCalibration,
    target_release_range: tuple[float, float],
    qvel_input: bool,
) -> dict[str, Any]:
    episode_path = episodes_dir / f"episode_{row['episode_id']}.hdf5"
    with h5py.File(episode_path, "r") as handle:
        qpos = np.asarray(handle["observations/qpos"][()], dtype=np.float32)
        qvel = np.asarray(handle["observations/qvel"][()], dtype=np.float32)
        expert = np.asarray(handle["action"][()], dtype=np.float32)
        condition = np.asarray(
            handle["conditions/real_transition_condition_v1"][()], dtype=np.float32
        )
        cycle_phase = (
            np.asarray(
                handle[f"conditions/{CYCLE_PHASE_KEY}"][()], dtype=np.float32
            )
            if f"conditions/{CYCLE_PHASE_KEY}" in handle
            else np.zeros((len(qpos), 1), dtype=np.float32)
        )
        if condition.shape != (len(qpos), 2):
            raise ValueError(f"episode {row['episode_id']} condition shape is invalid")
        if cycle_phase.shape != (len(qpos), 1):
            raise ValueError(f"episode {row['episode_id']} cycle phase shape is invalid")
        excursion_observed = (
            np.asarray(
                handle[f"conditions/{EXCURSION_OBSERVED_KEY}"][()],
                dtype=np.float32,
            )
            if f"conditions/{EXCURSION_OBSERVED_KEY}" in handle
            else derive_excursion_observed(
                qpos=qpos,
                minimum_delta_rad=0.08,
                minimum_consecutive_samples=3,
            )
        )
        if excursion_observed.shape != (len(qpos), 1):
            raise ValueError(
                f"episode {row['episode_id']} excursion state shape is invalid"
            )
        return_commit = (
            np.asarray(
                handle[f"conditions/{RETURN_COMMIT_KEY}"][()], dtype=np.float32
            )
            if f"conditions/{RETURN_COMMIT_KEY}" in handle
            else np.zeros((len(qpos), 1), dtype=np.float32)
        )
        if return_commit.shape != (len(qpos), 1):
            raise ValueError(
                f"episode {row['episode_id']} return commit shape is invalid"
            )
        apex_index = int(np.argmax(qpos[:, 0]))
        target_side = str(goal.target_side)
        ready_index = _reference_ready_index(
            qpos=qpos,
            qvel=qvel,
            apex_index=apex_index,
            target_side=target_side,
            calibration=calibration,
        )
        target_low, target_high = calibration.target_ranges[target_side]
        release_index = _first_target_entry(
            qpos=qpos,
            apex_index=apex_index,
            low=target_low,
            high=target_high,
        )
        images_and_actions = []
        for step in range(len(qpos)):
            if float(excursion_observed[step, 0]) > 0.0:
                source.set_cycle_excursion_observed(observed=True)
            if float(cycle_phase[step, 0]) > 0.0:
                source.set_cycle_phase(return_phase=True)
            if float(return_commit[step, 0]) > 0.0:
                source.set_return_commit(committed=True)
            images = {
                camera: _read_camera_image(handle, camera, step)
                for camera in CAMERAS
            }
            observation = {
                "qpos": qpos[step],
                "qvel": qvel[step],
                "images": images,
                EXCURSION_OBSERVED_KEY: excursion_observed[step],
                CYCLE_PHASE_KEY: cycle_phase[step],
                RETURN_COMMIT_KEY: return_commit[step],
            }
            action, info = source.next_action(observation)
            extras = dict(getattr(info, "extras", {}) or {})
            raw = np.asarray(extras.get("policy_action", action), dtype=np.float32)
            safe, triggered = guard.check(
                action,
                qpos[step],
                deadman_pressed=True,
                estop_active=False,
                manual_override_active=False,
                sensor_age_s=0.0,
            )
            effective_raw = _effective_signs(raw, calibration)
            effective_safe = _effective_signs(safe, calibration)
            images_and_actions.append(
                {
                    "observation": observation,
                    "raw": raw,
                    "safe": np.asarray(safe, dtype=np.float32),
                    "expert": expert[step],
                    "effective_raw": effective_raw,
                    "effective_safe": effective_safe,
                    "guard_triggered": bool(triggered),
                    "guard_reasons": list(guard.last_info.reasons),
                    "policy_error": str(extras.get("policy_error", "")),
                }
            )
        counterfactual = _counterfactual_release_probe(
            policy=counterfactual_policy,
            steps=images_and_actions,
            calibration=calibration,
            target_side=target_side,
            release_index=release_index,
            qvel_input=qvel_input,
        )
        supported_target_release = _supported_target_release_probe(
            policy=counterfactual_policy,
            steps=images_and_actions,
            calibration=calibration,
            apex_index=apex_index,
            decision_range=target_release_range,
            qvel_input=qvel_input,
        )
    actual = _score_cycle_actions(
        steps=images_and_actions,
        qpos=qpos,
        qvel=qvel,
        expert=expert,
        apex_index=apex_index,
        ready_index=ready_index,
        release_index=release_index,
        target_side=target_side,
        calibration=calibration,
    )
    actual.update(
        {
            "cycle_index": int(goal.cycle_index),
            "goal_epoch": int(goal.goal_epoch),
            "transition": str(goal.transition),
            "target_side": target_side,
            "expected_transition": str(row["transition_type"]),
            "planner_transition_matches_reference": bool(
                str(goal.transition) == str(row["transition_type"])
            ),
            "condition_from_hdf5_matches_planner": bool(
                np.all(condition[:, 0] == float(goal.target_side_code))
                and np.all(condition[:, 1] == 1.0)
            ),
            "counterfactual_release_probe": counterfactual,
            "supported_target_release_probe": supported_target_release,
            "reference_episode_id": int(row["episode_id"]),
        }
    )
    return actual


def _score_cycle_actions(
    *,
    steps: list[dict[str, Any]],
    qpos: np.ndarray,
    qvel: np.ndarray,
    expert: np.ndarray,
    apex_index: int,
    ready_index: int | None,
    release_index: int | None,
    target_side: str,
    calibration: OpenLoopCalibration,
) -> dict[str, Any]:
    signs = np.asarray([step["effective_safe"] for step in steps], dtype=np.int8)
    expert_signs = _effective_signs(expert, calibration)
    dig_end = min(apex_index + 1, len(steps))
    return_start = min(apex_index, len(steps))
    release_start = (
        min(max(release_index, return_start), len(steps))
        if release_index is not None
        else len(steps)
    )
    ready_end = (
        min(max(ready_index + calibration.stable_steps, release_start), len(steps))
        if ready_index is not None
        else len(steps)
    )
    dig_slice = slice(0, dig_end)
    approach_slice = slice(return_start, release_start)
    release_slice = slice(release_start, ready_end)
    target_expected = np.zeros(len(steps), dtype=np.int8)
    low, high = calibration.target_ranges[target_side]
    for index in range(return_start, len(steps)):
        if qpos[index, 0] > high:
            target_expected[index] = -1
        elif qpos[index, 0] < low:
            target_expected[index] = 1
    target_expected[:return_start] = 0
    result = {
        "status": (
            "REFERENCE_CYCLE_COMPLETE"
            if ready_index is not None
            and bool(np.max(qpos[:, 0]) >= calibration.working_swing_apex_min)
            else "REFERENCE_CYCLE_INCOMPLETE"
        ),
        "failure_reason": (
            None
            if ready_index is not None
            and bool(np.max(qpos[:, 0]) >= calibration.working_swing_apex_min)
            else "missing_apex_or_ready_boundary"
        ),
        "steps": len(steps),
        "apex_index": int(apex_index),
        "apex_qpos": float(np.max(qpos[:, 0])),
        "ready_index": None if ready_index is None else int(ready_index),
        "release_index": None if release_index is None else int(release_index),
        "final_qpos": qpos[-1].astype(np.float32).tolist(),
        "final_qvel": qvel[-1].astype(np.float32).tolist(),
        "reference_target_ready": ready_index is not None,
        "all_policy_actions_finite": bool(
            all(np.isfinite(step["raw"]).all() for step in steps)
        ),
        "all_safe_actions_finite": bool(
            all(np.isfinite(step["safe"]).all() for step in steps)
        ),
        "policy_error_count": int(sum(bool(step["policy_error"]) for step in steps)),
        "guard_trigger_count": int(sum(step["guard_triggered"] for step in steps)),
        "guard_reasons": sorted(
            {reason for step in steps for reason in step["guard_reasons"]}
        ),
        "raw_action_mae": float(
            np.mean(np.abs(np.asarray([step["raw"] for step in steps]) - expert))
        ),
        "safe_action_mae": float(
            np.mean(np.abs(np.asarray([step["safe"] for step in steps]) - expert))
        ),
        "expert_dig_positive_effective_rate": _rate(
            expert_signs[dig_slice, 0] == 1
        ),
        "policy_dig_positive_effective_rate": _rate(signs[dig_slice, 0] == 1),
        "expert_return_negative_effective_rate": _rate(
            expert_signs[approach_slice, 0] == -1
        ),
        "policy_return_negative_effective_rate": _rate(
            signs[approach_slice, 0] == -1
        ),
        "policy_target_geometry_hit_rate": _rate(
            signs[return_start:, 0] == target_expected[return_start:]
        ),
        "expert_target_geometry_hit_rate": _rate(
            expert_signs[return_start:, 0] == target_expected[return_start:]
        ),
        "expert_release_idle_rate": _rate(expert_signs[release_slice, 0] == 0),
        "policy_release_idle_rate": _rate(signs[release_slice, 0] == 0),
        "policy_release_wrong_effective_rate": _rate(
            signs[release_slice, 0] != 0
        ),
        "expert_action_effective_steps": int(np.sum(expert_signs[:, 0] != 0)),
        "policy_action_effective_steps": int(np.sum(signs[:, 0] != 0)),
        "policy_action_sign_agreement_rate": _rate(
            signs[:, 0] == expert_signs[:, 0]
        ),
        "phase_steps": {
            "dig": int(dig_end),
            "return_approach": int(max(0, release_start - return_start)),
            "release": int(max(0, ready_end - release_start)),
        },
    }
    return result


def _counterfactual_release_probe(
    *,
    policy: Any,
    steps: list[dict[str, Any]],
    calibration: OpenLoopCalibration,
    target_side: str,
    release_index: int | None,
    qvel_input: bool,
) -> dict[str, Any]:
    if release_index is None:
        return {
            "status": "NO_REFERENCE_RELEASE",
            "sample_count": 0,
        }
    target_low, target_high = calibration.target_ranges[target_side]
    indices = [
        index
        for index in range(release_index, len(steps))
        if target_low <= float(steps[index]["observation"]["qpos"][0]) <= target_high
    ]
    # Keep the intervention bounded while covering the complete target-side
    # entry window.  The first/last rows are more informative than a large
    # repeated sample of nearly identical stopped states.
    if len(indices) > 16:
        indices = np.linspace(indices[0], indices[-1], 16, dtype=int).tolist()
    target_signs: list[int] = []
    flipped_signs: list[int] = []
    for index in indices:
        observation = steps[index]["observation"]
        positive_target = target_side
        flipped_target = "A" if positive_target == "B" else "B"
        for side, output in (
            (positive_target, target_signs),
            (flipped_target, flipped_signs),
        ):
            policy.reset()
            probe = dict(observation)
            probe["real_transition_condition_v1"] = np.asarray(
                [1.0 if side == "B" else -1.0, 1.0], dtype=np.float32
            )
            policy_observation = _policy_obs_from_real_obs(
                probe,
                camera_names=CAMERAS,
            )
            action = np.asarray(policy.predict(policy_observation), dtype=np.float32)
            output.append(int(_effective_signs(action.reshape(1, 4), calibration)[0, 0]))
    qpos_values = [
        float(steps[index]["observation"]["qpos"][0]) for index in indices
    ]
    expected_target = [
        _target_geometry_sign(value, target_side, calibration) for value in qpos_values
    ]
    flipped_target = "A" if target_side == "B" else "B"
    expected_flipped = [
        _target_geometry_sign(value, flipped_target, calibration)
        for value in qpos_values
    ]
    return {
        "status": "OK",
        "sample_count": len(indices),
        "indices": [int(index) for index in indices],
        "target_idle_rate": _rate(np.asarray(target_signs) == 0),
        "flipped_expected_effective_rate": _rate(
            np.asarray(expected_flipped) != 0
        ),
        "flipped_effective_hit_rate": _rate(
            np.asarray(flipped_signs) == np.asarray(expected_flipped)
        ),
        "target_expected_sign_hit_rate": _rate(
            np.asarray(target_signs) == np.asarray(expected_target)
        ),
        "target_signs": target_signs,
        "flipped_signs": flipped_signs,
        "expected_target_signs": expected_target,
        "expected_flipped_signs": expected_flipped,
        "qpos_values": qpos_values,
        "qvel_input_used": bool(qvel_input),
    }


def _supported_target_release_probe(
    *,
    policy: Any,
    steps: list[dict[str, Any]],
    calibration: OpenLoopCalibration,
    apex_index: int,
    decision_range: tuple[float, float],
    qvel_input: bool,
) -> dict[str, Any]:
    """Probe the train-supported A-continue/B-stop decision on one state."""

    low, high = decision_range
    indices = [
        index
        for index in range(max(0, apex_index), len(steps))
        if low <= float(steps[index]["observation"]["qpos"][0]) <= high
    ]
    if len(indices) > 16:
        indices = np.linspace(indices[0], indices[-1], 16, dtype=int).tolist()
    a_actions: list[float] = []
    b_actions: list[float] = []
    a_signs: list[int] = []
    b_signs: list[int] = []
    for index in indices:
        observation = steps[index]["observation"]
        for side, actions_out, signs_out in (
            ("A", a_actions, a_signs),
            ("B", b_actions, b_signs),
        ):
            policy.reset()
            probe = dict(observation)
            probe["real_transition_condition_v1"] = np.asarray(
                [-1.0 if side == "A" else 1.0, 1.0], dtype=np.float32
            )
            action = np.asarray(
                policy.predict(
                    _policy_obs_from_real_obs(probe, camera_names=CAMERAS)
                ),
                dtype=np.float32,
            )
            actions_out.append(float(action[0]))
            signs_out.append(
                int(_effective_signs(action.reshape(1, 4), calibration)[0, 0])
            )
    a_sign_array = np.asarray(a_signs, dtype=np.int8)
    b_sign_array = np.asarray(b_signs, dtype=np.int8)
    pair_hits = (a_sign_array == -1) & (b_sign_array == 0)
    return {
        "status": "OK" if indices else "NO_SUPPORTED_DECISION_OBSERVATION",
        "sample_count": len(indices),
        "pair_hit_count": int(np.sum(pair_hits)),
        "pair_hit_rate": _rate(pair_hits),
        "A_continue_negative_rate": _rate(a_sign_array == -1),
        "B_release_idle_rate": _rate(b_sign_array == 0),
        "indices": [int(value) for value in indices],
        "qpos_values": [
            float(steps[index]["observation"]["qpos"][0]) for index in indices
        ],
        "A_swing_actions": a_actions,
        "B_swing_actions": b_actions,
        "A_signs": a_signs,
        "B_signs": b_signs,
        "decision_range": [float(low), float(high)],
        "qvel_input_used": bool(qvel_input),
    }


def _reference_ready_index(
    *,
    qpos: np.ndarray,
    qvel: np.ndarray,
    apex_index: int,
    target_side: str,
    calibration: OpenLoopCalibration,
) -> int | None:
    low, high = calibration.target_ranges[target_side]
    stable = np.abs(qvel[:, 0]) <= calibration.stable_qvel_abs
    last_start = len(qpos) - calibration.stable_steps
    for index in range(max(0, apex_index), max(0, last_start + 1)):
        if low <= float(qpos[index, 0]) <= high and np.all(
            stable[index : index + calibration.stable_steps]
        ):
            return index
    return None


def _first_target_entry(
    *, qpos: np.ndarray, apex_index: int, low: float, high: float
) -> int | None:
    for index in range(max(0, apex_index), len(qpos)):
        if low <= float(qpos[index, 0]) <= high:
            return index
    return None


def _target_geometry_sign(value: float, side: str, calibration: OpenLoopCalibration) -> int:
    low, high = calibration.target_ranges[side]
    if value > high:
        return -1
    if value < low:
        return 1
    return 0


def _effective_signs(action: np.ndarray, calibration: OpenLoopCalibration) -> np.ndarray:
    array = np.asarray(action, dtype=np.float64)
    positive = array >= calibration.deadzone_positive
    negative = array <= -calibration.deadzone_negative
    return np.where(positive, 1, np.where(negative, -1, 0)).astype(np.int8)


def _rate(mask: np.ndarray | list[bool]) -> float:
    values = np.asarray(mask, dtype=bool)
    return 0.0 if values.size == 0 else float(np.mean(values))


def _summarize_mode(
    *,
    bundle_dir: Path,
    mode: str,
    qvel_input: bool,
    calibration: OpenLoopCalibration,
    run_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    cycles = [cycle for run in run_reports for cycle in run["cycles"]]
    transition_summary: dict[str, dict[str, int]] = defaultdict(
        lambda: {"planned": 0, "attempted": 0, "reference_complete": 0}
    )
    for cycle in cycles:
        summary = transition_summary[cycle["transition"]]
        summary["planned"] += 1
        summary["attempted"] += 1
        summary["reference_complete"] += int(
            cycle["status"] == "REFERENCE_CYCLE_COMPLETE"
        )
    supported_probes = [
        cycle["supported_target_release_probe"] for cycle in cycles
    ]
    supported_sample_count = sum(
        int(probe["sample_count"]) for probe in supported_probes
    )
    supported_pair_hit_count = sum(
        int(probe["pair_hit_count"]) for probe in supported_probes
    )
    supported_a_hit_count = sum(
        sum(int(sign) == -1 for sign in probe["A_signs"])
        for probe in supported_probes
    )
    supported_b_hit_count = sum(
        sum(int(sign) == 0 for sign in probe["B_signs"])
        for probe in supported_probes
    )
    return {
        "mode": mode,
        "bundle_dir": str(bundle_dir),
        "qvel_input": bool(qvel_input),
        "run_count": len(run_reports),
        "completed_reference_run_count": sum(
            run["status"] == "COMPLETED_REFERENCE_REPLAY" for run in run_reports
        ),
        "planned_cycle_count": sum(
            int(run["planned_cycle_count"]) for run in run_reports
        ),
        "attempted_cycle_count": len(cycles),
        "completed_reference_cycle_count": sum(
            cycle["status"] == "REFERENCE_CYCLE_COMPLETE" for cycle in cycles
        ),
        "all_policy_actions_finite": all(
            cycle["all_policy_actions_finite"] for cycle in cycles
        ),
        "all_safe_actions_finite": all(
            cycle["all_safe_actions_finite"] for cycle in cycles
        ),
        "transition_summary": dict(sorted(transition_summary.items())),
        "supported_target_release": {
            "sample_count": int(supported_sample_count),
            "pair_hit_count": int(supported_pair_hit_count),
            "pair_hit_rate": (
                0.0
                if supported_sample_count == 0
                else float(supported_pair_hit_count / supported_sample_count)
            ),
            "A_continue_negative_rate": (
                0.0
                if supported_sample_count == 0
                else float(supported_a_hit_count / supported_sample_count)
            ),
            "B_release_idle_rate": (
                0.0
                if supported_sample_count == 0
                else float(supported_b_hit_count / supported_sample_count)
            ),
        },
        "calibration_apex_min": float(calibration.working_swing_apex_min),
        "calibration_cycle_duration_p95_steps": int(
            calibration.cycle_duration_p95_steps
        ),
        "calibration_cycle_timeout_steps": int(calibration.cycle_timeout_steps),
        "run_failure_reasons": _count_run_failures(run_reports),
        "cycle_failure_reasons": _count_cycle_failures(run_reports),
        "runs": run_reports,
    }


def _read_ready_ids(path: Path) -> set[int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {int(str(value).split("_")[-1]) for value in payload["train_ready_episode_ids"]}


def _read_cycle_rows(path: Path, ready_ids: set[int]) -> list[dict[str, Any]]:
    return [
        dict(row)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for row in [json.loads(line)]
        if int(row["episode_id"]) in ready_ids
    ]


def _complete_run_groups(rows: list[dict[str, Any]], split: str) -> list[list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["split"] == split:
            grouped[str(row["source_run_id"])].append(row)
    result = []
    for _run_id, group in sorted(grouped.items()):
        ordered = sorted(group, key=lambda row: int(row["cycle_index"]))
        if [int(row["cycle_index"]) for row in ordered] != list(range(len(ordered))):
            continue
        if any(
            index > 0
            and ordered[index - 1]["scripted_target_side"]
            != ordered[index]["current_ready_side"]
            for index in range(1, len(ordered))
        ):
            continue
        result.append(ordered)
    return result


def _count_run_failures(run_reports: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for run in run_reports:
        if run["status"] != "COMPLETED_REFERENCE_REPLAY":
            counts[str(run["failure_reason"])] += 1
    return dict(sorted(counts.items()))


def _count_cycle_failures(run_reports: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for run in run_reports:
        for cycle in run["cycles"]:
            if cycle["status"] != "REFERENCE_CYCLE_COMPLETE":
                counts[str(cycle["failure_reason"])] += 1
    return dict(sorted(counts.items()))


def _resolve_bundle_deadzone_thresholds(bundle_dir: Path) -> Path:
    resolved = yaml.safe_load(
        (bundle_dir / "resolved_config.yaml").read_text(encoding="utf-8")
    ) or {}
    raw = dict(resolved.get("train", {}) or {}).get("deadzone_loss", {})
    path = Path(str(raw["threshold_json"]))
    if path.is_file():
        return path
    bundled = bundle_dir / path.name
    if bundled.is_file():
        return bundled
    raise FileNotFoundError(f"deadzone threshold file does not exist: {path}")


def _resolve_bundle_target_release_contract(
    *, bundle_dir: Path, resolved: dict[str, Any]
) -> dict[str, Any] | None:
    train_cfg = dict(resolved.get("train", {}) or {})
    policy_cfg = dict(resolved.get("policy", {}) or {})
    raw = dict(
        train_cfg.get(
            "target_release_loss", policy_cfg.get("target_release_loss", {})
        )
        or {}
    )
    if not bool(raw.get("enabled", False)):
        return None
    path = Path(str(raw.get("contract_json", "")))
    if not path.is_file():
        bundled = bundle_dir / path.name
        if bundled.is_file():
            path = bundled
        else:
            raise FileNotFoundError(
                f"target release contract does not exist: {path}"
            )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "real_transition_target_release_contract_v1":
        raise ValueError("bundle target release contract schema is invalid")
    return payload


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "__dataclass_fields__"):
        from dataclasses import asdict

        return _jsonable(asdict(value))
    return value


if __name__ == "__main__":
    main()
