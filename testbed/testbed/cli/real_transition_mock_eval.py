"""Run a real-data-calibrated, image-retrieval mock closed-loop evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from testbed.actions.policy import PolicyActionSource, load_act_policy_from_bundle
from testbed.backends.real.backend import RealExcavatorBackend
from testbed.data.mock_closed_loop import (
    DataCalibratedMockStateReader,
    H5ImageBank,
    MockClosedLoopProfile,
)
from testbed.runtime.guard import ActionGuard
from testbed.tasks.act_cycle_planner import ABCyclePlanner

CAMERAS = ("video4", "video5", "video6", "video7")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="tb-real-transition-mock-eval",
        description=(
            "Evaluate conditioned ACT in a real-data-calibrated mock loop. "
            "This is a support-gated diagnostic, not hydraulic proof."
        ),
    )
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=None,
        help="Optional checkpoint override; defaults to <bundle-dir>/policy_best.ckpt.",
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--train-ready-manifest", type=Path, required=True)
    parser.add_argument("--ready-contract", type=Path, required=True)
    parser.add_argument(
        "--deadzone-thresholds",
        type=Path,
        default=None,
        help="Optional direct policy-output deadzone JSON; defaults to the bundle config.",
    )
    parser.add_argument("--cycle-manifest", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--no-temporal-agg",
        action="store_true",
        help="Use direct query-0 inference instead of the runtime aggregation path.",
    )
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument(
        "--episode-ids",
        default="",
        help="Optional comma-separated validation/locked episode ids.",
    )
    args = parser.parse_args()
    try:
        result = evaluate(
            bundle_dir=args.bundle_dir,
            ckpt_path=args.ckpt,
            dataset_dir=args.dataset_dir,
            train_ready_manifest=args.train_ready_manifest,
            ready_contract=args.ready_contract,
            deadzone_thresholds=args.deadzone_thresholds,
            cycle_manifest=args.cycle_manifest,
            device=str(args.device),
            temporal_agg=not bool(args.no_temporal_agg),
            max_steps=int(args.max_steps),
            episode_ids=_parse_episode_ids(args.episode_ids),
        )
        if args.output.exists() and not args.overwrite:
            raise FileExistsError(f"mock evaluation report already exists: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except (OSError, ValueError, RuntimeError) as exc:
        _print_json({"status": "FAIL", "error": str(exc)})
        raise SystemExit(2) from exc
    _print_json({**result, "output": str(args.output)})


def evaluate(
    *,
    bundle_dir: Path,
    ckpt_path: Path | None = None,
    dataset_dir: Path,
    train_ready_manifest: Path,
    ready_contract: Path,
    deadzone_thresholds: Path | None = None,
    cycle_manifest: Path | None,
    device: str,
    temporal_agg: bool,
    max_steps: int,
    episode_ids: list[int] | None,
) -> dict[str, Any]:
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    ready_payload = json.loads(train_ready_manifest.read_text(encoding="utf-8"))
    ready_ids = {int(str(value).split("_")[-1]) for value in ready_payload["train_ready_episode_ids"]}
    cycle_path = cycle_manifest or dataset_dir.parent / "cycle_manifest.jsonl"
    manifest_rows = [
        json.loads(line)
        for line in cycle_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows_by_id = {
        int(row["episode_id"]): row
        for row in manifest_rows
        if int(row["episode_id"]) in ready_ids
    }
    train_ids = sorted(
        episode_id for episode_id, row in rows_by_id.items() if row.get("split") == "train"
    )
    evaluation_ids = sorted(
        episode_id
        for episode_id, row in rows_by_id.items()
        if row.get("split") in {"validation", "locked_test"}
    )
    if episode_ids is not None:
        unknown = sorted(set(episode_ids) - set(evaluation_ids))
        if unknown:
            raise ValueError(
                "episode ids must be validation/locked_test train-ready ids: "
                + ", ".join(str(value) for value in unknown)
            )
        evaluation_ids = sorted(episode_ids)
    if not train_ids or not evaluation_ids:
        raise ValueError("mock evaluation requires train and evaluation episodes")
    profile = MockClosedLoopProfile.from_dataset(
        dataset_dir=dataset_dir,
        episode_ids=train_ids,
        ready_contract_path=ready_contract,
        deadzone_threshold_path=(
            deadzone_thresholds
            if deadzone_thresholds is not None
            else _resolve_bundle_deadzone_thresholds(bundle_dir)
        ),
    )
    support_bank = H5ImageBank(
        [dataset_dir / f"episode_{episode_id}.hdf5" for episode_id in train_ids],
        camera_names=CAMERAS,
        qpos_state_scale=profile.qpos_state_scale,
        qvel_state_scale=profile.qvel_state_scale,
    )
    policy = load_act_policy_from_bundle(
        bundle_dir=bundle_dir,
        ckpt_path=ckpt_path,
        device=device,
        temporal_agg=temporal_agg,
    )
    results = []
    try:
        for episode_id in evaluation_ids:
            results.append(
                _evaluate_episode(
                    policy=policy,
                    profile=profile,
                    dataset_dir=dataset_dir,
                    episode_id=episode_id,
                    manifest_row=rows_by_id[episode_id],
                    support_bank=support_bank,
                    max_steps=max_steps,
                )
            )
    finally:
        close = getattr(policy, "close", None)
        if callable(close):
            close()
    completed = sum(result["status"] == "COMPLETED" for result in results)
    directional_start = sum(
        bool(result.get("first_action_target_direction", False))
        for result in results
    )
    target_range_entered = sum(
        bool(result.get("target_range_entered_after_excursion", False))
        for result in results
    )
    policy_finite = sum(
        bool(result.get("all_raw_policy_actions_finite", False))
        for result in results
    )
    safe_finite = sum(
        bool(result.get("all_safe_actions_finite", False))
        for result in results
    )
    swing_effective_cases = sum(
        bool(result.get("swing_effective_action_seen", False))
        for result in results
    )
    return {
        "schema": "real_transition_mock_closed_loop_eval_v1",
        "status": "PASS" if completed == len(results) else "DIAGNOSTIC_ONLY",
        "plant": "DataCalibratedMockStateReader + H5ImageBank",
        "checkpoint": str(
            ckpt_path if ckpt_path is not None else bundle_dir / "policy_best.ckpt"
        ),
        "temporal_aggregation": bool(temporal_agg),
        "train_episode_count": len(train_ids),
        "image_pool_episode_count": len(evaluation_ids),
        "support_pool_episode_count": len(train_ids),
        "evaluation_episode_count": len(results),
        "case_policy_reset": "source.reset_before_each_case",
        "goal_epoch_policy_reset": "commit_cycle_goal_resets_policy_cache",
        "profile": _jsonable(profile),
        "completed_episode_count": int(completed),
        "first_action_target_direction_count": int(directional_start),
        "target_range_entered_after_excursion_count": int(target_range_entered),
        "all_policy_actions_finite": f"{policy_finite}/{len(results)}",
        "all_safe_actions_finite": f"{safe_finite}/{len(results)}",
        "all_commanded_actions_finite": f"{safe_finite}/{len(results)}",
        "swing_effective_action_cases": f"{swing_effective_cases}/{len(results)}",
        "results": results,
        "boundary": (
            "Held-out episode images are retrieved by predicted qpos while the "
            "hard support gate is calibrated from train-only qpos states. Both "
            "image retrieval and fitted state response are real-data surrogates; "
            "this report is not a hydraulic or field-effect proof."
        ),
    }


def _evaluate_episode(
    *,
    policy: Any,
    profile: MockClosedLoopProfile,
    dataset_dir: Path,
    episode_id: int,
    manifest_row: dict[str, Any],
    support_bank: H5ImageBank,
    max_steps: int,
) -> dict[str, Any]:
    import h5py

    episode_path = dataset_dir / f"episode_{episode_id}.hdf5"
    with h5py.File(episode_path, "r") as handle:
        qpos = np.asarray(handle["observations/qpos"][()], dtype=np.float32)
        qvel = np.asarray(handle["observations/qvel"][()], dtype=np.float32)
        initial_qpos = qpos[0].copy()
        initial_qvel = qvel[0].copy()
    image_bank = H5ImageBank(
        episode_path,
        camera_names=CAMERAS,
        qpos_state_scale=profile.qpos_state_scale,
        qvel_state_scale=profile.qvel_state_scale,
    )
    transition = str(manifest_row.get("transition_type", ""))
    if "->" not in transition:
        raise ValueError(f"episode {episode_id} has invalid transition_type {transition!r}")
    current_side, target_side = transition.split("->", 1)
    target_side_code = 1 if target_side == "B" else -1
    planner = ABCyclePlanner(current_side + target_side, loop=False)
    source = PolicyActionSource(
        policy=policy,
        source_id=f"mock-eval:{episode_id}",
        camera_name=CAMERAS[0],
        camera_names=list(CAMERAS),
        action_scale=[1.0] * 4,
        clip=1.0,
        output_mode="control",
        qvel_mode="raw",
        cycle_planner=planner,
    )
    reader = DataCalibratedMockStateReader(
        profile=profile,
        image_bank=image_bank,
        support_bank=support_bank,
        initial_qpos=initial_qpos,
        initial_qvel=initial_qvel,
    )
    backend = RealExcavatorBackend(
        controller_mode="mock",
        state_reader=reader,
        camera_names=CAMERAS,
        control_hz=1.0 / profile.dt,
    )
    guard = ActionGuard(action_clip=1.0, max_delta=1.0, sensor_timeout_s=1.0)
    try:
        # ``evaluate`` reuses one loaded ACT instance for all held-out cases;
        # reset the source before each case so temporal/chunk state cannot
        # leak from a previous episode.  ``commit_cycle_goal`` also resets at
        # every planner goal boundary within a multi-cycle run.
        source.reset()
        backend.start_episode(seed=0)
        observation = backend.read_state()
        goal = source.commit_cycle_goal()
        anchor = float(observation["qpos"][0])
        stable_count = 0
        excursion_seen = False
        actions: list[np.ndarray] = []
        safe_actions: list[np.ndarray] = []
        raw_policy_actions: list[np.ndarray] = []
        policy_error_count = 0
        max_raw_policy_action_abs = np.zeros(4, dtype=np.float32)
        swing_effective_action_steps = 0
        stop_reason = "max_steps"
        first_action_swing: float | None = None
        first_action_target_direction: bool | None = None
        target_range_entered_after_excursion = False
        max_signed_swing_progress = 0.0
        # The limit is the empirical 99th-percentile cross-episode state
        # distance.  The 95th percentile remains in the profile as a warning
        # reference; using it as a hard stop would reject ordinary held-out
        # pose variation before the target/safety gates can be exercised.
        support_limit = max(1e-6, float(profile.qpos_support_distance_p99))
        support_warning_limit = max(
            1e-6, float(profile.qpos_support_distance_p95)
        )
        support_warning_count = 0
        max_data_support_distance = 0.0
        for step in range(max_steps):
            if reader.last_data_support_distance is not None:
                max_data_support_distance = max(
                    max_data_support_distance,
                    float(reader.last_data_support_distance),
                )
                if reader.last_data_support_distance > support_warning_limit:
                    support_warning_count += 1
            if (
                reader.last_data_support_distance is not None
                and reader.last_data_support_distance > support_limit
            ):
                stop_reason = "data_support_exceeded"
                break
            action, action_info = source.next_action(observation)
            info_extras = dict(getattr(action_info, "extras", {}) or {})
            raw_policy_action = np.asarray(
                info_extras.get("policy_action", action),
                dtype=np.float32,
            ).reshape(-1)
            if raw_policy_action.shape != (4,) or not np.isfinite(raw_policy_action).all():
                stop_reason = "policy_output_invalid"
                break
            raw_policy_actions.append(raw_policy_action.copy())
            max_raw_policy_action_abs = np.maximum(
                max_raw_policy_action_abs,
                np.abs(raw_policy_action),
            )
            swing_threshold = float(
                profile.deadzone_positive[0]
                if raw_policy_action[0] >= 0.0
                else profile.deadzone_negative[0]
            )
            if abs(float(raw_policy_action[0])) >= swing_threshold:
                swing_effective_action_steps += 1
            if str(info_extras.get("policy_error", "")):
                policy_error_count += 1
            safe_action, triggered = guard.check(
                action,
                observation["qpos"],
                deadman_pressed=True,
                estop_active=False,
                manual_override_active=False,
                sensor_age_s=0.0,
            )
            if triggered:
                stop_reason = "guard_triggered"
                break
            if first_action_swing is None:
                first_action_swing = float(raw_policy_action[0])
                first_action_target_direction = bool(
                    abs(first_action_swing) > 1e-6
                    and np.sign(first_action_swing) == target_side_code
                )
            actions.append(np.asarray(action, dtype=np.float32).copy())
            safe_actions.append(np.asarray(safe_action, dtype=np.float32).copy())
            timestep = backend.step(safe_action)
            observation = timestep.observation
            if reader.last_data_support_distance is not None:
                max_data_support_distance = max(
                    max_data_support_distance,
                    float(reader.last_data_support_distance),
                )
            swing = float(observation["qpos"][0])
            if not (
                profile.safe_swing_range[0]
                <= swing
                <= profile.safe_swing_range[1]
            ):
                stop_reason = "safe_swing_range_exceeded"
                break
            if abs(swing - anchor) >= profile.excursion_delta:
                excursion_seen = True
            max_signed_swing_progress = max(
                max_signed_swing_progress,
                float(target_side_code * (swing - anchor)),
            )
            if excursion_seen:
                target_low, target_high = profile.target_ranges[target_side]
                target_range_entered_after_excursion |= bool(
                    target_low <= swing <= target_high
                )
            if excursion_seen and profile.target_ready(
                qpos=observation["qpos"],
                qvel=observation["qvel"],
                target_side=target_side,
            ):
                stable_count += 1
            else:
                stable_count = 0
            if stable_count >= profile.stable_steps:
                source.mark_cycle_target_ready(target_side)
                stop_reason = "target_ready"
                break
        action_array = np.asarray(actions, dtype=np.float32)
        safe_action_array = np.asarray(safe_actions, dtype=np.float32)
        raw_action_array = np.asarray(raw_policy_actions, dtype=np.float32)
        completed = stop_reason == "target_ready"
        if completed:
            status = "COMPLETED"
        elif stop_reason == "data_support_exceeded":
            status = "UNSUPPORTED"
        elif stop_reason in {"safe_swing_range_exceeded", "guard_triggered"}:
            status = "SAFETY_STOP"
        elif stop_reason == "policy_output_invalid":
            status = "POLICY_ERROR"
        else:
            status = "INCOMPLETE"
        return {
            "episode_id": int(episode_id),
            "split": str(manifest_row.get("split", "")),
            "transition": transition,
            "target_side": target_side,
            "target_side_code": int(target_side_code),
            "goal": goal.as_dict(),
            "status": status,
            "stop_reason": stop_reason,
            "steps": len(actions),
            "started_with_finite_nonzero_action": bool(
                raw_action_array.size
                and np.isfinite(raw_action_array[0]).all()
                and np.any(np.abs(raw_action_array[0]) > 1e-6)
            ),
            "first_action_swing": first_action_swing,
            "first_action_target_direction": first_action_target_direction,
            "max_signed_swing_progress": float(max_signed_swing_progress),
            "target_range_entered_after_excursion": bool(
                target_range_entered_after_excursion
            ),
            "all_actions_finite": bool(
                action_array.size == 0 or np.isfinite(action_array).all()
            ),
            "all_safe_actions_finite": bool(
                safe_action_array.size == 0 or np.isfinite(safe_action_array).all()
            ),
            "all_commanded_actions_finite": bool(
                safe_action_array.size == 0 or np.isfinite(safe_action_array).all()
            ),
            "all_raw_policy_actions_finite": bool(
                raw_action_array.size == 0 or np.isfinite(raw_action_array).all()
            ),
            "policy_error_count": int(policy_error_count),
            "max_raw_policy_action_abs": max_raw_policy_action_abs.tolist(),
            "swing_effective_action_steps": int(swing_effective_action_steps),
            "swing_effective_action_seen": bool(swing_effective_action_steps > 0),
            "final_qpos": np.asarray(observation["qpos"], dtype=np.float32).tolist(),
            "final_qvel": np.asarray(observation["qvel"], dtype=np.float32).tolist(),
            "image_retrieval_distance": reader.last_image_distance,
            "data_support_distance": reader.last_data_support_distance,
            "data_support_warning_limit": support_warning_limit,
            "data_support_limit": support_limit,
            "data_support_warning_count": int(support_warning_count),
            "max_data_support_distance": float(max_data_support_distance),
            "guard_trigger_count": int(guard.trigger_count),
        }
    finally:
        source.close()
        backend.close()


def _parse_episode_ids(raw: str) -> list[int] | None:
    text = str(raw or "").strip()
    if not text:
        return None
    values = [int(item.strip().split("_")[-1]) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("--episode-ids must contain at least one id")
    return sorted(set(values))


def _resolve_bundle_deadzone_thresholds(bundle_dir: Path) -> Path:
    """Resolve the frozen direct-output deadzone table from the ACT bundle."""

    import yaml

    resolved_path = Path(bundle_dir) / "resolved_config.yaml"
    if not resolved_path.is_file():
        raise FileNotFoundError(
            f"cannot resolve deadzone thresholds without {resolved_path}"
        )
    resolved = yaml.safe_load(resolved_path.read_text(encoding="utf-8")) or {}
    raw = (
        dict(resolved.get("train", {}) or {}).get("deadzone_loss", {})
        or dict(resolved.get("policy", {}) or {}).get("deadzone_loss", {})
    )
    threshold_raw = raw.get("threshold_json")
    if not threshold_raw:
        raise ValueError(
            "bundle resolved_config.yaml has no direct-output deadzone threshold_json"
        )
    threshold_path = Path(str(threshold_raw))
    if not threshold_path.is_file():
        bundled_path = Path(bundle_dir) / threshold_path.name
        if bundled_path.is_file():
            threshold_path = bundled_path
        else:
            raise FileNotFoundError(
                f"deadzone threshold file does not exist: {threshold_path}"
            )
    return threshold_path


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _print_json(value: Any) -> None:
    json.dump(value, sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
