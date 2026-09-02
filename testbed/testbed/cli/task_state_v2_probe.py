"""Frozen near-closed-loop open-loop probe for task-state-v2 policies."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from testbed.actions.policy import (
    _policy_obs_from_real_obs,
    load_act_policy_from_bundle,
)
from testbed.data.action_primitive_islands import AXIS_NAMES
from testbed.data.dataset import REAL_TRANSITION_CONDITION_KEY, _read_camera_image
from testbed.data.task_state_v2 import (
    TASK_STATE_V2_KEY,
    TASK_STATE_V2_SCHEMA,
    build_task_state_sequence,
    load_task_state_v2_manifest,
    task_state_manifest_by_episode,
    task_state_vector,
)
from testbed.data.work_return_context import WORK_CONTEXT_SCHEMA
from testbed.policies.deadzone_eval import load_deadzone_thresholds
from testbed.tasks.real_transition import sha256_file, write_immutable_text

CAMERAS = ("video4", "video5", "video6", "video7")
PROBE_SCHEMA_V1 = "real_transition_task_state_v2_probe_manifest_v1"
PROBE_SCHEMA = "real_transition_task_state_v2_probe_manifest_v2"
RESULT_SCHEMA = "real_transition_task_state_v2_probe_result_v1"
HOLD_TICKS = 20
PREFIX_TICKS = HOLD_TICKS - 1


def main() -> None:
    parser = argparse.ArgumentParser(prog="tb-task-state-v2-probe")
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--task-state-manifest", type=Path, required=True)
    freeze.add_argument("--field-manifest", type=Path, required=True)
    freeze.add_argument("--deadzone-thresholds", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--probe-manifest", type=Path, required=True)
    evaluate_parser.add_argument("--bundle-dir", type=Path, required=True)
    evaluate_parser.add_argument("--checkpoint-name", required=True)
    evaluate_parser.add_argument("--model-name", required=True)
    evaluate_parser.add_argument(
        "--interface",
        choices=(
            "legacy_target_condition",
            "state_visual_condition_qvel",
            "task_state_v2",
        ),
        required=True,
    )
    evaluate_parser.add_argument("--device", default="cuda")
    evaluate_parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        result = freeze_probe_manifest(
            task_state_manifest=args.task_state_manifest,
            field_manifest=args.field_manifest,
            deadzone_thresholds=args.deadzone_thresholds,
            output_path=args.output,
        )
    else:
        result = evaluate(
            probe_manifest=args.probe_manifest,
            bundle_dir=args.bundle_dir,
            checkpoint_name=str(args.checkpoint_name),
            model_name=str(args.model_name),
            interface=str(args.interface),
            device=str(args.device),
            output_dir=args.output_dir,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def freeze_probe_manifest(
    *,
    task_state_manifest: Path | str,
    field_manifest: Path | str,
    deadzone_thresholds: Path | str,
    output_path: Path | str,
) -> dict[str, Any]:
    task_path = Path(task_state_manifest).resolve()
    field_path = Path(field_manifest).resolve()
    deadzone_path = Path(deadzone_thresholds).resolve()
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen probe: {output}")
    task_manifest = load_task_state_v2_manifest(task_path)
    work_path = Path(
        str(task_manifest["source_files"]["work_context_manifest"]["path"])
    ).resolve()
    work_manifest = _json(work_path)
    if work_manifest.get("schema") != WORK_CONTEXT_SCHEMA:
        raise ValueError("work-context source schema mismatch")
    if sha256_file(work_path) != str(
        task_manifest["source_files"]["work_context_manifest"]["sha256"]
    ):
        raise ValueError("work-context source changed after task-state freeze")
    field = _json(field_path)
    thresholds = load_deadzone_thresholds(deadzone_path)
    negative_swing = float(thresholds["swing"]["neg"])
    root = Path(str(task_manifest["dataset_root"])).resolve()
    task_rows = task_state_manifest_by_episode(task_manifest)
    work_rows = {
        int(row["episode_id"]): dict(row) for row in work_manifest["episodes"]
    }
    heldout = []
    for episode_id, task_row in sorted(task_rows.items()):
        if str(task_row["split"]) not in {"validation", "locked_test"}:
            continue
        work_row = work_rows[episode_id]
        if str(work_row["episode_sha256"]) != str(task_row["episode_sha256"]):
            raise ValueError(f"episode {episode_id} identity differs across manifests")
        probe_row = {
            **task_row,
            "outbound_segment": list(work_row["outbound_segment"]),
            "bucket_segment": list(work_row["bucket_segment"]),
        }
        if str(task_row["transition_type"]) == "B->A":
            episode_path = root / str(task_row["episode_path"])
            with h5py.File(episode_path, "r") as handle:
                qpos = np.asarray(
                    handle["observations/qpos"][()], dtype=np.float32
                )
                action = np.asarray(handle["action"][()], dtype=np.float32)
            return_start, return_end = (
                int(value) for value in task_row["return_effective_segment"]
            )
            rows = np.arange(return_start, return_end + 1, dtype=np.int64)
            rows = rows[action[rows, 0] <= -negative_swing]
            if rows.size == 0:
                raise ValueError(
                    f"episode {episode_id} has no effective return crossing"
                )
            distances = np.abs(qpos[rows, 0] - qpos[0, 0])
            crossing = int(rows[int(np.argmin(distances))])
            probe_row["return_ready_crossing_row"] = crossing
            probe_row["return_ready_crossing_swing_qpos_distance"] = float(
                abs(float(qpos[crossing, 0] - qpos[0, 0]))
            )
        heldout.append(probe_row)
    if len(heldout) != 30:
        raise ValueError(f"expected 30 source-disjoint heldout cycles, got {len(heldout)}")
    b_to_a = [row for row in heldout if row["transition_type"] == "B->A"]
    if len(b_to_a) != 8:
        raise ValueError(f"expected 8 heldout B-to-A cycles, got {len(b_to_a)}")
    field_probes = [dict(row) for row in field["field_hybrid_probes"]]
    if [row["role"] for row in field_probes].count("normal") != 1 or [
        row["role"] for row in field_probes
    ].count("abnormal") != 2:
        raise ValueError("field population must retain one normal and two abnormal hybrids")
    for row in field_probes:
        episode_id = int(row["source_episode_id"])
        if episode_id not in task_rows:
            raise ValueError(f"field source episode {episode_id} is absent")
        if int(row["source_row"]) >= int(task_rows[episode_id]["n_rows"]):
            raise ValueError(f"field source row is outside episode {episode_id}")
        row["source_episode_sha256_v5"] = str(
            task_rows[episode_id]["episode_sha256"]
        )
    payload = {
        "schema": PROBE_SCHEMA,
        "path": str(output),
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "frozen_before_long_training": True,
        "dataset_root": str(root),
        "task_state_manifest": {
            "path": str(task_path),
            "sha256": sha256_file(task_path),
            "schema": TASK_STATE_V2_SCHEMA,
        },
        "work_context_manifest": {
            "path": str(work_path),
            "sha256": sha256_file(work_path),
        },
        "field_manifest_source": {
            "path": str(field_path),
            "sha256": sha256_file(field_path),
        },
        "deadzone_thresholds": {
            "path": str(deadzone_path),
            "sha256": sha256_file(deadzone_path),
        },
        "hold_ticks": HOLD_TICKS,
        "aggregation_prefix_ticks": PREFIX_TICKS,
        "population": {
            "heldout_cycle_count": len(heldout),
            "heldout_b_to_a_count": len(b_to_a),
            "split_counts": _counts(heldout, "split"),
            "transition_counts": _counts(heldout, "transition_type"),
            "cycles": heldout,
            "field_hybrids": field_probes,
        },
        "window_contract": {
            "work_start": "row 0 through at most 20 recorded rows",
            "outbound": "first 20 rows of the action-derived positive-swing segment",
            "bucket": "first 20 rows of the action-derived positive-bucket segment",
            "boundary_state": "first task-state boundary, ending before the second boundary",
            "return_effective": "first 20 rows of the main mechanically effective negative-swing segment",
            "return_ready_crossing": "B-to-A only: effective return row whose swing qpos is nearest the same cycle's stopped B-ready swing qpos",
            "tail": "last 20 rows",
            "temporal_aggregation": "replay the preceding 19 recorded observations, bounded by the latest policy reset",
            "reset": "legacy baseline resets at cycle goal commit; task_state_v2 resets at each event-bit transition",
        },
        "metrics": [
            "raw_action_chunk",
            "aggregated_policy_action",
            "mechanically_effective_direction",
            "work_start_correct_motion",
            "direct_shortcut",
            "positive_excursion_action_proxy",
            "return_negative_action_proxy",
            "first_effective_swing_direction_and_row",
            "tail_idle",
            "task_pair_adherence",
            "zero_qvel_ready_and_return_behaviour",
            "heldout_action_mae_and_effective_sign_agreement",
        ],
        "synthetic_policy": {
            "field_hybrids_are_non_gating": True,
            "counterfactual_task_and_qvel_pairs_are_separate_from_recorded_cycles": True,
        },
        "evidence_boundary": (
            "Recorded qpos/qvel/images advance independently of predicted actions. "
            "The 19-row prefix reproduces the available ACT aggregation history, "
            "but no policy action changes a future image or robot state. Field rows "
            "remain synthetic hybrids because the actual field images are absent."
        ),
    }
    written = write_immutable_text(
        output,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return {"probe_manifest": str(written), "sha256": sha256_file(written)}


def evaluate(
    *,
    probe_manifest: Path | str,
    bundle_dir: Path | str,
    checkpoint_name: str,
    model_name: str,
    interface: str,
    device: str,
    output_dir: Path | str,
) -> dict[str, Any]:
    probe_path = Path(probe_manifest).resolve()
    probe = _json(probe_path)
    if probe.get("schema") not in {PROBE_SCHEMA_V1, PROBE_SCHEMA}:
        raise ValueError("task-state-v2 probe manifest schema mismatch")
    _verify_file(probe["task_state_manifest"])
    _verify_file(probe["work_context_manifest"])
    _verify_file(probe["deadzone_thresholds"])
    root = Path(str(probe["dataset_root"])).resolve()
    bundle = Path(bundle_dir).resolve()
    checkpoint = bundle / checkpoint_name
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite probe output: {output}")
    for path in (checkpoint, bundle / "dataset_stats.pkl", bundle / "resolved_config.yaml"):
        if not path.is_file():
            raise FileNotFoundError(f"missing policy artifact: {path}")
    thresholds = load_deadzone_thresholds(
        Path(str(probe["deadzone_thresholds"]["path"]))
    )
    positive = np.asarray(
        [float(thresholds[axis]["pos"]) for axis in AXIS_NAMES], dtype=np.float32
    )
    negative = np.asarray(
        [float(thresholds[axis]["neg"]) for axis in AXIS_NAMES], dtype=np.float32
    )
    policy = load_act_policy_from_bundle(
        bundle_dir=bundle,
        ckpt_path=checkpoint,
        device=device,
        temporal_agg=True,
    )
    _validate_interface(policy, interface)
    cycle_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    field_rows: list[dict[str, Any]] = []
    traces: dict[str, np.ndarray] = {}
    try:
        for spec in probe["population"]["cycles"]:
            cycle, windows, cycle_traces = _evaluate_cycle(
                policy=policy,
                root=root,
                spec=dict(spec),
                interface=interface,
                positive=positive,
                negative=negative,
            )
            cycle_rows.append(cycle)
            window_rows.extend(windows)
            traces.update(cycle_traces)
            if str(spec["transition_type"]) == "B->A":
                pair_rows.extend(
                    _evaluate_ready_pairs(
                        policy=policy,
                        root=root,
                        spec=dict(spec),
                        interface=interface,
                        positive=positive,
                        negative=negative,
                    )
                )
        for spec in probe["population"]["field_hybrids"]:
            field_rows.append(
                _evaluate_field_hybrid(
                    policy=policy,
                    root=root,
                    spec=dict(spec),
                    interface=interface,
                    positive=positive,
                    negative=negative,
                )
            )
    finally:
        close = getattr(policy, "close", None)
        if callable(close):
            close()
    summary = _summarise(cycle_rows, pair_rows, field_rows)
    payload = {
        "schema": RESULT_SCHEMA,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "OFFLINE_OPEN_LOOP_REPLAY_COMPLETE",
        "model_name": model_name,
        "interface": interface,
        "probe_manifest": str(probe_path),
        "probe_manifest_sha256": sha256_file(probe_path),
        "bundle_dir": str(bundle),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "resolved_config_sha256": sha256_file(bundle / "resolved_config.yaml"),
        "dataset_stats_sha256": sha256_file(bundle / "dataset_stats.pkl"),
        "low_dim_keys": list(getattr(policy, "low_dim_keys", ()) or ()),
        "summary": summary,
        "cycles": cycle_rows,
        "windows": window_rows,
        "ready_counterfactual_pairs": pair_rows,
        "field_hybrids": field_rows,
        "test_applicability": (
            "RECORDED_STATE_HISTORY_CONDITIONED_OPEN_LOOP_WITH_TEMPORAL_AGGREGATION; "
            "synthetic counterfactual and field rows are reported separately"
        ),
        "evidence_boundary": probe["evidence_boundary"],
    }
    output.mkdir(parents=True)
    result_path = write_immutable_text(
        output / "probe_result.json",
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    _write_csv(output / "cycle_metrics.csv", cycle_rows)
    _write_csv(output / "window_metrics.csv", window_rows)
    _write_csv(output / "ready_counterfactual_metrics.csv", pair_rows)
    _write_csv(output / "field_hybrid_metrics.csv", field_rows)
    np.savez_compressed(output / "action_traces.npz", **traces)
    sums = [
        f"{sha256_file(path)}  {path.name}"
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "SHA256SUMS.txt"
    ]
    write_immutable_text(output / "SHA256SUMS.txt", "\n".join(sums) + "\n")
    return {
        "result": str(result_path),
        "result_sha256": sha256_file(result_path),
        "summary": summary,
    }


def _evaluate_cycle(
    *,
    policy: Any,
    root: Path,
    spec: dict[str, Any],
    interface: str,
    positive: np.ndarray,
    negative: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, np.ndarray]]:
    episode_id = int(spec["episode_id"])
    path = root / str(spec["episode_path"])
    if sha256_file(path) != str(spec["episode_sha256"]):
        raise ValueError(f"episode {episode_id} changed after probe freeze")
    first_boundary = min(
        int(spec["work_complete_row"]), int(spec["return_commit_row"])
    )
    last_boundary = max(
        int(spec["work_complete_row"]), int(spec["return_commit_row"])
    )
    with h5py.File(path, "r") as handle:
        total_steps = int(handle["action"].shape[0])
        anchors = _cycle_anchors(spec, total_steps=total_steps)
        windows: list[dict[str, Any]] = []
        traces: dict[str, np.ndarray] = {}
        for name, anchor, stop in anchors:
            trace = _replay_recorded_window(
                policy=policy,
                handle=handle,
                spec=spec,
                interface=interface,
                anchor=anchor,
                stop=stop,
            )
            metrics = _window_metrics(
                trace=trace, positive=positive, negative=negative
            )
            windows.append(
                {
                    "episode_id": episode_id,
                    "cycle_id": str(spec["cycle_id"]),
                    "split": str(spec["split"]),
                    "transition_type": str(spec["transition_type"]),
                    "window": name,
                    "anchor_row": anchor,
                    "stop_row_exclusive": stop,
                    "task_state_at_anchor": trace["task_state"][0].tolist(),
                    **metrics,
                }
            )
            key = f"episode_{episode_id}_{name}"
            for trace_name in ("aggregated", "raw_chunk", "expert", "rows"):
                traces[f"{key}_{trace_name}"] = trace[trace_name]
    by_name = {row["window"]: row for row in windows}
    work = by_name["work_start"]
    outbound = by_name["outbound"]
    bucket = by_name["bucket"]
    returning = by_name["return_effective"]
    return_crossing = by_name.get("return_ready_crossing")
    tail = by_name["tail"]
    boundary = by_name.get("boundary_state")
    all_mae = [float(row["aggregated_action_mae"]) for row in windows]
    sign_values = [
        float(row["effective_sign_agreement"])
        for row in windows
        if row["effective_sign_agreement"] is not None
    ]
    correct_initial = bool(
        not work["negative_swing_within_window"]
        and work["any_axis_effective_within_window"]
    )
    return (
        {
            "episode_id": episode_id,
            "cycle_id": str(spec["cycle_id"]),
            "split": str(spec["split"]),
            "transition_type": str(spec["transition_type"]),
            "current_side": str(spec["current_side"]),
            "next_target": str(spec["next_target"]),
            "work_complete_row": int(spec["work_complete_row"]),
            "return_commit_row": int(spec["return_commit_row"]),
            "first_task_boundary_row": first_boundary,
            "last_task_boundary_row": last_boundary,
            "work_start_correct_motion": correct_initial,
            "direct_shortcut": bool(work["negative_swing_within_window"]),
            "work_start_tool_liveness": bool(
                work["non_swing_effective_within_window"]
            ),
            "outbound_positive_swing": bool(
                outbound["positive_swing_within_window"]
            ),
            "bucket_tool_liveness": bool(
                bucket["non_swing_effective_within_window"]
            ),
            "bucket_positive": bool(bucket["bucket_positive_within_window"]),
            "return_negative_swing": bool(
                returning["negative_swing_within_window"]
            ),
            "return_ready_crossing_negative_swing": (
                None
                if return_crossing is None
                else bool(return_crossing["negative_swing_within_window"])
            ),
            "boundary_negative_before_second_event": (
                None
                if boundary is None
                else bool(boundary["negative_swing_within_window"])
            ),
            "boundary_all_axes_idle": (
                None if boundary is None else bool(boundary["all_axes_idle"])
            ),
            "tail_all_axes_idle": bool(tail["all_axes_idle"]),
            "ordered_action_proxy": bool(
                correct_initial
                and outbound["positive_swing_within_window"]
                and bucket["non_swing_effective_within_window"]
                and returning["negative_swing_within_window"]
            ),
            "positive_excursion_action_integral_proxy": float(
                outbound["positive_swing_action_integral"]
            ),
            "return_negative_action_integral_proxy": float(
                returning["negative_swing_action_integral"]
            ),
            "heldout_aggregated_action_mae": float(np.mean(all_mae)),
            "heldout_effective_sign_agreement": (
                None if not sign_values else float(np.mean(sign_values))
            ),
        },
        windows,
        traces,
    )


def _cycle_anchors(
    spec: dict[str, Any], *, total_steps: int
) -> list[tuple[str, int, int]]:
    first_boundary = min(
        int(spec["work_complete_row"]), int(spec["return_commit_row"])
    )
    last_boundary = max(
        int(spec["work_complete_row"]), int(spec["return_commit_row"])
    )
    outbound_start = int(spec["outbound_segment"][0])
    outbound_stop = min(int(spec["outbound_segment"][1]) + 1, outbound_start + HOLD_TICKS)
    bucket_start = int(spec["bucket_segment"][0])
    bucket_stop = min(int(spec["bucket_segment"][1]) + 1, bucket_start + HOLD_TICKS)
    return_start = int(spec["return_effective_segment"][0])
    result = [
        ("work_start", 0, min(first_boundary, HOLD_TICKS)),
        ("outbound", outbound_start, outbound_stop),
        ("bucket", bucket_start, bucket_stop),
    ]
    if first_boundary < last_boundary:
        result.append(
            (
                "boundary_state",
                first_boundary,
                min(last_boundary, first_boundary + HOLD_TICKS),
            )
        )
    result.extend(
        (
            ("return_effective", return_start, min(total_steps, return_start + HOLD_TICKS)),
            ("tail", max(last_boundary, total_steps - HOLD_TICKS), total_steps),
        )
    )
    if spec.get("return_ready_crossing_row") is not None:
        crossing = int(spec["return_ready_crossing_row"])
        return_end = int(spec["return_effective_segment"][1]) + 1
        result.append(
            (
                "return_ready_crossing",
                crossing,
                min(return_end, crossing + HOLD_TICKS),
            )
        )
    return result


def _replay_recorded_window(
    *,
    policy: Any,
    handle: h5py.File,
    spec: dict[str, Any],
    interface: str,
    anchor: int,
    stop: int,
) -> dict[str, np.ndarray]:
    total_steps = int(handle["action"].shape[0])
    reset_rows = [0]
    if interface == "task_state_v2":
        reset_rows.extend(
            sorted(
                {
                    int(spec["work_complete_row"]),
                    int(spec["return_commit_row"]),
                }
            )
        )
    latest_reset = max(row for row in reset_rows if row <= anchor)
    prefix_start = max(latest_reset, anchor - PREFIX_TICKS)
    task_sequence = build_task_state_sequence(
        total_steps=total_steps,
        current_side=str(spec["current_side"]),
        dig_target=str(spec["dig_target"]),
        next_target=str(spec["next_target"]),
        work_complete_row=int(spec["work_complete_row"]),
        return_commit_row=int(spec["return_commit_row"]),
    )
    condition = np.asarray(
        handle[f"conditions/{REAL_TRANSITION_CONDITION_KEY}"][()], dtype=np.float32
    )
    qpos = np.asarray(handle["observations/qpos"][()], dtype=np.float32)
    qvel = np.asarray(handle["observations/qvel"][()], dtype=np.float32)
    expert = np.asarray(handle["action"][()], dtype=np.float32)
    policy.reset()
    aggregated = []
    raw_chunks = []
    task_states = []
    rows = []
    for step in range(prefix_start, stop):
        if step != prefix_start and step in reset_rows:
            policy.reset()
        obs = _observation(
            handle=handle,
            timestep=step,
            qpos=qpos[step],
            qvel=qvel[step],
            condition=condition[step],
            task_state=task_sequence[step],
        )
        action = policy.predict(
            _policy_obs_from_real_obs(obs, camera_names=CAMERAS)
        )
        if step >= anchor:
            aggregated.append(np.asarray(action, dtype=np.float32))
            raw_chunks.append(_direct_chunk(policy))
            task_states.append(task_sequence[step])
            rows.append(step)
    row_array = np.asarray(rows, dtype=np.int64)
    return {
        "rows": row_array,
        "aggregated": np.stack(aggregated),
        "raw_chunk": np.stack(raw_chunks),
        "expert": expert[row_array],
        "expert_full": expert,
        "task_state": np.stack(task_states),
    }


def _window_metrics(
    *, trace: dict[str, np.ndarray], positive: np.ndarray, negative: np.ndarray
) -> dict[str, Any]:
    action = trace["aggregated"]
    expert = trace["expert"]
    direction = _direction(action, positive, negative)
    expert_direction = _direction(expert, positive, negative)
    effective = expert_direction != 0
    sign_agreement = (
        None
        if not np.any(effective)
        else float(np.mean(direction[effective] == expert_direction[effective]))
    )
    raw_errors = []
    metric_stop = int(trace["rows"][-1]) + 1
    for row, chunk in zip(trace["rows"], trace["raw_chunk"], strict=True):
        target = trace["expert_full"][
            int(row) : min(int(row) + len(chunk), metric_stop)
        ]
        raw_errors.append(np.abs(chunk[: len(target)] - target))
    first_swing = np.flatnonzero(direction[:, 0] != 0)
    first_swing_row = (
        None if first_swing.size == 0 else int(trace["rows"][int(first_swing[0])])
    )
    first_swing_direction = (
        0 if first_swing.size == 0 else int(direction[int(first_swing[0]), 0])
    )
    return {
        "row_count": int(len(action)),
        "query0_action": action[0].tolist(),
        "query0_raw_chunk": trace["raw_chunk"][0].tolist(),
        "any_axis_effective_within_window": bool(np.any(direction != 0)),
        "non_swing_effective_within_window": bool(np.any(direction[:, 1:] != 0)),
        "positive_swing_within_window": bool(np.any(direction[:, 0] == 1)),
        "negative_swing_within_window": bool(np.any(direction[:, 0] == -1)),
        "bucket_positive_within_window": bool(np.any(direction[:, 3] == 1)),
        "all_axes_idle": bool(np.all(direction == 0)),
        "first_effective_swing_row": first_swing_row,
        "first_effective_swing_direction": first_swing_direction,
        "positive_swing_action_integral": float(np.maximum(action[:, 0], 0.0).sum()),
        "negative_swing_action_integral": float(np.maximum(-action[:, 0], 0.0).sum()),
        "aggregated_action_mae": float(np.mean(np.abs(action - expert))),
        "raw_chunk_action_mae": float(
            np.mean(np.concatenate([value.reshape(-1) for value in raw_errors]))
        ),
        "effective_sign_agreement": sign_agreement,
    }


def _evaluate_ready_pairs(
    *,
    policy: Any,
    root: Path,
    spec: dict[str, Any],
    interface: str,
    positive: np.ndarray,
    negative: np.ndarray,
) -> list[dict[str, Any]]:
    path = root / str(spec["episode_path"])
    with h5py.File(path, "r") as handle:
        qpos = np.asarray(handle["observations/qpos"][0], dtype=np.float32)
        qvel = np.asarray(handle["observations/qvel"][0], dtype=np.float32)
        condition = np.asarray(
            handle[f"conditions/{REAL_TRANSITION_CONDITION_KEY}"][0],
            dtype=np.float32,
        )
        images = {
            camera: _read_camera_image(handle, camera, 0) for camera in CAMERAS
        }
    states = {
        "work": task_state_vector(
            current_side=str(spec["current_side"]),
            dig_target=str(spec["dig_target"]),
            next_target=str(spec["next_target"]),
            dig_complete=0,
            return_commit=0,
        ),
        "return": task_state_vector(
            current_side=str(spec["current_side"]),
            dig_target=str(spec["dig_target"]),
            next_target=str(spec["next_target"]),
            dig_complete=1,
            return_commit=1,
        ),
    }
    outputs: dict[tuple[str, str], dict[str, Any]] = {}
    for state_name, task_state in states.items():
        for qvel_name, qvel_value in (("factual", qvel), ("zero", np.zeros_like(qvel))):
            outputs[(state_name, qvel_name)] = _fixed_hold(
                policy=policy,
                qpos=qpos,
                qvel=qvel_value,
                images=images,
                condition=condition,
                task_state=task_state,
                positive=positive,
                negative=negative,
            )
    rows = []
    for qvel_name in ("factual", "zero"):
        work = outputs[("work", qvel_name)]
        returning = outputs[("return", qvel_name)]
        rows.append(
            {
                "episode_id": int(spec["episode_id"]),
                "cycle_id": str(spec["cycle_id"]),
                "split": str(spec["split"]),
                "qvel_variant": qvel_name,
                "work_no_negative": not work["negative_swing"],
                "work_any_axis_effective": work["any_axis_effective"],
                "work_non_swing_liveness": work["non_swing_effective"],
                "return_negative": returning["negative_swing"],
                "task_pair_hit": bool(
                    not work["negative_swing"]
                    and work["any_axis_effective"]
                    and returning["negative_swing"]
                ),
                "work_query0": work["query0"],
                "return_query0": returning["query0"],
                "work_raw_chunk": work["raw_chunk"],
                "return_raw_chunk": returning["raw_chunk"],
                "task_raw_chunk_l1_delta": float(
                    np.mean(
                        np.abs(
                            np.asarray(work["raw_chunk"], dtype=np.float32)
                            - np.asarray(returning["raw_chunk"], dtype=np.float32)
                        )
                    )
                ),
            }
        )
    return rows


def _evaluate_field_hybrid(
    *,
    policy: Any,
    root: Path,
    spec: dict[str, Any],
    interface: str,
    positive: np.ndarray,
    negative: np.ndarray,
) -> dict[str, Any]:
    path = root / "episodes" / f"episode_{int(spec['source_episode_id'])}.hdf5"
    if sha256_file(path) != str(spec["source_episode_sha256_v5"]):
        raise ValueError(f"field source episode changed: {path}")
    row = int(spec["source_row"])
    with h5py.File(path, "r") as handle:
        qvel = np.asarray(handle["observations/qvel"][row], dtype=np.float32)
        images = {
            camera: _read_camera_image(handle, camera, row) for camera in CAMERAS
        }
    if spec.get("documented_swing_qvel") is not None:
        qvel = qvel.copy()
        qvel[0] = float(spec["documented_swing_qvel"])
    result = _fixed_hold(
        policy=policy,
        qpos=np.asarray(spec["qpos"], dtype=np.float32),
        qvel=qvel,
        images=images,
        condition=np.asarray([-1.0, 1.0], dtype=np.float32),
        task_state=task_state_vector(
            current_side="B",
            dig_target="B",
            next_target="A",
            dig_complete=0,
            return_commit=0,
        ),
        positive=positive,
        negative=negative,
    )
    return {
        "probe_id": str(spec["probe_id"]),
        "role": str(spec["role"]),
        "synthetic": True,
        "query0": result["query0"],
        "negative_swing_shortcut_query0": result["query0_negative_swing"],
        "negative_swing_shortcut_within20": result["negative_swing"],
        "non_swing_effective_within20": result["non_swing_effective"],
        "any_axis_effective_within20": result["any_axis_effective"],
        "all_axes_idle_20": result["all_axes_idle"],
    }


def _fixed_hold(
    *,
    policy: Any,
    qpos: np.ndarray,
    qvel: np.ndarray,
    images: dict[str, np.ndarray],
    condition: np.ndarray,
    task_state: np.ndarray,
    positive: np.ndarray,
    negative: np.ndarray,
) -> dict[str, Any]:
    policy.reset()
    actions = []
    first_chunk = None
    for _ in range(HOLD_TICKS):
        observation = {
            "qpos": qpos,
            "qvel": qvel,
            "images": images,
            REAL_TRANSITION_CONDITION_KEY: condition,
            TASK_STATE_V2_KEY: task_state,
        }
        actions.append(
            np.asarray(
                policy.predict(
                    _policy_obs_from_real_obs(observation, camera_names=CAMERAS)
                ),
                dtype=np.float32,
            )
        )
        if first_chunk is None:
            first_chunk = _direct_chunk(policy)
    action = np.stack(actions)
    direction = _direction(action, positive, negative)
    return {
        "query0": action[0].tolist(),
        "raw_chunk": np.asarray(first_chunk).tolist(),
        "query0_negative_swing": bool(direction[0, 0] == -1),
        "negative_swing": bool(np.any(direction[:, 0] == -1)),
        "positive_swing": bool(np.any(direction[:, 0] == 1)),
        "non_swing_effective": bool(np.any(direction[:, 1:] != 0)),
        "any_axis_effective": bool(np.any(direction != 0)),
        "all_axes_idle": bool(np.all(direction == 0)),
    }


def _observation(
    *,
    handle: h5py.File,
    timestep: int,
    qpos: np.ndarray,
    qvel: np.ndarray,
    condition: np.ndarray,
    task_state: np.ndarray,
) -> dict[str, Any]:
    return {
        "qpos": qpos,
        "qvel": qvel,
        "images": {
            camera: _read_camera_image(handle, camera, int(timestep))
            for camera in CAMERAS
        },
        REAL_TRANSITION_CONDITION_KEY: condition,
        TASK_STATE_V2_KEY: task_state,
    }


def _summarise(
    cycles: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    field: list[dict[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    populations = {
        "heldout_all": cycles,
        "heldout_b_to_a": [row for row in cycles if row["transition_type"] == "B->A"],
        "heldout_other": [row for row in cycles if row["transition_type"] != "B->A"],
    }
    for name, rows in populations.items():
        sign = [
            float(row["heldout_effective_sign_agreement"])
            for row in rows
            if row["heldout_effective_sign_agreement"] is not None
        ]
        result[name] = {
            "count": len(rows),
            "work_start_correct_motion_rate": _rate(rows, "work_start_correct_motion"),
            "direct_shortcut_rate": _rate(rows, "direct_shortcut"),
            "work_start_tool_liveness_rate": _rate(rows, "work_start_tool_liveness"),
            "outbound_positive_swing_rate": _rate(rows, "outbound_positive_swing"),
            "bucket_tool_liveness_rate": _rate(rows, "bucket_tool_liveness"),
            "return_negative_swing_rate": _rate(rows, "return_negative_swing"),
            "return_ready_crossing_negative_swing_rate": _optional_rate(
                rows, "return_ready_crossing_negative_swing"
            ),
            "ordered_action_proxy_rate": _rate(rows, "ordered_action_proxy"),
            "tail_all_axes_idle_rate": _rate(rows, "tail_all_axes_idle"),
            "heldout_aggregated_action_mae": float(
                np.mean([row["heldout_aggregated_action_mae"] for row in rows])
            ),
            "heldout_effective_sign_agreement": (
                None if not sign else float(np.mean(sign))
            ),
        }
    uncommitted_boundary = [
        row
        for row in cycles
        if int(row["work_complete_row"]) < int(row["return_commit_row"])
    ]
    result["uncommitted_boundary_state"] = {
        "count": len(uncommitted_boundary),
        "no_negative_swing_rate": 1.0
        - _rate(uncommitted_boundary, "boundary_negative_before_second_event"),
        "all_axes_idle_rate": _rate(uncommitted_boundary, "boundary_all_axes_idle"),
    }
    for qvel_variant in ("factual", "zero"):
        rows = [row for row in pairs if row["qvel_variant"] == qvel_variant]
        result[f"b_to_a_ready_pair_{qvel_variant}_qvel"] = {
            "count": len(rows),
            "work_no_negative_rate": _rate(rows, "work_no_negative"),
            "work_any_axis_effective_rate": _rate(rows, "work_any_axis_effective"),
            "work_non_swing_liveness_rate": _rate(rows, "work_non_swing_liveness"),
            "return_negative_rate": _rate(rows, "return_negative"),
            "task_pair_hit_rate": _rate(rows, "task_pair_hit"),
            "task_raw_chunk_l1_delta_mean": float(
                np.mean([row["task_raw_chunk_l1_delta"] for row in rows])
            ),
        }
    pair_by_episode = {
        (int(row["episode_id"]), str(row["qvel_variant"])): row for row in pairs
    }
    qvel_deltas = {"work": [], "return": []}
    for episode_id in sorted({int(row["episode_id"]) for row in pairs}):
        factual = pair_by_episode[(episode_id, "factual")]
        zero = pair_by_episode[(episode_id, "zero")]
        for state in qvel_deltas:
            qvel_deltas[state].append(
                float(
                    np.mean(
                        np.abs(
                            np.asarray(
                                factual[f"{state}_raw_chunk"], dtype=np.float32
                            )
                            - np.asarray(
                                zero[f"{state}_raw_chunk"], dtype=np.float32
                            )
                        )
                    )
                )
            )
    result["b_to_a_ready_qvel_sensitivity"] = {
        f"{state}_raw_chunk_l1_delta_mean": float(np.mean(values))
        for state, values in qvel_deltas.items()
    }
    abnormal = [row for row in field if row["role"] == "abnormal"]
    normal = [row for row in field if row["role"] == "normal"]
    result["field_abnormal_synthetic_non_gating"] = {
        "count": len(abnormal),
        "query0_shortcut_rate": _rate(abnormal, "negative_swing_shortcut_query0"),
        "held20_shortcut_rate": _rate(abnormal, "negative_swing_shortcut_within20"),
        "any_axis_effective_rate": _rate(abnormal, "any_axis_effective_within20"),
    }
    result["field_normal_synthetic_non_gating"] = {
        "count": len(normal),
        "held20_shortcut_rate": _rate(normal, "negative_swing_shortcut_within20"),
        "any_axis_effective_rate": _rate(normal, "any_axis_effective_within20"),
    }
    return result


def _validate_interface(policy: Any, interface: str) -> None:
    keys = list(getattr(policy, "low_dim_keys", ()) or ())
    if interface == "legacy_target_condition":
        expected = ["qpos", REAL_TRANSITION_CONDITION_KEY]
    elif interface == "state_visual_condition_qvel":
        expected = ["qpos", REAL_TRANSITION_CONDITION_KEY, "qvel"]
    else:
        expected = ["qpos", "qvel", TASK_STATE_V2_KEY]
    if keys != expected:
        raise ValueError(
            f"model interface mismatch: expected low_dim_keys={expected}, got {keys}"
        )


def _direct_chunk(policy: Any) -> np.ndarray:
    normalised = np.asarray(policy.last_raw_action_chunk(), dtype=np.float32)
    mean = np.asarray(policy.norm_stats["action_mean"], dtype=np.float32)
    std = np.asarray(policy.norm_stats["action_std"], dtype=np.float32)
    return normalised * std + mean


def _direction(
    action: np.ndarray, positive: np.ndarray, negative: np.ndarray
) -> np.ndarray:
    return np.where(
        action >= positive.reshape(1, -1),
        1,
        np.where(action <= -negative.reshape(1, -1), -1, 0),
    ).astype(np.int8)


def _rate(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([bool(row[key]) for row in rows])) if rows else float("nan")


def _optional_rate(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [bool(row[key]) for row in rows if row.get(key) is not None]
    return None if not values else float(np.mean(values))


def _counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    values: dict[str, int] = defaultdict(int)
    for row in rows:
        values[str(row[key])] += 1
    return dict(sorted(values.items()))


def _verify_file(spec: dict[str, Any]) -> None:
    path = Path(str(spec["path"]))
    if sha256_file(path) != str(spec["sha256"]):
        raise ValueError(f"frozen probe input changed: {path}")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(row.get(key), ensure_ascii=False)
                    if isinstance(row.get(key), (list, dict))
                    else row.get(key)
                    for key in fields
                }
            )


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


if __name__ == "__main__":
    main()
