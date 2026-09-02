"""Recorded-state replay of automatic task-state-v2 progress ownership."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from testbed.actions.policy import (
    _policy_obs_from_real_obs,
    load_act_policy_from_bundle,
)
from testbed.cli.task_state_v2_probe import CAMERAS, PREFIX_TICKS, _observation
from testbed.data.task_state_v2 import task_state_vector
from testbed.tasks.real_transition import sha256_file, write_immutable_text
from testbed.tasks.task_state_auto_progress import TaskStateAutoProgress

SCHEMA = "real_transition_task_state_v2_auto_progress_replay_v1"


class _EventSink:
    def set_task_dig_complete(self, *, completed: bool) -> bool:
        return bool(completed)

    def set_task_return_commit(self, *, committed: bool) -> bool:
        return bool(committed)


def main() -> None:
    parser = argparse.ArgumentParser(prog="tb-task-state-v2-auto-progress-replay")
    parser.add_argument("--probe-manifest", type=Path, required=True)
    parser.add_argument("--work-context-manifest", type=Path, required=True)
    parser.add_argument("--auto-progress-contract", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-name", default="policy_accepted.ckpt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(
        probe_manifest=args.probe_manifest,
        work_context_manifest=args.work_context_manifest,
        auto_progress_contract=args.auto_progress_contract,
        bundle_dir=args.bundle_dir,
        checkpoint_name=str(args.checkpoint_name),
        device=str(args.device),
        output_dir=args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def evaluate(
    *,
    probe_manifest: Path | str,
    work_context_manifest: Path | str,
    auto_progress_contract: Path | str,
    bundle_dir: Path | str,
    checkpoint_name: str,
    device: str,
    output_dir: Path | str,
) -> dict[str, Any]:
    probe_path = Path(probe_manifest).resolve()
    work_path = Path(work_context_manifest).resolve()
    contract_path = Path(auto_progress_contract).resolve()
    bundle = Path(bundle_dir).resolve()
    checkpoint = bundle / checkpoint_name
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite replay output: {output}")
    probe = _json(probe_path)
    work = _json(work_path)
    contract = _json(contract_path)
    ready_contract = _json(bundle / "contracts/ready_contract.json")
    swing_ready = dict(ready_contract["swing_axis"])
    excursion_delta_rad = float(swing_ready["cycle_excursion_min_abs_delta_rad"])
    excursion_required_steps = int(
        swing_ready["cycle_excursion_min_consecutive_samples"]
    )
    root = Path(str(probe["dataset_root"])).resolve()
    work_rows = {int(row["episode_id"]): row for row in work["episodes"]}
    positive = np.asarray(
        contract["runtime_config"]["positive_action_thresholds"],
        dtype=np.float32,
    )
    negative = np.asarray(
        contract["runtime_config"]["negative_action_thresholds"],
        dtype=np.float32,
    )
    policy = load_act_policy_from_bundle(
        bundle_dir=bundle,
        ckpt_path=checkpoint,
        device=device,
        temporal_agg=True,
        device_uint8_preprocess=True,
    )
    rows: list[dict[str, Any]] = []
    try:
        for spec_raw in probe["population"]["cycles"]:
            spec = dict(spec_raw)
            rows.append(
                _evaluate_cycle(
                    policy=policy,
                    root=root,
                    spec=spec,
                    work_row=work_rows[int(spec["episode_id"])],
                    contract=contract,
                    positive=positive,
                    negative=negative,
                    excursion_delta_rad=excursion_delta_rad,
                    excursion_required_steps=excursion_required_steps,
                )
            )
    finally:
        close = getattr(policy, "close", None)
        if callable(close):
            close()
    summary = {
        "heldout_all": _summary(rows),
        "heldout_b_to_a": _summary(
            [row for row in rows if row["transition_type"] == "B->A"]
        ),
        "heldout_other": _summary(
            [row for row in rows if row["transition_type"] != "B->A"]
        ),
    }
    payload = {
        "schema": SCHEMA,
        "status": "RECORDED_STATE_AUTOMATIC_PROGRESS_REPLAY_COMPLETE",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "probe_manifest": {
            "path": str(probe_path),
            "sha256": sha256_file(probe_path),
        },
        "work_context_manifest": {
            "path": str(work_path),
            "sha256": sha256_file(work_path),
        },
        "auto_progress_contract": {
            "path": str(contract_path),
            "sha256": sha256_file(contract_path),
        },
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": sha256_file(checkpoint),
        },
        "runtime_semantics": {
            "progress_owner": "automatic_policy_state",
            "policy_reset_on_each_task_state_change": True,
            "model_action_drives_progress_detector": True,
            "recorded_qpos_qvel_images_drive_future_observations": True,
        },
        "summary": summary,
        "cycles": rows,
        "test_applicability": (
            "STATE_HISTORY_CONDITIONED_RECORDED_OBSERVATION_REPLAY; the automatic "
            "detector is driven by model actions, while future qpos/qvel/images "
            "remain recorded and are not changed by those actions"
        ),
        "evidence_boundary": (
            "This proves that the frozen automatic detector and candidate model "
            "interact on held-out recorded sequences. It is not physical-policy "
            "closed-loop evidence."
        ),
    }
    output.mkdir(parents=True)
    result_path = write_immutable_text(
        output / "auto_progress_replay.json",
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    _write_csv(output / "cycle_metrics.csv", rows)
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
    work_row: dict[str, Any],
    contract: dict[str, Any],
    positive: np.ndarray,
    negative: np.ndarray,
    excursion_delta_rad: float,
    excursion_required_steps: int,
) -> dict[str, Any]:
    episode_id = int(spec["episode_id"])
    episode_path = root / str(spec["episode_path"])
    if sha256_file(episode_path) != str(spec["episode_sha256"]):
        raise ValueError(f"episode {episode_id} SHA-256 mismatch")
    with h5py.File(episode_path, "r") as handle:
        qpos = np.asarray(handle["observations/qpos"][()], dtype=np.float32)
        qvel = np.asarray(handle["observations/qvel"][()], dtype=np.float32)
        condition = np.asarray(
            handle["conditions/real_transition_condition_v1"][()],
            dtype=np.float32,
        )
        bucket_start = int(work_row["bucket_segment"][0])
        recorded_return = int(spec["return_effective_segment"][0])
        replay_start = max(0, bucket_start - PREFIX_TICKS)
        replay_stop = min(len(qpos), recorded_return + 30)
        progress = TaskStateAutoProgress(contract)
        progress.reset_goal(qpos[0])
        excursion_observed = False
        excursion_count = 0
        for row in range(1, replay_start + 1):
            progress.observe_qpos(qpos[row])
            excursion_count = (
                excursion_count + 1
                if float(qpos[row, 0] - qpos[0, 0]) >= excursion_delta_rad
                else 0
            )
            excursion_observed = (
                excursion_observed or excursion_count >= excursion_required_steps
            )

        policy.reset()
        sink = _EventSink()
        work_complete = False
        return_commit = False
        work_complete_row = None
        return_commit_row = None
        actions: list[np.ndarray] = []
        action_rows: list[int] = []
        for row in range(replay_start, replay_stop):
            progress.observe_qpos(qpos[row])
            excursion_count = (
                excursion_count + 1
                if float(qpos[row, 0] - qpos[0, 0]) >= excursion_delta_rad
                else 0
            )
            excursion_observed = (
                excursion_observed or excursion_count >= excursion_required_steps
            )
            event, changed = progress.apply_pending(sink)
            if event == "work_complete" and changed:
                work_complete = True
                work_complete_row = row
                policy.reset()
            elif event == "return_commit" and changed:
                return_commit = True
                return_commit_row = row
                policy.reset()
            task_state = task_state_vector(
                current_side=str(spec["current_side"]),
                dig_target=str(spec["dig_target"]),
                next_target=str(spec["next_target"]),
                dig_complete=work_complete,
                return_commit=return_commit,
            )
            observation = _observation(
                handle=handle,
                timestep=row,
                qpos=qpos[row],
                qvel=qvel[row],
                condition=condition[row],
                task_state=task_state,
            )
            action = np.asarray(
                policy.predict(
                    _policy_obs_from_real_obs(observation, camera_names=CAMERAS)
                ),
                dtype=np.float32,
            )
            actions.append(action)
            action_rows.append(row)
            progress.observe_policy_action(
                action,
                excursion_observed=excursion_observed,
                task_dig_complete=work_complete,
                task_return_commit=return_commit,
            )
    action_array = np.stack(actions)
    row_array = np.asarray(action_rows, dtype=np.int64)
    precommit = (
        np.ones(len(row_array), dtype=bool)
        if return_commit_row is None
        else row_array < return_commit_row
    )
    postcommit = ~precommit
    return {
        "episode_id": episode_id,
        "cycle_id": str(spec["cycle_id"]),
        "split": str(spec["split"]),
        "transition_type": str(spec["transition_type"]),
        "replay_start_row": replay_start,
        "replay_stop_row_exclusive": replay_stop,
        "recorded_work_complete_row": int(spec["work_complete_row"]),
        "recorded_return_commit_row": int(spec["return_commit_row"]),
        "recorded_return_effective_row": recorded_return,
        "automatic_work_complete_row": work_complete_row,
        "automatic_return_commit_row": return_commit_row,
        "automatic_work_complete": work_complete_row is not None,
        "automatic_return_commit": return_commit_row is not None,
        "work_liveness_observed": progress.work_liveness_observed,
        "bucket_effective_observed": bool(
            progress.status()["bucket_effective_observed"]
        ),
        "precommit_effective_negative_swing": bool(
            np.any(action_array[precommit, 0] <= -negative[0])
        ),
        "postcommit_effective_negative_swing": bool(
            np.any(action_array[postcommit, 0] <= -negative[0])
        ),
        "automatic_commit_before_recorded_return": bool(
            return_commit_row is not None and return_commit_row < recorded_return
        ),
        "automatic_work_complete_delta_rows": (
            None
            if work_complete_row is None
            else work_complete_row - int(spec["work_complete_row"])
        ),
        "automatic_return_commit_delta_rows": (
            None
            if return_commit_row is None
            else return_commit_row - int(spec["return_commit_row"])
        ),
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"count": 0}
    return {
        "count": len(rows),
        "automatic_work_complete_rate": _rate(rows, "automatic_work_complete"),
        "automatic_return_commit_rate": _rate(rows, "automatic_return_commit"),
        "work_liveness_rate": _rate(rows, "work_liveness_observed"),
        "bucket_effective_rate": _rate(rows, "bucket_effective_observed"),
        "precommit_effective_negative_swing_rate": _rate(
            rows, "precommit_effective_negative_swing"
        ),
        "postcommit_effective_negative_swing_rate": _rate(
            rows, "postcommit_effective_negative_swing"
        ),
        "automatic_commit_before_recorded_return_rate": _rate(
            rows, "automatic_commit_before_recorded_return"
        ),
        "work_complete_delta_rows": _numeric_summary(
            rows, "automatic_work_complete_delta_rows"
        ),
        "return_commit_delta_rows": _numeric_summary(
            rows, "automatic_return_commit_delta_rows"
        ),
    }


def _rate(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([bool(row[key]) for row in rows]))


def _numeric_summary(rows: list[dict[str, Any]], key: str) -> dict[str, float] | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "max": float(np.max(array)),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


if __name__ == "__main__":
    main()
