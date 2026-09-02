"""Build an immutable hindsight task-state sidecar for real-transition cycles."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from testbed.data.dataset import _read_train_exclude_mask, _valid_start_indices
from testbed.data.task_state_v2 import (
    TASK_STATE_V2_DIM,
    TASK_STATE_V2_SCHEMA,
    TASK_STATE_V2_TIERS,
    task_state_candidate_starts,
)
from testbed.data.work_return_context import WORK_CONTEXT_SCHEMA
from testbed.tasks.real_transition import sha256_file, write_immutable_text
from testbed.tasks.real_transition_return_commit import RETURN_COMMIT_KEY


def main() -> None:
    parser = argparse.ArgumentParser(prog="tb-task-state-v2-manifest")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--work-context-manifest", type=Path, required=True)
    parser.add_argument("--label-audit-dir", type=Path, required=True)
    parser.add_argument("--chunk-steps", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_manifest(
        dataset_root=args.dataset_root,
        work_context_manifest=args.work_context_manifest,
        label_audit_dir=args.label_audit_dir,
        chunk_steps=int(args.chunk_steps),
        output_path=args.output,
    )
    path = write_immutable_text(
        args.output,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(
        json.dumps(
            {
                "output": str(path),
                "episode_count": len(payload["episodes"]),
                "population_counts": payload["population_counts"],
                "commit_before_complete_episode_ids": payload[
                    "boundary_order"
                ]["commit_before_complete_episode_ids"],
            },
            ensure_ascii=False,
        )
    )


def build_manifest(
    *,
    dataset_root: Path | str,
    work_context_manifest: Path | str,
    label_audit_dir: Path | str,
    chunk_steps: int,
    output_path: Path | str,
) -> dict[str, Any]:
    root = Path(dataset_root).resolve()
    work_path = Path(work_context_manifest).resolve()
    audit_dir = Path(label_audit_dir).resolve()
    output_file = Path(output_path).resolve()
    if output_file.exists():
        raise FileExistsError(f"refusing to overwrite frozen manifest: {output_file}")
    if not (root / "episodes").is_dir():
        raise FileNotFoundError(f"dataset root has no episodes directory: {root}")
    window = int(chunk_steps)
    if window <= 0:
        raise ValueError("chunk_steps must be positive")

    work = _json(work_path)
    if work.get("schema") != WORK_CONTEXT_SCHEMA:
        raise ValueError("work-context manifest schema mismatch")
    if Path(str(work.get("dataset_root", ""))).resolve() != root:
        raise ValueError("work-context manifest dataset root mismatch")
    if int(work.get("chunk_steps", -1)) != window:
        raise ValueError("work-context manifest chunk size mismatch")
    work_rows = _episode_map(work.get("episodes", ()), name="work context")

    stage_csv = audit_dir / "cycle_stage_rows.csv"
    audit_result_path = audit_dir / "label_audit_result.json"
    audit_contract_path = audit_dir / "frozen_label_audit_contract_v1.json"
    audit_result = _json(audit_result_path)
    if audit_result.get("status") != "DIAGNOSTIC_COMPLETE":
        raise ValueError("hindsight label audit is not complete")
    stage_rows = {
        int(row["episode_id"]): row
        for row in csv.DictReader(stage_csv.open(encoding="utf-8"))
    }
    if set(stage_rows) != set(work_rows):
        raise ValueError("label-audit and work-context episode populations differ")

    populations: dict[str, Counter[str]] = defaultdict(Counter)
    state_populations: dict[str, Counter[str]] = defaultdict(Counter)
    transitions: dict[str, Counter[str]] = defaultdict(Counter)
    episode_rows: list[dict[str, Any]] = []
    commit_before_complete: list[int] = []
    for episode_id in sorted(work_rows):
        source = work_rows[episode_id]
        stage = stage_rows[episode_id]
        episode_rel = str(source["episode_path"])
        episode_path = root / episode_rel
        actual_sha = sha256_file(episode_path)
        expected_sha = str(source["episode_sha256"])
        if actual_sha != expected_sha:
            raise ValueError(f"episode {episode_id} SHA-256 mismatch")
        work_complete = int(source["work_complete_boundary_row"])
        if int(stage["work_complete_row"]) != work_complete:
            raise ValueError(f"episode {episode_id} work-complete boundary changed")

        with h5py.File(episode_path, "r") as handle:
            total_steps = int(handle["action"].shape[0])
            if int(source["n_rows"]) != total_steps or int(stage["n_rows"]) != total_steps:
                raise ValueError(f"episode {episode_id} row count changed")
            metadata = dict(handle["metadata"].attrs) if "metadata" in handle else {}
            if not _bool_attr(metadata.get("action_prealigned", False)):
                raise ValueError(f"episode {episode_id} action is not prealigned")
            return_commit = np.asarray(
                handle[f"conditions/{RETURN_COMMIT_KEY}"][()], dtype=np.float32
            ).reshape(-1)
            return_commit_row = _return_commit_transition(return_commit)
            if int(stage["return_commit_row"]) != return_commit_row:
                raise ValueError(f"episode {episode_id} return-commit row changed")
            valid_starts = _valid_start_indices(
                total_steps=total_steps,
                train_exclude_mask=_read_train_exclude_mask(handle, total_steps),
                action_chunk_size=window,
            )
        candidates = task_state_candidate_starts(
            total_steps=total_steps,
            work_complete_row=work_complete,
            return_commit_row=return_commit_row,
            action_window_steps=window,
            valid_starts=valid_starts,
        )
        by_name = candidates.by_name()
        empty = [name for name, values in by_name.items() if values.size == 0]
        if empty:
            raise ValueError(
                f"episode {episode_id} has empty task-state tiers: {empty}"
            )

        split = str(source["split"])
        for name, values in by_name.items():
            populations[split][name] += int(values.size)
        first_boundary = min(work_complete, return_commit_row)
        last_boundary = max(work_complete, return_commit_row)
        state_lengths = {
            "dig_incomplete_uncommitted": first_boundary,
            (
                "dig_complete_uncommitted"
                if work_complete < return_commit_row
                else "dig_incomplete_committed"
            ): last_boundary - first_boundary,
            "dig_complete_committed": total_steps - last_boundary,
        }
        for name, count in state_lengths.items():
            state_populations[split][name] += int(count)
        if return_commit_row < work_complete:
            commit_before_complete.append(episode_id)
        transition_type = str(source["transition_type"])
        transitions[split][transition_type] += 1
        episode_rows.append(
            {
                "episode_id": episode_id,
                "episode_path": episode_rel,
                "episode_sha256": actual_sha,
                "split": split,
                "source_block_id": str(source["source_block_id"]),
                "source_run_id": str(source["source_run_id"]),
                "cycle_id": str(source["cycle_id"]),
                "transition_type": transition_type,
                "current_side": str(source["current_anchor"]),
                "dig_target": str(source["dig_target"]),
                "next_target": str(source["next_target"]),
                "n_rows": total_steps,
                "work_complete_row": work_complete,
                "return_commit_row": return_commit_row,
                "return_effective_segment": [
                    int(stage["return_effective_start"]),
                    int(stage["return_effective_end"]),
                ],
                "boundary_order": (
                    "work_complete_then_return_commit"
                    if work_complete < return_commit_row
                    else "return_commit_then_work_complete"
                ),
                "candidate_starts": {
                    name: values.astype(int).tolist()
                    for name, values in by_name.items()
                },
                "candidate_counts": {
                    name: int(values.size) for name, values in by_name.items()
                },
            }
        )

    splits = ("train", "validation", "locked_test")
    for split in splits:
        for tier in TASK_STATE_V2_TIERS:
            if populations[split][tier] <= 0:
                raise ValueError(f"empty {split}:{tier} task-state population")
    expected_overlap = sorted(
        int(value)
        for value in audit_result["label_integrity"][
            "strict_order_violation_episode_ids"
        ]
    )
    if commit_before_complete != expected_overlap:
        raise ValueError(
            "commit-before-complete population differs from the frozen label audit"
        )

    dataset_files = {}
    for name in (
        "SHA256SUMS.txt",
        "cycle_manifest.jsonl",
        "train_ready_manifest.json",
        "split_manifest.json",
        "annotations/cycle_annotations_v2.jsonl",
    ):
        path = root / name
        dataset_files[name] = {"path": str(path), "sha256": sha256_file(path)}
    source_files = {
        "work_context_manifest": {
            "path": str(work_path),
            "sha256": sha256_file(work_path),
        },
        "label_audit_contract": {
            "path": str(audit_contract_path),
            "sha256": sha256_file(audit_contract_path),
        },
        "label_audit_result": {
            "path": str(audit_result_path),
            "sha256": sha256_file(audit_result_path),
        },
        "cycle_stage_rows": {
            "path": str(stage_csv),
            "sha256": sha256_file(stage_csv),
        },
    }
    return {
        "schema": TASK_STATE_V2_SCHEMA,
        "path": str(output_file),
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "frozen_before_model_training": True,
        "maximum_decision": "OFFLINE_DEVELOPMENT_CANDIDATE_INPUT_ONLY",
        "dataset_root": str(root),
        "dataset_files": dataset_files,
        "source_files": source_files,
        "chunk_steps": window,
        "task_state_key": "real_transition_task_state_v2",
        "task_state_dim": TASK_STATE_V2_DIM,
        "task_state_fields": [
            "current_side_code",
            "dig_target_code",
            "dig_complete",
            "return_commit",
            "gated_next_target_code",
        ],
        "tier_names": list(TASK_STATE_V2_TIERS),
        "task_state_contract": {
            "current_side": "cycle_manifest.current_ready_side; cycle-semantic anchor, not instantaneous swing qpos",
            "dig_target": "current_side for every cycle in this source dataset",
            "dig_complete": "0 before work_complete_row, 1 at and after it; hindsight label",
            "return_commit": "recorded v5 hindsight state; 0 before its single rising edge, 1 at and after it",
            "next_target": "cycle_manifest.scripted_target_side, exposed to ACT only when return_commit=1",
            "independent_event_bits": True,
            "commit_before_complete_overlap_preserved": True,
            "chunk_crossing_any_task_state_transition": "masked",
        },
        "sampling_contract": {
            "work_start": "row 0, one sample per episode",
            "work_body": "random full chunk after row 0 and before the first task-state transition",
            "boundary_state": "the first task-state transition row; chunk is masked at the next transition",
            "return_body": "random full chunk at or after the last task-state transition",
            "samples_per_episode": len(TASK_STATE_V2_TIERS),
        },
        "population_counts": {
            split: {
                tier: int(populations[split][tier])
                for tier in TASK_STATE_V2_TIERS
            }
            for split in splits
        },
        "state_row_counts": {
            split: dict(sorted(state_populations[split].items()))
            for split in splits
        },
        "transition_counts": {
            split: dict(sorted(transitions[split].items())) for split in splits
        },
        "boundary_order": {
            "work_complete_then_return_commit_count": len(episode_rows)
            - len(commit_before_complete),
            "commit_before_complete_count": len(commit_before_complete),
            "commit_before_complete_episode_ids": commit_before_complete,
        },
        "episodes": episode_rows,
        "runtime_contract": {
            "owner": "not implemented",
            "reset_on_task_state_transition": "required_before_runtime use",
            "field_ready": False,
        },
        "evidence_boundary": (
            "All task-state events are traced to existing recorded data or "
            "audited hindsight boundaries. This sidecar changes no HDF5 source "
            "and provides no policy-driven, hydraulic, soil-effect, or physical "
            "closed-loop evidence."
        ),
    }


def _return_commit_transition(values: np.ndarray) -> int:
    state = np.asarray(values, dtype=np.float32).reshape(-1)
    if state.size < 2 or not np.all(np.isin(state, [0.0, 1.0])):
        raise ValueError("return-commit state must be a finite binary sequence")
    changes = np.flatnonzero(np.diff(state) != 0.0) + 1
    if (
        changes.size != 1
        or state[0] != 0.0
        or state[-1] != 1.0
        or state[int(changes[0])] != 1.0
    ):
        raise ValueError("return-commit state must contain exactly one 0-to-1 edge")
    return int(changes[0])


def _episode_map(values: Any, *, name: str) -> dict[int, dict[str, Any]]:
    if not isinstance(values, list):
        raise ValueError(f"{name} episodes must be a list")
    result: dict[int, dict[str, Any]] = {}
    for raw in values:
        row = dict(raw)
        episode_id = int(row["episode_id"])
        if episode_id in result:
            raise ValueError(f"duplicate {name} episode {episode_id}")
        result[episode_id] = row
    return result


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _bool_attr(value: Any) -> bool:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


if __name__ == "__main__":
    main()
