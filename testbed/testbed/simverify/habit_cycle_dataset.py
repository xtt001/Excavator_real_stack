"""Immutable 20 Hz ready-to-ready dataset builder for fixed habit scenarios."""

from __future__ import annotations

import json
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from testbed.simverify.artifacts import (
    artifact_identity,
    verify_checksums,
    write_checksums,
    write_json,
)
from testbed.simverify.contracts import git_provenance, sha256_file
from testbed.simverify.habit_cycle import RELATIVE_INTENTS, SECTORS

DEFAULT_DEFINITION_ROOT = Path(
    "/data/pingfan/Excavator_real_stack_data/"
    "simverify_habit_cycle_definition_v5"
)
DEFAULT_M0_ROOT = Path(
    "/data/pingfan/Excavator_real_stack_data/sim_observable_cycle_v3"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/data/pingfan/Excavator_real_stack_data/"
    "sim_expert_habit_ready_cycle_v1"
)
CAMERAS = ("video4", "video5", "video6", "video7")
NUMERIC_DATASETS = (
    "action",
    "observations/qpos",
    "observations/qvel",
    "timestamps/sim_time_s",
    "timestamps/step_id",
    "diagnostics/selection_error_s",
    "diagnostics/source_action_index",
    "diagnostics/source_observation_index",
    "diagnostics/source_sim_time_s",
    "diagnostics/source_step_id",
    "diagnostics/target_sim_time_s",
    "diagnostics/target_tick",
)


def build_habit_cycle_dataset(
    *,
    repo_root: str | Path,
    definition_root: str | Path = DEFAULT_DEFINITION_ROOT,
    m0_root: str | Path = DEFAULT_M0_ROOT,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    user_approved_freeze: bool = False,
    action_chunk_size: int = 20,
) -> dict[str, Any]:
    """Freeze scenarios and copy complete cycle slices into new HDF5 files."""

    if not user_approved_freeze:
        raise ValueError("scenario freezing requires explicit user approval")
    if int(action_chunk_size) <= 0:
        raise ValueError("action_chunk_size must be positive")
    repository = Path(repo_root).resolve(strict=True)
    repo = git_provenance(repository)
    if repo.get("branch") != "v2.0.0-simVerify" or bool(repo.get("dirty")):
        raise ValueError("dataset build requires clean v2.0.0-simVerify worktree")
    definition = Path(definition_root).resolve(strict=True)
    m0 = Path(m0_root).resolve(strict=True)
    destination = Path(output_root).resolve(strict=False)
    if destination.exists():
        raise FileExistsError(f"immutable dataset output exists: {destination}")
    definition_verification = verify_checksums(
        definition,
        definition / "checksums.sha256",
    )
    if not definition_verification["ok"]:
        raise ValueError("definition artifact checksum verification failed")
    decision = _read_json(definition / "definition_falsification_decision_v1.json")
    if decision.get("decision") != "accept":
        raise ValueError("definition decision must be accept")
    boundaries = _read_json(definition / "habit_cycle_boundaries_v1.json")
    scenarios = _read_json(
        definition / "expert_habit_scenario_candidates_v1.json"
    )
    split_groups = _read_json(m0 / "split_groups.json")
    if split_groups["splits"]["held_out_test"] != [1, 13, 25, 33]:
        raise ValueError("held-out split identity changed")

    records = list(boundaries["records"])
    by_key = {
        (int(row["episode_id"]), int(row["cycle_id"])): row for row in records
    }
    train_keys = sorted(
        {
            (int(scenario["source_episode_id"]), int(cycle_id))
            for scenario in scenarios["candidates"]
            for cycle_id in scenario["source_cycle_ids"]
        }
    )
    validation_keys = sorted(
        (int(row["episode_id"]), int(row["cycle_id"]))
        for row in records
        if row["split"] == "validation"
        and row["relative_intent"] in RELATIVE_INTENTS
        and row["causal_confirm_matches_reference"]
        and "cycle_ready_start_step" in row
    )
    held_out = set(map(int, split_groups["splits"]["held_out_test"]))
    if any(episode_id in held_out for episode_id, _cycle_id in train_keys + validation_keys):
        raise AssertionError("held-out episode entered derived cycle set")

    selected: list[tuple[str, Mapping[str, Any]]] = []
    for split_name, keys in (("train", train_keys), ("validation", validation_keys)):
        for key in keys:
            row = by_key.get(key)
            if row is None:
                raise KeyError(f"scenario references missing boundary: {key}")
            selected.append((split_name, row))

    generated_at = datetime.now(timezone.utc).isoformat()
    temporary = destination.parent / f".{destination.name}.tmp-{os.getpid()}"
    temporary.mkdir(parents=True)
    identities: list[dict[str, Any]] = []
    derived_rows: list[dict[str, Any]] = []
    excluded_rows: list[dict[str, Any]] = []
    try:
        derived_id = 0
        for split_name, row in selected:
            source_episode = int(row["episode_id"])
            source_path = m0 / "episodes" / f"episode_{source_episode}.hdf5"
            output_path = temporary / "episodes" / f"episode_{derived_id}.hdf5"
            result = _write_cycle_slice(
                source_path=source_path,
                output_path=output_path,
                row=row,
                derived_episode_id=derived_id,
                split=split_name,
                action_chunk_size=int(action_chunk_size),
            )
            if result["status"] != "written":
                excluded_rows.append(result)
                continue
            identity = artifact_identity(output_path)
            identities.append(identity)
            derived_rows.append(
                {
                    **result,
                    "path": f"episodes/episode_{derived_id}.hdf5",
                    "size_bytes": identity["size_bytes"],
                    "sha256": identity["sha256"],
                }
            )
            derived_id += 1
        if not derived_rows:
            raise ValueError("no structurally trainable cycle slice was produced")

        train_ids = [
            int(row["derived_episode_id"])
            for row in derived_rows
            if row["split"] == "train"
        ]
        validation_ids = [
            int(row["derived_episode_id"])
            for row in derived_rows
            if row["split"] == "validation"
        ]
        if not train_ids or not validation_ids:
            raise ValueError("derived dataset requires train and validation cycles")

        smoke_candidates = select_semantic_smoke_scenarios(
            scenarios["candidates"]
        )
        frozen_scenarios = write_json(
            temporary / "frozen_scenario_manifest.json",
            {
                "schema": "sim_expert_habit_frozen_scenarios_v1",
                "user_approval_source": (
                    "conversation_instruction_2026-07-27_run_slicing_training_testing"
                ),
                "training_support_pool": "all_ranked_candidates_natural_frequency",
                "candidate_count": int(scenarios["candidate_count"]),
                "candidates": scenarios["candidates"],
                "main_smoke_selection_rule": (
                    "highest_ranked_candidate_per_observed_script_semantic_signature"
                ),
                "main_smoke_scenarios": smoke_candidates,
                "scenario_freeze_authorized": True,
                "planner_model_trained": False,
            },
        )
        identities.append(frozen_scenarios)
        split_identity = write_json(
            temporary / "derived_split.yaml",
            {
                "schema_version": 1,
                "assignment_method": (
                    "source_episode_split_inherited_then_one_file_per_full_cycle"
                ),
                "dataset_dir": str(destination / "episodes"),
                "available_episode_ids": train_ids + validation_ids,
                "train_ids": train_ids,
                "val_ids": validation_ids,
                "held_out_test": "locked_unread",
                "held_out_source_episode_ids": sorted(held_out),
                "source_split_manifest": str(m0 / "split_groups.json"),
                "source_split_manifest_sha256": sha256_file(
                    m0 / "split_groups.json"
                ),
            },
        )
        identities.append(split_identity)
        manifest = write_json(
            temporary / "dataset_manifest.json",
            {
                "schema": "sim_expert_habit_ready_cycle_dataset_v1",
                "status": "complete_training_not_started",
                "evidence_scope": "recorded-observation/offline_training_input",
                "condition_contract": {
                    "key": "cycle_condition_v1",
                    "schema_id": "cycle_condition_v1_dump_end_gated_v1",
                    "pre_dump": "all_zero_inactive",
                    "activation": "first_20hz_row_strictly_after_dump_end",
                    "post_dump": (
                        "current_sector_one_hot_plus_committed_target_one_hot"
                    ),
                    "historical_label_source": "hindsight_observable_next_dig_entry",
                    "recorded_command": "unknown_not_recorded",
                },
                "cycle_contract": {
                    "range": "half_open_ready_start_to_ready_end",
                    "one_hdf5_file_per_cycle": True,
                    "action_chunk_size": int(action_chunk_size),
                    "cross_cycle_action_supervision": False,
                },
                "counts": {
                    "derived_cycle_count": len(derived_rows),
                    "train_cycle_count": len(train_ids),
                    "validation_cycle_count": len(validation_ids),
                    "excluded_short_or_invalid_count": len(excluded_rows),
                    "by_split_intent": _split_intent_counts(derived_rows),
                },
                "cycles": derived_rows,
                "excluded": excluded_rows,
                "sources": {
                    "definition_manifest": artifact_identity(
                        definition / "audit_manifest.json"
                    ),
                    "definition_boundaries": artifact_identity(
                        definition / "habit_cycle_boundaries_v1.json"
                    ),
                    "m0_dataset_manifest": artifact_identity(
                        m0 / "dataset_manifest.json"
                    ),
                },
                "provenance": {
                    "generated_at": generated_at,
                    "git": repo,
                    "held_out_observation_read_count": 0,
                    "privilege_used": False,
                    "training_executed": False,
                },
            },
        )
        identities.append(manifest)
        write_checksums(
            temporary,
            identities,
            path=temporary / "checksums.sha256",
        )
        os.replace(temporary, destination)
        verification = verify_checksums(
            destination,
            destination / "checksums.sha256",
        )
        if not verification["ok"]:
            raise RuntimeError("derived dataset checksum verification failed")
        return {
            "schema": "sim_expert_habit_dataset_build_result_v1",
            "output_root": str(destination),
            "manifest": artifact_identity(destination / "dataset_manifest.json"),
            "checksums": artifact_identity(destination / "checksums.sha256"),
            "counts": {
                "train": len(train_ids),
                "validation": len(validation_ids),
                "excluded": len(excluded_rows),
            },
            "verification": verification,
        }
    finally:
        if temporary.exists():
            for path in sorted(temporary.rglob("*"), reverse=True):
                if path.is_file() or path.is_symlink():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
            temporary.rmdir()


def select_semantic_smoke_scenarios(
    candidates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Select one already-ranked scenario for each observed script signature."""

    selected: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for row in candidates:
        signature = (
            str(row["family"]),
            str(row["current_sector"]),
            str(row["relative_intents"][0]),
        )
        selected.setdefault(signature, row)
    return [
        {
            "signature": list(signature),
            "scenario_id": str(row["scenario_id"]),
            "source_episode_id": int(row["source_episode_id"]),
            "source_cycle_ids": list(map(int, row["source_cycle_ids"])),
            "source_row_range": list(map(int, row["source_row_range"])),
        }
        for signature, row in sorted(selected.items())
    ]


def _write_cycle_slice(
    *,
    source_path: Path,
    output_path: Path,
    row: Mapping[str, Any],
    derived_episode_id: int,
    split: str,
    action_chunk_size: int,
) -> dict[str, Any]:
    source_episode = int(row["episode_id"])
    source_cycle = int(row["cycle_id"])
    start_raw = int(row["cycle_ready_start_step"])
    end_raw = int(row["cycle_ready_end_step"])
    dump_end_raw = int(row["dump_end_step"])
    with h5py.File(source_path, "r") as source:
        source_index = np.asarray(
            source["diagnostics/source_observation_index"],
            dtype=np.int64,
        )
        if np.any(np.diff(source_index) < 0):
            raise ValueError(f"source observation indices are not monotonic: {source_path}")
        start_tick = int(np.searchsorted(source_index, start_raw, side="left"))
        end_tick = int(np.searchsorted(source_index, end_raw, side="left"))
        commit_tick = int(np.searchsorted(source_index, dump_end_raw, side="right"))
        length = end_tick - start_tick
        base = {
            "derived_episode_id": int(derived_episode_id),
            "split": str(split),
            "source_episode_id": source_episode,
            "source_cycle_id": source_cycle,
            "relative_intent": str(row["relative_intent"]),
            "current_sector": str(row["current_sector"]),
            "hindsight_expert_target_sector": str(
                row["hindsight_expert_target_sector"]
            ),
            "raw_source_range": [start_raw, end_raw],
            "source_20hz_range": [start_tick, end_tick],
            "source_dump_end_raw_step": dump_end_raw,
            "source_commit_20hz_tick": commit_tick,
            "cycle_length_20hz": length,
        }
        if length < int(action_chunk_size):
            return {
                **base,
                "status": "excluded",
                "reason": "cycle_shorter_than_action_chunk",
            }
        if not start_tick <= commit_tick < end_tick:
            return {
                **base,
                "status": "excluded",
                "reason": "dump_end_not_inside_cycle",
            }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(output_path, "w") as target:
            target.attrs["is_real"] = False
            target.attrs["sim"] = True
            target.attrs["simverify_habit_cycle"] = True
            for dataset_path in NUMERIC_DATASETS:
                values = source[dataset_path][start_tick:end_tick]
                created = target.create_dataset(
                    dataset_path,
                    data=values,
                    compression="gzip" if np.asarray(values).ndim > 0 else None,
                    compression_opts=1 if np.asarray(values).ndim > 0 else None,
                )
                for key, value in source[dataset_path].attrs.items():
                    created.attrs[key] = value
            for camera in CAMERAS:
                source_dataset = source[f"observations/encoded_images/{camera}"]
                target_dataset = target.create_dataset(
                    f"observations/encoded_images/{camera}",
                    shape=(length,),
                    dtype=h5py.vlen_dtype(np.dtype("uint8")),
                )
                for offset in range(length):
                    target_dataset[offset] = np.asarray(
                        source_dataset[start_tick + offset],
                        dtype=np.uint8,
                    )
                for key, value in source_dataset.attrs.items():
                    target_dataset.attrs[key] = value

            condition = np.zeros((length, 6), dtype=np.float32)
            active = np.zeros(length, dtype=np.uint8)
            relative_commit = commit_tick - start_tick
            current_index = SECTORS.index(str(row["current_sector"]))
            target_index = SECTORS.index(
                str(row["hindsight_expert_target_sector"])
            )
            condition[relative_commit:, current_index] = 1.0
            condition[relative_commit:, 3 + target_index] = 1.0
            active[relative_commit:] = 1
            condition_ds = target.create_dataset(
                "conditions/cycle_condition_v1",
                data=condition,
            )
            condition_ds.attrs["schema_id"] = (
                "cycle_condition_v1_dump_end_gated_v1"
            )
            condition_ds.attrs["pre_dump"] = "all_zero_inactive"
            condition_ds.attrs["post_dump"] = "current_plus_committed_target"
            target.create_dataset(
                "conditions/target_committed_mask",
                data=active,
            )
            target.create_dataset(
                "conditions/valid_mask",
                data=np.ones(length, dtype=np.uint8),
            )
            target.create_dataset(
                "conditions/cycle_id",
                data=np.full(length, source_cycle, dtype=np.int64),
            )
            metadata = target.create_group("metadata")
            metadata.attrs["derived_episode_id"] = int(derived_episode_id)
            metadata.attrs["source_episode_id"] = source_episode
            metadata.attrs["source_cycle_id"] = source_cycle
            metadata.attrs["split"] = str(split)
            metadata.attrs["current_sector"] = str(row["current_sector"])
            metadata.attrs["scripted_target_sector"] = str(
                row["hindsight_expert_target_sector"]
            )
            metadata.attrs["hindsight_expert_target_sector"] = str(
                row["hindsight_expert_target_sector"]
            )
            metadata.attrs["realized_target_sector"] = str(
                row["hindsight_expert_target_sector"]
            )
            metadata.attrs["condition_source"] = (
                "hindsight_observable_next_dig_entry"
            )
            metadata.attrs["recorded_command"] = "unknown_not_recorded"
            metadata.attrs["observable_cycle_completed"] = True
            metadata.attrs["physical_effect_validated"] = False
            metadata.attrs["action_prealigned"] = True
            metadata.attrs["evidence_scope"] = (
                "recorded-observation/offline_training_input"
            )
        return {**base, "status": "written"}


def _split_intent_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split in ("train", "validation"):
        counts = Counter(
            str(row["relative_intent"])
            for row in rows
            if row["split"] == split
        )
        result[split] = dict(sorted(counts.items()))
    return result


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
