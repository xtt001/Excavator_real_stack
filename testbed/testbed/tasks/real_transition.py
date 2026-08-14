"""Contracts and immutable packaging for v2.0.1 real-transition data.

This module deliberately owns no actuator control.  It provides the scripted
P0/P1 plan, the append-only task-event state machine, and the final seal that
binds every event to an exact HDF5 ``step_id``/``step_ns`` row.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import random
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

DATA_CONTRACT_VERSION = "real_transition_raw_v1"
CONDITION_SCHEMA = "real_transition_condition_v1"
SEQUENCE_MANIFEST_SCHEMA = "real_transition_sequence_manifest_v1"
SPLIT_MANIFEST_SCHEMA = "real_transition_split_manifest_v1"
TASK_EVENT_SCHEMA = "real_transition_task_event_v1"
RUN_MANIFEST_SCHEMA = "real_transition_run_manifest_v1"
SESSION_PREPARATION_SCHEMA = "real_transition_session_preparation_v1"

SIDE_CODES = {"A": -1, "B": 1}
SEQUENCE_TEMPLATES: dict[str, tuple[str, ...]] = {
    "P0": ("A", "B", "B", "A", "A"),
    "P1": ("B", "A", "A", "B", "B"),
}
REQUIRED_GOAL_ACK_SOURCES = ("recorder", "router", "display")
TERMINAL_EVENT_TYPES = {"run_complete", "run_abort", "safety_stop"}
PASSIVE_EVENT_TYPES = {"manual_intervention"}
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class TransitionContractError(RuntimeError):
    """Raised when a plan, event stream, or run package violates the contract."""


@dataclass(frozen=True)
class TransitionRunSpec:
    session_id: str
    block_id: str
    run_id: str
    split: str
    template_id: str
    collection_rank: int
    run_rank_in_block: int
    sequence: tuple[str, ...]

    @property
    def initial_side(self) -> str:
        return self.sequence[0]

    @property
    def targets(self) -> tuple[str, ...]:
        return self.sequence[1:]

    @property
    def transitions(self) -> tuple[str, ...]:
        return tuple(
            f"{current}->{target}"
            for current, target in zip(self.sequence[:-1], self.sequence[1:])
        )

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        session_id: str,
        block_id: str,
        split: str,
        collection_rank: int,
    ) -> TransitionRunSpec:
        sequence = tuple(str(side) for side in value.get("sequence", ()))
        spec = cls(
            session_id=_safe_id(session_id, "session_id"),
            block_id=_safe_id(block_id, "block_id"),
            run_id=_safe_id(value.get("run_id", ""), "run_id"),
            split=str(split),
            template_id=str(value.get("template_id", "")),
            collection_rank=int(collection_rank),
            run_rank_in_block=int(value.get("run_rank_in_block", -1)),
            sequence=sequence,
        )
        spec.validate()
        return spec

    def validate(self) -> None:
        if self.template_id not in SEQUENCE_TEMPLATES:
            raise TransitionContractError(
                f"unsupported template_id {self.template_id!r} for {self.run_id}"
            )
        expected = SEQUENCE_TEMPLATES[self.template_id]
        if self.sequence != expected:
            raise TransitionContractError(
                f"run {self.run_id} sequence {self.sequence!r} does not match "
                f"{self.template_id}={expected!r}"
            )
        if self.split not in {"train", "validation", "locked_test"}:
            raise TransitionContractError(
                f"run {self.run_id} has unsupported split {self.split!r}"
            )
        if self.run_rank_in_block < 0:
            raise TransitionContractError(
                f"run {self.run_id} has invalid run_rank_in_block"
            )
        if Counter(self.transitions) != Counter(
            {"A->A": 1, "A->B": 1, "B->A": 1, "B->B": 1}
        ):
            raise TransitionContractError(
                f"run {self.run_id} does not contain every atomic transition once"
            )


def build_session_manifests(
    *,
    session_id: str,
    seed: int,
    created_at_utc: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a deterministic six-block, twenty-four-run recording plan."""

    session_id = _safe_id(session_id, "session_id")
    seed = int(seed)
    created_at_utc = str(created_at_utc or _utc_now())
    rng = random.Random(seed)

    core_splits = ["train", "train", "validation", "locked_test"]
    rng.shuffle(core_splits)
    block_specs = [
        {"split": split, "priority_tier": "minimum_64_cycle"} for split in core_splits
    ]
    expansion = [
        {"split": "train", "priority_tier": "train_expansion_96_cycle"},
        {"split": "train", "priority_tier": "train_expansion_96_cycle"},
    ]
    rng.shuffle(expansion)
    block_specs.extend(expansion)

    blocks: list[dict[str, Any]] = []
    for collection_rank, block_plan in enumerate(block_specs):
        block_id = f"b{collection_rank + 1:02d}"
        templates = ["P0", "P0", "P1", "P1"]
        rng.shuffle(templates)
        occurrence: Counter[str] = Counter()
        runs: list[dict[str, Any]] = []
        for run_rank, template_id in enumerate(templates):
            occurrence[template_id] += 1
            run_id = f"{block_id}_r{run_rank + 1:02d}"
            sequence = SEQUENCE_TEMPLATES[template_id]
            cycles = []
            for cycle_index, (current_side, target_side) in enumerate(
                zip(sequence[:-1], sequence[1:])
            ):
                cycles.append(
                    {
                        "cycle_id": f"{run_id}_c{cycle_index + 1:02d}",
                        "cycle_index": cycle_index,
                        "current_side": current_side,
                        "scripted_target_side": target_side,
                        "target_side_code": SIDE_CODES[target_side],
                        "transition": f"{current_side}->{target_side}",
                    }
                )
            runs.append(
                {
                    "run_id": run_id,
                    "run_rank_in_block": run_rank,
                    "template_id": template_id,
                    "template_occurrence": occurrence[template_id],
                    "sequence": list(sequence),
                    "initial_side": sequence[0],
                    "scripted_targets": list(sequence[1:]),
                    "cycles": cycles,
                    "replacement_of": None,
                }
            )
        blocks.append(
            {
                "block_id": block_id,
                "collection_rank": collection_rank,
                "priority_tier": block_plan["priority_tier"],
                "split": block_plan["split"],
                "runs": runs,
            }
        )

    sequence_manifest: dict[str, Any] = {
        "schema": SEQUENCE_MANIFEST_SCHEMA,
        "data_contract_version": DATA_CONTRACT_VERSION,
        "condition_schema": CONDITION_SCHEMA,
        "session_id": session_id,
        "seed": seed,
        "created_at_utc": created_at_utc,
        "immutable_after_recording_starts": True,
        "side_codes": dict(SIDE_CODES),
        "templates": {
            template_id: list(sequence)
            for template_id, sequence in SEQUENCE_TEMPLATES.items()
        },
        "blocks": blocks,
    }
    validate_sequence_manifest(sequence_manifest)

    block_assignments = [
        {
            "block_id": block["block_id"],
            "split": block["split"],
            "priority_tier": block["priority_tier"],
            "collection_rank": block["collection_rank"],
            "run_ids": [run["run_id"] for run in block["runs"]],
        }
        for block in blocks
    ]
    split_manifest: dict[str, Any] = {
        "schema": SPLIT_MANIFEST_SCHEMA,
        "data_contract_version": DATA_CONTRACT_VERSION,
        "session_id": session_id,
        "created_at_utc": created_at_utc,
        "sequence_manifest_sha256": _sha256_bytes(
            _canonical_json_bytes(sequence_manifest)
        ),
        "rules": {
            "unit": "whole_source_block",
            "cycles_from_one_run_stay_in_one_split": True,
            "locked_test_task_results_hidden_until_authorized": True,
            "post_collection_reassignment_allowed": False,
        },
        "block_assignments": block_assignments,
        "expected_counts": {
            "blocks": {"train": 4, "validation": 1, "locked_test": 1},
            "runs": {"train": 16, "validation": 4, "locked_test": 4},
            "cycles": {"train": 64, "validation": 16, "locked_test": 16},
            "cycles_per_transition": {
                "train": 16,
                "validation": 4,
                "locked_test": 4,
            },
        },
    }
    validate_split_manifest(split_manifest, sequence_manifest=sequence_manifest)
    return sequence_manifest, split_manifest


def prepare_session_directory(
    *,
    output_root: Path | str,
    session_id: str,
    seed: int,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    """Write immutable pre-recording sequence and split artifacts.

    Existing byte-identical files make this operation idempotent.  A different
    payload under the same session id is rejected instead of overwritten.
    """

    safe_session_id = _safe_id(session_id, "session_id")
    session_dir = Path(output_root) / f"session_{safe_session_id}"
    sequence_path = session_dir / "sequence_manifest.json"
    if created_at_utc is None and sequence_path.is_file():
        existing = load_sequence_manifest(sequence_path)
        if existing.get("session_id") != safe_session_id or int(
            existing.get("seed", -1)
        ) != int(seed):
            raise TransitionContractError(
                "existing sequence manifest session_id/seed does not match this request"
            )
        created_at_utc = str(existing["created_at_utc"])
    sequence_manifest, split_manifest = build_session_manifests(
        session_id=session_id,
        seed=seed,
        created_at_utc=created_at_utc,
    )
    session_dir.mkdir(parents=True, exist_ok=True)
    split_path = session_dir / "split_manifest.json"
    _write_json_immutable(sequence_path, sequence_manifest)
    _write_json_immutable(split_path, split_manifest)

    preparation = {
        "schema": SESSION_PREPARATION_SCHEMA,
        "data_contract_version": DATA_CONTRACT_VERSION,
        "session_id": sequence_manifest["session_id"],
        "created_at_utc": sequence_manifest["created_at_utc"],
        "seed": int(seed),
        "status": "sequence_and_split_frozen",
        "artifacts": {
            "sequence_manifest.json": sha256_file(sequence_path),
            "split_manifest.json": sha256_file(split_path),
        },
        "remaining_field_owned_artifacts": [
            "home_side_contract.json",
            "resolved_record_config.yaml",
            "session_manifest.json",
        ],
    }
    preparation_path = session_dir / "preparation_manifest.json"
    _write_json_immutable(preparation_path, preparation)
    return {
        "session_dir": str(session_dir),
        "sequence_manifest": str(sequence_path),
        "split_manifest": str(split_path),
        "preparation_manifest": str(preparation_path),
        "counts": summarize_sequence_manifest(sequence_manifest),
    }


def load_sequence_manifest(path: Path | str) -> dict[str, Any]:
    manifest = _read_json_object(Path(path))
    validate_sequence_manifest(manifest)
    return manifest


def load_split_manifest(
    path: Path | str,
    *,
    sequence_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    manifest = _read_json_object(Path(path))
    validate_split_manifest(manifest, sequence_manifest=sequence_manifest)
    return manifest


def validate_sequence_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema") != SEQUENCE_MANIFEST_SCHEMA:
        raise TransitionContractError(
            f"sequence schema must be {SEQUENCE_MANIFEST_SCHEMA!r}"
        )
    session_id = _safe_id(manifest.get("session_id", ""), "session_id")
    blocks = manifest.get("blocks", ())
    if not isinstance(blocks, Sequence) or isinstance(blocks, (str, bytes)):
        raise TransitionContractError("sequence blocks must be a list")
    if len(blocks) != 6:
        raise TransitionContractError(f"expected 6 blocks, found {len(blocks)}")

    seen_blocks: set[str] = set()
    seen_runs: set[str] = set()
    collection_ranks: set[int] = set()
    transition_counts: Counter[str] = Counter()
    split_blocks: Counter[str] = Counter()
    for block in blocks:
        if not isinstance(block, Mapping):
            raise TransitionContractError("each sequence block must be an object")
        block_id = _safe_id(block.get("block_id", ""), "block_id")
        if block_id in seen_blocks:
            raise TransitionContractError(f"duplicate block_id {block_id}")
        seen_blocks.add(block_id)
        collection_rank = int(block.get("collection_rank", -1))
        collection_ranks.add(collection_rank)
        split = str(block.get("split", ""))
        split_blocks[split] += 1
        runs = block.get("runs", ())
        if not isinstance(runs, Sequence) or isinstance(runs, (str, bytes)):
            raise TransitionContractError(f"block {block_id} runs must be a list")
        if len(runs) != 4:
            raise TransitionContractError(
                f"block {block_id} must contain 4 runs, found {len(runs)}"
            )
        templates: Counter[str] = Counter()
        block_transitions: Counter[str] = Counter()
        for run in runs:
            if not isinstance(run, Mapping):
                raise TransitionContractError(f"block {block_id} run must be an object")
            spec = TransitionRunSpec.from_mapping(
                run,
                session_id=session_id,
                block_id=block_id,
                split=split,
                collection_rank=collection_rank,
            )
            if spec.run_id in seen_runs:
                raise TransitionContractError(f"duplicate run_id {spec.run_id}")
            seen_runs.add(spec.run_id)
            templates[spec.template_id] += 1
            block_transitions.update(spec.transitions)
            transition_counts.update(spec.transitions)
        if templates != Counter({"P0": 2, "P1": 2}):
            raise TransitionContractError(
                f"block {block_id} must contain two P0 and two P1 runs"
            )
        if block_transitions != Counter({"A->A": 4, "A->B": 4, "B->A": 4, "B->B": 4}):
            raise TransitionContractError(
                f"block {block_id} transition coverage is not balanced"
            )

    if collection_ranks != set(range(6)):
        raise TransitionContractError(
            f"collection ranks must be 0..5, found {sorted(collection_ranks)}"
        )
    if split_blocks != Counter({"train": 4, "validation": 1, "locked_test": 1}):
        raise TransitionContractError(
            f"unexpected block split counts: {dict(split_blocks)}"
        )
    if transition_counts != Counter({"A->A": 24, "A->B": 24, "B->A": 24, "B->B": 24}):
        raise TransitionContractError(
            f"unexpected global transition coverage: {dict(transition_counts)}"
        )


def validate_split_manifest(
    manifest: Mapping[str, Any],
    *,
    sequence_manifest: Mapping[str, Any],
) -> None:
    if manifest.get("schema") != SPLIT_MANIFEST_SCHEMA:
        raise TransitionContractError(f"split schema must be {SPLIT_MANIFEST_SCHEMA!r}")
    validate_sequence_manifest(sequence_manifest)
    if manifest.get("session_id") != sequence_manifest.get("session_id"):
        raise TransitionContractError("split and sequence session_id differ")
    expected_sequence_sha = _sha256_bytes(
        _canonical_json_bytes(dict(sequence_manifest))
    )
    if manifest.get("sequence_manifest_sha256") != expected_sequence_sha:
        raise TransitionContractError("split manifest sequence checksum mismatch")

    expected = {
        str(block["block_id"]): str(block["split"])
        for block in sequence_manifest["blocks"]
    }
    assignments = manifest.get("block_assignments", ())
    actual: dict[str, str] = {}
    for assignment in assignments:
        block_id = str(assignment.get("block_id", ""))
        if block_id in actual:
            raise TransitionContractError(f"duplicate split assignment for {block_id}")
        actual[block_id] = str(assignment.get("split", ""))
    if actual != expected:
        raise TransitionContractError(
            "split assignments do not match sequence manifest"
        )


def summarize_sequence_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    validate_sequence_manifest(manifest)
    split_blocks: Counter[str] = Counter()
    split_runs: Counter[str] = Counter()
    split_cycles: Counter[str] = Counter()
    transitions: Counter[str] = Counter()
    for block in manifest["blocks"]:
        split = str(block["split"])
        split_blocks[split] += 1
        for run in block["runs"]:
            split_runs[split] += 1
            sequence = tuple(run["sequence"])
            run_transitions = [
                f"{current}->{target}"
                for current, target in zip(sequence[:-1], sequence[1:])
            ]
            split_cycles[split] += len(run_transitions)
            transitions.update(run_transitions)
    return {
        "blocks": dict(sorted(split_blocks.items())),
        "runs": dict(sorted(split_runs.items())),
        "cycles": dict(sorted(split_cycles.items())),
        "transitions": dict(sorted(transitions.items())),
    }


def iter_run_specs(manifest: Mapping[str, Any]) -> Iterable[TransitionRunSpec]:
    validate_sequence_manifest(manifest)
    session_id = str(manifest["session_id"])
    for block in sorted(manifest["blocks"], key=lambda item: item["collection_rank"]):
        for run in sorted(block["runs"], key=lambda item: item["run_rank_in_block"]):
            yield TransitionRunSpec.from_mapping(
                run,
                session_id=session_id,
                block_id=str(block["block_id"]),
                split=str(block["split"]),
                collection_rank=int(block["collection_rank"]),
            )


def find_run_spec(manifest: Mapping[str, Any], run_id: str) -> TransitionRunSpec:
    run_id = _safe_id(run_id, "run_id")
    for spec in iter_run_specs(manifest):
        if spec.run_id == run_id:
            return spec
    raise TransitionContractError(f"run_id {run_id!r} is absent from sequence manifest")


class TransitionRunPackage:
    """Append-only event recorder and immutable run-package sealer."""

    def __init__(
        self,
        *,
        run_dir: Path | str,
        run_spec: TransitionRunSpec,
        create: bool = True,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.run_spec = run_spec
        self.events_path = self.run_dir / "task_events.jsonl"
        self.manifest_path = self.run_dir / "run_manifest.json"
        self.checksums_path = self.run_dir / "SHA256SUMS.txt"
        self.raw_path = self.run_dir / "raw.hdf5"
        self._events: list[dict[str, Any]] = []
        self._phase = "new"
        self._cycle_index = 0
        self._goal_epoch = 0
        self._sealed = False
        if create:
            self._create_empty_package()

    @property
    def cycle_index(self) -> int:
        return self._cycle_index

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def next_target_side(self) -> str | None:
        if self._cycle_index >= len(self.run_spec.targets):
            return None
        return self.run_spec.targets[self._cycle_index]

    def start_run(
        self, *, step_id: int, step_ns: int, notes: str = ""
    ) -> dict[str, Any]:
        self._require_phase("new")
        event = self._append_event(
            event_type="run_start",
            step_id=step_id,
            step_ns=step_ns,
            event_source="system",
            notes=notes,
        )
        self._phase = "started"
        return event

    def mark_initial_ready(
        self,
        *,
        step_id: int,
        step_ns: int,
        notes: str = "",
    ) -> dict[str, Any]:
        self._require_phase("started")
        event = self._append_event(
            event_type="initial_ready_mark",
            step_id=step_id,
            step_ns=step_ns,
            event_source="experimenter",
            notes=notes,
            scripted_target_side=self.run_spec.initial_side,
        )
        self._phase = "ready"
        return event

    def commit_next_goal(
        self,
        *,
        step_id: int,
        step_ns: int,
        commit_ack_sources: Sequence[str],
        expected_return_swing_sign: int | None = None,
        notes: str = "",
    ) -> dict[str, Any]:
        self._require_phase("ready")
        target = self.next_target_side
        if target is None:
            raise TransitionContractError("all four goals have already been committed")
        ack_sources = tuple(dict.fromkeys(str(item) for item in commit_ack_sources))
        missing = [
            item for item in REQUIRED_GOAL_ACK_SOURCES if item not in ack_sources
        ]
        if missing:
            raise TransitionContractError(
                f"goal_commit missing acknowledgements: {', '.join(missing)}"
            )
        if expected_return_swing_sign not in {-1, 1, None}:
            raise TransitionContractError(
                "expected_return_swing_sign must be -1, +1, or null"
            )
        self._goal_epoch += 1
        event = self._append_event(
            event_type="goal_commit",
            step_id=step_id,
            step_ns=step_ns,
            event_source="sequencer",
            notes=notes,
            scripted_target_side=target,
            expected_return_swing_sign=expected_return_swing_sign,
            commit_ack_sources=ack_sources,
            goal_epoch=self._goal_epoch,
        )
        self._phase = "goal_committed"
        return event

    def mark_dump_end(
        self,
        *,
        step_id: int,
        step_ns: int,
        notes: str = "",
    ) -> dict[str, Any]:
        self._require_phase("goal_committed")
        event = self._append_event(
            event_type="dump_end_mark",
            step_id=step_id,
            step_ns=step_ns,
            event_source="experimenter",
            notes=notes,
            scripted_target_side=self.next_target_side,
            goal_epoch=self._goal_epoch,
        )
        self._phase = "dump_marked"
        return event

    def mark_target_ready(
        self,
        *,
        step_id: int,
        step_ns: int,
        notes: str = "",
    ) -> dict[str, Any]:
        self._require_phase("dump_marked")
        event = self._append_event(
            event_type="target_ready_mark",
            step_id=step_id,
            step_ns=step_ns,
            event_source="experimenter",
            notes=notes,
            scripted_target_side=self.next_target_side,
            goal_epoch=self._goal_epoch,
        )
        self._cycle_index += 1
        self._phase = (
            "cycles_complete"
            if self._cycle_index == len(self.run_spec.targets)
            else "ready"
        )
        return event

    def record_manual_intervention(
        self,
        *,
        step_id: int,
        step_ns: int,
        notes: str,
    ) -> dict[str, Any]:
        if self._phase in {"new", "complete", "aborted", "sealed"}:
            raise TransitionContractError(
                f"manual_intervention is invalid while phase={self._phase}"
            )
        return self._append_event(
            event_type="manual_intervention",
            step_id=step_id,
            step_ns=step_ns,
            event_source="experimenter",
            notes=notes,
            scripted_target_side=self.next_target_side,
            goal_epoch=self._goal_epoch or None,
        )

    def complete_run(
        self,
        *,
        step_id: int,
        step_ns: int,
        notes: str = "",
    ) -> dict[str, Any]:
        self._require_phase("cycles_complete")
        event = self._append_event(
            event_type="run_complete",
            step_id=step_id,
            step_ns=step_ns,
            event_source="system",
            notes=notes,
        )
        self._phase = "complete"
        return event

    def abort_run(
        self,
        *,
        step_id: int,
        step_ns: int,
        reason: str,
        safety_stop: bool = False,
    ) -> dict[str, Any]:
        if self._phase in {"new", "complete", "aborted", "sealed"}:
            raise TransitionContractError(
                f"run abort is invalid while phase={self._phase}"
            )
        reason = str(reason).strip()
        if not reason:
            raise TransitionContractError("run abort requires a non-empty reason")
        event = self._append_event(
            event_type="safety_stop" if safety_stop else "run_abort",
            step_id=step_id,
            step_ns=step_ns,
            event_source="system" if safety_stop else "experimenter",
            notes=reason,
            scripted_target_side=self.next_target_side,
            goal_epoch=self._goal_epoch or None,
        )
        self._phase = "aborted"
        return event

    def seal(
        self,
        *,
        raw_hdf5_path: Path | str | None = None,
        git_commit: str,
        resolved_config_sha256: str,
        owner_artifacts: Mapping[str, Path | str],
        field_context: Mapping[str, Any] | None = None,
        stop_reason: str = "",
    ) -> dict[str, Any]:
        """Seal a complete or aborted run after exact HDF5 alignment checks."""

        if self._phase not in {"complete", "aborted"}:
            raise TransitionContractError(
                f"cannot seal run while phase={self._phase}; terminal event is missing"
            )
        if self._sealed or self.manifest_path.exists() or self.checksums_path.exists():
            raise TransitionContractError(
                f"run package is already sealed: {self.run_dir}"
            )
        raw_path = Path(raw_hdf5_path) if raw_hdf5_path is not None else self.raw_path
        if raw_path.resolve() != self.raw_path.resolve():
            raise TransitionContractError(
                f"raw HDF5 must be written directly to {self.raw_path}, got {raw_path}"
            )
        alignment = validate_event_hdf5_alignment(raw_path, self._events)
        validate_event_sequence(self._events, self.run_spec)

        owner_entries: dict[str, Any] = {}
        for name, path_value in sorted(owner_artifacts.items()):
            owner_path = Path(path_value)
            if not owner_path.is_file():
                raise TransitionContractError(
                    f"owner artifact {name!r} does not exist: {owner_path}"
                )
            owner_entries[str(name)] = {
                "path": str(owner_path),
                "sha256": sha256_file(owner_path),
            }

        event_counts = Counter(str(event["event_type"]) for event in self._events)
        manifest = {
            "schema": RUN_MANIFEST_SCHEMA,
            "data_contract_version": DATA_CONTRACT_VERSION,
            "condition_schema": CONDITION_SCHEMA,
            "sealed_at_utc": _utc_now(),
            "immutable": True,
            "session_id": self.run_spec.session_id,
            "block_id": self.run_spec.block_id,
            "run_id": self.run_spec.run_id,
            "split": self.run_spec.split,
            "template_id": self.run_spec.template_id,
            "collection_rank": self.run_spec.collection_rank,
            "run_rank_in_block": self.run_spec.run_rank_in_block,
            "planned_sequence": list(self.run_spec.sequence),
            "planned_targets": list(self.run_spec.targets),
            "planned_transitions": list(self.run_spec.transitions),
            "status": "complete" if self._phase == "complete" else "aborted",
            "stop_reason": str(stop_reason),
            "completed_cycles": int(self._cycle_index),
            "first_failed_cycle_index": (
                None if self._phase == "complete" else int(self._cycle_index)
            ),
            "manual_intervention": int(event_counts["manual_intervention"] > 0),
            "safety_stop": int(event_counts["safety_stop"] > 0),
            "event_counts": dict(sorted(event_counts.items())),
            "alignment": alignment,
            "field_context": dict(field_context or {}),
            "provenance": {
                "git_commit": str(git_commit),
                "resolved_record_config_sha256": _require_sha256(
                    resolved_config_sha256,
                    "resolved_config_sha256",
                ),
                "owner_artifacts": owner_entries,
            },
            "artifacts": {
                "raw.hdf5": sha256_file(raw_path),
                "task_events.jsonl": sha256_file(self.events_path),
            },
        }
        _write_json_immutable(self.manifest_path, manifest)
        checksum_lines = [
            f"{sha256_file(self.raw_path)}  raw.hdf5",
            f"{sha256_file(self.events_path)}  task_events.jsonl",
            f"{sha256_file(self.manifest_path)}  run_manifest.json",
        ]
        _write_text_immutable(self.checksums_path, "\n".join(checksum_lines) + "\n")
        self._sealed = True
        self._phase = "sealed"
        return manifest

    def _create_empty_package(self) -> None:
        if self.run_dir.exists() and any(self.run_dir.iterdir()):
            raise TransitionContractError(
                f"refusing to reuse non-empty run directory: {self.run_dir}"
            )
        self.run_dir.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.events_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        os.close(fd)

    def _require_phase(self, expected: str) -> None:
        if self._phase != expected:
            raise TransitionContractError(
                f"expected phase={expected}, current phase={self._phase}"
            )

    def _append_event(
        self,
        *,
        event_type: str,
        step_id: int,
        step_ns: int,
        event_source: str,
        notes: str,
        scripted_target_side: str | None = None,
        expected_return_swing_sign: int | None = None,
        commit_ack_sources: Sequence[str] = (),
        goal_epoch: int | None = None,
    ) -> dict[str, Any]:
        if self._sealed:
            raise TransitionContractError("cannot append an event after package seal")
        step_id = int(step_id)
        step_ns = int(step_ns)
        if step_id < 0 or step_ns <= 0:
            raise TransitionContractError(
                "event step_id must be >=0 and step_ns must be >0"
            )
        if self._events:
            previous = self._events[-1]
            previous_pair = (
                int(previous["event_step_id"]),
                int(previous["event_step_ns"]),
            )
            current_pair = (step_id, step_ns)
            if current_pair < previous_pair:
                raise TransitionContractError(
                    f"event time moved backwards: {current_pair} < {previous_pair}"
                )

        cycle_index = self._cycle_index if self._cycle_index < 4 else None
        cycle_id = (
            f"{self.run_spec.run_id}_c{cycle_index + 1:02d}"
            if cycle_index is not None
            else None
        )
        target = scripted_target_side
        if target is not None and target not in SIDE_CODES:
            raise TransitionContractError(f"invalid scripted target side {target!r}")
        epoch = int(goal_epoch) if goal_epoch is not None else None
        goal_id = f"{self.run_spec.run_id}_g{epoch:02d}" if epoch is not None else None
        event = {
            "schema": TASK_EVENT_SCHEMA,
            "event_id": f"{self.run_spec.run_id}_e{len(self._events) + 1:03d}",
            "event_type": str(event_type),
            "event_step_id": step_id,
            "event_step_ns": step_ns,
            "session_id": self.run_spec.session_id,
            "block_id": self.run_spec.block_id,
            "run_id": self.run_spec.run_id,
            "cycle_id": cycle_id,
            "cycle_index": cycle_index,
            "goal_id": goal_id,
            "goal_epoch": epoch,
            "scripted_target_side": target,
            "target_side_code": SIDE_CODES[target] if target is not None else None,
            "expected_return_swing_sign": expected_return_swing_sign,
            "event_source": str(event_source),
            "commit_ack_sources": list(commit_ack_sources),
            "notes": str(notes),
        }
        encoded = json.dumps(
            event,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(encoded + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._events.append(event)
        return dict(event)


def validate_event_sequence(
    events: Sequence[Mapping[str, Any]],
    run_spec: TransitionRunSpec,
) -> dict[str, Any]:
    if not events:
        raise TransitionContractError("task event stream is empty")
    phase = "new"
    cycle_index = 0
    goal_epoch = 0
    terminal = ""
    previous_pair: tuple[int, int] | None = None
    for index, event in enumerate(events):
        if event.get("schema") != TASK_EVENT_SCHEMA:
            raise TransitionContractError(f"event {index} has an unsupported schema")
        for field, expected in (
            ("session_id", run_spec.session_id),
            ("block_id", run_spec.block_id),
            ("run_id", run_spec.run_id),
        ):
            if event.get(field) != expected:
                raise TransitionContractError(
                    f"event {index} {field}={event.get(field)!r}, expected {expected!r}"
                )
        pair = (int(event["event_step_id"]), int(event["event_step_ns"]))
        if previous_pair is not None and pair < previous_pair:
            raise TransitionContractError(f"event {index} is not monotonic")
        previous_pair = pair
        event_type = str(event.get("event_type", ""))
        if event_type in PASSIVE_EVENT_TYPES:
            if phase in {"new", "complete", "aborted"}:
                raise TransitionContractError(
                    f"event {index} {event_type} is invalid while phase={phase}"
                )
            continue
        if event_type == "run_start" and phase == "new":
            phase = "started"
            continue
        if event_type == "initial_ready_mark" and phase == "started":
            if event.get("scripted_target_side") != run_spec.initial_side:
                raise TransitionContractError("initial_ready_mark side is incorrect")
            phase = "ready"
            continue
        if event_type == "goal_commit" and phase == "ready":
            expected_target = run_spec.targets[cycle_index]
            if event.get("scripted_target_side") != expected_target:
                raise TransitionContractError(
                    f"cycle {cycle_index} goal target does not match the frozen sequence"
                )
            goal_epoch += 1
            if int(event.get("goal_epoch", -1)) != goal_epoch:
                raise TransitionContractError("goal_epoch is not contiguous")
            ack_sources = set(str(item) for item in event.get("commit_ack_sources", ()))
            missing = set(REQUIRED_GOAL_ACK_SOURCES) - ack_sources
            if missing:
                raise TransitionContractError(
                    f"goal_commit is missing acknowledgements: {sorted(missing)}"
                )
            phase = "goal_committed"
            continue
        if event_type == "dump_end_mark" and phase == "goal_committed":
            if event.get("scripted_target_side") != run_spec.targets[cycle_index]:
                raise TransitionContractError("dump_end_mark target is incorrect")
            phase = "dump_marked"
            continue
        if event_type == "target_ready_mark" and phase == "dump_marked":
            if event.get("scripted_target_side") != run_spec.targets[cycle_index]:
                raise TransitionContractError("target_ready_mark target is incorrect")
            cycle_index += 1
            phase = "cycles_complete" if cycle_index == 4 else "ready"
            continue
        if event_type == "run_complete" and phase == "cycles_complete":
            phase = "complete"
            terminal = event_type
            continue
        if event_type in {"run_abort", "safety_stop"} and phase not in {
            "new",
            "complete",
            "aborted",
        }:
            phase = "aborted"
            terminal = event_type
            continue
        raise TransitionContractError(
            f"event {index} type={event_type!r} is invalid while phase={phase}"
        )
    if phase not in {"complete", "aborted"}:
        raise TransitionContractError(
            f"event stream has no terminal event; phase={phase}"
        )
    return {
        "terminal_event": terminal,
        "completed_cycles": cycle_index,
        "goal_commits": goal_epoch,
    }


def validate_event_hdf5_alignment(
    raw_hdf5_path: Path | str,
    events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    path = Path(raw_hdf5_path)
    if not path.is_file():
        raise TransitionContractError(f"raw HDF5 does not exist: {path}")
    with h5py.File(path, "r") as handle:
        required = [
            "observations/qpos",
            "observations/qvel",
            "action",
            "timestamps/step_id",
            "timestamps/step_ns",
        ]
        missing = [name for name in required if name not in handle]
        if missing:
            raise TransitionContractError(
                f"raw HDF5 is missing required datasets: {', '.join(missing)}"
            )
        step_ids = np.asarray(handle["timestamps/step_id"][()], dtype=np.int64)
        step_ns = np.asarray(handle["timestamps/step_ns"][()], dtype=np.int64)
        lengths = {
            "qpos": int(handle["observations/qpos"].shape[0]),
            "qvel": int(handle["observations/qvel"].shape[0]),
            "action": int(handle["action"].shape[0]),
            "step_id": int(step_ids.shape[0]),
            "step_ns": int(step_ns.shape[0]),
        }
    if len(set(lengths.values())) != 1:
        raise TransitionContractError(f"raw HDF5 row lengths differ: {lengths}")
    if not len(step_ids):
        raise TransitionContractError("raw HDF5 contains no rows")
    if np.any(np.diff(step_ids) <= 0):
        raise TransitionContractError("timestamps/step_id must be strictly increasing")
    if np.any(np.diff(step_ns) <= 0):
        raise TransitionContractError("timestamps/step_ns must be strictly increasing")
    row_by_step_id = {int(step): int(stamp) for step, stamp in zip(step_ids, step_ns)}
    for index, event in enumerate(events):
        event_step = int(event["event_step_id"])
        event_ns = int(event["event_step_ns"])
        if event_step not in row_by_step_id:
            raise TransitionContractError(
                f"event {index} step_id={event_step} is absent from raw HDF5"
            )
        if row_by_step_id[event_step] != event_ns:
            raise TransitionContractError(
                f"event {index} step/time pair does not match raw HDF5: "
                f"event=({event_step},{event_ns}) "
                f"raw=({event_step},{row_by_step_id[event_step]})"
            )
    return {
        "status": "exact_step_time_match",
        "n_rows": int(len(step_ids)),
        "n_events": int(len(events)),
        "first_step_id": int(step_ids[0]),
        "last_step_id": int(step_ids[-1]),
        "first_step_ns": int(step_ns[0]),
        "last_step_ns": int(step_ns[-1]),
    }


def verify_run_package(run_dir: Path | str) -> dict[str, Any]:
    run_dir = Path(run_dir)
    raw_path = run_dir / "raw.hdf5"
    events_path = run_dir / "task_events.jsonl"
    manifest_path = run_dir / "run_manifest.json"
    checksums_path = run_dir / "SHA256SUMS.txt"
    for path in (raw_path, events_path, manifest_path, checksums_path):
        if not path.is_file():
            raise TransitionContractError(f"run package is missing {path.name}")

    expected_checksums = _read_sha256sums(checksums_path)
    required_names = {"raw.hdf5", "task_events.jsonl", "run_manifest.json"}
    if set(expected_checksums) != required_names:
        raise TransitionContractError(
            f"SHA256SUMS.txt must contain exactly {sorted(required_names)}"
        )
    for name, expected in expected_checksums.items():
        actual = sha256_file(run_dir / name)
        if actual != expected:
            raise TransitionContractError(
                f"checksum mismatch for {name}: expected {expected}, got {actual}"
            )

    manifest = _read_json_object(manifest_path)
    if manifest.get("schema") != RUN_MANIFEST_SCHEMA:
        raise TransitionContractError("run manifest schema mismatch")
    spec = TransitionRunSpec(
        session_id=_safe_id(manifest.get("session_id", ""), "session_id"),
        block_id=_safe_id(manifest.get("block_id", ""), "block_id"),
        run_id=_safe_id(manifest.get("run_id", ""), "run_id"),
        split=str(manifest.get("split", "")),
        template_id=str(manifest.get("template_id", "")),
        collection_rank=int(manifest.get("collection_rank", -1)),
        run_rank_in_block=int(manifest.get("run_rank_in_block", -1)),
        sequence=tuple(str(side) for side in manifest.get("planned_sequence", ())),
    )
    spec.validate()
    events = _read_jsonl(events_path)
    event_summary = validate_event_sequence(events, spec)
    alignment = validate_event_hdf5_alignment(raw_path, events)
    embedded = manifest.get("artifacts", {})
    for name in ("raw.hdf5", "task_events.jsonl"):
        if embedded.get(name) != sha256_file(run_dir / name):
            raise TransitionContractError(
                f"run manifest artifact checksum mismatch: {name}"
            )
    expected_status = (
        "complete" if event_summary["terminal_event"] == "run_complete" else "aborted"
    )
    if manifest.get("status") != expected_status:
        raise TransitionContractError(
            f"manifest status {manifest.get('status')!r} disagrees with event stream"
        )
    return {
        "status": "PASS",
        "run_dir": str(run_dir),
        "run_id": spec.run_id,
        "run_status": expected_status,
        "checksums_verified": sorted(expected_checksums),
        "event_summary": event_summary,
        "alignment": alignment,
    }


def sha256_file(path: Path | str, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def write_immutable_text(path: Path | str, value: str) -> Path:
    """Write a text artifact once, allowing only byte-identical retries."""

    resolved = Path(path)
    _write_text_immutable(resolved, str(value))
    return resolved


def _safe_id(value: Any, field: str) -> str:
    text = str(value)
    if not _SAFE_ID_RE.fullmatch(text):
        raise TransitionContractError(
            f"{field} must match {_SAFE_ID_RE.pattern!r}, got {text!r}"
        )
    return text


def _require_sha256(value: Any, field: str) -> str:
    text = str(value).lower()
    if not re.fullmatch(r"[0-9a-f]{64}", text):
        raise TransitionContractError(f"{field} must be a 64-character SHA-256")
    return text


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_json_immutable(path: Path, value: Mapping[str, Any]) -> None:
    _write_bytes_immutable(path, _canonical_json_bytes(value))


def _write_text_immutable(path: Path, value: str) -> None:
    _write_bytes_immutable(path, value.encode("utf-8"))


def _write_bytes_immutable(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == payload:
            return
        raise TransitionContractError(
            f"refusing to overwrite immutable artifact: {path}"
        )
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TransitionContractError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TransitionContractError(f"JSON root must be an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TransitionContractError(
                        f"{path}:{line_number} event must be a JSON object"
                    )
                events.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise TransitionContractError(f"cannot read JSONL {path}: {exc}") from exc
    return events


def _read_sha256sums(path: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 2:
            raise TransitionContractError(f"invalid checksum line {path}:{line_number}")
        checksum, name = parts
        _require_sha256(checksum, f"{path}:{line_number}")
        if Path(name).name != name or name in checksums:
            raise TransitionContractError(f"unsafe or duplicate checksum path {name!r}")
        checksums[name] = checksum
    return checksums


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")
