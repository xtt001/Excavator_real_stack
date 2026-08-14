"""Contracts and immutable packaging for v2.0.1 real-transition data.

This module deliberately owns no actuator control.  It creates a seeded,
field-observation-independent sequence plan, records append-only task events,
and seals each run against exact HDF5 ``step_id``/``step_ns`` rows.  Legacy v1
P0/P1 artifacts remain readable, but only the multi-sequence v2 plan is allowed
for new recording sessions.
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
from functools import lru_cache
from itertools import combinations, product
from pathlib import Path
from typing import Any

import h5py
import numpy as np

LEGACY_DATA_CONTRACT_VERSION = "real_transition_raw_v1"
DATA_CONTRACT_VERSION = "real_transition_raw_v2"
CONDITION_SCHEMA = "real_transition_condition_v1"
LEGACY_SEQUENCE_MANIFEST_SCHEMA = "real_transition_sequence_manifest_v1"
SEQUENCE_MANIFEST_SCHEMA = "real_transition_sequence_manifest_v2"
LEGACY_SPLIT_MANIFEST_SCHEMA = "real_transition_split_manifest_v1"
SPLIT_MANIFEST_SCHEMA = "real_transition_split_manifest_v2"
LEGACY_TASK_EVENT_SCHEMA = "real_transition_task_event_v1"
TASK_EVENT_SCHEMA = "real_transition_task_event_v2"
LEGACY_RUN_MANIFEST_SCHEMA = "real_transition_run_manifest_v1"
RUN_MANIFEST_SCHEMA = "real_transition_run_manifest_v2"
SESSION_PREPARATION_SCHEMA = "real_transition_session_preparation_v2"

SIDE_CODES = {"A": -1, "B": 1}
ATOMIC_TRANSITIONS = ("A->A", "A->B", "B->A", "B->B")
LEGACY_SEQUENCE_TEMPLATES: dict[str, tuple[str, ...]] = {
    "P0": ("A", "B", "B", "A", "A"),
    "P1": ("B", "A", "A", "B", "B"),
}
COLLECTION_PROFILE_ID = "balanced_multisequence_3_5_v1"
COLLECTION_CYCLE_COUNTS = (3, 4, 5)
COLLECTION_LENGTH_COUNTS = {3: 8, 4: 8, 5: 8}
_BLOCK_LENGTH_LAYOUTS = {
    "train_short": (3, 3, 4, 5),
    "evaluation": (3, 4, 4, 5),
    "train_long": (3, 4, 5, 5),
}
_BALANCE_AXES = (
    "split_x_atomic_transition",
    "block_x_initial_side_x_first_target",
    "block_x_first_three_cycle_positions_x_target",
    "pair_order_group_x_initial_side_x_first_target",
    "cycle_count_x_initial_side",
    "cycle_count_x_target_side",
)
_PRIMARY_EXCLUDED_SEQUENCES = frozenset(LEGACY_SEQUENCE_TEMPLATES.values())
_SEQUENCE_SELECTION_ATTEMPTS = 1_000_000
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
    sequence_id: str
    collection_rank: int
    run_rank_in_block: int
    sequence: tuple[str, ...]
    manifest_schema: str = SEQUENCE_MANIFEST_SCHEMA
    data_contract_version: str = DATA_CONTRACT_VERSION
    task_event_schema: str = TASK_EVENT_SCHEMA
    run_manifest_schema: str = RUN_MANIFEST_SCHEMA
    legacy_template_id: str | None = None
    matched_start_pair_id: str | None = None
    paired_run_id: str | None = None
    matched_start_pair_member_rank: int | None = None

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

    @property
    def cycle_count(self) -> int:
        return len(self.targets)

    @property
    def template_id(self) -> str | None:
        """Legacy compatibility accessor; new plans use ``sequence_id``."""

        return self.legacy_template_id

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        session_id: str,
        block_id: str,
        split: str,
        collection_rank: int,
        manifest_schema: str,
        data_contract_version: str,
    ) -> TransitionRunSpec:
        sequence = tuple(str(side) for side in value.get("sequence", ()))
        if manifest_schema == LEGACY_SEQUENCE_MANIFEST_SCHEMA:
            legacy_template_id = str(value.get("template_id", ""))
            sequence_id = f"legacy_{legacy_template_id}"
            task_event_schema = LEGACY_TASK_EVENT_SCHEMA
            run_manifest_schema = LEGACY_RUN_MANIFEST_SCHEMA
        elif manifest_schema == SEQUENCE_MANIFEST_SCHEMA:
            legacy_template_id = None
            sequence_id = _safe_id(value.get("sequence_id", ""), "sequence_id")
            task_event_schema = TASK_EVENT_SCHEMA
            run_manifest_schema = RUN_MANIFEST_SCHEMA
        else:
            raise TransitionContractError(
                f"unsupported sequence manifest schema {manifest_schema!r}"
            )
        spec = cls(
            session_id=_safe_id(session_id, "session_id"),
            block_id=_safe_id(block_id, "block_id"),
            run_id=_safe_id(value.get("run_id", ""), "run_id"),
            split=str(split),
            sequence_id=sequence_id,
            collection_rank=int(collection_rank),
            run_rank_in_block=int(value.get("run_rank_in_block", -1)),
            sequence=sequence,
            manifest_schema=manifest_schema,
            data_contract_version=str(data_contract_version),
            task_event_schema=task_event_schema,
            run_manifest_schema=run_manifest_schema,
            legacy_template_id=legacy_template_id,
            matched_start_pair_id=_optional_safe_id(
                value.get("matched_start_pair_id"), "matched_start_pair_id"
            ),
            paired_run_id=_optional_safe_id(
                value.get("paired_run_id"), "paired_run_id"
            ),
            matched_start_pair_member_rank=_optional_nonnegative_int(
                value.get("matched_start_pair_member_rank"),
                "matched_start_pair_member_rank",
            ),
        )
        spec.validate()
        if "initial_side" in value and str(value["initial_side"]) != spec.initial_side:
            raise TransitionContractError(
                f"run {spec.run_id} initial_side disagrees with sequence"
            )
        if "scripted_targets" in value and tuple(
            str(side) for side in value["scripted_targets"]
        ) != spec.targets:
            raise TransitionContractError(
                f"run {spec.run_id} scripted_targets disagree with sequence"
            )
        if "cycle_count" in value and int(value["cycle_count"]) != spec.cycle_count:
            raise TransitionContractError(
                f"run {spec.run_id} cycle_count disagrees with sequence"
            )
        return spec

    def validate(self) -> None:
        if self.split not in {"train", "validation", "locked_test"}:
            raise TransitionContractError(
                f"run {self.run_id} has unsupported split {self.split!r}"
            )
        if self.run_rank_in_block < 0:
            raise TransitionContractError(
                f"run {self.run_id} has invalid run_rank_in_block"
            )
        if len(self.sequence) < 2:
            raise TransitionContractError(
                f"run {self.run_id} must contain an initial side and at least "
                "one target"
            )
        invalid_sides = sorted(set(self.sequence) - set(SIDE_CODES))
        if invalid_sides:
            raise TransitionContractError(
                f"run {self.run_id} has invalid sides: {invalid_sides}"
            )
        if self.manifest_schema == LEGACY_SEQUENCE_MANIFEST_SCHEMA:
            if self.legacy_template_id not in LEGACY_SEQUENCE_TEMPLATES:
                raise TransitionContractError(
                    f"unsupported legacy template_id {self.legacy_template_id!r}"
                )
            expected = LEGACY_SEQUENCE_TEMPLATES[str(self.legacy_template_id)]
            if self.sequence != expected:
                raise TransitionContractError(
                    f"legacy run {self.run_id} sequence does not match "
                    f"{self.legacy_template_id}={expected!r}"
                )
        elif self.manifest_schema == SEQUENCE_MANIFEST_SCHEMA:
            _safe_id(self.sequence_id, "sequence_id")
            if self.legacy_template_id is not None:
                raise TransitionContractError(
                    f"v2 run {self.run_id} must not carry legacy template_id"
                )
        else:
            raise TransitionContractError(
                f"run {self.run_id} has unsupported manifest schema"
            )
        pair_fields_present = (
            self.matched_start_pair_id is not None,
            self.paired_run_id is not None,
            self.matched_start_pair_member_rank is not None,
        )
        if len(set(pair_fields_present)) != 1:
            raise TransitionContractError(
                f"run {self.run_id} must define all matched pair fields or none"
            )
        if (
            self.matched_start_pair_member_rank is not None
            and self.matched_start_pair_member_rank not in {0, 1}
        ):
            raise TransitionContractError(
                f"run {self.run_id} matched pair rank must be 0 or 1"
            )


def build_session_manifests(
    *,
    session_id: str,
    seed: int,
    created_at_utc: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the seeded 24-run/96-cycle multi-sequence recording plan.

    The only generator inputs are ``seed`` and identifiers.  No field image,
    terrain state, realized endpoint, or operator choice can affect target
    assignment after the plan is frozen.
    """

    session_id = _safe_id(session_id, "session_id")
    seed = int(seed)
    created_at_utc = str(created_at_utc or _utc_now())
    rng = random.Random(seed)
    selected = _select_balanced_experiment_blocks(rng)
    train_pairs = _pair_complementary_train_blocks(
        selected["train_short"],
        selected["train_long"],
    )
    rng.shuffle(train_pairs)
    core_train_short, core_train_long = train_pairs[0]
    expansion_train_short, expansion_train_long = train_pairs[1]
    evaluation = list(selected["evaluation"])
    rng.shuffle(evaluation)

    core_plans = [
        {
            "split": "train",
            "priority_tier": "minimum_64_cycle",
            "layout_id": "train_short",
            "sequences": core_train_short,
        },
        {
            "split": "train",
            "priority_tier": "minimum_64_cycle",
            "layout_id": "train_long",
            "sequences": core_train_long,
        },
        {
            "split": "validation",
            "priority_tier": "minimum_64_cycle",
            "layout_id": "evaluation",
            "sequences": evaluation.pop(),
        },
        {
            "split": "locked_test",
            "priority_tier": "minimum_64_cycle",
            "layout_id": "evaluation",
            "sequences": evaluation.pop(),
        },
    ]
    rng.shuffle(core_plans)
    expansion_plans = [
        {
            "split": "train",
            "priority_tier": "train_expansion_96_cycle",
            "layout_id": "train_short",
            "sequences": expansion_train_short,
        },
        {
            "split": "train",
            "priority_tier": "train_expansion_96_cycle",
            "layout_id": "train_long",
            "sequences": expansion_train_long,
        },
    ]
    rng.shuffle(expansion_plans)
    block_specs = core_plans + expansion_plans
    _assign_counterbalanced_pair_order(block_specs, rng=rng)

    blocks: list[dict[str, Any]] = []
    for collection_rank, block_plan in enumerate(block_specs):
        block_id = f"b{collection_rank + 1:02d}"
        pair_groups = []
        for initial_side in ("A", "B"):
            members = [
                sequence
                for sequence in block_plan["sequences"]
                if sequence[0] == initial_side
            ]
            if {sequence[1] for sequence in members} != {"A", "B"}:
                raise TransitionContractError(
                    f"internal sequence design lost {initial_side} matched-start pair"
                )
            first_target = block_plan["pair_first_target_by_initial"][initial_side]
            members.sort(key=lambda sequence: sequence[1] != first_target)
            pair_groups.append((initial_side, members))
        rng.shuffle(pair_groups)
        ordered_sequences = [
            sequence for _initial_side, members in pair_groups for sequence in members
        ]
        runs: list[dict[str, Any]] = []
        for run_rank, sequence in enumerate(ordered_sequences):
            run_id = f"{block_id}_r{run_rank + 1:02d}"
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
                    "sequence_id": _sequence_id(sequence),
                    "sequence": list(sequence),
                    "initial_side": sequence[0],
                    "scripted_targets": list(sequence[1:]),
                    "cycle_count": len(sequence) - 1,
                    "cycles": cycles,
                    "replacement_of": None,
                }
            )
        for initial_side in ("A", "B"):
            pair_runs = [run for run in runs if run["initial_side"] == initial_side]
            pair_id = f"{block_id}_matched_start_{initial_side}"
            if len(pair_runs) != 2:
                raise TransitionContractError(
                    f"internal sequence design expected two {initial_side} starts"
                )
            for member_rank, (run, paired) in enumerate(
                zip(pair_runs, reversed(pair_runs))
            ):
                run["matched_start_pair_id"] = pair_id
                run["paired_run_id"] = paired["run_id"]
                run["matched_start_pair_member_rank"] = member_rank
                run["matched_start_current_side"] = initial_side
                run["matched_start_target_side"] = run["scripted_targets"][0]
        blocks.append(
            {
                "block_id": block_id,
                "collection_rank": collection_rank,
                "priority_tier": block_plan["priority_tier"],
                "split": block_plan["split"],
                "layout_id": block_plan["layout_id"],
                "pair_order_balance_group": block_plan["pair_order_balance_group"],
                "pair_first_target_by_initial": dict(
                    block_plan["pair_first_target_by_initial"]
                ),
                "cycle_count": sum(run["cycle_count"] for run in runs),
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
        "collection_profile": {
            "profile_id": COLLECTION_PROFILE_ID,
            "execution_schema_accepts_arbitrary_positive_cycle_count": True,
            "authorized_recording_cycle_counts": list(COLLECTION_CYCLE_COUNTS),
            "run_count_by_cycle_count": {
                str(length): count
                for length, count in sorted(COLLECTION_LENGTH_COUNTS.items())
            },
            "sequence_reuse_within_session": False,
            "legacy_diagnostic_sequences_excluded_from_primary": {
                template_id: list(sequence)
                for template_id, sequence in LEGACY_SEQUENCE_TEMPLATES.items()
            },
            "field_observation_inputs_to_generator": [],
            "targets_frozen_before_recording": True,
            "balance_axes": list(_BALANCE_AXES),
        },
        "blocks": blocks,
    }
    sequence_manifest["coverage_report"] = _summarize_blocks(blocks)
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
            key: sequence_manifest["coverage_report"][key]
            for key in (
                "blocks",
                "runs",
                "cycles",
                "cycle_lengths",
                "cycle_lengths_by_split",
                "transitions_by_split",
            )
        },
    }
    validate_split_manifest(split_manifest, sequence_manifest=sequence_manifest)
    return sequence_manifest, split_manifest


def _assign_counterbalanced_pair_order(
    block_specs: Sequence[dict[str, Any]], *, rng: random.Random
) -> None:
    """Freeze A-first/B-first pair order without looking at field state.

    Each balance group contains two blocks.  For both initial sides, one block
    records the A-target member first and the other records the B-target member
    first.  This prevents pair position (and its associated terrain freshness)
    from becoming a deterministic proxy for the first goal.
    """

    groups: dict[str, list[dict[str, Any]]] = {
        "minimum_train": [],
        "train_expansion": [],
        "evaluation": [],
    }
    for block_spec in block_specs:
        split = str(block_spec["split"])
        priority_tier = str(block_spec["priority_tier"])
        if split == "train" and priority_tier == "minimum_64_cycle":
            group = "minimum_train"
        elif split == "train" and priority_tier == "train_expansion_96_cycle":
            group = "train_expansion"
        elif split in {"validation", "locked_test"}:
            group = "evaluation"
        else:
            raise TransitionContractError(
                f"cannot assign pair-order balance group for {split}/{priority_tier}"
            )
        block_spec["pair_order_balance_group"] = group
        block_spec["pair_first_target_by_initial"] = {}
        groups[group].append(block_spec)

    for group, members in groups.items():
        if len(members) != 2:
            raise TransitionContractError(
                f"pair-order balance group {group} must contain two blocks"
            )
        for initial_side in ("A", "B"):
            first_targets = ["A", "B"]
            rng.shuffle(first_targets)
            for block_spec, first_target in zip(members, first_targets):
                block_spec["pair_first_target_by_initial"][initial_side] = first_target


def _pair_complementary_train_blocks(
    short_blocks: Sequence[tuple[tuple[str, ...], ...]],
    long_blocks: Sequence[tuple[tuple[str, ...], ...]],
) -> list[
    tuple[tuple[tuple[str, ...], ...], tuple[tuple[str, ...], ...]]
]:
    """Pair 15-cycle and 17-cycle blocks into balanced 32-cycle train tiers."""

    remaining_long = list(long_blocks)
    pairs = []
    for short_block in short_blocks:
        short_counts = _transition_counts(short_block)
        deficit = [
            transition
            for transition in ATOMIC_TRANSITIONS
            if short_counts[transition] == 3
        ]
        if len(deficit) != 1:
            raise TransitionContractError(
                "short train block must have exactly one transition deficit"
            )
        matched_index = next(
            (
                index
                for index, long_block in enumerate(remaining_long)
                if _transition_counts(long_block)[deficit[0]] == 5
            ),
            None,
        )
        if matched_index is None:
            raise TransitionContractError(
                "cannot pair short and long train blocks into a balanced tier"
            )
        long_block = remaining_long.pop(matched_index)
        combined_counts = _transition_counts(short_block + long_block)
        if combined_counts != Counter(
            {transition: 8 for transition in ATOMIC_TRANSITIONS}
        ):
            raise TransitionContractError(
                "paired train blocks do not contain eight of each transition"
            )
        pairs.append((short_block, long_block))
    if remaining_long or len(pairs) != 2:
        raise TransitionContractError(
            "expected two complementary short/long train block pairs"
        )
    return pairs


@lru_cache(maxsize=None)
def _candidate_sequences(cycle_count: int) -> tuple[tuple[str, ...], ...]:
    candidates = []
    for sequence in product(("A", "B"), repeat=int(cycle_count) + 1):
        transitions = tuple(zip(sequence[:-1], sequence[1:]))
        if sequence in _PRIMARY_EXCLUDED_SEQUENCES:
            continue
        if len(set(sequence[1:])) < 2:
            continue
        if not any(current == target for current, target in transitions):
            continue
        if not any(current != target for current, target in transitions):
            continue
        candidates.append(tuple(sequence))
    return tuple(candidates)


@lru_cache(maxsize=None)
def _candidate_blocks(
    layout: tuple[int, ...],
) -> tuple[tuple[tuple[str, ...], ...], ...]:
    grouped_choices = []
    length_counts = Counter(layout)
    for cycle_count in sorted(length_counts):
        grouped_choices.append(
            tuple(
                combinations(
                    _candidate_sequences(cycle_count),
                    length_counts[cycle_count],
                )
            )
        )
    candidates = []
    for grouped in product(*grouped_choices):
        sequences = tuple(sequence for group in grouped for sequence in group)
        if _block_design_is_balanced(sequences, expected_cycles=sum(layout)):
            candidates.append(sequences)
    return tuple(candidates)


def _block_design_is_balanced(
    sequences: Sequence[Sequence[str]], *, expected_cycles: int
) -> bool:
    if Counter(sequence[0] for sequence in sequences) != Counter({"A": 2, "B": 2}):
        return False
    if Counter(
        f"{sequence[0]}->{sequence[1]}" for sequence in sequences
    ) != Counter({transition: 1 for transition in ATOMIC_TRANSITIONS}):
        return False
    for target_position in range(1, 4):
        if Counter(sequence[target_position] for sequence in sequences) != Counter(
            {"A": 2, "B": 2}
        ):
            return False
    counts = _transition_counts(sequences)
    quotient, remainder = divmod(int(expected_cycles), len(ATOMIC_TRANSITIONS))
    expected = [quotient] * (len(ATOMIC_TRANSITIONS) - remainder) + [
        quotient + 1
    ] * remainder
    return sorted(counts[transition] for transition in ATOMIC_TRANSITIONS) == expected


def _aggregate_design_is_balanced(
    sequences: Sequence[Sequence[str]], *, transitions_per_type: int
) -> bool:
    if _transition_counts(sequences) != Counter(
        {transition: int(transitions_per_type) for transition in ATOMIC_TRANSITIONS}
    ):
        return False
    max_cycles = max(len(sequence) - 1 for sequence in sequences)
    for target_position in range(1, max_cycles + 1):
        targets = [
            sequence[target_position]
            for sequence in sequences
            if len(sequence) > target_position
        ]
        if Counter(targets)["A"] != Counter(targets)["B"]:
            return False
    for cycle_count in COLLECTION_CYCLE_COUNTS:
        same_length = [
            sequence for sequence in sequences if len(sequence) - 1 == cycle_count
        ]
        if not same_length:
            continue
        initial_counts = Counter(sequence[0] for sequence in same_length)
        if initial_counts["A"] != initial_counts["B"]:
            return False
        target_counts = Counter(
            target for sequence in same_length for target in sequence[1:]
        )
        if target_counts["A"] != target_counts["B"]:
            return False
    return True


def _select_balanced_experiment_blocks(
    rng: random.Random,
) -> dict[str, tuple[tuple[tuple[str, ...], ...], ...]]:
    short_candidates = _candidate_blocks(_BLOCK_LENGTH_LAYOUTS["train_short"])
    long_candidates = _candidate_blocks(_BLOCK_LENGTH_LAYOUTS["train_long"])
    evaluation_candidates = _candidate_blocks(_BLOCK_LENGTH_LAYOUTS["evaluation"])

    train_blocks: tuple[tuple[tuple[str, ...], ...], ...] | None = None
    for _attempt in range(_SEQUENCE_SELECTION_ATTEMPTS):
        selected = (
            rng.choice(short_candidates),
            rng.choice(short_candidates),
            rng.choice(long_candidates),
            rng.choice(long_candidates),
        )
        flattened = [sequence for block in selected for sequence in block]
        if len(set(flattened)) != len(flattened):
            continue
        if _aggregate_design_is_balanced(flattened, transitions_per_type=16):
            train_blocks = selected
            break
    if train_blocks is None:
        raise TransitionContractError(
            "cannot generate a balanced train sequence plan for this seed"
        )

    used_train = {sequence for block in train_blocks for sequence in block}
    available_evaluation = tuple(
        block for block in evaluation_candidates if not (set(block) & used_train)
    )
    evaluation_blocks: tuple[tuple[tuple[str, ...], ...], ...] | None = None
    for _attempt in range(_SEQUENCE_SELECTION_ATTEMPTS):
        first = rng.choice(available_evaluation)
        second = rng.choice(available_evaluation)
        if set(first) & set(second):
            continue
        if _aggregate_design_is_balanced(first + second, transitions_per_type=8):
            evaluation_blocks = (first, second)
            break
    if evaluation_blocks is None:
        raise TransitionContractError(
            "cannot generate balanced validation/locked-test sequences for this seed"
        )
    return {
        "train_short": (train_blocks[0], train_blocks[1]),
        "train_long": (train_blocks[2], train_blocks[3]),
        "evaluation": evaluation_blocks,
    }


def _transition_counts(sequences: Sequence[Sequence[str]]) -> Counter[str]:
    return Counter(
        f"{current}->{target}"
        for sequence in sequences
        for current, target in zip(sequence[:-1], sequence[1:])
    )


def _sequence_id(sequence: Sequence[str]) -> str:
    return f"L{len(sequence) - 1}_{''.join(sequence)}"


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
    schema = str(manifest.get("schema", ""))
    if schema == SEQUENCE_MANIFEST_SCHEMA:
        _validate_sequence_manifest_v2(manifest)
        return
    if schema == LEGACY_SEQUENCE_MANIFEST_SCHEMA:
        _validate_sequence_manifest_v1(manifest)
        return
    raise TransitionContractError(
        "sequence schema must be one of "
        f"{LEGACY_SEQUENCE_MANIFEST_SCHEMA!r}, {SEQUENCE_MANIFEST_SCHEMA!r}"
    )


def _validate_sequence_manifest_v2(manifest: Mapping[str, Any]) -> None:
    if manifest.get("data_contract_version") != DATA_CONTRACT_VERSION:
        raise TransitionContractError(
            f"v2 sequence data_contract_version must be {DATA_CONTRACT_VERSION!r}"
        )
    if manifest.get("condition_schema") != CONDITION_SCHEMA:
        raise TransitionContractError(
            f"condition schema must be {CONDITION_SCHEMA!r}"
        )
    profile = manifest.get("collection_profile", {})
    if (
        not isinstance(profile, Mapping)
        or profile.get("profile_id") != COLLECTION_PROFILE_ID
    ):
        raise TransitionContractError(
            f"collection profile must be {COLLECTION_PROFILE_ID!r}"
        )
    if (
        profile.get("execution_schema_accepts_arbitrary_positive_cycle_count")
        is not True
    ):
        raise TransitionContractError(
            "execution schema must retain arbitrary positive cycle counts"
        )
    authorized_counts = tuple(
        int(value)
        for value in profile.get("authorized_recording_cycle_counts", ())
    )
    if authorized_counts != COLLECTION_CYCLE_COUNTS:
        raise TransitionContractError("collection profile cycle counts are not 3/4/5")
    expected_length_counts = {
        str(length): count
        for length, count in sorted(COLLECTION_LENGTH_COUNTS.items())
    }
    if profile.get("run_count_by_cycle_count") != expected_length_counts:
        raise TransitionContractError("collection profile length counts are incorrect")
    if profile.get("sequence_reuse_within_session") is not False:
        raise TransitionContractError(
            "sequence reuse must be disabled for this profile"
        )
    expected_legacy_exclusions = {
        template_id: list(sequence)
        for template_id, sequence in LEGACY_SEQUENCE_TEMPLATES.items()
    }
    if (
        profile.get("legacy_diagnostic_sequences_excluded_from_primary")
        != expected_legacy_exclusions
    ):
        raise TransitionContractError("legacy diagnostic exclusions are incorrect")
    if profile.get("field_observation_inputs_to_generator") != []:
        raise TransitionContractError(
            "sequence generation must not accept field-observation inputs"
        )
    if not bool(profile.get("targets_frozen_before_recording", False)):
        raise TransitionContractError("targets must be frozen before recording")
    if profile.get("balance_axes") != list(_BALANCE_AXES):
        raise TransitionContractError("collection profile balance axes are incorrect")
    if manifest.get("immutable_after_recording_starts") is not True:
        raise TransitionContractError("sequence manifest must be immutable after start")
    if manifest.get("side_codes") != SIDE_CODES:
        raise TransitionContractError("sequence side codes are incorrect")

    session_id = _safe_id(manifest.get("session_id", ""), "session_id")
    blocks = manifest.get("blocks", ())
    if not isinstance(blocks, Sequence) or isinstance(blocks, (str, bytes)):
        raise TransitionContractError("sequence blocks must be a list")
    if len(blocks) != 6:
        raise TransitionContractError(f"expected 6 blocks, found {len(blocks)}")

    seen_blocks: set[str] = set()
    seen_runs: set[str] = set()
    seen_sequences: set[tuple[str, ...]] = set()
    collection_ranks: set[int] = set()
    split_blocks: Counter[str] = Counter()
    priority_blocks: Counter[str] = Counter()
    transitions_by_priority: dict[str, Counter[str]] = {}
    pair_order_blocks: Counter[str] = Counter()
    pair_first_targets: dict[str, dict[str, Counter[str]]] = {}
    for block in blocks:
        if not isinstance(block, Mapping):
            raise TransitionContractError("each sequence block must be an object")
        block_id = _safe_id(block.get("block_id", ""), "block_id")
        if block_id in seen_blocks:
            raise TransitionContractError(f"duplicate block_id {block_id}")
        seen_blocks.add(block_id)
        collection_rank = int(block.get("collection_rank", -1))
        if collection_rank in collection_ranks:
            raise TransitionContractError(
                f"duplicate collection_rank {collection_rank}"
            )
        collection_ranks.add(collection_rank)
        split = str(block.get("split", ""))
        split_blocks[split] += 1
        priority_tier = str(block.get("priority_tier", ""))
        priority_blocks[priority_tier] += 1
        transitions_by_priority.setdefault(priority_tier, Counter())
        layout_id = str(block.get("layout_id", ""))
        if layout_id not in _BLOCK_LENGTH_LAYOUTS:
            raise TransitionContractError(
                f"block {block_id} has unknown layout_id {layout_id!r}"
            )
        pair_order_group = str(block.get("pair_order_balance_group", ""))
        if pair_order_group not in {
            "minimum_train",
            "train_expansion",
            "evaluation",
        }:
            raise TransitionContractError(
                f"block {block_id} has invalid pair-order balance group"
            )
        pair_order_blocks[pair_order_group] += 1
        pair_first_targets.setdefault(
            pair_order_group,
            {"A": Counter(), "B": Counter()},
        )
        runs = block.get("runs", ())
        if not isinstance(runs, Sequence) or isinstance(runs, (str, bytes)):
            raise TransitionContractError(f"block {block_id} runs must be a list")
        if len(runs) != 4:
            raise TransitionContractError(
                f"block {block_id} must contain 4 runs, found {len(runs)}"
            )
        run_ranks: set[int] = set()
        pair_members: dict[str, list[TransitionRunSpec]] = {}
        block_sequences: list[tuple[str, ...]] = []
        for run in runs:
            if not isinstance(run, Mapping):
                raise TransitionContractError(f"block {block_id} run must be an object")
            spec = TransitionRunSpec.from_mapping(
                run,
                session_id=session_id,
                block_id=block_id,
                split=split,
                collection_rank=collection_rank,
                manifest_schema=SEQUENCE_MANIFEST_SCHEMA,
                data_contract_version=DATA_CONTRACT_VERSION,
            )
            if spec.run_id in seen_runs:
                raise TransitionContractError(f"duplicate run_id {spec.run_id}")
            seen_runs.add(spec.run_id)
            if spec.sequence in seen_sequences:
                raise TransitionContractError(
                    f"sequence {_sequence_id(spec.sequence)} is reused within "
                    "the session"
                )
            seen_sequences.add(spec.sequence)
            if spec.sequence in _PRIMARY_EXCLUDED_SEQUENCES:
                raise TransitionContractError(
                    "legacy diagnostic sequence appears in primary plan: "
                    f"{spec.sequence}"
                )
            if spec.sequence_id != _sequence_id(spec.sequence):
                raise TransitionContractError(
                    f"run {spec.run_id} sequence_id does not match its sequence"
                )
            if spec.cycle_count not in COLLECTION_CYCLE_COUNTS:
                raise TransitionContractError(
                    f"run {spec.run_id} is outside the current 3/4/5 recording profile"
                )
            run_ranks.add(spec.run_rank_in_block)
            block_sequences.append(spec.sequence)
            _validate_run_cycle_payload(run, spec)
            assert spec.matched_start_pair_id is not None
            pair_members.setdefault(spec.matched_start_pair_id, []).append(spec)
        if run_ranks != set(range(4)):
            raise TransitionContractError(
                f"block {block_id} run ranks must be 0..3"
            )
        expected_layout = tuple(sorted(_BLOCK_LENGTH_LAYOUTS[layout_id]))
        actual_layout = tuple(sorted(len(sequence) - 1 for sequence in block_sequences))
        if actual_layout != expected_layout:
            raise TransitionContractError(
                f"block {block_id} cycle lengths {actual_layout} do not match "
                f"{layout_id}"
            )
        expected_block_cycles = sum(expected_layout)
        if int(block.get("cycle_count", -1)) != expected_block_cycles:
            raise TransitionContractError(f"block {block_id} cycle_count is incorrect")
        if not _block_design_is_balanced(
            block_sequences, expected_cycles=expected_block_cycles
        ):
            raise TransitionContractError(
                f"block {block_id} violates the balanced matched-start design"
            )
        transitions_by_priority[priority_tier].update(
            _transition_counts(block_sequences)
        )
        actual_pair_order = _validate_matched_start_pairs(pair_members)
        declared_pair_order = block.get("pair_first_target_by_initial", {})
        if declared_pair_order != actual_pair_order:
            raise TransitionContractError(
                f"block {block_id} declared pair order disagrees with run order"
            )
        for initial_side, first_target in actual_pair_order.items():
            pair_first_targets[pair_order_group][initial_side][first_target] += 1

    if collection_ranks != set(range(6)):
        raise TransitionContractError(
            f"collection ranks must be 0..5, found {sorted(collection_ranks)}"
        )
    if split_blocks != Counter({"train": 4, "validation": 1, "locked_test": 1}):
        raise TransitionContractError(
            f"unexpected block split counts: {dict(split_blocks)}"
        )
    if priority_blocks != Counter(
        {"minimum_64_cycle": 4, "train_expansion_96_cycle": 2}
    ):
        raise TransitionContractError(
            f"unexpected priority block counts: {dict(priority_blocks)}"
        )
    if pair_order_blocks != Counter(
        {"minimum_train": 2, "train_expansion": 2, "evaluation": 2}
    ):
        raise TransitionContractError(
            f"unexpected pair-order groups: {dict(pair_order_blocks)}"
        )
    for group, by_initial_side in pair_first_targets.items():
        for initial_side, counts in by_initial_side.items():
            if counts != Counter({"A": 1, "B": 1}):
                raise TransitionContractError(
                    f"pair order {group}/{initial_side} is not A/B counterbalanced"
                )
    expected_priority_transitions = {
        "minimum_64_cycle": Counter(
            {transition: 16 for transition in ATOMIC_TRANSITIONS}
        ),
        "train_expansion_96_cycle": Counter(
            {transition: 8 for transition in ATOMIC_TRANSITIONS}
        ),
    }
    if transitions_by_priority != expected_priority_transitions:
        raise TransitionContractError(
            "priority tiers do not preserve atomic-transition balance"
        )
    summary = _summarize_blocks(blocks)
    expected_summary = {
        "blocks": {"locked_test": 1, "train": 4, "validation": 1},
        "runs": {"locked_test": 4, "train": 16, "validation": 4},
        "cycles": {"locked_test": 16, "train": 64, "validation": 16},
        "transitions": {transition: 24 for transition in ATOMIC_TRANSITIONS},
        "transitions_by_split": {
            "locked_test": {transition: 4 for transition in ATOMIC_TRANSITIONS},
            "train": {transition: 16 for transition in ATOMIC_TRANSITIONS},
            "validation": {transition: 4 for transition in ATOMIC_TRANSITIONS},
        },
        "cycle_lengths": {"3": 8, "4": 8, "5": 8},
        "cycle_lengths_by_split": {
            "locked_test": {"3": 1, "4": 2, "5": 1},
            "train": {"3": 6, "4": 4, "5": 6},
            "validation": {"3": 1, "4": 2, "5": 1},
        },
    }
    for key, expected in expected_summary.items():
        if summary.get(key) != expected:
            raise TransitionContractError(
                f"sequence coverage {key}={summary.get(key)!r}, expected {expected!r}"
            )
    core_cycles = sum(
        int(block["cycle_count"])
        for block in blocks
        if block.get("priority_tier") == "minimum_64_cycle"
    )
    if core_cycles != 64:
        raise TransitionContractError(
            f"minimum priority blocks must contain 64 cycles, found {core_cycles}"
        )
    if manifest.get("coverage_report") != summary:
        raise TransitionContractError(
            "embedded coverage_report does not match the sequence blocks"
        )


def _validate_sequence_manifest_v1(manifest: Mapping[str, Any]) -> None:
    if manifest.get("data_contract_version") not in {
        None,
        LEGACY_DATA_CONTRACT_VERSION,
    }:
        raise TransitionContractError("legacy sequence data contract version mismatch")
    session_id = _safe_id(manifest.get("session_id", ""), "session_id")
    blocks = manifest.get("blocks", ())
    if not isinstance(blocks, Sequence) or isinstance(blocks, (str, bytes)):
        raise TransitionContractError("legacy sequence blocks must be a list")
    if len(blocks) != 6:
        raise TransitionContractError(
            f"legacy plan expected 6 blocks, found {len(blocks)}"
        )
    collection_ranks: set[int] = set()
    split_blocks: Counter[str] = Counter()
    transition_counts: Counter[str] = Counter()
    seen_runs: set[str] = set()
    for block in blocks:
        block_id = _safe_id(block.get("block_id", ""), "block_id")
        collection_rank = int(block.get("collection_rank", -1))
        collection_ranks.add(collection_rank)
        split = str(block.get("split", ""))
        split_blocks[split] += 1
        runs = block.get("runs", ())
        if not isinstance(runs, Sequence) or len(runs) != 4:
            raise TransitionContractError(
                f"legacy block {block_id} must contain four runs"
            )
        templates: Counter[str] = Counter()
        for run in runs:
            spec = TransitionRunSpec.from_mapping(
                run,
                session_id=session_id,
                block_id=block_id,
                split=split,
                collection_rank=collection_rank,
                manifest_schema=LEGACY_SEQUENCE_MANIFEST_SCHEMA,
                data_contract_version=LEGACY_DATA_CONTRACT_VERSION,
            )
            if spec.run_id in seen_runs:
                raise TransitionContractError(f"duplicate legacy run_id {spec.run_id}")
            seen_runs.add(spec.run_id)
            templates[str(spec.template_id)] += 1
            transition_counts.update(spec.transitions)
        if templates != Counter({"P0": 2, "P1": 2}):
            raise TransitionContractError(
                f"legacy block {block_id} must contain two P0 and two P1 runs"
            )
    if collection_ranks != set(range(6)):
        raise TransitionContractError("legacy collection ranks must be 0..5")
    if split_blocks != Counter({"train": 4, "validation": 1, "locked_test": 1}):
        raise TransitionContractError("legacy split block counts are invalid")
    if transition_counts != Counter(
        {transition: 24 for transition in ATOMIC_TRANSITIONS}
    ):
        raise TransitionContractError("legacy transition coverage is invalid")


def _validate_run_cycle_payload(
    run: Mapping[str, Any], spec: TransitionRunSpec
) -> None:
    cycles = run.get("cycles", ())
    if not isinstance(cycles, Sequence) or isinstance(cycles, (str, bytes)):
        raise TransitionContractError(f"run {spec.run_id} cycles must be a list")
    if len(cycles) != spec.cycle_count:
        raise TransitionContractError(
            f"run {spec.run_id} cycle payload length is wrong"
        )
    for index, (cycle, current, target) in enumerate(
        zip(cycles, spec.sequence[:-1], spec.sequence[1:])
    ):
        expected = {
            "cycle_id": f"{spec.run_id}_c{index + 1:02d}",
            "cycle_index": index,
            "current_side": current,
            "scripted_target_side": target,
            "target_side_code": SIDE_CODES[target],
            "transition": f"{current}->{target}",
        }
        for field, value in expected.items():
            if cycle.get(field) != value:
                raise TransitionContractError(
                    f"run {spec.run_id} cycle {index} {field} is inconsistent"
                )


def _validate_matched_start_pairs(
    pair_members: Mapping[str, Sequence[TransitionRunSpec]],
) -> dict[str, str]:
    if len(pair_members) != 2:
        raise TransitionContractError("each block must contain two matched-start pairs")
    covered_sides: set[str] = set()
    first_target_by_initial: dict[str, str] = {}
    for pair_id, members in pair_members.items():
        if len(members) != 2:
            raise TransitionContractError(
                f"matched-start pair {pair_id} must contain exactly two runs"
            )
        if {member.matched_start_pair_member_rank for member in members} != {0, 1}:
            raise TransitionContractError(
                f"matched-start pair {pair_id} ranks must be 0 and 1"
            )
        first, second = sorted(
            members,
            key=lambda member: int(member.matched_start_pair_member_rank or 0),
        )
        if first.initial_side != second.initial_side:
            raise TransitionContractError(
                f"matched-start pair {pair_id} initial sides differ"
            )
        if {first.targets[0], second.targets[0]} != {"A", "B"}:
            raise TransitionContractError(
                f"matched-start pair {pair_id} must branch to A and B"
            )
        if first.paired_run_id != second.run_id or second.paired_run_id != first.run_id:
            raise TransitionContractError(
                f"matched-start pair {pair_id} paired_run_id is not symmetric"
            )
        covered_sides.add(first.initial_side)
        first_target_by_initial[first.initial_side] = first.targets[0]
    if covered_sides != {"A", "B"}:
        raise TransitionContractError("matched-start pairs must cover current A and B")
    return first_target_by_initial


def validate_split_manifest(
    manifest: Mapping[str, Any],
    *,
    sequence_manifest: Mapping[str, Any],
) -> None:
    validate_sequence_manifest(sequence_manifest)
    sequence_schema = str(sequence_manifest.get("schema", ""))
    expected_split_schema = (
        SPLIT_MANIFEST_SCHEMA
        if sequence_schema == SEQUENCE_MANIFEST_SCHEMA
        else LEGACY_SPLIT_MANIFEST_SCHEMA
    )
    if manifest.get("schema") != expected_split_schema:
        raise TransitionContractError(
            f"split schema must be {expected_split_schema!r}"
        )
    expected_data_contract = (
        DATA_CONTRACT_VERSION
        if sequence_schema == SEQUENCE_MANIFEST_SCHEMA
        else LEGACY_DATA_CONTRACT_VERSION
    )
    if manifest.get("data_contract_version") not in {
        None,
        expected_data_contract,
    }:
        raise TransitionContractError("split data contract version mismatch")
    if manifest.get("session_id") != sequence_manifest.get("session_id"):
        raise TransitionContractError("split and sequence session_id differ")
    expected_sequence_sha = _sha256_bytes(
        _canonical_json_bytes(dict(sequence_manifest))
    )
    if manifest.get("sequence_manifest_sha256") != expected_sequence_sha:
        raise TransitionContractError("split manifest sequence checksum mismatch")

    expected: dict[str, dict[str, Any]] = {
        str(block["block_id"]): {
            "split": str(block["split"]),
            "run_ids": [str(run["run_id"]) for run in block["runs"]],
        }
        for block in sequence_manifest["blocks"]
    }
    if sequence_schema == SEQUENCE_MANIFEST_SCHEMA:
        for block in sequence_manifest["blocks"]:
            expected[str(block["block_id"])].update(
                {
                    "priority_tier": str(block["priority_tier"]),
                    "collection_rank": int(block["collection_rank"]),
                }
            )
    assignments = manifest.get("block_assignments", ())
    if not isinstance(assignments, Sequence) or isinstance(
        assignments, (str, bytes)
    ):
        raise TransitionContractError("split block_assignments must be a list")
    actual: dict[str, dict[str, Any]] = {}
    for assignment in assignments:
        if not isinstance(assignment, Mapping):
            raise TransitionContractError("each split assignment must be an object")
        block_id = str(assignment.get("block_id", ""))
        if block_id in actual:
            raise TransitionContractError(f"duplicate split assignment for {block_id}")
        actual[block_id] = {
            "split": str(assignment.get("split", "")),
            "run_ids": [str(run_id) for run_id in assignment.get("run_ids", ())],
        }
        if sequence_schema == SEQUENCE_MANIFEST_SCHEMA:
            actual[block_id].update(
                {
                    "priority_tier": str(assignment.get("priority_tier", "")),
                    "collection_rank": int(assignment.get("collection_rank", -1)),
                }
            )
    if actual != expected:
        raise TransitionContractError(
            "split assignments do not match sequence manifest"
        )
    if sequence_schema == SEQUENCE_MANIFEST_SCHEMA:
        expected_rules = {
            "unit": "whole_source_block",
            "cycles_from_one_run_stay_in_one_split": True,
            "locked_test_task_results_hidden_until_authorized": True,
            "post_collection_reassignment_allowed": False,
        }
        if manifest.get("rules") != expected_rules:
            raise TransitionContractError("split rules do not match the v2 contract")
        summary = _summarize_blocks(sequence_manifest["blocks"])
        expected_counts = {
            key: summary[key]
            for key in (
                "blocks",
                "runs",
                "cycles",
                "cycle_lengths",
                "cycle_lengths_by_split",
                "transitions_by_split",
            )
        }
        if manifest.get("expected_counts") != expected_counts:
            raise TransitionContractError(
                "split expected_counts do not match sequence coverage"
            )


def summarize_sequence_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    validate_sequence_manifest(manifest)
    return _summarize_blocks(manifest["blocks"])


def _summarize_blocks(blocks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    split_blocks: Counter[str] = Counter()
    split_runs: Counter[str] = Counter()
    split_cycles: Counter[str] = Counter()
    transitions: Counter[str] = Counter()
    transitions_by_split: dict[str, Counter[str]] = {}
    transitions_by_priority_tier: dict[str, Counter[str]] = {}
    cycle_lengths: Counter[str] = Counter()
    cycle_lengths_by_split: dict[str, Counter[str]] = {}
    initial_sides_by_split: dict[str, Counter[str]] = {}
    targets_by_cycle_index_by_split: dict[str, dict[str, Counter[str]]] = {}
    matched_pairs_by_split: Counter[str] = Counter()
    pair_first_targets_by_balance_group: dict[str, dict[str, Counter[str]]] = {}
    sequences: set[tuple[str, ...]] = set()
    for block in blocks:
        split = str(block["split"])
        split_blocks[split] += 1
        transitions_by_split.setdefault(split, Counter())
        priority_tier = str(block.get("priority_tier", ""))
        transitions_by_priority_tier.setdefault(priority_tier, Counter())
        cycle_lengths_by_split.setdefault(split, Counter())
        initial_sides_by_split.setdefault(split, Counter())
        targets_by_cycle_index_by_split.setdefault(split, {})
        matched_pairs_by_split[split] += len(
            {str(run.get("matched_start_pair_id")) for run in block["runs"]}
            - {"None", ""}
        )
        pair_order_group = str(block.get("pair_order_balance_group", ""))
        declared_pair_order = block.get("pair_first_target_by_initial", {})
        if pair_order_group and isinstance(declared_pair_order, Mapping):
            by_initial = pair_first_targets_by_balance_group.setdefault(
                pair_order_group,
                {"A": Counter(), "B": Counter()},
            )
            for initial_side in ("A", "B"):
                first_target = str(declared_pair_order.get(initial_side, ""))
                if first_target in SIDE_CODES:
                    by_initial[initial_side][first_target] += 1
        for run in block["runs"]:
            split_runs[split] += 1
            sequence = tuple(run["sequence"])
            sequences.add(sequence)
            run_transitions = [
                f"{current}->{target}"
                for current, target in zip(sequence[:-1], sequence[1:])
            ]
            split_cycles[split] += len(run_transitions)
            transitions.update(run_transitions)
            transitions_by_split[split].update(run_transitions)
            transitions_by_priority_tier[priority_tier].update(run_transitions)
            length_key = str(len(run_transitions))
            cycle_lengths[length_key] += 1
            cycle_lengths_by_split[split][length_key] += 1
            initial_sides_by_split[split][sequence[0]] += 1
            for cycle_index, target in enumerate(sequence[1:]):
                index_key = str(cycle_index)
                targets_by_cycle_index_by_split[split].setdefault(
                    index_key, Counter()
                )[target] += 1
    return {
        "blocks": dict(sorted(split_blocks.items())),
        "runs": dict(sorted(split_runs.items())),
        "cycles": dict(sorted(split_cycles.items())),
        "transitions": dict(sorted(transitions.items())),
        "transitions_by_split": {
            split: dict(sorted(counts.items()))
            for split, counts in sorted(transitions_by_split.items())
        },
        "transitions_by_priority_tier": {
            priority_tier: dict(sorted(counts.items()))
            for priority_tier, counts in sorted(
                transitions_by_priority_tier.items()
            )
        },
        "cycle_lengths": dict(sorted(cycle_lengths.items())),
        "cycle_lengths_by_split": {
            split: dict(sorted(counts.items()))
            for split, counts in sorted(cycle_lengths_by_split.items())
        },
        "initial_sides_by_split": {
            split: dict(sorted(counts.items()))
            for split, counts in sorted(initial_sides_by_split.items())
        },
        "targets_by_cycle_index_by_split": {
            split: {
                cycle_index: dict(sorted(counts.items()))
                for cycle_index, counts in sorted(
                    by_index.items(), key=lambda item: int(item[0])
                )
            }
            for split, by_index in sorted(targets_by_cycle_index_by_split.items())
        },
        "matched_start_pairs_by_split": dict(sorted(matched_pairs_by_split.items())),
        "pair_first_targets_by_balance_group": {
            group: {
                initial_side: dict(sorted(counts.items()))
                for initial_side, counts in sorted(by_initial.items())
            }
            for group, by_initial in sorted(
                pair_first_targets_by_balance_group.items()
            )
        },
        "unique_sequence_count": len(sequences),
    }


def iter_run_specs(manifest: Mapping[str, Any]) -> Iterable[TransitionRunSpec]:
    validate_sequence_manifest(manifest)
    session_id = str(manifest["session_id"])
    manifest_schema = str(manifest["schema"])
    data_contract_version = str(
        manifest.get(
            "data_contract_version",
            LEGACY_DATA_CONTRACT_VERSION,
        )
    )
    for block in sorted(manifest["blocks"], key=lambda item: item["collection_rank"]):
        for run in sorted(block["runs"], key=lambda item: item["run_rank_in_block"]):
            yield TransitionRunSpec.from_mapping(
                run,
                session_id=session_id,
                block_id=str(block["block_id"]),
                split=str(block["split"]),
                collection_rank=int(block["collection_rank"]),
                manifest_schema=manifest_schema,
                data_contract_version=data_contract_version,
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
            raise TransitionContractError(
                "all planned goals have already been committed"
            )
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
        realized_target_side: str | None = None,
        notes: str = "",
    ) -> dict[str, Any]:
        self._require_phase("dump_marked")
        expected_target = self.next_target_side
        if self.run_spec.manifest_schema == SEQUENCE_MANIFEST_SCHEMA:
            if realized_target_side not in SIDE_CODES:
                raise TransitionContractError(
                    "v2 target_ready_mark requires realized_target_side A or B"
                )
            if realized_target_side != expected_target:
                raise TransitionContractError(
                    "realized target does not match the scripted target; abort the run"
                )
        elif realized_target_side is None:
            realized_target_side = expected_target
        event = self._append_event(
            event_type="target_ready_mark",
            step_id=step_id,
            step_ns=step_ns,
            event_source="experimenter",
            notes=notes,
            scripted_target_side=expected_target,
            realized_target_side=realized_target_side,
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
        realized_target_side: str | None = None,
    ) -> dict[str, Any]:
        if self._phase in {"new", "complete", "aborted", "sealed"}:
            raise TransitionContractError(
                f"run abort is invalid while phase={self._phase}"
            )
        reason = str(reason).strip()
        if not reason:
            raise TransitionContractError("run abort requires a non-empty reason")
        if realized_target_side is not None and realized_target_side not in SIDE_CODES:
            raise TransitionContractError(
                "realized_target_side must be A, B, or null"
            )
        event = self._append_event(
            event_type="safety_stop" if safety_stop else "run_abort",
            step_id=step_id,
            step_ns=step_ns,
            event_source="system" if safety_stop else "experimenter",
            notes=reason,
            scripted_target_side=self.next_target_side,
            realized_target_side=realized_target_side,
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
        realized_targets = [
            str(event["realized_target_side"])
            for event in self._events
            if event.get("event_type") == "target_ready_mark"
            and event.get("realized_target_side") in SIDE_CODES
        ]
        manifest = {
            "schema": self.run_spec.run_manifest_schema,
            "data_contract_version": self.run_spec.data_contract_version,
            "condition_schema": CONDITION_SCHEMA,
            "sealed_at_utc": _utc_now(),
            "immutable": True,
            "session_id": self.run_spec.session_id,
            "block_id": self.run_spec.block_id,
            "run_id": self.run_spec.run_id,
            "split": self.run_spec.split,
            "sequence_id": self.run_spec.sequence_id,
            "legacy_template_id": self.run_spec.legacy_template_id,
            "matched_start_pair_id": self.run_spec.matched_start_pair_id,
            "paired_run_id": self.run_spec.paired_run_id,
            "matched_start_pair_member_rank": (
                self.run_spec.matched_start_pair_member_rank
            ),
            "collection_rank": self.run_spec.collection_rank,
            "run_rank_in_block": self.run_spec.run_rank_in_block,
            "planned_sequence": list(self.run_spec.sequence),
            "planned_targets": list(self.run_spec.targets),
            "planned_transitions": list(self.run_spec.transitions),
            "planned_cycle_count": self.run_spec.cycle_count,
            "realized_targets": realized_targets,
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
        realized_target_side: str | None = None,
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

        cycle_index = (
            self._cycle_index
            if self._cycle_index < len(self.run_spec.targets)
            else None
        )
        cycle_id = (
            f"{self.run_spec.run_id}_c{cycle_index + 1:02d}"
            if cycle_index is not None
            else None
        )
        target = scripted_target_side
        if target is not None and target not in SIDE_CODES:
            raise TransitionContractError(f"invalid scripted target side {target!r}")
        if realized_target_side is not None and realized_target_side not in SIDE_CODES:
            raise TransitionContractError(
                f"invalid realized target side {realized_target_side!r}"
            )
        epoch = int(goal_epoch) if goal_epoch is not None else None
        goal_id = f"{self.run_spec.run_id}_g{epoch:02d}" if epoch is not None else None
        event = {
            "schema": self.run_spec.task_event_schema,
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
            "realized_target_side": realized_target_side,
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
    realized_targets: list[str] = []
    for index, event in enumerate(events):
        if event.get("schema") != run_spec.task_event_schema:
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
            if cycle_index >= run_spec.cycle_count:
                raise TransitionContractError(
                    "goal committed after all cycles completed"
                )
            expected_target = run_spec.targets[cycle_index]
            if event.get("scripted_target_side") != expected_target:
                raise TransitionContractError(
                    f"cycle {cycle_index} goal target does not match the "
                    "frozen sequence"
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
            expected_target = run_spec.targets[cycle_index]
            if event.get("scripted_target_side") != expected_target:
                raise TransitionContractError("target_ready_mark target is incorrect")
            if run_spec.task_event_schema == TASK_EVENT_SCHEMA:
                realized = event.get("realized_target_side")
                if realized != expected_target:
                    raise TransitionContractError(
                        "target_ready_mark realized target does not match "
                        "scripted target"
                    )
                realized_targets.append(str(realized))
            cycle_index += 1
            phase = (
                "cycles_complete"
                if cycle_index == run_spec.cycle_count
                else "ready"
            )
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
        "realized_targets": realized_targets,
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
    manifest_schema = str(manifest.get("schema", ""))
    if manifest_schema == RUN_MANIFEST_SCHEMA:
        sequence_manifest_schema = SEQUENCE_MANIFEST_SCHEMA
        task_event_schema = TASK_EVENT_SCHEMA
        data_contract_version = DATA_CONTRACT_VERSION
        sequence_id = _safe_id(manifest.get("sequence_id", ""), "sequence_id")
        legacy_template_id = None
    elif manifest_schema == LEGACY_RUN_MANIFEST_SCHEMA:
        sequence_manifest_schema = LEGACY_SEQUENCE_MANIFEST_SCHEMA
        task_event_schema = LEGACY_TASK_EVENT_SCHEMA
        data_contract_version = LEGACY_DATA_CONTRACT_VERSION
        legacy_template_id = str(
            manifest.get("template_id") or manifest.get("legacy_template_id") or ""
        )
        sequence_id = f"legacy_{legacy_template_id}"
    else:
        raise TransitionContractError("run manifest schema mismatch")
    spec = TransitionRunSpec(
        session_id=_safe_id(manifest.get("session_id", ""), "session_id"),
        block_id=_safe_id(manifest.get("block_id", ""), "block_id"),
        run_id=_safe_id(manifest.get("run_id", ""), "run_id"),
        split=str(manifest.get("split", "")),
        sequence_id=sequence_id,
        collection_rank=int(manifest.get("collection_rank", -1)),
        run_rank_in_block=int(manifest.get("run_rank_in_block", -1)),
        sequence=tuple(str(side) for side in manifest.get("planned_sequence", ())),
        manifest_schema=sequence_manifest_schema,
        data_contract_version=data_contract_version,
        task_event_schema=task_event_schema,
        run_manifest_schema=manifest_schema,
        legacy_template_id=legacy_template_id,
        matched_start_pair_id=_optional_safe_id(
            manifest.get("matched_start_pair_id"), "matched_start_pair_id"
        ),
        paired_run_id=_optional_safe_id(
            manifest.get("paired_run_id"), "paired_run_id"
        ),
        matched_start_pair_member_rank=_optional_nonnegative_int(
            manifest.get("matched_start_pair_member_rank"),
            "matched_start_pair_member_rank",
        ),
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
    if int(manifest.get("completed_cycles", -1)) != int(
        event_summary["completed_cycles"]
    ):
        raise TransitionContractError(
            "run manifest completed_cycles disagrees with event stream"
        )
    if manifest_schema == RUN_MANIFEST_SCHEMA:
        if int(manifest.get("planned_cycle_count", -1)) != spec.cycle_count:
            raise TransitionContractError(
                "run manifest planned_cycle_count is incorrect"
            )
        if list(manifest.get("planned_targets", ())) != list(spec.targets):
            raise TransitionContractError("run manifest planned_targets are incorrect")
        if list(manifest.get("realized_targets", ())) != list(
            event_summary["realized_targets"]
        ):
            raise TransitionContractError(
                "run manifest realized_targets disagree with event stream"
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


def _optional_safe_id(value: Any, field: str) -> str | None:
    if value is None or str(value).strip() == "":
        return None
    return _safe_id(value, field)


def _optional_nonnegative_int(value: Any, field: str) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise TransitionContractError(f"{field} must be an integer or null") from exc
    if parsed < 0:
        raise TransitionContractError(f"{field} must be non-negative or null")
    return parsed


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
