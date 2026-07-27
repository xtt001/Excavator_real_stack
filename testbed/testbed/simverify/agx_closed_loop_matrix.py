"""Recompute a checksum-bound paired AGX closed-loop diagnostic matrix."""

from __future__ import annotations

import json
import math
import os
import shutil
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from testbed.simverify.artifacts import (
    artifact_identity,
    verify_checksums,
    write_checksums,
    write_json,
)
from testbed.simverify.contracts import FROZEN_TARGET_HZ, sha256_file

MATRIX_SCHEMA = "simverify_agx_closed_loop_paired_diagnostic_v1"
EXPECTED_BASELINES = ("B1.4", "B2.4")
SECTOR_LABELS = ("left", "center", "right")


def classify_swing_sector(
    swing_qpos: float,
    *,
    boundaries: Sequence[float],
    review_margin: float,
) -> str:
    """Classify source-domain swing using the frozen M0 review envelope."""

    if len(boundaries) != 2:
        raise ValueError("sector boundaries must contain exactly two values")
    lower, upper = map(float, boundaries)
    value = float(swing_qpos)
    margin = float(review_margin)
    if (
        not all(math.isfinite(item) for item in (lower, upper, value, margin))
        or lower >= upper
        or margin < 0.0
    ):
        raise ValueError("invalid sector boundary contract")
    if abs(value - lower) <= margin or abs(value - upper) <= margin:
        return "boundary_review"
    if value < lower:
        return "left"
    if value < upper:
        return "center"
    return "right"


def extract_observable_cycle_entries(
    policy_rows: Sequence[Mapping[str, Any]],
    *,
    action_deadzone: float,
    dump_swing_threshold: float,
    minimum_policy_ticks: int,
    sector_boundaries: Sequence[float],
    sector_review_margin: float,
) -> dict[str, Any]:
    """Find dig-entry proxies separated by observable dump-release events."""

    if not policy_rows:
        raise ValueError("policy_rows must not be empty")
    if int(minimum_policy_ticks) <= 0:
        raise ValueError("minimum_policy_ticks must be positive")
    ticks = [int(row["policy_tick"]) for row in policy_rows]
    if ticks != list(range(len(policy_rows))):
        raise ValueError("policy rows must be contiguous and zero-based")
    qpos = np.asarray([row["qpos"] for row in policy_rows], dtype=np.float64)
    action = np.asarray(
        [row["actual_sent_action"] for row in policy_rows],
        dtype=np.float64,
    )
    if qpos.shape != (len(policy_rows), 4) or action.shape != (
        len(policy_rows),
        4,
    ):
        raise ValueError("policy rows require qpos/action shape (N,4)")
    if not np.all(np.isfinite(qpos)) or not np.all(np.isfinite(action)):
        raise ValueError("policy rows contain non-finite qpos/action")

    deadzone = float(action_deadzone)
    dump_threshold = float(dump_swing_threshold)
    minimum = int(minimum_policy_ticks)
    positive_runs = _runs(action[:, 3] > deadzone, minimum=minimum)
    raw_releases = _runs(
        (action[:, 3] < -deadzone) & (qpos[:, 0] > dump_threshold),
        minimum=minimum,
    )
    releases = _merge_release_runs(
        raw_releases,
        swing=qpos[:, 0],
        dump_swing_threshold=dump_threshold,
    )

    selected_positive_runs: list[tuple[int, int]] = []
    if positive_runs:
        selected_positive_runs.append(positive_runs[0])
        last_start = positive_runs[0][0]
        for _release_start, release_end in releases:
            next_run = next(
                (
                    run
                    for run in positive_runs
                    if run[0] >= release_end and run[0] > last_start
                ),
                None,
            )
            if next_run is not None:
                selected_positive_runs.append(next_run)
                last_start = next_run[0]

    entries = []
    for index, (start, end) in enumerate(selected_positive_runs):
        row = policy_rows[start]
        swing = float(qpos[start, 0])
        route = row.get("condition_route")
        entries.append(
            {
                "schema": "simverify_agx_observable_dig_entry_proxy_v1",
                "entry_index": int(index),
                "policy_tick": int(start),
                "positive_bucket_run_end_tick_exclusive": int(end),
                "swing_qpos": swing,
                "sector": classify_swing_sector(
                    swing,
                    boundaries=sector_boundaries,
                    review_margin=sector_review_margin,
                ),
                "cycle_index": int(row.get("cycle_index", 0)),
                "condition_route": (
                    None
                    if not isinstance(route, Mapping)
                    else str(route.get("route", ""))
                ),
            }
        )
    return {
        "schema": "simverify_agx_observable_cycle_entries_v1",
        "dig_entries": entries,
        "dump_releases": [
            {
                "start_tick": int(start),
                "end_tick": int(end),
            }
            for start, end in releases
        ],
        "positive_bucket_runs": [
            {"start_tick": int(start), "end_tick": int(end)}
            for start, end in positive_runs
        ],
    }


def build_closed_loop_paired_matrix(
    *,
    run_roots: Sequence[str | Path],
    m0_root: str | Path,
    output_root: str | Path,
    current_git: Mapping[str, Any],
) -> dict[str, Any]:
    """Build an immutable paired diagnostic without defining a new Gate."""

    if len(run_roots) < 2:
        raise ValueError("paired matrix requires at least two run roots")
    destination = Path(output_root).resolve(strict=False)
    if destination.exists():
        raise FileExistsError(f"immutable matrix output exists: {destination}")
    m0 = Path(m0_root).resolve(strict=True)
    m0_verification = verify_checksums(m0, m0 / "checksums.sha256")
    if not m0_verification["ok"]:
        raise ValueError("M0 checksum verification failed")
    thresholds_path = m0 / "annotation_thresholds_v2.json"
    thresholds = json.loads(thresholds_path.read_text(encoding="utf-8"))
    numeric = thresholds["numeric"]
    sector = thresholds["sector"]
    source_dt = float(numeric["source_dt_s"])
    minimum_source_steps = int(numeric["dump_release"]["minimum_release_steps"])
    minimum_policy_ticks = int(
        math.ceil(minimum_source_steps * source_dt * FROZEN_TARGET_HZ - 1.0e-12)
    )
    extraction_contract = {
        "schema": "simverify_agx_cycle_entry_extraction_contract_v1",
        "observable_inputs": [
            "actual_sent_action",
            "swing_qpos",
            "condition_route_diagnostic",
        ],
        "privilege_used": False,
        "action_deadzone": float(numeric["action_deadzone"]),
        "dump_swing_threshold": float(numeric["dump_release"]["swing_threshold"]),
        "minimum_source_steps": minimum_source_steps,
        "minimum_policy_ticks": minimum_policy_ticks,
        "policy_hz": FROZEN_TARGET_HZ,
        "sector_boundaries": list(map(float, sector["boundaries_low_to_high"])),
        "sector_review_margin": float(sector["boundary_review_margin"]),
        "dig_entry_rule": (
            "first_sustained_positive_bucket_run_then_first_sustained_"
            "positive_bucket_run_after_each_observable_dump_release"
        ),
        "new_numeric_threshold_fitted_after_runs": False,
    }

    records = [
        _load_run_record(
            root,
            extraction_contract=extraction_contract,
        )
        for root in run_roots
    ]
    records.sort(key=lambda item: (int(item["seed"]), str(item["baseline_id"])))
    _validate_pairing(records)
    paired = _paired_seed_rows(records)
    candidate = [record for record in records if record["baseline_id"] == "B1.4"]
    null = [record for record in records if record["baseline_id"] == "B2.4"]
    candidate_second = _exact_match_rate(candidate, entry_index=1)
    candidate_third = _exact_match_rate(candidate, entry_index=2)
    null_second = _exact_match_rate(null, entry_index=1)
    null_third = _exact_match_rate(null, entry_index=2)
    if candidate_second == 1.0 and candidate_third == 1.0:
        observation = "full_requested_three_sector_sequence_observed"
    elif candidate_second > null_second and candidate_second > 0.0:
        observation = "partial_condition_response_not_full_sequence"
    else:
        observation = "condition_sequence_response_not_established"

    result = {
        "schema": MATRIX_SCHEMA,
        "status": "completed_paired_diagnostic",
        "evidence_scope": "sim_closed_loop_diagnostic_non_promotable",
        "closed_loop_execution": True,
        "formal_gate_result": False,
        "task_success_claimed": False,
        "held_out_test_read": False,
        "real_control_candidate": False,
        "control_candidate": False,
        "decision_enum_changed": False,
        "observed_result": observation,
        "requested_sector_sequence": records[0]["requested_sector_sequence"],
        "run_count": len(records),
        "seed_count": len(paired),
        "metrics": {
            "candidate_b1_4": {
                "ready_reset_rate": _reset_rate(candidate),
                "second_entry_exact_sector_rate": candidate_second,
                "third_entry_exact_sector_rate": candidate_third,
                "full_three_entry_exact_sequence_rate": _full_sequence_rate(candidate),
                "reset_action_l2_discontinuity": [
                    record["reset_action_l2_discontinuity"]
                    for record in candidate
                    if record["reset_action_l2_discontinuity"] is not None
                ],
            },
            "shuffled_null_b2_4": {
                "ready_reset_rate": _reset_rate(null),
                "second_entry_exact_sector_rate": null_second,
                "third_entry_exact_sector_rate": null_third,
                "full_three_entry_exact_sequence_rate": _full_sequence_rate(null),
            },
        },
        "paired_seeds": paired,
        "runs": records,
        "interpretation_lock": {
            "can_state": [
                "live_action_to_next_observation_feedback_executed",
                "observable_ready_router_reset_behavior_measured",
                "requested_sector_sequence_response_measured",
            ],
            "cannot_state": [
                "closed_loop_task_success",
                "held_out_generalization",
                "real_machine_transfer",
                "deployment_readiness",
            ],
        },
    }
    temporary = destination.parent / f".{destination.name}.tmp-{os.getpid()}"
    if temporary.exists():
        raise FileExistsError(f"matrix temporary path exists: {temporary}")
    temporary.mkdir(parents=True)
    try:
        identities = [
            write_json(temporary / "paired_diagnostic.json", result),
        ]
        manifest = {
            "schema": "simverify_agx_closed_loop_matrix_manifest_v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "evidence_scope": result["evidence_scope"],
            "current_git": dict(current_git),
            "m0": {
                "root": str(m0),
                "checksums_sha256": sha256_file(m0 / "checksums.sha256"),
                "annotation_thresholds_v2": artifact_identity(thresholds_path),
                "checksum_verification": m0_verification,
            },
            "extraction_contract": extraction_contract,
            "input_runs": [
                {
                    "root": record["root"],
                    "baseline_id": record["baseline_id"],
                    "seed": record["seed"],
                    "run_manifest_sha256": record["input_identity"][
                        "run_manifest_sha256"
                    ],
                    "checksums_sha256": record["input_identity"]["checksums_sha256"],
                    "verified_file_count": record["input_identity"][
                        "verified_file_count"
                    ],
                }
                for record in records
            ],
        }
        identities.append(write_json(temporary / "matrix_manifest.json", manifest))
        write_checksums(
            temporary,
            identities,
            path=temporary / "checksums.sha256",
        )
        temporary.rename(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return result


def _load_run_record(
    root_value: str | Path,
    *,
    extraction_contract: Mapping[str, Any],
) -> dict[str, Any]:
    root = Path(root_value).resolve(strict=True)
    verification = verify_checksums(root, root / "checksums.sha256")
    if not verification["ok"]:
        raise ValueError(f"run checksum verification failed: {root}")
    manifest_path = root / "run_manifest.json"
    rows_path = root / "policy_ticks.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = [
        json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines()
    ]
    if (
        manifest.get("status") != "completed_bounded_diagnostic"
        or manifest.get("evidence_scope") != "sim_closed_loop_diagnostic_non_promotable"
        or manifest.get("closed_loop_execution") is not True
        or manifest.get("task_success_claimed") is not False
        or manifest.get("real_control_candidate") is not False
    ):
        raise ValueError(f"run evidence contract mismatch: {root}")
    baseline_id = str(manifest["bundle_contract"]["baseline_id"])
    if baseline_id not in EXPECTED_BASELINES:
        raise ValueError(f"unexpected paired baseline {baseline_id!r}")
    intervention = manifest["test_intent"]["intervention"]
    sequence = [
        str(intervention["current_sector"]),
        str(intervention["next_sector"]),
        str(intervention["second_next_sector"]),
    ]
    if any(label not in SECTOR_LABELS for label in sequence):
        raise ValueError(f"invalid requested sector sequence: {sequence}")
    extracted = extract_observable_cycle_entries(
        rows,
        action_deadzone=float(extraction_contract["action_deadzone"]),
        dump_swing_threshold=float(extraction_contract["dump_swing_threshold"]),
        minimum_policy_ticks=int(extraction_contract["minimum_policy_ticks"]),
        sector_boundaries=extraction_contract["sector_boundaries"],
        sector_review_margin=float(extraction_contract["sector_review_margin"]),
    )
    entries = extracted["dig_entries"]
    for index, entry in enumerate(entries):
        entry["requested_sector"] = sequence[index] if index < len(sequence) else None
        entry["exact_sector_match"] = bool(
            index < len(sequence) and entry["sector"] == sequence[index]
        )
    reset_tick = manifest["condition_lifecycle_contract"]["reset_policy_tick"]
    reset_jump = None
    if reset_tick is not None:
        tick = int(reset_tick)
        if tick <= 0 or tick >= len(rows):
            raise ValueError(f"invalid reset tick in {root}")
        before = np.asarray(rows[tick - 1]["actual_sent_action"], dtype=np.float64)
        after = np.asarray(rows[tick]["actual_sent_action"], dtype=np.float64)
        reset_jump = float(np.linalg.norm(after - before))
    return {
        "schema": "simverify_agx_closed_loop_run_diagnostic_v1",
        "root": str(root),
        "baseline_id": baseline_id,
        "seed": int(intervention["seed"]),
        "requested_sector_sequence": sequence,
        "policy_tick_count": int(manifest["policy_tick_count"]),
        "reset_count": int(manifest["condition_lifecycle_contract"]["reset_count"]),
        "reset_policy_tick": reset_tick,
        "reset_action_l2_discontinuity": reset_jump,
        "observable_cycle_entries": extracted,
        "provenance": manifest["provenance"],
        "input_identity": {
            "run_manifest_sha256": sha256_file(manifest_path),
            "policy_ticks_sha256": sha256_file(rows_path),
            "checksums_sha256": sha256_file(root / "checksums.sha256"),
            "verified_file_count": int(verification["verified_file_count"]),
        },
    }


def _validate_pairing(records: Sequence[Mapping[str, Any]]) -> None:
    grouped: dict[int, set[str]] = {}
    sequences = {
        tuple(map(str, record["requested_sector_sequence"])) for record in records
    }
    if len(sequences) != 1:
        raise ValueError("paired runs must use one requested sector sequence")
    run_commits = {
        str(record["provenance"]["current_repo"]["commit"]) for record in records
    }
    if len(run_commits) != 1:
        raise ValueError("paired runs must use one Real Stack commit")
    for record in records:
        seed = int(record["seed"])
        grouped.setdefault(seed, set()).add(str(record["baseline_id"]))
    expected = set(EXPECTED_BASELINES)
    if not grouped or any(baselines != expected for baselines in grouped.values()):
        raise ValueError("each seed requires exactly one B1.4 and one B2.4 run")
    if len(records) != 2 * len(grouped):
        raise ValueError("duplicate baseline run detected for a paired seed")


def _paired_seed_rows(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_seed: dict[int, dict[str, Mapping[str, Any]]] = {}
    for record in records:
        by_seed.setdefault(int(record["seed"]), {})[str(record["baseline_id"])] = record
    output = []
    for seed, pair in sorted(by_seed.items()):
        output.append(
            {
                "seed": seed,
                "candidate_b1_4": _compact_run(pair["B1.4"]),
                "shuffled_null_b2_4": _compact_run(pair["B2.4"]),
            }
        )
    return output


def _compact_run(record: Mapping[str, Any]) -> dict[str, Any]:
    entries = record["observable_cycle_entries"]["dig_entries"]
    return {
        "reset_count": int(record["reset_count"]),
        "reset_policy_tick": record["reset_policy_tick"],
        "dig_entry_ticks": [int(entry["policy_tick"]) for entry in entries],
        "dig_entry_swing_qpos": [float(entry["swing_qpos"]) for entry in entries],
        "observed_sectors": [str(entry["sector"]) for entry in entries],
        "exact_sector_matches": [
            bool(entry["exact_sector_match"]) for entry in entries[:3]
        ],
    }


def _exact_match_rate(
    records: Sequence[Mapping[str, Any]],
    *,
    entry_index: int,
) -> float:
    values = []
    for record in records:
        entries = record["observable_cycle_entries"]["dig_entries"]
        values.append(
            bool(
                len(entries) > entry_index
                and entries[entry_index]["exact_sector_match"]
            )
        )
    return float(np.mean(values))


def _full_sequence_rate(records: Sequence[Mapping[str, Any]]) -> float:
    values = []
    for record in records:
        entries = record["observable_cycle_entries"]["dig_entries"]
        values.append(
            bool(
                len(entries) >= 3
                and all(entry["exact_sector_match"] for entry in entries[:3])
            )
        )
    return float(np.mean(values))


def _reset_rate(records: Sequence[Mapping[str, Any]]) -> float:
    return float(np.mean([record["reset_count"] == 1 for record in records]))


def _runs(mask: np.ndarray, *, minimum: int) -> list[tuple[int, int]]:
    values = np.asarray(mask, dtype=bool).reshape(-1)
    padded = np.concatenate(([False], values, [False]))
    edges = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return [
        (int(start), int(end))
        for start, end in zip(starts, ends)
        if int(end - start) >= int(minimum)
    ]


def _merge_release_runs(
    runs: Sequence[tuple[int, int]],
    *,
    swing: np.ndarray,
    dump_swing_threshold: float,
) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for start, end in runs:
        if merged and np.min(swing[merged[-1][1] : start + 1]) > dump_swing_threshold:
            merged[-1][1] = int(end)
        else:
            merged.append([int(start), int(end)])
    return [(start, end) for start, end in merged]
