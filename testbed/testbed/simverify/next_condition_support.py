"""Train-derived support Gate for the B1.4 next-sector-only experiment."""

from __future__ import annotations

import itertools
import json
import math
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from testbed.simverify.artifacts import (
    verify_checksums,
    write_checksums,
    write_json,
    write_jsonl,
)
from testbed.simverify.contracts import git_provenance, sha256_file

EVIDENCE_SCOPE = "recorded-observation/offline"
SECTORS = ("left", "center", "right")
HELD_OUT_EPISODES = {1, 13, 25, 33}


def build_next_condition_support(
    *,
    repo_root: str | Path,
    output_root: str | Path,
    m0_root: str | Path = (
        "/data/pingfan/Excavator_real_stack_data/sim_observable_cycle_v3"
    ),
    m2_root: str | Path = (
        "/data/pingfan/Excavator_real_stack_data/"
        "sim_observable_cycle_v3_m2_contract_v1"
    ),
) -> dict[str, Any]:
    """Freeze the support inventory that must precede B1.4 training."""

    repository = Path(repo_root).resolve(strict=True)
    git = git_provenance(repository)
    if (
        git.get("branch") != "v2.0.0-simVerify"
        or bool(git.get("dirty"))
        or not git.get("git_available")
    ):
        raise ValueError("next-condition support Gate requires a clean worktree")
    destination = Path(output_root).resolve(strict=False)
    if destination.exists():
        raise FileExistsError(
            f"immutable next-condition support output exists: {destination}"
        )
    m0 = Path(m0_root).resolve(strict=True)
    m2 = Path(m2_root).resolve(strict=True)
    m0_verification = verify_checksums(m0, m0 / "checksums.sha256")
    m2_verification = verify_checksums(m2, m2 / "checksums.sha256")
    if not m0_verification["ok"] or not m2_verification["ok"]:
        raise ValueError("M0/M2 checksum verification failed")

    annotations = _read_jsonl(m0 / "cycle_annotations.jsonl")
    centers = _train_sector_centers(annotations)
    anchors = _read_jsonl(m2 / "condition_counterfactual_anchors_v1.jsonl")
    contract = derive_next_condition_support(anchors, sector_centers=centers)
    gate_passed = bool(contract["gate"]["passed"])
    decision = (
        "pass_next_condition_semantic_support_prerequisite"
        if gate_passed
        else "insufficient_next_condition_semantic_support"
    )

    temporary = destination.parent / f".{destination.name}.tmp-{os.getpid()}"
    temporary.mkdir(parents=True)
    identities: list[dict[str, Any]] = []
    try:
        identities.append(
            write_json(
                temporary / "next_condition_support_thresholds_v1.json",
                contract["thresholds"],
            )
        )
        identities.append(
            write_jsonl(
                temporary / "next_condition_source_episode_support.jsonl",
                contract["source_episode_rows"],
            )
        )
        identities.append(
            write_json(
                temporary / "next_condition_permutation_support_v1.json",
                contract["permutation_support"],
            )
        )
        gate_identity = write_json(
            temporary / "next_condition_support_gate_v1.json",
            {
                **contract["gate"],
                "decision": decision,
                "authorizes_b1_4_training": gate_passed,
                "evidence_scope": EVIDENCE_SCOPE,
                "held_out_test_read": False,
                "closed_loop_execution": False,
            },
        )
        identities.append(gate_identity)
        manifest_identity = write_json(
            temporary / "next_condition_support_manifest.json",
            {
                "schema": "simverify_next_condition_support_manifest_v1",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "git": git,
                "decision": decision,
                "authorizes_b1_4_training": gate_passed,
                "evidence_scope": EVIDENCE_SCOPE,
                "held_out_test_read": False,
                "closed_loop_execution": False,
                "m0_root": str(m0),
                "m0_manifest_sha256": sha256_file(m0 / "dataset_manifest.json"),
                "m0_checksums_sha256": sha256_file(m0 / "checksums.sha256"),
                "m2_root": str(m2),
                "m2_manifest_sha256": sha256_file(m2 / "m2_manifest.json"),
                "m2_checksums_sha256": sha256_file(m2 / "checksums.sha256"),
                "anchor_registry_sha256": sha256_file(
                    m2 / "condition_counterfactual_anchors_v1.jsonl"
                ),
                "cycle_annotations_sha256": sha256_file(
                    m0 / "cycle_annotations.jsonl"
                ),
                "sector_centers_fit_split": "train",
                "sector_swing_qpos_median": centers,
                "eligible_validation_source_episode_ids": contract["gate"][
                    "eligible_validation_source_episode_ids"
                ],
                "excluded_validation_source_episode_ids": contract["gate"][
                    "excluded_validation_source_episode_ids"
                ],
            },
        )
        identities.append(manifest_identity)
        checksums_identity = write_checksums(
            temporary,
            identities,
            path=temporary / "checksums.sha256",
        )
        os.rename(temporary, destination)
        return {
            "status": "completed",
            "output_root": str(destination),
            "decision": decision,
            "authorizes_b1_4_training": gate_passed,
            "manifest_sha256": manifest_identity["sha256"],
            "gate_sha256": gate_identity["sha256"],
            "checksums_sha256": checksums_identity["sha256"],
            "held_out_test_read": False,
            "closed_loop_execution": False,
        }
    except BaseException as exc:
        if temporary.exists():
            write_json(
                temporary / "BUILD_FAILED.json",
                {
                    "schema": "simverify_next_condition_support_failure_v1",
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                    "held_out_test_read": False,
                    "closed_loop_execution": False,
                    "git": git,
                },
            )
        raise


def derive_next_condition_support(
    anchors: Sequence[Mapping[str, Any]],
    *,
    sector_centers: Mapping[str, float],
) -> dict[str, Any]:
    """Derive integer support thresholds from train and audit validation once."""

    if set(sector_centers) != set(SECTORS):
        raise ValueError("sector centers must contain left, center, and right")
    if len(set(map(float, sector_centers.values()))) != len(SECTORS):
        raise ValueError("sector centers must be distinct")
    selected = [
        row
        for row in anchors
        if list(row.get("changed_factors", [])) == ["next_sector"]
        and bool(row.get("supported"))
    ]
    if any(int(row["episode_id"]) in HELD_OUT_EPISODES for row in selected):
        raise ValueError("held-out episode entered next-condition support audit")
    by_split_episode: dict[str, dict[int, list[Mapping[str, Any]]]] = {
        "train": defaultdict(list),
        "validation": defaultdict(list),
    }
    for row in selected:
        split = str(row["split"])
        if split not in by_split_episode:
            raise ValueError(f"unexpected support split: {split}")
        by_split_episode[split][int(row["episode_id"])].append(row)
    if not by_split_episode["train"] or not by_split_episode["validation"]:
        raise ValueError("next-condition support requires train and validation")

    train_counts = np.asarray(
        [len(rows) for rows in by_split_episode["train"].values()],
        dtype=np.float64,
    )
    train_q02_5 = float(np.quantile(train_counts, 0.025))
    minimum_anchors = int(math.ceil(train_q02_5))
    minimum_source_episodes = 2
    eligible: dict[str, set[int]] = {}
    for split, grouped in by_split_episode.items():
        eligible[split] = {
            episode_id
            for episode_id, rows in grouped.items()
            if len(rows) >= minimum_anchors
        }

    permutations = _semantic_permutations()
    permutation_rows: list[dict[str, Any]] = []
    for mapping in permutations:
        name = _permutation_name(mapping)
        split_payload: dict[str, Any] = {}
        for split, grouped in by_split_episode.items():
            episode_payload = []
            for episode_id in sorted(eligible[split]):
                informative = [
                    row
                    for row in grouped[episode_id]
                    if _permutation_changes_expected_direction(
                        row,
                        mapping=mapping,
                        sector_centers=sector_centers,
                    )
                ]
                if informative:
                    episode_payload.append(
                        {
                            "episode_id": episode_id,
                            "informative_anchor_count": len(informative),
                            "anchor_keys": [
                                {
                                    "episode_id": int(row["episode_id"]),
                                    "cycle_id": int(row["cycle_id"]),
                                    "base_sector": str(
                                        row["base_condition"]["next_sector"]
                                    ),
                                    "target_sector": str(
                                        row["target_condition"]["next_sector"]
                                    ),
                                }
                                for row in informative
                            ],
                        }
                    )
            split_payload[split] = {
                "informative_source_episode_count": len(episode_payload),
                "source_episodes": episode_payload,
            }
        permutation_rows.append(
            {
                "permutation": name,
                "mapping": dict(mapping),
                "train": split_payload["train"],
                "validation": split_payload["validation"],
                "validation_support_passed": (
                    split_payload["validation"][
                        "informative_source_episode_count"
                    ]
                    >= minimum_source_episodes
                ),
            }
        )

    source_episode_rows = []
    for split, grouped in by_split_episode.items():
        for episode_id, rows in sorted(grouped.items()):
            source_episode_rows.append(
                {
                    "schema": (
                        "simverify_next_condition_source_episode_support_v1"
                    ),
                    "split": split,
                    "episode_id": episode_id,
                    "supported_anchor_count": len(rows),
                    "eligible": episode_id in eligible[split],
                    "minimum_supported_anchor_count": minimum_anchors,
                    "pair_counts": _pair_counts(rows),
                }
            )
    validation_passed = (
        len(eligible["validation"]) >= minimum_source_episodes
        and all(row["validation_support_passed"] for row in permutation_rows)
    )
    thresholds = {
        "schema": "simverify_next_condition_support_thresholds_v1",
        "fit_split": "train",
        "validation_used_for_threshold_generation": False,
        "supported_anchor_count_train_source_episode_q02_5": train_q02_5,
        "minimum_supported_anchors_per_source_episode": minimum_anchors,
        "minimum_informative_source_episodes_per_semantic_permutation": (
            minimum_source_episodes
        ),
        "source_episode_minimum_reason": (
            "two independent source episodes are the minimum for "
            "between-source bootstrap variation"
        ),
        "held_out_test_read": False,
    }
    return {
        "thresholds": thresholds,
        "source_episode_rows": source_episode_rows,
        "permutation_support": {
            "schema": "simverify_next_condition_permutation_support_v1",
            "sector_centers": {
                key: float(value) for key, value in sector_centers.items()
            },
            "informative_rule": (
                "identity expected swing direction differs from the "
                "permuted expected swing direction"
            ),
            "permutations": permutation_rows,
        },
        "gate": {
            "schema": "simverify_next_condition_support_gate_v1",
            "passed": validation_passed,
            "criteria": {
                "eligible_validation_source_episode_count": {
                    "value": len(eligible["validation"]),
                    "minimum": minimum_source_episodes,
                    "passed": (
                        len(eligible["validation"]) >= minimum_source_episodes
                    ),
                },
                "all_five_permutations_have_informative_validation_support": {
                    "value": sum(
                        row["validation_support_passed"]
                        for row in permutation_rows
                    ),
                    "required": len(permutation_rows),
                    "passed": all(
                        row["validation_support_passed"]
                        for row in permutation_rows
                    ),
                },
            },
            "eligible_train_source_episode_ids": sorted(eligible["train"]),
            "eligible_validation_source_episode_ids": sorted(
                eligible["validation"]
            ),
            "excluded_validation_source_episode_ids": sorted(
                set(by_split_episode["validation"]) - eligible["validation"]
            ),
            "held_out_test_read": False,
            "closed_loop_execution": False,
        },
    }


def _train_sector_centers(
    annotations: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    values: dict[str, list[float]] = {sector: [] for sector in SECTORS}
    for row in annotations:
        if (
            row.get("split") != "train"
            or row.get("quality", {}).get("status") != "accepted"
        ):
            continue
        evidence = row["numeric_sector_evidence"]
        condition = row["policy_condition"]
        current = evidence.get("current_swing_qpos")
        next_value = evidence.get("next_swing_qpos")
        if current is not None:
            values[str(condition["current_sector"])].append(float(current))
        if next_value is not None:
            values[str(condition["next_ready_sector"])].append(float(next_value))
    if any(not values[sector] for sector in SECTORS):
        raise ValueError("train annotations lack a sector center")
    return {
        sector: float(np.median(values[sector]))
        for sector in SECTORS
    }


def _permutation_changes_expected_direction(
    row: Mapping[str, Any],
    *,
    mapping: Mapping[str, str],
    sector_centers: Mapping[str, float],
) -> bool:
    base = str(row["base_condition"]["next_sector"])
    target = str(row["target_condition"]["next_sector"])
    identity_sign = int(
        np.sign(float(sector_centers[target]) - float(sector_centers[base]))
    )
    permutation_sign = int(
        np.sign(
            float(sector_centers[mapping[target]])
            - float(sector_centers[mapping[base]])
        )
    )
    return identity_sign != permutation_sign


def _semantic_permutations() -> list[dict[str, str]]:
    identity = tuple(SECTORS)
    return [
        dict(zip(SECTORS, values, strict=True))
        for values in itertools.permutations(SECTORS)
        if values != identity
    ]


def _permutation_name(mapping: Mapping[str, str]) -> str:
    return "__".join(f"{sector}_to_{mapping[sector]}" for sector in SECTORS)


def _pair_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = (
            f"{row['base_condition']['next_sector']}->"
            f"{row['target_condition']['next_sector']}"
        )
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
