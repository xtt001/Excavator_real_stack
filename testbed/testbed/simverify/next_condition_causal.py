"""Support-aware fixed-observation causal Gate for a frozen next-sector pair."""

from __future__ import annotations

import json
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
from testbed.simverify.m3_condition_causal_v2 import (
    _mean,
    _metric_mean,
    _package_identity,
    _permutation_name,
    _q,
    _require_mask_is_null,
    _require_matched,
    _semantic_permutations,
    _validated_package,
    phase_specificity,
    signed_semantic_margin,
)
from testbed.simverify.m3_condition_gate import paired_metric_result

EVIDENCE_SCOPE = "recorded-observation/offline"
SUPPORTED_BASELINE_PAIRS = frozenset(
    {
        ("B1.4", "B2.4"),
        ("B1.5", "B2.5"),
    }
)


def build_next_condition_causal_gate(
    *,
    repo_root: str | Path,
    output_root: str | Path,
    b1_replay_roots: Sequence[str | Path],
    b2_replay_root: str | Path,
    masked_b1_replay_root: str | Path,
    support_root: str | Path,
    candidate_baseline_id: str = "B1.4",
    null_baseline_id: str = "B2.4",
    bootstrap_repetitions: int = 100_000,
    bootstrap_seed: int = 20_260_726,
) -> dict[str, Any]:
    """Build the immutable next-sector semantic decision for a frozen pair."""

    if len(b1_replay_roots) < 3:
        raise ValueError("next-condition Gate requires at least three candidate repeats")
    _validate_baseline_pair(candidate_baseline_id, null_baseline_id)
    if bootstrap_repetitions < 10_000:
        raise ValueError("next-condition Gate requires at least 10000 draws")
    repository = Path(repo_root).resolve(strict=True)
    git = git_provenance(repository)
    if (
        git.get("branch") != "v2.0.0-simVerify"
        or bool(git.get("dirty"))
        or not git.get("git_available")
    ):
        raise ValueError("next-condition Gate requires a clean worktree")
    destination = Path(output_root).resolve(strict=False)
    if destination.exists():
        raise FileExistsError(
            f"immutable next-condition Gate exists: {destination}"
        )

    support = _validated_support(Path(support_root).resolve(strict=True))
    candidates = [
        _validated_package(
            Path(root).resolve(strict=True),
            candidate_baseline_id,
            "requested",
        )
        for root in b1_replay_roots
    ]
    shuffled = _validated_package(
        Path(b2_replay_root).resolve(strict=True),
        null_baseline_id,
        "requested",
    )
    masked = _validated_package(
        Path(masked_b1_replay_root).resolve(strict=True),
        candidate_baseline_id,
        "masked_canonical",
    )
    _require_matched([*candidates, shuffled, masked])
    repeat_ids = [int(package["manifest"]["repeat_id"]) for package in candidates]
    if len(set(repeat_ids)) != len(repeat_ids):
        raise ValueError("candidate repeat ids must be unique")
    reference_index = int(np.argmin(repeat_ids))
    reference = candidates[reference_index]
    repeats = [
        package
        for index, package in enumerate(candidates)
        if index != reference_index
    ]
    if (
        masked["manifest"]["checkpoint"]["sha256"]
        != reference["manifest"]["checkpoint"]["sha256"]
    ):
        raise ValueError("masked control must use the candidate reference checkpoint")
    _require_mask_is_null(masked)
    if reference["manifest"].get("intervention_factors") != ["next_sector"]:
        raise ValueError("candidate replay is not next-sector-only")
    if reference["manifest"].get("sector_direction_fit_splits") != ["train"]:
        raise ValueError("candidate direction semantics were not fit train-only")

    direction = _read_json(reference["root"] / "sector_action_direction_v1.json")
    source_rows, criteria, noise = _evaluate(
        reference=reference,
        repeats=repeats,
        shuffled=shuffled,
        masked=masked,
        support=support,
        sector_centers=direction["sector_swing_qpos_median"],
        action_direction_sign=int(direction["action_to_qpos_direction_sign"]),
        repetitions=bootstrap_repetitions,
        seed=bootstrap_seed,
        null_baseline_id=null_baseline_id,
    )
    passed = all(bool(value["passed"]) for value in criteria.values())
    decision = (
        "next_condition_understanding_established_offline"
        if passed
        else "next_condition_understanding_not_established"
    )
    gate = {
        "schema": "simverify_next_condition_causal_gate_v1",
        "decision": decision,
        "recommended_terminal_status": None if passed else "revise_condition",
        "next_condition_understanding_established": passed,
        "factor_pass": {"next_sector": passed},
        "criteria": criteria,
        "decision_rule": "next_sector must pass every frozen criterion",
        "support_manifest_sha256": support["manifest_sha256"],
        "eligible_validation_source_episode_ids": support[
            "eligible_validation_source_episode_ids"
        ],
        "excluded_validation_source_episode_ids": support[
            "excluded_validation_source_episode_ids"
        ],
        "evidence_scope": EVIDENCE_SCOPE,
        "held_out_test_read": False,
        "closed_loop_execution": False,
        "interpretation_guard": (
            "offline semantic response is not closed-loop task completion"
        ),
    }

    temporary = destination.parent / f".{destination.name}.tmp-{os.getpid()}"
    temporary.mkdir(parents=True)
    identities: list[dict[str, Any]] = []
    try:
        identities.append(
            write_jsonl(
                temporary / "next_condition_source_episode_metrics.jsonl",
                source_rows,
            )
        )
        identities.append(
            write_json(
                temporary / "next_condition_repeat_noise_v1.json",
                noise,
            )
        )
        gate_identity = write_json(
            temporary / "next_condition_causal_gate_v1.json",
            gate,
        )
        identities.append(gate_identity)
        manifest_identity = write_json(
            temporary / "next_condition_causal_manifest.json",
            {
                "schema": "simverify_next_condition_causal_manifest_v1",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "git": git,
                "decision": decision,
                "candidate_baseline_id": candidate_baseline_id,
                "null_baseline_id": null_baseline_id,
                "support_artifact": {
                    "path": str(support["root"]),
                    "manifest_sha256": support["manifest_sha256"],
                    "checksums_sha256": support["checksums_sha256"],
                },
                "bootstrap": {
                    "unit": "source_episode",
                    "repetitions": bootstrap_repetitions,
                    "seed": bootstrap_seed,
                },
                "condition_replay_packages": [
                    _package_identity(package)
                    for package in [*candidates, shuffled, masked]
                ],
                "evidence_scope": EVIDENCE_SCOPE,
                "held_out_test_read": False,
                "closed_loop_execution": False,
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
            "next_condition_understanding_established": passed,
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
                    "schema": "simverify_next_condition_causal_failure_v1",
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                    "held_out_test_read": False,
                    "closed_loop_execution": False,
                    "git": git,
                },
            )
        raise


def _evaluate(
    *,
    reference: Mapping[str, Any],
    repeats: Sequence[Mapping[str, Any]],
    shuffled: Mapping[str, Any],
    masked: Mapping[str, Any],
    support: Mapping[str, Any],
    sector_centers: Mapping[str, float],
    action_direction_sign: int,
    repetitions: int,
    seed: int,
    null_baseline_id: str = "B2.4",
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    eligible_episodes = set(support["eligible_validation_source_episode_ids"])
    reference_rows = reference["supported_rows"]
    grouped: dict[int, list[int]] = defaultdict(list)
    for anchor_id, row in reference_rows.items():
        episode_id = int(row["episode_id"])
        if episode_id in eligible_episodes:
            grouped[episode_id].append(anchor_id)
    if set(grouped) != eligible_episodes:
        raise ValueError("eligible source episode lacks replay anchors")

    identity = {"left": "left", "center": "center", "right": "right"}
    source_rows = []
    for episode_id, ids in sorted(grouped.items()):
        candidate_margin = _semantic_mean(
            reference_rows,
            ids,
            mapping=identity,
            centers=sector_centers,
            direction_sign=action_direction_sign,
        )
        source_rows.append(
            {
                "schema": "simverify_next_condition_source_episode_metrics_v1",
                "episode_id": episode_id,
                "supported_anchor_count": len(ids),
                "candidate_action_effect": _metric_mean(
                    reference_rows,
                    ids,
                    "token_swap_action_effect",
                ),
                "masked_action_effect": _metric_mean(
                    masked["supported_rows"],
                    ids,
                    "token_swap_action_effect",
                ),
                "candidate_signed_semantic_margin": candidate_margin,
                "shuffled_signed_semantic_margin": _semantic_mean(
                    shuffled["supported_rows"],
                    ids,
                    mapping=identity,
                    centers=sector_centers,
                    direction_sign=action_direction_sign,
                ),
                "candidate_phase_specificity": _mean(
                    [phase_specificity(reference_rows[index]) for index in ids]
                ),
                "masked_phase_specificity": _mean(
                    [
                        phase_specificity(masked["supported_rows"][index])
                        for index in ids
                    ]
                ),
                "shuffled_phase_specificity": _mean(
                    [
                        phase_specificity(shuffled["supported_rows"][index])
                        for index in ids
                    ]
                ),
                "repeat_action_effect_abs_delta": _repeat_metric_delta(
                    reference,
                    repeats,
                    ids,
                    metric="token_swap_action_effect",
                ),
                "repeat_signed_margin_abs_delta": max(
                    abs(
                        candidate_margin
                        - _semantic_mean(
                            package["supported_rows"],
                            ids,
                            mapping=identity,
                            centers=sector_centers,
                            direction_sign=action_direction_sign,
                        )
                    )
                    for package in repeats
                ),
                "repeat_phase_specificity_abs_delta": max(
                    abs(
                        _mean(
                            [
                                phase_specificity(reference_rows[index])
                                for index in ids
                            ]
                        )
                        - _mean(
                            [
                                phase_specificity(
                                    package["supported_rows"][index]
                                )
                                for index in ids
                            ]
                        )
                    )
                    for package in repeats
                ),
                "event_coverage_delta": _metric_mean(
                    reference_rows,
                    ids,
                    "event_coverage_delta",
                ),
                "event_order_violation_rate": _mean(
                    [
                        not bool(
                            reference_rows[index]["metrics"][
                                "target_event_order_valid"
                            ]
                        )
                        for index in ids
                    ]
                ),
                "evidence_scope": EVIDENCE_SCOPE,
            }
        )

    action_noise = _q(
        [row["repeat_action_effect_abs_delta"] for row in source_rows],
        0.975,
    )
    margin_noise = _q(
        [row["repeat_signed_margin_abs_delta"] for row in source_rows],
        0.975,
    )
    phase_noise = _q(
        [row["repeat_phase_specificity_abs_delta"] for row in source_rows],
        0.975,
    )
    action = paired_metric_result(
        _array(source_rows, "candidate_action_effect"),
        _array(source_rows, "masked_action_effect"),
        repeat_noise=action_noise,
        lower_is_better=False,
        repetitions=repetitions,
        seed=seed + 1,
    )
    margin = paired_metric_result(
        _array(source_rows, "candidate_signed_semantic_margin"),
        _array(source_rows, "shuffled_signed_semantic_margin"),
        repeat_noise=margin_noise,
        lower_is_better=False,
        repetitions=repetitions,
        seed=seed + 2,
    )
    phase_masked = paired_metric_result(
        _array(source_rows, "candidate_phase_specificity"),
        _array(source_rows, "masked_phase_specificity"),
        repeat_noise=phase_noise,
        lower_is_better=False,
        repetitions=repetitions,
        seed=seed + 3,
    )
    phase_shuffled = paired_metric_result(
        _array(source_rows, "candidate_phase_specificity"),
        _array(source_rows, "shuffled_phase_specificity"),
        repeat_noise=phase_noise,
        lower_is_better=False,
        repetitions=repetitions,
        seed=seed + 4,
    )

    permutation_results = {}
    permutations = {
        row["permutation"]: row
        for row in support["permutation_support"]["permutations"]
    }
    for permutation_index, mapping in enumerate(_semantic_permutations()):
        name = _permutation_name(mapping)
        support_row = permutations[name]
        episode_deltas = []
        for episode_support in support_row["validation"]["source_episodes"]:
            episode_id = int(episode_support["episode_id"])
            if episode_id not in eligible_episodes:
                continue
            keys = {
                (
                    int(row["episode_id"]),
                    int(row["cycle_id"]),
                    str(row["base_sector"]),
                    str(row["target_sector"]),
                )
                for row in episode_support["anchor_keys"]
            }
            ids = [
                anchor_id
                for anchor_id in grouped[episode_id]
                if _row_support_key(reference_rows[anchor_id]) in keys
            ]
            if not ids:
                raise ValueError(
                    f"permutation support/replay mismatch: {name}/{episode_id}"
                )
            identity_margin = _semantic_mean(
                reference_rows,
                ids,
                mapping=identity,
                centers=sector_centers,
                direction_sign=action_direction_sign,
            )
            wrong_margin = _semantic_mean(
                reference_rows,
                ids,
                mapping=mapping,
                centers=sector_centers,
                direction_sign=action_direction_sign,
            )
            episode_deltas.append(identity_margin - wrong_margin)
        if len(episode_deltas) < 2:
            raise ValueError(f"permutation lacks two informative episodes: {name}")
        values = np.asarray(episode_deltas, dtype=np.float64)
        permutation_results[name] = {
            **paired_metric_result(
                values,
                np.zeros_like(values),
                repeat_noise=margin_noise,
                lower_is_better=False,
                repetitions=repetitions,
                seed=seed + 10 + permutation_index,
            ),
            "informative_source_episode_count": len(episode_deltas),
            "identity_minus_permutation_source_episode_values": episode_deltas,
        }
    semantic_identifiability = {
        "permutation_results": permutation_results,
        "all_five_non_identity_permutations_rejected": all(
            result["passed"] for result in permutation_results.values()
        ),
    }
    semantic_identifiability["passed"] = semantic_identifiability[
        "all_five_non_identity_permutations_rejected"
    ]
    preservation = {
        "event_coverage_delta_source_episode_values": [
            row["event_coverage_delta"] for row in source_rows
        ],
        "event_order_violation_rate_source_episode_values": [
            row["event_order_violation_rate"] for row in source_rows
        ],
        "passed": bool(
            min(row["event_coverage_delta"] for row in source_rows) >= 0.0
            and max(
                row["event_order_violation_rate"] for row in source_rows
            )
            == 0.0
        ),
    }
    phase = {
        "positive_vs_masked": phase_masked,
        "greater_than_shuffled": phase_shuffled,
        "passed": bool(phase_masked["passed"] and phase_shuffled["passed"]),
    }
    criteria = {
        "action_sensitivity_vs_masked": action,
        (
            "signed_semantic_margin_vs_"
            + null_baseline_id.lower().replace(".", "_")
        ): margin,
        "semantic_identifiability": semantic_identifiability,
        "phase_specificity": phase,
        "task_envelope_preservation": preservation,
    }
    noise = {
        "schema": "simverify_next_condition_repeat_noise_v1",
        "reference_repeat_id": reference["manifest"]["repeat_id"],
        "comparison_repeat_ids": [
            package["manifest"]["repeat_id"] for package in repeats
        ],
        "action_effect_q97_5": action_noise,
        "signed_semantic_margin_q97_5": margin_noise,
        "phase_specificity_q97_5": phase_noise,
    }
    return source_rows, criteria, noise


def _validate_baseline_pair(candidate: str, null: str) -> None:
    if (str(candidate), str(null)) not in SUPPORTED_BASELINE_PAIRS:
        raise ValueError(
            "next-condition Gate baseline pair must be one of "
            f"{sorted(SUPPORTED_BASELINE_PAIRS)}"
        )


def _validated_support(root: Path) -> dict[str, Any]:
    verification = verify_checksums(root, root / "checksums.sha256")
    if not verification["ok"]:
        raise ValueError("next-condition support checksum verification failed")
    manifest = _read_json(root / "next_condition_support_manifest.json")
    gate = _read_json(root / "next_condition_support_gate_v1.json")
    permutation_support = _read_json(
        root / "next_condition_permutation_support_v1.json"
    )
    if (
        manifest.get("schema") != "simverify_next_condition_support_manifest_v1"
        or manifest.get("authorizes_b1_4_training") is not True
        or manifest.get("held_out_test_read") is not False
        or gate.get("decision")
        != "pass_next_condition_semantic_support_prerequisite"
        or gate.get("authorizes_b1_4_training") is not True
        or gate.get("held_out_test_read") is not False
    ):
        raise ValueError("support artifact does not authorize B1.4")
    return {
        "root": root,
        "manifest_sha256": sha256_file(
            root / "next_condition_support_manifest.json"
        ),
        "checksums_sha256": sha256_file(root / "checksums.sha256"),
        "eligible_validation_source_episode_ids": list(
            map(int, gate["eligible_validation_source_episode_ids"])
        ),
        "excluded_validation_source_episode_ids": list(
            map(int, gate["excluded_validation_source_episode_ids"])
        ),
        "permutation_support": permutation_support,
    }


def _semantic_mean(
    rows: Mapping[int, Mapping[str, Any]],
    ids: Sequence[int],
    *,
    mapping: Mapping[str, str],
    centers: Mapping[str, float],
    direction_sign: int,
) -> float:
    return _mean(
        [
            signed_semantic_margin(
                rows[index],
                semantic_mapping=mapping,
                sector_centers=centers,
                action_direction_sign=direction_sign,
            )
            for index in ids
        ]
    )


def _repeat_metric_delta(
    reference: Mapping[str, Any],
    repeats: Sequence[Mapping[str, Any]],
    ids: Sequence[int],
    *,
    metric: str,
) -> float:
    reference_value = _metric_mean(reference["supported_rows"], ids, metric)
    return max(
        abs(
            reference_value
            - _metric_mean(package["supported_rows"], ids, metric)
        )
        for package in repeats
    )


def _row_support_key(row: Mapping[str, Any]) -> tuple[int, int, str, str]:
    return (
        int(row["episode_id"]),
        int(row["cycle_id"]),
        str(row["base_condition"]["next_sector"]),
        str(row["target_condition"]["next_sector"]),
    )


def _array(rows: Sequence[Mapping[str, Any]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=np.float64)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
