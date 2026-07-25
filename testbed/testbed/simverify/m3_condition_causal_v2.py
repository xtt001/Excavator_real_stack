"""Fixed-observation causal test for cycle_condition_v1 semantics."""

from __future__ import annotations

import itertools
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
from testbed.simverify.m3_condition_gate import paired_metric_result

FACTORS = ("current_sector", "next_sector")
SECTORS = ("left", "center", "right")
HELD_OUT_EPISODES = {1, 13, 25, 33}


def build_condition_causal_v2(
    *,
    repo_root: str | Path,
    output_root: str | Path,
    b1_replay_roots: Sequence[str | Path],
    b2_replay_root: str | Path,
    masked_b1_replay_root: str | Path,
    bootstrap_repetitions: int = 100_000,
    bootstrap_seed: int = 20_260_725,
) -> dict[str, Any]:
    """Build an immutable, held-out-free condition-understanding decision."""

    if len(b1_replay_roots) < 3:
        raise ValueError("condition causal v2 requires at least three B1 repeats")
    if bootstrap_repetitions < 10_000:
        raise ValueError("condition causal v2 requires at least 10000 bootstrap draws")
    repository = Path(repo_root).resolve(strict=True)
    git = git_provenance(repository)
    if (
        git.get("branch") != "v2.0.0-simVerify"
        or bool(git.get("dirty"))
        or not git.get("git_available")
    ):
        raise ValueError("condition causal v2 requires a clean SimVerify worktree")
    destination = Path(output_root).resolve(strict=False)
    if destination.exists():
        raise FileExistsError(
            f"immutable condition causal output exists: {destination}"
        )

    b1_packages = [
        _validated_package(Path(root).resolve(strict=True), "B1", "requested")
        for root in b1_replay_roots
    ]
    b2 = _validated_package(
        Path(b2_replay_root).resolve(strict=True), "B2", "requested"
    )
    masked = _validated_package(
        Path(masked_b1_replay_root).resolve(strict=True),
        "B1",
        "masked_canonical",
    )
    _require_matched([*b1_packages, b2, masked])
    repeat_ids = [int(package["manifest"]["repeat_id"]) for package in b1_packages]
    if len(set(repeat_ids)) != len(repeat_ids):
        raise ValueError("B1 repeat ids must be unique")
    reference_index = int(np.argmin(repeat_ids))
    reference = b1_packages[reference_index]
    repeats = [
        package for index, package in enumerate(b1_packages) if index != reference_index
    ]
    if (
        masked["manifest"]["checkpoint"]["sha256"]
        != reference["manifest"]["checkpoint"]["sha256"]
    ):
        raise ValueError("masked control must use the reference B1 checkpoint")
    _require_mask_is_null(masked)

    direction = _read_json(reference["root"] / "sector_action_direction_v1.json")
    sector_centers = direction["sector_swing_qpos_median"]
    action_direction_sign = int(direction["action_to_qpos_direction_sign"])
    permutations = _semantic_permutations()
    source_rows, criteria, noise = _evaluate(
        reference=reference,
        repeats=repeats,
        b2=b2,
        masked=masked,
        permutations=permutations,
        sector_centers=sector_centers,
        action_direction_sign=action_direction_sign,
        repetitions=bootstrap_repetitions,
        seed=bootstrap_seed,
    )
    factor_pass = {
        factor: all(bool(value["passed"]) for value in criteria[factor].values())
        for factor in FACTORS
    }
    passed = all(factor_pass.values())
    decision = (
        "condition_understanding_established_offline"
        if passed
        else "condition_understanding_not_established"
    )
    gate = {
        "schema": "simverify_condition_causal_gate_v2",
        "decision": decision,
        "recommended_terminal_status": None if passed else "revise_condition",
        "evidence_scope": "recorded-observation/offline",
        "closed_loop_execution": False,
        "held_out_test_read": False,
        "condition_understanding_established": passed,
        "factor_pass": factor_pass,
        "criteria": criteria,
        "decision_rule": (
            "both current_sector and next_sector must pass every frozen criterion"
        ),
        "interpretation_guard": (
            "action sensitivity alone does not establish semantic or phase understanding"
        ),
    }

    temporary = destination.parent / f".{destination.name}.tmp-{os.getpid()}"
    temporary.mkdir(parents=True)
    identities: list[dict[str, Any]] = []
    try:
        identities.append(
            write_jsonl(temporary / "source_episode_causal_metrics.jsonl", source_rows)
        )
        identities.append(write_json(temporary / "repeat_noise_v2.json", noise))
        identities.append(
            write_json(
                temporary / "semantic_permutations_v1.json",
                {
                    "schema": "simverify_semantic_permutations_v1",
                    "sectors": list(SECTORS),
                    "identity": dict(zip(SECTORS, SECTORS, strict=True)),
                    "non_identity": permutations,
                    "count": len(permutations),
                },
            )
        )
        identities.append(write_json(temporary / "condition_causal_gate_v2.json", gate))
        manifest_identity = write_json(
            temporary / "condition_causal_manifest.json",
            {
                "schema": "simverify_condition_causal_manifest_v2",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "git": git,
                "decision": decision,
                "evidence_scope": "recorded-observation/offline",
                "closed_loop_execution": False,
                "held_out_test_read": False,
                "bootstrap": {
                    "unit": "source_episode",
                    "repetitions": bootstrap_repetitions,
                    "seed": bootstrap_seed,
                },
                "condition_replay_packages": [
                    _package_identity(package) for package in [*b1_packages, b2, masked]
                ],
            },
        )
        identities.append(manifest_identity)
        checksums_identity = write_checksums(
            temporary, identities, path=temporary / "checksums.sha256"
        )
        os.rename(temporary, destination)
        return {
            "status": "completed",
            "output_root": str(destination),
            "decision": decision,
            "condition_understanding_established": passed,
            "factor_pass": factor_pass,
            "manifest_sha256": manifest_identity["sha256"],
            "checksums_sha256": checksums_identity["sha256"],
            "held_out_test_read": False,
            "closed_loop_execution": False,
        }
    except BaseException as exc:
        if temporary.exists():
            write_json(
                temporary / "BUILD_FAILED.json",
                {
                    "schema": "simverify_condition_causal_failure_v2",
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                    "held_out_test_read": False,
                    "closed_loop_execution": False,
                    "git": git,
                },
            )
        raise


def signed_semantic_margin(
    row: Mapping[str, Any],
    *,
    semantic_mapping: Mapping[str, str],
    sector_centers: Mapping[str, float],
    action_direction_sign: int,
) -> float:
    """Return signed swing response under one label-to-meaning mapping."""

    factor = str(row["changed_factor"])
    key = "current_sector" if factor == "current_sector" else "next_sector"
    base = semantic_mapping[str(row["base_condition"][key])]
    target = semantic_mapping[str(row["target_condition"][key])]
    expected = int(
        np.sign(float(sector_centers[target]) - float(sector_centers[base]))
    ) * int(action_direction_sign)
    return float(expected * float(row["metrics"]["swing_action_delta_mean"]))


def phase_specificity(row: Mapping[str, Any]) -> float:
    """Return intended-window effect minus off-window effect."""

    values = np.asarray(row["metrics"]["per_tick_effect_l1"], dtype=np.float64)
    start, end = map(int, row["metrics"]["relevant_window_local"])
    if not 0 <= start < end <= values.size:
        raise ValueError("invalid relevant condition window")
    intended = values[start:end]
    off = np.concatenate((values[:start], values[end:]))
    if off.size == 0:
        raise ValueError("phase specificity requires an off-window")
    return float(np.mean(intended) - np.mean(off))


def _evaluate(
    *,
    reference: Mapping[str, Any],
    repeats: Sequence[Mapping[str, Any]],
    b2: Mapping[str, Any],
    masked: Mapping[str, Any],
    permutations: Sequence[Mapping[str, str]],
    sector_centers: Mapping[str, float],
    action_direction_sign: int,
    repetitions: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    identity = dict(zip(SECTORS, SECTORS, strict=True))
    source_rows: list[dict[str, Any]] = []
    criteria: dict[str, Any] = {}
    noise_output: dict[str, Any] = {
        "schema": "simverify_condition_causal_repeat_noise_v2",
        "reference_repeat_id": reference["manifest"]["repeat_id"],
        "comparison_repeat_ids": [
            package["manifest"]["repeat_id"] for package in repeats
        ],
        "factors": {},
    }
    for factor_index, factor in enumerate(FACTORS):
        anchor_ids = sorted(
            anchor_id
            for anchor_id, row in reference["supported_rows"].items()
            if row["changed_factor"] == factor
        )
        grouped: dict[int, list[int]] = defaultdict(list)
        for anchor_id in anchor_ids:
            grouped[int(reference["supported_rows"][anchor_id]["episode_id"])].append(
                anchor_id
            )
        if len(grouped) < 2:
            raise ValueError(f"{factor} lacks two supported source episodes")

        episode_rows: list[dict[str, Any]] = []
        for episode_id, ids in sorted(grouped.items()):
            ref_rows = reference["supported_rows"]
            b2_rows = b2["supported_rows"]
            masked_rows = masked["supported_rows"]
            reference_margin = _mean(
                [
                    signed_semantic_margin(
                        ref_rows[index],
                        semantic_mapping=identity,
                        sector_centers=sector_centers,
                        action_direction_sign=action_direction_sign,
                    )
                    for index in ids
                ]
            )
            permutation_margins = {
                _permutation_name(mapping): _mean(
                    [
                        signed_semantic_margin(
                            ref_rows[index],
                            semantic_mapping=mapping,
                            sector_centers=sector_centers,
                            action_direction_sign=action_direction_sign,
                        )
                        for index in ids
                    ]
                )
                for mapping in permutations
            }
            repeat_margin_delta = max(
                abs(
                    reference_margin
                    - _mean(
                        [
                            signed_semantic_margin(
                                package["supported_rows"][index],
                                semantic_mapping=identity,
                                sector_centers=sector_centers,
                                action_direction_sign=action_direction_sign,
                            )
                            for index in ids
                        ]
                    )
                )
                for package in repeats
            )
            reference_phase = _mean(
                [phase_specificity(ref_rows[index]) for index in ids]
            )
            repeat_phase_delta = max(
                abs(
                    reference_phase
                    - _mean(
                        [
                            phase_specificity(package["supported_rows"][index])
                            for index in ids
                        ]
                    )
                )
                for package in repeats
            )
            row = {
                "schema": "simverify_condition_source_episode_causal_metrics_v2",
                "factor": factor,
                "episode_id": episode_id,
                "supported_anchor_count": len(ids),
                "b1_action_effect": _metric_mean(
                    ref_rows, ids, "token_swap_action_effect"
                ),
                "masked_action_effect": _metric_mean(
                    masked_rows, ids, "token_swap_action_effect"
                ),
                "b1_signed_semantic_margin": reference_margin,
                "b2_signed_semantic_margin": _mean(
                    [
                        signed_semantic_margin(
                            b2_rows[index],
                            semantic_mapping=identity,
                            sector_centers=sector_centers,
                            action_direction_sign=action_direction_sign,
                        )
                        for index in ids
                    ]
                ),
                "b1_identity_minus_permutation": {
                    name: reference_margin - value
                    for name, value in permutation_margins.items()
                },
                "b1_phase_specificity": reference_phase,
                "b2_phase_specificity": _mean(
                    [phase_specificity(b2_rows[index]) for index in ids]
                ),
                "masked_phase_specificity": _mean(
                    [phase_specificity(masked_rows[index]) for index in ids]
                ),
                "repeat_signed_margin_abs_delta": repeat_margin_delta,
                "repeat_phase_specificity_abs_delta": repeat_phase_delta,
                "event_coverage_delta": _metric_mean(
                    ref_rows, ids, "event_coverage_delta"
                ),
                "event_order_violation_rate": _mean(
                    [
                        not bool(ref_rows[index]["metrics"]["target_event_order_valid"])
                        for index in ids
                    ]
                ),
                "evidence_scope": "recorded-observation/offline",
            }
            episode_rows.append(row)
            source_rows.append(row)

        margin_noise = _q(
            [row["repeat_signed_margin_abs_delta"] for row in episode_rows], 0.975
        )
        phase_noise = _q(
            [row["repeat_phase_specificity_abs_delta"] for row in episode_rows], 0.975
        )
        action_noise = _repeat_action_noise(reference, repeats, grouped=grouped)
        noise_output["factors"][factor] = {
            "action_effect_q97_5": action_noise,
            "signed_semantic_margin_q97_5": margin_noise,
            "phase_specificity_q97_5": phase_noise,
        }
        base_seed = seed + factor_index * 100
        action_vs_masked = paired_metric_result(
            _array(episode_rows, "b1_action_effect"),
            _array(episode_rows, "masked_action_effect"),
            repeat_noise=action_noise,
            lower_is_better=False,
            repetitions=repetitions,
            seed=base_seed + 1,
        )
        margin_vs_b2 = paired_metric_result(
            _array(episode_rows, "b1_signed_semantic_margin"),
            _array(episode_rows, "b2_signed_semantic_margin"),
            repeat_noise=margin_noise,
            lower_is_better=False,
            repetitions=repetitions,
            seed=base_seed + 2,
        )
        permutation_results = {}
        for permutation_index, mapping in enumerate(permutations):
            name = _permutation_name(mapping)
            values = np.asarray(
                [row["b1_identity_minus_permutation"][name] for row in episode_rows],
                dtype=np.float64,
            )
            permutation_results[name] = paired_metric_result(
                values,
                np.zeros_like(values),
                repeat_noise=margin_noise,
                lower_is_better=False,
                repetitions=repetitions,
                seed=base_seed + 10 + permutation_index,
            )
        semantic_identifiability = {
            "permutation_results": permutation_results,
            "all_five_non_identity_permutations_rejected": all(
                result["passed"] for result in permutation_results.values()
            ),
        }
        semantic_identifiability["passed"] = semantic_identifiability[
            "all_five_non_identity_permutations_rejected"
        ]
        phase_positive = paired_metric_result(
            _array(episode_rows, "b1_phase_specificity"),
            _array(episode_rows, "masked_phase_specificity"),
            repeat_noise=phase_noise,
            lower_is_better=False,
            repetitions=repetitions,
            seed=base_seed + 30,
        )
        phase_vs_b2 = paired_metric_result(
            _array(episode_rows, "b1_phase_specificity"),
            _array(episode_rows, "b2_phase_specificity"),
            repeat_noise=phase_noise,
            lower_is_better=False,
            repetitions=repetitions,
            seed=base_seed + 31,
        )
        phase_specific = {
            "positive_vs_masked": phase_positive,
            "greater_than_b2": phase_vs_b2,
            "passed": bool(phase_positive["passed"] and phase_vs_b2["passed"]),
        }
        preservation = {
            "event_coverage_delta_source_episode_values": [
                row["event_coverage_delta"] for row in episode_rows
            ],
            "event_order_violation_rate_source_episode_values": [
                row["event_order_violation_rate"] for row in episode_rows
            ],
            "passed": bool(
                min(row["event_coverage_delta"] for row in episode_rows) >= 0.0
                and max(row["event_order_violation_rate"] for row in episode_rows)
                == 0.0
            ),
        }
        criteria[factor] = {
            "action_sensitivity_vs_masked": action_vs_masked,
            "signed_semantic_margin_vs_b2": margin_vs_b2,
            "semantic_identifiability": semantic_identifiability,
            "phase_specificity": phase_specific,
            "task_envelope_preservation": preservation,
        }
    return source_rows, criteria, noise_output


def _validated_package(root: Path, baseline: str, mode: str) -> dict[str, Any]:
    verification = verify_checksums(root, root / "checksums.sha256")
    if not verification["ok"]:
        raise ValueError(f"condition replay checksum verification failed: {root}")
    manifest = _read_json(root / "condition_replay_manifest.json")
    actual_mode = manifest.get("condition_delivery_mode", "requested")
    if (
        manifest.get("baseline_id") != baseline
        or manifest.get("split") != "validation"
        or actual_mode != mode
        or manifest.get("held_out_test_read") is not False
        or manifest.get("closed_loop_execution") is not False
        or manifest.get("observation_history_changed") is not False
        or set(map(int, manifest["episode_ids"])) & HELD_OUT_EPISODES
    ):
        raise ValueError(f"invalid {baseline}/{mode} condition replay: {root}")
    all_rows = _read_jsonl(root / "condition_swap_metrics.jsonl")
    return {
        "root": root,
        "manifest": manifest,
        "all_rows": all_rows,
        "supported_rows": {
            int(row["anchor_index"]): row for row in all_rows if row["supported"]
        },
    }


def _require_matched(packages: Sequence[Mapping[str, Any]]) -> None:
    for field in (
        "m0_dataset_manifest_sha256",
        "m2_manifest_sha256",
        "anchor_registry_sha256",
    ):
        if len({package["manifest"][field] for package in packages}) != 1:
            raise ValueError(f"condition replay packages disagree on {field}")
    signatures = [
        [
            (
                int(row["anchor_index"]),
                int(row["episode_id"]),
                str(row["changed_factor"]),
                bool(row["supported"]),
            )
            for row in package["all_rows"]
        ]
        for package in packages
    ]
    if any(signature != signatures[0] for signature in signatures[1:]):
        raise ValueError("condition replay anchor inventories do not match")


def _require_mask_is_null(package: Mapping[str, Any]) -> None:
    if (
        package["manifest"].get("requested_condition_diff_delivered_to_policy")
        is not False
    ):
        raise ValueError("masked replay did not suppress the requested difference")
    for row in package["supported_rows"].values():
        if row["delivered_base_condition"] != row["delivered_target_condition"]:
            raise ValueError("masked replay delivered unequal tokens")
        if float(row["metrics"]["token_swap_action_effect"]) != 0.0 or any(
            float(value) != 0.0 for value in row["metrics"]["per_tick_effect_l1"]
        ):
            raise ValueError("masked replay is not an exact action null")


def _semantic_permutations() -> list[dict[str, str]]:
    identity = tuple(SECTORS)
    return [
        dict(zip(SECTORS, values, strict=True))
        for values in itertools.permutations(SECTORS)
        if values != identity
    ]


def _permutation_name(mapping: Mapping[str, str]) -> str:
    return "__".join(f"{sector}_to_{mapping[sector]}" for sector in SECTORS)


def _repeat_action_noise(
    reference: Mapping[str, Any],
    repeats: Sequence[Mapping[str, Any]],
    *,
    grouped: Mapping[int, Sequence[int]],
) -> float:
    values = []
    for ids in grouped.values():
        reference_value = _metric_mean(
            reference["supported_rows"], ids, "token_swap_action_effect"
        )
        values.append(
            max(
                abs(
                    reference_value
                    - _metric_mean(
                        package["supported_rows"], ids, "token_swap_action_effect"
                    )
                )
                for package in repeats
            )
        )
    return _q(values, 0.975)


def _metric_mean(
    rows: Mapping[int, Mapping[str, Any]], ids: Sequence[int], metric: str
) -> float:
    return _mean([float(rows[index]["metrics"][metric]) for index in ids])


def _array(rows: Sequence[Mapping[str, Any]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=np.float64)


def _mean(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("mean values must be finite and non-empty")
    return float(np.mean(array))


def _q(values: Sequence[float], quantile: float) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("quantile values must be finite and non-empty")
    return float(np.quantile(array, quantile))


def _package_identity(package: Mapping[str, Any]) -> dict[str, Any]:
    root = package["root"]
    manifest = package["manifest"]
    return {
        "path": str(root),
        "baseline_id": manifest["baseline_id"],
        "repeat_id": manifest["repeat_id"],
        "condition_delivery_mode": manifest.get("condition_delivery_mode", "requested"),
        "checkpoint_sha256": manifest["checkpoint"]["sha256"],
        "manifest_sha256": sha256_file(root / "condition_replay_manifest.json"),
        "checksums_sha256": sha256_file(root / "checksums.sha256"),
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
