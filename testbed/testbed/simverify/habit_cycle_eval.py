"""Recorded-observation evaluation for the frozen ready-to-ready habit task."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from testbed.data.dataset import _read_camera_image
from testbed.policies.offline_eval import (
    compute_action_metrics,
    load_policy_for_episode,
)
from testbed.simverify.artifacts import (
    artifact_identity,
    write_checksums,
    write_json,
    write_jsonl,
)
from testbed.simverify.contracts import git_provenance, sha256_file
from testbed.simverify.m2_eval import (
    extract_ordered_task_events,
    validate_replay_trace_arrays,
)
from testbed.simverify.m3_replay import cycle_action_metrics

CAMERAS = ("video4", "video5", "video6", "video7")
SECTORS = ("left", "center", "right")
HELD_OUT_SOURCE_EPISODES = frozenset({1, 13, 25, 33})


def sector_condition(current_sector: str, target_sector: str) -> np.ndarray:
    """Build the frozen current+target one-hot condition."""

    if current_sector not in SECTORS or target_sector not in SECTORS:
        raise ValueError("condition sectors must be left, center, or right")
    vector = np.zeros(6, dtype=np.float32)
    vector[SECTORS.index(current_sector)] = 1.0
    vector[3 + SECTORS.index(target_sector)] = 1.0
    return vector


def delivered_condition_rows(
    recorded: np.ndarray,
    committed_mask: np.ndarray,
    *,
    target_override: str | None = None,
) -> np.ndarray:
    """Return causal condition rows, optionally changing only the committed target."""

    rows = np.asarray(recorded, dtype=np.float32)
    mask = np.asarray(committed_mask, dtype=np.uint8)
    if rows.ndim != 2 or rows.shape[1] != 6:
        raise ValueError("recorded condition must have shape (T, 6)")
    if mask.shape != (rows.shape[0],):
        raise ValueError("committed mask must have shape (T,)")
    if not np.isin(mask, [0, 1]).all():
        raise ValueError("committed mask must be binary")
    first_committed = np.flatnonzero(mask)
    if first_committed.size == 0:
        raise ValueError("cycle has no post-dump committed rows")
    commit_index = int(first_committed[0])
    if mask[:commit_index].any() or not mask[commit_index:].all():
        raise ValueError("committed mask must be a single false-to-true transition")
    if not np.allclose(rows[:commit_index], 0.0):
        raise ValueError("pre-dump condition rows must be inactive zeros")
    committed = rows[commit_index:]
    if not np.allclose(committed, committed[0]):
        raise ValueError("committed condition must remain atomic through cycle end")
    if not np.allclose(committed[:, :3].sum(axis=1), 1.0):
        raise ValueError("committed current sector must be one-hot")
    if not np.allclose(committed[:, 3:].sum(axis=1), 1.0):
        raise ValueError("committed target sector must be one-hot")

    delivered = rows.copy()
    if target_override is not None:
        if target_override not in SECTORS:
            raise ValueError("target override must be left, center, or right")
        delivered[commit_index:, 3:] = 0.0
        delivered[commit_index:, 3 + SECTORS.index(target_override)] = 1.0
    return delivered


def split_action_metrics(
    expert_action: np.ndarray,
    policy_action: np.ndarray,
    committed_mask: np.ndarray,
) -> dict[str, Any]:
    """Report full-cycle, pre-dump, and post-commit teacher-forced metrics."""

    expert = np.asarray(expert_action, dtype=np.float32)
    policy = np.asarray(policy_action, dtype=np.float32)
    mask = np.asarray(committed_mask, dtype=bool)
    if expert.shape != policy.shape or expert.ndim != 2 or expert.shape[1] != 4:
        raise ValueError("expert and policy action must share shape (T, 4)")
    if mask.shape != (expert.shape[0],) or not mask.any() or mask.all():
        raise ValueError("metrics require non-empty pre-dump and post-commit windows")
    return {
        "full_cycle": compute_action_metrics(expert, policy),
        "pre_dump": compute_action_metrics(expert[~mask], policy[~mask]),
        "post_commit": compute_action_metrics(expert[mask], policy[mask]),
    }


def condition_swap_metrics(
    base_action: np.ndarray,
    alternate_action: np.ndarray,
    committed_mask: np.ndarray,
    *,
    expected_swing_delta_sign: int | None = None,
) -> dict[str, Any]:
    """Measure a fixed-observation target intervention without claiming success."""

    base = np.asarray(base_action, dtype=np.float32)
    alternate = np.asarray(alternate_action, dtype=np.float32)
    mask = np.asarray(committed_mask, dtype=bool)
    if base.shape != alternate.shape or base.ndim != 2 or base.shape[1] != 4:
        raise ValueError("condition swap actions must share shape (T, 4)")
    if mask.shape != (base.shape[0],) or not mask.any() or mask.all():
        raise ValueError("condition swap requires pre and post windows")
    delta = alternate - base
    pre = delta[~mask]
    post = delta[mask]
    observed_sign = int(np.sign(float(np.mean(post[:, 0]))))
    if expected_swing_delta_sign not in {None, -1, 1}:
        raise ValueError("expected swing delta sign must be -1, 1, or None")
    return {
        "pre_dump_effect_l1": float(np.mean(np.abs(pre))),
        "post_commit_effect_l1": float(np.mean(np.abs(post))),
        "post_commit_swing_delta_mean": float(np.mean(post[:, 0])),
        "expected_swing_delta_sign": expected_swing_delta_sign,
        "observed_swing_delta_sign": observed_sign,
        "semantic_direction_correct": (
            None
            if expected_swing_delta_sign is None
            else observed_sign == expected_swing_delta_sign
        ),
        "post_commit_non_swing_effect_l1": float(np.mean(np.abs(post[:, 1:]))),
        "causal_localization_ratio": float(
            np.mean(np.abs(post)) / max(np.mean(np.abs(pre)), 1e-12)
        ),
        "closed_loop_execution": False,
    }


def replay_habit_cycle(
    *,
    policy: Any,
    episode: h5py.File,
    pass_condition_to_policy: bool,
    target_override: str | None = None,
    camera_names: Sequence[str] = CAMERAS,
) -> dict[str, np.ndarray]:
    """Replay one derived ready-to-ready cycle on recorded observations."""

    step_count = int(episode["action"].shape[0])
    if step_count < 2:
        raise ValueError("habit cycle must have at least two rows")
    recorded_condition = np.asarray(
        episode["conditions/cycle_condition_v1"],
        dtype=np.float32,
    )
    committed_mask = np.asarray(
        episode["conditions/target_committed_mask"],
        dtype=np.uint8,
    )
    delivered = delivered_condition_rows(
        recorded_condition,
        committed_mask,
        target_override=target_override,
    )
    if target_override is not None and not pass_condition_to_policy:
        raise ValueError("unconditioned replay cannot accept a target override")
    if hasattr(policy, "reset"):
        policy.reset()

    raw_normalized_rows: list[np.ndarray] = []
    raw_direct_rows: list[np.ndarray] = []
    aggregated = np.zeros((step_count, 4), dtype=np.float32)
    query_count: int | None = None
    qpos = episode["observations/qpos"]
    qvel = episode["observations/qvel"]
    for index in range(step_count):
        observation: dict[str, Any] = {
            "qpos": np.asarray(qpos[index], dtype=np.float32),
            "qvel": np.asarray(qvel[index], dtype=np.float32),
        }
        if pass_condition_to_policy:
            observation["cycle_condition_v1"] = delivered[index].copy()
        for camera in camera_names:
            observation[f"image_{camera}"] = _read_camera_image(
                episode,
                camera,
                index,
            )
        aggregated[index] = np.asarray(
            policy.predict(observation),
            dtype=np.float32,
        ).reshape(4)
        raw_normalized = np.asarray(
            policy.last_raw_action_chunk(),
            dtype=np.float32,
        )
        raw_direct = np.asarray(
            policy.last_raw_action_chunk_direct(),
            dtype=np.float32,
        )
        if (
            raw_normalized.ndim != 2
            or raw_normalized.shape[1] != 4
            or raw_direct.shape != raw_normalized.shape
        ):
            raise ValueError("policy raw chunks must share shape (Q, 4)")
        if query_count is None:
            query_count = int(raw_normalized.shape[0])
        if int(raw_normalized.shape[0]) != query_count:
            raise ValueError("policy query count changed during replay")
        raw_normalized_rows.append(raw_normalized.copy())
        raw_direct_rows.append(raw_direct.copy())

    arrays = {
        "raw_policy_chunk_normalized": np.stack(raw_normalized_rows),
        "raw_policy_chunk_direct": np.stack(raw_direct_rows),
        "temporal_aggregation_action": aggregated.copy(),
        "future_runtime_safe_action": aggregated.copy(),
        "expert_action": np.asarray(episode["action"], dtype=np.float32),
        "condition": delivered.copy(),
        "condition_recorded": recorded_condition,
        "condition_delivered": delivered,
        "target_committed_mask": committed_mask,
        "condition_valid_mask": np.asarray(
            episode["conditions/valid_mask"],
            dtype=np.uint8,
        ),
        "condition_cycle_id": np.asarray(
            episode["conditions/cycle_id"],
            dtype=np.int64,
        ),
        "target_tick": np.asarray(
            episode["diagnostics/target_tick"],
            dtype=np.int64,
        ),
        "source_observation_index": np.asarray(
            episode["diagnostics/source_observation_index"],
            dtype=np.int64,
        ),
        "observation_age_ticks": np.zeros(step_count, dtype=np.int64),
        "action_age_ticks": np.zeros(step_count, dtype=np.int64),
    }
    validate_replay_trace_arrays(arrays, chunk_size=int(query_count or 0))
    return arrays


def run_habit_validation_replay(
    *,
    repo_root: str | Path,
    output_root: str | Path,
    dataset_root: str | Path,
    bundle_roots: Mapping[str, str | Path],
    event_envelope_path: str | Path = (
        "/data/pingfan/Excavator_real_stack_data/"
        "sim_observable_cycle_v3_m2_contract_v1/expert_event_envelope_v1.json"
    ),
) -> dict[str, Any]:
    """Run matched B0/B1/B2 validation replay without reading held-out sources."""

    repository = Path(repo_root).resolve(strict=True)
    git = git_provenance(repository)
    if (
        not git.get("git_available")
        or git.get("branch") != "v2.0.0-simVerify"
        or bool(git.get("dirty"))
    ):
        raise ValueError("habit validation replay requires a clean SimVerify worktree")
    if set(bundle_roots) != {"B0", "B1", "B2"}:
        raise ValueError("bundle roots must contain exactly B0, B1, and B2")
    destination = Path(output_root).resolve(strict=False)
    if destination.exists():
        raise FileExistsError(f"immutable habit replay exists: {destination}")
    dataset = Path(dataset_root).resolve(strict=True)
    manifest = _read_json(dataset / "dataset_manifest.json")
    split = _read_json(dataset / "derived_split.yaml")
    val_ids = list(map(int, split["val_ids"]))
    cycle_rows = {
        int(row["derived_episode_id"]): row
        for row in manifest["cycles"]
        if row["status"] == "written"
    }
    if set(val_ids) - set(cycle_rows):
        raise ValueError("validation split references missing derived cycles")
    source_ids = {int(cycle_rows[index]["source_episode_id"]) for index in val_ids}
    if source_ids & HELD_OUT_SOURCE_EPISODES:
        raise ValueError("held-out source episode entered validation replay")
    train_targets = _train_target_support(manifest)
    action_to_qpos_direction = _fit_swing_action_to_qpos_direction(
        dataset,
        split["train_ids"],
    )
    event_envelope_file = Path(event_envelope_path).resolve(strict=True)
    event_envelope = _read_json(event_envelope_file)
    templates = event_envelope["templates"]
    deadzone = list(map(float, event_envelope["effective_deadzone"]))
    bundles = {
        baseline: _validate_bundle(Path(path).resolve(strict=True), baseline)
        for baseline, path in bundle_roots.items()
    }

    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    temporary = destination.parent / f".{destination.name}.tmp-{os.getpid()}"
    temporary.mkdir(parents=True)
    identities: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    swap_rows: list[dict[str, Any]] = []
    try:
        for baseline in ("B0", "B1", "B2"):
            bundle = bundles[baseline]["root"]
            policy = load_policy_for_episode(
                bundle_dir=bundle,
                ckpt_path=bundle / "policy_best.ckpt",
                resolved_config_path=None,
                stats_path=None,
                max_episode_len=max(
                    int(cycle_rows[index]["cycle_length_20hz"])
                    for index in val_ids
                ),
                temporal_agg=True,
                device="cuda",
                inference_precision="fp32",
            )
            for derived_id in val_ids:
                cycle = cycle_rows[derived_id]
                path = dataset / str(cycle["path"])
                with h5py.File(path, "r") as episode:
                    arrays = replay_habit_cycle(
                        policy=policy,
                        episode=episode,
                        pass_condition_to_policy=baseline != "B0",
                    )
                    relative = Path("traces") / baseline / f"episode_{derived_id}.npz"
                    trace_path = temporary / relative
                    trace_path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(trace_path, **arrays)
                    identities.append(artifact_identity(trace_path))
                    metric_rows.append(
                        {
                            "schema": "simverify_habit_cycle_metric_v1",
                            "baseline_id": baseline,
                            "derived_episode_id": derived_id,
                            "source_episode_id": int(cycle["source_episode_id"]),
                            "source_cycle_id": int(cycle["source_cycle_id"]),
                            "current_sector": cycle["current_sector"],
                            "target_sector": cycle[
                                "hindsight_expert_target_sector"
                            ],
                            "relative_intent": cycle["relative_intent"],
                            "trace_path": str(relative),
                            "metrics": split_action_metrics(
                                arrays["expert_action"],
                                arrays["temporal_aggregation_action"],
                                arrays["target_committed_mask"],
                            ),
                            "action_grammar": cycle_action_metrics(
                                arrays,
                                templates=templates,
                                deadzone=deadzone,
                            ),
                            "expert_action_grammar": extract_ordered_task_events(
                                arrays["expert_action"],
                                templates,
                                deadzone=deadzone,
                            ),
                            "evidence_scope": "recorded-observation/offline",
                            "closed_loop_execution": False,
                        }
                    )
                    if baseline == "B0":
                        continue
                    alternate = _select_supported_alternate(
                        current_sector=str(cycle["current_sector"]),
                        base_target=str(cycle["hindsight_expert_target_sector"]),
                        train_targets=train_targets,
                    )
                    if alternate is None:
                        swap_rows.append(
                            {
                                "schema": "simverify_habit_condition_swap_v1",
                                "baseline_id": baseline,
                                "derived_episode_id": derived_id,
                                "status": "coverage_gap",
                                "base_target": cycle[
                                    "hindsight_expert_target_sector"
                                ],
                                "alternate_target": None,
                                "included_in_success_denominator": False,
                            }
                        )
                        continue
                    alternate_arrays = replay_habit_cycle(
                        policy=policy,
                        episode=episode,
                        pass_condition_to_policy=True,
                        target_override=alternate,
                    )
                    alt_relative = (
                        Path("counterfactual_traces")
                        / baseline
                        / f"episode_{derived_id}_target_{alternate}.npz"
                    )
                    alt_path = temporary / alt_relative
                    alt_path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(alt_path, **alternate_arrays)
                    identities.append(artifact_identity(alt_path))
                    swap_rows.append(
                        {
                            "schema": "simverify_habit_condition_swap_v1",
                            "baseline_id": baseline,
                            "derived_episode_id": derived_id,
                            "source_episode_id": int(cycle["source_episode_id"]),
                            "current_sector": cycle["current_sector"],
                            "base_target": cycle[
                                "hindsight_expert_target_sector"
                            ],
                            "alternate_target": alternate,
                            "status": "supported_fixed_observation_intervention",
                            "included_in_success_denominator": True,
                            "alternate_trace_path": str(alt_relative),
                            "metrics": condition_swap_metrics(
                                arrays["temporal_aggregation_action"],
                                alternate_arrays["temporal_aggregation_action"],
                                arrays["target_committed_mask"],
                                expected_swing_delta_sign=(
                                    int(action_to_qpos_direction["sign"])
                                    * int(
                                        np.sign(
                                            SECTORS.index(alternate)
                                            - SECTORS.index(
                                                str(
                                                    cycle[
                                                        "hindsight_expert_target_sector"
                                                    ]
                                                )
                                            )
                                        )
                                    )
                                ),
                            ),
                            "evidence_scope": "recorded-observation/offline",
                            "closed_loop_execution": False,
                        }
                    )
        identities.append(write_jsonl(temporary / "cycle_metrics.jsonl", metric_rows))
        identities.append(
            write_jsonl(temporary / "condition_swap_metrics.jsonl", swap_rows)
        )
        summary = _aggregate_validation_metrics(metric_rows, swap_rows)
        identities.append(write_json(temporary / "validation_summary.json", summary))
        manifest_identity = write_json(
            temporary / "validation_replay_manifest.json",
            {
                "schema": "simverify_habit_validation_replay_manifest_v1",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "git": git,
                "dataset_manifest_sha256": sha256_file(
                    dataset / "dataset_manifest.json"
                ),
                "split_manifest_sha256": sha256_file(dataset / "derived_split.yaml"),
                "event_envelope_sha256": sha256_file(event_envelope_file),
                "swing_action_to_qpos_direction": action_to_qpos_direction,
                "validation_derived_episode_ids": val_ids,
                "validation_source_episode_ids": sorted(source_ids),
                "held_out_source_episode_ids": sorted(HELD_OUT_SOURCE_EPISODES),
                "held_out_test_read": False,
                "bundle_checkpoints": {
                    baseline: {
                        "path": str(package["root"] / "policy_best.ckpt"),
                        "sha256": package["checkpoint_sha256"],
                    }
                    for baseline, package in bundles.items()
                },
                "condition_intervention": (
                    "fixed recorded observation; only committed target one-hot changes"
                ),
                "trace_contract": [
                    "raw_policy_chunk_normalized",
                    "raw_policy_chunk_direct",
                    "temporal_aggregation_action",
                    "future_runtime_safe_action",
                ],
                "evidence_scope": "recorded-observation/offline",
                "closed_loop_execution": False,
                "physical_effect_validated": False,
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
            "summary": summary,
            "manifest_sha256": manifest_identity["sha256"],
            "checksums_sha256": checksums_identity["sha256"],
            "held_out_test_read": False,
            "closed_loop_execution": False,
        }
    except BaseException as exc:
        if temporary.exists() and not (temporary / "BUILD_FAILED.json").exists():
            write_json(
                temporary / "BUILD_FAILED.json",
                {
                    "schema": "simverify_habit_validation_replay_failure_v1",
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                    "git": git,
                    "held_out_test_read": False,
                    "closed_loop_execution": False,
                },
            )
        raise


def _train_target_support(manifest: Mapping[str, Any]) -> dict[str, set[str]]:
    support = {sector: set() for sector in SECTORS}
    for row in manifest["cycles"]:
        if row["status"] == "written" and row["split"] == "train":
            support[str(row["current_sector"])].add(
                str(row["hindsight_expert_target_sector"])
            )
    return support


def _fit_swing_action_to_qpos_direction(
    dataset_root: Path,
    train_ids: Sequence[int],
) -> dict[str, Any]:
    products: list[np.ndarray] = []
    source_episode_ids: set[int] = set()
    manifest = _read_json(dataset_root / "dataset_manifest.json")
    rows = {
        int(row["derived_episode_id"]): row
        for row in manifest["cycles"]
        if row["status"] == "written"
    }
    for derived_id in map(int, train_ids):
        source_episode_ids.add(int(rows[derived_id]["source_episode_id"]))
        with h5py.File(dataset_root / str(rows[derived_id]["path"]), "r") as episode:
            action = np.asarray(episode["action"][:, 0], dtype=np.float64)
            qvel = np.asarray(
                episode["observations/qvel"][:, 0],
                dtype=np.float64,
            )
        active = (np.abs(action) > 0.05) & (np.abs(qvel) > 1e-6)
        if active.any():
            products.append(action[active] * qvel[active])
    if not products:
        raise ValueError("train split has no active swing action/qvel samples")
    values = np.concatenate(products)
    median = float(np.median(values))
    sign = int(np.sign(median))
    if sign == 0:
        raise ValueError("swing action-to-qpos direction is ambiguous")
    return {
        "schema": "simverify_swing_action_to_qpos_direction_v1",
        "fit_split": "train",
        "fit_source_episode_ids": sorted(source_episode_ids),
        "active_sample_count": int(values.size),
        "median_action_times_qvel": median,
        "sign": sign,
        "privilege_used": False,
        "held_out_test_read": False,
    }


def _select_supported_alternate(
    *,
    current_sector: str,
    base_target: str,
    train_targets: Mapping[str, set[str]],
) -> str | None:
    current_index = SECTORS.index(current_sector)
    legal = {
        sector
        for index, sector in enumerate(SECTORS)
        if abs(index - current_index) <= 1
    }
    candidates = sorted(
        (set(train_targets[current_sector]) & legal) - {base_target},
        key=SECTORS.index,
    )
    return candidates[0] if candidates else None


def _validate_bundle(path: Path, baseline: str) -> dict[str, Any]:
    metadata = _read_json(path / "run_metadata.json")
    if metadata.get("status") != "completed":
        raise ValueError(f"{baseline} training is not completed")
    if metadata.get("experiment_contract", {}).get("baseline_id") != baseline:
        raise ValueError(f"{baseline} bundle has mismatched experiment contract")
    semantics = metadata.get("checkpoint_semantics", {})
    if (
        semantics.get("real_control_allowed") is not False
        or semantics.get("evidence_scope") != "recorded-observation/offline"
    ):
        raise ValueError(f"{baseline} checkpoint semantics are unsafe")
    checkpoint = path / "policy_best.ckpt"
    return {
        "root": path,
        "metadata": metadata,
        "checkpoint_sha256": sha256_file(checkpoint),
    }


def _aggregate_validation_metrics(
    rows: Sequence[Mapping[str, Any]],
    swaps: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    baseline_summary: dict[str, Any] = {}
    for baseline in ("B0", "B1", "B2"):
        selected = [row for row in rows if row["baseline_id"] == baseline]
        if not selected:
            raise ValueError(f"validation metrics have no {baseline} rows")
        baseline_summary[baseline] = {}
        for window in ("full_cycle", "pre_dump", "post_commit"):
            values = [
                float(row["metrics"][window]["overall"]["mae"])
                for row in selected
            ]
            baseline_summary[baseline][f"{window}_episode_macro_mae"] = float(
                np.mean(values)
            )
        baseline_summary[baseline]["event_order_valid_rate"] = float(
            np.mean(
                [
                    bool(row["action_grammar"]["event_order_valid"])
                    for row in selected
                ]
            )
        )
        baseline_summary[baseline]["required_event_coverage_mean"] = float(
            np.mean(
                [
                    float(row["action_grammar"]["required_event_coverage"])
                    for row in selected
                ]
            )
        )
        baseline_summary[baseline]["deadzone_effective_recall_mean"] = float(
            np.mean(
                [
                    float(row["action_grammar"]["deadzone_effective_recall"])
                    for row in selected
                ]
            )
        )
        baseline_summary[baseline]["expert_required_event_coverage_mean"] = float(
            np.mean(
                [
                    float(row["expert_action_grammar"]["required_event_coverage"])
                    for row in selected
                ]
            )
        )
    supported_swaps = [
        row
        for row in swaps
        if row["status"] == "supported_fixed_observation_intervention"
    ]
    swap_summary = {}
    for baseline in ("B1", "B2"):
        selected = [
            row for row in supported_swaps if row["baseline_id"] == baseline
        ]
        swap_summary[baseline] = {
            "supported_anchor_count": len(selected),
            "post_commit_effect_l1_mean": (
                float(
                    np.mean(
                        [
                            row["metrics"]["post_commit_effect_l1"]
                            for row in selected
                        ]
                    )
                )
                if selected
                else None
            ),
            "pre_dump_effect_l1_max": (
                float(
                    np.max(
                        [row["metrics"]["pre_dump_effect_l1"] for row in selected]
                    )
                )
                if selected
                else None
            ),
            "semantic_direction_correct_rate": (
                float(
                    np.mean(
                        [
                            bool(row["metrics"]["semantic_direction_correct"])
                            for row in selected
                        ]
                    )
                )
                if selected
                else None
            ),
        }
    return {
        "schema": "simverify_habit_validation_summary_v1",
        "baseline_metrics": baseline_summary,
        "condition_swap_metrics": swap_summary,
        "development_split_only": True,
        "gate_thresholds_frozen": False,
        "held_out_test_read": False,
        "evidence_scope": "recorded-observation/offline",
        "closed_loop_execution": False,
        "interpretation_guard": (
            "teacher-forced replay and target sensitivity do not prove target arrival "
            "or closed-loop cycle completion"
        ),
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
