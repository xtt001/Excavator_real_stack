"""Raw-source falsification audit for the expert-habit cycle definition."""

from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from testbed.simverify.annotations import (
    EpisodeSignals,
    annotate_numeric_cycles,
    bootstrap_numeric_thresholds,
    classify_sector,
    fit_numeric_annotation_thresholds,
    fit_sector_thresholds,
)
from testbed.simverify.artifacts import (
    artifact_identity,
    verify_checksums,
    write_checksums,
    write_json,
)
from testbed.simverify.contracts import (
    assert_source_provenance_unchanged,
    collect_hdf5_source_provenance,
    git_provenance,
)
from testbed.simverify.features import FrozenResNet18FeatureExtractor
from testbed.simverify.gates import assign_episode_splits
from testbed.simverify.habit_cycle import (
    DIAGNOSTIC_NONADJACENT,
    RELATIVE_INTENTS,
    SECTORS,
    relative_intent,
    resolve_target_sector,
)

DEFAULT_SOURCE_ROOT = Path(
    "/data/pingfan/excavator_testbed_data/"
    "yulong_v2_2_pro_full_task_four_camera_jpeg_20260717_cycle_clean_v1"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/data/pingfan/Excavator_real_stack_data/"
    "simverify_habit_cycle_definition_v5"
)
DEFAULT_RESNET18_WEIGHTS = Path(
    "/home/pingfan/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth"
)
DEFAULT_RESNET18_SHA256 = (
    "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
)
DEFAULT_SPLIT_SEED = "simverify-m0-v1:20260724"
DEFAULT_BOOTSTRAP_SEED = 20260728
DEFAULT_BOOTSTRAP_SAMPLES = 1024
DEFAULT_NULL_SAMPLES = 1024
HISTORY_SECONDS = (0.0, 0.5, 1.0, 2.0)
EVIDENCE_SCOPE = "recorded-observation/offline_definition_audit"


def run_habit_cycle_definition_audit(
    *,
    source_root: str | Path = DEFAULT_SOURCE_ROOT,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    repo_root: str | Path,
    weights_path: str | Path = DEFAULT_RESNET18_WEIGHTS,
    expected_weights_sha256: str = DEFAULT_RESNET18_SHA256,
    split_seed: str = DEFAULT_SPLIT_SEED,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    null_samples: int = DEFAULT_NULL_SAMPLES,
    feature_device: str | None = None,
    feature_batch_size: int = 64,
    extractor: FrozenResNet18FeatureExtractor | None = None,
) -> dict[str, Any]:
    """Build an immutable, no-training definition-audit package."""

    source = Path(source_root).resolve(strict=True)
    clean_dir = source / "clean_all_vds"
    if not clean_dir.is_dir():
        raise FileNotFoundError(clean_dir)
    destination = Path(output_root).resolve(strict=False)
    if destination.exists():
        raise FileExistsError(f"immutable audit output already exists: {destination}")
    repository = Path(repo_root).resolve(strict=True)
    repo = git_provenance(repository)
    if repo.get("branch") != "v2.0.0-simVerify":
        raise ValueError("habit audit requires branch v2.0.0-simVerify")
    if bool(repo.get("dirty")):
        raise ValueError("habit audit requires a clean committed worktree")
    if int(bootstrap_samples) <= 0 or int(null_samples) <= 0:
        raise ValueError("bootstrap_samples and null_samples must be positive")

    paths = discover_source_episodes(clean_dir)
    source_chains = {
        episode_id: collect_hdf5_source_provenance(path)
        for episode_id, path in paths.items()
    }
    source_snapshot_records = [
        row for episode_id in sorted(source_chains) for row in source_chains[episode_id]
    ]
    metadata = {
        episode_id: read_episode_metadata(path)
        for episode_id, path in paths.items()
    }
    split = assign_episode_splits(
        {
            episode_id: str(row["controller_epoch"])
            for episode_id, row in metadata.items()
        },
        seed=str(split_seed),
    )
    train_ids = list(map(int, split["splits"]["train"]))
    validation_ids = list(map(int, split["splits"]["validation"]))
    held_out_ids = list(map(int, split["splits"]["held_out_test"]))
    observable_ids = train_ids + validation_ids
    signals = {episode_id: load_episode_signals(paths[episode_id]) for episode_id in observable_ids}
    assert_source_provenance_unchanged(source_snapshot_records)

    numeric_thresholds = fit_numeric_annotation_thresholds(
        [signals[episode_id] for episode_id in train_ids],
        action_deadzone=0.05,
    )
    numeric_bootstrap = bootstrap_numeric_thresholds(
        [signals[episode_id] for episode_id in train_ids],
        action_deadzone=0.05,
        samples=int(bootstrap_samples),
        seed=int(bootstrap_seed),
    )
    numeric_cycles = {
        episode_id: annotate_numeric_cycles(signals[episode_id], numeric_thresholds)
        for episode_id in observable_ids
    }
    sector_thresholds, sector_bootstrap = fit_sector_thresholds_with_bootstrap(
        numeric_cycles,
        train_ids=train_ids,
        samples=int(bootstrap_samples),
        seed=int(bootstrap_seed) + 1,
    )

    raw_candidates = build_transition_candidates(
        numeric_cycles,
        signals=signals,
        metadata=metadata,
        split=split,
        sector_thresholds=sector_thresholds,
        dump_swing_threshold=float(
            numeric_thresholds["dump_release"]["swing_threshold"]
        ),
        swing_speed_threshold=float(
            numeric_thresholds["ready"]["swing_speed_threshold"]
        ),
        ready_envelope_steps=int(
            numeric_thresholds["ready"]["minimum_envelope_steps"]
        ),
    )
    dwell_contract = fit_causal_confirmation_dwell(
        raw_candidates,
        signals=signals,
        sector_thresholds=sector_thresholds,
        dump_swing_threshold=float(
            numeric_thresholds["dump_release"]["swing_threshold"]
        ),
        swing_speed_threshold=float(
            numeric_thresholds["ready"]["swing_speed_threshold"]
        ),
    )
    candidates = enumerate_causal_candidates(
        raw_candidates,
        dwell_steps=int(dwell_contract["selected_dwell_steps"]),
        signals=signals,
        sector_thresholds=sector_thresholds,
        dump_swing_threshold=float(
            numeric_thresholds["dump_release"]["swing_threshold"]
        ),
        swing_speed_threshold=float(
            numeric_thresholds["ready"]["swing_speed_threshold"]
        ),
    )

    if extractor is None:
        device = feature_device or (
            "cuda" if _cuda_is_available() else "cpu"
        )
        extractor = FrozenResNet18FeatureExtractor(
            weights_path,
            expected_checkpoint_sha256=expected_weights_sha256,
            device=device,
            batch_size=int(feature_batch_size),
        )
    feature_rows = extract_candidate_features(
        candidates,
        paths=paths,
        extractor=extractor,
    )
    visual_audit, candidates = build_visual_boundary_audit(
        candidates,
        feature_rows=feature_rows,
        null_samples=int(null_samples),
        seed=int(bootstrap_seed) + 2,
    )
    resolved_by_key = {
        (int(row["episode_id"]), int(row["cycle_id"])): row
        for row in candidates
        if row["causal_confirm_matches_reference"]
    }
    for row in candidates:
        if row["causal_confirm_matches_reference"]:
            _full_cycle_source_range(row, resolved_by_key)
    observation_audit = build_observation_sufficiency_audit(
        candidates,
        signals=signals,
    )
    support_payload = build_habit_condition_support(
        candidates,
        feature_rows=feature_rows,
        signals=signals,
        null_samples=int(null_samples),
        seed=int(bootstrap_seed) + 4,
    )
    transition_payload = build_habit_transition_inventory(
        candidates,
        numeric_cycles=numeric_cycles,
        metadata=metadata,
        split=split,
        held_out_ids=held_out_ids,
    )
    scenario_payload = build_scenario_candidates(
        candidates,
        signals=signals,
        source_chains=source_chains,
        transition_inventory=transition_payload,
        condition_support=support_payload,
    )
    boundary_payload = build_dig_ready_boundary_audit(
        candidates,
        numeric_thresholds=numeric_thresholds,
        numeric_bootstrap=numeric_bootstrap,
        sector_thresholds=sector_thresholds,
        sector_bootstrap=sector_bootstrap,
        dwell_contract=dwell_contract,
        visual_audit=visual_audit,
        observation_audit=observation_audit,
    )
    decision = definition_decision(
        transition_inventory=transition_payload,
        boundary_audit=boundary_payload,
        condition_support=support_payload,
    )

    generated_at = datetime.now(timezone.utc).isoformat()
    temporary = destination.parent / f".{destination.name}.tmp-{os.getpid()}"
    if temporary.exists():
        raise FileExistsError(f"temporary audit output exists: {temporary}")
    temporary.mkdir(parents=True)
    try:
        source_snapshot = write_json(
            temporary / "source_snapshot_manifest.json",
            {
                "schema": "simverify_habit_source_snapshot_v1",
                "source_root": str(source),
                "episode_count": len(paths),
                "held_out_observation_read_count": 0,
                "held_out_episode_ids": held_out_ids,
                "episodes": [
                    {
                        "episode_id": int(episode_id),
                        "split": _split_name(split, episode_id),
                        "metadata": metadata[episode_id],
                        "resolved_source_chain": source_chains[episode_id],
                    }
                    for episode_id in sorted(paths)
                ],
                "provenance": {
                    "generated_at": generated_at,
                    "git": repo,
                    "evidence_scope": EVIDENCE_SCOPE,
                },
            },
        )
        common = {
            "generated_at": generated_at,
            "git": repo,
            "source_root": str(source),
            "source_snapshot_sha256": source_snapshot["sha256"],
            "evidence_scope": EVIDENCE_SCOPE,
            "held_out_observation_read_count": 0,
            "training_executed": False,
            "planner_model_trained": False,
            "physical_effect_validated": False,
        }
        identities = [source_snapshot]
        payloads = (
            (
                "habit_cycle_boundaries_v1.json",
                {
                    "schema": "habit_cycle_boundaries_v1",
                    "range_semantics": (
                        "raw_source_step_half_open_ready_start_to_ready_end"
                    ),
                    "records": candidates,
                },
            ),
            ("habit_transition_inventory_v1.json", transition_payload),
            ("dig_ready_boundary_audit_v1.json", boundary_payload),
            ("habit_condition_support_v1.json", support_payload),
            ("expert_habit_scenario_candidates_v1.json", scenario_payload),
            ("definition_falsification_decision_v1.json", decision),
        )
        for name, payload in payloads:
            identity = write_json(
                temporary / name,
                {**payload, "provenance": common},
            )
            identities.append(identity)
        manifest = write_json(
            temporary / "audit_manifest.json",
            {
                "schema": "simverify_habit_cycle_definition_audit_manifest_v1",
                "status": "complete_user_review_required",
                "definition_decision": decision["decision"],
                "scenario_freeze_authorized": False,
                "training_authorized": False,
                "held_out_test_authorized": False,
                "artifacts": [
                    _relative_identity(identity, temporary) for identity in identities
                ],
                "feature_extractor": extractor.provenance,
                "provenance": common,
            },
        )
        identities.append(manifest)
        checksums = write_checksums(
            temporary,
            identities,
            path=temporary / "checksums.sha256",
        )
        assert_source_provenance_unchanged(source_snapshot_records)
        os.replace(temporary, destination)
        verification = verify_checksums(
            destination,
            destination / "checksums.sha256",
        )
        if not verification["ok"]:
            raise RuntimeError("written habit audit checksum verification failed")
        return {
            "schema": "simverify_habit_cycle_definition_audit_result_v1",
            "output_root": str(destination),
            "decision": decision["decision"],
            "training_authorized": False,
            "scenario_freeze_authorized": False,
            "manifest": artifact_identity(destination / "audit_manifest.json"),
            "checksums": artifact_identity(destination / checksums["path"].split("/")[-1]),
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


def discover_source_episodes(clean_dir: Path) -> dict[int, Path]:
    result = {
        int(path.stem.split("_", 1)[1]): path.resolve(strict=True)
        for path in clean_dir.glob("episode_*.hdf5")
    }
    expected = {
        1, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 16,
        19, 20, 23, 24, 25, 27, 28, 29, 30, 32, 33, 34,
    }
    if set(result) != expected:
        raise ValueError(f"clean source episode inventory changed: {sorted(result)}")
    return result


def read_episode_metadata(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        metadata = handle["metadata"].attrs
        return {
            "episode_id": str(metadata["episode_id"]),
            "controller_epoch": str(metadata["controller_epoch"]),
            "dt": float(metadata["dt"]),
            "control_hz": int(metadata["control_hz"]),
            "action_semantics": str(metadata["action_semantics"]),
            "qpos_order": str(metadata["qpos_order"]),
            "qvel_order": str(metadata["qvel_order"]),
            "action_order": str(metadata["action_order"]),
            "camera_names": str(metadata["camera_names"]),
            "operator_id": str(metadata.get("operator_id", "unknown")),
            "teleop_input": str(metadata.get("teleop_input", "unknown")),
            "n_steps": int(handle["action"].shape[0]),
        }


def load_episode_signals(path: Path) -> EpisodeSignals:
    episode_id = int(path.stem.split("_", 1)[1])
    with h5py.File(path, "r") as handle:
        result = EpisodeSignals(
            episode_id=episode_id,
            step_id=np.asarray(handle["timestamps/step_id"], dtype=np.int64),
            qpos=np.asarray(handle["observations/qpos"], dtype=np.float32),
            qvel=np.asarray(handle["observations/qvel"], dtype=np.float32),
            action=np.asarray(handle["action"], dtype=np.float32),
            dt=float(handle["metadata"].attrs["dt"]),
        )
    result.validate()
    return result


def fit_sector_thresholds_with_bootstrap(
    cycles: Mapping[int, Sequence[Mapping[str, Any]]],
    *,
    train_ids: Sequence[int],
    samples: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    base_records = [
        record for episode_id in train_ids for record in cycles[int(episode_id)]
    ]
    fitted = fit_sector_thresholds(base_records)
    rng = np.random.default_rng(int(seed))
    boundaries: list[list[float]] = []
    centers: list[list[float]] = []
    failures = 0
    ids = np.asarray(list(map(int, train_ids)), dtype=np.int64)
    for _ in range(int(samples)):
        draw = rng.choice(ids, size=len(ids), replace=True)
        records = [record for episode_id in draw for record in cycles[int(episode_id)]]
        try:
            sample = fit_sector_thresholds(records)
        except ValueError:
            failures += 1
            continue
        boundaries.append(list(map(float, sample["boundaries_low_to_high"])))
        centers.append(list(map(float, sample["cluster_centers_low_to_high"])))
    if not boundaries:
        raise ValueError("all sector threshold bootstrap samples failed")
    boundary_array = np.asarray(boundaries, dtype=np.float64)
    base_boundary = np.asarray(fitted["boundaries_low_to_high"], dtype=np.float64)
    margin = float(
        np.max(
            np.maximum(
                base_boundary - np.quantile(boundary_array, 0.025, axis=0),
                np.quantile(boundary_array, 0.975, axis=0) - base_boundary,
            )
        )
    )
    fitted = dict(fitted)
    fitted["boundary_review_margin"] = margin
    fitted["boundary_review_margin_source"] = (
        "source_episode_bootstrap_maximum_ci95_half_width"
    )
    return fitted, {
        "schema": "habit_sector_source_episode_bootstrap_v1",
        "seed": int(seed),
        "requested_samples": int(samples),
        "successful_samples": len(boundaries),
        "failed_samples": int(failures),
        "boundary_p02_5": np.quantile(boundary_array, 0.025, axis=0).tolist(),
        "boundary_p97_5": np.quantile(boundary_array, 0.975, axis=0).tolist(),
        "center_p02_5": np.quantile(
            np.asarray(centers, dtype=np.float64), 0.025, axis=0
        ).tolist(),
        "center_p97_5": np.quantile(
            np.asarray(centers, dtype=np.float64), 0.975, axis=0
        ).tolist(),
        "boundary_review_margin": margin,
    }


def build_transition_candidates(
    cycles: Mapping[int, Sequence[Mapping[str, Any]]],
    *,
    signals: Mapping[int, EpisodeSignals],
    metadata: Mapping[int, Mapping[str, Any]],
    split: Mapping[str, Any],
    sector_thresholds: Mapping[str, Any],
    dump_swing_threshold: float,
    swing_speed_threshold: float,
    ready_envelope_steps: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for episode_id in sorted(cycles):
        episode = signals[episode_id]
        episode_cycles = cycles[episode_id]
        episode_records: list[dict[str, Any]] = []
        for cycle_index, cycle in enumerate(episode_cycles):
            reasons: list[str] = []
            current_value = cycle["numeric_sector_evidence"].get(
                "current_swing_qpos"
            )
            current_result = (
                (None, 0.0, True)
                if current_value is None
                else classify_sector(float(current_value), sector_thresholds)
            )
            current, current_confidence, current_boundary = current_result
            target: str | None = None
            target_confidence = 0.0
            target_boundary = True
            dump_event = cycle["observable_events"].get("dump_end_proxy")
            if dump_event is None:
                reasons.append("dump_end_not_identifiable")
            next_cycle = (
                episode_cycles[cycle_index + 1]
                if cycle_index + 1 < len(episode_cycles)
                else None
            )
            next_dump_event = (
                None
                if next_cycle is None
                else next_cycle["observable_events"].get("dump_start_proxy")
            )
            if next_dump_event is None:
                reasons.append("next_dump_start_not_identifiable")
            dump_step = (
                None
                if dump_event is None
                else int(dump_event["representative_step"])
            )
            next_dig_step: int | None = None
            interval: list[int] | None = None
            if dump_step is not None and next_dump_event is not None:
                search_end = int(next_dump_event["representative_step"])
                if search_end <= int(dump_step) + 1:
                    reasons.append("invalid_dump_to_next_ready_search_order")
                else:
                    search_start = _ready_search_start(
                        episode,
                        dump_step=int(dump_step),
                        search_end=search_end,
                        swing_speed_threshold=float(
                            swing_speed_threshold
                        ),
                    )
                    ready_candidate = (
                        np.asarray(episode.qpos[:, 0], dtype=np.float64)
                        < float(dump_swing_threshold)
                    ) & (
                        np.abs(
                            np.asarray(
                                episode.qvel[:, 0],
                                dtype=np.float64,
                            )
                        )
                        <= float(swing_speed_threshold)
                    )
                    runs = [
                        run
                        for run in _true_runs(
                            ready_candidate,
                            start=search_start,
                            end=search_end,
                        )
                        if run[1] - run[0] >= int(ready_envelope_steps)
                    ]
                    if not runs:
                        reasons.append("dig_ready_reference_not_identifiable")
                    else:
                        # This is the first causal low-speed work-area
                        # envelope after dump. Its own qpos defines the
                        # hindsight target; the earlier bucket proxy is not
                        # allowed to label the sector.
                        selected = runs[0]
                        interval = [
                            int(selected[0]),
                            int(selected[0]) + int(ready_envelope_steps),
                        ]
                        next_dig_step = int(interval[1])
                        target_value = float(
                            episode.qpos[int(selected[0]), 0]
                        )
                        (
                            target,
                            target_confidence,
                            target_boundary,
                        ) = classify_sector(
                            target_value,
                            sector_thresholds,
                        )
            if current is None:
                reasons.append("current_sector_not_identifiable")
            if target is None:
                reasons.append("next_sector_not_identifiable")
            intent = (
                None
                if current is None or target is None
                else relative_intent(current, target)
            )
            episode_records.append(
                {
                    "schema": "habit_transition_candidate_v1",
                    "episode_id": int(episode_id),
                    "cycle_id": int(cycle["cycle_id"]),
                    "split": _split_name(split, episode_id),
                    "controller_epoch": str(
                        metadata[episode_id]["controller_epoch"]
                    ),
                    "source_dt_s": float(episode.dt),
                    "current_sector": current,
                    "hindsight_expert_target_sector": target,
                    "sector_evidence": {
                        "current_confidence": float(current_confidence),
                        "target_confidence": float(target_confidence),
                        "current_boundary_review": bool(current_boundary),
                        "target_boundary_review": bool(target_boundary),
                    },
                    "relative_intent": intent,
                    "training_main_scope": (
                        intent in RELATIVE_INTENTS
                        if intent is not None
                        else False
                    ),
                    "diagnostic_nonadjacent": (
                        intent == DIAGNOSTIC_NONADJACENT
                    ),
                    "dump_end_step": dump_step,
                    "next_dig_entry_step": next_dig_step,
                    "dig_ready_reference_interval": interval,
                    "numeric_causal_candidate_steps": [],
                    "causal_confirm_step": None,
                    "causal_confirmed": False,
                    "causal_confirm_matches_reference": False,
                    "reason_codes": sorted(set(reasons)),
                    "command": {
                        "scripted_target_sector": None,
                        "source": "not_recorded_historical_data",
                    },
                    "outcome": {
                        "hindsight_expert_target_sector": target,
                        "source": "observable_ready_capture",
                    },
                }
            )
        for index in range(1, len(episode_records)):
            previous = episode_records[index - 1]
            row = episode_records[index]
            if (
                int(row["cycle_id"]) != int(previous["cycle_id"]) + 1
                or previous["hindsight_expert_target_sector"] is None
            ):
                continue
            row["current_sector"] = previous[
                "hindsight_expert_target_sector"
            ]
            row["sector_evidence"]["current_confidence"] = previous[
                "sector_evidence"
            ]["target_confidence"]
            row["sector_evidence"]["current_boundary_review"] = previous[
                "sector_evidence"
            ]["target_boundary_review"]
            row["reason_codes"] = [
                code
                for code in row["reason_codes"]
                if code != "current_sector_not_identifiable"
            ]
            target = row["hindsight_expert_target_sector"]
            intent = (
                None
                if target is None
                else relative_intent(str(row["current_sector"]), str(target))
            )
            row["relative_intent"] = intent
            row["training_main_scope"] = intent in RELATIVE_INTENTS
            row["diagnostic_nonadjacent"] = (
                intent == DIAGNOSTIC_NONADJACENT
            )
        records.extend(episode_records)
    return records


def fit_causal_confirmation_dwell(
    candidates: Sequence[Mapping[str, Any]],
    *,
    signals: Mapping[int, EpisodeSignals],
    sector_thresholds: Mapping[str, Any],
    dump_swing_threshold: float,
    swing_speed_threshold: float,
) -> dict[str, Any]:
    """Fit the shortest dwell with the best train run discrimination.

    The reference interval may use hindsight, but the candidate runs are
    enumerated forward from dump_end.  The selected dwell is therefore a
    reproducible train-derived operating point, not a hand-written duration.
    """

    positive_lengths: list[int] = []
    negative_lengths: list[int] = []
    for row in candidates:
        reference = row["dig_ready_reference_interval"]
        if (
            row["split"] != "train"
            or reference is None
            or row["relative_intent"] not in RELATIVE_INTENTS
        ):
            continue
        episode = signals[int(row["episode_id"])]
        eligible = _target_work_sector_mask(
            episode.qpos[:, 0],
            target=str(row["hindsight_expert_target_sector"]),
            sector_thresholds=sector_thresholds,
            dump_swing_threshold=float(dump_swing_threshold),
        ) & (
            np.abs(np.asarray(episode.qvel[:, 0], dtype=np.float64))
            <= float(swing_speed_threshold)
        )
        start = int(row["dump_end_step"]) + 1
        end = int(row["next_dig_entry_step"])
        start = _ready_search_start(
            episode,
            dump_step=int(row["dump_end_step"]),
            search_end=end,
            swing_speed_threshold=float(swing_speed_threshold),
        )
        reference_start, reference_end = map(int, reference)
        positive_lengths.append(reference_end - reference_start)
        for run_start, run_end in _true_runs(eligible, start=start, end=end):
            if run_end <= reference_start:
                negative_lengths.append(run_end - run_start)
    if not positive_lengths:
        raise ValueError("no train dig-ready reference interval is available")
    upper = max(positive_lengths + negative_lengths)
    scored: list[dict[str, Any]] = []
    for dwell in range(1, upper + 1):
        true_positive_rate = float(
            np.mean([length >= dwell for length in positive_lengths])
        )
        false_positive_rate = (
            float(np.mean([length >= dwell for length in negative_lengths]))
            if negative_lengths
            else 0.0
        )
        scored.append(
            {
                "dwell_steps": dwell,
                "true_positive_rate": true_positive_rate,
                "false_positive_rate": false_positive_rate,
                "balanced_accuracy": (
                    true_positive_rate + (1.0 - false_positive_rate)
                )
                / 2.0,
            }
        )
    best = max(row["true_positive_rate"] for row in scored)
    selected = next(
        row for row in scored if math.isclose(row["true_positive_rate"], best)
    )
    return {
        "schema": "habit_causal_dwell_contract_v1",
        "selection_rule": (
            "smallest_train_dwell_maximizing_reference_ready_candidate_recall;"
            "earlier_low_speed_target_sector_runs_are_rejected_by_the_"
            "separate_visual_confirmation_stage"
        ),
        "selected_dwell_steps": int(selected["dwell_steps"]),
        "selected_operating_point": selected,
        "positive_reference_run_lengths": _summary(positive_lengths),
        "earlier_false_run_lengths": _summary(negative_lengths),
        "candidate_operating_points": scored,
    }


def enumerate_causal_candidates(
    candidates: Sequence[Mapping[str, Any]],
    *,
    dwell_steps: int,
    signals: Mapping[int, EpisodeSignals],
    sector_thresholds: Mapping[str, Any],
    dump_swing_threshold: float,
    swing_speed_threshold: float,
) -> list[dict[str, Any]]:
    """Enumerate runtime-safe candidate confirmation rows in forward order."""

    result: list[dict[str, Any]] = []
    for source in candidates:
        row = json.loads(json.dumps(source))
        interval = row["dig_ready_reference_interval"]
        if (
            interval is not None
            and row["hindsight_expert_target_sector"] is not None
        ):
            episode = signals[int(row["episode_id"])]
            eligible = _target_work_sector_mask(
                episode.qpos[:, 0],
                target=str(row["hindsight_expert_target_sector"]),
                sector_thresholds=sector_thresholds,
                dump_swing_threshold=float(dump_swing_threshold),
            ) & (
                np.abs(np.asarray(episode.qvel[:, 0], dtype=np.float64))
                <= float(swing_speed_threshold)
            )
            candidate_steps = [
                run_start + int(dwell_steps) - 1
                for run_start, run_end in _true_runs(
                    eligible,
                    start=_ready_search_start(
                        episode,
                        dump_step=int(row["dump_end_step"]),
                        search_end=int(row["next_dig_entry_step"]),
                        swing_speed_threshold=float(
                            swing_speed_threshold
                        ),
                    ),
                    end=int(row["next_dig_entry_step"]),
                )
                if run_end - run_start >= int(dwell_steps)
            ]
            row["numeric_causal_candidate_steps"] = candidate_steps
            if not candidate_steps:
                row["reason_codes"].append(
                    "no_forward_numeric_candidate_reaches_train_dwell"
                )
        row["reason_codes"] = sorted(set(row["reason_codes"]))
        result.append(row)
    return result


def _ready_search_start(
    episode: EpisodeSignals,
    *,
    dump_step: int,
    search_end: int,
    swing_speed_threshold: float,
) -> int:
    """Arm ready capture only after observable return swing activation."""

    start = int(dump_step) + 1
    if int(search_end) <= start:
        return start
    active = np.flatnonzero(
        np.abs(
            np.asarray(
                episode.qvel[start:int(search_end), 0],
                dtype=np.float64,
            )
        )
        > float(swing_speed_threshold)
    )
    if active.size == 0:
        return int(search_end)
    return int(start + int(active[0]))


def extract_candidate_features(
    candidates: Sequence[Mapping[str, Any]],
    *,
    paths: Mapping[int, Path],
    extractor: FrozenResNet18FeatureExtractor,
) -> dict[tuple[int, int, str], np.ndarray]:
    requested: dict[int, set[int]] = defaultdict(set)
    for row in candidates:
        if row["dig_ready_reference_interval"] is None:
            continue
        episode_id = int(row["episode_id"])
        reference_start, reference_end = map(
            int, row["dig_ready_reference_interval"]
        )
        reference_step = reference_end - 1
        requested[episode_id].add(reference_step)
        for step in row["numeric_causal_candidate_steps"]:
            requested[episode_id].add(int(step))
        requested[episode_id].add(int(row["dump_end_step"]))
    result: dict[tuple[int, int, str], np.ndarray] = {}
    for episode_id in sorted(requested):
        indices = sorted(requested[episode_id])
        eye = extractor.extract_hdf5_eye_pair(paths[episode_id], indices)
        stick = extractor.extract_hdf5_stick_pair(paths[episode_id], indices)
        for position, step in enumerate(indices):
            result[(episode_id, step, "eye")] = eye[position]
            result[(episode_id, step, "stick")] = stick[position]
    return result


def build_visual_boundary_audit(
    candidates: Sequence[Mapping[str, Any]],
    *,
    feature_rows: Mapping[tuple[int, int, str], np.ndarray],
    null_samples: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    eligible = [
        row
        for row in candidates
        if row["dig_ready_reference_interval"] is not None
        and row["relative_intent"] in RELATIVE_INTENTS
    ]
    train = [row for row in eligible if row["split"] == "train"]
    validation = [row for row in eligible if row["split"] == "validation"]
    if not train or not validation:
        return {
            "status": "not_computable",
            "reason": "missing_train_or_validation_confirmed_boundaries",
        }, list(map(dict, candidates))

    ready_train_x = np.stack(
        [
            feature_rows[
                (
                    int(row["episode_id"]),
                    int(row["dig_ready_reference_interval"][1]) - 1,
                    "eye",
                )
            ]
            for row in train
        ]
    )
    ready_val_x = np.stack(
        [
            feature_rows[
                (
                    int(row["episode_id"]),
                    int(row["dig_ready_reference_interval"][1]) - 1,
                    "eye",
                )
            ]
            for row in validation
        ]
    )
    train_sector = [str(row["hindsight_expert_target_sector"]) for row in train]
    val_sector = [str(row["hindsight_expert_target_sector"]) for row in validation]
    sector_centroids = _fit_episode_balanced_centroids(
        ready_train_x,
        train_sector,
        [int(row["episode_id"]) for row in train],
    )
    sector_prediction = _predict_centroids(ready_val_x, sector_centroids)
    sector_metrics = _classification_metrics(val_sector, sector_prediction)
    sector_null = _label_shuffle_null(
        ready_train_x,
        train_sector,
        [int(row["episode_id"]) for row in train],
        ready_val_x,
        val_sector,
        samples=int(null_samples),
        seed=int(seed),
    )

    train_pair_x, train_binary, train_episode = _ready_dump_features(
        train,
        feature_rows,
    )
    val_pair_x, val_binary, _ = _ready_dump_features(validation, feature_rows)
    binary_centroids = _fit_episode_balanced_centroids(
        train_pair_x,
        train_binary,
        train_episode,
    )
    binary_prediction = _predict_centroids(val_pair_x, binary_centroids)
    binary_metrics = _classification_metrics(val_binary, binary_prediction)
    binary_null = _label_shuffle_null(
        train_pair_x,
        train_binary,
        train_episode,
        val_pair_x,
        val_binary,
        samples=int(null_samples),
        seed=int(seed) + 1,
    )

    right_train = [
        row for row in train if row["hindsight_expert_target_sector"] == "right"
    ]
    right_val = [
        row
        for row in validation
        if row["hindsight_expert_target_sector"] == "right"
    ]
    right_result: dict[str, Any]
    if right_train and right_val:
        rtx, rty, rte = _ready_dump_features(right_train, feature_rows)
        rvx, rvy, _ = _ready_dump_features(right_val, feature_rows)
        rcentroids = _fit_episode_balanced_centroids(rtx, rty, rte)
        rpred = _predict_centroids(rvx, rcentroids)
        right_null = _label_shuffle_null(
            rtx,
            rty,
            rte,
            rvx,
            rvy,
            samples=int(null_samples),
            seed=int(seed) + 2,
        )
        right_metrics = _classification_metrics(rvy, rpred)
        right_result = {
            "status": "computed",
            **right_metrics,
            "permutation_null_p95_balanced_accuracy": right_null,
            "wilson_lower_95": _wilson_lower(
                sum(a == b for a, b in zip(rvy, rpred)),
                len(rvy),
            ),
            "passed": _wilson_lower(
                sum(a == b for a, b in zip(rvy, rpred)),
                len(rvy),
            )
            > 0.5,
            "gate_null": "binary_chance_accuracy_0_5",
            "shuffled_label_p95_gate_eligible": False,
        }
    else:
        right_result = {
            "status": "not_computable",
            "reason": "missing_train_or_validation_right_ready",
        }
    runtime_centroids = _fit_runtime_ready_centroids(train, feature_rows)
    resolved: list[dict[str, Any]] = []
    for source in candidates:
        row = json.loads(json.dumps(source))
        reference = row["dig_ready_reference_interval"]
        if reference is not None:
            for step in row["numeric_causal_candidate_steps"]:
                feature = _ready_feature(row, int(step), feature_rows)
                if _predict_centroids(feature[None, :], runtime_centroids)[0] == "ready":
                    row["causal_confirm_step"] = int(step)
                    row["causal_confirmed"] = True
                    row["causal_confirm_matches_reference"] = bool(
                        int(reference[0]) <= int(step) < int(reference[1])
                    )
                    row["confirm_to_next_dig_s"] = float(
                        (
                            int(row["next_dig_entry_step"]) - int(step)
                        )
                        * float(row["source_dt_s"])
                    )
                    if not row["causal_confirm_matches_reference"]:
                        row["reason_codes"].append(
                            "causal_visual_confirmation_precedes_reference_ready"
                        )
                    break
            if not row["causal_confirmed"]:
                row["reason_codes"].append(
                    "no_numeric_visual_candidate_confirmed_ready"
                )
        row["reason_codes"] = sorted(set(row["reason_codes"]))
        resolved.append(row)
    report = {
        "status": "computed",
        "ready_sector_eye_pair": {
            **sector_metrics,
            "permutation_null_p95_balanced_accuracy": sector_null,
            "passed": sector_metrics["balanced_accuracy"] > sector_null,
        },
        "ready_vs_dump_eye_stick": {
            **binary_metrics,
            "permutation_null_p95_balanced_accuracy": binary_null,
            "wilson_lower_95": _wilson_lower(
                sum(a == b for a, b in zip(val_binary, binary_prediction)),
                len(val_binary),
            ),
            "passed": _wilson_lower(
                sum(a == b for a, b in zip(val_binary, binary_prediction)),
                len(val_binary),
            )
            > 0.5,
            "gate_null": "binary_chance_accuracy_0_5",
            "shuffled_label_p95_gate_eligible": False,
        },
        "right_ready_vs_dump_eye_stick": right_result,
        "ready_feature_calibration": (
            "separate_pre_dig_ready_rows_not_dig_entry_prototypes"
        ),
        "runtime_confirmation_classifier": {
            "labels": ["ready", "not_ready"],
            "features": "frozen_eye_plus_stick_pair",
            "training_positive": "reference_ready_end_minus_one",
            "training_negative": (
                "dump_end_plus_forward_numeric_candidates_outside_reference"
            ),
            "selection": "first_forward_numeric_candidate_predicted_ready",
        },
    }
    return report, resolved


def build_observation_sufficiency_audit(
    candidates: Sequence[Mapping[str, Any]],
    *,
    signals: Mapping[int, EpisodeSignals],
) -> dict[str, Any]:
    eligible = [
        row
        for row in candidates
        if row["causal_confirmed"]
        and row["causal_confirm_matches_reference"]
        and row["relative_intent"] in RELATIVE_INTENTS
    ]
    train = [row for row in eligible if row["split"] == "train"]
    validation = [row for row in eligible if row["split"] == "validation"]
    if not train or not validation:
        return {
            "schema": "habit_observation_sufficiency_audit_v1",
            "status": "not_computable",
            "reason": "missing_train_or_validation_reference_matched_cycles",
            "runtime_contract": {
                "external_lifecycle_state_required": True,
                "scripted_target_required": True,
            },
        }
    rows: list[dict[str, Any]] = []
    for seconds in HISTORY_SECONDS:
        train_x = np.stack(
            [
                _history_feature(
                    signals[int(row["episode_id"])],
                    int(row["dump_end_step"]),
                    seconds=float(seconds),
                )
                for row in train
            ]
        )
        val_x = np.stack(
            [
                _history_feature(
                    signals[int(row["episode_id"])],
                    int(row["dump_end_step"]),
                    seconds=float(seconds),
                )
                for row in validation
            ]
        )
        train_family = [
            _intent_family(str(row["relative_intent"])) for row in train
        ]
        val_family = [
            _intent_family(str(row["relative_intent"])) for row in validation
        ]
        family_prediction = _nearest_centroid_standardized(
            train_x,
            train_family,
            val_x,
        )
        train_target = [
            str(row["hindsight_expert_target_sector"]) for row in train
        ]
        val_target = [
            str(row["hindsight_expert_target_sector"]) for row in validation
        ]
        target_prediction = _nearest_centroid_standardized(
            train_x,
            train_target,
            val_x,
        )
        phase_train_x: list[np.ndarray] = []
        phase_train_y: list[str] = []
        phase_val_x: list[np.ndarray] = []
        phase_val_y: list[str] = []
        for collection, output_x, output_y in (
            (train, phase_train_x, phase_train_y),
            (validation, phase_val_x, phase_val_y),
        ):
            for row in collection:
                episode = signals[int(row["episode_id"])]
                for label, step in (
                    ("dump_end", int(row["dump_end_step"])),
                    (
                        "dig_ready",
                        int(row["dig_ready_reference_interval"][1]) - 1,
                    ),
                ):
                    output_x.append(
                        _history_feature(
                            episode,
                            step,
                            seconds=float(seconds),
                        )
                    )
                    output_y.append(label)
        phase_prediction = _nearest_centroid_standardized(
            np.stack(phase_train_x),
            phase_train_y,
            np.stack(phase_val_x),
        )
        train_action_target = np.stack(
            [
                _future_action_summary(
                    signals[int(row["episode_id"])],
                    int(row["dump_end_step"]),
                    seconds=0.5,
                )
                for row in train
            ]
        )
        val_action_target = np.stack(
            [
                _future_action_summary(
                    signals[int(row["episode_id"])],
                    int(row["dump_end_step"]),
                    seconds=0.5,
                )
                for row in validation
            ]
        )
        action_prediction = _nearest_neighbor_values(
            train_x,
            train_action_target,
            val_x,
        )
        rows.append(
            {
                "history_seconds": float(seconds),
                "habit_family": _classification_metrics(
                    val_family,
                    family_prediction,
                ),
                "target_sector": _classification_metrics(
                    val_target,
                    target_prediction,
                ),
                "phase": _classification_metrics(
                    phase_val_y,
                    phase_prediction,
                ),
                "next_action_0_5s": {
                    "mae": float(
                        np.mean(np.abs(action_prediction - val_action_target))
                    ),
                    "active_direction_agreement": _active_direction_agreement(
                        action_prediction,
                        val_action_target,
                    ),
                },
            }
        )
    best_history = min(
        rows,
        key=lambda row: (
            -float(row["phase"]["balanced_accuracy"]),
            -float(row["target_sector"]["balanced_accuracy"]),
            float(row["next_action_0_5s"]["mae"]),
            float(row["history_seconds"]),
        ),
    )
    return {
        "schema": "habit_observation_sufficiency_audit_v1",
        "decision_time": "dump_end",
        "targets": [
            "observable_phase",
            "hindsight_target_sector",
            "expert_next_action_0_5s",
        ],
        "history_results": rows,
        "best_observed_history_seconds": best_history["history_seconds"],
        "runtime_contract": {
            "external_lifecycle_state_required": True,
            "scripted_target_required": True,
            "reason": (
                "target_is_a_committed_command_and_must_not_be_inferred_from_"
                "expert_frequency_or_future_outcome"
            ),
        },
        "interpretation": (
            "diagnostic_only_fixed_script_does_not_train_a_planner_model"
        ),
    }


def build_habit_condition_support(
    candidates: Sequence[Mapping[str, Any]],
    *,
    feature_rows: Mapping[tuple[int, int, str], np.ndarray],
    signals: Mapping[int, EpisodeSignals],
    null_samples: int,
    seed: int,
) -> dict[str, Any]:
    eligible = [
        row
        for row in candidates
        if row["causal_confirmed"]
        and row["causal_confirm_matches_reference"]
        and row["relative_intent"] in RELATIVE_INTENTS
    ]
    train = [row for row in eligible if row["split"] == "train"]
    if not train:
        return {
            "schema": "habit_condition_support_v1",
            "status": "not_computable",
            "reason": "no_train_reference_matched_cycles",
            "counts": {
                "entry_count": 0,
                "validation_entry_with_supported_alternative": 0,
            },
            "entries": [],
            "unsupported_alternative_semantics": (
                "coverage_gap_not_success_or_failure"
            ),
        }
    numeric_train = np.stack(
        [
            _history_feature(
                signals[int(row["episode_id"])],
                int(row["dump_end_step"]),
                seconds=1.0,
            )
            for row in train
        ]
    )
    mean = numeric_train.mean(axis=0)
    std = numeric_train.std(axis=0)
    std[std < 1.0e-8] = 1.0

    features: list[np.ndarray] = []
    for row in eligible:
        episode_id = int(row["episode_id"])
        dump_step = int(row["dump_end_step"])
        numeric = (
            _history_feature(
                signals[episode_id],
                dump_step,
                seconds=1.0,
            )
            - mean
        ) / std
        eye = feature_rows[(episode_id, dump_step, "eye")]
        features.append(_unit(np.concatenate((numeric, eye))))
    nearest_same: list[float] = []
    for index, row in enumerate(eligible):
        if row["split"] != "train":
            continue
        distances = [
            _cosine_distance(features[index], features[other])
            for other, candidate in enumerate(eligible)
            if other != index
            and candidate["split"] == "train"
            and int(candidate["episode_id"]) != int(row["episode_id"])
            and candidate["relative_intent"] == row["relative_intent"]
            and candidate["current_sector"] == row["current_sector"]
        ]
        if distances:
            nearest_same.append(min(distances))
    if not nearest_same:
        raise ValueError("no leave-source-episode-out same-intent support")
    threshold = float(np.quantile(nearest_same, 0.95))

    entries: list[dict[str, Any]] = []
    counts = Counter()
    for index, row in enumerate(eligible):
        alternatives: dict[str, Any] = {}
        for intent in RELATIVE_INTENTS:
            if intent == row["relative_intent"]:
                continue
            try:
                target = resolve_target_sector(str(row["current_sector"]), intent)
            except ValueError:
                continue
            neighbors = [
                (
                    _cosine_distance(features[index], features[other]),
                    int(candidate["episode_id"]),
                    int(candidate["cycle_id"]),
                )
                for other, candidate in enumerate(eligible)
                if other != index
                and candidate["split"] == "train"
                and int(candidate["episode_id"]) != int(row["episode_id"])
                and candidate["current_sector"] == row["current_sector"]
                and candidate["relative_intent"] == intent
            ]
            neighbors.sort()
            nearest = None if not neighbors else neighbors[0]
            supported = bool(nearest is not None and nearest[0] <= threshold)
            alternatives[intent] = {
                "target_sector": target,
                "supported": supported,
                "nearest_distance": None if nearest is None else float(nearest[0]),
                "nearest_episode_id": None if nearest is None else int(nearest[1]),
                "nearest_cycle_id": None if nearest is None else int(nearest[2]),
                "distance_threshold": threshold,
            }
            counts["alternative_count"] += 1
            counts["supported_alternative_count"] += supported
        any_supported = any(
            bool(value["supported"]) for value in alternatives.values()
        )
        counts["entry_count"] += 1
        counts["entry_with_supported_alternative"] += any_supported
        counts[f"{row['split']}_entry_count"] += 1
        counts[f"{row['split']}_entry_with_supported_alternative"] += any_supported
        entries.append(
            {
                "episode_id": int(row["episode_id"]),
                "cycle_id": int(row["cycle_id"]),
                "split": row["split"],
                "current_sector": row["current_sector"],
                "observed_relative_intent": row["relative_intent"],
                "hindsight_expert_target_sector": row[
                    "hindsight_expert_target_sector"
                ],
                "alternatives": alternatives,
                "support_status": (
                    "supported_alternative_available"
                    if any_supported
                    else "coverage_gap"
                ),
            }
        )
    train_indices = [
        index for index, row in enumerate(eligible) if row["split"] == "train"
    ]
    validation_indices = [
        index
        for index, row in enumerate(eligible)
        if row["split"] == "validation"
    ]
    predictability: dict[str, Any]
    if validation_indices:
        train_x = np.stack([features[index] for index in train_indices])
        validation_x = np.stack([features[index] for index in validation_indices])
        train_y = [
            str(eligible[index]["hindsight_expert_target_sector"])
            for index in train_indices
        ]
        validation_y = [
            str(eligible[index]["hindsight_expert_target_sector"])
            for index in validation_indices
        ]
        train_episode = [
            int(eligible[index]["episode_id"]) for index in train_indices
        ]
        centroids = _fit_episode_balanced_centroids(
            train_x,
            train_y,
            train_episode,
        )
        state_prediction = _predict_centroids(validation_x, centroids)
        global_mode = Counter(train_y).most_common(1)[0][0]
        by_current = {
            sector: Counter(
                str(row["hindsight_expert_target_sector"])
                for row in train
                if row["current_sector"] == sector
            ).most_common(1)[0][0]
            for sector in SECTORS
            if any(row["current_sector"] == sector for row in train)
        }
        current_prediction = [
            by_current.get(str(eligible[index]["current_sector"]), global_mode)
            for index in validation_indices
        ]
        predictability = {
            "status": "computed",
            "target": "hindsight_expert_target_sector",
            "state_history_plus_eye": _classification_metrics(
                validation_y,
                state_prediction,
            ),
            "frequency_prior_accuracy": float(
                np.mean([value == global_mode for value in validation_y])
            ),
            "current_sector_prior_accuracy": float(
                np.mean(
                    [
                        actual == predicted
                        for actual, predicted in zip(
                            validation_y,
                            current_prediction,
                        )
                    ]
                )
            ),
            "shuffled_target_null_p95_balanced_accuracy": _label_shuffle_null(
                train_x,
                train_y,
                train_episode,
                validation_x,
                validation_y,
                samples=int(null_samples),
                seed=int(seed),
            ),
            "interpretation": (
                "high_prior_accuracy_is_a_condition_ignored_risk_not_policy_"
                "condition_understanding"
            ),
        }
    else:
        predictability = {
            "status": "not_computable",
            "reason": "no_validation_entries",
        }
    return {
        "schema": "habit_condition_support_v1",
        "feature_schema": (
            "dump_end_causal_history_1s_plus_frozen_eye_pair_resnet18"
        ),
        "leave_source_episode_out": True,
        "alternative_neighbor_pool": "train_only",
        "distance": "one_minus_cosine",
        "distance_threshold": threshold,
        "distance_threshold_source": (
            "train_leave_source_episode_out_same_intent_nearest_p95"
        ),
        "counts": dict(counts),
        "entries": entries,
        "habit_predictability_null_controls": predictability,
        "preregistered_policy_test": {
            "B0": "same_ACT_without_target_condition",
            "B1": "same_ACT_with_supported_correct_target_condition",
            "B2": "same_ACT_with_matched_shuffled_target_condition",
            "success_rule": (
                "paired_supported_B1_must_beat_B2_without_phase_degradation"
            ),
        },
        "unsupported_alternative_semantics": (
            "coverage_gap_not_success_or_failure"
        ),
    }


def build_habit_transition_inventory(
    candidates: Sequence[Mapping[str, Any]],
    *,
    numeric_cycles: Mapping[int, Sequence[Mapping[str, Any]]],
    metadata: Mapping[int, Mapping[str, Any]],
    split: Mapping[str, Any],
    held_out_ids: Sequence[int],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema": "habit_transition_inventory_v1",
        "habit_source": "sim_expert_demonstration_habit",
        "command_source": "unknown_not_recorded",
        "label_source": "hindsight_observable_next_dig_entry",
        "coverage_policy": (
            "natural_frequency_inventory_not_combinatorial_gate"
        ),
        "nonadjacent_policy": "diagnostic_only_not_training_or_primary_eval",
        "splits": {},
    }
    for split_name in ("train", "validation"):
        rows = [row for row in candidates if row["split"] == split_name]
        by_episode: dict[str, Any] = {}
        total = Counter()
        epoch_counts: dict[str, Counter[str]] = defaultdict(Counter)
        for episode_id in split["splits"][split_name]:
            selected = [row for row in rows if int(row["episode_id"]) == episode_id]
            counts = Counter(
                str(row["relative_intent"])
                for row in selected
                if row["relative_intent"] is not None
            )
            by_episode[str(episode_id)] = {
                "controller_epoch": metadata[episode_id]["controller_epoch"],
                "numeric_cycle_candidate_count": len(numeric_cycles[episode_id]),
                "labeled_transition_count": int(sum(counts.values())),
                "intent_counts": dict(counts),
            }
            total.update(counts)
            epoch_counts[str(metadata[episode_id]["controller_epoch"])].update(
                counts
            )
        run_lengths = _same_sector_run_lengths(rows)
        result["splits"][split_name] = {
            "episode_ids": list(map(int, split["splits"][split_name])),
            "intent_counts": dict(total),
            "intent_frequencies": {
                key: float(value / max(1, sum(total.values())))
                for key, value in sorted(total.items())
            },
            "source_episode_support": {
                intent: sum(
                    int(
                        any(
                            row["relative_intent"] == intent
                            and int(row["episode_id"]) == episode_id
                            for row in rows
                        )
                    )
                    for episode_id in split["splits"][split_name]
                )
                for intent in (*RELATIVE_INTENTS, DIAGNOSTIC_NONADJACENT)
            },
            "same_sector_run_lengths": _summary(run_lengths),
            "same_sector_run_length_values": run_lengths,
            "controller_epoch_intent_counts": {
                key: dict(value) for key, value in sorted(epoch_counts.items())
            },
            "habit_stability": {
                family: {
                    "source_episode_count": sum(
                        int(
                            any(
                                (
                                    row["relative_intent"] == "stay"
                                    if family == "stay"
                                    else row["relative_intent"]
                                    in ("step_left", "step_right")
                                )
                                and int(row["episode_id"]) == episode_id
                                for row in rows
                            )
                        )
                        for episode_id in split["splits"][split_name]
                    ),
                    "controller_epochs": sorted(
                        {
                            str(row["controller_epoch"])
                            for row in rows
                            if (
                                row["relative_intent"] == "stay"
                                if family == "stay"
                                else row["relative_intent"]
                                in ("step_left", "step_right")
                            )
                        }
                    ),
                }
                for family in ("stay", "adjacent")
            },
            "episodes": by_episode,
        }
    result["splits"]["held_out_test"] = {
        "episode_ids": list(map(int, held_out_ids)),
        "status": "locked_unread",
        "intent_counts": None,
        "same_sector_run_lengths": None,
    }
    return result


def build_scenario_candidates(
    candidates: Sequence[Mapping[str, Any]],
    *,
    signals: Mapping[int, EpisodeSignals],
    source_chains: Mapping[int, Sequence[Mapping[str, Any]]],
    transition_inventory: Mapping[str, Any],
    condition_support: Mapping[str, Any],
) -> dict[str, Any]:
    resolved_train = [
        row
        for row in candidates
        if row["split"] == "train"
        and row["causal_confirmed"]
        and row["causal_confirm_matches_reference"]
        and row["relative_intent"] in RELATIVE_INTENTS
    ]
    by_key = {
        (int(row["episode_id"]), int(row["cycle_id"])): row
        for row in resolved_train
    }
    train = [
        row
        for row in resolved_train
        if _full_cycle_source_range(row, by_key) is not None
    ]
    scenario_rows: list[dict[str, Any]] = []
    ordered_by_episode: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in train:
        ordered_by_episode[int(row["episode_id"])].append(row)
    for episode_id in ordered_by_episode:
        ordered_by_episode[episode_id].sort(key=lambda row: int(row["cycle_id"]))

    for episode_id, rows in sorted(ordered_by_episode.items()):
        index = 0
        while index < len(rows):
            row = rows[index]
            if row["relative_intent"] != "stay":
                index += 1
                continue
            run = [row]
            cursor = index + 1
            while (
                cursor < len(rows)
                and int(rows[cursor]["cycle_id"])
                == int(run[-1]["cycle_id"]) + 1
                and rows[cursor]["relative_intent"] == "stay"
                and rows[cursor]["current_sector"] == run[-1]["current_sector"]
            ):
                run.append(rows[cursor])
                cursor += 1
            if len(run) >= 2:
                scenario_rows.append(
                    _scenario_record(
                        family="repeat_same",
                        rows=run,
                        signals=signals,
                        source_chains=source_chains,
                    )
                )
            index = cursor
        for row in rows:
            if row["relative_intent"] in ("step_left", "step_right"):
                scenario_rows.append(
                    _scenario_record(
                        family="move_adjacent",
                        rows=[row],
                        signals=signals,
                        source_chains=source_chains,
                    )
                )
    episode_support = {
        "repeat_same": len(
            {
                int(row["source_episode_id"])
                for row in scenario_rows
                if row["family"] == "repeat_same"
            }
        ),
        "move_adjacent": len(
            {
                int(row["source_episode_id"])
                for row in scenario_rows
                if row["family"] == "move_adjacent"
            }
        ),
    }
    support_entries = {
        (int(row["episode_id"]), int(row["cycle_id"])): row
        for row in condition_support["entries"]
    }
    family_episode_support = {
        family: len(
            {
                int(row["source_episode_id"])
                for row in scenario_rows
                if row["family"] == family
            }
        )
        for family in ("repeat_same", "move_adjacent")
    }
    for row in scenario_rows:
        source_rows = [
            support_entries.get((int(row["source_episode_id"]), int(cycle_id)))
            for cycle_id in row["source_cycle_ids"]
        ]
        unsupported = sorted(
            {
                value["target_sector"]
                for entry in source_rows
                if entry is not None
                for value in entry["alternatives"].values()
                if not value["supported"]
            }
        )
        row["source_episode_support_for_family"] = family_episode_support[
            row["family"]
        ]
        row["unsupported_alternative_targets"] = unsupported
        row["applicable_support_scope"] = (
            "same_current_sector_train_neighbors_within_frozen_distance"
        )
    scenario_rows.sort(
        key=lambda row: (
            row["family"],
            -int(row["source_episode_support_for_family"]),
            -int(row["cycle_count"]),
            -float(row["boundary_confidence_min"]),
            -float(row["return_action_active_fraction"]),
            int(row["source_episode_id"]),
            int(row["source_cycle_ids"][0]),
        )
    )
    family_rank = Counter()
    for row in scenario_rows:
        family_rank[row["family"]] += 1
        row["representativeness_rank_within_family"] = int(
            family_rank[row["family"]]
        )
    return {
        "schema": "expert_habit_scenario_candidates_v1",
        "status": "user_review_required_not_frozen",
        "habit_source": "sim_expert_demonstration_habit",
        "planner_model_trained": False,
        "script_behavior": (
            "commit_relative_intent_at_dump_end_then_resolve_absolute_target"
        ),
        "natural_frequency_preserved": True,
        "uniform_resampling": False,
        "nonadjacent_included": False,
        "source_episode_support": episode_support,
        "transition_inventory_summary": {
            "train_intent_counts": transition_inventory["splits"]["train"][
                "intent_counts"
            ],
            "validation_intent_counts": transition_inventory["splits"][
                "validation"
            ]["intent_counts"],
        },
        "candidate_count": len(scenario_rows),
        "candidates": scenario_rows,
    }


def build_dig_ready_boundary_audit(
    candidates: Sequence[Mapping[str, Any]],
    *,
    numeric_thresholds: Mapping[str, Any],
    numeric_bootstrap: Mapping[str, Any],
    sector_thresholds: Mapping[str, Any],
    sector_bootstrap: Mapping[str, Any],
    dwell_contract: Mapping[str, Any],
    visual_audit: Mapping[str, Any],
    observation_audit: Mapping[str, Any],
) -> dict[str, Any]:
    counts = Counter()
    lags: dict[str, list[float]] = defaultdict(list)
    for row in candidates:
        counts[f"{row['split']}_candidate_count"] += 1
        if row["dig_ready_reference_interval"] is not None:
            counts[f"{row['split']}_reference_count"] += 1
        if row["causal_confirmed"]:
            counts[f"{row['split']}_confirmed_count"] += 1
            lags[str(row["split"])].append(float(row["confirm_to_next_dig_s"]))
        if row["causal_confirm_matches_reference"]:
            counts[f"{row['split']}_reference_match_count"] += 1
        for reason in row["reason_codes"]:
            counts[f"reason:{reason}"] += 1
            counts[f"{row['split']}:reason:{reason}"] += 1
    train_reference_n = counts["train_reference_count"]
    train_matches = counts["train_reference_match_count"]
    validation_n = counts["validation_reference_count"]
    validation_matches = counts["validation_reference_match_count"]
    causal_lower = _wilson_lower(validation_matches, validation_n)
    train_match_lower = _wilson_lower(train_matches, train_reference_n)
    validation_match_rate = float(validation_matches / max(1, validation_n))
    visual_status = visual_audit.get("status")
    visual_pass = bool(
        visual_status == "computed"
        and visual_audit["ready_sector_eye_pair"]["passed"]
        and visual_audit["ready_vs_dump_eye_stick"]["passed"]
        and visual_audit["right_ready_vs_dump_eye_stick"].get("passed", False)
    )
    return {
        "schema": "dig_ready_boundary_audit_v1",
        "boundary_semantics": {
            "candidate_enter": (
                "first row of a contiguous target-work-sector run whose "
                "absolute swing speed is within the train-fitted low-speed "
                "cluster"
            ),
            "causal_confirm": (
                "candidate_enter plus train-fitted dwell and frozen visual "
                "confirmation using current and past rows only"
            ),
            "runtime_future_observation_used": False,
            "offline_hindsight_used_only_for_reference_target": True,
            "dump_corridor_exclusion": (
                "swing_qpos_must_not_exceed_train_fitted_dump_threshold"
            ),
            "ready_action_requirement": (
                "no_hard_action_deadzone;action_contributes_to_activity_and_"
                "visual_candidate_history"
            ),
        },
        "numeric_thresholds": numeric_thresholds,
        "numeric_bootstrap": numeric_bootstrap,
        "sector_thresholds": sector_thresholds,
        "sector_bootstrap": sector_bootstrap,
        "causal_dwell_contract": dwell_contract,
        "causal_dwell_source_steps": int(dwell_contract["selected_dwell_steps"]),
        "causal_dwell_s": float(
            int(dwell_contract["selected_dwell_steps"])
            * float(numeric_thresholds["source_dt_s"])
        ),
        "counts": dict(counts),
        "confirm_to_next_dig_s": {
            key: _summary(value) for key, value in sorted(lags.items())
        },
        "causal_confirmation_validation_wilson_lower_95": causal_lower,
        "causal_confirmation_train_wilson_lower_95_threshold": train_match_lower,
        "causal_confirmation_validation_rate": validation_match_rate,
        "causal_confirmation_threshold_source": (
            "train_reference_match_wilson_lower_95"
        ),
        "causal_confirmation_passed": (
            validation_n > 0
            and validation_match_rate >= train_match_lower
            and counts[
                "validation:reason:"
                "causal_visual_confirmation_precedes_reference_ready"
            ]
            == 0
        ),
        "visual_audit": visual_audit,
        "observation_sufficiency": observation_audit,
        "visual_boundary_passed": visual_pass,
        "privilege_used": False,
    }


def definition_decision(
    *,
    transition_inventory: Mapping[str, Any],
    boundary_audit: Mapping[str, Any],
    condition_support: Mapping[str, Any],
) -> dict[str, Any]:
    reasons: list[str] = []
    decision = "accept"
    if not bool(boundary_audit["causal_confirmation_passed"]):
        decision = "revise_boundary"
        reasons.append("causal_dig_ready_confirmation_not_stable")
    elif not bool(boundary_audit["visual_boundary_passed"]):
        decision = "revise_observation"
        reasons.append("ready_sector_or_dump_visual_separation_not_established")
    train_support = transition_inventory["splits"]["train"][
        "source_episode_support"
    ]
    validation_support = transition_inventory["splits"]["validation"][
        "source_episode_support"
    ]
    if train_support.get("stay", 0) < 2 or (
        train_support.get("step_left", 0) + train_support.get("step_right", 0)
    ) < 2:
        if decision == "accept":
            decision = "collect_interventional_data"
        reasons.append("expert_habit_family_not_source_episode_stable")
    train_stability = transition_inventory["splits"]["train"][
        "habit_stability"
    ]
    if any(
        len(train_stability[family]["controller_epochs"]) < 2
        for family in ("stay", "adjacent")
    ):
        if decision == "accept":
            decision = "collect_interventional_data"
        reasons.append("expert_habit_family_not_cross_controller_epoch")
    if validation_support.get("stay", 0) < 1 or (
        validation_support.get("step_left", 0)
        + validation_support.get("step_right", 0)
    ) < 1:
        if decision == "accept":
            decision = "collect_interventional_data"
        reasons.append("validation_missing_habit_family")
    if (
        int(
            condition_support["counts"].get(
                "validation_entry_with_supported_alternative", 0
            )
        )
        == 0
    ):
        if decision == "accept":
            decision = "collect_interventional_data"
        reasons.append("no_supported_validation_condition_alternative")
    return {
        "schema": "habit_cycle_definition_falsification_decision_v1",
        "decision": decision,
        "reason_codes": reasons,
        "scenario_freeze_authorized": False,
        "training_authorized": False,
        "held_out_test_authorized": False,
        "next_required_action": (
            "user_review_candidate_contract_and_scenarios"
            if decision == "accept"
            else decision
        ),
        "capability_boundary": {
            "planner_model": False,
            "sim_expert_habit_only": True,
            "real_domain_generalization": False,
            "closed_loop_execution": False,
            "physical_effect": False,
        },
    }


def _target_work_sector_mask(
    swing_qpos: np.ndarray,
    *,
    target: str,
    sector_thresholds: Mapping[str, Any],
    dump_swing_threshold: float,
) -> np.ndarray:
    result = np.zeros(len(swing_qpos), dtype=bool)
    for index, value in enumerate(np.asarray(swing_qpos, dtype=np.float64)):
        label, _confidence, _boundary = classify_sector(
            float(value),
            sector_thresholds,
        )
        result[index] = (
            label == target and float(value) <= float(dump_swing_threshold)
        )
    return result


def _true_runs(
    mask: np.ndarray,
    *,
    start: int,
    end: int,
) -> list[tuple[int, int]]:
    """Return half-open true runs without inspecting rows at or after end."""

    values = np.asarray(mask, dtype=bool)
    left = max(0, int(start))
    right = min(len(values), int(end))
    runs: list[tuple[int, int]] = []
    cursor = left
    while cursor < right:
        if not bool(values[cursor]):
            cursor += 1
            continue
        run_start = cursor
        while cursor < right and bool(values[cursor]):
            cursor += 1
        runs.append((run_start, cursor))
    return runs


def _ready_feature(
    row: Mapping[str, Any],
    step: int,
    features: Mapping[tuple[int, int, str], np.ndarray],
) -> np.ndarray:
    episode_id = int(row["episode_id"])
    return _unit(
        np.concatenate(
            (
                features[(episode_id, int(step), "eye")],
                features[(episode_id, int(step), "stick")],
            )
        )
    )


def _fit_runtime_ready_centroids(
    rows: Sequence[Mapping[str, Any]],
    features: Mapping[tuple[int, int, str], np.ndarray],
) -> dict[str, np.ndarray]:
    arrays: list[np.ndarray] = []
    labels: list[str] = []
    episodes: list[int] = []
    for row in rows:
        reference_start, reference_end = map(
            int, row["dig_ready_reference_interval"]
        )
        ready_step = reference_end - 1
        episode_id = int(row["episode_id"])
        arrays.append(_ready_feature(row, ready_step, features))
        labels.append("ready")
        episodes.append(episode_id)
        negative_steps = {int(row["dump_end_step"])}
        negative_steps.update(
            int(step)
            for step in row["numeric_causal_candidate_steps"]
            if not reference_start <= int(step) < reference_end
        )
        for step in sorted(negative_steps):
            arrays.append(_ready_feature(row, step, features))
            labels.append("not_ready")
            episodes.append(episode_id)
    return _fit_episode_balanced_centroids(
        np.stack(arrays),
        labels,
        episodes,
    )


def _ready_dump_features(
    rows: Sequence[Mapping[str, Any]],
    features: Mapping[tuple[int, int, str], np.ndarray],
) -> tuple[np.ndarray, list[str], list[int]]:
    arrays: list[np.ndarray] = []
    labels: list[str] = []
    episodes: list[int] = []
    for row in rows:
        episode_id = int(row["episode_id"])
        reference_step = int(row["dig_ready_reference_interval"][1]) - 1
        for label, step in (
            ("ready", reference_step),
            ("dump", int(row["dump_end_step"])),
        ):
            arrays.append(
                _unit(
                    np.concatenate(
                        (
                            features[(episode_id, step, "eye")],
                            features[(episode_id, step, "stick")],
                        )
                    )
                )
            )
            labels.append(label)
            episodes.append(episode_id)
    return np.stack(arrays), labels, episodes


def _fit_episode_balanced_centroids(
    features: np.ndarray,
    labels: Sequence[str],
    episode_ids: Sequence[int],
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for label in sorted(set(labels)):
        per_episode = []
        for episode_id in sorted(set(episode_ids)):
            indices = [
                index
                for index, (value, source) in enumerate(zip(labels, episode_ids))
                if value == label and int(source) == int(episode_id)
            ]
            if indices:
                per_episode.append(_unit(np.mean(features[indices], axis=0)))
        if not per_episode:
            raise ValueError(f"no centroid rows for label {label!r}")
        result[label] = _unit(np.mean(per_episode, axis=0))
    return result


def _predict_centroids(
    features: np.ndarray,
    centroids: Mapping[str, np.ndarray],
) -> list[str]:
    labels = sorted(centroids)
    matrix = np.stack([centroids[label] for label in labels])
    normalized = np.stack([_unit(row) for row in features])
    scores = normalized @ matrix.T
    return [labels[int(index)] for index in np.argmax(scores, axis=1)]


def _classification_metrics(
    expected: Sequence[str],
    predicted: Sequence[str],
) -> dict[str, Any]:
    if len(expected) != len(predicted) or not expected:
        raise ValueError("classification vectors must be equal and non-empty")
    labels = sorted(set(expected))
    accuracy = float(
        np.mean([actual == estimate for actual, estimate in zip(expected, predicted)])
    )
    recalls = [
        float(
            np.mean(
                [
                    estimate == label
                    for actual, estimate in zip(expected, predicted)
                    if actual == label
                ]
            )
        )
        for label in labels
    ]
    return {
        "count": len(expected),
        "labels": labels,
        "accuracy": accuracy,
        "balanced_accuracy": float(np.mean(recalls)),
        "expected_counts": dict(Counter(expected)),
        "predicted_counts": dict(Counter(predicted)),
    }


def _label_shuffle_null(
    train_x: np.ndarray,
    train_y: Sequence[str],
    train_episode: Sequence[int],
    val_x: np.ndarray,
    val_y: Sequence[str],
    *,
    samples: int,
    seed: int,
) -> float:
    rng = np.random.default_rng(int(seed))
    values: list[float] = []
    labels = np.asarray(train_y, dtype=object)
    for _ in range(int(samples)):
        shuffled = labels.copy()
        rng.shuffle(shuffled)
        centroids = _fit_episode_balanced_centroids(
            train_x,
            shuffled.tolist(),
            train_episode,
        )
        prediction = _predict_centroids(val_x, centroids)
        values.append(
            float(_classification_metrics(val_y, prediction)["balanced_accuracy"])
        )
    return float(np.quantile(values, 0.95))


def _history_feature(
    episode: EpisodeSignals,
    end_step: int,
    *,
    seconds: float,
) -> np.ndarray:
    end = min(max(0, int(end_step)), len(episode.step_id) - 1)
    window = max(1, int(round(float(seconds) / float(episode.dt))))
    start = max(0, end - window + 1)
    values = np.concatenate(
        (
            episode.qpos[start : end + 1],
            episode.qvel[start : end + 1],
            episode.action[start : end + 1],
        ),
        axis=1,
    ).astype(np.float64)
    return np.concatenate((values[-1], values.mean(axis=0), values.std(axis=0)))


def _future_action_summary(
    episode: EpisodeSignals,
    start_step: int,
    *,
    seconds: float,
) -> np.ndarray:
    start = min(max(0, int(start_step) + 1), len(episode.step_id) - 1)
    count = max(1, int(round(float(seconds) / float(episode.dt))))
    end = min(len(episode.step_id), start + count)
    values = np.asarray(episode.action[start:end], dtype=np.float64)
    return np.concatenate((values.mean(axis=0), values.std(axis=0)))


def _nearest_neighbor_values(
    train_x: np.ndarray,
    train_values: np.ndarray,
    validation_x: np.ndarray,
) -> np.ndarray:
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std[std < 1.0e-8] = 1.0
    train = (train_x - mean) / std
    validation = (validation_x - mean) / std
    distance = ((validation[:, None, :] - train[None, :, :]) ** 2).sum(axis=2)
    indices = np.argmin(distance, axis=1)
    return np.asarray(train_values[indices], dtype=np.float64)


def _active_direction_agreement(
    predicted: np.ndarray,
    expected: np.ndarray,
    *,
    deadzone: float = 0.05,
) -> float:
    predicted_mean = np.asarray(predicted, dtype=np.float64)[:, :4]
    expected_mean = np.asarray(expected, dtype=np.float64)[:, :4]
    expected_active = np.abs(expected_mean) > float(deadzone)
    if not np.any(expected_active):
        return 1.0
    return float(
        np.mean(
            np.sign(predicted_mean[expected_active])
            == np.sign(expected_mean[expected_active])
        )
    )


def _nearest_centroid_standardized(
    train_x: np.ndarray,
    train_y: Sequence[str],
    validation_x: np.ndarray,
) -> list[str]:
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std[std < 1.0e-8] = 1.0
    train = (train_x - mean) / std
    validation = (validation_x - mean) / std
    labels = sorted(set(train_y))
    centers = np.stack(
        [
            train[
                np.asarray([value == label for value in train_y], dtype=bool)
            ].mean(axis=0)
            for label in labels
        ]
    )
    distances = ((validation[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    return [labels[int(index)] for index in np.argmin(distances, axis=1)]


def _intent_family(intent: str) -> str:
    if intent == "stay":
        return "stay"
    if intent in ("step_left", "step_right"):
        return "adjacent"
    raise ValueError(f"intent is not in primary habit scope: {intent!r}")


def _same_sector_run_lengths(rows: Sequence[Mapping[str, Any]]) -> list[int]:
    by_episode: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["relative_intent"] == "stay":
            by_episode[int(row["episode_id"])].append(row)
    lengths: list[int] = []
    for episode_rows in by_episode.values():
        episode_rows.sort(key=lambda row: int(row["cycle_id"]))
        run = 0
        previous_cycle: int | None = None
        previous_sector: str | None = None
        for row in episode_rows:
            cycle_id = int(row["cycle_id"])
            sector = str(row["current_sector"])
            if (
                previous_cycle is not None
                and cycle_id == previous_cycle + 1
                and sector == previous_sector
            ):
                run += 1
            else:
                if run:
                    lengths.append(run)
                run = 1
            previous_cycle = cycle_id
            previous_sector = sector
        if run:
            lengths.append(run)
    return lengths


def _scenario_record(
    *,
    family: str,
    rows: Sequence[Mapping[str, Any]],
    signals: Mapping[int, EpisodeSignals],
    source_chains: Mapping[int, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    episode_id = int(rows[0]["episode_id"])
    first = rows[0]
    last = rows[-1]
    start = int(first["cycle_ready_start_step"])
    end = int(last["causal_confirm_step"])
    action = signals[episode_id].action[start:end]
    active_fraction = float(np.mean(np.any(np.abs(action) > 0.05, axis=1)))
    source_identity = source_chains[episode_id][0]
    return {
        "scenario_id": (
            f"{family}:episode_{episode_id}:"
            f"cycle_{int(first['cycle_id'])}_{int(last['cycle_id'])}"
        ),
        "status": "candidate_user_review_required",
        "family": family,
        "habit_source": "sim_expert_demonstration_habit",
        "source_episode_id": episode_id,
        "source_cycle_ids": [int(row["cycle_id"]) for row in rows],
        "source_row_range": [start, end],
        "source_row_range_semantics": (
            "raw_source_step_half_open_ready_start_to_ready_end"
        ),
        "source_vds_sha256": str(source_identity["sha256"]),
        "cycle_count": len(rows),
        "current_sector": first["current_sector"],
        "relative_intents": [row["relative_intent"] for row in rows],
        "scripted_target_sectors": [
            row["hindsight_expert_target_sector"] for row in rows
        ],
        "commit_event": "dump_end",
        "command_source_if_frozen": "fixed_script",
        "historical_label_source": "hindsight_observable_next_dig_entry",
        "return_action_active_fraction": active_fraction,
        "boundary_confidence_min": float(
            min(
                min(
                    float(row["sector_evidence"]["current_confidence"]),
                    float(row["sector_evidence"]["target_confidence"]),
                )
                for row in rows
            )
        ),
        "physical_effect_validated": False,
        "privilege_used": False,
    }


def _full_cycle_source_range(
    row: Mapping[str, Any],
    by_key: Mapping[tuple[int, int], Mapping[str, Any]],
) -> list[int] | None:
    previous = by_key.get(
        (int(row["episode_id"]), int(row["cycle_id"]) - 1)
    )
    if (
        previous is None
        or not previous["causal_confirm_matches_reference"]
        or previous["hindsight_expert_target_sector"] != row["current_sector"]
    ):
        return None
    start = int(previous["causal_confirm_step"])
    end = int(row["causal_confirm_step"])
    if end <= start:
        return None
    if isinstance(row, dict):
        row["cycle_ready_start_step"] = start
        row["cycle_ready_end_step"] = end
    return [start, end]


def _summary(values: Sequence[float | int]) -> dict[str, Any] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "minimum": float(np.min(array)),
        "p25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "p75": float(np.quantile(array, 0.75)),
        "p95": float(np.quantile(array, 0.95)),
        "maximum": float(np.max(array)),
    }


def _wilson_lower(successes: int, count: int, z: float = 1.959963984540054) -> float:
    if count <= 0:
        return 0.0
    p = float(successes) / float(count)
    denominator = 1.0 + z * z / count
    center = p + z * z / (2.0 * count)
    radius = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * count)) / count)
    return float((center - radius) / denominator)


def _unit(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(array))
    if not np.isfinite(norm) or norm <= 1.0e-12:
        raise ValueError("cannot normalize zero or non-finite feature")
    return (array / norm).astype(np.float32)


def _cosine_distance(first: np.ndarray, second: np.ndarray) -> float:
    return float(1.0 - float(np.dot(_unit(first), _unit(second))))


def _split_name(split: Mapping[str, Any], episode_id: int) -> str:
    for name, ids in split["splits"].items():
        if int(episode_id) in set(map(int, ids)):
            return str(name)
    raise KeyError(f"episode {episode_id} missing from split")


def _relative_identity(identity: Mapping[str, Any], root: Path) -> dict[str, Any]:
    return {
        "path": str(Path(str(identity["path"])).resolve().relative_to(root.resolve())),
        "size_bytes": int(identity["size_bytes"]),
        "sha256": str(identity["sha256"]),
    }


def _cuda_is_available() -> bool:
    import torch

    return bool(torch.cuda.is_available())
