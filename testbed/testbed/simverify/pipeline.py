"""End-to-end M0 builder for the observable-only SimVerify data package."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import yaml

from testbed.simverify.annotations import (
    EpisodeSignals,
    annotate_numeric_cycles,
    bootstrap_numeric_thresholds,
    classify_sector,
    fit_numeric_annotation_thresholds,
    fit_sector_thresholds,
    fit_visual_sector_centroids,
    fuse_cycle_sectors,
    map_source_interval_to_target,
    unit_normalize,
)
from testbed.simverify.artifacts import (
    artifact_identity,
    write_checksums,
    write_json,
    write_jsonl,
)
from testbed.simverify.contracts import (
    DATASET_MANIFEST_SCHEMA,
    FROZEN_OUTPUT_JPEG_QUALITY,
    SOURCE_ACTION_GENERATION,
    assert_source_provenance_unchanged,
    camera_transform_contract,
    collect_hdf5_source_provenance,
    file_provenance,
    git_provenance,
    git_ref_provenance,
    sha256_file,
    state_action_time_contract,
)
from testbed.simverify.event_selector import (
    apply_event_selections,
    assess_point_selection_stability,
    bootstrap_event_selected_sector,
    event_selector_gate_report,
    fit_event_null_control,
    fit_event_selector,
    public_selector,
    refit_outer_sector_with_stability_mask,
    select_event_corpus,
    selected_sector_records,
)
from testbed.simverify.event_selector import (
    prototype_arrays as event_prototype_arrays,
)
from testbed.simverify.export import (
    materialize_sim_episode,
    select_sim_time_indices,
)
from testbed.simverify.features import (
    FrozenResNet18FeatureExtractor,
)
from testbed.simverify.gates import (
    assign_episode_splits,
    build_condition_support_index,
    cycle_condition_schema,
    gate_thresholds_contract,
    transition_inventory,
    validate_condition_materialization,
    validate_gate_contract,
)

DEFAULT_SOURCE_ROOT = Path(
    "/data/pingfan/excavator_testbed_data/"
    "yulong_v2_2_pro_full_task_four_camera_jpeg_20260717_cycle_clean_v1"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/data/pingfan/Excavator_real_stack_data/sim_observable_cycle_v1"
)
DEFAULT_RESNET18_WEIGHTS = Path(
    "/home/pingfan/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth"
)
DEFAULT_RESNET18_SHA256 = (
    "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
)
DEFAULT_SPLIT_SEED = "simverify-m0-v1:20260724"
DEFAULT_BOOTSTRAP_SEED = 20260724
BASELINE_LABEL = "refs/tags/g49-n5-live-frozen-20260723"
BASELINE_TAG_OBJECT_SHA = "5a7424e020d0528e51a5dbe5c64bd58ad5cf6e60"
BASELINE_COMMIT_SHA = "a8c9eef0c86d80e96bff1d0649c07e76ceaedfed"
EVENT_NAMES = (
    "ready_start",
    "dig_entry_proxy",
    "carry_transition_proxy",
    "dump_start_proxy",
    "dump_end_proxy",
    "ready_end",
)
EVENT_PROTOTYPE_NAME = {
    "ready_start": "ready",
    "ready_end": "ready",
    "dig_entry_proxy": "dig_entry_proxy",
    "carry_transition_proxy": "carry_transition_proxy",
    "dump_start_proxy": "dump_start_proxy",
    "dump_end_proxy": "dump_end_proxy",
}


def run_m0_pipeline(
    *,
    source_root: str | Path = DEFAULT_SOURCE_ROOT,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    repo_root: str | Path,
    weights_path: str | Path = DEFAULT_RESNET18_WEIGHTS,
    expected_weights_sha256: str = DEFAULT_RESNET18_SHA256,
    split_seed: str = DEFAULT_SPLIT_SEED,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    bootstrap_samples: int = 256,
    feature_device: str | None = None,
    feature_batch_size: int = 64,
    jpeg_quality: int = 95,
) -> dict[str, Any]:
    """Build the immutable M0 package and a physically isolated oracle audit."""

    if int(jpeg_quality) != FROZEN_OUTPUT_JPEG_QUALITY:
        raise ValueError(
            "jpeg_quality is frozen by the camera transform contract at "
            f"{FROZEN_OUTPUT_JPEG_QUALITY}"
        )
    source = Path(source_root).resolve(strict=True)
    clean_dir = source / "clean_all_vds"
    if not clean_dir.is_dir():
        raise FileNotFoundError(clean_dir)
    destination = Path(output_root).resolve(strict=False)
    if destination.exists():
        raise FileExistsError(f"immutable M0 output already exists: {destination}")
    repository = Path(repo_root).resolve(strict=True)
    repo_snapshot = git_provenance(repository)
    _require_target_repo(repo_snapshot)
    baseline = git_ref_provenance(repository, BASELINE_LABEL)
    if (
        baseline["object_sha"] != BASELINE_TAG_OBJECT_SHA
        or baseline["commit_sha"] != BASELINE_COMMIT_SHA
    ):
        raise ValueError("frozen g49-n5 baseline tag identity changed")
    weights = Path(weights_path).resolve(strict=True)
    if sha256_file(weights) != expected_weights_sha256:
        raise ValueError("frozen ResNet-18 checkpoint SHA does not match")

    episode_paths = _discover_episode_paths(clean_dir)
    # Capture every VDS wrapper and resolved backing file before reading
    # metadata, numeric observations, or image features.  All later artifacts
    # are bound to this byte snapshot and the same records are checked again
    # after annotation and materialization.
    source_chains = {
        episode_id: collect_hdf5_source_provenance(path)
        for episode_id, path in episode_paths.items()
    }
    source_inventory_rows = _source_inventory_files(source)
    source_snapshot_records = [
        row for episode_id in sorted(source_chains) for row in source_chains[episode_id]
    ] + source_inventory_rows
    episode_metadata = {
        episode_id: _read_episode_metadata(path)
        for episode_id, path in episode_paths.items()
    }
    assert_source_provenance_unchanged(source_snapshot_records)
    split = assign_episode_splits(
        {
            episode_id: str(metadata["controller_epoch"])
            for episode_id, metadata in episode_metadata.items()
        },
        seed=split_seed,
    )

    temporary = destination.parent / f".{destination.name}.tmp-{os.getpid()}"
    if temporary.exists():
        raise FileExistsError(f"temporary output already exists: {temporary}")
    temporary.mkdir(parents=True)
    (temporary / "episodes").mkdir()
    artifact_ids: list[dict[str, Any]] = []
    try:
        generated_at = datetime.now(timezone.utc).isoformat()
        source_snapshot_identity = write_json(
            temporary / "source_snapshot_manifest.json",
            {
                "schema": "simverify_source_snapshot_v1",
                "captured_before_observation_reads": True,
                "source_root": str(source),
                "episode_count": len(episode_paths),
                "episodes": [
                    {
                        "episode_id": int(episode_id),
                        "input_vds": source_chains[episode_id][0],
                        "resolved_source_chain": source_chains[episode_id],
                    }
                    for episode_id in sorted(episode_paths)
                ],
                "source_inventory_files": source_inventory_rows,
                "git": repo_snapshot,
                "generated_at": generated_at,
            },
        )
        artifact_ids.append(source_snapshot_identity)
        common_provenance = {
            "generated_at": generated_at,
            "git": repo_snapshot,
            "baseline": baseline,
            "source_root": str(source),
            "source_snapshot": _relative_identity(
                source_snapshot_identity,
                temporary,
            ),
            "evidence_scope": "recorded-observation/offline",
        }

        camera_identity = write_json(
            temporary / "camera_mapping.json",
            {
                **camera_transform_contract(),
                "provenance": common_provenance,
            },
        )
        artifact_ids.append(camera_identity)
        state_contract_identity = write_json(
            temporary / "state_action_contract.json",
            {
                **state_action_time_contract(),
                "provenance": common_provenance,
            },
        )
        artifact_ids.append(state_contract_identity)
        condition_schema_identity = write_json(
            temporary / "cycle_condition_v1.schema.json",
            {
                **cycle_condition_schema(),
                "provenance": common_provenance,
            },
        )
        artifact_ids.append(condition_schema_identity)
        split_identity = write_json(
            temporary / "split_groups.json",
            {
                **split,
                "provenance": common_provenance,
            },
        )
        artifact_ids.append(split_identity)
        privilege_contract_identity = write_json(
            temporary / "privilege_scan_contract_v1.json",
            _privilege_scan_contract(common_provenance),
        )
        artifact_ids.append(privilege_contract_identity)

        # Threshold generation is train/validation only.  Held-out observations
        # are not opened until annotation thresholds and prototypes are hashed.
        train_ids = list(map(int, split["splits"]["train"]))
        validation_ids = list(map(int, split["splits"]["validation"]))
        calibration_ids = train_ids + validation_ids
        calibration_signals = {
            episode_id: _load_episode_signals(episode_paths[episode_id])
            for episode_id in calibration_ids
        }
        numeric_thresholds = fit_numeric_annotation_thresholds(
            [calibration_signals[episode_id] for episode_id in train_ids],
            action_deadzone=0.05,
        )
        numeric_bootstrap = bootstrap_numeric_thresholds(
            [calibration_signals[episode_id] for episode_id in train_ids],
            action_deadzone=0.05,
            samples=int(bootstrap_samples),
            seed=int(bootstrap_seed),
        )
        calibration_cycles = {
            episode_id: annotate_numeric_cycles(
                calibration_signals[episode_id],
                numeric_thresholds,
            )
            for episode_id in calibration_ids
        }
        resolved_device = (
            feature_device
            if feature_device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        extractor = FrozenResNet18FeatureExtractor(
            weights,
            expected_checkpoint_sha256=expected_weights_sha256,
            device=resolved_device,
            batch_size=int(feature_batch_size),
        )
        calibration_features = _extract_cycle_features(
            extractor,
            episode_paths,
            calibration_cycles,
            episode_ids=calibration_ids,
            episode_lengths={
                episode_id: int(calibration_signals[episode_id].step_id.size)
                for episode_id in calibration_ids
            },
            chunk_rows=int(feature_batch_size),
        )
        feature_input_identity = write_json(
            temporary / "annotation_feature_input_manifest_v1.json",
            _feature_input_manifest(
                calibration_features,
                episode_ids=calibration_ids,
                chunk_rows=int(feature_batch_size),
                extractor_provenance=extractor.provenance,
                common_provenance=common_provenance,
            ),
        )
        artifact_ids.append(feature_input_identity)
        fitted_event_selector = fit_event_selector(
            calibration_cycles,
            calibration_features,
            train_draw=train_ids,
            validation_draw=validation_ids,
        )
        event_null_control = fit_event_null_control(
            fitted_event_selector,
            calibration_cycles,
            calibration_features,
            validation_ids=validation_ids,
            replicates=int(bootstrap_samples),
            seed=int(bootstrap_seed),
        )
        point_event_selections = select_event_corpus(
            fitted_event_selector,
            calibration_cycles,
            calibration_features,
            episode_ids=calibration_ids,
        )
        outer_bootstrap = bootstrap_event_selected_sector(
            calibration_cycles,
            calibration_signals,
            calibration_features,
            train_ids=train_ids,
            validation_ids=validation_ids,
            point_selector=fitted_event_selector,
            point_selections=point_event_selections,
            samples=int(bootstrap_samples),
            seed=int(bootstrap_seed),
        )
        point_stability_assessment = assess_point_selection_stability(
            fitted_event_selector,
            point_event_selections,
            event_null_control,
            outer_bootstrap,
        )

        event_selector_prototype_path = (
            temporary / "annotation_event_selector_prototypes_v1.npz"
        )
        _write_prototypes(
            event_selector_prototype_path,
            event_prototype_arrays(fitted_event_selector),
        )
        event_selector_prototype_identity = artifact_identity(
            event_selector_prototype_path
        )
        artifact_ids.append(event_selector_prototype_identity)
        event_selector_payload = public_selector(fitted_event_selector)
        event_selector_payload.update(
            {
                "status": ("core_frozen_before_point_stability_mask_and_sector_gate"),
                "fit_episode_ids": {
                    "train": train_ids,
                    "validation": validation_ids,
                },
                "bootstrap_scope": (
                    "conditional_on_frozen_numeric_candidate_intervals;"
                    "numeric_threshold_bootstrap_is_independent_preceding_gate"
                ),
                "permutation_null": event_null_control,
                "source_episode_outer_bootstrap": {
                    key: value
                    for key, value in outer_bootstrap.items()
                    if key != "selection_stability" and not key.startswith("_")
                },
                "prototype_artifact": _relative_identity(
                    event_selector_prototype_identity,
                    temporary,
                ),
                "feature_extractor": extractor.provenance,
                "feature_input_manifest": _relative_identity(
                    feature_input_identity,
                    temporary,
                ),
                "held_out_observation_access_count": 0,
                "provenance": common_provenance,
            }
        )
        event_selector_identity = write_json(
            temporary / "annotation_event_selector_v1.json",
            event_selector_payload,
        )
        artifact_ids.append(event_selector_identity)
        event_selection_pre_gate_identity = write_json(
            temporary / "annotation_event_selections_pre_gate_v1.json",
            {
                **point_event_selections,
                "status": "point_selection_before_event_selector_gate",
                "selector": _relative_identity(
                    event_selector_identity,
                    temporary,
                ),
                "selection_stability": outer_bootstrap.get(
                    "selection_stability",
                    {},
                ),
                "point_stability_assessment": point_stability_assessment,
                "held_out_observation_access_count": 0,
                "provenance": common_provenance,
            },
        )
        artifact_ids.append(event_selection_pre_gate_identity)
        refit_outer_sector_with_stability_mask(
            outer_bootstrap,
            point_stability_assessment,
        )
        sector_outer_identity = write_json(
            temporary / "annotation_event_selected_sector_bootstrap_v1.json",
            {
                "schema": ("observable_event_selected_sector_bootstrap_artifact_v1"),
                "status": "frozen_point_stability_mask_applied",
                "selector": _relative_identity(
                    event_selector_identity,
                    temporary,
                ),
                "point_selection_and_stability_mask": _relative_identity(
                    event_selection_pre_gate_identity,
                    temporary,
                ),
                "source_episode_outer_bootstrap": {
                    key: value
                    for key, value in outer_bootstrap.items()
                    if key != "selection_stability" and not key.startswith("_")
                },
                "held_out_observation_access_count": 0,
                "provenance": common_provenance,
            },
        )
        artifact_ids.append(sector_outer_identity)
        selector_gate_report = event_selector_gate_report(
            fitted_event_selector,
            event_null_control,
            outer_bootstrap,
            point_event_selections,
            point_stability_assessment,
        )
        selector_gate_identity = write_json(
            temporary / "annotation_event_selector_gate_report.json",
            {
                **selector_gate_report,
                "selector_sha256": event_selector_identity["sha256"],
                "point_selection_sha256": event_selection_pre_gate_identity["sha256"],
                "sector_outer_bootstrap_sha256": sector_outer_identity["sha256"],
                "provenance": common_provenance,
            },
        )
        artifact_ids.append(selector_gate_identity)
        if not selector_gate_report["passed"]:
            raise RuntimeError(
                "observable visual event selector is unstable: "
                + ",".join(selector_gate_report["failure_reasons"])
            )

        apply_event_selections(
            calibration_cycles,
            point_event_selections,
            stability=outer_bootstrap["selection_stability"],
            stability_assessment=point_stability_assessment,
            selector=fitted_event_selector,
            selector_sha256=event_selector_identity["sha256"],
            episode_ids=calibration_ids,
        )
        _rebuild_sector_evidence_after_visual_selection(
            calibration_cycles,
            calibration_signals,
            episode_ids=calibration_ids,
        )
        event_selection_identity = write_json(
            temporary / "annotation_event_selections_v1.json",
            {
                **point_event_selections,
                "status": "event_selector_gate_passed_applied",
                "selector": _relative_identity(
                    event_selector_identity,
                    temporary,
                ),
                "sector_outer_bootstrap": _relative_identity(
                    sector_outer_identity,
                    temporary,
                ),
                "selection_stability": outer_bootstrap["selection_stability"],
                "held_out_observation_access_count": 0,
                "provenance": common_provenance,
            },
        )
        artifact_ids.append(event_selection_identity)

        sector_thresholds = fit_sector_thresholds(
            selected_sector_records(
                point_event_selections,
                calibration_cycles,
                calibration_signals,
                episode_draw=train_ids,
            )
        )
        sector_bootstrap = outer_bootstrap.get("sector")
        if sector_bootstrap is None:
            raise RuntimeError(
                "event-selected sector outer bootstrap has no successful sample"
            )
        sector_boundary_low = np.asarray(
            sector_bootstrap["boundaries"]["p02_5"],
            dtype=np.float64,
        )
        sector_boundary_high = np.asarray(
            sector_bootstrap["boundaries"]["p97_5"],
            dtype=np.float64,
        )
        sector_thresholds["boundary_review_margin"] = float(
            np.max((sector_boundary_high - sector_boundary_low) / 2.0)
        )
        sector_thresholds["boundary_review_margin_source"] = (
            "maximum_full_selector_source_episode_outer_bootstrap_"
            "boundary_ci95_half_width"
        )
        assert_source_provenance_unchanged(source_snapshot_records)
        annotation_gate_report = _annotation_bootstrap_gate_report(
            numeric_thresholds,
            numeric_bootstrap,
            sector_thresholds,
            sector_bootstrap,
        )
        annotation_gate_identity = write_json(
            temporary / "annotation_bootstrap_gate_report.json",
            {
                **annotation_gate_report,
                "fit_episode_ids": {
                    "train": train_ids,
                    "validation": validation_ids,
                },
                "held_out_observation_access_count": 0,
                "cycle_candidate_count": {
                    str(episode_id): len(calibration_cycles[episode_id])
                    for episode_id in calibration_ids
                },
                "reason_counts": dict(
                    sorted(
                        Counter(
                            reason
                            for episode_id in calibration_ids
                            for record in calibration_cycles[episode_id]
                            for reason in record["quality"]["reason_codes"]
                        ).items()
                    )
                ),
                "numeric_thresholds": numeric_thresholds,
                "numeric_episode_bootstrap": numeric_bootstrap,
                "sector_thresholds": sector_thresholds,
                "sector_episode_bootstrap": sector_bootstrap,
                "sector_bootstrap_scope": (
                    "full_event_prototype_refit_interval_reselection_"
                    "point_stability_mask_reapplication_"
                    "local_event_order_recheck_then_sector_refit"
                ),
                "event_selector_sha256": event_selector_identity["sha256"],
                "event_selections_sha256": event_selection_identity["sha256"],
                "event_selector_gate_sha256": selector_gate_identity["sha256"],
                "sector_outer_bootstrap": _relative_identity(
                    sector_outer_identity,
                    temporary,
                ),
                "provenance": common_provenance,
            },
        )
        artifact_ids.append(annotation_gate_identity)
        if not annotation_gate_report["passed"]:
            raise RuntimeError(str(annotation_gate_report["failure_reason"]))

        sector_visual_calibration = _fit_sector_visual_calibration(
            calibration_cycles,
            calibration_features,
            sector_thresholds=sector_thresholds,
            train_ids=train_ids,
            validation_ids=validation_ids,
            seed=int(bootstrap_seed),
            null_replicates=int(bootstrap_samples),
        )
        _require_sector_visual_identifiability(sector_visual_calibration)
        visual_calibration = {
            "prototype_arrays": {
                **sector_visual_calibration["prototype_arrays"],
                **event_prototype_arrays(fitted_event_selector),
            },
            "sector_centroids": sector_visual_calibration["sector_centroids"],
            "sector": sector_visual_calibration["sector"],
            "events": {
                **public_selector(fitted_event_selector),
                "selector_artifact_sha256": event_selector_identity["sha256"],
                "selection_artifact_sha256": event_selection_identity["sha256"],
            },
        }
        prototype_path = temporary / "annotation_feature_prototypes_v1.npz"
        _write_prototypes(
            prototype_path,
            visual_calibration["prototype_arrays"],
        )
        prototype_identity = artifact_identity(prototype_path)
        artifact_ids.append(prototype_identity)

        annotation_threshold_payload = {
            "schema": "observable_annotation_thresholds_v1",
            "status": "frozen",
            "fit_splits": ["train", "validation"],
            "held_out_observation_access_before_freeze": 0,
            "held_out_annotation_application": "locked_until_gate_thresholds_v1",
            "numeric": numeric_thresholds,
            "numeric_episode_bootstrap": numeric_bootstrap,
            "sector": sector_thresholds,
            "sector_episode_bootstrap": sector_bootstrap,
            "visual_sector": visual_calibration["sector"],
            "visual_events": visual_calibration["events"],
            "visual_event_interval_selector": {
                "artifact": _relative_identity(
                    event_selector_identity,
                    temporary,
                ),
                "selections": _relative_identity(
                    event_selection_identity,
                    temporary,
                ),
                "gate_report": _relative_identity(
                    selector_gate_identity,
                    temporary,
                ),
                "sector_outer_bootstrap": _relative_identity(
                    sector_outer_identity,
                    temporary,
                ),
            },
            "feature_extractor": extractor.provenance,
            "prototype_artifact": _relative_identity(
                prototype_identity,
                temporary,
            ),
            "provenance": common_provenance,
        }
        thresholds_identity = write_json(
            temporary / "annotation_thresholds_v1.json",
            annotation_threshold_payload,
        )
        artifact_ids.append(thresholds_identity)

        # Held-out episodes stay unavailable to annotation, visual evaluation,
        # condition support, and transition scoring until the later finite
        # gate_thresholds_v1 artifact is frozen. M0 may only perform the
        # parameter-free export/integrity QC required by the data contract.
        held_out_ids = list(map(int, split["splits"]["held_out_test"]))
        held_out_signals = {
            episode_id: _load_episode_signals(episode_paths[episode_id])
            for episode_id in held_out_ids
        }
        all_signals = {**calibration_signals, **held_out_signals}
        fused_records = _fuse_all_annotations(
            calibration_cycles,
            calibration_features,
            sector_thresholds=sector_thresholds,
            visual_calibration=visual_calibration,
            split=split,
            episode_paths=episode_paths,
            all_signals=all_signals,
        )
        _attach_target_time_provenance(fused_records, all_signals)

        annotation_identity = write_jsonl(
            temporary / "cycle_annotations.jsonl",
            fused_records,
        )
        artifact_ids.append(annotation_identity)
        review_records = [
            record
            for record in fused_records
            if record["quality"]["status"] != "accepted"
        ]
        review_identity = write_jsonl(
            temporary / "review_queue.jsonl",
            review_records,
        )
        artifact_ids.append(review_identity)

        annotation_manifest_payload = _annotation_manifest(
            fused_records,
            split=split,
            thresholds_identity=thresholds_identity,
            prototype_identity=prototype_identity,
            extractor_provenance=extractor.provenance,
            visual_calibration=visual_calibration,
            common_provenance=common_provenance,
        )
        annotation_manifest_identity = write_json(
            temporary / "annotation_manifest.json",
            annotation_manifest_payload,
        )
        artifact_ids.append(annotation_manifest_identity)

        transition_payload = transition_inventory(
            fused_records,
            split,
            locked_splits=("held_out_test",),
        )
        transition_payload["provenance"] = common_provenance
        transition_identity = write_json(
            temporary / "transition_inventory.json",
            transition_payload,
        )
        artifact_ids.append(transition_identity)

        support_entries = _condition_support_entries(
            fused_records,
            all_features=calibration_features,
            all_signals=all_signals,
            split=split,
            train_ids=train_ids,
        )
        support_payload = build_condition_support_index(
            support_entries,
            split=split,
        )
        support_payload["provenance"] = common_provenance
        support_identity = write_json(
            temporary / "condition_support_index.json",
            support_payload,
        )
        artifact_ids.append(support_identity)

        gate_payload = gate_thresholds_contract(
            split_manifest_sha256=split_identity["sha256"],
            annotation_manifest_sha256=annotation_manifest_identity["sha256"],
            bootstrap_replicates=int(bootstrap_samples),
            bootstrap_seed=int(bootstrap_seed),
        )
        validate_gate_contract(gate_payload)
        gate_payload["provenance"] = common_provenance
        gate_identity = write_json(
            temporary / "gate_thresholds_contract_v1.json",
            gate_payload,
        )
        artifact_ids.append(gate_identity)

        records_by_episode: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for record in fused_records:
            records_by_episode[int(record["episode_id"])].append(record)

        materialization_rows: list[dict[str, Any]] = []
        source_rows: list[dict[str, Any]] = []
        privilege_rows: list[dict[str, Any]] = []
        episode_manifest_rows: list[dict[str, Any]] = []
        split_by_episode = {
            int(episode_id): split_name
            for split_name, episode_ids in split["splits"].items()
            for episode_id in episode_ids
        }
        for episode_id in sorted(episode_paths):
            signals = all_signals[episode_id]
            condition, condition_cycle_id, condition_valid = (
                _materialize_source_conditions(
                    signals,
                    records_by_episode[episode_id],
                )
            )
            validate_condition_materialization(
                condition,
                condition_cycle_id,
                condition_valid,
            )
            output_path = temporary / "episodes" / f"episode_{episode_id}.hdf5"
            result = materialize_sim_episode(
                episode_paths[episode_id],
                output_path,
                repo_root=repository,
                deadzone=0.05,
                condition_rows=condition,
                condition_cycle_id=condition_cycle_id,
                condition_valid=condition_valid,
                condition_materialized_from_sha256=annotation_identity["sha256"],
                condition_schema_sha256=condition_schema_identity["sha256"],
                jpeg_quality=int(jpeg_quality),
            )
            if result["source_chain"] != source_chains[episode_id]:
                raise RuntimeError(
                    f"source byte snapshot changed before materialization: "
                    f"episode_{episode_id}"
                )
            output_identity = result["output"]
            artifact_ids.append(output_identity)
            output_field_contract = _export_episode_field_contract(output_path)
            materialization_rows.append(
                {
                    "episode_id": episode_id,
                    "input_path": str(episode_paths[episode_id]),
                    "output_path": f"episodes/episode_{episode_id}.hdf5",
                    "source_steps": result["source_steps"],
                    "output_steps": result["output_steps"],
                    "selection": result["selection"],
                    "transition_preservation_qc": result["transition_preservation_qc"],
                    "condition_status": result["condition_status"],
                    "output_field_contract": output_field_contract,
                }
            )
            source_rows.append(
                _source_episode_manifest_row(
                    episode_id,
                    episode_paths[episode_id],
                    source_chains[episode_id],
                    episode_metadata[episode_id],
                    signals,
                )
            )
            privilege_rows.append(
                {
                    "episode_id": episode_id,
                    **result["privilege_scan"],
                }
            )
            episode_manifest_rows.append(
                {
                    "episode_id": episode_id,
                    "path": f"episodes/episode_{episode_id}.hdf5",
                    "split": split_by_episode[episode_id],
                    "steps": result["output_steps"],
                    "size_bytes": int(output_identity["size_bytes"]),
                    "sha256": str(output_identity["sha256"]),
                    "field_contract": output_field_contract,
                }
            )

        export_field_contract_identity = write_json(
            temporary / "export_field_contract.json",
            {
                "schema": "simverify_export_field_contract_v1",
                "qpos_qvel_action_domain": "sim_source_representation",
                "real_unit_mapping": None,
                "source_action_generation": SOURCE_ACTION_GENERATION,
                "global": _aggregate_export_field_contract(
                    [temporary / str(row["path"]) for row in episode_manifest_rows]
                ),
                "episodes": [
                    {
                        "episode_id": row["episode_id"],
                        "path": row["path"],
                        "field_contract": row["field_contract"],
                    }
                    for row in episode_manifest_rows
                ],
                "provenance": common_provenance,
            },
        )
        artifact_ids.append(export_field_contract_identity)
        resample_payload = _aggregate_resample_qc(
            materialization_rows,
            common_provenance=common_provenance,
        )
        resample_identity = write_json(
            temporary / "resample_20hz_qc.json",
            resample_payload,
        )
        artifact_ids.append(resample_identity)
        source_manifest_identity = write_json(
            temporary / "source_episode_manifest.json",
            {
                "schema": "simverify_source_episode_manifest_v1",
                "episode_count": len(source_rows),
                "episodes": source_rows,
                "source_inventory_files": source_inventory_rows,
                "provenance": common_provenance,
            },
        )
        artifact_ids.append(source_manifest_identity)
        privilege_report = {
            "schema": "simverify_privilege_scan_report_v1",
            "episode_count": len(privilege_rows),
            "ok": all(row["ok"] for row in privilege_rows),
            "violations": [
                {
                    "episode_id": row["episode_id"],
                    "errors": row["errors"],
                }
                for row in privilege_rows
                if not row["ok"]
            ],
            "episodes": privilege_rows,
            "policy_input_paths": [
                "image_video4",
                "image_video5",
                "image_video6",
                "image_video7",
                "qpos",
                "qvel",
                "cycle_condition_v1",
            ],
            "oracle_dependency": False,
            "provenance": common_provenance,
        }
        if not privilege_report["ok"]:
            raise RuntimeError("privilege scan failed")
        privilege_report_identity = write_json(
            temporary / "privilege_scan_report.json",
            privilege_report,
        )
        artifact_ids.append(privilege_report_identity)

        dataset_manifest = {
            "schema_version": DATASET_MANIFEST_SCHEMA,
            "export_id": destination.name,
            "stage": "M0",
            "status": "m0_artifacts_frozen_m1_import_smoke_pending",
            "evidence_scope": "recorded-observation/offline",
            "training_authorized": False,
            "held_out_test_authorized": False,
            "held_out_annotation_status": "locked_until_gate_thresholds_v1",
            "held_out_parameter_free_export_qc_only": True,
            "real_deployable": False,
            "control_candidate": False,
            "checkpoint_restriction": ("sim_state_domain_only_not_real_deployable"),
            "source_episode_count": len(episode_manifest_rows),
            "episodes": episode_manifest_rows,
            "oracle_dependency": False,
            "oracle_audit_referenced_by_main_artifacts": False,
            "artifacts": {
                Path(identity["path"]).name: _relative_identity(
                    identity,
                    temporary,
                )
                for identity in artifact_ids
                if "/episodes/" not in str(identity["path"])
            },
            "provenance": common_provenance,
        }
        assert_source_provenance_unchanged(source_snapshot_records)
        dataset_manifest_identity = write_json(
            temporary / "dataset_manifest.json",
            dataset_manifest,
        )
        artifact_ids.append(dataset_manifest_identity)
        checksums_identity = write_checksums(
            temporary,
            artifact_ids,
            path=temporary / "checksums.sha256",
        )

        # Oracle is generated after the observable sidecar, thresholds, and
        # main checksums are frozen.  It is deliberately absent from the main
        # manifest and main checksum inventory.
        _write_oracle_audit(
            temporary / "oracle_audit",
            episode_paths=episode_paths,
            records=fused_records,
            annotation_sha256=annotation_identity["sha256"],
            threshold_sha256=thresholds_identity["sha256"],
            common_provenance=common_provenance,
        )

        try:
            os.rename(temporary, destination)
        except FileExistsError:
            raise FileExistsError(
                f"immutable M0 output appeared during build: {destination}"
            ) from None
        return {
            "schema": "simverify_m0_completion_v1",
            "status": "completed",
            "output_root": str(destination),
            "dataset_manifest_sha256": dataset_manifest_identity["sha256"],
            "checksums_sha256": checksums_identity["sha256"],
            "annotation_thresholds_sha256": thresholds_identity["sha256"],
            "cycle_annotations_sha256": annotation_identity["sha256"],
            "episode_count": len(episode_manifest_rows),
            "accepted_cycle_count": sum(
                record["quality"]["status"] == "accepted" for record in fused_records
            ),
            "review_cycle_count": len(review_records),
            "evidence_scope": "recorded-observation/offline",
            "closed_loop_execution": False,
            "training_started": False,
        }
    except BaseException as exc:
        # Preserve a failed build for forensic inspection instead of deleting
        # potentially expensive partial artifacts.  It cannot be mistaken for
        # the requested immutable destination.
        if temporary.exists():
            failure_path = temporary / "BUILD_FAILED.json"
            if not failure_path.exists():
                failure_path.write_text(
                    json.dumps(
                        {
                            "schema": "simverify_m0_build_failure_v1",
                            "status": "failed",
                            "final_destination_created": False,
                            "exception_type": type(exc).__name__,
                            "exception_message": str(exc),
                            "evidence_scope": "recorded-observation/offline",
                            "training_started": False,
                            "m1_import_smoke_started": False,
                            "git": repo_snapshot,
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
        raise


def _require_target_repo(snapshot: Mapping[str, Any]) -> None:
    if not snapshot.get("git_available"):
        raise ValueError("M0 builder requires Git provenance")
    if snapshot.get("branch") != "v2.0.0-simVerify":
        raise ValueError(
            f"M0 builder requires v2.0.0-simVerify, got {snapshot.get('branch')!r}"
        )
    if bool(snapshot.get("dirty")):
        raise ValueError("M0 artifact generation requires a clean committed worktree")


def _discover_episode_paths(clean_dir: Path) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for path in clean_dir.glob("episode_*.hdf5"):
        episode_id = int(path.stem.split("_", 1)[1])
        result[episode_id] = path.resolve(strict=True)
    expected = {
        1,
        3,
        4,
        6,
        7,
        8,
        9,
        10,
        12,
        13,
        14,
        16,
        19,
        20,
        23,
        24,
        25,
        27,
        28,
        29,
        30,
        32,
        33,
        34,
    }
    if set(result) != expected:
        raise ValueError(f"clean source episode inventory changed: {sorted(result)}")
    return result


def _read_episode_metadata(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        metadata = handle["metadata"].attrs
        action_generation = _read_source_action_generation(metadata)
        if action_generation != SOURCE_ACTION_GENERATION:
            raise ValueError(f"{path}: source action-generation contract changed")
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
            "action_generation": action_generation,
            "n_steps": int(handle["action"].shape[0]),
        }


def _read_source_action_generation(
    metadata: h5py.AttributeManager,
) -> dict[str, Any]:
    try:
        record_config = yaml.safe_load(str(metadata["record_config_yaml"]))
        use_measured_dt = bool(
            record_config["teleop"]["joystick"]["response_profile"]["use_measured_dt"]
        )
    except (KeyError, TypeError, yaml.YAMLError) as exc:
        raise ValueError(
            "cannot trace response_profile.use_measured_dt from source metadata"
        ) from exc

    def values(name: str, dtype: Any) -> list[Any]:
        array = np.asarray(metadata[name], dtype=dtype).reshape(-1)
        if array.shape != (4,):
            raise ValueError(f"source metadata {name} must have four axis values")
        if np.issubdtype(array.dtype, np.floating):
            array = np.round(array, decimals=8)
        return array.tolist()

    return {
        "joystick_axis_map": values("axis_map", np.int64),
        "joystick_invert": values("invert", bool),
        "scale": values("scale", np.float64),
        "symmetric_limit": values("limit", np.float64),
        "deadzone": values("deadzone", np.float64),
        "response_profile": {
            "enabled": bool(metadata["response_profile_enabled"]),
            "attack_rate": values(
                "response_profile_attack_rate",
                np.float64,
            ),
            "release_rate": values(
                "response_profile_release_rate",
                np.float64,
            ),
            "recenter_rate": values(
                "response_profile_recenter_rate",
                np.float64,
            ),
            "exponent": values(
                "response_profile_exponent",
                np.float64,
            ),
            "use_measured_dt": use_measured_dt,
        },
    }


def _load_episode_signals(path: Path) -> EpisodeSignals:
    episode_id = int(path.stem.split("_", 1)[1])
    with h5py.File(path, "r") as handle:
        return EpisodeSignals(
            episode_id=episode_id,
            step_id=np.asarray(handle["timestamps/step_id"], dtype=np.int64),
            qpos=np.asarray(handle["observations/qpos"], dtype=np.float32),
            qvel=np.asarray(handle["observations/qvel"], dtype=np.float32),
            action=np.asarray(handle["action"], dtype=np.float32),
            dt=float(handle["metadata"].attrs["dt"]),
        )


def _bootstrap_sector_thresholds(
    cycles: Mapping[int, Sequence[Mapping[str, Any]]],
    *,
    train_ids: Sequence[int],
    samples: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    centers: list[list[float]] = []
    boundaries: list[list[float]] = []
    failures = 0
    ids = list(map(int, train_ids))
    for _ in range(samples):
        draw = rng.choice(ids, size=len(ids), replace=True)
        records = [record for episode_id in draw for record in cycles[int(episode_id)]]
        try:
            fitted = fit_sector_thresholds(records)
        except ValueError:
            failures += 1
            continue
        centers.append(fitted["cluster_centers_low_to_high"])
        boundaries.append(fitted["boundaries_low_to_high"])
    if not centers:
        raise ValueError("all sector bootstrap samples failed")

    def summarize(values: Sequence[Sequence[float]]) -> dict[str, Any]:
        array = np.asarray(values, dtype=np.float64)
        return {
            "median": np.median(array, axis=0).tolist(),
            "p02_5": np.quantile(array, 0.025, axis=0).tolist(),
            "p97_5": np.quantile(array, 0.975, axis=0).tolist(),
            "std": np.std(array, axis=0).tolist(),
        }

    return {
        "unit": "source_episode",
        "seed": int(seed),
        "requested_samples": int(samples),
        "successful_samples": len(centers),
        "failed_samples": int(failures),
        "cluster_centers": summarize(centers),
        "boundaries": summarize(boundaries),
    }


def _require_bootstrap_stability(
    numeric: Mapping[str, Any],
    numeric_bootstrap: Mapping[str, Any],
    sector: Mapping[str, Any],
    sector_bootstrap: Mapping[str, Any],
) -> None:
    if numeric_bootstrap["failed_samples"] > 0:
        failure_rate = (
            numeric_bootstrap["failed_samples"] / numeric_bootstrap["requested_samples"]
        )
        if failure_rate > 0.01:
            raise RuntimeError("numeric observable labeler bootstrap is unstable")
    dump_centers = np.asarray(
        numeric["dump_release"]["swing_cluster_centers"],
        dtype=np.float64,
    )
    dump_gap = float(np.diff(dump_centers)[0])
    dump_ci = numeric_bootstrap["dump_swing_threshold"]
    if float(dump_ci["p97_5"]) - float(dump_ci["p02_5"]) >= 0.25 * dump_gap:
        raise RuntimeError("dump-cluster threshold bootstrap is unstable")
    sector_failure_rate = int(sector_bootstrap["failed_samples"]) / int(
        sector_bootstrap["requested_samples"]
    )
    if sector_failure_rate > 0.01:
        raise RuntimeError("event-selected sector outer bootstrap is unstable")
    centers = np.asarray(
        sector["cluster_centers_low_to_high"],
        dtype=np.float64,
    )
    minimum_gap = float(np.min(np.diff(centers)))
    low = np.asarray(
        sector_bootstrap["boundaries"]["p02_5"],
        dtype=np.float64,
    )
    high = np.asarray(
        sector_bootstrap["boundaries"]["p97_5"],
        dtype=np.float64,
    )
    if np.any(high - low >= 0.25 * minimum_gap):
        raise RuntimeError("sector boundary bootstrap is unstable")


def _annotation_bootstrap_gate_report(
    numeric: Mapping[str, Any],
    numeric_bootstrap: Mapping[str, Any],
    sector: Mapping[str, Any],
    sector_bootstrap: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate the unchanged M0 annotation gate and retain its operands."""

    dump_centers = np.asarray(
        numeric["dump_release"]["swing_cluster_centers"],
        dtype=np.float64,
    )
    dump_gap = float(np.diff(dump_centers)[0])
    dump_ci = numeric_bootstrap["dump_swing_threshold"]
    dump_width = float(dump_ci["p97_5"]) - float(dump_ci["p02_5"])

    sector_centers = np.asarray(
        sector["cluster_centers_low_to_high"],
        dtype=np.float64,
    )
    minimum_sector_gap = float(np.min(np.diff(sector_centers)))
    sector_low = np.asarray(
        sector_bootstrap["boundaries"]["p02_5"],
        dtype=np.float64,
    )
    sector_high = np.asarray(
        sector_bootstrap["boundaries"]["p97_5"],
        dtype=np.float64,
    )
    sector_widths = sector_high - sector_low

    failure_reason: str | None = None
    try:
        _require_bootstrap_stability(
            numeric,
            numeric_bootstrap,
            sector,
            sector_bootstrap,
        )
    except RuntimeError as exc:
        failure_reason = str(exc)
    return {
        "schema": "simverify_annotation_bootstrap_gate_report_v1",
        "stage": "M0",
        "evidence_scope": "recorded-observation/offline",
        "passed": failure_reason is None,
        "failure_reason": failure_reason,
        "training_authorized": False,
        "m1_import_smoke_authorized": failure_reason is None,
        "criteria": {
            "bootstrap_unit": "source_episode",
            "maximum_failed_sample_rate": 0.01,
            "maximum_ci95_width_fraction_of_cluster_gap": 0.25,
            "posthoc_threshold_change_allowed": False,
        },
        "dump_threshold": {
            "cluster_gap": dump_gap,
            "boundary_ci95_width": dump_width,
            "ci95_width_to_cluster_gap": dump_width / dump_gap,
        },
        "sector_thresholds": {
            "minimum_cluster_gap": minimum_sector_gap,
            "boundary_ci95_widths": sector_widths.tolist(),
            "ci95_width_to_minimum_cluster_gap": (
                sector_widths / minimum_sector_gap
            ).tolist(),
            "outer_bootstrap_failed_sample_rate": (
                int(sector_bootstrap["failed_samples"])
                / int(sector_bootstrap["requested_samples"])
            ),
        },
    }


def _extract_cycle_features(
    extractor: FrozenResNet18FeatureExtractor,
    episode_paths: Mapping[int, Path],
    cycles: Mapping[int, Sequence[Mapping[str, Any]]],
    *,
    episode_ids: Sequence[int],
    episode_lengths: Mapping[int, int],
    chunk_rows: int,
) -> dict[tuple[int, int], dict[str, np.ndarray]]:
    """Extract deduplicated eye/stick interval features with a one-row halo.

    JPEG decoding in the extractor precedes model batching, so source indices
    are explicitly chunked here as well.  This bounds decoded-image memory and
    keeps extraction count independent of bootstrap replicate count.
    """

    if int(chunk_rows) <= 0:
        raise ValueError("feature extraction chunk_rows must be positive")
    cache: dict[tuple[int, int], dict[str, np.ndarray]] = {}
    for episode_id in episode_ids:
        count = int(episode_lengths[int(episode_id)])
        indices = sorted(
            {
                int(index)
                for cycle in cycles[episode_id]
                for event in cycle["observable_events"].values()
                if event is not None
                for index in range(
                    max(0, int(event["interval"][0]) - 1),
                    min(count, int(event["interval"][1]) + 1),
                )
            }
        )
        if not indices:
            continue
        for begin in range(0, len(indices), int(chunk_rows)):
            chunk = indices[begin : begin + int(chunk_rows)]
            eye = extractor.extract_hdf5_eye_pair(
                episode_paths[episode_id],
                chunk,
            )
            stick = extractor.extract_hdf5_stick_pair(
                episode_paths[episode_id],
                chunk,
            )
            for row, index in enumerate(chunk):
                cache[(episode_id, index)] = {
                    "eye": np.asarray(eye[row], dtype=np.float32),
                    "stick": np.asarray(stick[row], dtype=np.float32),
                }
    return cache


def _feature_input_manifest(
    features: Mapping[tuple[int, int], Mapping[str, np.ndarray]],
    *,
    episode_ids: Sequence[int],
    chunk_rows: int,
    extractor_provenance: Mapping[str, Any],
    common_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    episodes: list[dict[str, Any]] = []
    total = 0
    for episode_id in map(int, episode_ids):
        steps = sorted(
            int(step)
            for feature_episode_id, step in features
            if int(feature_episode_id) == episode_id
        )
        if not steps:
            raise ValueError(f"episode_{episode_id}: no extracted event feature rows")
        ranges: list[list[int]] = []
        start = previous = steps[0]
        for step in steps[1:]:
            if step != previous + 1:
                ranges.append([int(start), int(previous + 1)])
                start = step
            previous = step
        ranges.append([int(start), int(previous + 1)])
        encoded = np.asarray(steps, dtype="<i8").tobytes()
        episodes.append(
            {
                "episode_id": episode_id,
                "source_row_count": len(steps),
                "source_row_minimum": int(steps[0]),
                "source_row_maximum": int(steps[-1]),
                "source_row_ranges_half_open": ranges,
                "source_row_int64_le_sha256": hashlib.sha256(encoded).hexdigest(),
            }
        )
        total += len(steps)
    sample = next(iter(features.values()))
    return {
        "schema": "observable_annotation_feature_input_manifest_v1",
        "selection_rule": (
            "deduplicated_union_of_numeric_event_half_open_intervals_"
            "plus_one_source_row_halo_clipped_to_episode"
        ),
        "feature_key": ["episode_id", "source_row"],
        "feature_roles": {
            role: {
                "dimension": int(np.asarray(sample[role]).size),
                "dtype": str(np.asarray(sample[role]).dtype),
                "normalization": "l2",
            }
            for role in ("eye", "stick")
        },
        "decode_chunk_rows": int(chunk_rows),
        "inference_batch_size": int(
            extractor_provenance["configured_default_batch_size"]
        ),
        "episode_count": len(episodes),
        "total_unique_source_rows": total,
        "episodes": episodes,
        "feature_extractor": copy.deepcopy(dict(extractor_provenance)),
        "provenance": copy.deepcopy(dict(common_provenance)),
    }


def _rebuild_sector_evidence_after_visual_selection(
    cycles: Mapping[int, Sequence[dict[str, Any]]],
    signals: Mapping[int, EpisodeSignals],
    *,
    episode_ids: Sequence[int],
) -> None:
    """Rebuild current/next qpos evidence from final visual dig representatives."""

    for episode_id in episode_ids:
        episode_cycles = cycles[int(episode_id)]
        qpos = signals[int(episode_id)].qpos
        for cycle in episode_cycles:
            validity = cycle["sector_validity"]["current"]
            event = cycle["observable_events"]["dig_entry_proxy"]
            confirmed = bool(
                event is not None
                and event.get("visual_interval_selection", {}).get("status")
                == "confirmed"
                and cycle["verification"].get(
                    "visual_current_sector_order_valid",
                    False,
                )
            )
            if not bool(validity["valid"]) or not confirmed:
                validity["valid"] = False
                validity["reason_codes"] = sorted(
                    set(
                        list(validity["reason_codes"])
                        + [
                            (
                                "dig_entry_visual_interval_not_confirmed"
                                if event is None
                                or event.get(
                                    "visual_interval_selection",
                                    {},
                                ).get("status")
                                != "confirmed"
                                else "visual_current_sector_order_invalid"
                            )
                        ]
                    )
                )
                cycle["sector_observations"]["current"] = None
                cycle["numeric_sector_evidence"]["current_swing_qpos"] = None
                continue
            representative = int(event["representative_step"])
            start, end = map(int, event["interval"])
            cycle["sector_observations"]["current"] = {
                "source": "dig_entry_proxy_visual_selected",
                "interval": [start, end],
                "numeric_representative_step": int(
                    event["numeric_representative_step"]
                ),
                "representative_step": representative,
                "swing_qpos_at_representative": float(qpos[representative, 0]),
            }
            cycle["numeric_sector_evidence"]["current_swing_qpos"] = float(
                qpos[representative, 0]
            )
        for index, cycle in enumerate(episode_cycles):
            next_cycle = (
                episode_cycles[index + 1] if index + 1 < len(episode_cycles) else None
            )
            if next_cycle is not None and bool(
                next_cycle["sector_validity"]["current"]["valid"]
            ):
                cycle["sector_validity"]["next"] = copy.deepcopy(
                    next_cycle["sector_validity"]["current"]
                )
                cycle["sector_observations"]["next"] = copy.deepcopy(
                    next_cycle["sector_observations"]["current"]
                )
                cycle["numeric_sector_evidence"]["next_swing_qpos"] = float(
                    next_cycle["numeric_sector_evidence"]["current_swing_qpos"]
                )
            else:
                cycle["sector_validity"]["next"] = {
                    "valid": False,
                    "source_cycle_id": (
                        None if next_cycle is None else int(next_cycle["cycle_id"])
                    ),
                    "reason_codes": [
                        (
                            "next_cycle_not_available"
                            if next_cycle is None
                            else "next_dig_entry_visual_interval_not_confirmed"
                        )
                    ],
                }
                cycle["sector_observations"]["next"] = None
                cycle["numeric_sector_evidence"]["next_swing_qpos"] = None


def _fit_sector_visual_calibration(
    cycles: Mapping[int, Sequence[Mapping[str, Any]]],
    features: Mapping[tuple[int, int], Mapping[str, np.ndarray]],
    *,
    sector_thresholds: Mapping[str, Any],
    train_ids: Sequence[int],
    validation_ids: Sequence[int],
    seed: int,
    null_replicates: int,
) -> dict[str, Any]:
    """Fit eye-only sector confirmation after the event/sector Gate passes."""

    train_rows: list[tuple[int, str, np.ndarray]] = []
    train_seen: set[tuple[int, int]] = set()
    for episode_id in map(int, train_ids):
        for cycle in cycles[episode_id]:
            for role, numeric_key in (
                ("current", "current_swing_qpos"),
                ("next", "next_swing_qpos"),
            ):
                numeric = cycle["numeric_sector_evidence"].get(numeric_key)
                observation = cycle["sector_observations"].get(role)
                validity = cycle["sector_validity"].get(role)
                if (
                    numeric is None
                    or observation is None
                    or validity is None
                    or not bool(validity["valid"])
                ):
                    continue
                label, _confidence, boundary = classify_sector(
                    float(numeric),
                    sector_thresholds,
                )
                if boundary or label is None:
                    continue
                step = int(observation["representative_step"])
                if (episode_id, step) in train_seen:
                    continue
                feature = features.get((episode_id, step))
                if feature is not None:
                    train_rows.append((episode_id, label, feature["eye"]))
                    train_seen.add((episode_id, step))
    centroids = {
        label: np.asarray(value, dtype=np.float32)
        for label, value in fit_visual_sector_centroids(train_rows).items()
    }

    validation_rows: list[dict[str, Any]] = []
    validation_seen: set[tuple[int, int]] = set()
    for episode_id in map(int, validation_ids):
        for cycle in cycles[episode_id]:
            for role, numeric_key in (
                ("current", "current_swing_qpos"),
                ("next", "next_swing_qpos"),
            ):
                numeric = cycle["numeric_sector_evidence"].get(numeric_key)
                observation = cycle["sector_observations"].get(role)
                validity = cycle["sector_validity"].get(role)
                if (
                    numeric is None
                    or observation is None
                    or validity is None
                    or not bool(validity["valid"])
                ):
                    continue
                step = int(observation["representative_step"])
                if (episode_id, step) in validation_seen:
                    continue
                qpos_label, _confidence, boundary = classify_sector(
                    float(numeric),
                    sector_thresholds,
                )
                feature = features.get((episode_id, step))
                if boundary or qpos_label is None or feature is None:
                    continue
                normalized = unit_normalize(feature["eye"].reshape(1, -1))[0]
                scores = {
                    label: float(np.dot(normalized, centroids[label]))
                    for label in centroids
                }
                ranked = sorted(
                    scores.items(),
                    key=lambda item: (-item[1], item[0]),
                )
                validation_rows.append(
                    {
                        "episode_id": episode_id,
                        "role": role,
                        "representative_step": step,
                        "label": qpos_label,
                        "prediction": ranked[0][0],
                        "feature": normalized,
                        "true_similarity": scores[qpos_label],
                        "true_margin": scores[qpos_label]
                        - max(
                            value
                            for label, value in scores.items()
                            if label != qpos_label
                        ),
                    }
                )
                validation_seen.add((episode_id, step))
    if not validation_rows:
        raise ValueError("no validation visual-sector rows")
    correct_rows = [row for row in validation_rows if row["prediction"] == row["label"]]
    if not correct_rows:
        raise RuntimeError("visual sector validation has zero correct rows")
    minimum_similarity = float(
        np.quantile(
            [row["true_similarity"] for row in correct_rows],
            0.01,
        )
    )
    minimum_margin = max(
        0.0,
        float(
            np.quantile(
                [row["true_margin"] for row in correct_rows],
                0.01,
            )
        ),
    )
    observed_accuracy = float(
        np.mean([row["prediction"] == row["label"] for row in validation_rows])
    )
    balanced_accuracy = float(
        np.mean(
            [
                np.mean(
                    [
                        row["prediction"] == label
                        for row in validation_rows
                        if row["label"] == label
                    ]
                )
                for label in ("left", "center", "right")
                if any(row["label"] == label for row in validation_rows)
            ]
        )
    )
    rng = np.random.default_rng(int(seed))
    null_accuracy = [
        _episode_block_null_accuracy(
            validation_rows,
            rng,
            labels=("left", "center", "right"),
        )
        for _ in range(int(null_replicates))
    ]
    bootstrap = _bootstrap_sector_visual_labeler(
        train_rows=train_rows,
        validation_rows=validation_rows,
        train_ids=train_ids,
        validation_ids=validation_ids,
        samples=int(null_replicates),
        seed=int(seed),
    )
    return {
        "prototype_arrays": {
            f"sector_{key}": value for key, value in centroids.items()
        },
        "sector_centroids": centroids,
        "sector": {
            "method": "eye_pair_cosine_nearest_centroid",
            "fit_split": "train",
            "calibration_split": "validation",
            "event_selector_dependency": (
                "frozen_event_selector_gate_passed_selected_dig_rows"
            ),
            "validation_count": len(validation_rows),
            "validation_accuracy": observed_accuracy,
            "validation_balanced_accuracy": balanced_accuracy,
            "permutation_null_replicates": int(null_replicates),
            "permutation_unit": "source_episode_sector_mapping",
            "permutation_null_p95_accuracy": float(np.quantile(null_accuracy, 0.95)),
            "minimum_similarity": minimum_similarity,
            "minimum_margin": minimum_margin,
            "source_episode_bootstrap": bootstrap,
        },
    }


def _bootstrap_sector_visual_labeler(
    *,
    train_rows: Sequence[tuple[int, str, np.ndarray]],
    validation_rows: Sequence[Mapping[str, Any]],
    train_ids: Sequence[int],
    validation_ids: Sequence[int],
    samples: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed) + 41)
    accuracy: list[float] = []
    balanced_accuracy: list[float] = []
    similarity_thresholds: list[float] = []
    margin_thresholds: list[float] = []
    failures = 0
    train_episode_ids = list(map(int, train_ids))
    validation_episode_ids = list(map(int, validation_ids))
    for _ in range(int(samples)):
        train_draw = rng.choice(
            train_episode_ids,
            size=len(train_episode_ids),
            replace=True,
        ).tolist()
        validation_draw = rng.choice(
            validation_episode_ids,
            size=len(validation_episode_ids),
            replace=True,
        ).tolist()
        try:
            fitted_centroids = fit_visual_sector_centroids(
                _resample_episode_rows(
                    train_rows,
                    train_draw,
                    episode_id_at=0,
                )
            )
            sampled_validation = _resample_episode_rows(
                validation_rows,
                validation_draw,
                episode_id_at="episode_id",
            )
            evaluated: list[dict[str, Any]] = []
            for row in sampled_validation:
                feature = np.asarray(row["feature"], dtype=np.float64)
                scores = {
                    label: float(np.dot(feature, fitted_centroids[label]))
                    for label in fitted_centroids
                }
                true_label = str(row["label"])
                prediction = max(scores, key=scores.get)
                evaluated.append(
                    {
                        "label": true_label,
                        "prediction": prediction,
                        "true_similarity": scores[true_label],
                        "true_margin": scores[true_label]
                        - max(
                            value
                            for label, value in scores.items()
                            if label != true_label
                        ),
                    }
                )
            correct = [row for row in evaluated if row["prediction"] == row["label"]]
            if not correct:
                raise ValueError("bootstrap sector sample has no correct rows")
            accuracy.append(
                float(np.mean([row["prediction"] == row["label"] for row in evaluated]))
            )
            balanced_accuracy.append(
                float(
                    np.mean(
                        [
                            np.mean(
                                [
                                    row["prediction"] == label
                                    for row in evaluated
                                    if row["label"] == label
                                ]
                            )
                            for label in ("left", "center", "right")
                            if any(row["label"] == label for row in evaluated)
                        ]
                    )
                )
            )
            similarity_thresholds.append(
                float(
                    np.quantile(
                        [row["true_similarity"] for row in correct],
                        0.01,
                    )
                )
            )
            margin_thresholds.append(
                max(
                    0.0,
                    float(
                        np.quantile(
                            [row["true_margin"] for row in correct],
                            0.01,
                        )
                    ),
                )
            )
        except (KeyError, ValueError):
            failures += 1
    if not accuracy:
        raise RuntimeError("all visual-sector source-episode bootstrap samples failed")
    return {
        "unit": "source_episode",
        "seed": int(seed) + 41,
        "requested_samples": int(samples),
        "successful_samples": len(accuracy),
        "failed_samples": int(failures),
        "validation_accuracy": _bootstrap_summary(accuracy),
        "validation_balanced_accuracy": _bootstrap_summary(balanced_accuracy),
        "minimum_similarity": _bootstrap_summary(similarity_thresholds),
        "minimum_margin": _bootstrap_summary(margin_thresholds),
    }


def _require_sector_visual_identifiability(
    calibration: Mapping[str, Any],
) -> None:
    sector = calibration["sector"]
    null = float(sector["permutation_null_p95_accuracy"])
    if float(sector["validation_accuracy"]) <= null:
        raise RuntimeError(
            "observable eye-pair sector labeler is not above its null control"
        )
    bootstrap = sector["source_episode_bootstrap"]
    failure_rate = int(bootstrap["failed_samples"]) / int(
        bootstrap["requested_samples"]
    )
    if failure_rate > 0.01:
        raise RuntimeError(
            "observable visual-sector source-episode bootstrap is unstable"
        )
    if float(bootstrap["validation_accuracy"]["p02_5"]) <= null:
        raise RuntimeError(
            "visual-sector bootstrap lower bound is not above null control"
        )


def _fit_visual_calibration(
    cycles: Mapping[int, Sequence[Mapping[str, Any]]],
    features: Mapping[tuple[int, int], Mapping[str, np.ndarray]],
    *,
    sector_thresholds: Mapping[str, Any],
    train_ids: Sequence[int],
    validation_ids: Sequence[int],
    seed: int,
    null_replicates: int,
) -> dict[str, Any]:
    train_rows: list[tuple[int, str, np.ndarray]] = []
    train_seen: set[tuple[int, int]] = set()
    for episode_id in train_ids:
        for cycle in cycles[episode_id]:
            for role, numeric_key in (
                ("current", "current_swing_qpos"),
                ("next", "next_swing_qpos"),
            ):
                numeric = cycle["numeric_sector_evidence"].get(numeric_key)
                observation = cycle["sector_observations"].get(role)
                validity = cycle["sector_validity"].get(role)
                if (
                    numeric is None
                    or observation is None
                    or validity is None
                    or not bool(validity["valid"])
                ):
                    continue
                label, _confidence, boundary = classify_sector(
                    float(numeric),
                    sector_thresholds,
                )
                if boundary or label is None:
                    continue
                step = int(observation["representative_step"])
                if (episode_id, step) in train_seen:
                    continue
                feature = features.get((episode_id, step))
                if feature is not None:
                    train_rows.append((episode_id, label, feature["eye"]))
                    train_seen.add((episode_id, step))
    centroids = fit_visual_sector_centroids(train_rows)

    validation_rows: list[dict[str, Any]] = []
    validation_seen: set[tuple[int, int]] = set()
    for episode_id in validation_ids:
        for cycle in cycles[episode_id]:
            for role, numeric_key in (
                ("current", "current_swing_qpos"),
                ("next", "next_swing_qpos"),
            ):
                numeric = cycle["numeric_sector_evidence"].get(numeric_key)
                observation = cycle["sector_observations"].get(role)
                validity = cycle["sector_validity"].get(role)
                if (
                    numeric is None
                    or observation is None
                    or validity is None
                    or not bool(validity["valid"])
                ):
                    continue
                step = int(observation["representative_step"])
                if (episode_id, step) in validation_seen:
                    continue
                qpos_label, _confidence, boundary = classify_sector(
                    float(numeric),
                    sector_thresholds,
                )
                feature = features.get((episode_id, step))
                if boundary or qpos_label is None or feature is None:
                    continue
                normalized = unit_normalize(feature["eye"].reshape(1, -1))[0]
                scores = {
                    label: float(np.dot(normalized, centroids[label]))
                    for label in centroids
                }
                ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
                validation_rows.append(
                    {
                        "episode_id": episode_id,
                        "role": role,
                        "representative_step": step,
                        "label": qpos_label,
                        "prediction": ranked[0][0],
                        "feature": normalized,
                        "true_similarity": scores[qpos_label],
                        "true_margin": scores[qpos_label]
                        - max(
                            value
                            for label, value in scores.items()
                            if label != qpos_label
                        ),
                    }
                )
                validation_seen.add((episode_id, step))
    if not validation_rows:
        raise ValueError("no validation visual-sector rows")
    correct_rows = [row for row in validation_rows if row["prediction"] == row["label"]]
    if not correct_rows:
        raise RuntimeError("visual sector validation has zero correct rows")
    minimum_similarity = float(
        np.quantile(
            [row["true_similarity"] for row in correct_rows],
            0.01,
        )
    )
    minimum_margin = max(
        0.0,
        float(
            np.quantile(
                [row["true_margin"] for row in correct_rows],
                0.01,
            )
        ),
    )
    observed_accuracy = float(
        np.mean([row["prediction"] == row["label"] for row in validation_rows])
    )
    balanced_accuracy = float(
        np.mean(
            [
                np.mean(
                    [
                        row["prediction"] == label
                        for row in validation_rows
                        if row["label"] == label
                    ]
                )
                for label in ("left", "center", "right")
                if any(row["label"] == label for row in validation_rows)
            ]
        )
    )
    rng = np.random.default_rng(int(seed))
    null_accuracy = [
        _episode_block_null_accuracy(
            validation_rows,
            rng,
            labels=("left", "center", "right"),
        )
        for _ in range(null_replicates)
    ]

    event_train: dict[str, list[tuple[int, np.ndarray]]] = defaultdict(list)
    event_validation: dict[str, list[tuple[int, np.ndarray]]] = defaultdict(list)
    for split_ids, target, seen in (
        (train_ids, event_train, set()),
        (validation_ids, event_validation, set()),
    ):
        for episode_id in split_ids:
            for cycle in cycles[episode_id]:
                for event_name, event in cycle["observable_events"].items():
                    if event is None:
                        continue
                    feature = features.get(
                        (episode_id, int(event["representative_step"]))
                    )
                    prototype_name = EVENT_PROTOTYPE_NAME[event_name]
                    key = (
                        int(episode_id),
                        prototype_name,
                        int(event["representative_step"]),
                    )
                    if feature is not None and key not in seen:
                        target[prototype_name].append((episode_id, feature["four"]))
                        seen.add(key)
    event_centroids = {
        name: unit_normalize(
            np.mean(
                np.stack([feature for _episode_id, feature in rows], axis=0),
                axis=0,
            ).reshape(1, -1)
        )[0]
        for name, rows in event_train.items()
    }
    expected_event_names = set(EVENT_PROTOTYPE_NAME.values())
    if set(event_centroids) != expected_event_names:
        raise ValueError("not all event visual prototypes are identifiable")
    event_thresholds: dict[str, float] = {}
    event_margin_thresholds: dict[str, float] = {}
    event_validation_summary: dict[str, Any] = {}
    event_validation_rows: list[dict[str, Any]] = []
    for name in sorted(event_centroids):
        rows: list[dict[str, Any]] = []
        for episode_id, feature in event_validation[name]:
            normalized = unit_normalize(np.asarray(feature).reshape(1, -1))[0]
            scores = {
                candidate: float(np.dot(normalized, event_centroids[candidate]))
                for candidate in event_centroids
            }
            prediction = max(scores, key=scores.get)
            rows.append(
                {
                    "episode_id": int(episode_id),
                    "label": name,
                    "prediction": prediction,
                    "feature": normalized,
                    "true_similarity": scores[name],
                    "true_margin": scores[name]
                    - max(
                        value
                        for candidate, value in scores.items()
                        if candidate != name
                    ),
                }
            )
        if not rows:
            raise ValueError(f"no validation features for event {name}")
        correct = [row for row in rows if row["prediction"] == name]
        if not correct:
            raise RuntimeError(
                f"event visual prototype has zero correct validation rows: {name}"
            )
        similarities = [row["true_similarity"] for row in rows]
        margins = [row["true_margin"] for row in rows]
        event_thresholds[name] = float(np.quantile(similarities, 0.01))
        event_margin_thresholds[name] = max(
            0.0,
            float(np.quantile(margins, 0.01)),
        )
        event_validation_summary[name] = {
            "count": len(rows),
            "correct_count": len(correct),
            "similarity_p01": event_thresholds[name],
            "margin_p01": event_margin_thresholds[name],
            "similarity_p50": float(np.median(similarities)),
            "similarity_min": float(np.min(similarities)),
            "margin_p50": float(np.median(margins)),
        }
        event_validation_rows.extend(rows)

    event_accuracy = float(
        np.mean([row["prediction"] == row["label"] for row in event_validation_rows])
    )
    event_balanced_accuracy = float(
        np.mean(
            [
                np.mean(
                    [
                        row["prediction"] == label
                        for row in event_validation_rows
                        if row["label"] == label
                    ]
                )
                for label in sorted(event_centroids)
            ]
        )
    )
    event_null_accuracy = [
        _episode_block_null_accuracy(
            event_validation_rows,
            rng,
            labels=tuple(sorted(event_centroids)),
        )
        for _ in range(null_replicates)
    ]

    visual_bootstrap = _bootstrap_visual_labeler(
        train_rows=train_rows,
        validation_rows=validation_rows,
        event_train=event_train,
        event_validation_rows=event_validation_rows,
        event_centroids=event_centroids,
        train_ids=train_ids,
        validation_ids=validation_ids,
        samples=int(null_replicates),
        seed=int(seed),
    )
    prototype_arrays = {
        **{f"sector_{key}": value for key, value in centroids.items()},
        **{f"event_{key}": value for key, value in event_centroids.items()},
    }
    return {
        "prototype_arrays": prototype_arrays,
        "sector_centroids": centroids,
        "event_centroids": event_centroids,
        "sector": {
            "method": "eye_pair_cosine_nearest_centroid",
            "fit_split": "train",
            "calibration_split": "validation",
            "validation_count": len(validation_rows),
            "validation_accuracy": observed_accuracy,
            "validation_balanced_accuracy": balanced_accuracy,
            "permutation_null_replicates": int(null_replicates),
            "permutation_unit": "source_episode_sector_mapping",
            "permutation_null_p95_accuracy": float(np.quantile(null_accuracy, 0.95)),
            "minimum_similarity": minimum_similarity,
            "minimum_margin": minimum_margin,
            "source_episode_bootstrap": visual_bootstrap["sector"],
        },
        "events": {
            "method": "four_camera_feature_prototype_similarity",
            "fit_split": "train",
            "calibration_split": "validation",
            "prototype_thresholds": event_thresholds,
            "prototype_margin_thresholds": event_margin_thresholds,
            "validation_accuracy": event_accuracy,
            "validation_balanced_accuracy": event_balanced_accuracy,
            "permutation_null_replicates": int(null_replicates),
            "permutation_unit": "source_episode_event_mapping",
            "permutation_null_p95_accuracy": float(
                np.quantile(event_null_accuracy, 0.95)
            ),
            "validation": event_validation_summary,
            "source_episode_bootstrap": visual_bootstrap["events"],
        },
    }


def _episode_block_null_accuracy(
    rows: Sequence[Mapping[str, Any]],
    rng: np.random.Generator,
    *,
    labels: Sequence[str],
) -> float:
    ordered_labels = tuple(map(str, labels))
    mappings = {
        int(episode_id): dict(zip(ordered_labels, permutation))
        for episode_id, permutation in (
            (
                episode_id,
                _non_identity_permutation(ordered_labels, rng),
            )
            for episode_id in sorted({int(row["episode_id"]) for row in rows})
        )
    }
    return float(
        np.mean(
            [
                str(row["prediction"])
                == mappings[int(row["episode_id"])][str(row["label"])]
                for row in rows
            ]
        )
    )


def _non_identity_permutation(
    values: Sequence[str],
    rng: np.random.Generator,
) -> tuple[str, ...]:
    original = tuple(map(str, values))
    if len(original) < 2:
        raise ValueError("null mapping requires at least two labels")
    while True:
        candidate = tuple(rng.permutation(original).tolist())
        if candidate != original:
            return candidate


def _bootstrap_summary(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("cannot summarize an empty bootstrap distribution")
    return {
        "median": float(np.median(array)),
        "p02_5": float(np.quantile(array, 0.025)),
        "p97_5": float(np.quantile(array, 0.975)),
        "std": float(np.std(array)),
    }


def _resample_episode_rows(
    rows: Sequence[Any],
    draw: Sequence[int],
    *,
    episode_id_at: int | str,
) -> list[Any]:
    result: list[Any] = []
    for episode_id in draw:
        for row in rows:
            if isinstance(episode_id_at, int):
                row_episode = int(row[episode_id_at])
            else:
                row_episode = int(row[episode_id_at])
            if row_episode == int(episode_id):
                result.append(row)
    return result


def _bootstrap_visual_labeler(
    *,
    train_rows: Sequence[tuple[int, str, np.ndarray]],
    validation_rows: Sequence[Mapping[str, Any]],
    event_train: Mapping[str, Sequence[tuple[int, np.ndarray]]],
    event_validation_rows: Sequence[Mapping[str, Any]],
    event_centroids: Mapping[str, np.ndarray],
    train_ids: Sequence[int],
    validation_ids: Sequence[int],
    samples: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed) + 1)
    accuracy: list[float] = []
    balanced_accuracy: list[float] = []
    similarity_thresholds: list[float] = []
    margin_thresholds: list[float] = []
    event_thresholds: dict[str, list[float]] = defaultdict(list)
    event_margin_thresholds: dict[str, list[float]] = defaultdict(list)
    event_prototype_similarity: dict[str, list[float]] = defaultdict(list)
    event_accuracy: list[float] = []
    event_balanced_accuracy: list[float] = []
    failures = 0
    train_episode_ids = list(map(int, train_ids))
    validation_episode_ids = list(map(int, validation_ids))
    for _ in range(int(samples)):
        train_draw = rng.choice(
            train_episode_ids,
            size=len(train_episode_ids),
            replace=True,
        ).tolist()
        validation_draw = rng.choice(
            validation_episode_ids,
            size=len(validation_episode_ids),
            replace=True,
        ).tolist()
        try:
            fitted_centroids = fit_visual_sector_centroids(
                _resample_episode_rows(
                    train_rows,
                    train_draw,
                    episode_id_at=0,
                )
            )
            sampled_validation = _resample_episode_rows(
                validation_rows,
                validation_draw,
                episode_id_at="episode_id",
            )
            evaluated: list[dict[str, Any]] = []
            for row in sampled_validation:
                feature = np.asarray(row["feature"], dtype=np.float64)
                scores = {
                    label: float(np.dot(feature, fitted_centroids[label]))
                    for label in fitted_centroids
                }
                prediction = max(scores, key=scores.get)
                true_label = str(row["label"])
                evaluated.append(
                    {
                        "label": true_label,
                        "prediction": prediction,
                        "true_similarity": scores[true_label],
                        "true_margin": scores[true_label]
                        - max(
                            value
                            for label, value in scores.items()
                            if label != true_label
                        ),
                    }
                )
            correct = [row for row in evaluated if row["prediction"] == row["label"]]
            if not correct:
                raise ValueError("bootstrap sector sample has no correct rows")
            sample_accuracy = float(
                np.mean([row["prediction"] == row["label"] for row in evaluated])
            )
            sample_balanced_accuracy = float(
                np.mean(
                    [
                        np.mean(
                            [
                                row["prediction"] == label
                                for row in evaluated
                                if row["label"] == label
                            ]
                        )
                        for label in ("left", "center", "right")
                        if any(row["label"] == label for row in evaluated)
                    ]
                )
            )
            sample_similarity_threshold = float(
                np.quantile(
                    [row["true_similarity"] for row in correct],
                    0.01,
                )
            )
            sample_margin_threshold = max(
                0.0,
                float(
                    np.quantile(
                        [row["true_margin"] for row in correct],
                        0.01,
                    )
                ),
            )
            sample_event_thresholds: dict[str, float] = {}
            sample_event_margin_thresholds: dict[str, float] = {}
            sample_event_self_similarity: dict[str, float] = {}
            sample_event_centroids: dict[str, np.ndarray] = {}
            for name in sorted(event_centroids):
                sampled_train = _resample_episode_rows(
                    event_train[name],
                    train_draw,
                    episode_id_at=0,
                )
                if not sampled_train:
                    raise ValueError(f"bootstrap event sample is empty for {name}")
                centroid = unit_normalize(
                    np.mean(
                        np.stack(
                            [row[1] for row in sampled_train],
                            axis=0,
                        ),
                        axis=0,
                    ).reshape(1, -1)
                )[0]
                sample_event_centroids[name] = centroid
                sample_event_self_similarity[name] = float(
                    np.dot(centroid, event_centroids[name])
                )
            sampled_event_validation = _resample_episode_rows(
                event_validation_rows,
                validation_draw,
                episode_id_at="episode_id",
            )
            evaluated_events: list[dict[str, Any]] = []
            for row in sampled_event_validation:
                feature = np.asarray(row["feature"], dtype=np.float64)
                scores = {
                    name: float(np.dot(feature, centroid))
                    for name, centroid in sample_event_centroids.items()
                }
                true_label = str(row["label"])
                prediction = max(scores, key=scores.get)
                evaluated_events.append(
                    {
                        "label": true_label,
                        "prediction": prediction,
                        "true_similarity": scores[true_label],
                        "true_margin": scores[true_label]
                        - max(
                            value
                            for name, value in scores.items()
                            if name != true_label
                        ),
                    }
                )
            if not evaluated_events:
                raise ValueError("bootstrap event validation sample is empty")
            for name in sorted(event_centroids):
                labeled_name = [row for row in evaluated_events if row["label"] == name]
                if not labeled_name:
                    raise ValueError(f"bootstrap event sample has no rows for {name}")
                sample_event_thresholds[name] = float(
                    np.quantile(
                        [row["true_similarity"] for row in labeled_name],
                        0.01,
                    )
                )
                sample_event_margin_thresholds[name] = max(
                    0.0,
                    float(
                        np.quantile(
                            [row["true_margin"] for row in labeled_name],
                            0.01,
                        )
                    ),
                )
            sample_event_accuracy = float(
                np.mean([row["prediction"] == row["label"] for row in evaluated_events])
            )
            sample_event_balanced_accuracy = float(
                np.mean(
                    [
                        np.mean(
                            [
                                row["prediction"] == name
                                for row in evaluated_events
                                if row["label"] == name
                            ]
                        )
                        for name in sorted(event_centroids)
                    ]
                )
            )
            accuracy.append(sample_accuracy)
            balanced_accuracy.append(sample_balanced_accuracy)
            similarity_thresholds.append(sample_similarity_threshold)
            margin_thresholds.append(sample_margin_threshold)
            for name, value in sample_event_thresholds.items():
                event_thresholds[name].append(value)
            for name, value in sample_event_margin_thresholds.items():
                event_margin_thresholds[name].append(value)
            for name, value in sample_event_self_similarity.items():
                event_prototype_similarity[name].append(value)
            event_accuracy.append(sample_event_accuracy)
            event_balanced_accuracy.append(sample_event_balanced_accuracy)
        except (KeyError, ValueError):
            failures += 1
    if not accuracy:
        raise RuntimeError("all visual source-episode bootstrap samples failed")
    return {
        "sector": {
            "unit": "source_episode",
            "seed": int(seed) + 1,
            "requested_samples": int(samples),
            "successful_samples": len(accuracy),
            "failed_samples": int(failures),
            "validation_accuracy": _bootstrap_summary(accuracy),
            "validation_balanced_accuracy": _bootstrap_summary(balanced_accuracy),
            "minimum_similarity": _bootstrap_summary(similarity_thresholds),
            "minimum_margin": _bootstrap_summary(margin_thresholds),
        },
        "events": {
            "unit": "source_episode",
            "seed": int(seed) + 1,
            "requested_samples": int(samples),
            "successful_samples": len(accuracy),
            "failed_samples": int(failures),
            "validation_accuracy": _bootstrap_summary(event_accuracy),
            "validation_balanced_accuracy": _bootstrap_summary(event_balanced_accuracy),
            "prototype_thresholds": {
                name: _bootstrap_summary(values)
                for name, values in sorted(event_thresholds.items())
            },
            "prototype_margin_thresholds": {
                name: _bootstrap_summary(values)
                for name, values in sorted(event_margin_thresholds.items())
            },
            "prototype_self_similarity": {
                name: _bootstrap_summary(values)
                for name, values in sorted(event_prototype_similarity.items())
            },
        },
    }


def _require_visual_identifiability(calibration: Mapping[str, Any]) -> None:
    sector = calibration["sector"]
    if float(sector["validation_accuracy"]) <= float(
        sector["permutation_null_p95_accuracy"]
    ):
        raise RuntimeError(
            "observable eye-pair sector labeler is not above its null control"
        )
    bootstrap = sector["source_episode_bootstrap"]
    failure_rate = int(bootstrap["failed_samples"]) / int(
        bootstrap["requested_samples"]
    )
    if failure_rate > 0.01:
        raise RuntimeError(
            "observable visual labeler source-episode bootstrap is unstable"
        )
    if float(bootstrap["validation_accuracy"]["p02_5"]) <= float(
        sector["permutation_null_p95_accuracy"]
    ):
        raise RuntimeError(
            "visual-sector bootstrap lower bound is not above null control"
        )
    events = calibration["events"]
    if float(events["validation_accuracy"]) <= float(
        events["permutation_null_p95_accuracy"]
    ):
        raise RuntimeError(
            "observable event visual labeler is not above its null control"
        )
    event_bootstrap = events["source_episode_bootstrap"]
    if float(event_bootstrap["validation_accuracy"]["p02_5"]) <= float(
        events["permutation_null_p95_accuracy"]
    ):
        raise RuntimeError(
            "visual-event bootstrap lower bound is not above null control"
        )
    for name, self_similarity in event_bootstrap["prototype_self_similarity"].items():
        threshold = event_bootstrap["prototype_thresholds"][name]
        if float(self_similarity["p02_5"]) <= float(threshold["p97_5"]):
            raise RuntimeError(
                f"visual-event prototype bootstrap is unstable for {name}"
            )


def _write_prototypes(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    if path.exists():
        raise FileExistsError(path)
    with path.open("xb") as handle:
        np.savez_compressed(
            handle,
            **{
                key: np.asarray(value, dtype=np.float32)
                for key, value in arrays.items()
            },
        )


def _fuse_all_annotations(
    cycles: Mapping[int, Sequence[dict[str, Any]]],
    features: Mapping[tuple[int, int], Mapping[str, np.ndarray]],
    *,
    sector_thresholds: Mapping[str, Any],
    visual_calibration: Mapping[str, Any],
    split: Mapping[str, Any],
    episode_paths: Mapping[int, Path],
    all_signals: Mapping[int, EpisodeSignals],
) -> list[dict[str, Any]]:
    del all_signals
    split_by_episode = {
        int(episode_id): split_name
        for split_name, episode_ids in split["splits"].items()
        for episode_id in episode_ids
    }
    records: list[dict[str, Any]] = []
    for episode_id in sorted(cycles):
        for original in cycles[episode_id]:
            record = copy.deepcopy(original)
            current_observation = record["sector_observations"]["current"]
            next_observation = record["sector_observations"]["next"]
            current_feature = (
                None
                if current_observation is None
                else features.get(
                    (
                        episode_id,
                        int(current_observation["representative_step"]),
                    ),
                    {},
                ).get("eye")
            )
            next_feature = (
                None
                if next_observation is None
                else features.get(
                    (
                        episode_id,
                        int(next_observation["representative_step"]),
                    ),
                    {},
                ).get("eye")
            )
            fuse_cycle_sectors(
                record,
                sector_thresholds=sector_thresholds,
                current_eye_feature=current_feature,
                next_eye_feature=next_feature,
                visual_centroids=visual_calibration["sector_centroids"],
                visual_minimum_similarity=float(
                    visual_calibration["sector"]["minimum_similarity"]
                ),
                visual_minimum_margin=float(
                    visual_calibration["sector"]["minimum_margin"]
                ),
            )
            event_reasons: list[str] = []
            event_confidences: list[float] = []
            evaluated_event_count = 0
            for event_name, event in record["observable_events"].items():
                if event is None:
                    continue
                evaluated_event_count += 1
                prototype_name = EVENT_PROTOTYPE_NAME[event_name]
                selection = event.get("visual_interval_selection")
                if selection is None:
                    event["visual_confirmation"] = {
                        "status": "missing",
                        "prototype": prototype_name,
                        "reason": "interval_selector_not_evaluated",
                    }
                    event_reasons.append(f"{event_name}_visual_interval_not_evaluated")
                    continue
                event["visual_confirmation"] = {
                    "status": selection["status"],
                    "prototype": prototype_name,
                    "numeric_representative_step": selection[
                        "numeric_representative_step"
                    ],
                    "representative_step": selection["representative_step"],
                    "absolute_offset_steps": selection["absolute_offset_steps"],
                    "signed_offset_steps": selection["signed_offset_steps"],
                    "signed_offset_bounds": selection["signed_offset_bounds"],
                    "confidence": selection["confidence"],
                    "acceptance_rule": selection["acceptance_rule"],
                    "role_metrics": (
                        None
                        if selection["selected"] is None
                        else selection["selected"]["role_metrics"]
                    ),
                    "role_change": (
                        None
                        if selection["selected"] is None
                        else selection["selected"]["role_change"]
                    ),
                }
                if selection["status"] == "confirmed":
                    event_confidences.append(float(selection["confidence"]["joint"]))
                else:
                    event_reasons.append(f"{event_name}_visual_interval_not_confirmed")
            if event_reasons:
                record["quality"]["status"] = "ambiguous"
                record["quality"]["review_required"] = True
                record["quality"]["reason_codes"] = sorted(
                    set(record["quality"]["reason_codes"] + event_reasons)
                )
                record["policy_condition"]["vector"] = None
                record["policy_condition"]["current_sector"] = None
                record["policy_condition"]["next_ready_sector"] = None
            event_confidence = min(event_confidences) if event_confidences else 0.0
            record["quality"]["event_visual_confidence"] = event_confidence
            record["quality"]["confidence"] = min(
                float(record["quality"]["confidence"]),
                event_confidence,
            )
            record["verification"]["event_visual_confirmation_complete"] = (
                evaluated_event_count > 0
                and evaluated_event_count == len(event_confidences)
            )
            record["verification"]["visual_confirmation_complete"] = bool(
                record["verification"]["visual_confirmation_complete"]
                and record["verification"]["event_visual_confirmation_complete"]
            )
            record["annotation_id"] = (
                f"episode_{episode_id}:cycle_{int(record['cycle_id'])}"
            )
            record["source"] = {
                "episode_id": f"episode_{episode_id}",
                "input_path": str(episode_paths[episode_id]),
                "source_row_range": list(record["source_steps"]),
            }
            record["split"] = split_by_episode[episode_id]
            records.append(record)
    return records


def _attach_target_time_provenance(
    records: Sequence[dict[str, Any]],
    signals: Mapping[int, EpisodeSignals],
) -> None:
    selection_by_episode = {
        episode_id: select_sim_time_indices(
            episode.step_id,
            source_dt_s=episode.dt,
        )
        for episode_id, episode in signals.items()
    }
    for record in records:
        selection = selection_by_episode[int(record["episode_id"])]
        record["target_steps_20hz"] = map_source_interval_to_target(
            record["source_steps"],
            selection.source_indices,
        )
        for event in record["observable_events"].values():
            if event is None:
                continue
            event["target_tick_interval"] = map_source_interval_to_target(
                event["interval"],
                selection.source_indices,
            )
            representative = int(event["representative_step"])
            event["representative_target_tick"] = int(
                np.searchsorted(
                    selection.source_indices,
                    representative,
                    side="left",
                )
            )
            if "numeric_representative_step" in event:
                event["numeric_representative_target_tick"] = int(
                    np.searchsorted(
                        selection.source_indices,
                        int(event["numeric_representative_step"]),
                        side="left",
                    )
                )
        record["time_contract"] = {
            "source_time_basis": "step_id_times_metadata_dt",
            "source_dt_s": float(signals[int(record["episode_id"])].dt),
            "target_hz": 20.0,
            "action_label_offset_s": 0.0,
            "same_source_row_all_fields": True,
        }


def _condition_support_entries(
    records: Sequence[Mapping[str, Any]],
    *,
    all_features: Mapping[tuple[int, int], Mapping[str, np.ndarray]],
    all_signals: Mapping[int, EpisodeSignals],
    split: Mapping[str, Any],
    train_ids: Sequence[int],
) -> list[dict[str, Any]]:
    accepted = [
        record for record in records if record["quality"]["status"] == "accepted"
    ]
    train_states: list[np.ndarray] = []
    for record in accepted:
        if int(record["episode_id"]) not in set(map(int, train_ids)):
            continue
        episode = all_signals[int(record["episode_id"])]
        for role in ("current", "next"):
            step = int(record["sector_observations"][role]["representative_step"])
            train_states.append(
                np.concatenate((episode.qpos[step], episode.qvel[step]))
            )
    state_array = np.stack(train_states, axis=0).astype(np.float64)
    state_mean = np.mean(state_array, axis=0)
    state_std = np.maximum(np.std(state_array, axis=0), 1e-6)
    split_by_episode = {
        int(episode_id): split_name
        for split_name, episode_ids in split["splits"].items()
        for episode_id in episode_ids
    }

    entries: list[dict[str, Any]] = []
    for record in accepted:
        episode_id = int(record["episode_id"])
        episode = all_signals[episode_id]
        role_features: dict[str, np.ndarray] = {}
        for role in ("current", "next"):
            step = int(record["sector_observations"][role]["representative_step"])
            state = np.concatenate((episode.qpos[step], episode.qvel[step]))
            state_z = unit_normalize(((state - state_mean) / state_std).reshape(1, -1))[
                0
            ]
            eye = all_features[(episode_id, step)]["eye"]
            role_features[role] = unit_normalize(
                np.concatenate((eye, state_z)).reshape(1, -1)
            )[0]
        entries.append(
            {
                "episode_id": episode_id,
                "cycle_id": int(record["cycle_id"]),
                "split": split_by_episode[episode_id],
                "current_sector": record["outcome"]["actual_current_sector"],
                "next_sector": record["outcome"]["actual_next_ready_sector"],
                "current_feature": role_features["current"],
                "next_feature": role_features["next"],
            }
        )
    return entries


def _materialize_source_conditions(
    signals: EpisodeSignals,
    records: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    condition = np.zeros((signals.step_id.size, 6), dtype=np.float32)
    cycle_id = np.full(signals.step_id.size, -1, dtype=np.int64)
    valid = np.zeros(signals.step_id.size, dtype=bool)
    for record in records:
        if record["quality"]["status"] != "accepted":
            continue
        start, end = map(int, record["source_steps"])
        if end <= start:
            continue
        rows = slice(max(0, start), min(signals.step_id.size, end))
        if np.any(valid[rows]):
            raise ValueError(
                f"episode_{signals.episode_id}: accepted cycle ranges overlap"
            )
        vector = np.asarray(
            record["policy_condition"]["vector"],
            dtype=np.float32,
        )
        condition[rows] = vector
        cycle_id[rows] = int(record["cycle_id"])
        valid[rows] = True
    return condition, cycle_id, valid


def _annotation_manifest(
    records: Sequence[Mapping[str, Any]],
    *,
    split: Mapping[str, Any],
    thresholds_identity: Mapping[str, Any],
    prototype_identity: Mapping[str, Any],
    extractor_provenance: Mapping[str, Any],
    visual_calibration: Mapping[str, Any],
    common_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    quality = Counter(record["quality"]["status"] for record in records)
    reasons = Counter(
        reason for record in records for reason in record["quality"]["reason_codes"]
    )
    by_split = {
        split_name: {
            "cycle_count": sum(record["split"] == split_name for record in records),
            "accepted_count": sum(
                record["split"] == split_name
                and record["quality"]["status"] == "accepted"
                for record in records
            ),
        }
        for split_name in split["splits"]
    }
    return {
        "schema": "observable_cycle_annotation_manifest_v1",
        "evidence_scope": "recorded-observation/offline",
        "observable_inputs": [
            "four_camera_images",
            "qpos",
            "qvel",
            "action",
        ],
        "privilege_used_for_annotation": False,
        "historical_command": "unknown_not_recorded",
        "condition_source": "hindsight_outcome",
        "annotation_thresholds": {
            "path": "annotation_thresholds_v1.json",
            "sha256": thresholds_identity["sha256"],
        },
        "feature_prototypes": {
            "path": "annotation_feature_prototypes_v1.npz",
            "sha256": prototype_identity["sha256"],
        },
        "feature_extractor": extractor_provenance,
        "visual_validation": {
            "sector": visual_calibration["sector"],
            "events": visual_calibration["events"],
        },
        "counts": {
            "total": len(records),
            "by_quality": dict(sorted(quality.items())),
            "by_split": by_split,
            "review_reason_counts": dict(sorted(reasons.items())),
        },
        "capability_boundary": {
            "closed_loop_execution": False,
            "soil_contact_truth": False,
            "payload_truth": False,
            "real_domain_generalization": False,
        },
        "provenance": common_provenance,
    }


def _source_episode_manifest_row(
    episode_id: int,
    path: Path,
    source_chain: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
    signals: EpisodeSignals,
) -> dict[str, Any]:
    raw = next(
        (
            row
            for row in source_chain
            if Path(str(row["path"])).parent.name
            == "yulong_v2_2_pro_full_task_four_camera_jpeg_20260717"
        ),
        None,
    )
    return {
        "episode_id": episode_id,
        "input_vds": file_provenance(path),
        "resolved_source_chain": list(source_chain),
        "raw_source": raw,
        "metadata": dict(metadata),
        "source_step_range": [
            int(signals.step_id[0]),
            int(signals.step_id[-1]),
        ],
        "field_contract": {
            "qpos": _array_stats(signals.qpos),
            "qvel": _array_stats(signals.qvel),
            "action": _array_stats(signals.action),
        },
    }


def _array_stats(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "finite": bool(np.isfinite(array).all()),
        "axis_min": np.min(array, axis=0).tolist(),
        "axis_max": np.max(array, axis=0).tolist(),
        "axis_mean": np.mean(array, axis=0).tolist(),
        "axis_std": np.std(array, axis=0).tolist(),
        "axis_p01": np.quantile(array, 0.01, axis=0).tolist(),
        "axis_p50": np.quantile(array, 0.50, axis=0).tolist(),
        "axis_p99": np.quantile(array, 0.99, axis=0).tolist(),
    }


def _export_episode_field_contract(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        return {
            "qpos": _array_stats(
                np.asarray(handle["observations/qpos"], dtype=np.float32)
            ),
            "qvel": _array_stats(
                np.asarray(handle["observations/qvel"], dtype=np.float32)
            ),
            "action": _array_stats(np.asarray(handle["action"], dtype=np.float32)),
            "condition": {
                "shape": list(handle["conditions/cycle_condition_v1"].shape),
                "dtype": str(handle["conditions/cycle_condition_v1"].dtype),
                "valid_row_count": int(
                    np.count_nonzero(
                        np.asarray(
                            handle["conditions/valid_mask"],
                            dtype=np.uint8,
                        )
                    )
                ),
            },
        }


def _aggregate_export_field_contract(
    paths: Sequence[Path],
) -> dict[str, Any]:
    fields: dict[str, list[np.ndarray]] = {
        "qpos": [],
        "qvel": [],
        "action": [],
    }
    for path in paths:
        with h5py.File(path, "r") as handle:
            fields["qpos"].append(
                np.asarray(handle["observations/qpos"], dtype=np.float32)
            )
            fields["qvel"].append(
                np.asarray(handle["observations/qvel"], dtype=np.float32)
            )
            fields["action"].append(np.asarray(handle["action"], dtype=np.float32))
    return {
        name: _array_stats(np.concatenate(values, axis=0))
        for name, values in fields.items()
    }


def _source_inventory_files(source_root: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for name in (
        "source_manifest.json",
        "summary.json",
        "acceptance_report.json",
        "cycle_eligibility.jsonl",
        "clean_all_vds/lineage.json",
    ):
        path = source_root / name
        if path.is_file():
            result.append(file_provenance(path))
    return result


def _aggregate_resample_qc(
    rows: Sequence[Mapping[str, Any]],
    *,
    common_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    valid = sum(
        int(row["transition_preservation_qc"]["valid_segment_count"]) for row in rows
    )
    preserved = sum(
        int(row["transition_preservation_qc"]["preserved_segment_count"])
        for row in rows
    )
    missing_rows = [
        segment
        for row in rows
        for segment in row["transition_preservation_qc"]["missing_segments"]
    ]
    durable_missing = [segment for segment in missing_rows if segment["durable"]]
    max_delay = max(
        float(row["transition_preservation_qc"]["max_preserved_onset_delay_s"])
        for row in rows
    )
    return {
        "schema": "sim_20hz_full_dataset_qc_v1",
        "episode_count": len(rows),
        "source_time_basis": "step_id_times_metadata_dt",
        "wall_clock_step_ns_used": False,
        "action_label_offset_s": 0.0,
        "same_source_row_all_fields": True,
        "source_row_count": int(sum(row["source_steps"] for row in rows)),
        "output_row_count": int(sum(row["output_steps"] for row in rows)),
        "valid_action_sign_segment_count": int(valid),
        "preserved_action_sign_segment_count": int(preserved),
        "missing_action_sign_segment_count": len(missing_rows),
        "preservation_rate": float(preserved / valid) if valid else 1.0,
        "durable_min_duration_s": 0.05,
        "durable_missing_segment_count": len(durable_missing),
        "all_missing_segments_shorter_than_50ms": not durable_missing,
        "max_preserved_onset_delay_s": max_delay,
        "episodes": list(rows),
        "provenance": common_provenance,
    }


def _privilege_scan_contract(
    common_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "privilege_scan_contract_v1",
        "construction": "allowlist_from_empty_output",
        "source_fields_allowed": [
            "observations/qpos",
            "observations/qvel",
            "observations/encoded_images/{eye_left,eye_right,stick_down,stick_up}",
            "action",
            "timestamps/step_id",
        ],
        "policy_inputs_allowed": [
            "image_video4",
            "image_video5",
            "image_video6",
            "image_video7",
            "qpos",
            "qvel",
            "cycle_condition_v1",
        ],
        "training_labels_allowed": ["action"],
        "denied_source_paths": [
            "observations/env_state",
            "rewards",
            "v2/**",
            "timestamps/step_ns",
        ],
        "denied_semantics": [
            "bucket_mass",
            "soil_contact",
            "removed_depth",
            "terrain_grid",
            "bucket_tip_world",
            "planner_state",
            "planner_goal",
            "goal_tokens",
            "oracle",
            "future_action",
        ],
        "external_hdf5_links_allowed": False,
        "virtual_datasets_allowed": False,
        "oracle_dependency_allowed": False,
        "provenance": common_provenance,
    }


def _relative_identity(
    identity: Mapping[str, Any],
    root: Path,
) -> dict[str, Any]:
    path = Path(str(identity["path"])).resolve(strict=True)
    return {
        "path": str(path.relative_to(root.resolve(strict=True))),
        "size_bytes": int(identity["size_bytes"]),
        "sha256": str(identity["sha256"]),
    }


def _write_oracle_audit(
    output_dir: Path,
    *,
    episode_paths: Mapping[int, Path],
    records: Sequence[Mapping[str, Any]],
    annotation_sha256: str,
    threshold_sha256: str,
    common_provenance: Mapping[str, Any],
) -> None:
    output_dir.mkdir(parents=True)
    observable_by_episode: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        observable_by_episode[int(record["episode_id"])].append(record)
    matches: list[dict[str, Any]] = []
    episode_matching: list[dict[str, Any]] = []
    sector_map = {0: "left", 1: "center", 2: "right"}
    for episode_id in sorted(observable_by_episode):
        path = episode_paths[episode_id]
        with h5py.File(path, "r") as handle:
            oracle_start = np.asarray(
                handle["v2/cycle/start_step"],
                dtype=np.int64,
            )
            oracle_dump = np.asarray(
                handle["v2/cycle/dump_end_step"],
                dtype=np.int64,
            )
            oracle_current = np.asarray(
                handle["v2/cycle/curr_src_sector_id"],
                dtype=np.int64,
            )
            oracle_next = np.asarray(
                handle["v2/cycle/next_src_sector_id"],
                dtype=np.int64,
            )
            replay = np.asarray(
                handle["v2/cycle/cleaning_replay_candidate"],
                dtype=bool,
            )
        episode_records = sorted(
            observable_by_episode[episode_id],
            key=lambda record: int(
                record["observable_events"]["dump_end_proxy"]["representative_step"]
            ),
        )
        candidates = np.flatnonzero(replay & (oracle_dump >= 0))
        observable_dump = [
            int(record["observable_events"]["dump_end_proxy"]["representative_step"])
            for record in episode_records
        ]
        local_pairs = _monotonic_one_to_one_pairs(
            observable_dump,
            oracle_dump[candidates].astype(int).tolist(),
        )
        matched_observable: set[int] = set()
        matched_oracle: set[int] = set()
        for observable_index, candidate_index in local_pairs:
            record = episode_records[observable_index]
            dump = observable_dump[observable_index]
            oracle_index = int(candidates[candidate_index])
            matched_observable.add(observable_index)
            matched_oracle.add(oracle_index)
            current = record["outcome"]["actual_current_sector"]
            next_sector = record["outcome"]["actual_next_ready_sector"]
            matches.append(
                {
                    "episode_id": episode_id,
                    "observable_cycle_id": int(record["cycle_id"]),
                    "observable_quality": record["quality"]["status"],
                    "oracle_cycle_id": oracle_index,
                    "ready_start_error_steps": int(
                        record["observable_events"]["ready_start"][
                            "representative_step"
                        ]
                        - oracle_start[oracle_index]
                    )
                    if record["observable_events"]["ready_start"] is not None
                    else None,
                    "dump_end_error_steps": int(dump - oracle_dump[oracle_index]),
                    "current_sector_agrees": (
                        current == sector_map.get(int(oracle_current[oracle_index]))
                        if current is not None
                        else None
                    ),
                    "next_sector_agrees": (
                        next_sector == sector_map.get(int(oracle_next[oracle_index]))
                        if next_sector is not None
                        and int(oracle_next[oracle_index]) >= 0
                        else None
                    ),
                }
            )
        episode_matching.append(
            {
                "episode_id": episode_id,
                "observable_cycle_count": len(episode_records),
                "oracle_replay_candidate_count": int(candidates.size),
                "matched_pair_count": len(local_pairs),
                "observable_match_coverage": (
                    float(len(local_pairs) / len(episode_records))
                    if episode_records
                    else None
                ),
                "oracle_match_coverage": (
                    float(len(local_pairs) / candidates.size)
                    if candidates.size
                    else None
                ),
                "unmatched_observable_cycle_ids": [
                    int(record["cycle_id"])
                    for index, record in enumerate(episode_records)
                    if index not in matched_observable
                ],
                "unmatched_oracle_cycle_ids": [
                    int(index)
                    for index in candidates
                    if int(index) not in matched_oracle
                ],
            }
        )
    ready_errors = [
        abs(int(row["ready_start_error_steps"]))
        for row in matches
        if row["ready_start_error_steps"] is not None
    ]
    dump_errors = [abs(int(row["dump_end_error_steps"])) for row in matches]
    current_agreement = [
        bool(row["current_sector_agrees"])
        for row in matches
        if row["current_sector_agrees"] is not None
    ]
    next_agreement = [
        bool(row["next_sector_agrees"])
        for row in matches
        if row["next_sector_agrees"] is not None
    ]
    report = {
        "schema": "observable_vs_pact_privilege_oracle_audit_v2",
        "physical_isolation": True,
        "post_hoc_only": True,
        "exploratory_only": True,
        "gate_decision_authorized": False,
        "annotation_sha256": annotation_sha256,
        "annotation_thresholds_sha256": threshold_sha256,
        "oracle_fields_used": [
            "v2/cycle/start_step",
            "v2/cycle/dump_end_step",
            "v2/cycle/curr_src_sector_id",
            "v2/cycle/next_src_sector_id",
            "v2/cycle/cleaning_replay_candidate",
        ],
        "main_artifacts_modified_after_oracle": False,
        "matching_method": (
            "episode_local_monotonic_one_to_one_minimum_dump_step_error"
        ),
        "match_count": len(matches),
        "episode_matching": episode_matching,
        "ready_start_absolute_error_steps": _scalar_stats(ready_errors),
        "dump_end_absolute_error_steps": _scalar_stats(dump_errors),
        "current_sector_agreement": (
            float(np.mean(current_agreement)) if current_agreement else None
        ),
        "next_sector_agreement": (
            float(np.mean(next_agreement)) if next_agreement else None
        ),
        "boundary_near_observable_count": int(
            sum(
                any(
                    reason
                    in {
                        "current_qpos_sector_boundary",
                        "next_qpos_sector_boundary",
                    }
                    for reason in row["quality"]["reason_codes"]
                )
                for row in records
            )
        ),
        "observable_failure_reason_counts": dict(
            sorted(
                Counter(
                    reason
                    for row in records
                    for reason in row["quality"]["reason_codes"]
                ).items()
            )
        ),
        "matches": matches,
        "provenance": common_provenance,
    }
    report_identity = write_json(output_dir / "oracle_report.json", report)
    manifest_identity = write_json(
        output_dir / "oracle_manifest.json",
        {
            "schema": "simverify_oracle_audit_manifest_v1",
            "main_package_dependency": False,
            "can_delete_without_changing_main_import": True,
            "report": {
                "path": "oracle_report.json",
                "sha256": report_identity["sha256"],
            },
        },
    )
    write_checksums(
        output_dir,
        [report_identity, manifest_identity],
        path=output_dir / "checksums.sha256",
    )


def _monotonic_one_to_one_pairs(
    left_steps: Sequence[int],
    right_steps: Sequence[int],
) -> list[tuple[int, int]]:
    """Match the shorter ordered sequence to a subset of the longer one."""

    if not left_steps or not right_steps:
        return []
    if len(left_steps) > len(right_steps):
        swapped = _monotonic_one_to_one_pairs(right_steps, left_steps)
        return [(right, left) for left, right in swapped]
    left = np.asarray(left_steps, dtype=np.int64)
    right = np.asarray(right_steps, dtype=np.int64)
    rows = left.size
    columns = right.size
    cost = np.full((rows + 1, columns + 1), np.inf, dtype=np.float64)
    take = np.zeros((rows + 1, columns + 1), dtype=bool)
    cost[0, :] = 0.0
    for row in range(1, rows + 1):
        for column in range(1, columns + 1):
            skip_cost = cost[row, column - 1]
            match_cost = cost[row - 1, column - 1] + abs(
                int(left[row - 1]) - int(right[column - 1])
            )
            if match_cost <= skip_cost:
                cost[row, column] = match_cost
                take[row, column] = True
            else:
                cost[row, column] = skip_cost
    if not np.isfinite(cost[rows, columns]):
        raise RuntimeError("cannot construct monotonic oracle matching")
    pairs: list[tuple[int, int]] = []
    row = rows
    column = columns
    while row > 0:
        if column <= 0:
            raise RuntimeError("oracle matching backtrack failed")
        if take[row, column]:
            pairs.append((row - 1, column - 1))
            row -= 1
            column -= 1
        else:
            column -= 1
    pairs.reverse()
    return pairs


def _scalar_stats(values: Sequence[float]) -> dict[str, Any] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
    }
