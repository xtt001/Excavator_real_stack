"""Materialize observable-only simulator episodes on the frozen 20 Hz grid."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np

from testbed.simverify.contracts import (
    CAMERA_MAPPING_ID,
    CONDITION_SCHEMA_VERSION,
    EXPORT_EPISODE_SCHEMA,
    FROZEN_ACTION_LABEL_OFFSET_S,
    FROZEN_OUTPUT_JPEG_QUALITY,
    FROZEN_SOURCE_DT_S,
    FROZEN_SOURCE_HZ,
    FROZEN_TARGET_HZ,
    IMAGE_TRANSFORM_ID,
    POLICY_AXIS_ORDER,
    POLICY_CAMERA_ORDER,
    SOURCE_ACTION_ORDER,
    SOURCE_CAMERA_ORDER,
    SOURCE_QPOS_ORDER,
    SOURCE_QVEL_ORDER,
    SOURCE_TO_POLICY_CAMERA,
    STATE_ACTION_TIME_CONTRACT_ID,
    assert_source_provenance_unchanged,
    camera_transform_contract,
    collect_hdf5_source_provenance,
    file_provenance,
    git_provenance,
    scan_export_for_privilege,
    state_action_time_contract,
)


@dataclass(frozen=True)
class SimTimeSelection:
    """One-to-one mapping from 20 Hz target ticks to complete source rows."""

    source_indices: np.ndarray
    source_step_ids: np.ndarray
    source_sim_time_s: np.ndarray
    target_ticks: np.ndarray
    target_sim_time_s: np.ndarray
    selection_error_s: np.ndarray


@dataclass(frozen=True)
class _SourceEpisodeContract:
    n_steps: int
    source_dt_s: float
    step_ids: np.ndarray
    action: np.ndarray
    episode_id: str


def select_sim_time_indices(
    step_ids: np.ndarray,
    *,
    source_dt_s: float,
    target_hz: float = FROZEN_TARGET_HZ,
) -> SimTimeSelection:
    """Select the first source row not earlier than each global 20 Hz tick.

    A tiny tolerance is used only to absorb float32 serialization of the frozen
    0.02 second source ``dt``.  Every selected row is unique; a source gap large
    enough to map multiple targets to the same row fails closed for QC review.
    """

    raw_steps = np.asarray(step_ids)
    if raw_steps.ndim != 1 or raw_steps.size == 0:
        raise ValueError("step_ids must be a non-empty 1D array")
    if not np.issubdtype(raw_steps.dtype, np.integer):
        raise ValueError(f"step_ids must have integer dtype, got {raw_steps.dtype}")
    steps = raw_steps.astype(np.int64, copy=False)
    if np.any(np.diff(steps) <= 0):
        raise ValueError("step_ids must be strictly increasing")

    dt = float(source_dt_s)
    hz = float(target_hz)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"source_dt_s must be finite and positive, got {source_dt_s}")
    if not np.isclose(dt, FROZEN_SOURCE_DT_S, rtol=0.0, atol=1e-6):
        raise ValueError(
            f"source_dt_s must match frozen 50 Hz dt={FROZEN_SOURCE_DT_S}, got {dt}"
        )
    if not np.isfinite(hz) or not np.isclose(
        hz, FROZEN_TARGET_HZ, rtol=0.0, atol=1e-9
    ):
        raise ValueError(
            f"target_hz must match frozen target {FROZEN_TARGET_HZ}, got {target_hz}"
        )

    source_time = steps.astype(np.float64) * dt
    tolerance = max(1e-9, dt * 1e-6)
    first_tick = int(np.ceil((source_time[0] - tolerance) * hz))
    last_tick = int(np.floor((source_time[-1] + tolerance) * hz))
    if last_tick < first_tick:
        raise ValueError("source episode does not cover one 20 Hz target tick")
    target_ticks = np.arange(first_tick, last_tick + 1, dtype=np.int64)
    target_time = target_ticks.astype(np.float64) / hz

    selected = np.searchsorted(source_time, target_time - tolerance, side="left")
    if np.any(selected >= source_time.size):
        raise ValueError("20 Hz target grid extends beyond the available source rows")
    selected = selected.astype(np.int64)
    if selected.size > 1 and np.any(np.diff(selected) <= 0):
        raise ValueError(
            "source cadence gap maps multiple target ticks to one source row"
        )

    selected_time = source_time[selected]
    selection_error = selected_time - target_time
    if np.any(selection_error < -tolerance):
        raise ValueError("selector chose a source row earlier than its target tick")
    selection_error[np.abs(selection_error) <= tolerance] = 0.0
    return SimTimeSelection(
        source_indices=selected,
        source_step_ids=steps[selected].copy(),
        source_sim_time_s=selected_time.astype(np.float64),
        target_ticks=target_ticks,
        target_sim_time_s=target_time.astype(np.float64),
        selection_error_s=selection_error.astype(np.float64),
    )


def transition_preservation_qc(
    actions: np.ndarray,
    *,
    step_ids: np.ndarray,
    source_dt_s: float,
    selected_indices: np.ndarray,
    deadzone: float | np.ndarray = 0.05,
    durable_min_duration_s: float = 0.05,
) -> dict[str, Any]:
    """Measure preservation of non-zero per-axis action-sign segments."""

    action = np.asarray(actions, dtype=np.float32)
    steps = np.asarray(step_ids, dtype=np.int64).reshape(-1)
    selected = np.asarray(selected_indices, dtype=np.int64).reshape(-1)
    if action.ndim != 2 or action.shape[1] != len(POLICY_AXIS_ORDER):
        raise ValueError(f"actions must have shape (T,4), got {action.shape}")
    if steps.shape != (action.shape[0],):
        raise ValueError("step_ids length must match actions")
    if not np.all(np.isfinite(action)):
        raise ValueError("actions contain NaN or infinity")
    if selected.size and (
        np.any(selected < 0)
        or np.any(selected >= action.shape[0])
        or np.any(np.diff(selected) <= 0)
    ):
        raise ValueError("selected_indices must be strictly increasing valid rows")

    thresholds = np.asarray(deadzone, dtype=np.float32)
    if thresholds.ndim == 0:
        thresholds = np.repeat(thresholds.reshape(1), action.shape[1])
    thresholds = thresholds.reshape(-1)
    if thresholds.shape != (action.shape[1],) or np.any(thresholds < 0.0):
        raise ValueError("deadzone must be one non-negative value per action axis")

    signs = np.zeros(action.shape, dtype=np.int8)
    signs[action > thresholds.reshape(1, -1)] = 1
    signs[action < -thresholds.reshape(1, -1)] = -1
    selected_set = set(int(index) for index in selected.tolist())

    segments: list[dict[str, Any]] = []
    axis_summaries: dict[str, dict[str, Any]] = {}
    for axis_index, axis_name in enumerate(POLICY_AXIS_ORDER):
        axis_segments: list[dict[str, Any]] = []
        cursor = 0
        while cursor < action.shape[0]:
            sign = int(signs[cursor, axis_index])
            if sign == 0:
                cursor += 1
                continue
            start = cursor
            cursor += 1
            while (
                cursor < action.shape[0]
                and int(signs[cursor, axis_index]) == sign
                and int(steps[cursor]) == int(steps[cursor - 1]) + 1
            ):
                cursor += 1
            end = cursor
            duration_s = (
                int(steps[end - 1]) - int(steps[start]) + 1
            ) * float(source_dt_s)
            hits = [index for index in range(start, end) if index in selected_set]
            preserved = bool(hits)
            onset_delay_s = (
                (int(steps[hits[0]]) - int(steps[start])) * float(source_dt_s)
                if hits
                else None
            )
            row = {
                "axis": axis_name,
                "axis_index": int(axis_index),
                "sign": int(sign),
                "source_start_index": int(start),
                "source_end_index_exclusive": int(end),
                "source_start_step_id": int(steps[start]),
                "source_end_step_id_inclusive": int(steps[end - 1]),
                "duration_s": float(duration_s),
                "durable": bool(duration_s + 1e-12 >= durable_min_duration_s),
                "preserved": preserved,
                "first_selected_source_index": int(hits[0]) if hits else None,
                "onset_delay_s": (
                    float(onset_delay_s) if onset_delay_s is not None else None
                ),
            }
            axis_segments.append(row)
            segments.append(row)

        axis_summaries[axis_name] = _summarize_segments(
            axis_segments,
            durable_min_duration_s=durable_min_duration_s,
        )

    summary = _summarize_segments(
        segments,
        durable_min_duration_s=durable_min_duration_s,
    )
    return {
        "schema_version": "sim_20hz_transition_preservation_qc_v1",
        "source_dt_s": float(source_dt_s),
        "target_hz": FROZEN_TARGET_HZ,
        "deadzone": thresholds.astype(float).tolist(),
        "durable_min_duration_s": float(durable_min_duration_s),
        **summary,
        "axes": axis_summaries,
        "segments": segments,
        "missing_segments": [row for row in segments if not row["preserved"]],
    }


def materialize_sim_episode(
    input_path: str | Path,
    output_path: str | Path,
    *,
    repo_root: str | Path | None = None,
    target_hz: float = FROZEN_TARGET_HZ,
    deadzone: float | np.ndarray | None = None,
    condition_rows: np.ndarray | None = None,
    condition_cycle_id: np.ndarray | None = None,
    condition_valid: np.ndarray | None = None,
    condition_materialized_from_sha256: str | None = None,
    condition_schema_sha256: str | None = None,
    jpeg_quality: int = 95,
    chunk_size: int = 64,
) -> dict[str, Any]:
    """Materialize one source episode without privilege or wall-clock fields."""

    src = Path(input_path).resolve(strict=True)
    dst = Path(output_path).resolve(strict=False)
    if dst.exists():
        raise FileExistsError(f"SimVerify output already exists: {dst}")
    if int(chunk_size) <= 0:
        raise ValueError("chunk_size must be positive")
    if int(jpeg_quality) != FROZEN_OUTPUT_JPEG_QUALITY:
        raise ValueError(
            "jpeg_quality is frozen by the camera transform contract at "
            f"{FROZEN_OUTPUT_JPEG_QUALITY}"
        )
    materialized_from_sha256 = _validate_optional_sha256(
        condition_materialized_from_sha256,
        name="condition_materialized_from_sha256",
    )
    schema_sha256 = _validate_optional_sha256(
        condition_schema_sha256,
        name="condition_schema_sha256",
    )

    source_chain = collect_hdf5_source_provenance(src)
    camera_contract = camera_transform_contract()
    state_contract = state_action_time_contract()
    resolved_repo = (
        Path(repo_root).resolve()
        if repo_root is not None
        else Path(__file__).resolve().parents[3]
    )
    repo_snapshot = git_provenance(resolved_repo)

    with h5py.File(src, "r") as in_handle:
        source_contract = _validate_source_episode(in_handle, src=src)
        selection = select_sim_time_indices(
            source_contract.step_ids,
            source_dt_s=source_contract.source_dt_s,
            target_hz=target_hz,
        )
        threshold = _resolve_deadzone(in_handle, deadzone)
        transition_qc = transition_preservation_qc(
            source_contract.action,
            step_ids=source_contract.step_ids,
            source_dt_s=source_contract.source_dt_s,
            selected_indices=selection.source_indices,
            deadzone=threshold,
        )
        (
            selected_condition,
            selected_condition_cycle_id,
            selected_condition_valid,
            condition_status,
        ) = (
            _prepare_condition(
                condition_rows=condition_rows,
                condition_cycle_id=condition_cycle_id,
                condition_valid=condition_valid,
                n_source_steps=source_contract.n_steps,
                selected_indices=selection.source_indices,
            )
        )

        dst.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{dst.name}.",
            suffix=".simverify-tmp",
            dir=dst.parent,
        )
        os.close(fd)
        temp_path = Path(temp_name)
        try:
            _write_materialized_episode(
                in_handle=in_handle,
                output_path=temp_path,
                source_contract=source_contract,
                source_sha256=str(source_chain[0]["sha256"]),
                selection=selection,
                selected_condition=selected_condition,
                selected_condition_cycle_id=selected_condition_cycle_id,
                selected_condition_valid=selected_condition_valid,
                condition_status=condition_status,
                condition_materialized_from_sha256=materialized_from_sha256,
                condition_schema_sha256=schema_sha256,
                camera_contract=camera_contract,
                state_contract=state_contract,
                repo_snapshot=repo_snapshot,
                jpeg_quality=int(jpeg_quality),
                chunk_size=int(chunk_size),
            )
            scan = scan_export_for_privilege(temp_path)
            if not scan["ok"]:
                raise ValueError(
                    "materialized export failed privilege scan: "
                    + ", ".join(scan["errors"])
                )
            assert_source_provenance_unchanged(source_chain)
            try:
                os.link(temp_path, dst)
            except FileExistsError:
                raise FileExistsError(
                    f"SimVerify output appeared during export: {dst}"
                ) from None
        finally:
            if temp_path.exists():
                temp_path.unlink()

    final_scan = scan_export_for_privilege(dst)
    output_identity = file_provenance(dst)
    return {
        "schema_version": "sim_observable_cycle_materialization_v1",
        "evidence_scope": "recorded-observation/offline",
        "input_path": str(src),
        "output_path": str(dst),
        "source_episode_id": source_contract.episode_id,
        "source_steps": int(source_contract.n_steps),
        "output_steps": int(selection.source_indices.size),
        "source_chain": source_chain,
        "output": output_identity,
        "git": repo_snapshot,
        "camera_contract": camera_contract,
        "state_action_time_contract": state_contract,
        "selection": _selection_summary(selection),
        "transition_preservation_qc": transition_qc,
        "privilege_scan": final_scan,
        "condition_status": condition_status,
    }


def _validate_source_episode(
    handle: h5py.File,
    *,
    src: Path,
) -> _SourceEpisodeContract:
    required = (
        "observations/qpos",
        "observations/qvel",
        "action",
        "timestamps/step_id",
    )
    for path in required:
        if path not in handle:
            raise ValueError(f"{src}: missing required dataset {path}")
    if "observations/encoded_images" not in handle:
        raise ValueError(f"{src}: missing observations/encoded_images")
    for camera in SOURCE_CAMERA_ORDER:
        path = f"observations/encoded_images/{camera}"
        if path not in handle:
            raise ValueError(f"{src}: missing source camera {camera}")

    qpos = handle["observations/qpos"]
    qvel = handle["observations/qvel"]
    action_ds = handle["action"]
    if qpos.ndim != 2 or qpos.shape[1] != 4:
        raise ValueError(f"{src}: qpos must have shape (T,4), got {qpos.shape}")
    n_steps = int(qpos.shape[0])
    if n_steps <= 0:
        raise ValueError(f"{src}: source episode is empty")
    if qvel.shape != (n_steps, 4) or action_ds.shape != (n_steps, 4):
        raise ValueError(
            f"{src}: qvel/action must match qpos shape {(n_steps, 4)}"
        )
    if handle["timestamps/step_id"].shape != (n_steps,):
        raise ValueError(f"{src}: timestamps/step_id length mismatch")
    for camera in SOURCE_CAMERA_ORDER:
        if handle[f"observations/encoded_images/{camera}"].shape != (n_steps,):
            raise ValueError(f"{src}: camera {camera} length mismatch")

    metadata = handle.get("metadata")
    if metadata is None:
        raise ValueError(f"{src}: missing metadata group")
    dt = float(metadata.attrs.get("dt", np.nan))
    if not np.isclose(dt, FROZEN_SOURCE_DT_S, rtol=0.0, atol=1e-6):
        raise ValueError(f"{src}: metadata.dt must be frozen 0.02, got {dt}")
    _require_csv_attr(metadata, "qpos_order", SOURCE_QPOS_ORDER, src=src)
    _require_csv_attr(metadata, "qvel_order", SOURCE_QVEL_ORDER, src=src)
    _require_csv_attr(metadata, "action_order", SOURCE_ACTION_ORDER, src=src)
    _require_csv_attr(metadata, "camera_names", SOURCE_CAMERA_ORDER, src=src)
    action_semantics = _metadata_text(
        metadata.attrs.get("action_semantics", "")
    )
    if action_semantics != "actuator_speed_cmd":
        raise ValueError(
            f"{src}: action_semantics must be actuator_speed_cmd, "
            f"got {action_semantics!r}"
        )

    step_id_dataset = handle["timestamps/step_id"]
    if not np.issubdtype(step_id_dataset.dtype, np.integer):
        raise ValueError(
            f"{src}: timestamps/step_id must have integer dtype, "
            f"got {step_id_dataset.dtype}"
        )
    step_ids = np.asarray(step_id_dataset[()], dtype=np.int64)
    if np.any(np.diff(step_ids) <= 0):
        raise ValueError(f"{src}: step_id must be strictly increasing")
    action = np.asarray(action_ds[()], dtype=np.float32)
    if not np.all(np.isfinite(action)):
        raise ValueError(f"{src}: action contains NaN or infinity")
    episode_id = _metadata_text(metadata.attrs.get("episode_id", src.stem))
    return _SourceEpisodeContract(
        n_steps=n_steps,
        source_dt_s=dt,
        step_ids=step_ids,
        action=action,
        episode_id=episode_id,
    )


def _write_materialized_episode(
    *,
    in_handle: h5py.File,
    output_path: Path,
    source_contract: _SourceEpisodeContract,
    source_sha256: str,
    selection: SimTimeSelection,
    selected_condition: np.ndarray,
    selected_condition_cycle_id: np.ndarray,
    selected_condition_valid: np.ndarray,
    condition_status: str,
    condition_materialized_from_sha256: str | None,
    condition_schema_sha256: str | None,
    camera_contract: dict[str, Any],
    state_contract: dict[str, Any],
    repo_snapshot: dict[str, Any],
    jpeg_quality: int,
    chunk_size: int,
) -> None:
    n_output = int(selection.source_indices.size)
    row_chunk = min(max(1, int(chunk_size)), n_output)
    with h5py.File(output_path, "w") as out:
        out.attrs["sim"] = True
        out.attrs["is_real"] = False
        out.attrs["simverify_export"] = True

        metadata = out.create_group("metadata")
        metadata.attrs["schema_version"] = EXPORT_EPISODE_SCHEMA
        metadata.attrs["evidence_scope"] = "recorded-observation/offline"
        metadata.attrs["source_dataset_path"] = str(
            Path(in_handle.filename).resolve()
        )
        metadata.attrs["source_dataset_sha256"] = source_sha256
        metadata.attrs["source_episode_id"] = source_contract.episode_id
        metadata.attrs["source_n_steps"] = int(source_contract.n_steps)
        metadata.attrs["n_steps"] = n_output
        metadata.attrs["source_dt_s"] = float(source_contract.source_dt_s)
        metadata.attrs["source_hz"] = FROZEN_SOURCE_HZ
        metadata.attrs["record_hz"] = FROZEN_TARGET_HZ
        metadata.attrs["control_hz"] = FROZEN_TARGET_HZ
        metadata.attrs["dt"] = 1.0 / FROZEN_TARGET_HZ
        metadata.attrs["sampling_hz"] = FROZEN_TARGET_HZ
        metadata.attrs["source_time_basis"] = (
            "timestamps/step_id * metadata.dt"
        )
        metadata.attrs["source_step_ns_used"] = False
        metadata.attrs["output_step_id_semantics"] = "target_tick_20hz"
        metadata.attrs["action_label_offset_s"] = (
            FROZEN_ACTION_LABEL_OFFSET_S
        )
        metadata.attrs["action_prealigned"] = True
        metadata.attrs["camera_names"] = ",".join(POLICY_CAMERA_ORDER)
        metadata.attrs["source_camera_names"] = ",".join(SOURCE_CAMERA_ORDER)
        metadata.attrs["camera_mapping_id"] = CAMERA_MAPPING_ID
        metadata.attrs["camera_contract_sha256"] = camera_contract[
            "contract_sha256"
        ]
        metadata.attrs["image_transform_id"] = IMAGE_TRANSFORM_ID
        metadata.attrs["source_image_width"] = 512
        metadata.attrs["source_image_height"] = 288
        metadata.attrs["output_image_width"] = 384
        metadata.attrs["output_image_height"] = 216
        metadata.attrs["image_color_space"] = "RGB"
        metadata.attrs["image_resize_filter"] = "linear"
        metadata.attrs["image_crop_policy"] = "none"
        metadata.attrs["image_jpeg_quality"] = FROZEN_OUTPUT_JPEG_QUALITY
        metadata.attrs["geometric_equivalence"] = False
        metadata.attrs["qpos_order"] = ",".join(SOURCE_QPOS_ORDER)
        metadata.attrs["qvel_order"] = ",".join(SOURCE_QVEL_ORDER)
        metadata.attrs["action_order"] = ",".join(POLICY_AXIS_ORDER)
        metadata.attrs["qpos_semantics"] = "sim_source_representation"
        metadata.attrs["qvel_semantics"] = "sim_source_representation"
        metadata.attrs["action_semantics"] = "actuator_speed_cmd"
        metadata.attrs["state_domain"] = "sim_source_domain_only"
        metadata.attrs["action_domain"] = "sim_source_domain"
        metadata.attrs["checkpoint_restriction"] = (
            "sim_state_domain_only_not_real_deployable"
        )
        metadata.attrs["state_action_time_contract_id"] = (
            STATE_ACTION_TIME_CONTRACT_ID
        )
        metadata.attrs["state_action_time_contract_sha256"] = state_contract[
            "contract_sha256"
        ]
        metadata.attrs["condition_schema_version"] = CONDITION_SCHEMA_VERSION
        metadata.attrs["condition_dim"] = 6
        metadata.attrs["condition_status"] = condition_status
        condition_source = (
            "hindsight_outcome_pending"
            if condition_status == "placeholder_unlabeled"
            else "hindsight_outcome"
        )
        metadata.attrs["condition_source"] = condition_source
        metadata.attrs["command_source"] = "unknown_not_recorded"
        metadata.attrs["export_git_commit"] = str(
            repo_snapshot.get("commit", "")
        )
        metadata.attrs["export_git_branch"] = str(
            repo_snapshot.get("branch", "")
        )
        metadata.attrs["export_git_dirty"] = bool(
            repo_snapshot.get("dirty", False)
        )

        observations = out.create_group("observations")
        qpos_out = observations.create_dataset(
            "qpos",
            shape=(n_output, 4),
            dtype=np.float32,
            chunks=(row_chunk, 4),
            compression="lzf",
        )
        qvel_out = observations.create_dataset(
            "qvel",
            shape=(n_output, 4),
            dtype=np.float32,
            chunks=(row_chunk, 4),
            compression="lzf",
        )
        conditions = out.create_group("conditions")
        conditions.attrs["schema_id"] = CONDITION_SCHEMA_VERSION
        conditions.attrs["dim"] = 6
        conditions.attrs["encoding"] = (
            "current_sector_one_hot_3_plus_next_sector_one_hot_3"
        )
        conditions.attrs["normalization"] = "none_binary_one_hot"
        conditions.attrs["source"] = condition_source
        conditions.attrs["scope"] = "constant_within_observable_cycle"
        if condition_materialized_from_sha256 is not None:
            conditions.attrs["materialized_from_sha256"] = (
                condition_materialized_from_sha256
            )
        if condition_schema_sha256 is not None:
            conditions.attrs["schema_sha256"] = condition_schema_sha256
        condition_out = conditions.create_dataset(
            "cycle_condition_v1",
            data=selected_condition.astype(np.float32, copy=False),
            chunks=(row_chunk, 6),
            compression="lzf",
        )
        condition_out.attrs["schema_id"] = CONDITION_SCHEMA_VERSION
        condition_out.attrs["dim"] = 6
        condition_out.attrs["normalization"] = "none_binary_one_hot"
        condition_out.attrs["source"] = condition_source
        condition_out.attrs["scope"] = "constant_within_observable_cycle"
        conditions.create_dataset(
            "cycle_id",
            data=selected_condition_cycle_id.astype(np.int64, copy=False),
        )
        conditions.create_dataset(
            "valid_mask",
            data=selected_condition_valid.astype(np.uint8, copy=False),
        )
        images_out = observations.create_group("encoded_images")
        image_dtype = h5py.vlen_dtype(np.dtype("uint8"))
        output_image_datasets: dict[str, h5py.Dataset] = {}
        for source_camera, policy_camera in SOURCE_TO_POLICY_CAMERA.items():
            dataset = images_out.create_dataset(
                policy_camera,
                shape=(n_output,),
                dtype=image_dtype,
            )
            dataset.attrs["encoding"] = "jpeg"
            dataset.attrs["color_space"] = "RGB"
            dataset.attrs["width"] = 384
            dataset.attrs["height"] = 216
            dataset.attrs["jpeg_quality"] = FROZEN_OUTPUT_JPEG_QUALITY
            dataset.attrs["transform_id"] = IMAGE_TRANSFORM_ID
            dataset.attrs["source_camera"] = source_camera
            dataset.attrs["policy_camera"] = policy_camera
            output_image_datasets[policy_camera] = dataset

        action_out = out.create_dataset(
            "action",
            shape=(n_output, 4),
            dtype=np.float32,
            chunks=(row_chunk, 4),
            compression="lzf",
        )

        for start in range(0, n_output, row_chunk):
            stop = min(n_output, start + row_chunk)
            source_indices = selection.source_indices[start:stop]
            qpos = np.asarray(
                in_handle["observations/qpos"][source_indices],
                dtype=np.float32,
            )
            qvel = np.asarray(
                in_handle["observations/qvel"][source_indices],
                dtype=np.float32,
            )
            action = np.asarray(
                in_handle["action"][source_indices],
                dtype=np.float32,
            )
            if not (
                np.all(np.isfinite(qpos))
                and np.all(np.isfinite(qvel))
                and np.all(np.isfinite(action))
            ):
                raise ValueError(
                    f"non-finite observable row in output chunk {start}:{stop}"
                )
            qpos_out[start:stop] = qpos
            qvel_out[start:stop] = qvel
            action_out[start:stop] = action

            for output_index, source_index in enumerate(
                source_indices, start=start
            ):
                for source_camera, policy_camera in (
                    SOURCE_TO_POLICY_CAMERA.items()
                ):
                    encoded = np.asarray(
                        in_handle[
                            f"observations/encoded_images/{source_camera}"
                        ][int(source_index)],
                        dtype=np.uint8,
                    ).reshape(-1)
                    output_image_datasets[policy_camera][output_index] = (
                        _transform_jpeg(
                            encoded,
                            source_camera=source_camera,
                            jpeg_quality=jpeg_quality,
                        )
                    )

        timestamps = out.create_group("timestamps")
        timestamps.create_dataset(
            "step_id", data=selection.target_ticks.astype(np.int64)
        )
        timestamps.create_dataset(
            "sim_time_s", data=selection.target_sim_time_s.astype(np.float64)
        )

        diagnostics = out.create_group("diagnostics")
        diagnostics.create_dataset(
            "source_observation_index",
            data=selection.source_indices.astype(np.int64),
        )
        diagnostics.create_dataset(
            "source_action_index",
            data=selection.source_indices.astype(np.int64),
        )
        diagnostics.create_dataset(
            "source_step_id",
            data=selection.source_step_ids.astype(np.int64),
        )
        diagnostics.create_dataset(
            "source_sim_time_s",
            data=selection.source_sim_time_s.astype(np.float64),
        )
        diagnostics.create_dataset(
            "target_tick", data=selection.target_ticks.astype(np.int64)
        )
        diagnostics.create_dataset(
            "target_sim_time_s",
            data=selection.target_sim_time_s.astype(np.float64),
        )
        diagnostics.create_dataset(
            "selection_error_s",
            data=selection.selection_error_s.astype(np.float64),
        )


def _prepare_condition(
    *,
    condition_rows: np.ndarray | None,
    condition_cycle_id: np.ndarray | None,
    condition_valid: np.ndarray | None,
    n_source_steps: int,
    selected_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    if condition_rows is None:
        if condition_cycle_id is not None or condition_valid is not None:
            raise ValueError(
                "condition_cycle_id/condition_valid require condition_rows"
            )
        return (
            np.zeros((selected_indices.size, 6), dtype=np.float32),
            np.full(selected_indices.size, -1, dtype=np.int64),
            np.zeros(selected_indices.size, dtype=bool),
            "placeholder_unlabeled",
        )

    condition = np.asarray(condition_rows, dtype=np.float32)
    if condition.shape != (int(n_source_steps), 6):
        raise ValueError(
            "condition_rows must be source-row aligned with shape "
            f"({n_source_steps},6), got {condition.shape}"
        )
    if condition_cycle_id is None:
        raise ValueError("condition_rows requires source-row aligned condition_cycle_id")
    cycle_id = np.asarray(condition_cycle_id)
    if not np.issubdtype(cycle_id.dtype, np.integer):
        raise ValueError("condition_cycle_id must have integer dtype")
    cycle_id = cycle_id.astype(np.int64, copy=False).reshape(-1)
    if cycle_id.shape != (int(n_source_steps),):
        raise ValueError(
            f"condition_cycle_id must have shape ({n_source_steps},)"
        )
    if condition_valid is None:
        valid = np.ones(n_source_steps, dtype=bool)
    else:
        raw_valid = np.asarray(condition_valid).reshape(-1)
        if raw_valid.shape != (int(n_source_steps),):
            raise ValueError(
                f"condition_valid must have shape ({n_source_steps},)"
            )
        if not (
            np.issubdtype(raw_valid.dtype, np.bool_)
            or np.issubdtype(raw_valid.dtype, np.integer)
        ):
            raise ValueError("condition_valid must have boolean or integer dtype")
        if np.any((raw_valid != 0) & (raw_valid != 1)):
            raise ValueError("condition_valid values must be binary 0/1")
        valid = raw_valid.astype(bool, copy=False)
    if not np.all(np.isfinite(condition)):
        raise ValueError("condition_rows contains NaN or infinity")
    _validate_condition_values(condition, cycle_id, valid)
    selected_valid = valid[selected_indices].copy()
    return (
        condition[selected_indices].copy(),
        cycle_id[selected_indices].copy(),
        selected_valid,
        "labeled" if bool(np.all(selected_valid)) else "partially_labeled",
    )


def _validate_optional_sha256(value: str | None, *, name: str) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{name} must be a 64-character hexadecimal SHA-256")
    return normalized


def _validate_condition_values(
    condition: np.ndarray,
    cycle_id: np.ndarray,
    valid: np.ndarray,
) -> None:
    if np.any((condition < 0.0) | (condition > 1.0)):
        raise ValueError("cycle_condition_v1 values must be in [0,1]")
    valid_rows = condition[valid]
    if valid_rows.size:
        binary = np.isclose(valid_rows, 0.0) | np.isclose(valid_rows, 1.0)
        if not np.all(binary):
            raise ValueError("valid cycle_condition_v1 rows must be binary")
        if not (
            np.allclose(valid_rows[:, :3].sum(axis=1), 1.0)
            and np.allclose(valid_rows[:, 3:].sum(axis=1), 1.0)
        ):
            raise ValueError(
                "valid cycle_condition_v1 rows require one current and one next sector"
            )
    if np.any(condition[~valid] != 0.0):
        raise ValueError("invalid condition rows must use the all-zero placeholder")
    if np.any(cycle_id[valid] < 0):
        raise ValueError("valid condition rows require non-negative cycle_id")
    if np.any(cycle_id[~valid] != -1):
        raise ValueError("invalid condition rows require cycle_id=-1")


def _transform_jpeg(
    encoded: np.ndarray,
    *,
    source_camera: str,
    jpeg_quality: int,
) -> np.ndarray:
    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"failed to decode JPEG for source camera {source_camera}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if rgb.shape != (288, 512, 3):
        raise ValueError(
            f"source camera {source_camera} decoded shape must be "
            f"(288,512,3), got {rgb.shape}"
        )
    resized_rgb = cv2.resize(
        rgb,
        (384, 216),
        interpolation=cv2.INTER_LINEAR,
    )
    resized_bgr = cv2.cvtColor(resized_rgb, cv2.COLOR_RGB2BGR)
    ok, output = cv2.imencode(
        ".jpg",
        resized_bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
    )
    if not ok:
        raise ValueError(f"failed to encode JPEG for source camera {source_camera}")
    return np.asarray(output, dtype=np.uint8).reshape(-1)


def _resolve_deadzone(
    handle: h5py.File,
    explicit: float | np.ndarray | None,
) -> np.ndarray:
    if explicit is None:
        metadata = handle["metadata"]
        explicit = metadata.attrs.get("deadzone", 0.05)
    values = np.asarray(explicit, dtype=np.float32)
    if values.ndim == 0:
        values = np.repeat(values.reshape(1), 4)
    values = values.reshape(-1)
    if values.shape != (4,) or np.any(values < 0.0):
        raise ValueError("deadzone must resolve to four non-negative values")
    return values


def _require_csv_attr(
    metadata: h5py.Group,
    name: str,
    expected: tuple[str, ...],
    *,
    src: Path,
) -> None:
    value = _metadata_text(metadata.attrs.get(name, ""))
    parsed = tuple(item.strip() for item in value.split(",") if item.strip())
    if parsed != expected:
        raise ValueError(
            f"{src}: metadata.{name} must be {expected}, got {parsed}"
        )


def _metadata_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="strict")
    return str(value)


def _selection_summary(selection: SimTimeSelection) -> dict[str, Any]:
    error = selection.selection_error_s
    return {
        "source_row_count": int(selection.source_indices.size),
        "first_source_index": int(selection.source_indices[0]),
        "last_source_index": int(selection.source_indices[-1]),
        "first_target_tick": int(selection.target_ticks[0]),
        "last_target_tick": int(selection.target_ticks[-1]),
        "action_label_offset_s": FROZEN_ACTION_LABEL_OFFSET_S,
        "same_row_alignment": True,
        "selection_error_s": {
            "min": float(np.min(error)),
            "p50": float(np.percentile(error, 50)),
            "p95": float(np.percentile(error, 95)),
            "max": float(np.max(error)),
        },
    }


def _summarize_segments(
    segments: list[dict[str, Any]],
    *,
    durable_min_duration_s: float,
) -> dict[str, Any]:
    preserved = [row for row in segments if row["preserved"]]
    missing = [row for row in segments if not row["preserved"]]
    durable = [row for row in segments if row["durable"]]
    durable_missing = [row for row in durable if not row["preserved"]]
    delays = [
        float(row["onset_delay_s"])
        for row in preserved
        if row["onset_delay_s"] is not None
    ]
    return {
        "valid_segment_count": len(segments),
        "preserved_segment_count": len(preserved),
        "missing_segment_count": len(missing),
        "preservation_rate": (
            float(len(preserved) / len(segments)) if segments else 1.0
        ),
        "durable_segment_count": len(durable),
        "durable_missing_segment_count": len(durable_missing),
        "all_missing_segments_shorter_than_durable_min": bool(
            all(
                float(row["duration_s"]) + 1e-12
                < float(durable_min_duration_s)
                for row in missing
            )
        ),
        "max_preserved_onset_delay_s": max(delays) if delays else 0.0,
    }
