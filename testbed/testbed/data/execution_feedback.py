"""Causal execution-feedback sidecars and target-independent retry variants."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml

from testbed.data.deadzone_intent_labels import AXIS_NAMES

SIDECAR_SCHEMA_VERSION = 1
MANIFEST_SCHEMA_VERSION = 1
ALIGNMENT_MODE = "latest_raw_command_strictly_before_observation_v1"
COUNTERFACTUAL_MODE = "symmetric_target_independent_weak_command_v1"
COUNTERFACTUAL_VARIANT_COUNT = 2
MANIFEST_FILENAME = "execution_feedback_manifest.json"


@dataclass(frozen=True)
class CausalCommandAlignment:
    """Per-observation physical command state reconstructed from raw sends."""

    previous_final_command: np.ndarray
    raw_source_index: np.ndarray
    command_send_timestamp_ns: np.ndarray
    observation_timestamp_ns: np.ndarray
    train_exclude_mask: np.ndarray
    source_time_gap_exceeds_threshold: np.ndarray
    reset_mask: np.ndarray
    valid_mask: np.ndarray


@dataclass(frozen=True)
class ExecutionFeedbackSidecar(CausalCommandAlignment):
    """Validated sidecar arrays plus immutable source provenance."""

    episode_id: str
    raw_source_length: int
    resampled_path: str
    raw_source_path: str
    resampled_sha256: str
    raw_source_sha256: str

    def __len__(self) -> int:
        return int(self.previous_final_command.shape[0])


@dataclass(frozen=True)
class CounterfactualWeakCommandVariants:
    """A deterministic positive/negative weak-command pair for one axis."""

    mode: str
    episode_id: str
    timestep: int
    seed: int
    axis_index: int
    axis: str
    magnitude_fraction: float
    previous_final_command: np.ndarray
    qvel: np.ndarray


def resolve_execution_feedback_config(
    raw: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Resolve the dataset-facing execution-feedback configuration."""

    if raw is not None and not isinstance(raw, Mapping):
        raise ValueError("execution_feedback config must be a mapping")
    cfg = dict(raw or {})
    enabled = _strict_bool(cfg.get("enabled", False), name="enabled")
    if not enabled:
        return _disabled_execution_feedback_config()

    manifest_path = _required_file(
        cfg.get("manifest_path"),
        name="manifest_path",
    )
    base_norm_stats_path = _required_file(
        cfg.get("base_norm_stats_path"),
        name="base_norm_stats_path",
    )
    counterfactual_raw = cfg.get("counterfactual", {})
    if not isinstance(counterfactual_raw, Mapping):
        raise ValueError("execution_feedback.counterfactual must be a mapping")
    counterfactual_cfg = dict(counterfactual_raw)
    counterfactual_enabled = _strict_bool(
        counterfactual_cfg.get("enabled", False),
        name="counterfactual.enabled",
    )
    if not counterfactual_enabled:
        counterfactual = _disabled_counterfactual_config()
    else:
        if "seed" not in counterfactual_cfg:
            raise ValueError("execution_feedback.counterfactual.seed is required")
        seed = _integer(
            counterfactual_cfg["seed"],
            name="counterfactual.seed",
        )
        loss_weight = _finite_nonnegative_float(
            counterfactual_cfg.get("loss_weight", 1.0),
            name="counterfactual.loss_weight",
        )
        _validate_fixed_counterfactual_contract(counterfactual_cfg)
        thresholds = _resolve_threshold_source(
            counterfactual_cfg,
            prefix="execution_feedback.counterfactual",
        )
        counterfactual = {
            "enabled": True,
            "seed": seed,
            "loss_weight": loss_weight,
            "thresholds": thresholds,
        }

    return {
        "enabled": True,
        "manifest_path": str(manifest_path),
        "base_norm_stats_path": str(base_norm_stats_path),
        "counterfactual": counterfactual,
    }


def align_causal_previous_commands(
    *,
    observation_timestamp_ns: np.ndarray,
    raw_commanded_action: np.ndarray,
    raw_action_send_timestamp_ns: np.ndarray,
    train_exclude_mask: np.ndarray,
    source_time_gap_exceeds_threshold: np.ndarray,
) -> CausalCommandAlignment:
    """Align the latest strictly-prior raw command to each observation.

    The physical previous command is zero at episode start and on every reset
    sample.  After a reset, a command is valid only if it was sent at or after
    the most recent reset observation and strictly before the current one.
    """

    observation_ts = _timestamp_vector(
        observation_timestamp_ns,
        name="observation_timestamp_ns",
        strictly_increasing=True,
    )
    if observation_ts.size == 0:
        raise ValueError("observation_timestamp_ns must not be empty")
    raw_send_ts = _timestamp_vector(
        raw_action_send_timestamp_ns,
        name="raw_action_send_timestamp_ns",
        strictly_increasing=False,
    )
    raw_command = _command_matrix(
        raw_commanded_action,
        name="raw_commanded_action",
    )
    if raw_command.shape[0] != raw_send_ts.size:
        raise ValueError(
            "raw_commanded_action and raw_action_send_timestamp_ns lengths differ: "
            f"{raw_command.shape[0]} != {raw_send_ts.size}"
        )
    n_steps = int(observation_ts.size)
    train_exclude = _bool_mask(
        train_exclude_mask,
        n_steps=n_steps,
        name="train_exclude_mask",
    )
    source_gap = _bool_mask(
        source_time_gap_exceeds_threshold,
        n_steps=n_steps,
        name="source_time_gap_exceeds_threshold",
    )
    reset_mask = train_exclude | source_gap
    reset_mask[0] = True

    previous_command = np.zeros((n_steps, len(AXIS_NAMES)), dtype=np.float32)
    raw_source_index = np.full(n_steps, -1, dtype=np.int64)
    command_send_ts = np.full(n_steps, -1, dtype=np.int64)
    valid_mask = np.zeros(n_steps, dtype=bool)
    last_reset_ts = int(observation_ts[0])

    for timestep, observation_time in enumerate(observation_ts):
        if reset_mask[timestep]:
            last_reset_ts = int(observation_time)
            continue
        source_index = int(
            np.searchsorted(raw_send_ts, observation_time, side="left") - 1
        )
        if source_index < 0:
            continue
        send_time = int(raw_send_ts[source_index])
        if send_time < last_reset_ts:
            continue
        if send_time >= int(observation_time):
            raise ValueError(
                "internal causality violation: command send timestamp must be "
                "strictly earlier than observation timestamp"
            )
        previous_command[timestep] = raw_command[source_index]
        raw_source_index[timestep] = source_index
        command_send_ts[timestep] = send_time
        valid_mask[timestep] = True

    return CausalCommandAlignment(
        previous_final_command=previous_command,
        raw_source_index=raw_source_index,
        command_send_timestamp_ns=command_send_ts,
        observation_timestamp_ns=observation_ts.copy(),
        train_exclude_mask=train_exclude,
        source_time_gap_exceeds_threshold=source_gap,
        reset_mask=reset_mask,
        valid_mask=valid_mask,
    )


def build_episode_execution_feedback(
    *,
    episode_id: int | str,
    resampled_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Build one causal execution-feedback NPZ and return its manifest entry."""

    episode = _episode_id(episode_id)
    resampled = _required_file(resampled_path, name="resampled_path")
    output = Path(output_path).expanduser().resolve()

    with h5py.File(resampled, "r") as resampled_file:
        if "metadata" not in resampled_file:
            raise ValueError(f"resampled episode is missing metadata: {resampled}")
        source_path_raw = resampled_file["metadata"].attrs.get("source_dataset_path")
        raw_source = _resolve_source_path(
            source_path_raw,
            resampled_path=resampled,
        )
        observation_ts = _required_dataset(
            resampled_file,
            "diagnostics/source_observation_timestamp_ns",
            file_label="resampled episode",
        )
        train_exclude = _required_dataset(
            resampled_file,
            "diagnostics/train_exclude_mask",
            file_label="resampled episode",
        )
        source_gap = _required_dataset(
            resampled_file,
            "diagnostics/source_time_gap_exceeds_threshold",
            file_label="resampled episode",
        )

    if raw_source == resampled:
        raise ValueError("metadata.source_dataset_path must identify the raw episode")
    with h5py.File(raw_source, "r") as raw_file:
        raw_command = _required_dataset(
            raw_file,
            "diagnostics/commanded_action",
            file_label="raw episode",
        )
        raw_send_ts = _required_dataset(
            raw_file,
            "diagnostics/action_send_timestamp_ns",
            file_label="raw episode",
        )

    alignment = align_causal_previous_commands(
        observation_timestamp_ns=observation_ts,
        raw_commanded_action=raw_command,
        raw_action_send_timestamp_ns=raw_send_ts,
        train_exclude_mask=train_exclude,
        source_time_gap_exceeds_threshold=source_gap,
    )
    resampled_sha256 = sha256_file(resampled)
    raw_source_sha256 = sha256_file(raw_source)
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_sidecar(
        output_path=output,
        episode_id=episode,
        alignment=alignment,
        raw_source_length=int(np.asarray(raw_send_ts).reshape(-1).size),
        resampled_path=resampled,
        raw_source_path=raw_source,
        resampled_sha256=resampled_sha256,
        raw_source_sha256=raw_source_sha256,
    )
    sidecar = load_execution_feedback_sidecar(
        output,
        expected_episode_id=episode,
        expected_length=len(alignment.previous_final_command),
    )
    sidecar_sha256 = sha256_file(output)
    return {
        "episode_id": int(episode),
        "sidecar_path": str(output),
        "sidecar_sha256": sidecar_sha256,
        "resampled_path": str(resampled),
        "resampled_sha256": resampled_sha256,
        "raw_source_path": str(raw_source),
        "raw_source_sha256": raw_source_sha256,
        "length": len(sidecar),
        "valid_count": int(np.count_nonzero(sidecar.valid_mask)),
        "reset_counts": _reset_counts(sidecar),
        "causality_age_summary_ns": causality_age_summary_ns(sidecar),
    }


def load_execution_feedback_sidecar(
    path: str | Path,
    *,
    expected_episode_id: int | str | None = None,
    expected_length: int | None = None,
) -> ExecutionFeedbackSidecar:
    """Load and strictly validate a sidecar without pickle support."""

    sidecar_path = _required_file(path, name="sidecar_path")
    required = {
        "schema_version",
        "alignment_mode",
        "episode_id",
        "previous_final_command",
        "raw_source_index",
        "command_send_timestamp_ns",
        "observation_timestamp_ns",
        "train_exclude_mask",
        "source_time_gap_exceeds_threshold",
        "reset_mask",
        "valid_mask",
        "raw_source_length",
        "resampled_path",
        "raw_source_path",
        "resampled_sha256",
        "raw_source_sha256",
    }
    with np.load(sidecar_path, allow_pickle=False) as payload:
        missing = sorted(required - set(payload.files))
        if missing:
            raise ValueError(
                f"execution-feedback sidecar is missing keys: {', '.join(missing)}"
            )
        schema_version = _npz_scalar_int(payload, "schema_version")
        if schema_version != SIDECAR_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported execution-feedback sidecar schema {schema_version}"
            )
        alignment_mode = _npz_scalar_text(payload, "alignment_mode")
        if alignment_mode != ALIGNMENT_MODE:
            raise ValueError(
                f"unsupported execution-feedback alignment_mode {alignment_mode!r}"
            )
        episode = _episode_id(_npz_scalar_text(payload, "episode_id"))
        raw_source_length = _npz_scalar_int(payload, "raw_source_length")
        if raw_source_length < 0:
            raise ValueError("raw_source_length must be nonnegative")
        previous_command = _command_matrix(
            payload["previous_final_command"],
            name="previous_final_command",
        )
        n_steps = int(previous_command.shape[0])
        raw_source_index = _integer_vector(
            payload["raw_source_index"],
            n_steps=n_steps,
            name="raw_source_index",
            allow_negative_one=True,
        )
        command_send_ts = _integer_vector(
            payload["command_send_timestamp_ns"],
            n_steps=n_steps,
            name="command_send_timestamp_ns",
            allow_negative_one=True,
        )
        observation_ts = _timestamp_vector(
            payload["observation_timestamp_ns"],
            name="observation_timestamp_ns",
            strictly_increasing=True,
        )
        if observation_ts.size != n_steps:
            raise ValueError("observation_timestamp_ns length does not match commands")
        train_exclude = _bool_mask(
            payload["train_exclude_mask"],
            n_steps=n_steps,
            name="train_exclude_mask",
        )
        source_gap = _bool_mask(
            payload["source_time_gap_exceeds_threshold"],
            n_steps=n_steps,
            name="source_time_gap_exceeds_threshold",
        )
        reset_mask = _bool_mask(
            payload["reset_mask"],
            n_steps=n_steps,
            name="reset_mask",
        )
        valid_mask = _bool_mask(
            payload["valid_mask"],
            n_steps=n_steps,
            name="valid_mask",
        )
        resampled_path = _npz_scalar_text(payload, "resampled_path")
        raw_source_path = _npz_scalar_text(payload, "raw_source_path")
        resampled_sha256 = _sha256_text(
            _npz_scalar_text(payload, "resampled_sha256"),
            name="resampled_sha256",
        )
        raw_source_sha256 = _sha256_text(
            _npz_scalar_text(payload, "raw_source_sha256"),
            name="raw_source_sha256",
        )

    if n_steps == 0:
        raise ValueError("execution-feedback sidecar must not be empty")
    expected_reset = train_exclude | source_gap
    expected_reset[0] = True
    if not np.array_equal(reset_mask, expected_reset):
        raise ValueError(
            "reset_mask does not match episode-start/train-exclude/gap union"
        )
    if np.any(reset_mask & valid_mask):
        raise ValueError("reset samples must not carry a valid previous command")
    invalid = ~valid_mask
    if np.any(previous_command[invalid] != 0.0):
        raise ValueError("invalid previous commands must be exactly zero")
    if np.any(raw_source_index[invalid] != -1):
        raise ValueError("invalid raw_source_index entries must be -1")
    if np.any(command_send_ts[invalid] != -1):
        raise ValueError("invalid command_send_timestamp_ns entries must be -1")
    if np.any(raw_source_index[valid_mask] < 0) or np.any(
        raw_source_index[valid_mask] >= raw_source_length
    ):
        raise ValueError("valid raw_source_index is outside raw_source_length")
    if np.any(command_send_ts[valid_mask] < 0):
        raise ValueError("valid command send timestamps must be nonnegative")
    if np.any(command_send_ts[valid_mask] >= observation_ts[valid_mask]):
        raise ValueError(
            "noncausal sidecar: command send timestamp must be strictly earlier "
            "than observation timestamp"
        )
    reset_boundary = np.maximum.accumulate(
        np.where(reset_mask, observation_ts, np.int64(-1))
    )
    if np.any(command_send_ts[valid_mask] < reset_boundary[valid_mask]):
        raise ValueError("sidecar inherits a command from before the latest reset")
    if expected_episode_id is not None and episode != _episode_id(expected_episode_id):
        raise ValueError(
            f"sidecar episode_id {episode} != expected {_episode_id(expected_episode_id)}"
        )
    if expected_length is not None and n_steps != _nonnegative_integer(
        expected_length,
        name="expected_length",
    ):
        raise ValueError(
            f"sidecar length {n_steps} != expected length {int(expected_length)}"
        )

    return ExecutionFeedbackSidecar(
        previous_final_command=previous_command.copy(),
        raw_source_index=raw_source_index.copy(),
        command_send_timestamp_ns=command_send_ts.copy(),
        observation_timestamp_ns=observation_ts.copy(),
        train_exclude_mask=train_exclude.copy(),
        source_time_gap_exceeds_threshold=source_gap.copy(),
        reset_mask=reset_mask.copy(),
        valid_mask=valid_mask.copy(),
        episode_id=episode,
        raw_source_length=raw_source_length,
        resampled_path=resampled_path,
        raw_source_path=raw_source_path,
        resampled_sha256=resampled_sha256,
        raw_source_sha256=raw_source_sha256,
    )


def validate_execution_feedback_manifest(
    path: str | Path,
    *,
    verify_hashes: bool = True,
    expected_dataset_dir: str | Path | None = None,
    expected_split_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate manifest coverage, sidecar contracts, and optional file hashes."""

    manifest_path = _required_file(path, name="manifest_path")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("execution-feedback manifest must be a mapping")
    if int(payload.get("schema_version", -1)) != MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported execution-feedback manifest schema")
    if payload.get("alignment_mode") != ALIGNMENT_MODE:
        raise ValueError("execution-feedback manifest alignment_mode mismatch")
    if payload.get("counterfactual_mode") != COUNTERFACTUAL_MODE:
        raise ValueError("execution-feedback manifest counterfactual_mode mismatch")
    if int(payload.get("counterfactual_variant_count", -1)) != (
        COUNTERFACTUAL_VARIANT_COUNT
    ):
        raise ValueError("execution-feedback manifest variant count mismatch")

    dataset_dir = _required_directory(payload.get("dataset_dir"), name="dataset_dir")
    split_path = _required_file(payload.get("split_path"), name="split_path")
    split_sha256 = _sha256_text(
        str(payload.get("split_sha256", "")),
        name="split_sha256",
    )
    if sha256_file(split_path) != split_sha256:
        raise ValueError("execution-feedback manifest split SHA-256 mismatch")
    if (
        expected_dataset_dir is not None
        and dataset_dir != Path(expected_dataset_dir).expanduser().resolve()
    ):
        raise ValueError("execution-feedback manifest dataset_dir mismatch")
    if (
        expected_split_path is not None
        and split_path != Path(expected_split_path).expanduser().resolve()
    ):
        raise ValueError("execution-feedback manifest split_path mismatch")

    split_train_ids, split_val_ids = load_split_episode_ids(split_path)
    manifest_train_ids = _episode_id_list(payload.get("train_ids"), name="train_ids")
    manifest_val_ids = _episode_id_list(payload.get("val_ids"), name="val_ids")
    if manifest_train_ids != split_train_ids or manifest_val_ids != split_val_ids:
        raise ValueError("execution-feedback manifest IDs do not match split file")
    expected_ids = _ordered_union(split_train_ids, split_val_ids)
    manifest_ids = _episode_id_list(payload.get("episode_ids"), name="episode_ids")
    if manifest_ids != expected_ids:
        raise ValueError("execution-feedback manifest episode_ids mismatch")

    records = payload.get("episodes")
    if not isinstance(records, list):
        raise ValueError("execution-feedback manifest episodes must be a list")
    record_ids = [
        int(_episode_id(record.get("episode_id")))
        for record in records
        if isinstance(record, Mapping)
    ]
    if len(record_ids) != len(records) or record_ids != expected_ids:
        raise ValueError("execution-feedback manifest episode records mismatch")

    for record in records:
        _validate_manifest_episode_record(
            record,
            dataset_dir=dataset_dir,
            verify_hashes=bool(verify_hashes),
        )
    return payload


def load_split_episode_ids(path: str | Path) -> tuple[list[int], list[int]]:
    """Load exactly the train and validation IDs declared by a split file."""

    split_path = _required_file(path, name="split_path")
    payload = yaml.safe_load(split_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ValueError("split file must contain a mapping")
    train_ids = _episode_id_list(payload.get("train_ids"), name="train_ids")
    val_ids = _episode_id_list(payload.get("val_ids"), name="val_ids")
    if not train_ids or not val_ids:
        raise ValueError("split file must contain non-empty train_ids and val_ids")
    return train_ids, val_ids


def generate_symmetric_weak_command_variants(
    *,
    episode_id: int | str,
    timestep: int,
    seed: int,
    thresholds: Mapping[str, Mapping[str, Any]],
) -> CounterfactualWeakCommandVariants:
    """Generate a stable target-independent positive/negative weak-command pair."""

    episode = _episode_id(episode_id)
    step = _nonnegative_integer(timestep, name="timestep")
    fixed_seed = _integer(seed, name="seed")
    normalized_thresholds = _normalize_thresholds(
        thresholds,
        prefix="counterfactual thresholds",
    )
    digest = hashlib.sha256(
        (f"{COUNTERFACTUAL_MODE}\0{fixed_seed}\0{episode}\0{step}").encode()
    ).digest()
    axis_index = int.from_bytes(digest[:8], byteorder="big", signed=False) % len(
        AXIS_NAMES
    )
    fraction_bits = int.from_bytes(digest[8:16], byteorder="big", signed=False) >> 11
    magnitude_fraction = float(fraction_bits) / float(1 << 53)
    axis = AXIS_NAMES[axis_index]
    positive_threshold = normalized_thresholds[axis]["pos"]
    negative_threshold = normalized_thresholds[axis]["neg"]

    commands = np.zeros(
        (COUNTERFACTUAL_VARIANT_COUNT, len(AXIS_NAMES)),
        dtype=np.float32,
    )
    commands[0, axis_index] = _strictly_weak_float32(
        magnitude_fraction * positive_threshold,
        threshold=positive_threshold,
    )
    commands[1, axis_index] = -_strictly_weak_float32(
        magnitude_fraction * negative_threshold,
        threshold=negative_threshold,
    )
    return CounterfactualWeakCommandVariants(
        mode=COUNTERFACTUAL_MODE,
        episode_id=episode,
        timestep=step,
        seed=fixed_seed,
        axis_index=axis_index,
        axis=axis,
        magnitude_fraction=magnitude_fraction,
        previous_final_command=commands,
        qvel=np.zeros_like(commands),
    )


def build_execution_feedback_norm_stats(
    *,
    dataset_dir: str | Path,
    train_ids: Sequence[int],
    config: Mapping[str, Any],
    verify_manifest_hashes: bool = True,
) -> dict[str, Any]:
    """Compose parity-preserving qpos/action stats with train-only feedback stats."""

    resolved = resolve_execution_feedback_config(config)
    if not resolved["enabled"]:
        raise ValueError("execution_feedback must be enabled to build feedback stats")
    dataset = _required_directory(dataset_dir, name="dataset_dir")
    manifest_path = Path(resolved["manifest_path"])
    manifest = validate_execution_feedback_manifest(
        manifest_path,
        verify_hashes=bool(verify_manifest_hashes),
        expected_dataset_dir=dataset,
    )
    normalized_train_ids = [
        _nonnegative_integer(value, name="train_id") for value in train_ids
    ]
    if normalized_train_ids != list(manifest["train_ids"]):
        raise ValueError(
            "execution-feedback norm train_ids must exactly match manifest train_ids"
        )
    records = {int(record["episode_id"]): record for record in manifest["episodes"]}

    base_path = Path(resolved["base_norm_stats_path"])
    with base_path.open("rb") as file:
        base = pickle.load(file)
    qpos_mean = _norm_vector(base, "qpos_mean")
    qpos_std = _positive_norm_vector(base, "qpos_std")
    action_mean = _norm_vector(base, "action_mean")
    action_std = _positive_norm_vector(base, "action_std")

    qvel_rows: list[np.ndarray] = []
    command_rows: list[np.ndarray] = []
    example_qpos: np.ndarray | None = None
    example_qvel: np.ndarray | None = None
    example_command: np.ndarray | None = None
    included_steps = 0
    for episode_id in normalized_train_ids:
        record = records.get(episode_id)
        if record is None:
            raise ValueError(
                f"execution-feedback manifest has no train episode {episode_id}"
            )
        episode_path = dataset / f"episode_{episode_id}.hdf5"
        with h5py.File(episode_path, "r") as episode:
            qpos = np.asarray(episode["observations/qpos"][()], dtype=np.float32)
            qvel = np.asarray(episode["observations/qvel"][()], dtype=np.float32)
        sidecar = load_execution_feedback_sidecar(
            record["sidecar_path"],
            expected_episode_id=episode_id,
            expected_length=int(qvel.shape[0]),
        )
        if (
            qpos.shape != qvel.shape
            or qvel.shape != sidecar.previous_final_command.shape
        ):
            raise ValueError(
                f"execution-feedback episode {episode_id} shape mismatch: "
                f"qpos={qpos.shape}, qvel={qvel.shape}, "
                f"command={sidecar.previous_final_command.shape}"
            )
        include = ~(
            sidecar.train_exclude_mask | sidecar.source_time_gap_exceeds_threshold
        )
        if not np.any(include):
            raise ValueError(
                f"execution-feedback episode {episode_id} has no valid norm steps"
            )
        qvel_rows.append(qvel[include])
        command_rows.append(sidecar.previous_final_command[include])
        included_steps += int(np.count_nonzero(include))
        example_qpos = qpos
        example_qvel = qvel
        example_command = sidecar.previous_final_command

    qvel_all = np.concatenate(qvel_rows, axis=0)
    command_all = np.concatenate(command_rows, axis=0)
    qvel_mean, qvel_std = _mean_std(qvel_all)
    command_mean, command_std = _mean_std(command_all)
    assert example_qpos is not None
    assert example_qvel is not None
    assert example_command is not None
    example_proprio = np.concatenate(
        [example_qpos, example_qvel, example_command], axis=1
    ).astype(np.float32)
    return {
        "action_mean": action_mean,
        "action_std": action_std,
        "proprio_mean": np.concatenate([qpos_mean, qvel_mean, command_mean]).astype(
            np.float32
        ),
        "proprio_std": np.concatenate([qpos_std, qvel_std, command_std]).astype(
            np.float32
        ),
        "example_proprio": example_proprio,
        "proprio_keys": np.asarray(
            ["qpos", "qvel", "previous_final_command"], dtype=object
        ),
        "proprio_dim": 12,
        "qpos_only_dim": 4,
        "qpos_mean": qpos_mean,
        "qpos_std": qpos_std,
        "example_qpos": example_qpos,
        "execution_feedback_norm_provenance": {
            "contract_version": "inherit_base_qpos_action_train_only_feedback_v1",
            "base_norm_stats_path": str(base_path),
            "base_norm_stats_sha256": sha256_file(base_path),
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "train_ids": normalized_train_ids,
            "train_step_count": included_steps,
            "qpos_action_stats_source": "inherited_base",
            "qvel_previous_command_stats_source": "train_ids_only",
        },
    }


def causality_age_summary_ns(
    sidecar: ExecutionFeedbackSidecar | CausalCommandAlignment,
) -> dict[str, int | float | None]:
    """Summarize observation-minus-command age over valid causal matches."""

    valid = np.asarray(sidecar.valid_mask, dtype=bool)
    ages = (
        np.asarray(sidecar.observation_timestamp_ns, dtype=np.int64)[valid]
        - np.asarray(sidecar.command_send_timestamp_ns, dtype=np.int64)[valid]
    )
    if ages.size == 0:
        return {
            "count": 0,
            "min": None,
            "mean": None,
            "p50": None,
            "p95": None,
            "max": None,
        }
    return {
        "count": int(ages.size),
        "min": int(np.min(ages)),
        "mean": float(np.mean(ages)),
        "p50": float(np.percentile(ages, 50)),
        "p95": float(np.percentile(ages, 95)),
        "max": int(np.max(ages)),
    }


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA-256 digest for an artifact."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _norm_vector(payload: Mapping[str, Any], key: str) -> np.ndarray:
    if key not in payload:
        raise KeyError(f"base norm stats are missing {key!r}")
    value = np.asarray(payload[key], dtype=np.float32).reshape(-1)
    if value.shape != (len(AXIS_NAMES),) or not np.isfinite(value).all():
        raise ValueError(f"base norm stats {key} must contain four finite values")
    return value.copy()


def _positive_norm_vector(payload: Mapping[str, Any], key: str) -> np.ndarray:
    value = _norm_vector(payload, key)
    if np.any(value <= 0.0):
        raise ValueError(f"base norm stats {key} must be strictly positive")
    return value


def _mean_std(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    array = _command_matrix(value, name="execution feedback norm values")
    mean = np.mean(array, axis=0, dtype=np.float64).astype(np.float32)
    std = np.std(array, axis=0, ddof=0, dtype=np.float64).astype(np.float32)
    return mean, np.maximum(std, np.float32(1e-2)).astype(np.float32)


def _write_sidecar(
    *,
    output_path: Path,
    episode_id: str,
    alignment: CausalCommandAlignment,
    raw_source_length: int,
    resampled_path: Path,
    raw_source_path: Path,
    resampled_sha256: str,
    raw_source_sha256: str,
) -> None:
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    try:
        with temporary.open("wb") as file:
            np.savez_compressed(
                file,
                schema_version=np.asarray(SIDECAR_SCHEMA_VERSION, dtype=np.int64),
                alignment_mode=np.asarray(ALIGNMENT_MODE),
                episode_id=np.asarray(episode_id),
                previous_final_command=alignment.previous_final_command,
                raw_source_index=alignment.raw_source_index,
                command_send_timestamp_ns=alignment.command_send_timestamp_ns,
                observation_timestamp_ns=alignment.observation_timestamp_ns,
                train_exclude_mask=alignment.train_exclude_mask,
                source_time_gap_exceeds_threshold=(
                    alignment.source_time_gap_exceeds_threshold
                ),
                reset_mask=alignment.reset_mask,
                valid_mask=alignment.valid_mask,
                raw_source_length=np.asarray(raw_source_length, dtype=np.int64),
                resampled_path=np.asarray(str(resampled_path)),
                raw_source_path=np.asarray(str(raw_source_path)),
                resampled_sha256=np.asarray(resampled_sha256),
                raw_source_sha256=np.asarray(raw_source_sha256),
            )
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _validate_manifest_episode_record(
    record: Mapping[str, Any],
    *,
    dataset_dir: Path,
    verify_hashes: bool,
) -> None:
    if not isinstance(record, Mapping):
        raise ValueError("execution-feedback episode record must be a mapping")
    episode = _episode_id(record.get("episode_id"))
    sidecar_path = _required_file(record.get("sidecar_path"), name="sidecar_path")
    resampled_path = _required_file(
        record.get("resampled_path"),
        name="resampled_path",
    )
    raw_source_path = _required_file(
        record.get("raw_source_path"),
        name="raw_source_path",
    )
    expected_resampled = (dataset_dir / f"episode_{episode}.hdf5").resolve()
    if resampled_path != expected_resampled:
        raise ValueError(
            f"manifest resampled path for episode {episode} is outside dataset_dir"
        )
    sidecar_sha256 = _sha256_text(
        str(record.get("sidecar_sha256", "")),
        name="sidecar_sha256",
    )
    resampled_sha256 = _sha256_text(
        str(record.get("resampled_sha256", "")),
        name="resampled_sha256",
    )
    raw_source_sha256 = _sha256_text(
        str(record.get("raw_source_sha256", "")),
        name="raw_source_sha256",
    )
    if verify_hashes:
        if sha256_file(sidecar_path) != sidecar_sha256:
            raise ValueError(f"sidecar SHA-256 mismatch for episode {episode}")
        if sha256_file(resampled_path) != resampled_sha256:
            raise ValueError(f"resampled SHA-256 mismatch for episode {episode}")
        if sha256_file(raw_source_path) != raw_source_sha256:
            raise ValueError(f"raw source SHA-256 mismatch for episode {episode}")

    sidecar = load_execution_feedback_sidecar(
        sidecar_path,
        expected_episode_id=episode,
        expected_length=_nonnegative_integer(record.get("length"), name="length"),
    )
    if Path(sidecar.resampled_path).resolve() != resampled_path:
        raise ValueError(f"sidecar resampled provenance mismatch for episode {episode}")
    if Path(sidecar.raw_source_path).resolve() != raw_source_path:
        raise ValueError(f"sidecar raw provenance mismatch for episode {episode}")
    if sidecar.resampled_sha256 != resampled_sha256:
        raise ValueError(f"sidecar resampled hash mismatch for episode {episode}")
    if sidecar.raw_source_sha256 != raw_source_sha256:
        raise ValueError(f"sidecar raw hash mismatch for episode {episode}")
    _validate_sidecar_against_source_files(
        sidecar,
        resampled_path=resampled_path,
        raw_source_path=raw_source_path,
    )
    if _nonnegative_integer(record.get("valid_count"), name="valid_count") != int(
        np.count_nonzero(sidecar.valid_mask)
    ):
        raise ValueError(f"manifest valid_count mismatch for episode {episode}")
    if record.get("reset_counts") != _reset_counts(sidecar):
        raise ValueError(f"manifest reset_counts mismatch for episode {episode}")
    if record.get("causality_age_summary_ns") != causality_age_summary_ns(sidecar):
        raise ValueError(f"manifest causality age mismatch for episode {episode}")


def _reset_counts(
    sidecar: ExecutionFeedbackSidecar | CausalCommandAlignment,
) -> dict[str, int]:
    train_exclude = np.asarray(sidecar.train_exclude_mask, dtype=bool)
    source_gap = np.asarray(sidecar.source_time_gap_exceeds_threshold, dtype=bool)
    reset = np.asarray(sidecar.reset_mask, dtype=bool)
    return {
        "total": int(np.count_nonzero(reset)),
        "episode_start": 1 if reset.size else 0,
        "train_exclude": int(np.count_nonzero(train_exclude)),
        "source_time_gap": int(np.count_nonzero(source_gap)),
        "train_or_gap_union": int(np.count_nonzero(train_exclude | source_gap)),
    }


def _validate_sidecar_against_source_files(
    sidecar: ExecutionFeedbackSidecar,
    *,
    resampled_path: Path,
    raw_source_path: Path,
) -> None:
    with h5py.File(resampled_path, "r") as resampled_file:
        if "metadata" not in resampled_file:
            raise ValueError("manifest resampled source is missing metadata")
        declared_raw_source = _resolve_source_path(
            resampled_file["metadata"].attrs.get("source_dataset_path"),
            resampled_path=resampled_path,
        )
        if declared_raw_source != raw_source_path:
            raise ValueError("manifest raw source differs from resampled metadata")
        observation_ts = _required_dataset(
            resampled_file,
            "diagnostics/source_observation_timestamp_ns",
            file_label="resampled episode",
        )
        train_exclude = _required_dataset(
            resampled_file,
            "diagnostics/train_exclude_mask",
            file_label="resampled episode",
        )
        source_gap = _required_dataset(
            resampled_file,
            "diagnostics/source_time_gap_exceeds_threshold",
            file_label="resampled episode",
        )
    with h5py.File(raw_source_path, "r") as raw_file:
        raw_command = _required_dataset(
            raw_file,
            "diagnostics/commanded_action",
            file_label="raw episode",
        )
        raw_send_ts = _required_dataset(
            raw_file,
            "diagnostics/action_send_timestamp_ns",
            file_label="raw episode",
        )
    expected = align_causal_previous_commands(
        observation_timestamp_ns=observation_ts,
        raw_commanded_action=raw_command,
        raw_action_send_timestamp_ns=raw_send_ts,
        train_exclude_mask=train_exclude,
        source_time_gap_exceeds_threshold=source_gap,
    )
    if sidecar.raw_source_length != int(np.asarray(raw_send_ts).reshape(-1).size):
        raise ValueError("sidecar raw_source_length differs from raw source")
    for field in (
        "previous_final_command",
        "raw_source_index",
        "command_send_timestamp_ns",
        "observation_timestamp_ns",
        "train_exclude_mask",
        "source_time_gap_exceeds_threshold",
        "reset_mask",
        "valid_mask",
    ):
        if not np.array_equal(getattr(sidecar, field), getattr(expected, field)):
            raise ValueError(
                f"sidecar {field} differs from strict-causal source rebuild"
            )


def _disabled_execution_feedback_config() -> dict[str, Any]:
    return {
        "enabled": False,
        "manifest_path": None,
        "base_norm_stats_path": None,
        "counterfactual": _disabled_counterfactual_config(),
    }


def _disabled_counterfactual_config() -> dict[str, Any]:
    return {
        "enabled": False,
        "seed": 0,
        "loss_weight": 0.0,
        "thresholds": {},
    }


def _validate_fixed_counterfactual_contract(cfg: Mapping[str, Any]) -> None:
    if "mode" in cfg and str(cfg["mode"]) != COUNTERFACTUAL_MODE:
        raise ValueError(
            f"execution_feedback.counterfactual.mode must be {COUNTERFACTUAL_MODE!r}"
        )
    if "variant_count" in cfg:
        variant_count = _integer(
            cfg["variant_count"],
            name="counterfactual.variant_count",
        )
        if variant_count != COUNTERFACTUAL_VARIANT_COUNT:
            raise ValueError(
                "execution_feedback.counterfactual.variant_count must be "
                f"{COUNTERFACTUAL_VARIANT_COUNT}"
            )


def _resolve_threshold_source(
    cfg: Mapping[str, Any],
    *,
    prefix: str,
) -> dict[str, dict[str, float]]:
    inline_present = cfg.get("thresholds") is not None
    path_raw = cfg.get("threshold_json")
    path_present = path_raw is not None and str(path_raw).strip() != ""
    if inline_present == path_present:
        raise ValueError(
            f"{prefix} requires exactly one of thresholds or threshold_json"
        )
    if inline_present:
        payload: Any = cfg["thresholds"]
    else:
        threshold_path = _required_file(path_raw, name="counterfactual.threshold_json")
        payload = json.loads(threshold_path.read_text(encoding="utf-8"))
        _validate_direct_action_domain_metadata(payload, prefix=prefix)
    if isinstance(payload, Mapping) and "deadzone_action" in payload:
        payload = payload["deadzone_action"]
    return _normalize_thresholds(payload, prefix=f"{prefix}.thresholds")


def _normalize_thresholds(
    raw: Any,
    *,
    prefix: str,
) -> dict[str, dict[str, float]]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{prefix} must be a mapping")
    thresholds: dict[str, dict[str, float]] = {}
    for axis in AXIS_NAMES:
        axis_raw = raw.get(axis)
        if not isinstance(axis_raw, Mapping):
            raise ValueError(f"{prefix} missing axis {axis!r}")
        thresholds[axis] = {
            "pos": _positive_threshold(axis_raw.get("pos"), name=f"{axis}.pos"),
            "neg": _positive_threshold(axis_raw.get("neg"), name=f"{axis}.neg"),
        }
    return thresholds


def _positive_threshold(value: Any, *, name: str) -> float:
    if isinstance(value, Mapping):
        if "threshold_action_abs" in value:
            value = value["threshold_action_abs"]
        elif "value" in value:
            value = value["value"]
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"counterfactual threshold {name} must be finite and positive")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"counterfactual threshold {name} must be finite and positive"
        ) from exc
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"counterfactual threshold {name} must be finite and positive")
    return result


def _validate_direct_action_domain_metadata(payload: Any, *, prefix: str) -> None:
    if not isinstance(payload, Mapping):
        return
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        return
    action_domain = metadata.get("action_domain")
    if action_domain is not None and str(action_domain) != "direct_policy_output":
        raise ValueError(
            f"{prefix} threshold_json is not in direct policy output domain"
        )
    scale = metadata.get("policy_action_scale")
    if scale is not None:
        scale_arr = np.asarray(scale, dtype=np.float64)
        if scale_arr.shape != (len(AXIS_NAMES),) or not np.array_equal(
            scale_arr,
            np.ones(len(AXIS_NAMES), dtype=np.float64),
        ):
            raise ValueError(f"{prefix} threshold_json must use identity action scale")


def _strictly_weak_float32(value: float, *, threshold: float) -> np.float32:
    result = np.float32(value)
    threshold32 = np.float32(threshold)
    if result >= threshold32:
        result = np.nextafter(threshold32, np.float32(0.0))
    return result


def _required_dataset(
    h5_file: h5py.File,
    path: str,
    *,
    file_label: str,
) -> np.ndarray:
    if path not in h5_file:
        raise ValueError(f"{file_label} is missing {path}")
    return np.asarray(h5_file[path][()])


def _resolve_source_path(value: Any, *, resampled_path: Path) -> Path:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if value is None or not str(value).strip():
        raise ValueError("resampled metadata.source_dataset_path is missing")
    source = Path(str(value)).expanduser()
    if not source.is_absolute():
        source = resampled_path.parent / source
    return _required_file(source, name="metadata.source_dataset_path")


def _command_matrix(value: Any, *, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 2 or raw.shape[1] != len(AXIS_NAMES):
        raise ValueError(
            f"{name} must have shape (T, {len(AXIS_NAMES)}), got {raw.shape}"
        )
    if raw.dtype.kind not in {"f", "i", "u"}:
        raise ValueError(f"{name} must be numeric")
    array = raw.astype(np.float32, copy=False)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains nonfinite values")
    return array


def _timestamp_vector(
    value: Any,
    *,
    name: str,
    strictly_increasing: bool,
) -> np.ndarray:
    array = _integer_vector(value, n_steps=None, name=name, allow_negative_one=False)
    if array.size > 1:
        invalid = (
            array[1:] <= array[:-1] if strictly_increasing else array[1:] < array[:-1]
        )
        if np.any(invalid):
            qualifier = (
                "strictly increasing" if strictly_increasing else "nondecreasing"
            )
            raise ValueError(f"{name} must be {qualifier}")
    return array


def _integer_vector(
    value: Any,
    *,
    n_steps: int | None,
    name: str,
    allow_negative_one: bool,
) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.dtype.kind not in {"i", "u"}:
        raise ValueError(f"{name} must be a one-dimensional integer array")
    if raw.dtype.kind == "u" and raw.size and np.max(raw) > np.iinfo(np.int64).max:
        raise ValueError(f"{name} exceeds int64 range")
    array = raw.astype(np.int64, copy=False)
    if n_steps is not None and array.size != int(n_steps):
        raise ValueError(f"{name} length must be {n_steps}, got {array.size}")
    minimum = -1 if allow_negative_one else 0
    if np.any(array < minimum):
        raise ValueError(f"{name} contains an invalid negative value")
    return array


def _bool_mask(value: Any, *, n_steps: int, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.size != int(n_steps):
        raise ValueError(f"{name} must have shape ({n_steps},), got {raw.shape}")
    if raw.dtype.kind not in {"b", "i", "u"} or np.any((raw != 0) & (raw != 1)):
        raise ValueError(f"{name} must contain only boolean or 0/1 values")
    return raw.astype(bool, copy=True)


def _npz_scalar_text(payload: Any, key: str) -> str:
    value = np.asarray(payload[key])
    if value.shape != () or value.dtype.kind not in {"U", "S"}:
        raise ValueError(f"sidecar {key} must be a scalar string")
    item = value.item()
    return item.decode("utf-8") if isinstance(item, bytes) else str(item)


def _npz_scalar_int(payload: Any, key: str) -> int:
    value = np.asarray(payload[key])
    if value.shape != () or value.dtype.kind not in {"i", "u"}:
        raise ValueError(f"sidecar {key} must be a scalar integer")
    return int(value.item())


def _episode_id(value: Any) -> str:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("episode_id must be a nonnegative integer")
    if isinstance(value, str) and value.startswith("episode_"):
        value = value.removeprefix("episode_")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("episode_id must be a nonnegative integer") from exc
    if result < 0 or str(value).strip() != str(result):
        raise ValueError("episode_id must be a canonical nonnegative integer")
    return str(result)


def _episode_id_list(value: Any, *, name: str) -> list[int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of episode IDs")
    ids = [int(_episode_id(item)) for item in value]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{name} must not contain duplicate episode IDs")
    return ids


def _ordered_union(first: Sequence[int], second: Sequence[int]) -> list[int]:
    result: list[int] = []
    seen: set[int] = set()
    for item in [*first, *second]:
        value = int(item)
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _required_file(value: Any, *, name: str) -> Path:
    if value is None or not str(value).strip():
        raise ValueError(f"execution_feedback.{name} is required")
    path = Path(str(value)).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"execution_feedback.{name} does not exist: {path}")
    return path


def _required_directory(value: Any, *, name: str) -> Path:
    if value is None or not str(value).strip():
        raise ValueError(f"execution_feedback.{name} is required")
    path = Path(str(value)).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"execution_feedback.{name} does not exist: {path}")
    return path


def _strict_bool(value: Any, *, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"execution_feedback.{name} must be a boolean")
    return bool(value)


def _integer(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"execution_feedback.{name} must be an integer")
    return int(value)


def _nonnegative_integer(value: Any, *, name: str) -> int:
    result = _integer(value, name=name)
    if result < 0:
        raise ValueError(f"execution_feedback.{name} must be nonnegative")
    return result


def _finite_nonnegative_float(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"execution_feedback.{name} must be finite and nonnegative")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"execution_feedback.{name} must be finite and nonnegative"
        ) from exc
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"execution_feedback.{name} must be finite and nonnegative")
    return result


def _sha256_text(value: str, *, name: str) -> str:
    normalized = str(value).lower()
    if len(normalized) != 64 or any(
        char not in "0123456789abcdef" for char in normalized
    ):
        raise ValueError(f"{name} must be a SHA-256 hex digest")
    return normalized
