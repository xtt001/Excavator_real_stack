"""Derive deterministic single-demo intent events from action trajectories.

The sidecar separates intent active at an onset from directions that appear
later in the same demonstrated sequence.  Its support set belongs to this one
recording only: it is not task-wide support, correctness, or safety evidence.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml

from testbed.policies.deadzone_eval import (
    AXIS_NAMES,
    effective_direction_mask,
)

SCHEMA_VERSION = "single_demo_intent_events_v2"
MANIFEST_FILENAME = "expert_intent_events_manifest.json"
EVENTS_FILENAME = "expert_intent_events.jsonl"
EVENTS_CSV_FILENAME = "expert_intent_events.csv"
SUMMARY_FILENAME = "expert_intent_events_summary.json"
WINDOWS = {
    "immediate_0_1": (0, 1),
    "near_2_5": (2, 5),
    "near_6_10": (6, 10),
}

# These IDs are documented as selection-held-out or sealed source episodes.
# Composite train/validation views use different IDs and remain unaffected.
SEALED_TEST_EPISODE_IDS = frozenset({*range(105, 110), *range(156, 176)})
TEST_ROLE_KEYS = ("test_ids", "sealed_test_ids", "heldout_ids", "held_out_ids")


def load_episode_roles_from_split(
    path: str | Path,
    *,
    expected_dataset_dir: str | Path | None = None,
) -> tuple[list[int], list[int]]:
    """Load an explicit train/validation split and reject every test role."""

    split_path = _required_file(path, name="split_path")
    payload = yaml.safe_load(split_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ValueError("split file must contain a mapping")
    for key in TEST_ROLE_KEYS:
        values = _episode_ids(payload.get(key, []), name=key, allow_empty=True)
        if values:
            raise ValueError(f"split contains forbidden test role {key}: {values}")
    if expected_dataset_dir is not None and payload.get("dataset_dir") is not None:
        configured = Path(str(payload["dataset_dir"])).expanduser().resolve()
        expected = Path(expected_dataset_dir).expanduser().resolve()
        if configured != expected:
            raise ValueError(
                f"split dataset_dir does not match requested dataset: {configured}"
            )
    train_ids = _episode_ids(payload.get("train_ids"), name="train_ids")
    validation_ids = _episode_ids(payload.get("val_ids"), name="val_ids")
    _validate_episode_roles(train_ids, validation_ids)
    return train_ids, validation_ids


def build_expert_intent_event_sidecar(
    *,
    dataset_dir: str | Path,
    output_dir: str | Path,
    thresholds: Mapping[str, Mapping[str, float]],
    threshold_source_path: str | Path,
    train_episode_ids: Sequence[int],
    validation_episode_ids: Sequence[int],
    support_horizon_ticks: int,
    split_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build additive event artifacts without changing any source HDF5."""

    dataset = Path(dataset_dir).expanduser().resolve()
    if not dataset.is_dir():
        raise FileNotFoundError(f"dataset_dir does not exist: {dataset}")
    threshold_source = _required_file(
        threshold_source_path, name="threshold_source_path"
    )
    train_ids = _episode_ids(train_episode_ids, name="train_episode_ids")
    validation_ids = _episode_ids(validation_episode_ids, name="validation_episode_ids")
    _validate_episode_roles(train_ids, validation_ids)
    horizon = int(support_horizon_ticks)
    if horizon < 11:
        raise ValueError("support_horizon_ticks must be at least 11")
    normalized_thresholds = _normalize_thresholds(thresholds)
    resolved_split = (
        _required_file(split_path, name="split_path")
        if split_path is not None
        else None
    )
    if resolved_split is not None:
        split_train_ids, split_validation_ids = load_episode_roles_from_split(
            resolved_split,
            expected_dataset_dir=dataset,
        )
        if split_train_ids != train_ids or split_validation_ids != validation_ids:
            raise ValueError(
                "supplied train/validation IDs do not exactly match split file"
            )

    episode_sources: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    role_by_id = {episode_id: "train" for episode_id in train_ids}
    role_by_id.update({episode_id: "validation" for episode_id in validation_ids})
    for episode_id in [*train_ids, *validation_ids]:
        source_path = _required_file(
            dataset / f"episode_{episode_id}.hdf5",
            name=f"episode_{episode_id}",
        )
        source_hash = sha256_file(source_path)
        action, qpos, qvel = _read_episode_arrays(source_path)
        episode_events = derive_expert_intent_events(
            episode_id=episode_id,
            split=role_by_id[episode_id],
            action=action,
            qpos=qpos,
            qvel=qvel,
            thresholds=normalized_thresholds,
            support_horizon_ticks=horizon,
            source_path=source_path,
            source_sha256=source_hash,
        )
        events.extend(episode_events)
        episode_sources.append(
            {
                "episode_id": episode_id,
                "split": role_by_id[episode_id],
                "path": str(source_path),
                "sha256": source_hash,
                "steps": int(action.shape[0]),
                "event_count": len(episode_events),
            }
        )

    summary = summarize_expert_intent_events(
        events,
        train_episode_ids=train_ids,
        validation_episode_ids=validation_ids,
    )
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_events_jsonl(output / EVENTS_FILENAME, events)
    _write_events_csv(output / EVENTS_CSV_FILENAME, events)
    _write_json_atomic(output / SUMMARY_FILENAME, summary)
    artifact_hashes = {
        EVENTS_FILENAME: sha256_file(output / EVENTS_FILENAME),
        EVENTS_CSV_FILENAME: sha256_file(output / EVENTS_CSV_FILENAME),
        SUMMARY_FILENAME: sha256_file(output / SUMMARY_FILENAME),
    }
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "semantics": {
            "event_anchor": "any idle-to-effective axis/direction transition",
            "window_bounds": "inclusive tick offsets from event onset",
            "support_horizon": "tick count including event onset",
            "response_claim": "none",
        },
        "dataset_dir": str(dataset),
        "train_ids": train_ids,
        "validation_ids": validation_ids,
        "episode_ids": [*train_ids, *validation_ids],
        "support_horizon_ticks": horizon,
        "windows": {key: list(value) for key, value in WINDOWS.items()},
        "thresholds": normalized_thresholds,
        "threshold_source_path": str(threshold_source),
        "threshold_source_sha256": sha256_file(threshold_source),
        "split_path": str(resolved_split) if resolved_split is not None else None,
        "split_sha256": sha256_file(resolved_split)
        if resolved_split is not None
        else None,
        "episodes": episode_sources,
        "event_count": len(events),
        "artifacts": artifact_hashes,
    }
    _write_json_atomic(output / MANIFEST_FILENAME, manifest)
    return manifest


def derive_expert_intent_events(
    *,
    episode_id: int,
    split: str,
    action: np.ndarray,
    qpos: np.ndarray,
    qvel: np.ndarray,
    thresholds: Mapping[str, Mapping[str, float]],
    support_horizon_ticks: int,
    source_path: str | Path,
    source_sha256: str,
) -> list[dict[str, Any]]:
    """Derive one joint event for every unique transition timestep."""

    actions = np.asarray(action, dtype=np.float32)
    positions = np.asarray(qpos, dtype=np.float32)
    velocities = np.asarray(qvel, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != len(AXIS_NAMES):
        raise ValueError(f"action must have shape (T, {len(AXIS_NAMES)})")
    if positions.shape != actions.shape or velocities.shape != actions.shape:
        raise ValueError(
            "action/qpos/qvel shapes differ: "
            f"{actions.shape}, {positions.shape}, {velocities.shape}"
        )
    horizon = int(support_horizon_ticks)
    if horizon < 11:
        raise ValueError("support_horizon_ticks must be at least 11")
    effective = effective_direction_mask(actions, dict(thresholds))
    previous = np.zeros_like(effective)
    previous[1:] = effective[:-1]
    transitions = effective & ~previous
    onset_steps = np.flatnonzero(np.any(transitions, axis=(1, 2))).tolist()
    path = str(Path(source_path).expanduser().resolve())

    result: list[dict[str, Any]] = []
    for event_index, onset_step in enumerate(onset_steps):
        end = min(actions.shape[0], onset_step + horizon)
        support = effective[onset_step:end]
        newly_effective = _labels_from_mask(transitions[onset_step])
        window_intents = {
            name: _labels_from_window(support, start, stop)
            for name, (start, stop) in WINDOWS.items()
        }
        supported = _labels_from_mask(np.any(support, axis=0))
        details = _direction_details(
            effective=effective,
            event_onset_step=onset_step,
            support_end_step=end,
        )
        supported_details = [detail for detail in details if detail["supported"]]
        motif_groups: dict[int, list[str]] = {}
        for detail in supported_details:
            delay = int(detail["onset_delay_ticks"])
            motif_groups.setdefault(delay, []).append(str(detail["direction"]))
        ordered_motif = [
            {"onset_delay_ticks": delay, "directions": labels}
            for delay, labels in sorted(motif_groups.items())
        ]
        motif = ">".join(
            f"{group['onset_delay_ticks']}:" + ",".join(group["directions"])
            for group in ordered_motif
        )
        releases = [
            int(detail["release_step_exclusive"])
            for detail in supported_details
            if detail["release_step_exclusive"] is not None
        ]
        axis_delays = {
            axis: _axis_first_delay(supported_details, axis=axis) for axis in AXIS_NAMES
        }
        event = {
            "schema_version": SCHEMA_VERSION,
            "event_id": f"episode_{int(episode_id)}:event_{event_index:04d}:step_{onset_step}",
            "episode_id": int(episode_id),
            "split": str(split),
            "event_index": event_index,
            "onset_step": int(onset_step),
            "support_end_step_exclusive": int(end),
            "support_horizon_requested_ticks": horizon,
            "support_horizon_observed_ticks": int(end - onset_step),
            "newly_effective_directions": newly_effective,
            "anchor_intent": _labels_from_mask(effective[onset_step]),
            "immediate_intent_0_1": window_intents["immediate_0_1"],
            "near_intent_2_5": window_intents["near_2_5"],
            "near_intent_6_10": window_intents["near_6_10"],
            "single_demo_event_support_directions": supported,
            "ordered_first_onset_motif": ordered_motif,
            "motif": motif,
            "axis_first_onset_delay_ticks": axis_delays,
            "direction_details": supported_details,
            "release_bounds": {
                "earliest_release_step_exclusive": min(releases) if releases else None,
                "latest_release_step_exclusive": max(releases) if releases else None,
                "right_censored_direction_count": sum(
                    detail["release_step_exclusive"] is None
                    for detail in supported_details
                ),
            },
            "qpos_at_onset": {
                axis: float(positions[onset_step, axis_index])
                for axis_index, axis in enumerate(AXIS_NAMES)
            },
            "qvel_at_onset": {
                axis: float(velocities[onset_step, axis_index])
                for axis_index, axis in enumerate(AXIS_NAMES)
            },
            "source": {"path": path, "sha256": str(source_sha256)},
        }
        result.append(event)
    return result


def summarize_expert_intent_events(
    events: Sequence[Mapping[str, Any]],
    *,
    train_episode_ids: Sequence[int],
    validation_episode_ids: Sequence[int],
) -> dict[str, Any]:
    """Create deterministic split and motif counts for quick review."""

    by_split: dict[str, Any] = {}
    for split, episode_ids in (
        ("train", train_episode_ids),
        ("validation", validation_episode_ids),
    ):
        selected = [event for event in events if event["split"] == split]
        first_by_episode: dict[int, Mapping[str, Any]] = {}
        for event in selected:
            first_by_episode.setdefault(int(event["episode_id"]), event)
        first_events = list(first_by_episode.values())
        by_split[split] = {
            "episode_count": len(episode_ids),
            "event_count": len(selected),
            "episodes_with_events": len(
                {int(event["episode_id"]) for event in selected}
            ),
            "motif_counts": dict(
                sorted(Counter(str(event["motif"]) for event in selected).items())
            ),
            "newly_effective_counts": dict(
                sorted(
                    Counter(
                        direction
                        for event in selected
                        for direction in event["newly_effective_directions"]
                    ).items()
                )
            ),
            "first_event": {
                "event_count": len(first_events),
                "anchor_intent_set_counts": _count_direction_sets(
                    first_events, field="anchor_intent"
                ),
                "single_demo_event_support_direction_set_counts": _count_direction_sets(
                    first_events, field="single_demo_event_support_directions"
                ),
                "ordered_direction_motif_counts": dict(
                    sorted(
                        Counter(
                            _motif_without_delays(event) for event in first_events
                        ).items()
                    )
                ),
            },
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "event_count": len(events),
        "by_split": by_split,
    }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_episode_arrays(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        action = np.asarray(handle["/action"][()], dtype=np.float32)
        qpos = np.asarray(handle["/observations/qpos"][()], dtype=np.float32)
        qvel = np.asarray(handle["/observations/qvel"][()], dtype=np.float32)
    return action, qpos, qvel


def _direction_details(
    *,
    effective: np.ndarray,
    event_onset_step: int,
    support_end_step: int,
) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    support = effective[event_onset_step:support_end_step]
    for axis_index, axis in enumerate(AXIS_NAMES):
        for direction_index, direction in enumerate(("pos", "neg")):
            active_offsets = np.flatnonzero(support[:, axis_index, direction_index])
            if active_offsets.size == 0:
                continue
            first_observed_offset = int(active_offsets[0])
            run_start = event_onset_step + first_observed_offset
            if first_observed_offset == 0:
                while (
                    run_start > 0
                    and effective[run_start - 1, axis_index, direction_index]
                ):
                    run_start -= 1
            delay = run_start - event_onset_step
            run_end = run_start
            while (
                run_end < effective.shape[0]
                and effective[run_end, axis_index, direction_index]
            ):
                run_end += 1
            idle_start = run_start
            while idle_start > 0 and not bool(
                np.any(effective[idle_start - 1, axis_index])
            ):
                idle_start -= 1
            details.append(
                {
                    "axis": axis,
                    "direction": f"{axis}{'+' if direction == 'pos' else '-'}",
                    "sign": direction,
                    "supported": True,
                    "active_at_event_onset": bool(
                        effective[event_onset_step, axis_index, direction_index]
                    ),
                    "onset_delay_ticks": delay,
                    "onset_step": run_start,
                    "persistence_ticks": run_end - run_start,
                    "pre_idle_dwell_ticks": run_start - idle_start,
                    "release_step_exclusive": run_end
                    if run_end < effective.shape[0]
                    else None,
                    "right_censored": run_end == effective.shape[0],
                }
            )
    return details


def _labels_from_window(mask: np.ndarray, start: int, stop: int) -> list[str]:
    if start >= mask.shape[0]:
        return []
    observed = mask[start : min(mask.shape[0], stop + 1)]
    return _labels_from_mask(np.any(observed, axis=0))


def _labels_from_mask(mask: np.ndarray) -> list[str]:
    labels: list[str] = []
    for axis_index, axis in enumerate(AXIS_NAMES):
        if bool(mask[axis_index, 0]):
            labels.append(f"{axis}+")
        if bool(mask[axis_index, 1]):
            labels.append(f"{axis}-")
    return labels


def _axis_first_delay(details: Sequence[Mapping[str, Any]], *, axis: str) -> int | None:
    delays = [
        int(detail["onset_delay_ticks"]) for detail in details if detail["axis"] == axis
    ]
    return min(delays) if delays else None


def _count_direction_sets(
    events: Sequence[Mapping[str, Any]],
    *,
    field: str,
) -> dict[str, int]:
    counts = Counter(",".join(event[field]) or "idle" for event in events)
    return dict(sorted(counts.items()))


def _motif_without_delays(event: Mapping[str, Any]) -> str:
    groups = event["ordered_first_onset_motif"]
    return ">".join(",".join(group["directions"]) for group in groups) or "idle"


def _normalize_thresholds(
    thresholds: Mapping[str, Mapping[str, float]],
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for axis in AXIS_NAMES:
        raw = thresholds.get(axis)
        if not isinstance(raw, Mapping):
            raise ValueError(f"thresholds are missing axis {axis}")
        result[axis] = {}
        for direction in ("pos", "neg"):
            value = float(raw.get(direction, 0.0))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"threshold {axis}/{direction} must be positive")
            result[axis][direction] = value
    return result


def _episode_ids(
    values: Sequence[int] | Any,
    *,
    name: str,
    allow_empty: bool = False,
) -> list[int]:
    if values is None or isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a list of episode IDs")
    try:
        result = [int(value) for value in values]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a list of episode IDs") from error
    if len(result) != len(set(result)):
        raise ValueError(f"{name} contains duplicate episode IDs")
    if not result and not allow_empty:
        raise ValueError(f"{name} must not be empty")
    return result


def _validate_episode_roles(
    train_ids: Sequence[int], validation_ids: Sequence[int]
) -> None:
    overlap = sorted(set(train_ids) & set(validation_ids))
    if overlap:
        raise ValueError(f"train and validation episode IDs overlap: {overlap}")
    sealed = sorted((set(train_ids) | set(validation_ids)) & SEALED_TEST_EPISODE_IDS)
    if sealed:
        raise ValueError(f"sealed/test episode IDs are forbidden: {sealed}")


def _required_file(path: str | Path, *, name: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{name} does not exist: {resolved}")
    return resolved


def _write_events_jsonl(path: Path, events: Sequence[Mapping[str, Any]]) -> None:
    content = "".join(
        json.dumps(event, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for event in events
    )
    _write_text_atomic(path, content)


def _write_events_csv(path: Path, events: Sequence[Mapping[str, Any]]) -> None:
    fields = (
        "event_id",
        "episode_id",
        "split",
        "onset_step",
        "support_end_step_exclusive",
        "newly_effective_directions",
        "anchor_intent",
        "immediate_intent_0_1",
        "near_intent_2_5",
        "near_intent_6_10",
        "single_demo_event_support_directions",
        "motif",
        "qpos_at_onset",
        "qvel_at_onset",
        "source_sha256",
    )
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            for event in events:
                writer.writerow(
                    {
                        **{key: event[key] for key in fields[:5]},
                        **{key: "|".join(event[key]) for key in fields[5:11]},
                        "motif": event["motif"],
                        "qpos_at_onset": json.dumps(
                            event["qpos_at_onset"], sort_keys=True
                        ),
                        "qvel_at_onset": json.dumps(
                            event["qvel_at_onset"], sort_keys=True
                        ),
                        "source_sha256": event["source"]["sha256"],
                    }
                )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    _write_text_atomic(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _write_text_atomic(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


__all__ = [
    "EVENTS_CSV_FILENAME",
    "EVENTS_FILENAME",
    "MANIFEST_FILENAME",
    "SCHEMA_VERSION",
    "SUMMARY_FILENAME",
    "build_expert_intent_event_sidecar",
    "derive_expert_intent_events",
    "load_episode_roles_from_split",
    "sha256_file",
    "summarize_expert_intent_events",
]
