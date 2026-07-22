"""Causal short visual history for offline and live policy input assembly.

The helper intentionally owns only history assembly.  It does not decide an
action, apply a deadzone, or alter the observation contract.  A caller feeds
one image per configured camera at a time and receives a fixed-length,
oldest-to-newest window.  The first frame is repeated while the window warms
up, but the repeated entries are marked invalid so a policy can distinguish
startup padding from observed frames.

This module is request-local until a later slice explicitly wires it into the
dataset or adapter.  Leaving the current single-frame path untouched is the
rollback mechanism for temporal experiments.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


def resolve_temporal_input_config(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    """Resolve the opt-in visual history input contract.

    The default is deliberately disabled so existing single-frame ACT
    checkpoints and runtime paths remain byte-for-byte compatible.  Both
    ``history_steps`` (the public name) and the helper's older
    ``history_length`` spelling are accepted when loading an experiment
    config; the resolved result always uses ``history_steps``.
    """

    cfg = dict(raw or {})
    enabled = bool(cfg.get("enabled", False))
    value = cfg.get("history_steps", cfg.get("history_length", 4))
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError("temporal_input.history_steps must be an integer")
    history_steps = int(value)
    if history_steps <= 0:
        raise ValueError("temporal_input.history_steps must be positive")
    return {"enabled": enabled, "history_steps": history_steps}


def causal_window_indices(
    total_steps: int,
    target_step: int,
    history_length: int,
) -> np.ndarray:
    """Return oldest-to-newest causal indices ending at ``target_step``.

    The result always has ``history_length`` entries.  At episode startup,
    indices before zero are replaced with zero (the episode's first frame),
    while an invalid target or a target beyond the episode raises.  The
    function never produces an index greater than ``target_step`` and is
    suitable for building an HDF5 temporal sample without reading future
    observations.
    """

    steps = _positive_int(total_steps, name="total_steps")
    target = _nonnegative_int(target_step, name="target_step")
    length = _positive_int(history_length, name="history_length")
    if target >= steps:
        raise ValueError(
            f"target_step must be less than total_steps {steps}, got {target}"
        )

    indices = np.arange(target - length + 1, target + 1, dtype=np.int64)
    return np.maximum(indices, 0)


@dataclass(frozen=True)
class VisualHistorySnapshot:
    """One causal snapshot for all configured cameras.

    Each camera value in ``images`` has shape ``(history_length, *image_shape)``
    and is ordered oldest-to-newest.  ``valid_mask`` is false only for startup
    padding.  ``age_steps`` is the number of accepted camera frames since the
    frame was observed (zero is the newest frame and ``-1`` denotes padding).
    The dictionaries are fresh per snapshot and the arrays are read-only.
    """

    camera_names: tuple[str, ...]
    history_length: int
    images: Mapping[str, np.ndarray]
    timestamps_ns: Mapping[str, np.ndarray]
    valid_mask: Mapping[str, np.ndarray]
    age_steps: Mapping[str, np.ndarray]
    accepted: Mapping[str, bool]
    duplicate_timestamp: Mapping[str, bool]


@dataclass
class _Frame:
    timestamp_ns: int
    accepted_index: int
    image: np.ndarray


@dataclass
class _CameraState:
    frames: deque[_Frame]
    accepted_count: int
    image_shape: tuple[int, ...]


@dataclass(frozen=True)
class CausalVisualHistoryState:
    """Deep-copied mutable state for deterministic branch replay."""

    camera_states: Mapping[str, _CameraState]


class CausalVisualHistory:
    """Maintain a fixed-length causal image ring buffer per camera.

    ``append`` is atomic with respect to timestamp validation: all required
    camera keys and timestamp order are checked before any state is changed.
    Equal timestamps are treated as duplicate observations and are ignored;
    an older timestamp raises because accepting it would leak a stale frame
    into a future window.  Extra camera keys are ignored, matching the
    existing policy contract where a policy selects its configured cameras.
    """

    def __init__(
        self,
        camera_names: Sequence[str],
        *,
        history_length: int = 4,
    ) -> None:
        names = tuple(str(name) for name in camera_names)
        if not names:
            raise ValueError("camera_names must not be empty")
        if len(set(names)) != len(names):
            raise ValueError("camera_names must be unique")
        if any(not name for name in names):
            raise ValueError("camera_names must contain non-empty names")
        length = _positive_int(history_length, name="history_length")

        self._camera_names = names
        self._history_length = length
        self._states: dict[str, _CameraState] = {}

    @property
    def camera_names(self) -> tuple[str, ...]:
        """Configured camera order used by future policy adapters."""

        return self._camera_names

    @property
    def history_length(self) -> int:
        return self._history_length

    def reset(self) -> None:
        """Clear all camera histories at an episode or transport boundary."""

        self._states.clear()

    def snapshot_state(self) -> CausalVisualHistoryState:
        """Return a no-alias copy of every retained camera frame."""

        return CausalVisualHistoryState(
            camera_states=self._clone_camera_states(self._states)
        )

    def restore_state(self, state: CausalVisualHistoryState) -> None:
        """Restore a state produced by :meth:`snapshot_state`."""

        if not isinstance(state, CausalVisualHistoryState):
            raise TypeError("state must be CausalVisualHistoryState")
        unknown = set(state.camera_states) - set(self._camera_names)
        if unknown:
            raise ValueError(
                "visual history state contains unknown cameras: "
                + ", ".join(sorted(unknown))
            )
        self._states = self._clone_camera_states(state.camera_states)

    def _clone_camera_states(
        self,
        states: Mapping[str, _CameraState],
    ) -> dict[str, _CameraState]:
        return {
            camera_name: _CameraState(
                frames=deque(
                    (
                        _Frame(
                            timestamp_ns=int(frame.timestamp_ns),
                            accepted_index=int(frame.accepted_index),
                            image=np.asarray(frame.image).copy(),
                        )
                        for frame in camera_state.frames
                    ),
                    maxlen=self._history_length,
                ),
                accepted_count=int(camera_state.accepted_count),
                image_shape=tuple(camera_state.image_shape),
            )
            for camera_name, camera_state in states.items()
        }

    def append(
        self,
        images: Mapping[str, np.ndarray],
        timestamps_ns: Mapping[str, int | np.integer[Any]],
    ) -> VisualHistorySnapshot:
        """Append one observation and return its causal padded snapshot.

        ``images`` and ``timestamps_ns`` must contain every configured camera.
        The input arrays are copied before being retained, so mutating a caller
        buffer after this call cannot change a later snapshot.  A duplicate
        timestamp returns the unchanged window for that camera and reports
        ``duplicate_timestamp[camera] == True``.
        """

        normalized_images = self._validate_images(images)
        normalized_timestamps = self._validate_timestamps(timestamps_ns)

        # Validate all camera timestamps first.  This prevents a new frame for
        # one camera from being committed when another camera is out of order.
        for camera_name in self._camera_names:
            state = self._states.get(camera_name)
            if state is None:
                continue
            last_timestamp = state.frames[-1].timestamp_ns
            timestamp = normalized_timestamps[camera_name]
            if timestamp < last_timestamp:
                raise ValueError(
                    f"timestamp for camera {camera_name!r} must be monotonic: "
                    f"{timestamp} < {last_timestamp}"
                )
            if normalized_images[camera_name].shape != state.image_shape:
                raise ValueError(
                    f"image shape for camera {camera_name!r} changed from "
                    f"{state.image_shape} to {normalized_images[camera_name].shape}"
                )

        accepted: dict[str, bool] = {}
        duplicate: dict[str, bool] = {}
        for camera_name in self._camera_names:
            image = normalized_images[camera_name]
            timestamp = normalized_timestamps[camera_name]
            state = self._states.get(camera_name)
            if state is None:
                state = _CameraState(
                    frames=deque(maxlen=self._history_length),
                    accepted_count=0,
                    image_shape=tuple(image.shape),
                )
                self._states[camera_name] = state

            if state.frames and timestamp == state.frames[-1].timestamp_ns:
                accepted[camera_name] = False
                duplicate[camera_name] = True
                continue

            state.frames.append(
                _Frame(
                    timestamp_ns=timestamp,
                    accepted_index=state.accepted_count,
                    image=image.copy(),
                )
            )
            state.accepted_count += 1
            accepted[camera_name] = True
            duplicate[camera_name] = False

        return self.snapshot(accepted=accepted, duplicate_timestamp=duplicate)

    def snapshot(
        self,
        *,
        accepted: Mapping[str, bool] | None = None,
        duplicate_timestamp: Mapping[str, bool] | None = None,
    ) -> VisualHistorySnapshot:
        """Return the latest padded snapshot without adding a new frame.

        ``snapshot`` is useful after a duplicate timestamp.  It raises before
        the first append because there is no image with which to pad history.
        ``accepted`` and ``duplicate_timestamp`` are diagnostic annotations for
        the most recent append; omitted values default to false.
        """

        if set(self._states) != set(self._camera_names):
            missing = [name for name in self._camera_names if name not in self._states]
            raise RuntimeError(
                "cannot snapshot before every configured camera has a frame: "
                + ", ".join(missing)
            )

        accepted_values = {
            name: bool((accepted or {}).get(name, False))
            for name in self._camera_names
        }
        duplicate_values = {
            name: bool((duplicate_timestamp or {}).get(name, False))
            for name in self._camera_names
        }

        images: dict[str, np.ndarray] = {}
        timestamps: dict[str, np.ndarray] = {}
        valid: dict[str, np.ndarray] = {}
        ages: dict[str, np.ndarray] = {}
        for camera_name in self._camera_names:
            state = self._states[camera_name]
            frames = list(state.frames)
            pad_count = self._history_length - len(frames)
            first = frames[0]
            image_rows = [first.image] * pad_count + [frame.image for frame in frames]
            timestamp_rows = [first.timestamp_ns] * pad_count + [
                frame.timestamp_ns for frame in frames
            ]
            age_rows = [-1] * pad_count + [
                state.accepted_count - 1 - frame.accepted_index for frame in frames
            ]

            images[camera_name] = _readonly_copy(np.stack(image_rows, axis=0))
            timestamps[camera_name] = _readonly_copy(
                np.asarray(timestamp_rows, dtype=np.int64)
            )
            valid[camera_name] = _readonly_copy(
                np.asarray([False] * pad_count + [True] * len(frames), dtype=bool)
            )
            ages[camera_name] = _readonly_copy(np.asarray(age_rows, dtype=np.int64))

        return VisualHistorySnapshot(
            camera_names=self._camera_names,
            history_length=self._history_length,
            images=images,
            timestamps_ns=timestamps,
            valid_mask=valid,
            age_steps=ages,
            accepted=accepted_values,
            duplicate_timestamp=duplicate_values,
        )

    def _validate_images(self, images: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        if not isinstance(images, Mapping):
            raise TypeError("images must be a mapping from camera name to ndarray")
        missing = [name for name in self._camera_names if name not in images]
        if missing:
            raise ValueError("missing image cameras: " + ", ".join(missing))

        normalized: dict[str, np.ndarray] = {}
        for camera_name in self._camera_names:
            image = np.asarray(images[camera_name])
            if image.ndim != 3:
                raise ValueError(
                    f"image for camera {camera_name!r} must be rank 3, got {image.shape}"
                )
            normalized[camera_name] = image
        return normalized

    def _validate_timestamps(
        self,
        timestamps_ns: Mapping[str, int | np.integer[Any]],
    ) -> dict[str, int]:
        if not isinstance(timestamps_ns, Mapping):
            raise TypeError("timestamps_ns must be a mapping from camera name to integer")
        missing = [name for name in self._camera_names if name not in timestamps_ns]
        if missing:
            raise ValueError("missing camera timestamps: " + ", ".join(missing))

        normalized: dict[str, int] = {}
        for camera_name in self._camera_names:
            value = timestamps_ns[camera_name]
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, np.integer)
            ):
                raise TypeError(
                    f"timestamp for camera {camera_name!r} must be an integer"
                )
            normalized[camera_name] = int(value)
        return normalized


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative_int(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _readonly_copy(value: np.ndarray) -> np.ndarray:
    copied = np.asarray(value).copy()
    copied.setflags(write=False)
    return copied
