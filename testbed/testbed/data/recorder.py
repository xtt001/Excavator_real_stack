"""
EpisodeRecorder: buffers per-step data and flushes to HDF5 at episode end.

Supports optional step_id, action_source, env_state, and diagnostics fields.

Usage
-----
    recorder = EpisodeRecorder(
        output_dir=Path("data/real_teleop_v1"),
        episode_idx=0,
        metadata={
            "task_name":        "real_excavation_teleop_v1",
            "is_real":          True,
            "platform":         "real_excavator",
            "seed":             -1,
            "control_hz":       50,
            "dt":               0.02,
            "action_semantics": "normalized_teleop_cmd_v1",
            "camera_names":     "fpv",
            "image_format":     "raw_rgb",
        },
    )
    recorder.record(obs, action, reward=0.0, step_id=0,
                    action_src_type="teleop", action_src_id="joystick")
    path = recorder.save(success=True)
"""

from __future__ import annotations

import datetime
import os
from pathlib import Path
from typing import Any

import numpy as np

from testbed.data.hdf5_io import write_episode
from testbed.data.schema import (
    ATTR_CAMERA_FPS,
    ATTR_CAMERA_HEIGHT,
    ATTR_CAMERA_WIDTH,
    ATTR_EPISODE_ID,
    ATTR_IMAGE_FORMAT,
)


class EpisodeRecorder:
    """
    Buffer for one demonstration episode.

    Parameters
    ----------
    output_dir   Directory where episode_N.hdf5 will be written.
    episode_idx  Integer episode index used for the filename.
    metadata     Dict of scalar metadata written to /metadata group.
    camera_names Names of cameras to record from images dict.
                 If None, all cameras present in first obs are recorded.
    """

    def __init__(
        self,
        output_dir: Path | str,
        episode_idx: int,
        metadata: dict[str, Any] | None = None,
        camera_names: list[str] | None = None,
    ):
        self.output_dir   = Path(output_dir)
        self.episode_idx  = episode_idx
        self.metadata     = dict(metadata or {})
        self.camera_names = camera_names

        # ── Core buffers ─────────────────────────────────────────────────────
        self._qpos:    list[np.ndarray] = []
        self._qvel:    list[np.ndarray] = []
        self._actions: list[np.ndarray] = []
        self._rewards: list[float]      = []
        self._images:  dict[str, list[np.ndarray]] = {}
        self._encoded_images: dict[str, list[np.ndarray]] = {}
        self._encoded_image_shapes: dict[str, tuple[int, int, int]] = {}

        # ── Optional per-step buffers ────────────────────────────────────────
        self._step_ids:        list[int]        = []
        self._step_ns:         list[int]        = []
        self._env_states:      list[np.ndarray] = []
        self._action_src_types: list[str]       = []
        self._action_src_ids:   list[str]       = []
        self._diagnostics:      dict[str, list[Any]] = {}

    # ── Per-step recording ────────────────────────────────────────────────────

    def record(
        self,
        obs: dict,
        action: np.ndarray,
        reward: float = 0.0,
        *,
        step_id: int | None = None,
        step_ns: int | None = None,
        action_src_type: str = "teleop",
        action_src_id: str = "joystick",
        diagnostics: dict[str, Any] | None = None,
    ) -> None:
        """
        Buffer one timestep.

        Parameters
        ----------
        obs              Raw observation dict from backend step.
        action           Guard-filtered action applied at this step (shape: (4,)).
        reward           Scalar reward (default 0).
        step_id          Monotonic step counter from the backend.
        step_ns          Wall-clock nanoseconds.
        action_src_type  "teleop" | "policy" | "scripted".
        action_src_id    "joystick" | "keyboard" | ...
        diagnostics      Optional per-step diagnostics such as raw_action,
                         guard_reason, and controller result fields.
        """
        self._qpos.append(np.array(obs["qpos"], dtype=np.float32))
        self._qvel.append(np.array(obs["qvel"], dtype=np.float32))
        self._actions.append(np.array(action, dtype=np.float32))
        self._rewards.append(float(reward))

        self._step_ids.append(int(step_id) if step_id is not None else len(self._step_ids))
        self._step_ns.append(int(step_ns) if step_ns is not None else 0)
        self._action_src_types.append(action_src_type)
        self._action_src_ids.append(action_src_id)
        if diagnostics:
            for key, value in diagnostics.items():
                self._diagnostics.setdefault(str(key), []).append(value)

        env_s = obs.get("env_state")
        if env_s is not None:
            self._env_states.append(np.array(env_s, dtype=np.float32))

        images: dict = obs.get("images", {})
        encoded_images: dict = obs.get("encoded_images", {})
        cams = self.camera_names if self.camera_names else list(images.keys())
        if not cams and encoded_images:
            cams = list(encoded_images.keys())
        for cam in cams:
            if cam in images:
                if cam not in self._images:
                    self._images[cam] = []
                self._images[cam].append(np.array(images[cam], dtype=np.uint8))
            if cam in encoded_images:
                if cam not in self._encoded_images:
                    self._encoded_images[cam] = []
                data, shape = _encoded_image_to_data_and_shape(encoded_images[cam])
                self._encoded_images[cam].append(data)
                if shape is not None:
                    self._encoded_image_shapes[cam] = shape

    # ── Save ─────────────────────────────────────────────────────────────────

    def save(
        self,
        success: bool = False,
        *,
        path: Path | str | None = None,
        metadata_updates: dict[str, Any] | None = None,
    ) -> Path:
        """
        Flush buffers to disk as episode_{episode_idx}.hdf5.

        Returns
        -------
        Path  Path to the written HDF5 file.
        """
        if not self._qpos:
            raise RuntimeError("EpisodeRecorder.save() called on an empty buffer.")

        path = (
            Path(path)
            if path is not None
            else self.output_dir / f"episode_{self.episode_idx}.hdf5"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")

        meta = dict(self.metadata)
        meta.setdefault(ATTR_EPISODE_ID, f"episode_{self.episode_idx}")
        meta["success"]   = int(success)
        meta["timestamp"] = datetime.datetime.utcnow().isoformat()
        meta["n_steps"]   = len(self._qpos)
        if metadata_updates:
            meta.update(metadata_updates)

        qpos    = np.stack(self._qpos)
        qvel    = np.stack(self._qvel)
        actions = np.stack(self._actions)
        rewards = np.array(self._rewards, dtype=np.float32)

        images: dict[str, np.ndarray] = {
            cam: np.stack(frames) for cam, frames in self._images.items()
        }
        encoded_images: dict[str, list[np.ndarray]] = {
            cam: list(frames) for cam, frames in self._encoded_images.items()
        }

        env_state = (
            np.stack(self._env_states)
            if self._env_states else None
        )
        diagnostics = {
            key: _stack_diagnostic(values)
            for key, values in self._diagnostics.items()
            if values
        }
        _update_image_metadata(
            meta,
            images=images,
            encoded_images=encoded_images,
            encoded_image_shapes=self._encoded_image_shapes,
            camera_names=self.camera_names,
            diagnostics=diagnostics,
        )

        if tmp_path.exists():
            tmp_path.unlink()
        write_episode(
            tmp_path,
            qpos=qpos,
            qvel=qvel,
            actions=actions,
            images=images if images else None,
            encoded_images=encoded_images if encoded_images else None,
            rewards=rewards,
            metadata=meta,
            env_state=env_state,
            step_ids=np.array(self._step_ids, dtype=np.int64) if self._step_ids else None,
            step_ns=np.array(self._step_ns, dtype=np.int64) if any(self._step_ns) else None,
            action_src_types=self._action_src_types if self._action_src_types else None,
            action_src_ids=self._action_src_ids if self._action_src_ids else None,
            diagnostics=diagnostics if diagnostics else None,
        )
        tmp_path.replace(path)
        return path

    # ── Convenience ──────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._qpos)

    def reset(self, episode_idx: int | None = None) -> None:
        """Clear buffers and optionally update episode index (for reuse)."""
        if episode_idx is not None:
            self.episode_idx = episode_idx
        self._qpos.clear()
        self._qvel.clear()
        self._actions.clear()
        self._rewards.clear()
        self._images.clear()
        self._encoded_images.clear()
        self._encoded_image_shapes.clear()
        self._step_ids.clear()
        self._step_ns.clear()
        self._env_states.clear()
        self._action_src_types.clear()
        self._action_src_ids.clear()
        self._diagnostics.clear()


def _stack_diagnostic(values: list[Any]) -> np.ndarray | list[str]:
    if all(isinstance(value, str) for value in values):
        return [str(value) for value in values]
    first = values[0]
    if isinstance(first, (int, float, bool, np.integer, np.floating, np.bool_)):
        return np.asarray(values)
    return np.stack([np.asarray(value) for value in values])


def _encoded_image_to_data_and_shape(value: Any) -> tuple[np.ndarray, tuple[int, int, int] | None]:
    shape: tuple[int, int, int] | None = None
    if isinstance(value, dict):
        raw_shape = value.get("shape")
        if raw_shape:
            parsed_shape = tuple(int(v) for v in raw_shape)
            if len(parsed_shape) == 3:
                shape = parsed_shape  # type: ignore[assignment]
        value = value.get("data", value.get("bytes", b""))
    if isinstance(value, (bytes, bytearray, memoryview)):
        data = np.frombuffer(bytes(value), dtype=np.uint8).copy()
    else:
        data = np.asarray(value, dtype=np.uint8).reshape(-1).copy()
    return data, shape


def _update_image_metadata(
    metadata: dict[str, Any],
    *,
    images: dict[str, np.ndarray],
    encoded_images: dict[str, list[np.ndarray]],
    encoded_image_shapes: dict[str, tuple[int, int, int]],
    camera_names: list[str] | None,
    diagnostics: dict[str, Any],
) -> None:
    if not images and not encoded_images:
        return

    if images:
        camera_name = _primary_camera_name(images, camera_names)
        frames = np.asarray(images[camera_name])
        if frames.ndim >= 4:
            metadata[ATTR_CAMERA_HEIGHT] = int(frames.shape[1])
            metadata[ATTR_CAMERA_WIDTH] = int(frames.shape[2])
    else:
        camera_name = _primary_camera_name(encoded_images, camera_names)
        shape = encoded_image_shapes.get(camera_name)
        if shape is not None:
            metadata[ATTR_CAMERA_HEIGHT] = int(shape[0])
            metadata[ATTR_CAMERA_WIDTH] = int(shape[1])
        metadata[ATTR_IMAGE_FORMAT] = "jpeg"

    fps = _estimate_timestamp_fps(diagnostics.get("image_timestamp_ns"))
    if fps > 0.0:
        metadata[ATTR_CAMERA_FPS] = fps


def _primary_camera_name(
    images: dict[str, np.ndarray],
    camera_names: list[str] | None,
) -> str:
    for camera_name in camera_names or []:
        if camera_name in images:
            return camera_name
    return next(iter(images))


def _estimate_timestamp_fps(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        timestamps = np.asarray(value, dtype=np.int64).reshape(-1)
    except (TypeError, ValueError):
        return 0.0
    timestamps = timestamps[timestamps > 0]
    if timestamps.size < 2:
        return 0.0
    deltas = np.diff(timestamps)
    deltas = deltas[deltas > 0]
    if deltas.size == 0:
        return 0.0
    median_delta_ns = float(np.median(deltas))
    if median_delta_ns <= 0.0:
        return 0.0
    return float(1_000_000_000.0 / median_delta_ns)
