"""Real-data-calibrated mock plant helpers for offline policy tests.

The existing ``MockStateReader`` is intentionally a tiny linear integrator
for backend plumbing.  This module adds a stricter, data-derived test plant:
the action-to-qvel response is fitted from real HDF5 episodes and image
observations are retrieved by the predicted state rather than replayed by a
fixed timestamp.  Rollouts stop when the state leaves the observed support.

This is still a diagnostic surrogate.  It does not model hydraulics, soil, or
real camera rendering and must not be presented as physical-effect proof.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from testbed.backends.real.contracts import REAL_ACTION_DIM, as_real_action
from testbed.backends.real.state import RealStateReader, RealStateSamples
from testbed.backends.real.sync import TimestampedSample
from testbed.data.dataset import _read_camera_image

AXIS_COUNT = REAL_ACTION_DIM
SIDE_CODES = {"A": -1, "B": 1}
AXIS_NAMES = ("swing", "boom", "stick", "bucket")


class MockClosedLoopSupportError(RuntimeError):
    """Raised when a surrogate rollout leaves its measured data support."""


@dataclass(frozen=True)
class MockClosedLoopProfile:
    """Data-derived state response and target-ready ranges."""

    dt: float
    qvel_damping: np.ndarray
    action_to_qvel_gain: np.ndarray
    qvel_bias: np.ndarray
    qvel_abs_limit: np.ndarray
    qvel_residual_p95: np.ndarray
    qpos_state_scale: np.ndarray
    qvel_state_scale: np.ndarray
    qpos_support_distance_p95: float
    qpos_support_distance_p99: float
    home_swing_qpos: float
    excursion_delta: float
    stable_qvel_abs: float
    stable_steps: int
    safe_swing_range: tuple[float, float]
    target_ranges: dict[str, tuple[float, float]]
    target_quantile_ranges: dict[str, tuple[float, float]]
    target_band_margin: dict[str, float]
    target_endpoint_episode_count: int
    deadzone_positive: np.ndarray
    deadzone_negative: np.ndarray

    @classmethod
    def from_dataset(
        cls,
        *,
        dataset_dir: str | Path,
        episode_ids: Sequence[int],
        ready_contract_path: str | Path,
        dt: float | None = None,
        deadzone_threshold_path: str | Path | None = None,
    ) -> MockClosedLoopProfile:
        root = Path(dataset_dir)
        ids = [int(value) for value in episode_ids]
        if not ids:
            raise ValueError("episode_ids must not be empty")

        response_rows: list[np.ndarray] = []
        response_targets: list[np.ndarray] = []
        state_rows: list[np.ndarray] = []
        state_owner_rows: list[np.ndarray] = []
        endpoint_by_side: dict[str, list[float]] = {"A": [], "B": []}
        inferred_dt: list[float] = []
        for episode_id in ids:
            path = root / f"episode_{episode_id}.hdf5"
            if not path.is_file():
                raise FileNotFoundError(f"episode does not exist: {path}")
            with h5py.File(path, "r") as handle:
                qpos = np.asarray(handle["observations/qpos"][()], dtype=np.float64)
                qvel = np.asarray(handle["observations/qvel"][()], dtype=np.float64)
                action = np.asarray(handle["action"][()], dtype=np.float64)
                condition = np.asarray(
                    handle["conditions/real_transition_condition_v1"][()],
                    dtype=np.float64,
                )
                metadata = handle["metadata"].attrs if "metadata" in handle else {}
                inferred_dt.append(float(metadata.get("dt", 0.05)))
            if qpos.shape != qvel.shape or qpos.shape != action.shape:
                raise ValueError(f"episode {episode_id} state/action shapes disagree")
            if qpos.ndim != 2 or qpos.shape[1] != AXIS_COUNT:
                raise ValueError(f"episode {episode_id} must contain (T,4) arrays")
            if condition.shape != (qpos.shape[0], 2):
                raise ValueError(f"episode {episode_id} condition shape is invalid")
            if not np.isfinite(condition).all():
                raise ValueError(f"episode {episode_id} condition contains NaN/Inf")
            side_code = int(condition[-1, 0])
            if side_code not in SIDE_CODES.values() or not np.all(
                condition[:, 0] == side_code
            ) or not np.all(condition[:, 1] == 1.0):
                raise ValueError(
                    f"episode {episode_id} condition is not a constant target-side goal"
                )
            side = "A" if side_code < 0 else "B"
            endpoint_by_side[side].append(float(qpos[-1, 0]))
            state_rows.append(np.concatenate([qpos, qvel], axis=1))
            state_owner_rows.append(
                np.full(qpos.shape[0], int(episode_id), dtype=np.int64)
            )

            # Fit v[t+1] from the previous measured velocity and command.  A
            # short autoregressive term captures measured actuator lag without
            # pretending to be a hydraulic simulator.  The intercept is
            # deliberately omitted: the train-fold idle tails have near-zero
            # median qvel, so a learned bias would create artificial position
            # drift while the command is zero.
            if qpos.shape[0] >= 2:
                response_rows.append(
                    np.concatenate(
                        [qvel[:-1], action[:-1]],
                        axis=1,
                    )
                )
                response_targets.append(qvel[1:])

        endpoint_ids = ids
        if not all(endpoint_by_side.values()):
            raise ValueError("training data must contain both A and B target endpoints")
        X = np.concatenate(response_rows, axis=0)
        Y = np.concatenate(response_targets, axis=0)
        coef = np.stack(
            [np.linalg.lstsq(X, Y[:, axis], rcond=None)[0] for axis in range(AXIS_COUNT)],
            axis=1,
        )
        predicted = X @ coef
        residual = Y - predicted
        qvel_abs_limit = np.maximum(
            np.quantile(np.abs(Y), 0.995, axis=0), 0.05
        )
        qvel_residual_p95 = np.maximum(
            np.quantile(np.abs(residual), 0.95, axis=0), 1e-3
        )
        state_matrix = np.concatenate(state_rows, axis=0)
        state_owner_matrix = np.concatenate(state_owner_rows, axis=0)
        resolved_dt = float(dt if dt is not None else np.median(inferred_dt))
        # The policy bundle consumes qpos (plus the committed condition), so
        # the image bank is queried by absolute predicted qpos.  Local
        # one-step deltas (used by the old implementation) make ordinary
        # cross-episode pose variation look unsupported, so use the measured
        # train-state spread for each axis instead.  qvel remains available
        # for the ready/stability gate and for the fitted plant response.
        qpos_state_scale = np.maximum(
            np.std(state_matrix[:, :AXIS_COUNT], axis=0),
            1e-3,
        )
        qvel_state_scale = np.maximum(
            np.std(state_matrix[:, AXIS_COUNT:], axis=0),
            1e-3,
        )
        qpos_support_distance_p95, qpos_support_distance_p99 = _cross_episode_support_quantiles(
            state_matrix=state_matrix,
            episode_ids=state_owner_matrix,
            qpos_scale=qpos_state_scale,
            qvel_scale=qvel_state_scale,
            include_velocity=False,
        )
        contract = json.loads(Path(ready_contract_path).read_text(encoding="utf-8"))
        swing = dict(contract["swing_axis"])
        deadzone_positive, deadzone_negative = _load_deadzone_thresholds(
            deadzone_threshold_path
        )
        target_band_margin: dict[str, float] = {}
        target_ranges: dict[str, tuple[float, float]] = {}
        target_quantile_ranges: dict[str, tuple[float, float]] = {}
        for side, values in endpoint_by_side.items():
            low = float(np.quantile(values, 0.05))
            high = float(np.quantile(values, 0.95))
            # A small band around the empirical quantiles avoids turning
            # float32 quantisation and ordinary ready-pose jitter into a
            # false miss.  The floor is data-scale (one hundredth of a radian)
            # and remains far below the observed A/B separation.
            margin = max(0.01, 0.05 * (high - low))
            target_band_margin[side] = float(margin)
            # The gate accepts the complete observed train-ready endpoint
            # range.  Quantile bands remain in the report for diagnosing
            # typical versus edge target poses; they are not silently used to
            # reject a valid recorded endpoint.
            observed_low = float(np.min(values))
            observed_high = float(np.max(values))
            target_ranges[side] = (
                observed_low - margin,
                observed_high + margin,
            )
            target_quantile_ranges[side] = (low - margin, high + margin)
        return cls(
            dt=resolved_dt,
            # ``coef`` is fitted row-wise as [previous qvel, action] -> next
            # qvel.  Transpose the two blocks so ``step`` keeps the measured
            # cross-axis coupling instead of silently discarding it.
            qvel_damping=np.asarray(
                coef[0:AXIS_COUNT, :], dtype=np.float64
            ).T.copy(),
            action_to_qvel_gain=np.asarray(
                coef[AXIS_COUNT : 2 * AXIS_COUNT, :], dtype=np.float64
            ).T.copy(),
            qvel_bias=np.zeros(AXIS_COUNT, dtype=np.float64),
            qvel_abs_limit=qvel_abs_limit.astype(np.float64),
            qvel_residual_p95=qvel_residual_p95.astype(np.float64),
            qpos_state_scale=qpos_state_scale.astype(np.float64),
            qvel_state_scale=qvel_state_scale.astype(np.float64),
            qpos_support_distance_p95=float(qpos_support_distance_p95),
            qpos_support_distance_p99=float(qpos_support_distance_p99),
            home_swing_qpos=float(swing["home_swing_qpos_rad"]),
            excursion_delta=float(swing["cycle_excursion_min_abs_delta_rad"]),
            stable_qvel_abs=float(swing["swing_qvel_abs_max_rad_s"]),
            stable_steps=max(
                1,
                int(
                    round(
                        float(swing["stable_window_s"])
                        / max(resolved_dt, 1e-6)
                    )
                ),
            ),
            safe_swing_range=(
                float(swing["safe_swing_qpos_range_rad"][0]),
                float(swing["safe_swing_qpos_range_rad"][1]),
            ),
            target_ranges=target_ranges,
            target_quantile_ranges=target_quantile_ranges,
            target_band_margin=target_band_margin,
            target_endpoint_episode_count=len(endpoint_ids),
            deadzone_positive=deadzone_positive,
            deadzone_negative=deadzone_negative,
        )

    def target_ready(self, *, qpos: np.ndarray, qvel: np.ndarray, target_side: str) -> bool:
        side = str(target_side).upper()
        if side not in self.target_ranges:
            raise ValueError(f"target_side must be A or B, got {target_side!r}")
        swing = float(np.asarray(qpos, dtype=np.float64).reshape(-1)[0])
        velocity = float(np.asarray(qvel, dtype=np.float64).reshape(-1)[0])
        low, high = self.target_ranges[side]
        return low <= swing <= high and abs(velocity) <= self.stable_qvel_abs

    def step(self, *, qpos: np.ndarray, qvel: np.ndarray, action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        state_qpos = np.asarray(qpos, dtype=np.float64).reshape(AXIS_COUNT).copy()
        state_qvel = np.asarray(qvel, dtype=np.float64).reshape(AXIS_COUNT).copy()
        command = as_real_action(action, clip=True).astype(np.float64)
        effective = np.where(
            command >= 0.0,
            command >= self.deadzone_positive,
            -command >= self.deadzone_negative,
        )
        next_qvel = (
            self.qvel_damping @ state_qvel
            + self.action_to_qvel_gain @ command
            + self.qvel_bias
        )
        next_qvel = np.clip(next_qvel, -self.qvel_abs_limit, self.qvel_abs_limit)
        next_qpos = state_qpos + next_qvel * float(self.dt)
        # Reuse the one-cycle state-hold semantics for sub-deadzone axes:
        # without an effective command, that axis neither advances position
        # nor carries residual velocity into the next observation.
        held = ~effective
        next_qvel[held] = 0.0
        next_qpos[held] = state_qpos[held]
        return next_qpos.astype(np.float32), next_qvel.astype(np.float32)


class H5ImageBank:
    """Qpos-indexed real-image bank for a bounded diagnostic rollout."""

    def __init__(
        self,
        episode_path: str | Path | Sequence[str | Path],
        *,
        camera_names: Sequence[str],
        qpos_state_scale: np.ndarray,
        qvel_state_scale: np.ndarray,
        include_velocity: bool = False,
    ) -> None:
        if isinstance(episode_path, (str, Path)):
            paths = [Path(episode_path)]
        else:
            paths = [Path(value) for value in episode_path]
        if not paths:
            raise ValueError("episode_path must contain at least one HDF5 file")
        self.paths = tuple(paths)
        self.camera_names = tuple(str(name) for name in camera_names)
        self.qpos_state_scale = np.asarray(qpos_state_scale, dtype=np.float64)
        self.qvel_state_scale = np.asarray(qvel_state_scale, dtype=np.float64)
        self.include_velocity = bool(include_velocity)
        if self.qpos_state_scale.shape != (AXIS_COUNT,) or not np.isfinite(
            self.qpos_state_scale
        ).all() or np.any(self.qpos_state_scale <= 0.0):
            raise ValueError("qpos_state_scale must be finite and positive with shape (4,)")
        if self.qvel_state_scale.shape != (AXIS_COUNT,) or not np.isfinite(
            self.qvel_state_scale
        ).all() or np.any(self.qvel_state_scale <= 0.0):
            raise ValueError("qvel_state_scale must be finite and positive with shape (4,)")
        qpos_rows: list[np.ndarray] = []
        qvel_rows: list[np.ndarray] = []
        path_rows: list[int] = []
        local_index_rows: list[np.ndarray] = []
        for path_index, path in enumerate(self.paths):
            with h5py.File(path, "r") as handle:
                qpos = np.asarray(handle["observations/qpos"][()], dtype=np.float64)
                qvel = np.asarray(handle["observations/qvel"][()], dtype=np.float64)
            if qpos.shape != qvel.shape or qpos.ndim != 2 or qpos.shape[1] != AXIS_COUNT:
                raise ValueError(f"episode {path} must contain matching (T,4) states")
            qpos_rows.append(qpos)
            qvel_rows.append(qvel)
            path_rows.extend([path_index] * qpos.shape[0])
            local_index_rows.append(np.arange(qpos.shape[0], dtype=np.int64))
        self.qpos = np.concatenate(qpos_rows, axis=0)
        self.qvel = np.concatenate(qvel_rows, axis=0)
        self._path_indices = np.asarray(path_rows, dtype=np.int64)
        self._local_indices = np.concatenate(local_index_rows, axis=0)

    def query(
        self,
        *,
        qpos: np.ndarray,
        qvel: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], int, float]:
        distance = self._distance_vector(qpos=qpos, qvel=qvel)
        index = int(np.argmin(distance))
        path_index = int(self._path_indices[index])
        local_index = int(self._local_indices[index])
        with h5py.File(self.paths[path_index], "r") as handle:
            images = {
                camera: _read_camera_image(handle, camera, local_index)
                for camera in self.camera_names
            }
        return images, local_index, float(distance[index])

    def nearest_distance(
        self,
        *,
        qpos: np.ndarray,
        qvel: np.ndarray,
    ) -> float:
        """Return nearest-state distance without decoding a camera frame."""

        return float(np.min(self._distance_vector(qpos=qpos, qvel=qvel)))

    def _distance_vector(
        self,
        *,
        qpos: np.ndarray,
        qvel: np.ndarray,
    ) -> np.ndarray:
        position_delta = (
            self.qpos - np.asarray(qpos, dtype=np.float64).reshape(1, 4)
        ) / self.qpos_state_scale.reshape(1, 4)
        if self.include_velocity:
            velocity_delta = (
                self.qvel - np.asarray(qvel, dtype=np.float64).reshape(1, 4)
            ) / self.qvel_state_scale.reshape(1, 4)
            features = np.concatenate([position_delta, velocity_delta], axis=1)
        else:
            features = position_delta
        return np.linalg.norm(features, axis=1)


class DataCalibratedMockStateReader(RealStateReader):
    """State reader driven by a fitted profile and state-indexed images."""

    def __init__(
        self,
        *,
        profile: MockClosedLoopProfile,
        image_bank: H5ImageBank,
        support_bank: H5ImageBank | None = None,
        initial_qpos: np.ndarray,
        initial_qvel: np.ndarray,
    ) -> None:
        self.profile = profile
        self.image_bank = image_bank
        self.support_bank = support_bank
        self.initial_qpos = np.asarray(initial_qpos, dtype=np.float32).reshape(4).copy()
        self.initial_qvel = np.asarray(initial_qvel, dtype=np.float32).reshape(4).copy()
        self._qpos = self.initial_qpos.copy()
        self._qvel = self.initial_qvel.copy()
        self.last_image_index: int | None = None
        self.last_image_distance: float | None = None
        self.last_data_support_distance: float | None = None

    def reset(self, seed: int | None = None) -> None:
        self._qpos = self.initial_qpos.copy()
        self._qvel = self.initial_qvel.copy()
        self.last_image_index = None
        self.last_image_distance = None
        self.last_data_support_distance = None

    def apply_control_result(self, result: Any, *, dt: float) -> None:
        del dt
        self._qpos, self._qvel = self.profile.step(
            qpos=self._qpos,
            qvel=self._qvel,
            action=result.commanded_action,
        )

    def read(
        self,
        *,
        step_id: int,
        action_timestamp_ns: int | None = None,
    ) -> RealStateSamples:
        del step_id
        images, index, distance = self.image_bank.query(
            qpos=self._qpos,
            qvel=self._qvel,
        )
        self.last_image_index = index
        self.last_image_distance = distance
        self.last_data_support_distance = (
            self.image_bank.nearest_distance(qpos=self._qpos, qvel=self._qvel)
            if self.support_bank is None
            else self.support_bank.nearest_distance(
                qpos=self._qpos,
                qvel=self._qvel,
            )
        )
        receive_ns = int(action_timestamp_ns or 0) or int(
            time.time_ns()
        )
        joint = TimestampedSample(
            timestamp_ns=receive_ns,
            payload={
                "qpos": self._qpos.copy(),
                "qvel": self._qvel.copy(),
                "status": np.zeros(16, dtype=np.int32),
            },
            source="data_calibrated_mock_joint",
            receive_time_ns=receive_ns,
        )
        image_samples = {
            camera: TimestampedSample(
                timestamp_ns=receive_ns,
                payload=image,
                source=f"data_calibrated_mock_camera:{camera}",
                receive_time_ns=receive_ns,
            )
            for camera, image in images.items()
        }
        return RealStateSamples(joint=joint, images=image_samples)


def _shortest_angle_array(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return ((array + np.pi) % (2.0 * np.pi) - np.pi).astype(np.float64)


def _load_deadzone_thresholds(
    path: str | Path | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load direct policy-output deadzones used by the state-hold contract."""

    if path is None or not str(path).strip():
        zeros = np.zeros(AXIS_COUNT, dtype=np.float64)
        return zeros.copy(), zeros.copy()
    threshold_path = Path(path)
    if not threshold_path.is_file():
        raise FileNotFoundError(
            f"deadzone threshold file does not exist: {threshold_path}"
        )
    payload = json.loads(threshold_path.read_text(encoding="utf-8"))
    raw = payload.get("deadzone_action", payload)
    if not isinstance(raw, dict):
        raise ValueError("deadzone threshold payload must be a mapping")
    positive = []
    negative = []
    for axis in AXIS_NAMES:
        values = raw.get(axis)
        if not isinstance(values, dict):
            raise ValueError(f"deadzone threshold is missing axis {axis!r}")
        pos = float(values.get("pos"))
        neg = float(values.get("neg"))
        if not np.isfinite(pos) or not np.isfinite(neg) or pos < 0.0 or neg < 0.0:
            raise ValueError(f"deadzone thresholds for {axis!r} must be finite and non-negative")
        positive.append(pos)
        negative.append(neg)
    return np.asarray(positive, dtype=np.float64), np.asarray(negative, dtype=np.float64)


def _cross_episode_support_quantiles(
    *,
    state_matrix: np.ndarray,
    episode_ids: np.ndarray,
    qpos_scale: np.ndarray,
    qvel_scale: np.ndarray,
    include_velocity: bool = True,
) -> tuple[float, float]:
    """Estimate image-qpos support from train-only cross-episode neighbours.

    A row must be compared with a different episode; otherwise every sampled
    row would be its own zero-distance neighbour and the support threshold
    would collapse to zero.  A deterministic stride keeps profile fitting
    bounded while still covering the full train-state distribution.
    """

    states = np.asarray(state_matrix, dtype=np.float64)
    owners = np.asarray(episode_ids, dtype=np.int64).reshape(-1)
    if states.ndim != 2 or states.shape[1] != 2 * AXIS_COUNT:
        raise ValueError("state_matrix must have shape (N, 8)")
    if owners.shape[0] != states.shape[0]:
        raise ValueError("episode_ids must align with state_matrix")
    if states.shape[0] < 2 or np.unique(owners).size < 2:
        return (1.0, 1.0)
    stride = max(1, int(np.ceil(states.shape[0] / 4096)))
    sample_indices = np.arange(0, states.shape[0], stride, dtype=np.int64)
    sampled = states[sample_indices]
    sampled_owners = owners[sample_indices]
    if include_velocity:
        scale = np.concatenate(
            [
                np.asarray(qpos_scale, dtype=np.float64),
                np.asarray(qvel_scale, dtype=np.float64),
            ]
        )
        selected_states = states
    else:
        scale = np.asarray(qpos_scale, dtype=np.float64)
        selected_states = states[:, :AXIS_COUNT]
    features = selected_states / scale.reshape(1, -1)
    reference = sampled[:, : features.shape[1]] / scale.reshape(1, -1)
    distances = np.full(sampled.shape[0], np.inf, dtype=np.float64)
    for start in range(0, sampled.shape[0], 256):
        query = reference[start : start + 256]
        squared = (
            np.sum(query * query, axis=1, keepdims=True)
            + np.sum(features * features, axis=1, keepdims=True).T
            - 2.0 * query @ features.T
        )
        squared = np.maximum(squared, 0.0)
        same_episode = sampled_owners[start : start + len(query), None] == owners[None, :]
        squared[same_episode] = np.inf
        distances[start : start + len(query)] = np.sqrt(np.min(squared, axis=1))
    finite = distances[np.isfinite(distances)]
    if finite.size == 0:
        return (1.0, 1.0)
    return (
        float(max(np.quantile(finite, 0.95), 1e-6)),
        float(max(np.quantile(finite, 0.99), 1e-6)),
    )
