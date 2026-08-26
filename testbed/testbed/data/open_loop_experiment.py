"""Data-calibrated open-loop experiment primitives for planner evaluation.

The experiment is open-loop with respect to hardware: no bridge, CAN device,
or actuator is contacted.  This module freezes train-only ranges and response
diagnostics for the reference replay; it does not claim that a fitted linear
plant is a valid physical simulator.  Held-out episode images and states are
used only by the replay evaluator as an observation stream.

This module intentionally keeps the older ``MockClosedLoopProfile`` unchanged.
That profile is useful for backend plumbing, while this profile models the
observed cycle's common swing-to-dump excursion and only applies the A/B safe
range at a ready boundary.  The recorded data reaches roughly 1.6--1.9 rad
in the middle of a cycle, so treating the ready range as a per-tick work limit
would reject the real demonstrated trajectory.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from testbed.backends.real.contracts import REAL_ACTION_DIM
from testbed.data.mock_closed_loop import (
    MockClosedLoopProfile,
    _cross_episode_support_quantiles,
)

AXIS_COUNT = REAL_ACTION_DIM


@dataclass(frozen=True)
class OpenLoopCalibration:
    """Train-only response diagnostics, work envelope, gates, and timing."""

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
    working_qpos_ranges: np.ndarray
    working_swing_apex_min: float
    working_swing_apex_quantiles: np.ndarray
    target_ranges: dict[str, tuple[float, float]]
    target_quantile_ranges: dict[str, tuple[float, float]]
    target_band_margin: dict[str, float]
    home_swing_qpos: float
    excursion_delta: float
    stable_qvel_abs: float
    stable_steps: int
    safe_swing_range: tuple[float, float]
    deadzone_positive: np.ndarray
    deadzone_negative: np.ndarray
    cycle_duration_p95_steps: int
    cycle_timeout_steps: int

    @classmethod
    def from_dataset(
        cls,
        *,
        dataset_dir: str | Path,
        train_episode_ids: list[int],
        ready_contract_path: str | Path,
        deadzone_threshold_path: str | Path,
        cycle_timeout_s: float = 60.0,
    ) -> OpenLoopCalibration:
        """Fit only on train episodes and freeze all experiment constants."""

        base = MockClosedLoopProfile.from_dataset(
            dataset_dir=dataset_dir,
            episode_ids=train_episode_ids,
            ready_contract_path=ready_contract_path,
            deadzone_threshold_path=deadzone_threshold_path,
        )
        positive = base.deadzone_positive.astype(np.float64, copy=True)
        negative = base.deadzone_negative.astype(np.float64, copy=True)
        response_rows: list[np.ndarray] = []
        response_targets: list[np.ndarray] = []
        state_rows: list[np.ndarray] = []
        state_owners: list[np.ndarray] = []
        apex_values: list[float] = []
        lengths: list[int] = []
        root = Path(dataset_dir)
        for episode_id in train_episode_ids:
            path = root / f"episode_{int(episode_id)}.hdf5"
            with h5py.File(path, "r") as handle:
                qpos = np.asarray(handle["observations/qpos"][()], dtype=np.float64)
                qvel = np.asarray(handle["observations/qvel"][()], dtype=np.float64)
                action = np.asarray(handle["action"][()], dtype=np.float64)
            if qpos.shape != qvel.shape or qpos.shape != action.shape:
                raise ValueError(f"episode {episode_id} state/action shapes disagree")
            if qpos.ndim != 2 or qpos.shape[1] != AXIS_COUNT:
                raise ValueError(f"episode {episode_id} must contain (T,4) arrays")
            effective = np.where(
                action >= 0.0,
                action >= positive.reshape(1, -1),
                -action >= negative.reshape(1, -1),
            )
            effective_action = np.where(effective, action, 0.0)
            response_rows.append(
                np.concatenate([qvel[:-1], effective_action[:-1]], axis=1)
            )
            response_targets.append(qvel[1:])
            state_rows.append(np.concatenate([qpos, qvel], axis=1))
            state_owners.append(
                np.full(qpos.shape[0], int(episode_id), dtype=np.int64)
            )
            apex_values.append(float(np.max(qpos[:, 0])))
            lengths.append(int(qpos.shape[0]))

        states = np.concatenate(state_rows, axis=0)
        owners = np.concatenate(state_owners, axis=0)
        x = np.concatenate(response_rows, axis=0)
        y = np.concatenate(response_targets, axis=0)
        coefficient = np.stack(
            [np.linalg.lstsq(x, y[:, axis], rcond=None)[0] for axis in range(AXIS_COUNT)],
            axis=1,
        )
        residual = y - x @ coefficient
        qvel_limit = np.maximum(np.quantile(np.abs(y), 0.995, axis=0), 0.05)
        residual_p95 = np.maximum(
            np.quantile(np.abs(residual), 0.95, axis=0), 1e-3
        )
        qpos_scale = np.maximum(np.std(states[:, :AXIS_COUNT], axis=0), 1e-3)
        qvel_scale = np.maximum(np.std(states[:, AXIS_COUNT:], axis=0), 1e-3)
        support_p95, support_p99 = _cross_episode_support_quantiles(
            state_matrix=states,
            episode_ids=owners,
            qpos_scale=qpos_scale,
            qvel_scale=qvel_scale,
            include_velocity=False,
        )
        qpos_quantiles = np.quantile(
            states[:, :AXIS_COUNT], [0.001, 0.999], axis=0
        )
        qpos_margin = np.maximum(
            0.02,
            0.01 * (qpos_quantiles[1] - qpos_quantiles[0]),
        )
        working_ranges = np.stack(
            [qpos_quantiles[0] - qpos_margin, qpos_quantiles[1] + qpos_margin],
            axis=1,
        )
        apex_quantiles = np.quantile(
            np.asarray(apex_values, dtype=np.float64), [0.05, 0.5, 0.95]
        )
        contract = json.loads(
            Path(ready_contract_path).read_text(encoding="utf-8")
        )
        swing = dict(contract["swing_axis"])
        duration_p95 = int(np.ceil(np.quantile(lengths, 0.95)))
        timeout_steps = int(np.ceil(float(cycle_timeout_s) / base.dt))
        return cls(
            dt=float(base.dt),
            qvel_damping=np.asarray(coefficient[:AXIS_COUNT, :], dtype=np.float64).T,
            action_to_qvel_gain=np.asarray(
                coefficient[AXIS_COUNT:, :], dtype=np.float64
            ).T,
            qvel_bias=np.zeros(AXIS_COUNT, dtype=np.float64),
            qvel_abs_limit=qvel_limit.astype(np.float64),
            qvel_residual_p95=residual_p95.astype(np.float64),
            qpos_state_scale=qpos_scale.astype(np.float64),
            qvel_state_scale=qvel_scale.astype(np.float64),
            qpos_support_distance_p95=float(support_p95),
            qpos_support_distance_p99=float(support_p99),
            working_qpos_ranges=working_ranges.astype(np.float64),
            working_swing_apex_min=float(apex_quantiles[0]),
            working_swing_apex_quantiles=apex_quantiles.astype(np.float64),
            target_ranges=base.target_ranges,
            target_quantile_ranges=base.target_quantile_ranges,
            target_band_margin=base.target_band_margin,
            home_swing_qpos=float(swing["home_swing_qpos_rad"]),
            excursion_delta=float(swing["cycle_excursion_min_abs_delta_rad"]),
            stable_qvel_abs=float(swing["swing_qvel_abs_max_rad_s"]),
            stable_steps=int(base.stable_steps),
            safe_swing_range=base.safe_swing_range,
            deadzone_positive=positive,
            deadzone_negative=negative,
            cycle_duration_p95_steps=duration_p95,
            cycle_timeout_steps=timeout_steps,
        )

    def target_ready(
        self, *, qpos: np.ndarray, qvel: np.ndarray, target_side: str
    ) -> bool:
        side = str(target_side).upper()
        if side not in self.target_ranges:
            raise ValueError(f"target_side must be A or B, got {target_side!r}")
        swing = float(np.asarray(qpos, dtype=np.float64).reshape(-1)[0])
        velocity = float(np.asarray(qvel, dtype=np.float64).reshape(-1)[0])
        low, high = self.target_ranges[side]
        return (
            self.safe_swing_range[0] <= swing <= self.safe_swing_range[1]
            and low <= swing <= high
            and abs(velocity) <= self.stable_qvel_abs
        )

    def classify_ready_side(self, *, qpos: np.ndarray, qvel: np.ndarray) -> str | None:
        if not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
            return None
        for side in ("A", "B"):
            if self.target_ready(qpos=qpos, qvel=qvel, target_side=side):
                return side
        return None
