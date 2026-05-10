"""Shared contracts for real excavator backend integrations.

This module is intentionally pure Python and has no ROS/CAN dependency.  It is
the stable place where the testbed-side 4D action/observation contract is
mapped to lower-level real-machine adapters.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


REAL_ACTION_DIM = 4
EXCAVATOR_API_AXIS_COUNT = 8
REAL_ACTION_ORDER = ("swing", "boom", "stick", "bucket")
REAL_QPOS_ORDER = REAL_ACTION_ORDER
REAL_QVEL_ORDER = REAL_ACTION_ORDER
EXCAVATOR_API_AXIS_ORDER = (
    "swing",
    "boom",
    "stick",
    "bucket",
    "left_track",
    "right_track",
    "boom_offset",
    "chassis_dozer",
)


@dataclass(frozen=True)
class RealSafetyState:
    """Safety fields surfaced to the recorder and action guard."""

    deadman_pressed: bool = True
    estop_active: bool = False
    manual_override_active: bool = False
    sensor_stale: bool = False
    remote_mode_enabled: bool | None = None
    pilot_enabled: bool | None = None
    raw_status: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "deadman_pressed": bool(self.deadman_pressed),
            "estop_active": bool(self.estop_active),
            "manual_override_active": bool(self.manual_override_active),
            "sensor_stale": bool(self.sensor_stale),
            "remote_mode_enabled": self.remote_mode_enabled,
            "pilot_enabled": self.pilot_enabled,
            "raw_status": list(self.raw_status),
        }


@dataclass(frozen=True)
class RealMachineState:
    """Backend-neutral state snapshot used to build testbed observations."""

    qpos: np.ndarray
    qvel: np.ndarray
    qacc: np.ndarray | None = None
    status: np.ndarray | None = None
    motor_rpm: np.ndarray | None = None
    plan_rpm: np.ndarray | None = None
    sensor_timestamp_ns: int = 0
    joint_timestamp_ns: int = 0
    image_timestamp_ns: Mapping[str, int] = field(default_factory=dict)
    sync_timestamp_ns: int = 0
    sync_max_skew_ns: int = 0
    sync_warnings: tuple[str, ...] = ()
    images: Mapping[str, np.ndarray] = field(default_factory=dict)
    env_state: np.ndarray | None = None
    safety: RealSafetyState = field(default_factory=RealSafetyState)

    def to_observation(self, *, step_id: int = 0) -> dict[str, Any]:
        now_ns = int(time.time_ns())
        joint_ts_ns = int(
            self.joint_timestamp_ns
            or self.sensor_timestamp_ns
            or self.sync_timestamp_ns
            or now_ns
        )
        image_ts_ns = {
            str(name): int(ts_ns)
            for name, ts_ns in dict(self.image_timestamp_ns).items()
        }
        images = {
            str(name): np.asarray(image, dtype=np.uint8).copy()
            for name, image in dict(self.images).items()
        }
        for name in images:
            image_ts_ns.setdefault(name, int(self.sensor_timestamp_ns or joint_ts_ns))
        sync_ts_ns = int(self.sync_timestamp_ns or self.sensor_timestamp_ns or joint_ts_ns)
        obs: dict[str, Any] = {
            "qpos": as_real_vector4(self.qpos, name="qpos"),
            "qvel": as_real_vector4(self.qvel, name="qvel"),
            "images": images,
            "env_state": (
                None
                if self.env_state is None
                else np.asarray(self.env_state, dtype=np.float32).copy()
            ),
            "step_id": int(step_id),
            "timestamp_ns": sync_ts_ns,
            "sensor_timestamp_ns": sync_ts_ns,
            "joint_timestamp_ns": joint_ts_ns,
            "image_timestamp_ns": image_ts_ns,
            "sync_timestamp_ns": sync_ts_ns,
            "sync_max_skew_ns": int(self.sync_max_skew_ns),
            "sync_warnings": list(self.sync_warnings),
            "safety_state": self.safety.to_dict(),
            "qpos_order": REAL_QPOS_ORDER,
            "qvel_order": REAL_QVEL_ORDER,
            "action_order": REAL_ACTION_ORDER,
        }
        if self.qacc is not None:
            obs["qacc"] = as_real_vector4(self.qacc, name="qacc")
        if self.status is not None:
            obs["status"] = np.asarray(self.status, dtype=np.int32).reshape(-1).copy()
        if self.motor_rpm is not None:
            obs["motor_rpm"] = np.asarray(self.motor_rpm, dtype=np.float32).reshape(-1).copy()
        if self.plan_rpm is not None:
            obs["plan_rpm"] = np.asarray(self.plan_rpm, dtype=np.float32).reshape(-1).copy()
        return obs


def as_real_action(action: np.ndarray | Sequence[float], *, clip: bool = False) -> np.ndarray:
    """Validate a normalized real-excavator action vector."""

    arr = np.asarray(action, dtype=np.float32).reshape(-1)
    if arr.shape != (REAL_ACTION_DIM,):
        raise ValueError(
            f"real excavator action must have shape ({REAL_ACTION_DIM},), got {arr.shape}"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError("real excavator action contains NaN or Inf")
    if clip:
        arr = np.clip(arr, -1.0, 1.0)
    return arr.astype(np.float32, copy=True)


def as_real_vector4(value: np.ndarray | Sequence[float], *, name: str) -> np.ndarray:
    """Validate a real 4-axis qpos/qvel-style vector."""

    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.shape != (REAL_ACTION_DIM,):
        raise ValueError(f"{name} must have shape ({REAL_ACTION_DIM},), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or Inf")
    return arr.astype(np.float32, copy=True)


def action4_to_speed_scalar8(
    action: np.ndarray | Sequence[float],
    *,
    axis_signs: Sequence[float] | None = None,
    clip: bool = True,
) -> np.ndarray:
    """
    Map the testbed 4D normalized action to excavator_api SpeedScalarCmd[8].

    The lower C++ control library owns CAN layout details, joint 2/3 hardware
    swapping, PID, and motor-rpm conversion.  Repo A should only populate the
    semantic first four axes and keep the chassis/auxiliary axes at zero.
    """

    action4 = as_real_action(action, clip=clip)
    signs = (
        np.ones(REAL_ACTION_DIM, dtype=np.float32)
        if axis_signs is None
        else np.asarray(axis_signs, dtype=np.float32).reshape(-1)
    )
    if signs.shape != (REAL_ACTION_DIM,):
        raise ValueError(
            f"axis_signs must have shape ({REAL_ACTION_DIM},), got {signs.shape}"
        )
    speed = np.zeros(EXCAVATOR_API_AXIS_COUNT, dtype=np.float32)
    speed[:REAL_ACTION_DIM] = np.clip(action4 * signs, -1.0, 1.0)
    return speed


def safety_state_from_status(
    status: np.ndarray | Sequence[int] | None,
    *,
    deadman_pressed: bool = True,
    manual_override_active: bool = False,
    sensor_stale: bool = False,
) -> RealSafetyState:
    """Derive recorder/guard safety flags from the lower-library status vector."""

    if status is None:
        return RealSafetyState(
            deadman_pressed=deadman_pressed,
            manual_override_active=manual_override_active,
            sensor_stale=sensor_stale,
        )
    arr = np.asarray(status, dtype=np.int32).reshape(-1)
    estop_active = bool(arr[10]) if arr.size > 10 else False
    remote_mode_enabled = bool(arr[4]) if arr.size > 4 else None
    pilot_enabled = bool(arr[5]) if arr.size > 5 else None
    return RealSafetyState(
        deadman_pressed=deadman_pressed,
        estop_active=estop_active,
        manual_override_active=manual_override_active,
        sensor_stale=sensor_stale,
        remote_mode_enabled=remote_mode_enabled,
        pilot_enabled=pilot_enabled,
        raw_status=tuple(int(v) for v in arr.tolist()),
    )


def observation_from_real_vectors(
    *,
    qpos: np.ndarray | Sequence[float],
    qvel: np.ndarray | Sequence[float],
    step_id: int,
    sensor_timestamp_ns: int | None = None,
    joint_timestamp_ns: int | None = None,
    image_timestamp_ns: Mapping[str, int] | None = None,
    sync_timestamp_ns: int | None = None,
    sync_max_skew_ns: int = 0,
    sync_warnings: Sequence[str] | None = None,
    images: Mapping[str, np.ndarray] | None = None,
    env_state: np.ndarray | Sequence[float] | None = None,
    status: np.ndarray | Sequence[int] | None = None,
    motor_rpm: np.ndarray | Sequence[float] | None = None,
    plan_rpm: np.ndarray | Sequence[float] | None = None,
) -> dict[str, Any]:
    """Build a testbed observation from real-machine vector fields."""

    machine_state = RealMachineState(
        qpos=as_real_vector4(qpos, name="qpos"),
        qvel=as_real_vector4(qvel, name="qvel"),
        status=None if status is None else np.asarray(status, dtype=np.int32).reshape(-1),
        motor_rpm=(
            None if motor_rpm is None else np.asarray(motor_rpm, dtype=np.float32).reshape(-1)
        ),
        plan_rpm=(
            None if plan_rpm is None else np.asarray(plan_rpm, dtype=np.float32).reshape(-1)
        ),
        sensor_timestamp_ns=int(sensor_timestamp_ns or time.time_ns()),
        joint_timestamp_ns=int(joint_timestamp_ns or 0),
        image_timestamp_ns=dict(image_timestamp_ns or {}),
        sync_timestamp_ns=int(sync_timestamp_ns or 0),
        sync_max_skew_ns=int(sync_max_skew_ns),
        sync_warnings=tuple(str(warning) for warning in (sync_warnings or ())),
        images=dict(images or {}),
        env_state=None if env_state is None else np.asarray(env_state, dtype=np.float32),
        safety=safety_state_from_status(status),
    )
    return machine_state.to_observation(step_id=step_id)
