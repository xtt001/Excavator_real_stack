"""Near-home feedback controller for real excavator recording sessions."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from testbed.backends.real.contracts import (
    REAL_ACTION_DIM,
    REAL_ACTION_ORDER,
    align_real_qpos_to_reference_branch,
    as_real_vector4,
    real_qpos_error_rad,
)


_BUCKET_AXIS = 3
_BUCKET_QUATERNION_POLICY_OFFSET_RAD = -0.4060066694119653
_BUCKET_PRIMARY_CHART_MIN_STRENGTH = 0.35
_BUCKET_GRAVITY_HINGE_REFERENCE_RAD = 2.0839045979023254
_BUCKET_GRAVITY_HINGE_POLICY_OFFSET_RAD = -2.025561263010988
_BUCKET_GRAVITY_HINGE_MEDIAN_WINDOW = 21
_DAOYUAN_CHAIN_STICK_POLICY_OFFSET_RAD = 0.19801020488135143
_DAOYUAN_CHAIN_BUCKET_POLICY_OFFSET_RAD = -2.006833804661174


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _normalize_joint_rpy_profile(raw: Any) -> str:
    raw_text = str(raw or "legacy_diff").strip().lower()
    if raw_text in {"daoyuan_chain", "daoyuan", "chain_rpy"}:
        return "daoyuan_chain"
    return "legacy_diff"


def _joint_rpy_profile(imu_debug: Mapping[str, Any] | None = None) -> str:
    if isinstance(imu_debug, Mapping):
        raw_profile = imu_debug.get("joint_rpy_profile")
        if raw_profile:
            return _normalize_joint_rpy_profile(raw_profile)
        mapping = imu_debug.get("joint_velocity_mapping")
        if isinstance(mapping, Mapping):
            stick = mapping.get("stick")
            bucket = mapping.get("bucket")
            if isinstance(stick, Mapping) and "+" in str(stick.get("gyro_axis", "")):
                return "daoyuan_chain"
            if isinstance(bucket, Mapping) and "+" in str(bucket.get("position_axis", "")):
                return "daoyuan_chain"
    return _normalize_joint_rpy_profile(os.environ.get("EXCAVATOR_JOINT_RPY_PROFILE"))


def _daoyuan_chain_stick_policy_offset_rad(
    imu_debug: Mapping[str, Any] | None = None,
) -> float:
    if isinstance(imu_debug, Mapping):
        value = _finite_float(imu_debug.get("daoyuan_stick_policy_offset_rad"))
        if value is not None:
            return value
    value = _finite_float(
        os.environ.get(
            "EXCAVATOR_DAOYUAN_STICK_POLICY_OFFSET_RAD",
            str(_DAOYUAN_CHAIN_STICK_POLICY_OFFSET_RAD),
        )
    )
    return _DAOYUAN_CHAIN_STICK_POLICY_OFFSET_RAD if value is None else value


def _daoyuan_chain_bucket_policy_offset_rad(
    imu_debug: Mapping[str, Any] | None = None,
) -> float:
    if isinstance(imu_debug, Mapping):
        value = _finite_float(imu_debug.get("daoyuan_bucket_policy_offset_rad"))
        if value is not None:
            return value
    value = _finite_float(
        os.environ.get(
            "EXCAVATOR_DAOYUAN_BUCKET_POLICY_OFFSET_RAD",
            str(_DAOYUAN_CHAIN_BUCKET_POLICY_OFFSET_RAD),
        )
    )
    return _DAOYUAN_CHAIN_BUCKET_POLICY_OFFSET_RAD if value is None else value


def _bucket_imu0_profile() -> str:
    raw = os.environ.get("EXCAVATOR_BUCKET_IMU0_PROFILE", "legacy_y").strip().lower()
    if raw in {"roll_ccw90", "rotated_ccw90", "imu0_roll", "roll"}:
        return "roll_ccw90"
    return "legacy_y"


def _bucket_imu0_roll_profile_enabled() -> bool:
    return _bucket_imu0_profile() == "roll_ccw90"


def _bucket_qpos_source() -> str:
    raw = os.environ.get("EXCAVATOR_BUCKET_QPOS_SOURCE", "").strip().lower()
    if raw in {"gravity_hinge", "gravity", "accel_hinge"}:
        return "gravity_hinge"
    if raw in {"rpy", "native_rpy", "roll_ccw90"}:
        return "rpy"
    if raw in {"legacy_quaternion", "legacy", "quaternion"}:
        return "legacy_quaternion"
    return "rpy" if _bucket_imu0_roll_profile_enabled() else "legacy_quaternion"


def _bucket_imu0_reference_rad() -> float:
    if not _bucket_imu0_roll_profile_enabled():
        return 0.0
    try:
        value = float(os.environ.get("EXCAVATOR_BUCKET_IMU0_REFERENCE_RAD", "0"))
    except ValueError:
        return 0.0
    return value if np.isfinite(value) else 0.0


def _bucket_imu0_axis_sign() -> float:
    if not _bucket_imu0_roll_profile_enabled():
        return 1.0
    try:
        value = float(
            os.environ.get(
                "EXCAVATOR_BUCKET_IMU0_SIGN",
                os.environ.get("EXCAVATOR_BUCKET_IMU0_GYRO_SIGN", "1"),
            )
        )
    except ValueError:
        return 1.0
    return -1.0 if np.isfinite(value) and value < 0.0 else 1.0


def _bucket_gravity_hinge_reference_rad() -> float:
    try:
        value = float(
            os.environ.get(
                "EXCAVATOR_BUCKET_GRAVITY_HINGE_REFERENCE_RAD",
                str(_BUCKET_GRAVITY_HINGE_REFERENCE_RAD),
            )
        )
    except ValueError:
        return _BUCKET_GRAVITY_HINGE_REFERENCE_RAD
    return value if np.isfinite(value) else _BUCKET_GRAVITY_HINGE_REFERENCE_RAD


def _bucket_gravity_hinge_policy_offset_rad() -> float:
    try:
        value = float(
            os.environ.get(
                "EXCAVATOR_BUCKET_GRAVITY_HINGE_POLICY_OFFSET_RAD",
                str(_BUCKET_GRAVITY_HINGE_POLICY_OFFSET_RAD),
            )
        )
    except ValueError:
        return _BUCKET_GRAVITY_HINGE_POLICY_OFFSET_RAD
    return value if np.isfinite(value) else _BUCKET_GRAVITY_HINGE_POLICY_OFFSET_RAD


def _bucket_gravity_hinge_median_window() -> int:
    try:
        value = float(
            os.environ.get(
                "EXCAVATOR_BUCKET_GRAVITY_HINGE_MEDIAN_WINDOW",
                str(_BUCKET_GRAVITY_HINGE_MEDIAN_WINDOW),
            )
        )
    except ValueError:
        return _BUCKET_GRAVITY_HINGE_MEDIAN_WINDOW
    if not np.isfinite(value):
        return _BUCKET_GRAVITY_HINGE_MEDIAN_WINDOW
    return max(1, int(round(value)))


def _bucket_initial_position_rad(primary_phase_rad: float) -> float:
    return primary_phase_rad


def _vector4(value: Any, *, name: str, default: Sequence[float] | None = None) -> np.ndarray:
    if value is None:
        if default is None:
            raise ValueError(f"{name} is required")
        value = default
    if isinstance(value, (int, float)):
        value = [float(value)] * REAL_ACTION_DIM
    return as_real_vector4(value, name=name)


def _positive_vector4(
    value: Any,
    *,
    name: str,
    default: Sequence[float],
) -> np.ndarray:
    arr = _vector4(value, name=name, default=default)
    if np.any(arr <= 0.0):
        raise ValueError(f"{name} values must be positive")
    return arr


def _nonnegative_vector4(
    value: Any,
    *,
    name: str,
    default: Sequence[float],
) -> np.ndarray:
    arr = _vector4(value, name=name, default=default)
    if np.any(arr < 0.0):
        raise ValueError(f"{name} values must be non-negative")
    return arr


def _sign_vector4(
    value: Any,
    *,
    name: str,
    default: Sequence[float],
) -> np.ndarray:
    arr = _vector4(value, name=name, default=default)
    if np.any(arr == 0.0):
        raise ValueError(f"{name} values must be non-zero signs")
    return np.where(arr < 0.0, -1.0, 1.0).astype(np.float32)


def _axis_order(value: Any, *, name: str) -> tuple[int, ...]:
    if value is None:
        items: list[Any] = list(range(REAL_ACTION_DIM))
    elif isinstance(value, str):
        items = [part.strip() for part in value.split(",") if part.strip()]
    else:
        items = list(value)
    if not items:
        raise ValueError(f"{name} must include at least one axis")

    by_name = {axis_name: idx for idx, axis_name in enumerate(REAL_ACTION_ORDER)}
    parsed: list[int] = []
    for item in items:
        if isinstance(item, str):
            key = item.strip().lower()
            if key not in by_name:
                raise ValueError(
                    f"{name} unknown axis {item!r}; expected one of {REAL_ACTION_ORDER}"
                )
            idx = by_name[key]
        else:
            idx = int(item)
        if idx < 0 or idx >= REAL_ACTION_DIM:
            raise ValueError(f"{name} axis index out of range: {idx}")
        if idx in parsed:
            raise ValueError(f"{name} contains duplicate axis: {idx}")
        parsed.append(idx)

    parsed.extend(idx for idx in range(REAL_ACTION_DIM) if idx not in parsed)
    return tuple(parsed)


@dataclass(frozen=True)
class GoHomeConfig:
    """Configuration for safe near-home qpos feedback control."""

    home_pose_rad: np.ndarray
    near_tolerance_rad: np.ndarray
    success_tolerance_rad: np.ndarray
    center_tolerance_rad: np.ndarray
    resume_tolerance_rad: np.ndarray
    control_signs: np.ndarray
    p_gain: np.ndarray
    d_gain: np.ndarray
    min_action: np.ndarray
    min_action_positive: np.ndarray
    min_action_negative: np.ndarray
    max_action: np.ndarray
    near_max_action: np.ndarray
    coast_stop_time_s: np.ndarray
    coast_reactivation_delay_s: np.ndarray
    action_slew_rate: np.ndarray
    action_ramp_rate: np.ndarray
    center_approach_action: np.ndarray
    control_decision_period_s: float
    axis_order: tuple[int, ...]
    max_active_axes: int
    sign_reversal_delay_s: np.ndarray
    sign_reversal_min_error_rad: np.ndarray
    qpos_filter_tau_s: np.ndarray
    qvel_filter_tau_s: np.ndarray
    axis_center_hold_s: np.ndarray
    stall_action_step: np.ndarray
    stall_detection_s: float
    stall_boost_interval_s: float
    stall_error_progress_rad: np.ndarray
    stall_qvel_threshold_rad_s: np.ndarray
    wrong_direction_detection_s: float
    wrong_direction_error_increase_rad: np.ndarray
    wrong_direction_cooldown_s: np.ndarray
    qvel_stable_rad_s: np.ndarray
    max_policy_raw_qpos_delta_rad: np.ndarray
    dwell_s: float = 0.4
    timeout_s: float = 8.0
    runaway_error_factor: float = 1.5

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any] | None) -> "GoHomeConfig | None":
        raw = dict(cfg or {})
        if not bool(raw.get("enabled", False)):
            return None
        if raw.get("home_pose_rad") is None:
            raise ValueError("teleop.recording.go_home.home_pose_rad must be set")
        home = _vector4(raw.get("home_pose_rad"), name="home_pose_rad")
        near = _positive_vector4(
            raw.get("near_tolerance_rad"),
            name="near_tolerance_rad",
            default=[0.20] * REAL_ACTION_DIM,
        )
        success = _positive_vector4(
            raw.get("success_tolerance_rad"),
            name="success_tolerance_rad",
            default=[0.03] * REAL_ACTION_DIM,
        )
        center = _positive_vector4(
            raw.get("center_tolerance_rad"),
            name="center_tolerance_rad",
            default=success,
        )
        resume = _positive_vector4(
            raw.get("resume_tolerance_rad"),
            name="resume_tolerance_rad",
            default=np.maximum(success, center * 1.5),
        )
        if np.any(center > success):
            raise ValueError("center_tolerance_rad values must be <= success_tolerance_rad")
        if np.any(resume < center):
            raise ValueError("resume_tolerance_rad values must be >= center_tolerance_rad")
        control_signs = _sign_vector4(
            raw.get("control_signs"),
            name="control_signs",
            default=[1.0] * REAL_ACTION_DIM,
        )
        p_gain = _positive_vector4(
            raw.get("p_gain"),
            name="p_gain",
            default=[1.2] * REAL_ACTION_DIM,
        )
        d_gain = _nonnegative_vector4(
            raw.get("d_gain"),
            name="d_gain",
            default=[0.0] * REAL_ACTION_DIM,
        )
        min_action = _nonnegative_vector4(
            raw.get("min_action"),
            name="min_action",
            default=[0.0] * REAL_ACTION_DIM,
        )
        min_action_positive = _nonnegative_vector4(
            raw.get("min_action_positive"),
            name="min_action_positive",
            default=min_action,
        )
        min_action_negative = _nonnegative_vector4(
            raw.get("min_action_negative"),
            name="min_action_negative",
            default=min_action,
        )
        max_action = _positive_vector4(
            raw.get("max_action"),
            name="max_action",
            default=[0.20] * REAL_ACTION_DIM,
        )
        near_max_action = _positive_vector4(
            raw.get("near_max_action"),
            name="near_max_action",
            default=max_action,
        )
        if np.any(min_action_positive > max_action) or np.any(
            min_action_negative > max_action
        ):
            raise ValueError("min_action values must be <= max_action values")
        if np.any(min_action_positive > near_max_action) or np.any(
            min_action_negative > near_max_action
        ):
            raise ValueError("min_action values must be <= near_max_action values")
        if np.any(near_max_action > max_action):
            raise ValueError("near_max_action values must be <= max_action values")
        coast_stop_time_s = _nonnegative_vector4(
            raw.get("coast_stop_time_s"),
            name="coast_stop_time_s",
            default=[0.0] * REAL_ACTION_DIM,
        )
        coast_reactivation_delay_s = _nonnegative_vector4(
            raw.get("coast_reactivation_delay_s"),
            name="coast_reactivation_delay_s",
            default=[0.0] * REAL_ACTION_DIM,
        )
        action_slew_rate = _nonnegative_vector4(
            raw.get("action_slew_rate"),
            name="action_slew_rate",
            default=[0.0] * REAL_ACTION_DIM,
        )
        action_ramp_rate = _nonnegative_vector4(
            raw.get("action_ramp_rate"),
            name="action_ramp_rate",
            default=[0.0] * REAL_ACTION_DIM,
        )
        center_approach_action = _nonnegative_vector4(
            raw.get("center_approach_action"),
            name="center_approach_action",
            default=[0.0] * REAL_ACTION_DIM,
        )
        control_decision_period_s = float(raw.get("control_decision_period_s", 0.0))
        if control_decision_period_s < 0.0:
            raise ValueError("control_decision_period_s must be >= 0")
        axis_order = _axis_order(raw.get("axis_order"), name="axis_order")
        max_active_axes = int(raw.get("max_active_axes", REAL_ACTION_DIM))
        if max_active_axes <= 0 or max_active_axes > REAL_ACTION_DIM:
            raise ValueError(f"max_active_axes must be in [1,{REAL_ACTION_DIM}]")
        sign_reversal_delay_s = _nonnegative_vector4(
            raw.get("sign_reversal_delay_s"),
            name="sign_reversal_delay_s",
            default=[0.0] * REAL_ACTION_DIM,
        )
        sign_reversal_min_error_rad = _nonnegative_vector4(
            raw.get("sign_reversal_min_error_rad"),
            name="sign_reversal_min_error_rad",
            default=[0.0] * REAL_ACTION_DIM,
        )
        qpos_filter_tau_s = _nonnegative_vector4(
            raw.get("qpos_filter_tau_s"),
            name="qpos_filter_tau_s",
            default=[0.0] * REAL_ACTION_DIM,
        )
        qvel_filter_tau_s = _nonnegative_vector4(
            raw.get("qvel_filter_tau_s"),
            name="qvel_filter_tau_s",
            default=[0.0] * REAL_ACTION_DIM,
        )
        axis_center_hold_s = _nonnegative_vector4(
            raw.get("axis_center_hold_s"),
            name="axis_center_hold_s",
            default=[0.0] * REAL_ACTION_DIM,
        )
        stall_action_step = _nonnegative_vector4(
            raw.get("stall_action_step"),
            name="stall_action_step",
            default=[0.0] * REAL_ACTION_DIM,
        )
        stall_detection_s = float(raw.get("stall_detection_s", 1.0))
        stall_boost_interval_s = float(raw.get("stall_boost_interval_s", 0.75))
        if stall_detection_s < 0.0:
            raise ValueError("stall_detection_s must be >= 0")
        if stall_boost_interval_s <= 0.0:
            raise ValueError("stall_boost_interval_s must be positive")
        stall_error_progress = _nonnegative_vector4(
            raw.get("stall_error_progress_rad"),
            name="stall_error_progress_rad",
            default=[0.002] * REAL_ACTION_DIM,
        )
        stall_qvel_threshold = _nonnegative_vector4(
            raw.get("stall_qvel_threshold_rad_s"),
            name="stall_qvel_threshold_rad_s",
            default=[0.01] * REAL_ACTION_DIM,
        )
        wrong_direction_detection_s = float(
            raw.get("wrong_direction_detection_s", stall_detection_s)
        )
        if wrong_direction_detection_s < 0.0:
            raise ValueError("wrong_direction_detection_s must be >= 0")
        wrong_direction_error_increase = _nonnegative_vector4(
            raw.get("wrong_direction_error_increase_rad"),
            name="wrong_direction_error_increase_rad",
            default=[1.0e9] * REAL_ACTION_DIM,
        )
        wrong_direction_cooldown = _nonnegative_vector4(
            raw.get("wrong_direction_cooldown_s"),
            name="wrong_direction_cooldown_s",
            default=[0.0] * REAL_ACTION_DIM,
        )
        qvel_stable = _positive_vector4(
            raw.get("qvel_stable_rad_s"),
            name="qvel_stable_rad_s",
            default=[0.02] * REAL_ACTION_DIM,
        )
        max_policy_raw_delta = _positive_vector4(
            raw.get("max_policy_raw_qpos_delta_rad"),
            name="max_policy_raw_qpos_delta_rad",
            default=[0.08] * REAL_ACTION_DIM,
        )
        dwell_s = float(raw.get("dwell_s", 0.4))
        timeout_s = float(raw.get("timeout_s", 8.0))
        if dwell_s < 0.0:
            raise ValueError("dwell_s must be >= 0")
        if timeout_s <= 0.0:
            raise ValueError("timeout_s must be positive")
        return cls(
            home_pose_rad=home,
            near_tolerance_rad=near,
            success_tolerance_rad=success,
            center_tolerance_rad=center,
            resume_tolerance_rad=resume,
            control_signs=control_signs,
            p_gain=p_gain,
            d_gain=d_gain,
            min_action=min_action,
            min_action_positive=min_action_positive,
            min_action_negative=min_action_negative,
            max_action=max_action,
            near_max_action=near_max_action,
            coast_stop_time_s=coast_stop_time_s,
            coast_reactivation_delay_s=coast_reactivation_delay_s,
            action_slew_rate=action_slew_rate,
            action_ramp_rate=action_ramp_rate,
            center_approach_action=center_approach_action,
            control_decision_period_s=control_decision_period_s,
            axis_order=axis_order,
            max_active_axes=max_active_axes,
            sign_reversal_delay_s=sign_reversal_delay_s,
            sign_reversal_min_error_rad=sign_reversal_min_error_rad,
            qpos_filter_tau_s=qpos_filter_tau_s,
            qvel_filter_tau_s=qvel_filter_tau_s,
            axis_center_hold_s=axis_center_hold_s,
            stall_action_step=stall_action_step,
            stall_detection_s=stall_detection_s,
            stall_boost_interval_s=stall_boost_interval_s,
            stall_error_progress_rad=stall_error_progress,
            stall_qvel_threshold_rad_s=stall_qvel_threshold,
            wrong_direction_detection_s=wrong_direction_detection_s,
            wrong_direction_error_increase_rad=wrong_direction_error_increase,
            wrong_direction_cooldown_s=wrong_direction_cooldown,
            qvel_stable_rad_s=qvel_stable,
            max_policy_raw_qpos_delta_rad=max_policy_raw_delta,
            dwell_s=dwell_s,
            timeout_s=timeout_s,
            runaway_error_factor=float(raw.get("runaway_error_factor", 1.5)),
        )


@dataclass(frozen=True)
class GoHomeResult:
    """One controller update result."""

    action: np.ndarray
    done: bool = False
    failed: bool = False
    reason: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)


class GoHomeController:
    """Run a bounded PD controller from near-home qpos to the configured home pose."""

    def __init__(self, config: GoHomeConfig) -> None:
        self.config = config
        self.started = False
        self.start_time_ns = 0
        self.end_time_ns = 0
        self._start_s = 0.0
        self._last_update_s: float | None = None
        self._last_decision_s: float | None = None
        self._filter_last_s: float | None = None
        self._stable_since_s: float | None = None
        self._axis_active = np.zeros(REAL_ACTION_DIM, dtype=bool)
        self._axis_scheduled = np.zeros(REAL_ACTION_DIM, dtype=bool)
        self._center_approach_axis = np.zeros(REAL_ACTION_DIM, dtype=bool)
        self._axis_center_since_s = np.full(REAL_ACTION_DIM, np.nan, dtype=np.float64)
        self._axis_coast_until_s = np.zeros(REAL_ACTION_DIM, dtype=np.float64)
        self._axis_ramp_start_s = np.full(REAL_ACTION_DIM, np.nan, dtype=np.float64)
        self._axis_ramp_sign = np.zeros(REAL_ACTION_DIM, dtype=np.float32)
        self._action_ramp_limit = np.zeros(REAL_ACTION_DIM, dtype=np.float32)
        self._axis_command_sign = np.zeros(REAL_ACTION_DIM, dtype=np.float32)
        self._axis_last_sign_change_s = np.full(REAL_ACTION_DIM, -np.inf, dtype=np.float64)
        self._sign_reversal_blocked = np.zeros(REAL_ACTION_DIM, dtype=bool)
        self._last_action = np.zeros(REAL_ACTION_DIM, dtype=np.float32)
        self._held_unsmoothed_action = np.zeros(REAL_ACTION_DIM, dtype=np.float32)
        self._filtered_qpos = np.zeros(REAL_ACTION_DIM, dtype=np.float32)
        self._filtered_qvel = np.zeros(REAL_ACTION_DIM, dtype=np.float32)
        self._raw_error = np.zeros(REAL_ACTION_DIM, dtype=np.float32)
        self._raw_imu_qpos = np.zeros(REAL_ACTION_DIM, dtype=np.float32)
        self._has_raw_imu_qpos = False
        self._bucket_raw_imu_tracker = _BucketQuaternionPhaseTracker()
        self._policy_raw_delta = np.zeros(REAL_ACTION_DIM, dtype=np.float32)
        self._feedback_consistent = True
        self._stall_reference_error = np.zeros(REAL_ACTION_DIM, dtype=np.float32)
        self._stall_reference_s = np.zeros(REAL_ACTION_DIM, dtype=np.float64)
        self._stall_last_boost_s = np.full(REAL_ACTION_DIM, -np.inf, dtype=np.float64)
        self._stall_action_boost = np.zeros(REAL_ACTION_DIM, dtype=np.float32)
        self._axis_stalled = np.zeros(REAL_ACTION_DIM, dtype=bool)
        self._axis_wrong_direction = np.zeros(REAL_ACTION_DIM, dtype=bool)
        self._stall_action_sign = np.zeros(REAL_ACTION_DIM, dtype=np.float32)
        self._effective_min_action = np.zeros(REAL_ACTION_DIM, dtype=np.float32)
        self._action_limit = np.zeros(REAL_ACTION_DIM, dtype=np.float32)
        self.start_qpos = np.zeros(REAL_ACTION_DIM, dtype=np.float32)
        self.final_qpos = np.zeros(REAL_ACTION_DIM, dtype=np.float32)
        self.final_error = np.zeros(REAL_ACTION_DIM, dtype=np.float32)
        self.result_code = ""
        self.failed_reason = ""

    def start(self, obs: Mapping[str, Any], *, now_ns: int | None = None) -> None:
        qpos = _obs_qpos(obs)
        error = real_qpos_error_rad(self.config.home_pose_rad, qpos)
        policy_raw_delta, feedback_consistent, raw_imu_qpos = self._policy_raw_feedback(
            obs,
            qpos,
        )
        if not feedback_consistent:
            raise ValueError(
                "feedback_inconsistent policy_raw_delta_rad: "
                + _format_vector(policy_raw_delta)
            )
        if np.any(np.abs(error) > self.config.near_tolerance_rad):
            raise ValueError(
                "current pose is outside go-home near_tolerance_rad: "
                + _format_vector(error)
            )
        self.started = True
        self.start_time_ns = int(now_ns or time.time_ns())
        self.end_time_ns = 0
        self._start_s = time.monotonic()
        self._last_update_s = None
        self._last_decision_s = None
        self._filter_last_s = self._start_s
        self._stable_since_s = None
        self._axis_active = np.abs(error) > self.config.center_tolerance_rad
        self._axis_scheduled = np.zeros(REAL_ACTION_DIM, dtype=bool)
        self._center_approach_axis = np.zeros(REAL_ACTION_DIM, dtype=bool)
        self._axis_center_since_s = np.where(
            np.abs(error) <= self.config.center_tolerance_rad,
            self._start_s,
            np.nan,
        ).astype(np.float64)
        self._axis_coast_until_s = np.zeros(REAL_ACTION_DIM, dtype=np.float64)
        self._axis_ramp_start_s = np.full(REAL_ACTION_DIM, np.nan, dtype=np.float64)
        self._axis_ramp_sign = np.zeros(REAL_ACTION_DIM, dtype=np.float32)
        self._action_ramp_limit = np.zeros(REAL_ACTION_DIM, dtype=np.float32)
        self._axis_command_sign = np.zeros(REAL_ACTION_DIM, dtype=np.float32)
        self._axis_last_sign_change_s = np.full(REAL_ACTION_DIM, -np.inf, dtype=np.float64)
        self._sign_reversal_blocked = np.zeros(REAL_ACTION_DIM, dtype=bool)
        self._last_action = np.zeros(REAL_ACTION_DIM, dtype=np.float32)
        self._held_unsmoothed_action = np.zeros(REAL_ACTION_DIM, dtype=np.float32)
        self._filtered_qpos = qpos.astype(np.float32, copy=True)
        self._filtered_qvel = _obs_qvel(obs).astype(np.float32, copy=True)
        self._raw_error = error.astype(np.float32, copy=True)
        self._policy_raw_delta = policy_raw_delta.astype(np.float32, copy=True)
        self._feedback_consistent = bool(feedback_consistent)
        if raw_imu_qpos is None:
            self._raw_imu_qpos.fill(0.0)
            self._has_raw_imu_qpos = False
        else:
            self._raw_imu_qpos = raw_imu_qpos.astype(np.float32, copy=True)
            self._has_raw_imu_qpos = True
        abs_error = np.abs(error).astype(np.float32)
        self._stall_reference_error = abs_error.copy()
        self._stall_reference_s = np.full(REAL_ACTION_DIM, self._start_s, dtype=np.float64)
        self._stall_last_boost_s = np.full(REAL_ACTION_DIM, -np.inf, dtype=np.float64)
        self._stall_action_boost = np.zeros(REAL_ACTION_DIM, dtype=np.float32)
        self._axis_stalled = np.zeros(REAL_ACTION_DIM, dtype=bool)
        self._axis_wrong_direction = np.zeros(REAL_ACTION_DIM, dtype=bool)
        self._stall_action_sign = np.zeros(REAL_ACTION_DIM, dtype=np.float32)
        self._effective_min_action = np.zeros(REAL_ACTION_DIM, dtype=np.float32)
        self._action_limit = np.zeros(REAL_ACTION_DIM, dtype=np.float32)
        self.start_qpos = qpos.astype(np.float32, copy=True)
        self.final_qpos = qpos.astype(np.float32, copy=True)
        self.final_error = error.astype(np.float32, copy=True)
        self.result_code = "running"
        self.failed_reason = ""

    def update(self, obs: Mapping[str, Any], *, now_s: float | None = None) -> GoHomeResult:
        if not self.started:
            return self._fail("not_started", obs)
        now = time.monotonic() if now_s is None else float(now_s)
        try:
            raw_qpos = _obs_qpos(obs)
            raw_qvel = _obs_qvel(obs)
        except ValueError as exc:
            return self._fail(f"feedback_invalid:{exc}", obs)
        policy_raw_delta, feedback_consistent, raw_imu_qpos = self._policy_raw_feedback(
            obs,
            raw_qpos,
        )
        self._policy_raw_delta = policy_raw_delta.astype(np.float32, copy=True)
        self._feedback_consistent = bool(feedback_consistent)
        if raw_imu_qpos is None:
            self._raw_imu_qpos.fill(0.0)
            self._has_raw_imu_qpos = False
        else:
            self._raw_imu_qpos = raw_imu_qpos.astype(np.float32, copy=True)
            self._has_raw_imu_qpos = True
        qpos, qvel = self._filtered_feedback(raw_qpos, raw_qvel, now_s=now)

        error = real_qpos_error_rad(self.config.home_pose_rad, qpos)
        raw_error = real_qpos_error_rad(self.config.home_pose_rad, raw_qpos)
        self._raw_error = raw_error.astype(np.float32, copy=True)
        self.final_qpos = raw_qpos.astype(np.float32, copy=True)
        self.final_error = raw_error.astype(np.float32, copy=True)

        elapsed = max(0.0, now - self._start_s)

        runaway_limit = self.config.near_tolerance_rad * self.config.runaway_error_factor
        if np.any(np.abs(error) > runaway_limit):
            return self._fail("runaway_error", obs)

        abs_error = np.abs(error)
        acceptable_position = bool(
            feedback_consistent
            and np.all(abs_error <= self.config.success_tolerance_rad)
        )
        in_position = bool(
            feedback_consistent
            and np.all(abs_error <= self.config.center_tolerance_rad)
        )
        stable_velocity = np.all(np.abs(qvel) <= self.config.qvel_stable_rad_s)
        if elapsed > self.config.timeout_s and not (
            acceptable_position and stable_velocity
        ):
            return self._fail("timeout", obs)
        decision_updated = False
        if acceptable_position and stable_velocity:
            if self._stable_since_s is None:
                self._stable_since_s = now
            if now - self._stable_since_s >= self.config.dwell_s:
                return self._finish("succeeded", done=True)
        else:
            self._stable_since_s = None

        if in_position:
            self._axis_active = np.zeros(REAL_ACTION_DIM, dtype=bool)
            self._axis_scheduled = np.zeros(REAL_ACTION_DIM, dtype=bool)
            self._center_approach_axis = np.zeros(REAL_ACTION_DIM, dtype=bool)
            self._axis_stalled = np.zeros(REAL_ACTION_DIM, dtype=bool)
            self._axis_wrong_direction = np.zeros(REAL_ACTION_DIM, dtype=bool)
            self._stall_action_boost = np.zeros(REAL_ACTION_DIM, dtype=np.float32)
            self._effective_min_action = np.zeros(REAL_ACTION_DIM, dtype=np.float32)
            self._action_limit = np.zeros(REAL_ACTION_DIM, dtype=np.float32)
            self._action_ramp_limit = np.zeros(REAL_ACTION_DIM, dtype=np.float32)
            self._axis_ramp_start_s = np.full(REAL_ACTION_DIM, np.nan, dtype=np.float64)
            self._axis_ramp_sign = np.zeros(REAL_ACTION_DIM, dtype=np.float32)
            self._sign_reversal_blocked = np.zeros(REAL_ACTION_DIM, dtype=bool)
            self._held_unsmoothed_action = np.zeros(REAL_ACTION_DIM, dtype=np.float32)
            action = np.zeros(REAL_ACTION_DIM, dtype=np.float32)
        else:
            decision_updated = self._decision_due(now)
            if not decision_updated:
                action = self._held_unsmoothed_action.copy()
            else:
                self._last_decision_s = now
                action = self._compute_control_decision(
                    now=now,
                    error=error,
                    qvel=qvel,
                    abs_error=abs_error,
                )

        unsmoothed_action = np.asarray(action, dtype=np.float32)
        action = self._apply_slew_limit(unsmoothed_action, now_s=now)

        return GoHomeResult(
            action=np.asarray(action, dtype=np.float32),
            diagnostics=self._diagnostics(
                qpos=qpos,
                qvel=qvel,
                raw_qpos=raw_qpos,
                raw_qvel=raw_qvel,
                action=action,
                unsmoothed_action=unsmoothed_action,
                in_position=in_position,
                acceptable_position=acceptable_position,
                stable_velocity=stable_velocity,
                elapsed_s=elapsed,
                decision_updated=decision_updated,
            ),
        )

    def _decision_due(self, now_s: float) -> bool:
        period = float(self.config.control_decision_period_s)
        if period <= 0.0 or self._last_decision_s is None:
            return True
        return (now_s - self._last_decision_s) >= period

    def _compute_control_decision(
        self,
        *,
        now: float,
        error: np.ndarray,
        qvel: np.ndarray,
        abs_error: np.ndarray,
    ) -> np.ndarray:
        coast_waiting = now < self._axis_coast_until_s
        resume_axis = (abs_error > self.config.resume_tolerance_rad) & ~coast_waiting
        centered_axis = self._centered_axis_held(abs_error, now_s=now)
        self._axis_active = np.where(
            resume_axis,
            True,
            np.where(centered_axis, False, self._axis_active),
        )
        self._axis_active = np.where(coast_waiting, False, self._axis_active)
        moving_toward_center = (error * qvel) > 0.0
        approach_zone = (
            self._axis_active
            & moving_toward_center
            & (
                abs_error
                <= (
                    self.config.center_tolerance_rad
                    + np.abs(qvel) * self.config.coast_stop_time_s
                )
            )
        )
        low_speed_approach_axis = approach_zone & (
            self.config.center_approach_action > 0.0
        )
        coast_axis = approach_zone & ~low_speed_approach_axis
        if np.any(coast_axis):
            self._axis_coast_until_s = np.where(
                coast_axis,
                now + self.config.coast_reactivation_delay_s,
                self._axis_coast_until_s,
            )
        self._axis_active = np.where(coast_axis, False, self._axis_active)
        if not np.any(self._axis_active):
            outside_center = abs_error > self.config.center_tolerance_rad
            approach_to_center = (
                outside_center
                & moving_toward_center
                & (
                    abs_error
                    <= (
                        self.config.center_tolerance_rad
                        + np.abs(qvel) * self.config.coast_stop_time_s
                    )
                )
            )
            coasting_to_center = approach_to_center & (
                self.config.center_approach_action <= 0.0
            )
            reactivate_axis = outside_center & ~coasting_to_center & ~coast_waiting
            if np.any(reactivate_axis):
                relative_error = np.where(
                    reactivate_axis,
                    abs_error / self.config.center_tolerance_rad,
                    -np.inf,
                )
                self._axis_active[int(np.argmax(relative_error))] = True
        self._apply_axis_scheduler(abs_error=abs_error, coast_waiting=coast_waiting)
        pd_action = self.config.control_signs * (
            self.config.p_gain * error - self.config.d_gain * qvel
        )
        action_limit = np.where(
            abs_error <= self.config.success_tolerance_rad,
            self.config.near_max_action,
            self.config.max_action,
        ).astype(np.float32)
        self._center_approach_axis = (
            self._axis_active
            & moving_toward_center
            & (
                abs_error
                <= (
                    self.config.center_tolerance_rad
                    + np.abs(qvel) * self.config.coast_stop_time_s
                )
            )
            & (self.config.center_approach_action > 0.0)
        )
        action_limit = np.where(
            self._center_approach_axis,
            np.minimum(action_limit, self.config.center_approach_action),
            action_limit,
        ).astype(np.float32)
        action = np.where(self._axis_active, pd_action, 0.0)
        action = np.clip(action, -action_limit, action_limit)
        action_sign = _nonzero_sign(
            action,
            fallback=self.config.control_signs * error,
        )
        action = self._apply_sign_reversal_guard(
            now_s=now,
            action=action,
            action_sign=action_sign,
            abs_error=abs_error,
        )
        action_sign = _nonzero_sign(
            action,
            fallback=self.config.control_signs * error,
        )
        min_action = np.where(
            action_sign >= 0.0,
            self.config.min_action_positive,
            self.config.min_action_negative,
        )
        self._update_stall_boost(
            now_s=now,
            abs_error=abs_error,
            qvel=qvel,
            action_sign=action_sign,
            min_action=min_action,
        )
        action = self._apply_wrong_direction_guard(
            now_s=now,
            abs_error=abs_error,
            action=action,
        )
        action_sign = _nonzero_sign(
            action,
            fallback=self.config.control_signs * error,
        )
        boosted_axis = self._axis_stalled & (self._stall_action_boost > 0.0)
        action_limit = np.where(
            boosted_axis,
            np.maximum(action_limit, self.config.max_action),
            action_limit,
        ).astype(np.float32)
        effective_min_action = np.minimum(
            min_action + self._stall_action_boost,
            action_limit,
        )
        action_ramp_limit = self._update_action_ramp_limit(
            now_s=now,
            action_sign=action_sign,
            effective_min_action=effective_min_action,
            action_limit=action_limit,
        )
        self._effective_min_action = np.where(
            self._axis_active,
            effective_min_action,
            0.0,
        ).astype(np.float32)
        self._action_limit = action_limit.astype(np.float32)
        self._action_ramp_limit = np.where(
            self._axis_active,
            action_ramp_limit,
            0.0,
        ).astype(np.float32)
        action = np.clip(action, -action_ramp_limit, action_ramp_limit)
        small_active = (
            self._axis_active
            & (effective_min_action > 0.0)
            & (np.abs(action) < effective_min_action)
        )
        action = np.where(
            small_active,
            action_sign * effective_min_action,
            action,
        )
        action = np.clip(action, -action_ramp_limit, action_ramp_limit)
        self._held_unsmoothed_action = np.asarray(action, dtype=np.float32)
        return self._held_unsmoothed_action.copy()

    def _apply_axis_scheduler(
        self,
        *,
        abs_error: np.ndarray,
        coast_waiting: np.ndarray,
    ) -> None:
        active = np.asarray(self._axis_active, dtype=bool)
        if self.config.max_active_axes >= REAL_ACTION_DIM:
            self._axis_scheduled = active.copy()
            return

        eligible = (
            active
            & (abs_error > self.config.center_tolerance_rad)
            & ~np.asarray(coast_waiting, dtype=bool)
        )
        scheduled = np.zeros(REAL_ACTION_DIM, dtype=bool)
        count = 0
        for idx in self.config.axis_order:
            if not eligible[int(idx)]:
                continue
            scheduled[int(idx)] = True
            count += 1
            if count >= self.config.max_active_axes:
                break
        self._axis_scheduled = scheduled
        self._axis_active = active & scheduled

    def metadata(self) -> dict[str, Any]:
        return {
            "go_home_result": str(self.result_code),
            "go_home_failed_reason": str(self.failed_reason),
            "go_home_start_time_ns": int(self.start_time_ns),
            "go_home_end_time_ns": int(self.end_time_ns),
            "go_home_target_pose_rad": self.config.home_pose_rad.astype(np.float32),
            "go_home_start_qpos": self.start_qpos.astype(np.float32),
            "go_home_final_qpos": self.final_qpos.astype(np.float32),
            "go_home_final_error": self.final_error.astype(np.float32),
            "go_home_final_filtered_error": real_qpos_error_rad(
                self.config.home_pose_rad,
                self._filtered_qpos,
            ),
            "go_home_policy_raw_delta": self._policy_raw_delta.astype(np.float32),
            "go_home_feedback_consistent": int(bool(self._feedback_consistent)),
            "go_home_has_raw_imu_qpos": int(bool(self._has_raw_imu_qpos)),
        }

    def _finish(self, result_code: str, *, done: bool = False, failed: bool = False) -> GoHomeResult:
        self.result_code = result_code
        self.end_time_ns = int(time.time_ns())
        self.started = False
        self._last_decision_s = None
        self._last_action = np.zeros(REAL_ACTION_DIM, dtype=np.float32)
        self._held_unsmoothed_action = np.zeros(REAL_ACTION_DIM, dtype=np.float32)
        self._axis_scheduled = np.zeros(REAL_ACTION_DIM, dtype=bool)
        self._center_approach_axis = np.zeros(REAL_ACTION_DIM, dtype=bool)
        self._axis_command_sign = np.zeros(REAL_ACTION_DIM, dtype=np.float32)
        self._axis_last_sign_change_s = np.full(REAL_ACTION_DIM, -np.inf, dtype=np.float64)
        self._sign_reversal_blocked = np.zeros(REAL_ACTION_DIM, dtype=bool)
        self._axis_stalled = np.zeros(REAL_ACTION_DIM, dtype=bool)
        self._axis_wrong_direction = np.zeros(REAL_ACTION_DIM, dtype=bool)
        self._stall_action_boost = np.zeros(REAL_ACTION_DIM, dtype=np.float32)
        self._effective_min_action = np.zeros(REAL_ACTION_DIM, dtype=np.float32)
        self._action_limit = np.zeros(REAL_ACTION_DIM, dtype=np.float32)
        self._action_ramp_limit = np.zeros(REAL_ACTION_DIM, dtype=np.float32)
        self._axis_ramp_start_s = np.full(REAL_ACTION_DIM, np.nan, dtype=np.float64)
        self._axis_ramp_sign = np.zeros(REAL_ACTION_DIM, dtype=np.float32)
        filtered_error = real_qpos_error_rad(
            self.config.home_pose_rad,
            self._filtered_qpos,
        )
        return GoHomeResult(
            action=np.zeros(REAL_ACTION_DIM, dtype=np.float32),
            done=done,
            failed=failed,
            reason=self.failed_reason or result_code,
            diagnostics={
                "go_home_running": 0,
                "go_home_result_code": result_code,
                "go_home_error": filtered_error,
                "go_home_raw_error": self.final_error.astype(np.float32),
                "go_home_policy_raw_delta": self._policy_raw_delta.astype(np.float32),
                "go_home_feedback_consistent": int(bool(self._feedback_consistent)),
                "go_home_raw_imu_qpos": self._raw_imu_qpos.astype(np.float32),
                "go_home_has_raw_imu_qpos": int(bool(self._has_raw_imu_qpos)),
                "go_home_qvel": np.zeros(REAL_ACTION_DIM, dtype=np.float32),
                "go_home_raw_qvel": np.zeros(REAL_ACTION_DIM, dtype=np.float32),
                "go_home_filtered_qpos": self._filtered_qpos.astype(np.float32),
                "go_home_filtered_qvel": self._filtered_qvel.astype(np.float32),
                "go_home_commanded_action": np.zeros(REAL_ACTION_DIM, dtype=np.float32),
                "go_home_unsmoothed_action": np.zeros(REAL_ACTION_DIM, dtype=np.float32),
                "go_home_axis_active": np.zeros(REAL_ACTION_DIM, dtype=np.int32),
                "go_home_axis_scheduled": np.zeros(REAL_ACTION_DIM, dtype=np.int32),
                "go_home_center_approach_axis": np.zeros(
                    REAL_ACTION_DIM, dtype=np.int32
                ),
                "go_home_axis_stalled": np.zeros(REAL_ACTION_DIM, dtype=np.int32),
                "go_home_axis_wrong_direction": np.zeros(REAL_ACTION_DIM, dtype=np.int32),
                "go_home_stall_action_boost": np.zeros(REAL_ACTION_DIM, dtype=np.float32),
                "go_home_effective_min_action": np.zeros(REAL_ACTION_DIM, dtype=np.float32),
                "go_home_action_limit": np.zeros(REAL_ACTION_DIM, dtype=np.float32),
                "go_home_action_ramp_limit": np.zeros(REAL_ACTION_DIM, dtype=np.float32),
                "go_home_control_signs": self.config.control_signs.astype(np.float32),
                "go_home_command_sign": np.zeros(REAL_ACTION_DIM, dtype=np.float32),
                "go_home_sign_reversal_blocked": np.zeros(REAL_ACTION_DIM, dtype=np.int32),
                "go_home_decision_updated": 0,
                "go_home_control_decision_period_s": float(
                    self.config.control_decision_period_s
                ),
            },
        )

    def _fail(self, reason: str, obs: Mapping[str, Any]) -> GoHomeResult:
        try:
            qpos = _obs_qpos(obs)
            self.final_qpos = qpos.astype(np.float32, copy=True)
            self.final_error = real_qpos_error_rad(
                self.config.home_pose_rad,
                qpos,
            )
        except ValueError:
            pass
        self.failed_reason = str(reason)
        return self._finish("failed", failed=True)

    def _diagnostics(
        self,
        *,
        qpos: np.ndarray,
        qvel: np.ndarray,
        raw_qpos: np.ndarray,
        raw_qvel: np.ndarray,
        action: np.ndarray,
        unsmoothed_action: np.ndarray,
        in_position: bool,
        acceptable_position: bool,
        stable_velocity: bool,
        elapsed_s: float,
        decision_updated: bool,
    ) -> dict[str, Any]:
        return {
            "go_home_running": 1,
            "go_home_result_code": "running",
            "go_home_error": real_qpos_error_rad(self.config.home_pose_rad, qpos),
            "go_home_raw_error": real_qpos_error_rad(
                self.config.home_pose_rad,
                raw_qpos,
            ),
            "go_home_policy_raw_delta": self._policy_raw_delta.astype(np.float32),
            "go_home_feedback_consistent": int(bool(self._feedback_consistent)),
            "go_home_raw_imu_qpos": self._raw_imu_qpos.astype(np.float32),
            "go_home_has_raw_imu_qpos": int(bool(self._has_raw_imu_qpos)),
            "go_home_qvel": np.asarray(qvel, dtype=np.float32),
            "go_home_raw_qvel": np.asarray(raw_qvel, dtype=np.float32),
            "go_home_filtered_qpos": np.asarray(qpos, dtype=np.float32),
            "go_home_filtered_qvel": np.asarray(qvel, dtype=np.float32),
            "go_home_commanded_action": np.asarray(action, dtype=np.float32),
            "go_home_unsmoothed_action": np.asarray(unsmoothed_action, dtype=np.float32),
            "go_home_axis_active": self._axis_active.astype(np.int32),
            "go_home_axis_scheduled": self._axis_scheduled.astype(np.int32),
            "go_home_center_approach_axis": self._center_approach_axis.astype(np.int32),
            "go_home_axis_stalled": self._axis_stalled.astype(np.int32),
            "go_home_axis_wrong_direction": self._axis_wrong_direction.astype(np.int32),
            "go_home_stall_action_boost": self._stall_action_boost.astype(np.float32),
            "go_home_effective_min_action": self._effective_min_action.astype(np.float32),
            "go_home_action_limit": self._action_limit.astype(np.float32),
            "go_home_action_ramp_limit": self._action_ramp_limit.astype(np.float32),
            "go_home_control_signs": self.config.control_signs.astype(np.float32),
            "go_home_command_sign": self._axis_command_sign.astype(np.float32),
            "go_home_sign_reversal_blocked": self._sign_reversal_blocked.astype(np.int32),
            "go_home_decision_updated": int(bool(decision_updated)),
            "go_home_control_decision_period_s": float(
                self.config.control_decision_period_s
            ),
            "go_home_in_position": int(bool(in_position)),
            "go_home_acceptable_position": int(bool(acceptable_position)),
            "go_home_stable_velocity": int(bool(stable_velocity)),
            "go_home_elapsed_s": float(elapsed_s),
        }

    def _filtered_feedback(
        self,
        raw_qpos: np.ndarray,
        raw_qvel: np.ndarray,
        *,
        now_s: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        dt = 0.02 if self._filter_last_s is None else max(0.0, now_s - self._filter_last_s)
        self._filter_last_s = now_s
        aligned_qpos = align_real_qpos_to_reference_branch(raw_qpos, self._filtered_qpos)
        self._filtered_qpos = _lowpass_vector(
            aligned_qpos,
            self._filtered_qpos,
            tau_s=self.config.qpos_filter_tau_s,
            dt_s=dt,
        )
        self._filtered_qvel = _lowpass_vector(
            raw_qvel,
            self._filtered_qvel,
            tau_s=self.config.qvel_filter_tau_s,
            dt_s=dt,
        )
        return self._filtered_qpos.copy(), self._filtered_qvel.copy()

    def _policy_raw_feedback(
        self,
        obs: Mapping[str, Any],
        policy_qpos: np.ndarray,
    ) -> tuple[np.ndarray, bool, np.ndarray | None]:
        raw_imu_qpos = _obs_raw_imu_qpos(
            obs, bucket_tracker=self._bucket_raw_imu_tracker
        )
        if raw_imu_qpos is None:
            return np.zeros(REAL_ACTION_DIM, dtype=np.float32), True, None
        raw_imu_qpos = _align_raw_imu_qpos_to_policy_branch(raw_imu_qpos, policy_qpos)
        delta = real_qpos_error_rad(policy_qpos, raw_imu_qpos)
        gated_delta = np.abs(delta)
        if not _obs_has_explicit_raw_imu_qpos(obs):
            gated_delta = gated_delta.copy()
            gated_delta[_BUCKET_AXIS] = 0.0
        consistent = bool(
            np.all(gated_delta <= self.config.max_policy_raw_qpos_delta_rad)
        )
        return delta.astype(np.float32, copy=False), consistent, raw_imu_qpos

    def _centered_axis_held(self, abs_error: np.ndarray, *, now_s: float) -> np.ndarray:
        inside = abs_error <= self.config.center_tolerance_rad
        newly_inside = inside & np.isnan(self._axis_center_since_s)
        if np.any(newly_inside):
            self._axis_center_since_s = np.where(
                newly_inside,
                now_s,
                self._axis_center_since_s,
            )
        self._axis_center_since_s = np.where(
            inside,
            self._axis_center_since_s,
            np.nan,
        )
        held_s = np.where(
            np.isnan(self._axis_center_since_s),
            0.0,
            now_s - self._axis_center_since_s,
        )
        return inside & (held_s >= self.config.axis_center_hold_s)

    def _update_action_ramp_limit(
        self,
        *,
        now_s: float,
        action_sign: np.ndarray,
        effective_min_action: np.ndarray,
        action_limit: np.ndarray,
    ) -> np.ndarray:
        rate = self.config.action_ramp_rate.astype(np.float32)
        if np.all(rate <= 0.0):
            self._axis_ramp_start_s = np.where(
                self._axis_active,
                self._axis_ramp_start_s,
                np.nan,
            )
            self._axis_ramp_sign = np.where(
                self._axis_active,
                action_sign,
                0.0,
            ).astype(np.float32)
            return action_limit.astype(np.float32)

        active = np.asarray(self._axis_active, dtype=bool)
        sign = np.asarray(action_sign, dtype=np.float32)
        sign_changed = active & (self._axis_ramp_sign != 0.0) & (
            sign != self._axis_ramp_sign
        )
        needs_start = active & (np.isnan(self._axis_ramp_start_s) | sign_changed)
        if np.any(needs_start):
            self._axis_ramp_start_s = np.where(
                needs_start,
                now_s,
                self._axis_ramp_start_s,
            )
        inactive = ~active
        if np.any(inactive):
            self._axis_ramp_start_s[inactive] = np.nan
        self._axis_ramp_sign = np.where(active, sign, 0.0).astype(np.float32)

        age_s = np.where(
            np.isnan(self._axis_ramp_start_s),
            0.0,
            now_s - self._axis_ramp_start_s,
        )
        ramp_limit = effective_min_action + rate * age_s.astype(np.float32)
        ramp_limit = np.where(rate <= 0.0, action_limit, ramp_limit)
        return np.minimum(ramp_limit, action_limit).astype(np.float32)

    def _apply_sign_reversal_guard(
        self,
        *,
        now_s: float,
        action: np.ndarray,
        action_sign: np.ndarray,
        abs_error: np.ndarray,
    ) -> np.ndarray:
        active = np.asarray(self._axis_active, dtype=bool)
        sign = np.where(active, np.asarray(action_sign, dtype=np.float32), 0.0)
        previous = self._axis_command_sign.astype(np.float32)
        reversal = active & (previous != 0.0) & (sign != 0.0) & (sign != previous)
        enough_time = (
            now_s - self._axis_last_sign_change_s
        ) >= self.config.sign_reversal_delay_s
        enough_error = abs_error >= self.config.sign_reversal_min_error_rad
        blocked = reversal & ~(enough_time & enough_error)
        self._sign_reversal_blocked = blocked.astype(bool)
        if np.any(blocked):
            active = active.copy()
            active[blocked] = False
            self._axis_active = active
            self._axis_coast_until_s = np.where(
                blocked,
                np.maximum(
                    self._axis_coast_until_s,
                    now_s + self.config.sign_reversal_delay_s,
                ),
                self._axis_coast_until_s,
            )
            action = np.asarray(action, dtype=np.float32).copy()
            action[blocked] = 0.0

        allowed = active & (sign != 0.0) & ~blocked
        changed = allowed & ((previous == 0.0) | (sign != previous))
        if np.any(changed):
            self._axis_last_sign_change_s[changed] = float(now_s)
        self._axis_command_sign = np.where(allowed, sign, previous).astype(np.float32)
        return np.asarray(action, dtype=np.float32)

    def _apply_wrong_direction_guard(
        self,
        *,
        now_s: float,
        abs_error: np.ndarray,
        action: np.ndarray,
    ) -> np.ndarray:
        active = np.asarray(self._axis_active, dtype=bool)
        age_s = np.asarray(now_s - self._stall_reference_s, dtype=np.float32)
        growing_error = (
            abs_error
            >= (
                self._stall_reference_error
                + self.config.wrong_direction_error_increase_rad
            )
        )
        wrong = active & growing_error & (
            age_s >= float(self.config.wrong_direction_detection_s)
        )
        self._axis_wrong_direction = wrong.astype(bool)
        if not np.any(wrong):
            return np.asarray(action, dtype=np.float32)

        self._axis_active[wrong] = False
        self._axis_stalled[wrong] = False
        self._stall_action_boost[wrong] = 0.0
        self._effective_min_action[wrong] = 0.0
        self._action_limit[wrong] = 0.0
        self._action_ramp_limit[wrong] = 0.0
        self._axis_coast_until_s = np.where(
            wrong,
            np.maximum(
                self._axis_coast_until_s,
                now_s + self.config.wrong_direction_cooldown_s,
            ),
            self._axis_coast_until_s,
        )
        guarded = np.asarray(action, dtype=np.float32).copy()
        guarded[wrong] = 0.0
        return guarded

    def _apply_slew_limit(self, action: np.ndarray, *, now_s: float) -> np.ndarray:
        rate = self.config.action_slew_rate.astype(np.float32)
        if np.all(rate <= 0.0):
            self._last_update_s = now_s
            self._last_action = np.asarray(action, dtype=np.float32)
            return np.asarray(action, dtype=np.float32)
        dt = 0.02 if self._last_update_s is None else max(0.0, now_s - self._last_update_s)
        self._last_update_s = now_s
        unlimited = rate <= 0.0
        max_delta = rate * float(dt)
        delta = np.asarray(action, dtype=np.float32) - self._last_action
        limited_delta = np.where(unlimited, delta, np.clip(delta, -max_delta, max_delta))
        limited = self._last_action + limited_delta
        self._last_action = limited.astype(np.float32)
        return self._last_action.copy()

    def _update_stall_boost(
        self,
        *,
        now_s: float,
        abs_error: np.ndarray,
        qvel: np.ndarray,
        action_sign: np.ndarray,
        min_action: np.ndarray,
    ) -> None:
        active = np.asarray(self._axis_active, dtype=bool)
        sign = np.asarray(action_sign, dtype=np.float32)
        sign_changed = active & (self._stall_action_sign != 0.0) & (
            sign != self._stall_action_sign
        )
        reset = (~active) | sign_changed
        if np.any(reset):
            self._stall_reference_error[reset] = abs_error[reset].astype(np.float32)
            self._stall_reference_s[reset] = float(now_s)
            self._stall_last_boost_s[reset] = -np.inf
            self._stall_action_boost[reset] = 0.0
            self._axis_stalled[reset] = False
        self._stall_action_sign = np.where(active, sign, 0.0).astype(np.float32)

        improved = active & (
            abs_error
            <= (self._stall_reference_error - self.config.stall_error_progress_rad)
        )
        if np.any(improved):
            self._stall_reference_error[improved] = abs_error[improved].astype(np.float32)
            self._stall_reference_s[improved] = float(now_s)
            self._axis_stalled[improved] = False

        age_s = np.asarray(now_s - self._stall_reference_s, dtype=np.float32)
        no_progress = active & ~improved & (
            age_s >= float(self.config.stall_detection_s)
        )
        low_velocity = np.abs(qvel) <= self.config.stall_qvel_threshold_rad_s
        stalled = no_progress & low_velocity
        self._axis_stalled = stalled.astype(bool)
        boost_due = stalled & (
            (now_s - self._stall_last_boost_s)
            >= float(self.config.stall_boost_interval_s)
        )
        step = self.config.stall_action_step.astype(np.float32)
        boost_cap = np.maximum(
            self.config.max_action.astype(np.float32) - min_action.astype(np.float32),
            0.0,
        )
        if np.any(boost_due & (step > 0.0)):
            idx = boost_due & (step > 0.0)
            self._stall_action_boost[idx] = np.minimum(
                self._stall_action_boost[idx] + step[idx],
                boost_cap[idx],
            ).astype(np.float32)
            self._stall_last_boost_s[idx] = float(now_s)


def _obs_qpos(obs: Mapping[str, Any]) -> np.ndarray:
    if "qpos" not in obs:
        raise ValueError("missing qpos")
    return as_real_vector4(obs["qpos"], name="qpos")


def _obs_qvel(obs: Mapping[str, Any]) -> np.ndarray:
    if "qvel" not in obs:
        raise ValueError("missing qvel")
    return as_real_vector4(obs["qvel"], name="qvel")


def _quat_multiply_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.asarray(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )


def _quat_conjugate_np(q: np.ndarray) -> np.ndarray:
    return np.asarray([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def _normalize_quaternion_np(q: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(q))
    if not np.isfinite(norm) or norm <= 1e-9:
        return None
    return q / norm


def _bucket_relative_quaternion_wxyz(devices: Sequence[Any]) -> np.ndarray | None:
    def quaternion(device_index: int) -> np.ndarray | None:
        device = devices[device_index]
        if not isinstance(device, Mapping):
            return None
        try:
            if int(device.get("online", 1)) == 0 or int(device.get("valid_quaternion", 1)) == 0:
                return None
        except (TypeError, ValueError):
            return None
        try:
            q = np.asarray(device.get("quaternion_wxyz"), dtype=np.float64).reshape(-1)
        except (TypeError, ValueError):
            return None
        if q.shape != (4,) or not np.all(np.isfinite(q)):
            return None
        return _normalize_quaternion_np(q)

    imu1 = quaternion(0)
    imu2 = quaternion(1)
    if imu1 is None or imu2 is None:
        return None
    return _normalize_quaternion_np(_quat_multiply_np(_quat_conjugate_np(imu2), imu1))


def _bucket_quaternion_charts_from_relative_rad(
    relative: np.ndarray,
) -> tuple[float, float, float, float] | None:
    rel = _normalize_quaternion_np(relative)
    if rel is None:
        return None
    rel_w, rel_x, rel_y, rel_z = rel
    if _bucket_imu0_roll_profile_enabled():
        sign = _bucket_imu0_axis_sign()
        primary = float(
            sign
            * (np.remainder(2.0 * np.arctan2(rel_x, rel_w) + np.pi, 2.0 * np.pi)
               - np.pi)
        )
        secondary = float(
            sign
            * (np.remainder(2.0 * np.arctan2(rel_y, rel_z) + np.pi, 2.0 * np.pi)
               - np.pi)
        )
        primary_strength = float(np.hypot(rel_w, rel_x))
        secondary_strength = float(np.hypot(rel_y, rel_z))
    else:
        primary = float(
            np.remainder(2.0 * np.arctan2(rel_y, rel_w) + np.pi, 2.0 * np.pi)
            - np.pi
            + _BUCKET_QUATERNION_POLICY_OFFSET_RAD
        )
        secondary = float(
            np.remainder(-2.0 * np.arctan2(rel_x, rel_z) + np.pi, 2.0 * np.pi)
            - np.pi
        )
        primary_strength = float(np.hypot(rel_w, rel_y))
        secondary_strength = float(np.hypot(rel_x, rel_z))
    if not all(
        np.isfinite(v)
        for v in (primary, secondary, primary_strength, secondary_strength)
    ):
        return None
    return primary, secondary, primary_strength, secondary_strength


def _bucket_quaternion_charts_rad(
    devices: Sequence[Any],
    *,
    relative_reference: np.ndarray | None = None,
) -> tuple[float, float, float, float] | None:
    rel = _bucket_relative_quaternion_wxyz(devices)
    if rel is None:
        return None
    if _bucket_imu0_roll_profile_enabled() and relative_reference is not None:
        rel = _normalize_quaternion_np(_quat_multiply_np(_quat_conjugate_np(relative_reference), rel))
        if rel is None:
            return None
    return _bucket_quaternion_charts_from_relative_rad(rel)


def _bucket_quaternion_qpos_rad(devices: Sequence[Any]) -> float | None:
    charts = _bucket_quaternion_charts_rad(devices)
    return None if charts is None else _bucket_initial_position_rad(charts[0])


class _BucketQuaternionPhaseTracker:
    def __init__(self) -> None:
        self._ready = False
        self._primary_phase_rad = 0.0
        self._secondary_phase_rad = 0.0
        self._bucket_rad = 0.0
        self._profile = "legacy_y"
        self._sign = 1.0
        self._reference_rad = 0.0
        self._relative_reference: np.ndarray | None = None
        self._qpos_source = ""
        self._gravity_imu0_phase_rad = 0.0
        self._gravity_imu1_phase_rad = 0.0
        self._gravity_ready = False
        self._gravity_outer_zero_window: list[float] = []

    @staticmethod
    def _raw_roll_minus_pitch_rad(devices: Sequence[Any]) -> float | None:
        def raw_deg(device_index: int, axis_index: int) -> float | None:
            device = devices[device_index]
            if not isinstance(device, Mapping):
                return None
            try:
                rpy = np.asarray(device.get("rpy_raw_deg"), dtype=np.float64).reshape(-1)
            except (TypeError, ValueError):
                return None
            if rpy.shape != (3,) or not np.all(np.isfinite(rpy)):
                return None
            return float(rpy[axis_index])

        roll0 = raw_deg(0, 0)
        pitch1 = raw_deg(1, 1)
        if roll0 is None or pitch1 is None:
            return None
        return float(np.deg2rad(roll0 - pitch1))

    @staticmethod
    def _raw_daoyuan_chain_bucket_rad(devices: Sequence[Any]) -> float | None:
        def raw_deg(device_index: int, axis_index: int) -> float | None:
            device = devices[device_index]
            if not isinstance(device, Mapping):
                return None
            try:
                rpy = np.asarray(device.get("rpy_raw_deg"), dtype=np.float64).reshape(-1)
            except (TypeError, ValueError):
                return None
            if rpy.shape != (3,) or not np.all(np.isfinite(rpy)):
                return None
            return float(rpy[axis_index])

        roll0 = raw_deg(0, 0)
        pitch1 = raw_deg(1, 1)
        if roll0 is None or pitch1 is None:
            return None
        return -float(np.deg2rad(roll0 + pitch1))

    @staticmethod
    def _gravity_hinge_raw_rad(devices: Sequence[Any]) -> tuple[float, float, float] | None:
        def accel(device_index: int) -> np.ndarray | None:
            device = devices[device_index]
            if not isinstance(device, Mapping):
                return None
            try:
                if (
                    int(device.get("online", 1)) == 0
                    or int(device.get("valid_accel", 1)) == 0
                ):
                    return None
            except (TypeError, ValueError):
                return None
            try:
                values = np.asarray(device.get("accel_mps2"), dtype=np.float64).reshape(-1)
            except (TypeError, ValueError):
                return None
            if values.shape != (3,) or not np.all(np.isfinite(values)):
                return None
            if float(np.linalg.norm(values)) <= 1e-9:
                return None
            return values

        imu0 = accel(0)
        imu1 = accel(1)
        if imu0 is None or imu1 is None:
            return None
        imu0_phase = float(np.arctan2(float(imu0[2]), float(imu0[1])))
        imu1_phase = float(np.arctan2(-float(imu1[2]), float(imu1[0])))
        bucket = imu0_phase - imu1_phase
        if not all(np.isfinite(v) for v in (bucket, imu0_phase, imu1_phase)):
            return None
        return bucket, imu0_phase, imu1_phase

    def update(self, devices: Sequence[Any]) -> float | None:
        qpos_source = _bucket_qpos_source()
        profile = _bucket_imu0_profile()
        sign = _bucket_imu0_axis_sign()
        reference_rad = _bucket_imu0_reference_rad()
        if (
            self._ready
            and (
                qpos_source != self._qpos_source
                or profile != self._profile
                or sign != self._sign
                or abs(reference_rad - self._reference_rad) > 1e-12
            )
        ):
            self._ready = False
            self._gravity_ready = False
            self._gravity_outer_zero_window.clear()
            self._relative_reference = None
        self._qpos_source = qpos_source
        self._profile = profile
        self._sign = sign
        self._reference_rad = reference_rad
        if qpos_source == "gravity_hinge":
            raw = self._gravity_hinge_raw_rad(devices)
            if raw is None:
                return None
            _, imu0_phase, imu1_phase = raw
            if not self._gravity_ready:
                self._gravity_imu0_phase_rad = imu0_phase
                self._gravity_imu1_phase_rad = imu1_phase
                self._gravity_ready = True
            else:
                self._gravity_imu0_phase_rad = float(
                    self._gravity_imu0_phase_rad
                    + np.remainder(
                        imu0_phase - self._gravity_imu0_phase_rad + np.pi,
                        2.0 * np.pi,
                    )
                    - np.pi
                )
                self._gravity_imu1_phase_rad = float(
                    self._gravity_imu1_phase_rad
                    + np.remainder(
                        imu1_phase - self._gravity_imu1_phase_rad + np.pi,
                        2.0 * np.pi,
                    )
                    - np.pi
                )
            bucket_raw = self._gravity_imu0_phase_rad - self._gravity_imu1_phase_rad
            outer_zero = bucket_raw - _bucket_gravity_hinge_reference_rad()
            self._gravity_outer_zero_window.append(float(outer_zero))
            median_window = _bucket_gravity_hinge_median_window()
            if len(self._gravity_outer_zero_window) > median_window:
                del self._gravity_outer_zero_window[
                    : len(self._gravity_outer_zero_window) - median_window
                ]
            median_outer_zero = float(np.median(self._gravity_outer_zero_window))
            return (
                median_outer_zero
                + _bucket_gravity_hinge_reference_rad()
                + _bucket_gravity_hinge_policy_offset_rad()
            )
        if qpos_source == "rpy":
            phase = self._raw_roll_minus_pitch_rad(devices)
            if phase is None:
                return None
            phase = sign * (phase - reference_rad)
            if not self._ready:
                self._primary_phase_rad = phase
                self._secondary_phase_rad = phase
                self._bucket_rad = phase
                self._ready = True
                return self._bucket_rad
            phase = float(self._primary_phase_rad + np.remainder(phase - self._primary_phase_rad + np.pi, 2.0 * np.pi) - np.pi)
            self._bucket_rad += float(np.remainder(phase - self._primary_phase_rad + np.pi, 2.0 * np.pi) - np.pi)
            self._primary_phase_rad = phase
            self._secondary_phase_rad = phase
            return self._bucket_rad
        charts = _bucket_quaternion_charts_rad(
            devices,
            relative_reference=self._relative_reference,
        )
        if charts is None:
            return None
        primary, secondary, primary_strength, secondary_strength = charts
        if not self._ready:
            self._primary_phase_rad = primary
            self._secondary_phase_rad = secondary
            self._bucket_rad = _bucket_initial_position_rad(primary)
            self._ready = True
            return self._bucket_rad
        use_secondary = (
            primary_strength < _BUCKET_PRIMARY_CHART_MIN_STRENGTH
            and secondary_strength > primary_strength
        )
        primary_delta = float(
            np.remainder(primary - self._primary_phase_rad + np.pi, 2.0 * np.pi)
            - np.pi
        )
        secondary_delta = float(
            np.remainder(secondary - self._secondary_phase_rad + np.pi, 2.0 * np.pi)
            - np.pi
        )
        self._bucket_rad += secondary_delta if use_secondary else primary_delta
        self._primary_phase_rad = primary
        self._secondary_phase_rad = secondary
        return self._bucket_rad

    def update_daoyuan_chain(
        self,
        devices: Sequence[Any],
        *,
        reference_rad: float,
    ) -> float | None:
        qpos_source = "daoyuan_chain"
        profile = "daoyuan_chain"
        sign = 1.0
        if (
            self._ready
            and (
                qpos_source != self._qpos_source
                or profile != self._profile
                or sign != self._sign
                or abs(reference_rad - self._reference_rad) > 1e-12
            )
        ):
            self._ready = False
            self._gravity_ready = False
            self._gravity_outer_zero_window.clear()
            self._relative_reference = None
        self._qpos_source = qpos_source
        self._profile = profile
        self._sign = sign
        self._reference_rad = reference_rad
        phase = self._raw_daoyuan_chain_bucket_rad(devices)
        if phase is None:
            return None
        phase += reference_rad
        if not self._ready:
            self._primary_phase_rad = phase
            self._secondary_phase_rad = phase
            self._bucket_rad = phase
            self._ready = True
            return self._bucket_rad
        phase = float(
            self._primary_phase_rad
            + np.remainder(phase - self._primary_phase_rad + np.pi, 2.0 * np.pi)
            - np.pi
        )
        self._bucket_rad += float(
            np.remainder(phase - self._primary_phase_rad + np.pi, 2.0 * np.pi)
            - np.pi
        )
        self._primary_phase_rad = phase
        self._secondary_phase_rad = phase
        return self._bucket_rad


def _obs_raw_imu_qpos(
    obs: Mapping[str, Any],
    *,
    bucket_tracker: _BucketQuaternionPhaseTracker | None = None,
) -> np.ndarray | None:
    if "qpos_raw_imu" in obs:
        return as_real_vector4(obs["qpos_raw_imu"], name="qpos_raw_imu")
    if "qpos_raw_imu_deg" in obs:
        return np.deg2rad(
            as_real_vector4(obs["qpos_raw_imu_deg"], name="qpos_raw_imu_deg")
        ).astype(np.float32)

    imu_debug = obs.get("imu_debug")
    if not isinstance(imu_debug, Mapping):
        sensor_health = obs.get("sensor_health")
        if isinstance(sensor_health, Mapping):
            imu_debug = sensor_health.get("imu_debug")
    if not isinstance(imu_debug, Mapping):
        return None
    devices = imu_debug.get("devices")
    if not isinstance(devices, Sequence) or len(devices) < 4:
        return None

    def raw_deg(device_index: int, axis_index: int) -> float | None:
        device = devices[device_index]
        if not isinstance(device, Mapping):
            return None
        try:
            if int(device.get("online", 1)) == 0 or int(device.get("valid_attitude", 1)) == 0:
                return None
        except (TypeError, ValueError):
            return None
        try:
            rpy = np.asarray(device.get("rpy_raw_deg"), dtype=np.float32).reshape(-1)
        except (TypeError, ValueError):
            return None
        if rpy.shape != (3,) or not np.all(np.isfinite(rpy)):
            return None
        return float(rpy[axis_index])

    imu0_roll = raw_deg(0, 0)
    imu2_y = raw_deg(1, 1)
    imu3_y = raw_deg(2, 1)
    imu4_z = raw_deg(3, 2)
    joint_profile = _joint_rpy_profile(imu_debug)
    if joint_profile == "daoyuan_chain":
        bucket_offset = _daoyuan_chain_bucket_policy_offset_rad(imu_debug)
        if bucket_tracker is not None:
            bucket_rad = bucket_tracker.update_daoyuan_chain(
                devices,
                reference_rad=bucket_offset,
            )
        elif imu0_roll is not None and imu2_y is not None:
            bucket_rad = -float(np.deg2rad(float(imu0_roll) + float(imu2_y))) + bucket_offset
        else:
            bucket_rad = None
        if None in (imu2_y, imu3_y, imu4_z, bucket_rad):
            return None
        return np.asarray(
            [
                np.deg2rad(float(imu4_z)),
                np.deg2rad(float(imu3_y)),
                np.deg2rad(float(imu2_y) + float(imu3_y))
                + _daoyuan_chain_stick_policy_offset_rad(imu_debug),
                float(bucket_rad),
            ],
            dtype=np.float32,
        )
    if bucket_tracker is not None:
        bucket_rad = bucket_tracker.update(devices)
    elif _bucket_imu0_roll_profile_enabled():
        bucket_rad = (
            None
            if imu0_roll is None or imu2_y is None
            else _bucket_imu0_axis_sign()
            * (float(np.deg2rad(float(imu0_roll) - float(imu2_y))) - _bucket_imu0_reference_rad())
        )
    else:
        bucket_rad = _bucket_quaternion_qpos_rad(devices)
    if None in (imu2_y, imu3_y, imu4_z, bucket_rad):
        return None
    return np.asarray(
        [
            np.deg2rad(float(imu4_z)),
            np.deg2rad(float(imu3_y)),
            np.deg2rad(float(imu2_y) - float(imu3_y)),
            float(bucket_rad),
        ],
        dtype=np.float32,
    )


def _obs_has_explicit_raw_imu_qpos(obs: Mapping[str, Any]) -> bool:
    return "qpos_raw_imu" in obs or "qpos_raw_imu_deg" in obs


def _align_raw_imu_qpos_to_policy_branch(
    raw_imu_qpos: np.ndarray,
    policy_qpos: np.ndarray,
) -> np.ndarray:
    aligned = align_real_qpos_to_reference_branch(raw_imu_qpos, policy_qpos)
    policy = as_real_vector4(policy_qpos, name="policy_qpos")
    for axis in range(REAL_ACTION_DIM):
        if axis == 0:
            continue
        aligned[axis] += round(
            (float(policy[axis]) - float(aligned[axis])) / (2.0 * np.pi)
        ) * (2.0 * np.pi)
    return aligned.astype(np.float32, copy=False)


def _format_vector(value: Any) -> str:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    return "[" + ",".join(f"{float(v):+.4f}" for v in arr) + "]"


def _nonzero_sign(value: Any, *, fallback: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    fallback_arr = np.asarray(fallback, dtype=np.float32)
    sign = np.sign(arr)
    return np.where(sign == 0.0, np.sign(fallback_arr), sign)


def _lowpass_vector(
    raw: np.ndarray,
    previous: np.ndarray,
    *,
    tau_s: np.ndarray,
    dt_s: float,
) -> np.ndarray:
    raw_arr = np.asarray(raw, dtype=np.float32)
    prev_arr = np.asarray(previous, dtype=np.float32)
    tau = np.asarray(tau_s, dtype=np.float32)
    if dt_s <= 0.0 or np.all(tau <= 0.0):
        return raw_arr.astype(np.float32)
    alpha = np.where(tau <= 0.0, 1.0, float(dt_s) / (tau + float(dt_s)))
    return (prev_arr + alpha * (raw_arr - prev_arr)).astype(np.float32)
