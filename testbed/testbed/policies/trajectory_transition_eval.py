"""Short-horizon transition features for trajectory-support evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from testbed.policies.offline_eval import AXIS_NAMES
from testbed.policies.trajectory_support_eval import effective_action_channels


@dataclass(frozen=True)
class TransitionSamples:
    start_steps: np.ndarray
    target_qpos_delta: np.ndarray
    initial_qvel_displacement: np.ndarray
    action_impulse: np.ndarray


@dataclass(frozen=True)
class LinearTransitionModel:
    coefficients: np.ndarray

    def predict(
        self,
        qvel_displacement: np.ndarray,
        action_impulse: np.ndarray,
    ) -> np.ndarray:
        qvel = np.asarray(qvel_displacement, dtype=np.float64)
        action = np.asarray(action_impulse, dtype=np.float64)
        if qvel.shape != action.shape:
            raise ValueError(
                f"qvel and action features must share shape, got {qvel.shape} and {action.shape}"
            )
        if not np.all(np.isfinite(qvel)) or not np.all(np.isfinite(action)):
            raise ValueError("qvel and action features must be finite")
        return self.coefficients[0] + self.coefficients[1] * qvel + self.coefficients[2] * action


@dataclass(frozen=True)
class FeatureSupportModel:
    mean: np.ndarray
    inverse_covariance: np.ndarray
    distance_threshold: float

    def distances(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.mean.size:
            raise ValueError(f"features must have shape (N, {self.mean.size}), got {values.shape}")
        centered = values - self.mean
        squared = np.einsum("ni,ij,nj->n", centered, self.inverse_covariance, centered)
        return np.sqrt(np.maximum(squared, 0.0))


def fit_linear_transition_model(
    qvel_displacement: np.ndarray,
    action_impulse: np.ndarray,
    target_qpos_delta: np.ndarray,
) -> LinearTransitionModel:
    qvel = np.asarray(qvel_displacement, dtype=np.float64)
    action = np.asarray(action_impulse, dtype=np.float64)
    target = np.asarray(target_qpos_delta, dtype=np.float64)
    if qvel.ndim != 1 or qvel.shape != action.shape or qvel.shape != target.shape or qvel.size < 3:
        raise ValueError("transition fit requires at least three aligned one-dimensional samples")
    if not np.all(np.isfinite(qvel)) or not np.all(np.isfinite(action)) or not np.all(np.isfinite(target)):
        raise ValueError("transition fit samples must be finite")
    design = np.column_stack([np.ones(qvel.size), qvel, action])
    return LinearTransitionModel(coefficients=np.linalg.lstsq(design, target, rcond=None)[0])


def fit_feature_support_model(
    features: np.ndarray,
    *,
    quantile: float = 0.99,
    regularization: float = 1.0e-6,
) -> FeatureSupportModel:
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 3 or values.shape[1] == 0:
        raise ValueError("support fit requires at least three rows of non-empty features")
    if not np.all(np.isfinite(values)):
        raise ValueError("support fit features must be finite")
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be between zero and one")
    if not np.isfinite(regularization) or regularization <= 0.0:
        raise ValueError("regularization must be finite and positive")
    scale = np.maximum(np.var(values, axis=0), 1.0e-12)
    covariance = np.cov(values, rowvar=False) + np.diag(scale * float(regularization))
    model = FeatureSupportModel(
        mean=values.mean(axis=0),
        inverse_covariance=np.linalg.pinv(covariance),
        distance_threshold=0.0,
    )
    threshold = float(np.quantile(model.distances(values), quantile))
    return FeatureSupportModel(model.mean, model.inverse_covariance, threshold)


def build_transition_samples(
    *,
    qpos: np.ndarray,
    qvel: np.ndarray,
    action: np.ndarray,
    thresholds: Mapping[str, Mapping[str, float]],
    dt: float,
    horizon_steps: int,
    stride: int,
    qvel_to_qpos_sign: np.ndarray,
    action_to_qpos_sign: np.ndarray,
) -> TransitionSamples:
    """Build aligned state targets and causal command features."""

    qpos_values = _validate_matrix(qpos, name="qpos")
    qvel_values = _validate_matrix(qvel, name="qvel")
    action_values = _validate_matrix(action, name="action")
    if qpos_values.shape != qvel_values.shape or qpos_values.shape != action_values.shape:
        raise ValueError(
            f"qpos, qvel, and action must share shape, got {qpos_values.shape}, "
            f"{qvel_values.shape}, {action_values.shape}"
        )
    step_s = float(dt)
    if not np.isfinite(step_s) or step_s <= 0.0:
        raise ValueError("dt must be finite and positive")
    horizon = int(horizon_steps)
    if horizon <= 0 or horizon >= qpos_values.shape[0]:
        raise ValueError(f"horizon_steps must satisfy 0 < horizon < {qpos_values.shape[0]}")
    step_stride = int(stride)
    if step_stride <= 0:
        raise ValueError("stride must be positive")
    qvel_sign = _validate_signs(qvel_to_qpos_sign, name="qvel_to_qpos_sign")
    action_sign = _validate_signs(action_to_qpos_sign, name="action_to_qpos_sign")

    starts = np.arange(0, qpos_values.shape[0] - horizon, step_stride, dtype=np.int64)
    target = qpos_values[starts + horizon] - qpos_values[starts]
    target[:, 0] = (target[:, 0] + np.pi) % (2.0 * np.pi) - np.pi
    initial_velocity = qvel_values[starts] * qvel_sign.reshape(1, -1) * (horizon * step_s)
    channels = effective_action_channels(action_values, thresholds)
    signed_effective = (channels[:, :, 0] - channels[:, :, 1]) * action_sign.reshape(1, -1)
    impulse = np.stack(
        [signed_effective[start : start + horizon].sum(axis=0) * step_s for start in starts],
        axis=0,
    )
    return TransitionSamples(
        start_steps=starts,
        target_qpos_delta=target,
        initial_qvel_displacement=initial_velocity,
        action_impulse=impulse,
    )


def _validate_matrix(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != len(AXIS_NAMES):
        raise ValueError(f"{name} must have shape (T, {len(AXIS_NAMES)}), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    return array


def _validate_signs(values: np.ndarray, *, name: str) -> np.ndarray:
    signs = np.asarray(values, dtype=np.float64)
    if signs.shape != (len(AXIS_NAMES),) or not np.all(np.isin(signs, (-1.0, 1.0))):
        raise ValueError(f"{name} must contain four values chosen from -1 and 1")
    return signs
