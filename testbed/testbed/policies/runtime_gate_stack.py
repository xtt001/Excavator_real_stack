"""Runtime owner for the E52 ACT gate stack."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from testbed.policies.deadzone_eval import AXIS_NAMES, load_deadzone_thresholds

BASE_FEATURE_NAMES = [
    f"intent_{axis}_{direction}" for axis in AXIS_NAMES for direction in ("pos", "neg")
] + [f"{kind}_{axis}" for kind in ("qpos", "qvel") for axis in AXIS_NAMES]

_SCALAR_ARTIFACT_NAMES = (
    "phase_gate_model",
    "tail_candidate_model",
    "gohome_eligibility_model",
)
_TEMPORAL_ARTIFACT_NAME = "temporal_direction_model"
_TEMPORAL_METADATA_NAME = "temporal_direction_metadata"


@dataclass(frozen=True)
class RuntimeGateResult:
    action: np.ndarray
    gohome_requested: bool
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class _ModelBundle:
    model: torch.nn.Module
    mean: np.ndarray
    std: np.ndarray
    feature_names: list[str]


@dataclass(frozen=True)
class _TemporalModelBundle(_ModelBundle):
    offsets: list[int]


class RuntimeGateStack:
    """Apply E52 phase, snap, temporal-direction, and gohome gates per step."""

    def __init__(
        self,
        *,
        stack_id: str,
        phase_model: _ModelBundle,
        tail_model: _ModelBundle,
        eligibility_model: _ModelBundle,
        temporal_model: _TemporalModelBundle,
        phase_threshold: float,
        phase_inactive_scale: float,
        candidate_threshold: float,
        candidate_consecutive_steps: int,
        eligibility_threshold: float,
        eligibility_consecutive_steps: int,
        direction_threshold: float,
        direction_inactive_scale: float,
        snap_margin: float,
        snap_intent_threshold: float,
        snap_epsilon: float,
        deadzone_positive: np.ndarray,
        deadzone_negative: np.ndarray,
    ) -> None:
        for bundle in (phase_model, tail_model, eligibility_model):
            if bundle.feature_names != BASE_FEATURE_NAMES:
                raise ValueError(
                    f"gate feature names mismatch: {bundle.feature_names!r}"
                )
        if not temporal_model.offsets:
            raise ValueError("temporal direction model has no context offsets")
        if any(offset > 0 for offset in temporal_model.offsets):
            raise ValueError(
                "temporal direction model contains future offsets: "
                f"{temporal_model.offsets!r}"
            )
        expected_temporal_names = _temporal_feature_names(temporal_model.offsets)
        if temporal_model.feature_names != expected_temporal_names:
            raise ValueError(
                "temporal direction feature names do not match causal contract"
            )

        self.stack_id = str(stack_id)
        self._phase_model = phase_model
        self._tail_model = tail_model
        self._eligibility_model = eligibility_model
        self._temporal_model = temporal_model
        self._phase_threshold = _probability(phase_threshold, "phase threshold")
        self._phase_inactive_scale = _scale(
            phase_inactive_scale, "phase inactive scale"
        )
        self._candidate_threshold = _probability(
            candidate_threshold, "candidate threshold"
        )
        self._candidate_required = _positive_count(
            candidate_consecutive_steps, "candidate consecutive steps"
        )
        self._eligibility_threshold = _probability(
            eligibility_threshold, "eligibility threshold"
        )
        self._eligibility_required = _positive_count(
            eligibility_consecutive_steps, "eligibility consecutive steps"
        )
        self._direction_threshold = _probability(
            direction_threshold, "direction threshold"
        )
        self._direction_inactive_scale = _scale(
            direction_inactive_scale, "direction inactive scale"
        )
        self._snap_margin = _nonnegative(snap_margin, "snap margin")
        self._snap_intent_threshold = _probability(
            snap_intent_threshold, "snap intent threshold"
        )
        self._snap_epsilon = _nonnegative(snap_epsilon, "snap epsilon")
        self._deadzone_positive = _axis_vector(deadzone_positive, "positive deadzone")
        self._deadzone_negative = _axis_vector(deadzone_negative, "negative deadzone")
        if np.any(self._deadzone_positive <= 0.0) or np.any(
            self._deadzone_negative <= 0.0
        ):
            raise ValueError("deadzone thresholds must be positive")
        self.reset()

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        *,
        default_bundle_dir: str | Path | None = None,
    ) -> RuntimeGateStack:
        cfg = dict(config)
        if not bool(cfg.get("enabled", False)):
            raise ValueError(
                "runtime_gates.enabled must be true when constructing the stack"
            )

        bundle_dir_raw = cfg.get("bundle_dir", default_bundle_dir)
        manifest_raw = cfg.get("manifest_path")
        if manifest_raw is None and bundle_dir_raw is None:
            raise ValueError("runtime_gates requires manifest_path or bundle_dir")
        bundle_dir = (
            Path(bundle_dir_raw).expanduser() if bundle_dir_raw is not None else None
        )
        manifest_path = (
            Path(manifest_raw).expanduser()
            if manifest_raw is not None
            else bundle_dir / "candidate_package_manifest.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        selected = dict(manifest.get("selected_gates", {}) or {})
        artifacts = {
            str(item["name"]): dict(item)
            for item in manifest.get("artifacts", [])
            if isinstance(item, dict) and "name" in item
        }
        overrides = dict(cfg.get("artifacts", {}) or {})

        def artifact_path(name: str) -> Path:
            if name not in artifacts:
                raise ValueError(f"candidate manifest missing artifact {name!r}")
            entry = artifacts[name]
            configured = overrides.get(name)
            if configured is None and bundle_dir is not None:
                path = bundle_dir / Path(str(entry["path"])).name
            else:
                path = Path(
                    configured if configured is not None else entry["path"]
                ).expanduser()
                if not path.is_absolute() and bundle_dir is not None:
                    path = bundle_dir / path
            try:
                path_available = path.exists()
            except OSError:
                path_available = False
            if not path_available:
                raise FileNotFoundError(path)
            expected_sha = str(entry.get("sha256", ""))
            if expected_sha and _sha256(path) != expected_sha:
                raise ValueError(f"artifact sha256 mismatch for {name!r}: {path}")
            return path

        phase_gate = _parse_phase_gate(str(selected["phase_gate"]))
        gohome_gate = _parse_gohome_gate(str(selected["gohome_gate"]))
        scalar_models = {
            name: _load_scalar_model(artifact_path(name))
            for name in _SCALAR_ARTIFACT_NAMES
        }
        temporal_path = artifact_path(_TEMPORAL_ARTIFACT_NAME)
        metadata_path = artifact_path(_TEMPORAL_METADATA_NAME)
        temporal_model = _load_temporal_model(temporal_path, metadata_path)

        deadzone_raw = cfg.get("deadzone_json")
        if deadzone_raw is None:
            raise ValueError("runtime_gates.deadzone_json is required")
        deadzone_path = Path(deadzone_raw).expanduser()
        if not deadzone_path.is_absolute() and bundle_dir is not None:
            deadzone_path = bundle_dir / deadzone_path
        thresholds = load_deadzone_thresholds(deadzone_path)

        return cls(
            stack_id=str(manifest.get("candidate_id", "")),
            phase_model=scalar_models["phase_gate_model"],
            tail_model=scalar_models["tail_candidate_model"],
            eligibility_model=scalar_models["gohome_eligibility_model"],
            temporal_model=temporal_model,
            phase_threshold=phase_gate["threshold"],
            phase_inactive_scale=phase_gate["inactive_scale"],
            candidate_threshold=gohome_gate["candidate_threshold"],
            candidate_consecutive_steps=gohome_gate["candidate_consecutive_steps"],
            eligibility_threshold=gohome_gate["eligibility_threshold"],
            eligibility_consecutive_steps=gohome_gate["eligibility_consecutive_steps"],
            direction_threshold=float(selected["direction_threshold"]),
            direction_inactive_scale=float(selected["direction_inactive_scale"]),
            snap_margin=float(selected["snap_margin"]),
            snap_intent_threshold=float(selected["snap_intent_threshold"]),
            snap_epsilon=float(cfg.get("snap_epsilon", 0.001)),
            deadzone_positive=np.asarray(
                [thresholds[axis]["pos"] for axis in AXIS_NAMES], dtype=np.float32
            ),
            deadzone_negative=np.asarray(
                [thresholds[axis]["neg"] for axis in AXIS_NAMES], dtype=np.float32
            ),
        )

    def reset(self) -> None:
        self._feature_history: list[np.ndarray] = []
        self._candidate_run = 0
        self._eligibility_run = 0
        self._gohome_emitted = False

    def step(
        self,
        *,
        action: np.ndarray,
        intent_probabilities: np.ndarray,
        qpos: np.ndarray,
        qvel: np.ndarray,
    ) -> RuntimeGateResult:
        raw_action = _axis_vector(action, "action")
        intent = np.asarray(intent_probabilities, dtype=np.float32).reshape(-1)
        if intent.shape != (8,):
            raise ValueError(
                f"intent probabilities must have shape (8,), got {intent.shape}"
            )
        if not np.all(np.isfinite(intent)) or np.any(intent < 0.0) or np.any(intent > 1.0):
            raise ValueError("intent probabilities must be finite values in [0, 1]")
        feature_row = np.concatenate(
            [intent, _axis_vector(qpos, "qpos"), _axis_vector(qvel, "qvel")]
        ).astype(np.float32)
        self._feature_history.append(feature_row)
        max_history = 1 - min(self._temporal_model.offsets)
        if len(self._feature_history) > max_history:
            self._feature_history = self._feature_history[-max_history:]

        phase_prob = _predict(self._phase_model, feature_row)
        candidate_prob = _predict(self._tail_model, feature_row)
        eligibility_prob = _predict(self._eligibility_model, feature_row)
        temporal_features = self._temporal_features()
        direction_prob = _predict_vector(self._temporal_model, temporal_features)

        phase_active = phase_prob >= self._phase_threshold
        phase_action = raw_action.copy()
        if not phase_active:
            phase_action *= self._phase_inactive_scale

        snap_action, snap_mask = self._snap(phase_action, phase_active, intent)
        direction_active = direction_prob >= self._direction_threshold
        direction_action = snap_action.copy()
        positive = direction_action > 0.0
        negative = direction_action < 0.0
        direction_action[positive & ~direction_active[0::2]] *= (
            self._direction_inactive_scale
        )
        direction_action[negative & ~direction_active[1::2]] *= (
            self._direction_inactive_scale
        )

        self._candidate_run = (
            self._candidate_run + 1
            if candidate_prob >= self._candidate_threshold
            else 0
        )
        self._eligibility_run = (
            self._eligibility_run + 1
            if eligibility_prob >= self._eligibility_threshold
            else 0
        )
        candidate_active = self._candidate_run >= self._candidate_required
        eligibility_active = self._eligibility_run >= self._eligibility_required
        raw_gohome_active = candidate_active and eligibility_active
        gohome_requested = raw_gohome_active and not self._gohome_emitted
        if gohome_requested:
            self._gohome_emitted = True

        diagnostics = {
            "policy_gate_stack_id": self.stack_id,
            "policy_intent_probabilities": intent.copy(),
            "phase_gate_prob": float(phase_prob),
            "phase_gate_threshold": float(self._phase_threshold),
            "phase_gate_inactive_scale": float(self._phase_inactive_scale),
            "phase_gate_active": int(phase_active),
            "policy_phase_gated_action": phase_action.copy(),
            "policy_snap_active_mask": snap_mask.astype(np.int32),
            "policy_snap_action": snap_action.copy(),
            "policy_snap_margin": float(self._snap_margin),
            "policy_snap_intent_threshold": float(self._snap_intent_threshold),
            "temporal_direction_gate_probabilities": direction_prob.copy(),
            "temporal_direction_gate_threshold": float(self._direction_threshold),
            "temporal_direction_gate_inactive_scale": float(
                self._direction_inactive_scale
            ),
            "temporal_direction_gate_active_mask": direction_active.astype(np.int32),
            "policy_temporal_direction_action": direction_action.copy(),
            "gohome_candidate_probability": float(candidate_prob),
            "gohome_candidate_threshold": float(self._candidate_threshold),
            "gohome_candidate_required_steps": int(self._candidate_required),
            "gohome_candidate_consecutive_steps": int(self._candidate_run),
            "gohome_eligibility_probability": float(eligibility_prob),
            "gohome_eligibility_threshold": float(self._eligibility_threshold),
            "gohome_eligibility_required_steps": int(self._eligibility_required),
            "gohome_eligibility_consecutive_steps": int(self._eligibility_run),
            "gohome_request_probability": float(min(candidate_prob, eligibility_prob)),
            "gohome_raw_active": int(raw_gohome_active),
            "gohome_request_active": int(gohome_requested),
        }
        return RuntimeGateResult(
            action=direction_action.astype(np.float32),
            gohome_requested=bool(gohome_requested),
            diagnostics=diagnostics,
        )

    def _temporal_features(self) -> np.ndarray:
        current = len(self._feature_history) - 1
        chunks = [
            self._feature_history[max(0, current + offset)]
            for offset in self._temporal_model.offsets
        ]
        return np.concatenate(chunks).astype(np.float32)

    def _snap(
        self,
        action: np.ndarray,
        phase_active: bool,
        intent: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        out = action.copy()
        pos_intent = intent[0::2] >= self._snap_intent_threshold
        neg_intent = intent[1::2] >= self._snap_intent_threshold
        pos_near = (
            phase_active
            & pos_intent
            & (action >= self._deadzone_positive - self._snap_margin)
            & (action < self._deadzone_positive)
        )
        neg_near = (
            phase_active
            & neg_intent
            & (action <= -self._deadzone_negative + self._snap_margin)
            & (action > -self._deadzone_negative)
        )
        out[pos_near] = self._deadzone_positive[pos_near] + self._snap_epsilon
        out[neg_near] = -self._deadzone_negative[neg_near] - self._snap_epsilon
        return out.astype(np.float32), pos_near | neg_near


class _ScalarGateMlp(torch.nn.Module):
    def __init__(self, *, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _DirectionGateMlp(torch.nn.Module):
    def __init__(self, *, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 8),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _load_scalar_model(path: Path) -> _ModelBundle:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    names = list(payload["feature_names"])
    model = _ScalarGateMlp(input_dim=len(names), hidden_dim=int(payload["hidden_dim"]))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return _model_bundle(model, payload, names)


def _load_temporal_model(path: Path, metadata_path: Path) -> _TemporalModelBundle:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    names = list(metadata.get("feature_names", []))
    offsets = [int(value) for value in metadata.get("context_offsets", [])]
    if list(metadata.get("base_feature_names", [])) != BASE_FEATURE_NAMES:
        raise ValueError("temporal direction base feature names mismatch")
    mean = np.asarray(payload["feature_mean"], dtype=np.float32).reshape(-1)
    if len(names) != mean.shape[0]:
        raise ValueError("temporal direction metadata feature dimension mismatch")
    model = _DirectionGateMlp(
        input_dim=len(names), hidden_dim=int(payload["hidden_dim"])
    )
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    bundle = _model_bundle(model, payload, names)
    return _TemporalModelBundle(**bundle.__dict__, offsets=offsets)


def _model_bundle(
    model: torch.nn.Module,
    payload: dict[str, Any],
    names: list[str],
) -> _ModelBundle:
    mean = np.asarray(payload["feature_mean"], dtype=np.float32).reshape(-1)
    std = np.asarray(payload["feature_std"], dtype=np.float32).reshape(-1)
    if mean.shape != (len(names),) or std.shape != mean.shape:
        raise ValueError("gate normalization shape does not match feature names")
    if (
        not np.all(np.isfinite(mean))
        or not np.all(np.isfinite(std))
        or np.any(std <= 0.0)
    ):
        raise ValueError("gate normalization values must be finite with positive std")
    return _ModelBundle(model=model, mean=mean, std=std, feature_names=names)


def _predict(bundle: _ModelBundle, features: np.ndarray) -> float:
    return float(_predict_vector(bundle, features).reshape(-1)[0])


def _predict_vector(bundle: _ModelBundle, features: np.ndarray) -> np.ndarray:
    row = np.asarray(features, dtype=np.float32).reshape(-1)
    if row.shape != bundle.mean.shape:
        raise ValueError(
            f"gate feature shape mismatch: got {row.shape}, expected {bundle.mean.shape}"
        )
    normalized = (row - bundle.mean) / bundle.std
    with torch.inference_mode():
        logits = bundle.model(torch.as_tensor(normalized).unsqueeze(0))
        return torch.sigmoid(logits).squeeze(0).cpu().numpy().astype(np.float32)


def _parse_phase_gate(value: str) -> dict[str, float]:
    match = re.fullmatch(r"simple_([^_]+)_s(.+)", value)
    if match is None:
        raise ValueError(f"unsupported phase gate {value!r}")
    return {"threshold": float(match.group(1)), "inactive_scale": float(match.group(2))}


def _parse_gohome_gate(value: str) -> dict[str, float | int]:
    match = re.fullmatch(r"learned_tail_t([^_]+)_tc(\d+)_e([^_]+)_ec(\d+)", value)
    if match is None:
        raise ValueError(f"unsupported gohome gate {value!r}")
    return {
        "candidate_threshold": float(match.group(1)),
        "candidate_consecutive_steps": int(match.group(2)),
        "eligibility_threshold": float(match.group(3)),
        "eligibility_consecutive_steps": int(match.group(4)),
    }


def _temporal_feature_names(offsets: list[int]) -> list[str]:
    names: list[str] = []
    for offset in offsets:
        suffix = f"t{offset:+d}" if offset else "t0"
        names.extend(f"{name}_{suffix}" for name in BASE_FEATURE_NAMES)
    return names


def _axis_vector(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32).reshape(-1)
    if result.shape != (4,):
        raise ValueError(f"{name} must have shape (4,), got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result.astype(np.float32, copy=True)


def _scale(value: float, name: str) -> float:
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must satisfy 0 <= value <= 1")
    return result


def _probability(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be a finite value in [0, 1]")
    return result


def _nonnegative(value: float, name: str) -> float:
    result = float(value)
    if result < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _positive_count(value: int, name: str) -> int:
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be >= 1")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
