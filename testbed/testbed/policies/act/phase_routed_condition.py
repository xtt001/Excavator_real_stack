"""Frozen B1.3 condition-routing configuration and projection."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn

from testbed.simverify.observable_phase_router import ObservablePhaseRouter

STATE_DIM = 8
CONDITION_FACTOR_DIM = 3
TOTAL_DIM = 14


def resolve_phase_routed_condition_config(raw: Any) -> dict[str, Any]:
    cfg = dict(raw or {})
    if not bool(cfg.get("enabled", False)):
        return {
            "enabled": False,
            "state_slice": [0, 8],
            "current_condition_slice": [8, 11],
            "next_condition_slice": [11, 14],
        }
    expected = {
        "schema": "simverify_phase_routed_separated_condition_v1",
        "state_slice": [0, 8],
        "current_condition_slice": [8, 11],
        "next_condition_slice": [11, 14],
        "route_order": ["current", "neutral", "next"],
        "neutral_condition_influence": "exact_zero",
        "vae_encoder_inputs": ["qpos", "qvel", "actions"],
    }
    for key, expected_value in expected.items():
        if cfg.get(key) != expected_value:
            raise ValueError(
                f"phase_routed_condition.{key} must be "
                f"{expected_value!r}, got {cfg.get(key)!r}"
            )
    artifact_root = Path(str(cfg.get("router_artifact_root", ""))).resolve(
        strict=True
    )
    files = {
        "manifest": artifact_root / "phase_router_manifest.json",
        "params": artifact_root / "phase_router_params_v1.json",
        "assignments": artifact_root / "phase_route_assignments_v1.npz",
        "gate": artifact_root / "phase_router_gate_v1.json",
        "checksums": artifact_root / "checksums.sha256",
    }
    expected_shas = {
        "manifest": str(cfg.get("router_manifest_sha256", "")),
        "params": str(cfg.get("router_params_sha256", "")),
        "assignments": str(cfg.get("route_assignments_sha256", "")),
        "checksums": str(cfg.get("router_checksums_sha256", "")),
    }
    for key, expected_sha in expected_shas.items():
        if len(expected_sha) != 64:
            raise ValueError(
                f"phase_routed_condition {key} SHA-256 must be explicit"
            )
        actual = _sha256(files[key])
        if actual != expected_sha:
            raise ValueError(
                f"phase_routed_condition {key} SHA mismatch: "
                f"expected {expected_sha}, got {actual}"
            )
    manifest = _read_json(files["manifest"])
    params = _read_json(files["params"])
    gate = _read_json(files["gate"])
    if (
        manifest.get("decision") != "pass_observable_phase_router_prerequisite"
        or manifest.get("authorizes_b1_3_training") is not True
        or manifest.get("held_out_test_read") is not False
        or gate.get("authorizes_b1_3_training") is not True
        or gate.get("held_out_test_read") is not False
    ):
        raise ValueError("phase router artifact does not authorize B1.3")
    if manifest.get("params_sha256") != expected_shas["params"]:
        raise ValueError("phase router manifest params linkage mismatch")
    if manifest.get("assignments_sha256") != expected_shas["assignments"]:
        raise ValueError("phase router manifest assignments linkage mismatch")
    dwell_steps = int(params.get("dwell_steps", 0))
    if dwell_steps <= 0:
        raise ValueError("phase router artifact has invalid dwell")
    return {
        **cfg,
        "enabled": True,
        "router_artifact_root": str(artifact_root),
        "router_files": {key: str(value) for key, value in files.items()},
        "classifier": params,
        "dwell_steps": dwell_steps,
    }


def build_runtime_phase_router(
    config: Mapping[str, Any],
) -> ObservablePhaseRouter | None:
    if not bool(config.get("enabled", False)):
        return None
    return ObservablePhaseRouter(
        config["classifier"],
        dwell_steps=int(config["dwell_steps"]),
    )


class RoutedConditionProjection(nn.Module):
    """Project state and the two condition factors through disjoint paths."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.state = nn.Linear(STATE_DIM, hidden_dim)
        self.current = nn.Linear(CONDITION_FACTOR_DIM, hidden_dim)
        self.next = nn.Linear(CONDITION_FACTOR_DIM, hidden_dim)

    def forward(
        self,
        proprio: torch.Tensor,
        route: torch.Tensor,
    ) -> torch.Tensor:
        if proprio.ndim != 2 or proprio.shape[1] != TOTAL_DIM:
            raise ValueError(
                "phase-routed condition requires normalized proprio shape (B, 14)"
            )
        route_tensor = route.to(
            device=proprio.device,
            dtype=torch.int64,
        ).reshape(-1)
        if route_tensor.shape[0] != proprio.shape[0]:
            raise ValueError("condition route batch does not match proprio batch")
        if torch.any((route_tensor < 0) | (route_tensor > 2)):
            raise ValueError("condition routes must be current=0, neutral=1, next=2")
        state = self.state(proprio[:, :8])
        current_gate = (route_tensor == 0).to(proprio.dtype).unsqueeze(1)
        next_gate = (route_tensor == 2).to(proprio.dtype).unsqueeze(1)
        return (
            state
            + current_gate * self.current(proprio[:, 8:11])
            + next_gate * self.next(proprio[:, 11:14])
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

