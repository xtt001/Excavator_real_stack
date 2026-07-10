from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from testbed.policies.act.adapter import ACTAdapter
from testbed.policies.runtime_gate_stack import (
    BASE_FEATURE_NAMES,
    RuntimeGateStack,
    _DirectionGateMlp,
    _ScalarGateMlp,
    _temporal_feature_names,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _constant_scalar(path: Path, probability: float) -> None:
    model = _ScalarGateMlp(input_dim=16, hidden_dim=2)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.net[-1].bias.fill_(torch.logit(torch.tensor(probability)))
    torch.save(
        {
            "feature_names": BASE_FEATURE_NAMES,
            "hidden_dim": 2,
            "feature_mean": np.zeros(16, dtype=np.float32),
            "feature_std": np.ones(16, dtype=np.float32),
            "state_dict": model.state_dict(),
        },
        path,
    )


def _constant_temporal(path: Path, probabilities: list[float]) -> Path:
    offsets = [-1, 0]
    names = _temporal_feature_names(offsets)
    model = _DirectionGateMlp(input_dim=len(names), hidden_dim=2)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.net[-1].bias.copy_(torch.logit(torch.tensor(probabilities)))
    torch.save(
        {
            "feature_names": BASE_FEATURE_NAMES,
            "hidden_dim": 2,
            "feature_mean": np.zeros(len(names), dtype=np.float32),
            "feature_std": np.ones(len(names), dtype=np.float32),
            "model_state_dict": model.state_dict(),
        },
        path,
    )
    metadata = path.with_name("temporal_direction_gate_model_metadata.json")
    metadata.write_text(
        json.dumps(
            {
                "feature_names": names,
                "base_feature_names": BASE_FEATURE_NAMES,
                "context_offsets": offsets,
            }
        ),
        encoding="utf-8",
    )
    return metadata


def _runtime_bundle(tmp_path: Path) -> tuple[Path, Path]:
    phase = tmp_path / "phase.pt"
    tail = tmp_path / "tail.pt"
    eligibility = tmp_path / "eligibility.pt"
    temporal = tmp_path / "temporal.pt"
    _constant_scalar(phase, 0.9)
    _constant_scalar(tail, 0.9)
    _constant_scalar(eligibility, 0.9)
    temporal_metadata = _constant_temporal(
        temporal,
        [0.9, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1],
    )
    deadzone = tmp_path / "deadzone.json"
    deadzone.write_text(
        json.dumps(
            {
                "deadzone_action": {
                    axis: {"pos": 0.5, "neg": 0.5}
                    for axis in ("swing", "boom", "stick", "bucket")
                }
            }
        ),
        encoding="utf-8",
    )
    artifacts = []
    for name, path in (
        ("phase_gate_model", phase),
        ("tail_candidate_model", tail),
        ("gohome_eligibility_model", eligibility),
        ("temporal_direction_model", temporal),
        ("temporal_direction_metadata", temporal_metadata),
    ):
        artifacts.append({"name": name, "path": str(path), "sha256": _sha256(path)})
    manifest = tmp_path / "candidate_package_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "candidate_id": "E52-test",
                "selected_gates": {
                    "phase_gate": "simple_0.15_s0.50",
                    "gohome_gate": "learned_tail_t0.80_tc2_e0.80_ec2",
                    "direction_threshold": 0.5,
                    "direction_inactive_scale": 0.75,
                    "snap_margin": 0.02,
                    "snap_intent_threshold": 0.7,
                },
                "artifacts": artifacts,
            }
        ),
        encoding="utf-8",
    )
    return manifest, deadzone


def test_runtime_gate_stack_loads_bundle_applies_stack_and_resets(
    tmp_path: Path,
) -> None:
    manifest, deadzone = _runtime_bundle(tmp_path)
    stack = RuntimeGateStack.from_config(
        {
            "enabled": True,
            "manifest_path": str(manifest),
            "deadzone_json": str(deadzone),
        }
    )
    kwargs = {
        "action": np.array([0.49, -0.49, 0.2, -0.2], dtype=np.float32),
        "intent_probabilities": np.array(
            [0.9, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0], dtype=np.float32
        ),
        "qpos": np.zeros(4, dtype=np.float32),
        "qvel": np.zeros(4, dtype=np.float32),
    }

    first = stack.step(**kwargs)
    second = stack.step(**kwargs)
    third = stack.step(**kwargs)

    np.testing.assert_allclose(first.action, [0.501, -0.501, 0.15, -0.15], atol=1e-6)
    np.testing.assert_array_equal(
        first.diagnostics["policy_snap_active_mask"], [1, 1, 0, 0]
    )
    assert not first.gohome_requested
    assert second.gohome_requested
    assert second.diagnostics["gohome_request_active"] == 1
    assert not third.gohome_requested
    assert third.diagnostics["gohome_raw_active"] == 1

    stack.reset()
    after_reset = stack.step(**kwargs)
    assert not after_reset.gohome_requested
    assert after_reset.diagnostics["gohome_candidate_consecutive_steps"] == 1


def test_runtime_gate_stack_rejects_future_temporal_offset(tmp_path: Path) -> None:
    manifest, deadzone = _runtime_bundle(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    metadata_entry = next(
        item
        for item in payload["artifacts"]
        if item["name"] == "temporal_direction_metadata"
    )
    metadata_path = Path(metadata_entry["path"])
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["context_offsets"] = [0, 1]
    metadata["feature_names"] = _temporal_feature_names([0, 1])
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    metadata_entry["sha256"] = _sha256(metadata_path)
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="future offsets"):
        RuntimeGateStack.from_config(
            {
                "enabled": True,
                "manifest_path": str(manifest),
                "deadzone_json": str(deadzone),
            }
        )


def test_runtime_gate_stack_rejects_invalid_intent_probability(tmp_path: Path) -> None:
    manifest, deadzone = _runtime_bundle(tmp_path)
    stack = RuntimeGateStack.from_config(
        {
            "enabled": True,
            "manifest_path": str(manifest),
            "deadzone_json": str(deadzone),
        }
    )

    with pytest.raises(ValueError, match=r"finite values in \[0, 1\]"):
        stack.step(
            action=np.zeros(4, dtype=np.float32),
            intent_probabilities=np.array(
                [1.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                dtype=np.float32,
            ),
            qpos=np.zeros(4, dtype=np.float32),
            qvel=np.zeros(4, dtype=np.float32),
        )


class _FakeActModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, proprio, image, env_state):
        self.calls += 1
        actions = torch.tensor(
            [[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]],
            dtype=torch.float32,
        )
        intent_logits = torch.zeros((1, 2, 8), dtype=torch.float32)
        intent_logits[0, 0, 0] = torch.logit(torch.tensor(0.8))
        return actions, None, (None, None), intent_logits


class _FakeLegacyActModel(_FakeActModel):
    def forward(self, proprio, image, env_state):
        actions, is_pad, latent, _ = super().forward(proprio, image, env_state)
        return actions, is_pad, latent


def test_act_adapter_action_and_intent_uses_one_inference_and_one_step() -> None:
    adapter = ACTAdapter.__new__(ACTAdapter)
    adapter.device = torch.device("cpu")
    adapter._low_dim_keys = ["qpos", "qvel"]
    adapter._camera_names = ["fpv"]
    adapter._proprio_mean = torch.zeros(8)
    adapter._proprio_std = torch.ones(8)
    adapter._normalize = lambda value: value
    adapter._model = _FakeActModel()
    adapter.temporal_agg = False
    adapter._num_queries = 2
    adapter._t = 0
    adapter.norm_stats = {
        "action_mean": np.zeros(4, dtype=np.float32),
        "action_std": np.ones(4, dtype=np.float32),
    }
    obs = {
        "qpos": np.zeros(4, dtype=np.float32),
        "qvel": np.zeros(4, dtype=np.float32),
        "image_fpv": np.zeros((3, 4, 5), dtype=np.float32),
    }

    action, intent = adapter.predict_action_and_intent(obs)

    np.testing.assert_allclose(action, [0.1, 0.2, 0.3, 0.4])
    np.testing.assert_allclose(intent, [0.8, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    assert adapter._model.calls == 1
    assert adapter._t == 1

    np.testing.assert_allclose(adapter.predict(obs), [0.5, 0.6, 0.7, 0.8])
    assert adapter._model.calls == 2
    assert adapter._t == 2


def test_act_adapter_predict_preserves_legacy_model_without_intent_head() -> None:
    adapter = ACTAdapter.__new__(ACTAdapter)
    adapter.device = torch.device("cpu")
    adapter._low_dim_keys = ["qpos", "qvel"]
    adapter._camera_names = ["fpv"]
    adapter._proprio_mean = torch.zeros(8)
    adapter._proprio_std = torch.ones(8)
    adapter._normalize = lambda value: value
    adapter._model = _FakeLegacyActModel()
    adapter.temporal_agg = False
    adapter._num_queries = 2
    adapter._t = 0
    adapter.norm_stats = {
        "action_mean": np.zeros(4, dtype=np.float32),
        "action_std": np.ones(4, dtype=np.float32),
    }
    obs = {
        "qpos": np.zeros(4, dtype=np.float32),
        "qvel": np.zeros(4, dtype=np.float32),
        "image_fpv": np.zeros((3, 4, 5), dtype=np.float32),
    }

    np.testing.assert_allclose(adapter.predict(obs), [0.1, 0.2, 0.3, 0.4])
    assert adapter._model.calls == 1
    assert adapter._t == 1
    with pytest.raises(ValueError, match="no intent logits"):
        adapter.predict_action_and_intent(obs)
    assert adapter._model.calls == 2
    assert adapter._t == 2
