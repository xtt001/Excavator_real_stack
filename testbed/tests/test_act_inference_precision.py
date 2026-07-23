from __future__ import annotations

from contextlib import nullcontext
from unittest.mock import patch

import numpy as np
import pytest
import torch
from scripts.compare_act_inference_precision import _acceptance, _comparison
from torch import nn

from testbed.policies.act.adapter import (
    ACTAdapter,
    _resolve_inference_autocast_dtype,
    _single_frame_image_tensor,
)


def test_fp32_precision_disables_autocast_on_cpu() -> None:
    assert (
        _resolve_inference_autocast_dtype("fp32", device=torch.device("cpu"))
        is None
    )


def test_fp16_precision_requires_cuda() -> None:
    with pytest.raises(ValueError, match="requires a CUDA device"):
        _resolve_inference_autocast_dtype("fp16", device=torch.device("cpu"))


def test_fp16_precision_resolves_cuda_autocast_dtype() -> None:
    assert _resolve_inference_autocast_dtype(
        "fp16",
        device=torch.device("cuda"),
    ) is torch.float16


def test_unknown_inference_precision_is_rejected() -> None:
    with pytest.raises(ValueError, match="inference_precision"):
        _resolve_inference_autocast_dtype("int8", device=torch.device("cuda"))


class _HalfOutputModel(nn.Module):
    def forward(self, proprio, image, env_state):
        del proprio, image, env_state
        action = torch.ones((1, 3, 4), dtype=torch.float16)
        is_pad = torch.zeros((1, 3, 1), dtype=torch.float16)
        intent = torch.ones((1, 3, 8), dtype=torch.float16)
        return action, is_pad, [None, None], intent


def test_mixed_precision_scope_returns_fp32_policy_outputs() -> None:
    adapter = ACTAdapter.__new__(ACTAdapter)
    adapter.device = torch.device("cuda")
    adapter._inference_autocast_dtype = torch.float16
    adapter._model = _HalfOutputModel()

    with patch("torch.autocast", return_value=nullcontext()) as autocast:
        action, intent = adapter._forward_inference(
            torch.zeros((1, 4)),
            torch.zeros((1, 2, 3, 4, 4)),
        )

    autocast.assert_called_once_with(
        device_type="cuda",
        dtype=torch.float16,
        enabled=True,
    )
    assert action.dtype is torch.float32
    assert intent is not None
    assert intent.dtype is torch.float32


def test_device_uint8_preprocess_matches_legacy_tensor() -> None:
    rng = np.random.default_rng(7)
    images = [
        rng.integers(0, 256, size=(12, 16, 3), dtype=np.uint8),
        rng.integers(0, 256, size=(12, 16, 3), dtype=np.uint8),
    ]
    keys = ["image_video4", "image_video5"]

    legacy = _single_frame_image_tensor(
        images,
        keys=keys,
        device=torch.device("cpu"),
        device_uint8_preprocess=False,
    )
    optimized = _single_frame_image_tensor(
        images,
        keys=keys,
        device=torch.device("cpu"),
        device_uint8_preprocess=True,
    )

    assert optimized.shape == (1, 2, 3, 12, 16)
    torch.testing.assert_close(optimized, legacy, rtol=0.0, atol=1e-7)


def test_inference_compile_wraps_model_once() -> None:
    adapter = ACTAdapter.__new__(ACTAdapter)
    adapter._model = nn.Identity()
    adapter._inference_compile = True
    adapter._inference_compile_applied = False
    adapter._inference_compile_mode = "reduce-overhead"
    adapter._inference_compile_dynamic = False
    compiled = nn.Sequential(nn.Identity())

    with patch("torch.compile", return_value=compiled) as compile_fn:
        adapter._compile_model_for_inference()
        adapter._compile_model_for_inference()

    compile_fn.assert_called_once()
    assert compile_fn.call_args.kwargs == {
        "mode": "reduce-overhead",
        "dynamic": False,
    }
    assert adapter._model is compiled
    assert adapter._inference_compile_applied is True


def test_precision_comparison_detects_deadzone_class_change() -> None:
    thresholds = {
        axis: {"pos": 0.5, "neg": 0.5}
        for axis in ("swing", "boom", "stick", "bucket")
    }
    reference = torch.tensor([[0.49, 0.0, 0.0, 0.0]]).numpy()
    candidate = torch.tensor([[0.51, 0.0, 0.0, 0.0]]).numpy()

    result = _comparison(reference, candidate, thresholds=thresholds)

    assert result["deadzone_class_disagreement_count"] == 1
    assert result["max_abs_diff"] == pytest.approx(0.02)


def test_acceptance_requires_latency_improvement_and_requested_semantics() -> None:
    comparison = {
        "max_abs_diff": 0.001,
        "deadzone_class_disagreement_count": 0,
    }

    accepted = _acceptance(
        executed=comparison,
        raw_chunks=comparison,
        max_action_abs_diff=0.01,
        p95_speedup=1.2,
        min_p95_speedup=1.0,
        require_deadzone_equivalence=True,
    )
    slower = _acceptance(
        executed=comparison,
        raw_chunks=comparison,
        max_action_abs_diff=0.01,
        p95_speedup=0.9,
        min_p95_speedup=1.0,
        require_deadzone_equivalence=True,
    )

    assert accepted["passed"] is True
    assert slower["passed"] is False
