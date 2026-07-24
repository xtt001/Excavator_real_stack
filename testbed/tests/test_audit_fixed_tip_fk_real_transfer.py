from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "audit_fixed_tip_fk_real_transfer.py"
)
SPEC = importlib.util.spec_from_file_location(
    "audit_fixed_tip_fk_real_transfer", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_unity_normalize_maps_endpoints() -> None:
    minimum = np.asarray([-1.0, -2.0, -3.0, -4.0])
    maximum = np.asarray([1.0, 2.0, 3.0, 4.0])
    raw = np.stack((minimum, (minimum + maximum) / 2.0, maximum))
    normalized = MODULE.unity_normalize(raw, minimum, maximum)
    np.testing.assert_allclose(normalized[0], 0.0)
    np.testing.assert_allclose(normalized[1], 0.5)
    np.testing.assert_allclose(normalized[2], 1.0)


def test_unity_normalize_does_not_clip_extrapolation() -> None:
    minimum = np.zeros(4)
    maximum = np.ones(4)
    raw = np.asarray([[-0.1, 0.5, 1.1, 0.5]])
    normalized = MODULE.unity_normalize(raw, minimum, maximum)
    np.testing.assert_allclose(normalized, raw)
