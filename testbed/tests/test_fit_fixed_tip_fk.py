from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "fit_fixed_tip_fk.py"
)
SPEC = importlib.util.spec_from_file_location("fit_fixed_tip_fk", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_polynomial_power_count_and_fit() -> None:
    powers = MODULE.polynomial_powers(4, 3)
    assert powers.shape == (35, 4)
    rng = np.random.default_rng(4)
    inputs = rng.normal(size=(100, 4)).astype(np.float32)
    targets = np.stack(
        (
            inputs[:, 0] + inputs[:, 1] ** 2,
            inputs[:, 2] * inputs[:, 3],
            np.ones(len(inputs), dtype=np.float32) * 0.5,
        ),
        axis=1,
    )
    model = MODULE.fit_polynomial(inputs, targets, degree=2, ridge=1.0e-8)
    prediction = MODULE.predict(model, inputs)
    np.testing.assert_allclose(prediction, targets, atol=2.0e-4)


def test_grid_cell_mapping_uses_longer_z_axis() -> None:
    half_extents = np.asarray([1.25, 0.025, 1.5], dtype=np.float32)
    points = np.asarray(
        [
            [-0.5, 0.0, -1.0],
            [0.5, 0.0, -1.0],
            [-0.5, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [-0.5, 0.0, 1.0],
            [0.5, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    cells, in_bounds, _ = MODULE.grid_cells(points, half_extents)
    assert in_bounds.all()
    np.testing.assert_array_equal(cells, np.arange(6))


def test_polynomial_module_is_torchscript_portable(tmp_path: Path) -> None:
    rng = np.random.default_rng(5)
    inputs = rng.normal(size=(80, 4)).astype(np.float32)
    targets = rng.normal(size=(80, 3)).astype(np.float32)
    model = MODULE.fit_polynomial(inputs, targets, degree=2, ridge=1.0e-4)
    example = torch.zeros((1, 4), dtype=torch.float32)
    path = tmp_path / "fk.ts"
    torch.jit.trace(model, example).save(str(path))
    loaded = torch.jit.load(str(path), map_location="cpu")
    with torch.inference_mode():
        expected = model(torch.as_tensor(inputs[:4]))
        actual = loaded(torch.as_tensor(inputs[:4]))
    torch.testing.assert_close(actual, expected)
