#!/usr/bin/env python3
"""Fit and validate a qpos-only fixed bucket-tip forward-kinematics model."""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


DEFAULT_INPUT = Path(
    "docs/evidence/sim_fixed_tip_fk_v0_2/fixed_tip_samples.jsonl"
)
DEFAULT_OUTPUT = Path("docs/evidence/sim_fixed_tip_fk_v0_2")
EXPECTED_SCHEMA = "fixed_tip_fk_sample/v1"
EXPECTED_CANDIDATE = "bucket_center_tooth_leading_edge_midpoint_v0_1"
INPUT_ORDER = (
    "swing_position_norm",
    "boom_position_norm",
    "stick_position_norm",
    "bucket_position_norm",
)
OUTPUT_ORDER = (
    "bucket_tip_machine_root_x_m",
    "bucket_tip_machine_root_y_m",
    "bucket_tip_machine_root_z_m",
)
CELL_NAMES = ("D00", "D01", "D10", "D11", "D20", "D21")


@dataclass(frozen=True)
class Samples:
    qpos: np.ndarray
    tip_machine: np.ndarray
    tip_grid: np.ndarray
    split: np.ndarray
    trajectory: np.ndarray
    half_extents: np.ndarray
    rows: list[dict[str, Any]]


class FKRegressor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(4, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 3),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class NormalizedMLPFK(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        input_mean: np.ndarray,
        input_std: np.ndarray,
        output_mean: np.ndarray,
        output_std: np.ndarray,
    ) -> None:
        super().__init__()
        self.model = model
        self.register_buffer(
            "input_mean", torch.as_tensor(input_mean, dtype=torch.float32)
        )
        self.register_buffer(
            "input_std", torch.as_tensor(input_std, dtype=torch.float32)
        )
        self.register_buffer(
            "output_mean", torch.as_tensor(output_mean, dtype=torch.float32)
        )
        self.register_buffer(
            "output_std", torch.as_tensor(output_std, dtype=torch.float32)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        normalized = (inputs - self.input_mean) / self.input_std
        return self.model(normalized) * self.output_std + self.output_mean


class PolynomialFK(nn.Module):
    def __init__(
        self,
        powers: np.ndarray,
        coefficients: np.ndarray,
        input_mean: np.ndarray,
        input_std: np.ndarray,
        output_mean: np.ndarray,
        output_std: np.ndarray,
    ) -> None:
        super().__init__()
        self.register_buffer(
            "powers", torch.as_tensor(powers, dtype=torch.float32)
        )
        self.register_buffer(
            "coefficients", torch.as_tensor(coefficients, dtype=torch.float32)
        )
        self.register_buffer(
            "input_mean", torch.as_tensor(input_mean, dtype=torch.float32)
        )
        self.register_buffer(
            "input_std", torch.as_tensor(input_std, dtype=torch.float32)
        )
        self.register_buffer(
            "output_mean", torch.as_tensor(output_mean, dtype=torch.float32)
        )
        self.register_buffer(
            "output_std", torch.as_tensor(output_std, dtype=torch.float32)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        normalized = (inputs - self.input_mean) / self.input_std
        features = torch.prod(
            torch.pow(normalized.unsqueeze(-2), self.powers), dim=-1
        )
        normalized_output = torch.matmul(features, self.coefficients)
        return normalized_output * self.output_std + self.output_mean


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_samples(path: Path) -> Samples:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"No samples found in {path}")

    for index, row in enumerate(rows):
        if row.get("schema_version") != EXPECTED_SCHEMA:
            raise ValueError(f"row {index}: unexpected schema")
        probe = row.get("probe", {})
        if probe.get("candidate_id") != EXPECTED_CANDIDATE:
            raise ValueError(f"row {index}: fixed-tip candidate changed")
        if not probe.get("success", False):
            raise ValueError(f"row {index}: probe is not successful")
        echoed = np.asarray(probe.get("observed_qpos"), dtype=np.float64)
        observed = np.asarray(row.get("observed_qpos"), dtype=np.float64)
        if echoed.shape != (4,) or observed.shape != (4,):
            raise ValueError(f"row {index}: qpos must have four values")
        if not np.allclose(echoed, observed, atol=1.0e-7, rtol=0.0):
            raise ValueError(f"row {index}: probe qpos does not match response")

    qpos = np.asarray([row["observed_qpos"] for row in rows], dtype=np.float32)
    tip_machine = np.asarray(
        [row["probe"]["bucket_tip_machine_root_local_m"] for row in rows],
        dtype=np.float32,
    )
    tip_grid = np.asarray(
        [row["probe"]["bucket_tip_dig_area_local_m"] for row in rows],
        dtype=np.float32,
    )
    split = np.asarray([row["split"] for row in rows], dtype=object)
    trajectory = np.asarray(
        [int(row["trajectory_index"]) for row in rows], dtype=np.int64
    )
    half_extents_all = np.asarray(
        [row["probe"]["dig_area_half_extents_m"] for row in rows],
        dtype=np.float32,
    )
    if not np.isfinite(qpos).all() or not np.isfinite(tip_machine).all():
        raise ValueError("Samples contain non-finite qpos or tip coordinates")
    if not np.allclose(half_extents_all, half_extents_all[0], atol=1.0e-7):
        raise ValueError("Dig-area half extents changed during collection")
    return Samples(
        qpos=qpos,
        tip_machine=tip_machine,
        tip_grid=tip_grid,
        split=split,
        trajectory=trajectory,
        half_extents=half_extents_all[0],
        rows=rows,
    )


def normalization(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = values.std(axis=0, dtype=np.float64).astype(np.float32)
    return mean, np.maximum(std, 1.0e-6).astype(np.float32)


def polynomial_powers(input_dim: int, degree: int) -> np.ndarray:
    powers: list[list[int]] = []
    for total_degree in range(degree + 1):
        for indices in itertools.combinations_with_replacement(
            range(input_dim), total_degree
        ):
            row = [0] * input_dim
            for index in indices:
                row[index] += 1
            powers.append(row)
    return np.asarray(powers, dtype=np.int64)


def polynomial_features(inputs: np.ndarray, powers: np.ndarray) -> np.ndarray:
    return np.prod(
        np.power(inputs[:, None, :], powers[None, :, :]), axis=2
    ).astype(np.float64)


def fit_polynomial(
    train_x: np.ndarray,
    train_y: np.ndarray,
    *,
    degree: int,
    ridge: float,
) -> PolynomialFK:
    input_mean, input_std = normalization(train_x)
    output_mean, output_std = normalization(train_y)
    normalized_x = (train_x - input_mean) / input_std
    normalized_y = (train_y - output_mean) / output_std
    powers = polynomial_powers(train_x.shape[1], degree)
    features = polynomial_features(normalized_x, powers)
    gram = features.T @ features
    regularizer = np.eye(gram.shape[0], dtype=np.float64) * ridge
    regularizer[0, 0] = 0.0
    coefficients = np.linalg.solve(
        gram + regularizer, features.T @ normalized_y.astype(np.float64)
    )
    return PolynomialFK(
        powers,
        coefficients.astype(np.float32),
        input_mean,
        input_std,
        output_mean,
        output_std,
    )


def predict(model: nn.Module, inputs: np.ndarray) -> np.ndarray:
    model = model.cpu().eval()
    with torch.inference_mode():
        return (
            model(torch.as_tensor(inputs, dtype=torch.float32))
            .cpu()
            .numpy()
            .astype(np.float32)
        )


def error_summary(truth: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    axis_abs = np.abs(prediction - truth)
    distance = np.linalg.norm(prediction - truth, axis=1)
    return {
        "count": len(truth),
        "axis_mae_m": axis_abs.mean(axis=0),
        "axis_p95_abs_m": np.quantile(axis_abs, 0.95, axis=0),
        "distance_mean_m": float(distance.mean()),
        "distance_rmse_m": float(np.sqrt(np.mean(distance**2))),
        "distance_p50_m": float(np.quantile(distance, 0.50)),
        "distance_p90_m": float(np.quantile(distance, 0.90)),
        "distance_p95_m": float(np.quantile(distance, 0.95)),
        "distance_p99_m": float(np.quantile(distance, 0.99)),
        "distance_max_m": float(distance.max()),
    }


def train_mlp_candidate(
    train_x: np.ndarray,
    train_y: np.ndarray,
    tune_x: np.ndarray,
    tune_y: np.ndarray,
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: str,
    evaluation_interval: int,
    patience_evaluations: int,
) -> tuple[NormalizedMLPFK, int, dict[str, Any]]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    input_mean, input_std = normalization(train_x)
    output_mean, output_std = normalization(train_y)
    normalized_x = (train_x - input_mean) / input_std
    normalized_y = (train_y - output_mean) / output_std
    model = FKRegressor().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1.0e-6
    )
    loss_fn = nn.MSELoss()
    x_tensor = torch.as_tensor(normalized_x, dtype=torch.float32)
    y_tensor = torch.as_tensor(normalized_y, dtype=torch.float32)
    generator = torch.Generator().manual_seed(seed)

    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_tune_p95 = float("inf")
    no_improvement = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        permutation = torch.randperm(len(x_tensor), generator=generator)
        model.train()
        epoch_loss = 0.0
        for start in range(0, len(permutation), batch_size):
            indices = permutation[start : start + batch_size]
            batch_x = x_tensor[indices].to(device)
            batch_y = y_tensor[indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu()) * len(indices)

        if epoch % evaluation_interval != 0 and epoch != epochs:
            continue
        wrapper = NormalizedMLPFK(
            copy.deepcopy(model).cpu(),
            input_mean,
            input_std,
            output_mean,
            output_std,
        )
        tune_prediction = predict(wrapper, tune_x)
        tune_metrics = error_summary(tune_y, tune_prediction)
        tune_p95 = float(tune_metrics["distance_p95_m"])
        history.append(
            {
                "epoch": epoch,
                "train_normalized_mse": epoch_loss / len(permutation),
                "tune_distance_p95_m": tune_p95,
                "tune_distance_rmse_m": float(
                    tune_metrics["distance_rmse_m"]
                ),
            }
        )
        if tune_p95 < best_tune_p95:
            best_tune_p95 = tune_p95
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement >= patience_evaluations:
                break

    if best_state is None:
        raise RuntimeError("MLP training did not produce a checkpoint")
    selected = FKRegressor()
    selected.load_state_dict(best_state)
    wrapper = NormalizedMLPFK(
        selected, input_mean, input_std, output_mean, output_std
    )
    return wrapper, best_epoch, {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_tune_distance_p95_m": best_tune_p95,
        "history": history,
    }


def train_mlp_fixed_epochs(
    train_x: np.ndarray,
    train_y: np.ndarray,
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: str,
) -> NormalizedMLPFK:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    input_mean, input_std = normalization(train_x)
    output_mean, output_std = normalization(train_y)
    x_tensor = torch.as_tensor(
        (train_x - input_mean) / input_std, dtype=torch.float32
    )
    y_tensor = torch.as_tensor(
        (train_y - output_mean) / output_std, dtype=torch.float32
    )
    model = FKRegressor().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1.0e-6
    )
    generator = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        permutation = torch.randperm(len(x_tensor), generator=generator)
        model.train()
        for start in range(0, len(permutation), batch_size):
            indices = permutation[start : start + batch_size]
            batch_x = x_tensor[indices].to(device)
            batch_y = y_tensor[indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
    return NormalizedMLPFK(
        model.cpu(), input_mean, input_std, output_mean, output_std
    )


def affine_machine_to_grid(
    machine_points: np.ndarray, grid_points: np.ndarray
) -> np.ndarray:
    design = np.concatenate(
        (machine_points.astype(np.float64), np.ones((len(machine_points), 1))),
        axis=1,
    )
    coefficients, _, _, _ = np.linalg.lstsq(
        design, grid_points.astype(np.float64), rcond=None
    )
    return coefficients.astype(np.float32)


def apply_affine(points: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    design = np.concatenate(
        (points, np.ones((len(points), 1), dtype=np.float32)), axis=1
    )
    return design @ coefficients


def grid_coordinates(
    grid_tip: np.ndarray, half_extents: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    long_axis = 0 if half_extents[0] >= half_extents[2] else 2
    short_axis = 2 if long_axis == 0 else 0
    long_norm = grid_tip[:, long_axis] / half_extents[long_axis]
    short_norm = grid_tip[:, short_axis] / half_extents[short_axis]
    return long_norm, short_norm


def grid_cells(
    grid_tip: np.ndarray, half_extents: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    long_norm, short_norm = grid_coordinates(grid_tip, half_extents)
    in_bounds = (
        (long_norm >= -1.0)
        & (long_norm <= 1.0)
        & (short_norm >= -1.0)
        & (short_norm <= 1.0)
    )
    long_index = np.clip(
        np.floor((long_norm + 1.0) * 1.5).astype(np.int64), 0, 2
    )
    short_index = np.clip(
        np.floor(short_norm + 1.0).astype(np.int64), 0, 1
    )
    cells = long_index * 2 + short_index
    cells[~in_bounds] = -1
    internal_boundary_distance = np.minimum.reduce(
        (
            np.abs(long_norm + 1.0 / 3.0),
            np.abs(long_norm - 1.0 / 3.0),
            np.abs(short_norm),
        )
    )
    return cells, in_bounds, internal_boundary_distance


def grid_metrics(
    truth_grid: np.ndarray,
    prediction_grid: np.ndarray,
    half_extents: np.ndarray,
) -> dict[str, Any]:
    truth_cells, truth_in_bounds, boundary_distance = grid_cells(
        truth_grid, half_extents
    )
    prediction_cells, prediction_in_bounds, _ = grid_cells(
        prediction_grid, half_extents
    )
    result: dict[str, Any] = {
        "count": len(truth_grid),
        "true_in_bounds_count": int(truth_in_bounds.sum()),
        "true_in_bounds_fraction": float(truth_in_bounds.mean()),
        "in_bounds_agreement": float(
            np.mean(truth_in_bounds == prediction_in_bounds)
        ),
    }
    if truth_in_bounds.any():
        result["cell_accuracy_true_in_bounds"] = float(
            np.mean(
                prediction_cells[truth_in_bounds]
                == truth_cells[truth_in_bounds]
            )
        )
        for margin in (0.02, 0.05, 0.10):
            safe = truth_in_bounds & (boundary_distance >= margin)
            result[f"cell_accuracy_boundary_margin_{margin:.2f}"] = (
                float(np.mean(prediction_cells[safe] == truth_cells[safe]))
                if safe.any()
                else None
            )
            result[f"boundary_margin_{margin:.2f}_count"] = int(safe.sum())
        confusion = np.zeros((7, 7), dtype=np.int64)
        truth_bucket = np.where(truth_cells >= 0, truth_cells, 6)
        prediction_bucket = np.where(prediction_cells >= 0, prediction_cells, 6)
        for truth_cell, prediction_cell in zip(
            truth_bucket, prediction_bucket, strict=True
        ):
            confusion[truth_cell, prediction_cell] += 1
        result["confusion_rows_truth_cols_prediction_D00_to_D21_outside"] = (
            confusion
        )
    return result


def grid_cell_counts(
    grid_tip: np.ndarray, half_extents: np.ndarray
) -> dict[str, int]:
    cells, _, _ = grid_cells(grid_tip, half_extents)
    result = {"outside": int(np.sum(cells < 0))}
    result.update(
        {
            name: int(np.sum(cells == index))
            for index, name in enumerate(CELL_NAMES)
        }
    )
    return result


def nearest_neighbor_baseline(
    train_x: np.ndarray,
    train_y: np.ndarray,
    query_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean, std = normalization(train_x)
    train_normalized = (train_x - mean) / std
    query_normalized = (query_x - mean) / std
    predictions: list[np.ndarray] = []
    distances: list[float] = []
    chunk = 256
    for start in range(0, len(query_x), chunk):
        query = query_normalized[start : start + chunk]
        distance_sq = np.sum(
            (query[:, None, :] - train_normalized[None, :, :]) ** 2, axis=2
        )
        nearest = np.argmin(distance_sq, axis=1)
        predictions.append(train_y[nearest])
        distances.extend(np.sqrt(distance_sq[np.arange(len(query)), nearest]))
    return np.concatenate(predictions), np.asarray(distances, dtype=np.float32)


def historical_domain_coverage(
    dataset: Path,
    qpos_min: np.ndarray,
    qpos_max: np.ndarray,
) -> dict[str, Any]:
    values: list[np.ndarray] = []
    for path in sorted(dataset.glob("episode_*.hdf5")):
        with h5py.File(path, "r") as handle:
            metadata = handle.get("metadata")
            if metadata is None or "env_state_order" not in metadata.attrs:
                continue
            order = [
                part.strip()
                for part in str(metadata.attrs["env_state_order"]).split(",")
                if part.strip()
            ]
            if "bucket_dig_area_cell_in_bounds_mask" not in order:
                continue
            index = order.index("bucket_dig_area_cell_in_bounds_mask")
            mask = np.asarray(handle["observations/env_state"][:, index]) > 0.5
            qpos = np.asarray(handle["observations/qpos"], dtype=np.float32)
            values.append(qpos[mask & np.isfinite(qpos).all(axis=1)])
    if not values:
        return {"available": False}
    qpos = np.concatenate(values)
    per_axis_inside = (qpos >= qpos_min) & (qpos <= qpos_max)
    return {
        "available": True,
        "row_count": len(qpos),
        "per_axis_inside_fraction": per_axis_inside.mean(axis=0),
        "all_axes_inside_fraction": float(per_axis_inside.all(axis=1).mean()),
        "qpos_quantile_01": np.quantile(qpos, 0.01, axis=0),
        "qpos_quantile_50": np.quantile(qpos, 0.50, axis=0),
        "qpos_quantile_99": np.quantile(qpos, 0.99, axis=0),
    }


def save_plots(
    output_dir: Path,
    truth_machine: np.ndarray,
    prediction_machine: np.ndarray,
    truth_grid: np.ndarray,
    prediction_grid: np.ndarray,
    half_extents: np.ndarray,
    model_name: str,
) -> None:
    errors = np.linalg.norm(prediction_machine - truth_machine, axis=1)
    figure, axes = plt.subplots(2, 2, figsize=(11, 9))
    labels = ("machine x", "machine y", "machine z")
    for axis, label in enumerate(labels):
        panel = axes.flat[axis]
        panel.scatter(
            truth_machine[:, axis],
            prediction_machine[:, axis],
            s=8,
            alpha=0.45,
        )
        low = min(truth_machine[:, axis].min(), prediction_machine[:, axis].min())
        high = max(
            truth_machine[:, axis].max(), prediction_machine[:, axis].max()
        )
        panel.plot([low, high], [low, high], "k--", linewidth=1)
        panel.set_xlabel(f"Unity truth {label} (m)")
        panel.set_ylabel(f"FK prediction {label} (m)")
        panel.grid(alpha=0.2)
    ordered = np.sort(errors)
    axes.flat[3].plot(ordered, np.linspace(0.0, 1.0, len(ordered)))
    axes.flat[3].axvline(np.quantile(errors, 0.95), color="tab:red", linestyle="--")
    axes.flat[3].set_xlabel("3D fixed-tip error (m)")
    axes.flat[3].set_ylabel("Empirical CDF")
    axes.flat[3].grid(alpha=0.2)
    figure.suptitle(f"Held-out fixed-tip FK validation: {model_name}")
    figure.tight_layout()
    figure.savefig(output_dir / "fixed_tip_fk_validation.png", dpi=180)
    plt.close(figure)

    long_axis = 0 if half_extents[0] >= half_extents[2] else 2
    short_axis = 2 if long_axis == 0 else 0
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(
        truth_grid[:, short_axis],
        truth_grid[:, long_axis],
        s=8,
        alpha=0.5,
        label="Unity truth",
    )
    axes[1].scatter(
        prediction_grid[:, short_axis],
        prediction_grid[:, long_axis],
        s=8,
        alpha=0.5,
        color="tab:orange",
        label="FK prediction",
    )
    short_half = half_extents[short_axis]
    long_half = half_extents[long_axis]
    for panel in axes:
        for value in (-short_half, 0.0, short_half):
            panel.axvline(value, color="k", linewidth=0.8, alpha=0.5)
        for value in np.linspace(-long_half, long_half, 4):
            panel.axhline(value, color="k", linewidth=0.8, alpha=0.5)
        panel.set_xlim(-short_half * 1.25, short_half * 1.25)
        panel.set_ylim(-long_half * 1.25, long_half * 1.25)
        panel.set_xlabel("grid short axis (m)")
        panel.set_ylabel("grid long axis (m)")
        panel.legend()
        panel.grid(alpha=0.15)
    axes[0].set_title("Held-out Unity fixed point")
    axes[1].set_title("FK-projected fixed point")
    figure.tight_layout()
    figure.savefig(output_dir / "fixed_tip_grid_validation.png", dpi=180)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1.5e-3)
    parser.add_argument("--mlp-seeds", default="20260723,20260724,20260725")
    parser.add_argument("--evaluation-interval", type=int, default=5)
    parser.add_argument("--patience-evaluations", type=int, default=80)
    parser.add_argument("--polynomial-degrees", default="2,3,4,5,6")
    parser.add_argument("--ridge-values", default="1e-8,1e-6,1e-4,1e-2")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=("cpu", "cuda"),
    )
    return parser.parse_args()


def _parse_ints(text: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in text.split(",") if part.strip())


def _parse_floats(text: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in text.split(",") if part.strip())


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = load_samples(args.input.resolve())

    train_mask = samples.split == "train"
    validation_mask = samples.split == "validation"
    if not train_mask.any() or not validation_mask.any():
        raise ValueError("Both train and validation samples are required")
    tune_trajectory = int(samples.trajectory[train_mask].max())
    tune_mask = train_mask & (samples.trajectory == tune_trajectory)
    core_mask = train_mask & ~tune_mask
    if not core_mask.any() or not tune_mask.any():
        raise ValueError("At least two training trajectories are required")
    validation_trajectories = sorted(
        int(value) for value in np.unique(samples.trajectory[validation_mask])
    )
    if len(validation_trajectories) < 2:
        raise ValueError(
            "At least two validation trajectories are required: one for "
            "model-class selection and one untouched final test"
        )
    class_selection_trajectory = validation_trajectories[0]
    class_selection_mask = validation_mask & (
        samples.trajectory == class_selection_trajectory
    )
    final_test_mask = validation_mask & (
        samples.trajectory != class_selection_trajectory
    )

    core_x, core_y = samples.qpos[core_mask], samples.tip_machine[core_mask]
    tune_x, tune_y = samples.qpos[tune_mask], samples.tip_machine[tune_mask]
    all_train_x = samples.qpos[train_mask]
    all_train_y = samples.tip_machine[train_mask]
    class_selection_x = samples.qpos[class_selection_mask]
    class_selection_y = samples.tip_machine[class_selection_mask]
    final_test_x = samples.qpos[final_test_mask]
    final_test_y = samples.tip_machine[final_test_mask]

    polynomial_tuning: list[dict[str, Any]] = []
    best_polynomial_spec: tuple[int, float] | None = None
    best_polynomial_p95 = float("inf")
    for degree in _parse_ints(args.polynomial_degrees):
        for ridge in _parse_floats(args.ridge_values):
            candidate = fit_polynomial(
                core_x, core_y, degree=degree, ridge=ridge
            )
            candidate_metrics = error_summary(
                tune_y, predict(candidate, tune_x)
            )
            polynomial_tuning.append(
                {
                    "degree": degree,
                    "ridge": ridge,
                    "tune": candidate_metrics,
                }
            )
            p95 = float(candidate_metrics["distance_p95_m"])
            if p95 < best_polynomial_p95:
                best_polynomial_p95 = p95
                best_polynomial_spec = (degree, ridge)
    assert best_polynomial_spec is not None

    mlp_tuning: list[dict[str, Any]] = []
    best_mlp: NormalizedMLPFK | None = None
    best_mlp_epoch = 0
    best_mlp_seed = 0
    best_mlp_p95 = float("inf")
    for seed in _parse_ints(args.mlp_seeds):
        candidate, best_epoch, tuning = train_mlp_candidate(
            core_x,
            core_y,
            tune_x,
            tune_y,
            seed=seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=args.device,
            evaluation_interval=args.evaluation_interval,
            patience_evaluations=args.patience_evaluations,
        )
        mlp_tuning.append(tuning)
        p95 = float(tuning["best_tune_distance_p95_m"])
        if p95 < best_mlp_p95:
            best_mlp_p95 = p95
            best_mlp = candidate
            best_mlp_epoch = best_epoch
            best_mlp_seed = seed
    assert best_mlp is not None

    polynomial_model = fit_polynomial(
        all_train_x,
        all_train_y,
        degree=best_polynomial_spec[0],
        ridge=best_polynomial_spec[1],
    )
    mlp_model = train_mlp_fixed_epochs(
        all_train_x,
        all_train_y,
        seed=best_mlp_seed,
        epochs=best_mlp_epoch,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
    )
    models: dict[str, nn.Module] = {
        "polynomial": polynomial_model,
        "mlp": mlp_model,
    }
    class_selection_metrics = {
        name: error_summary(
            class_selection_y, predict(model, class_selection_x)
        )
        for name, model in models.items()
    }
    selected_model_type = min(
        class_selection_metrics,
        key=lambda name: class_selection_metrics[name]["distance_p95_m"],
    )
    selected_model = models[selected_model_type]

    machine_to_grid = affine_machine_to_grid(
        samples.tip_machine[train_mask], samples.tip_grid[train_mask]
    )
    transform_residual = np.linalg.norm(
        apply_affine(samples.tip_machine, machine_to_grid) - samples.tip_grid,
        axis=1,
    )
    if float(transform_residual.max()) > 1.0e-4:
        raise RuntimeError(
            "Machine-to-grid transform is not rigid/static enough: "
            f"max residual={transform_residual.max():.6g}m"
        )

    metrics: dict[str, Any] = {
        "sample_counts": {
            "core_train": int(core_mask.sum()),
            "tuning_trajectory": int(tune_mask.sum()),
            "all_train": int(train_mask.sum()),
            "model_class_selection": int(class_selection_mask.sum()),
            "untouched_final_test": int(final_test_mask.sum()),
        },
        "split_contract": {
            "tuning_trajectory_index": tune_trajectory,
            "model_class_selection_split": "validation",
            "model_class_selection_trajectory_index": (
                class_selection_trajectory
            ),
            "untouched_final_test_trajectory_indices": (
                validation_trajectories[1:]
            ),
            "hyperparameter_selection_metric": "tuning distance p95",
            "model_class_selection_metric": (
                "model-class-selection trajectory distance p95"
            ),
        },
        "polynomial_tuning": polynomial_tuning,
        "mlp_tuning": mlp_tuning,
        "model_class_selection": class_selection_metrics,
        "selected_model_type": selected_model_type,
        "selected_polynomial_degree": best_polynomial_spec[0],
        "selected_polynomial_ridge": best_polynomial_spec[1],
        "selected_mlp_seed": best_mlp_seed,
        "selected_mlp_epoch": best_mlp_epoch,
        "machine_to_grid_affine": machine_to_grid,
        "machine_to_grid_transform_residual_m": {
            "p95": float(np.quantile(transform_residual, 0.95)),
            "max": float(transform_residual.max()),
        },
        "observed_grid_cell_coverage": {
            "all_samples": grid_cell_counts(
                samples.tip_grid, samples.half_extents
            ),
            "all_train": grid_cell_counts(
                samples.tip_grid[train_mask], samples.half_extents
            ),
            "model_class_selection": grid_cell_counts(
                samples.tip_grid[class_selection_mask], samples.half_extents
            ),
            "untouched_final_test": grid_cell_counts(
                samples.tip_grid[final_test_mask], samples.half_extents
            ),
        },
        "models": {},
    }

    predictions: dict[str, np.ndarray] = {}
    for name, model in models.items():
        prediction_machine = predict(model, final_test_x)
        prediction_grid = apply_affine(prediction_machine, machine_to_grid)
        predictions[name] = prediction_machine
        metrics["models"][name] = {
            "untouched_final_test_machine_tip": error_summary(
                final_test_y, prediction_machine
            ),
            "untouched_final_test_grid_tip": error_summary(
                samples.tip_grid[final_test_mask], prediction_grid
            ),
            "untouched_final_test_grid_cells": grid_metrics(
                samples.tip_grid[final_test_mask],
                prediction_grid,
                samples.half_extents,
            ),
        }

    nearest_prediction, nearest_distance = nearest_neighbor_baseline(
        all_train_x, all_train_y, final_test_x
    )
    metrics["nearest_neighbor_reference"] = {
        "untouched_final_test_machine_tip": error_summary(
            final_test_y, nearest_prediction
        ),
        "normalized_qpos_neighbor_distance": {
            "p50": float(np.quantile(nearest_distance, 0.50)),
            "p95": float(np.quantile(nearest_distance, 0.95)),
            "max": float(nearest_distance.max()),
        },
    }

    qpos_min = all_train_x.min(axis=0)
    qpos_max = all_train_x.max(axis=0)
    final_test_inside = (final_test_x >= qpos_min) & (final_test_x <= qpos_max)
    collection_manifest_path = (
        args.input.resolve().parent / "collection_manifest.json"
    )
    dataset = None
    if collection_manifest_path.is_file():
        collection_manifest = json.loads(
            collection_manifest_path.read_text(encoding="utf-8")
        )
        dataset_value = collection_manifest.get("dataset")
        if dataset_value:
            dataset = Path(dataset_value)
    metrics["domain"] = {
        "train_qpos_min": qpos_min,
        "train_qpos_max": qpos_max,
        "train_qpos_quantile_01": np.quantile(all_train_x, 0.01, axis=0),
        "train_qpos_quantile_99": np.quantile(all_train_x, 0.99, axis=0),
        "final_test_per_axis_inside_train_range": final_test_inside.mean(axis=0),
        "final_test_all_axes_inside_train_range": float(
            final_test_inside.all(axis=1).mean()
        ),
        "historical_in_grid_proxy_qpos_coverage": (
            historical_domain_coverage(dataset, qpos_min, qpos_max)
            if dataset is not None and dataset.is_dir()
            else {"available": False}
        ),
    }

    for name, model in models.items():
        scripted = torch.jit.trace(
            model.cpu().eval(), torch.zeros((1, 4), dtype=torch.float32)
        )
        scripted.save(str(output_dir / f"fixed_tip_fk_{name}.ts"))
    selected_path = output_dir / "fixed_tip_fk_selected.ts"
    torch.jit.trace(
        selected_model.cpu().eval(), torch.zeros((1, 4), dtype=torch.float32)
    ).save(str(selected_path))

    selected_prediction = predictions[selected_model_type]
    selected_prediction_grid = apply_affine(
        selected_prediction, machine_to_grid
    )
    np.savez_compressed(
        output_dir / "final_test_predictions.npz",
        qpos=final_test_x,
        truth_machine_tip=final_test_y,
        prediction_machine_tip=selected_prediction,
        truth_grid_tip=samples.tip_grid[final_test_mask],
        prediction_grid_tip=selected_prediction_grid,
    )
    save_plots(
        output_dir,
        final_test_y,
        selected_prediction,
        samples.tip_grid[final_test_mask],
        selected_prediction_grid,
        samples.half_extents,
        selected_model_type,
    )

    manifest = {
        "schema_version": "fixed_tip_fk_model/v1",
        "selected_model_type": selected_model_type,
        "candidate_id": EXPECTED_CANDIDATE,
        "bucket_tip_bucket_local_m": samples.rows[0]["probe"][
            "bucket_tip_local_m"
        ],
        "input_order": INPUT_ORDER,
        "output_order": OUTPUT_ORDER,
        "input_semantics": "normalized qpos observed by the YuLong runtime",
        "output_semantics": (
            "fixed center-tooth leading-edge point in machine-root local meters"
        ),
        "machine_to_grid_affine": machine_to_grid,
        "dig_area_half_extents_m": samples.half_extents,
        "dig_area_grid_shape_long_by_short": [3, 2],
        "dig_area_cell_names": CELL_NAMES,
        "domain_qpos_min": qpos_min,
        "domain_qpos_max": qpos_max,
        "out_of_domain_policy": "reject_do_not_extrapolate",
        "real_qpos_radian_conversion": (
            "not_calibrated_do_not_assume_Unity_profile"
        ),
        "final_test_is_whole_trajectory_held_out": True,
        "final_test_was_not_used_for_hyperparameter_or_model_class_selection": True,
        "selected_model_path": str(selected_path),
        "selected_model_sha256": sha256(selected_path),
        "source_samples": str(args.input.resolve()),
        "source_samples_sha256": sha256(args.input.resolve()),
    }
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "model_manifest.json", manifest)
    print(
        json.dumps(
            jsonable({
                "selected_model_type": selected_model_type,
                "untouched_final_test": metrics["models"][selected_model_type][
                    "untouched_final_test_machine_tip"
                ],
                "grid_cells": metrics["models"][selected_model_type][
                    "untouched_final_test_grid_cells"
                ],
                "output": str(output_dir),
            }),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
