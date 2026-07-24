#!/usr/bin/env python3
"""Audit whether a simulator FK model may be applied to real qpos records.

This script deliberately stops at compatibility evidence.  It does not emit
real-machine grid labels because qpos sensor alignment and the real
machine-to-grid extrinsic must be calibrated first.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import h5py
import numpy as np


DEFAULT_REAL_DATASET = Path(
    "/data/pingfan/Excavator_real_stack_data/"
    "pro_real_teleop_20260713_20hz_v1"
)
DEFAULT_TRAIN_READY_MANIFEST = DEFAULT_REAL_DATASET / "qc_full/train_ready_manifest.json"
DEFAULT_MODEL_MANIFEST = Path(
    "docs/evidence/sim_fixed_tip_fk_v0_2/model_manifest.json"
)
DEFAULT_UNITY_NORMALIZATION = Path(
    "/home/pingfan/AGXUnityE85ExcavatorSim/"
    "Assets/AGXUnity_Excavator/AGXUnity_Excavator_Assets/"
    "Calibration/YuLong_norm.json"
)
DEFAULT_OUTPUT = Path(
    "docs/evidence/sim_fixed_tip_fk_v0_2/real_transfer_audit.json"
)
AXES = ("swing", "boom", "stick", "bucket")


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


def unity_normalize(
    raw_qpos_rad: np.ndarray,
    minimum_rad: np.ndarray,
    maximum_rad: np.ndarray,
) -> np.ndarray:
    raw = np.asarray(raw_qpos_rad, dtype=np.float64)
    minimum = np.asarray(minimum_rad, dtype=np.float64).reshape(4)
    maximum = np.asarray(maximum_rad, dtype=np.float64).reshape(4)
    scale = maximum - minimum
    if np.any(scale <= 0.0):
        raise ValueError("Unity normalization ranges must be increasing")
    return (raw - minimum) / scale


def load_real_qpos(
    dataset: Path, train_ready_manifest: Path
) -> tuple[np.ndarray, dict[str, Any]]:
    manifest = json.loads(train_ready_manifest.read_text(encoding="utf-8"))
    episode_names = manifest.get("train_ready_episode_ids", [])
    if not episode_names:
        raise ValueError(f"No train_ready_episode_ids in {train_ready_manifest}")

    values: list[np.ndarray] = []
    episode_rows: dict[str, int] = {}
    excluded_rows = 0
    for episode_name in episode_names:
        path = dataset / f"{episode_name}.hdf5"
        if not path.is_file():
            raise FileNotFoundError(path)
        with h5py.File(path, "r") as handle:
            qpos = np.asarray(handle["observations/qpos"], dtype=np.float32)
            metadata = handle.get("metadata")
            if metadata is None:
                raise ValueError(f"{path}: metadata group is required")
            order = str(metadata.attrs.get("qpos_order", ""))
            units = str(metadata.attrs.get("qpos_units", ""))
            if order != "swing,boom,stick,bucket":
                raise ValueError(f"{path}: unexpected qpos_order={order!r}")
            if units != "rad":
                raise ValueError(f"{path}: unexpected qpos_units={units!r}")
            valid = np.isfinite(qpos).all(axis=1)
            if "diagnostics/train_exclude_mask" in handle:
                excluded = (
                    np.asarray(handle["diagnostics/train_exclude_mask"]) > 0
                )
                excluded_rows += int(excluded.sum())
                valid &= ~excluded
        episode_rows[episode_name] = int(valid.sum())
        values.append(qpos[valid])

    return np.concatenate(values), {
        "episode_count": len(episode_names),
        "episode_valid_rows": episode_rows,
        "train_excluded_rows": excluded_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-dataset", type=Path, default=DEFAULT_REAL_DATASET)
    parser.add_argument(
        "--train-ready-manifest",
        type=Path,
        default=DEFAULT_TRAIN_READY_MANIFEST,
    )
    parser.add_argument(
        "--model-manifest", type=Path, default=DEFAULT_MODEL_MANIFEST
    )
    parser.add_argument(
        "--unity-normalization",
        type=Path,
        default=DEFAULT_UNITY_NORMALIZATION,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    real_qpos, source = load_real_qpos(
        args.real_dataset.resolve(), args.train_ready_manifest.resolve()
    )
    profile = json.loads(
        args.unity_normalization.resolve().read_text(encoding="utf-8")
    )
    minimum = np.asarray([profile[axis]["min"] for axis in AXES])
    maximum = np.asarray([profile[axis]["max"] for axis in AXES])
    normalized = unity_normalize(real_qpos, minimum, maximum)

    model_manifest = json.loads(
        args.model_manifest.resolve().read_text(encoding="utf-8")
    )
    domain_min = np.asarray(model_manifest["domain_qpos_min"], dtype=np.float64)
    domain_max = np.asarray(model_manifest["domain_qpos_max"], dtype=np.float64)
    inside_unit = (normalized >= 0.0) & (normalized <= 1.0)
    inside_model = (normalized >= domain_min) & (normalized <= domain_max)

    # This is only a domain proxy.  Swing in the calibrated source-facing band
    # does not prove that the bucket is digging or inside a particular cell.
    source_facing_swing_proxy = inside_model[:, 0]
    source_proxy_inside = inside_model[source_facing_swing_proxy]

    report = {
        "schema_version": "fixed_tip_fk_real_transfer_audit/v1",
        "verdict": "blocked_pending_real_sensor_and_grid_calibration",
        "reason": (
            "The simulator FK is accurate in its calibrated normalized-qpos "
            "domain, but the current Unity angle profile is not an established "
            "real-sensor conversion and no real machine-to-grid extrinsic is "
            "available."
        ),
        "real_dataset": str(args.real_dataset.resolve()),
        "train_ready_manifest": str(args.train_ready_manifest.resolve()),
        "model_manifest": str(args.model_manifest.resolve()),
        "unity_normalization_profile": str(
            args.unity_normalization.resolve()
        ),
        "source": source,
        "valid_row_count": len(real_qpos),
        "axis_order": AXES,
        "real_qpos_units": "rad",
        "unity_profile_min_rad": minimum,
        "unity_profile_max_rad": maximum,
        "real_qpos_quantiles_rad": {
            "p01": np.quantile(real_qpos, 0.01, axis=0),
            "p05": np.quantile(real_qpos, 0.05, axis=0),
            "p50": np.quantile(real_qpos, 0.50, axis=0),
            "p95": np.quantile(real_qpos, 0.95, axis=0),
            "p99": np.quantile(real_qpos, 0.99, axis=0),
        },
        "provisional_unity_normalized_quantiles": {
            "p01": np.quantile(normalized, 0.01, axis=0),
            "p05": np.quantile(normalized, 0.05, axis=0),
            "p50": np.quantile(normalized, 0.50, axis=0),
            "p95": np.quantile(normalized, 0.95, axis=0),
            "p99": np.quantile(normalized, 0.99, axis=0),
        },
        "provisional_unit_interval": {
            "per_axis_inside_fraction": inside_unit.mean(axis=0),
            "all_axes_inside_fraction": float(inside_unit.all(axis=1).mean()),
            "per_axis_outside_fraction": (~inside_unit).mean(axis=0),
        },
        "fk_calibration_domain": {
            "qpos_min": domain_min,
            "qpos_max": domain_max,
            "per_axis_inside_fraction": inside_model.mean(axis=0),
            "all_axes_inside_fraction": float(inside_model.all(axis=1).mean()),
        },
        "source_facing_swing_domain_proxy": {
            "definition": (
                "provisional normalized swing lies inside the simulator FK "
                "training range; this is not a dig-phase or grid-cell label"
            ),
            "row_count": int(source_facing_swing_proxy.sum()),
            "row_fraction": float(source_facing_swing_proxy.mean()),
            "other_axes_inside_fraction": (
                source_proxy_inside[:, 1:].mean(axis=0)
                if len(source_proxy_inside)
                else np.zeros(3)
            ),
            "all_axes_inside_fraction": (
                float(source_proxy_inside.all(axis=1).mean())
                if len(source_proxy_inside)
                else 0.0
            ),
        },
        "required_before_real_label_export": [
            "At least two matched real/simulator poses per axis, with additional "
            "interior checks, to fit and verify real-radian to simulator-qpos mapping.",
            "A measured machine-root to real dig-grid transform for the recording setup.",
            "Visual spot checks with visibility and boundary-confidence flags.",
            "Do not extrapolate the polynomial FK outside its model_manifest qpos domain.",
        ],
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(
        json.dumps(jsonable(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "verdict": report["verdict"],
                "valid_row_count": len(real_qpos),
                "unit_interval_outside_fraction": jsonable(
                    (~inside_unit).mean(axis=0)
                ),
                "fk_domain_all_axes_fraction": report[
                    "fk_calibration_domain"
                ]["all_axes_inside_fraction"],
                "source_swing_proxy_all_axes_fraction": report[
                    "source_facing_swing_domain_proxy"
                ]["all_axes_inside_fraction"],
                "output": str(args.output.resolve()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
