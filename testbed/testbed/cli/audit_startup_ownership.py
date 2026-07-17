"""Audit expert prestart ownership against saved open-loop policy actions."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml

from testbed.policies.action_start_distribution import (
    FORBIDDEN_HELDOUT,
    sha256_file,
)
from testbed.policies.deadzone_eval import load_deadzone_thresholds
from testbed.policies.startup_ownership import audit_startup_ownership


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m testbed.cli.audit_startup_ownership",
        description=(
            "Rescore saved open-loop actions under imitation-aligned and "
            "autonomy-aligned prestart ownership semantics."
        ),
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--deadzone-json", type=Path, required=True)
    parser.add_argument(
        "--policy-collection",
        action="append",
        required=True,
        metavar="MODEL=DIR",
    )
    parser.add_argument("--sample-hz", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    collections = dict(
        _parse_policy_collection(value) for value in args.policy_collection
    )
    if len(collections) != len(args.policy_collection):
        raise SystemExit("policy collection model labels must be unique")
    result = run_startup_ownership_audit(
        dataset_dir=args.dataset_dir,
        split_path=args.split,
        deadzone_json=args.deadzone_json,
        policy_collections=collections,
        sample_hz=float(args.sample_hz),
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "report": str(result["report"]),
                "report_sha256": result["report_sha256"],
                "source_manifest": str(result["source_manifest"]),
                "source_manifest_sha256": result["source_manifest_sha256"],
            },
            ensure_ascii=False,
        )
    )


def run_startup_ownership_audit(
    *,
    dataset_dir: str | Path,
    split_path: str | Path,
    deadzone_json: str | Path,
    policy_collections: Mapping[str, str | Path],
    sample_hz: float,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Load immutable sources, run post-hoc scoring, and write artifacts."""

    dataset = _required_directory(dataset_dir, "dataset_dir")
    split_file = _required_file(split_path, "split")
    deadzone_file = _required_file(deadzone_json, "deadzone_json")
    output = Path(output_dir).expanduser().resolve()
    split = _read_mapping_yaml(split_file)
    configured_dataset = split.get("dataset_dir")
    if configured_dataset is not None and Path(str(configured_dataset)).resolve() != (
        dataset
    ):
        raise ValueError(
            f"split dataset_dir does not match requested dataset: {configured_dataset}"
        )
    selected_numbers = list(
        dict.fromkeys(
            int(value)
            for value in [*split.get("train_ids", []), *split.get("val_ids", [])]
        )
    )
    if not selected_numbers:
        raise ValueError("split train_ids/val_ids must not be empty")
    forbidden = sorted(set(selected_numbers) & FORBIDDEN_HELDOUT)
    if forbidden:
        raise ValueError(f"held-out episodes are forbidden: {forbidden}")
    episode_ids = [f"episode_{value}" for value in sorted(selected_numbers)]

    expert_actions: dict[str, np.ndarray] = {}
    expert_sources: list[dict[str, Any]] = []
    for episode_id in episode_ids:
        path = _required_file(dataset / f"{episode_id}.hdf5", episode_id)
        with h5py.File(path, "r") as handle:
            if "action" not in handle:
                raise KeyError(f"{path} is missing action")
            expert_actions[episode_id] = np.asarray(
                handle["action"][()], dtype=np.float32
            )
        expert_sources.append(
            {
                "episode_id": episode_id,
                "path": str(path),
                "sha256": sha256_file(path),
            }
        )

    if not policy_collections:
        raise ValueError("policy_collections must not be empty")
    loaded_collections: dict[str, dict[str, np.ndarray]] = {}
    collection_sources: dict[str, Any] = {}
    for raw_model, raw_path in policy_collections.items():
        model = str(raw_model).strip()
        if not model or model in loaded_collections:
            raise ValueError(f"invalid or duplicate policy model label: {raw_model!r}")
        collection_dir = _required_directory(raw_path, f"policy collection {model}")
        summary_path = _required_file(
            collection_dir / "collection_summary.json",
            f"policy collection summary {model}",
        )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary_ids = [str(value) for value in summary.get("episode_ids", [])]
        if len(summary_ids) != len(set(summary_ids)) or set(summary_ids) != set(
            episode_ids
        ):
            raise ValueError(
                f"policy collection {model!r} episode ids do not exactly match split"
            )
        summary_dataset = summary.get("dataset_dir")
        if summary_dataset is not None and Path(str(summary_dataset)).resolve() != (
            dataset
        ):
            raise ValueError(
                f"policy collection {model!r} dataset_dir does not match dataset"
            )

        loaded_collections[model] = {}
        action_sources: list[dict[str, Any]] = []
        for episode_id in episode_ids:
            action_path = _required_file(
                collection_dir / "episodes" / episode_id / "actions.npz",
                f"{model}/{episode_id} actions",
            )
            with np.load(action_path) as data:
                missing = [
                    key
                    for key in ("time_s", "expert_action", "policy_action")
                    if key not in data
                ]
                if missing:
                    raise KeyError(
                        f"{action_path} is missing array(s): {', '.join(missing)}"
                    )
                time_s = np.asarray(data["time_s"], dtype=np.float64)
                stored_expert = np.asarray(data["expert_action"], dtype=np.float32)
                policy = np.asarray(data["policy_action"], dtype=np.float32)
            expected_expert = expert_actions[episode_id]
            if not np.array_equal(stored_expert, expected_expert):
                raise ValueError(
                    f"collection expert_action differs from HDF5 for {model}/{episode_id}"
                )
            if policy.shape != expected_expert.shape:
                raise ValueError(
                    f"collection policy_action shape differs for {model}/{episode_id}"
                )
            _validate_timebase(
                time_s,
                expected_steps=int(expected_expert.shape[0]),
                sample_hz=float(sample_hz),
                source=f"{model}/{episode_id}",
            )
            loaded_collections[model][episode_id] = policy
            action_sources.append(
                {
                    "episode_id": episode_id,
                    "path": str(action_path),
                    "sha256": sha256_file(action_path),
                }
            )
        collection_sources[model] = {
            "directory": str(collection_dir),
            "collection_summary": str(summary_path),
            "collection_summary_sha256": sha256_file(summary_path),
            "actions": action_sources,
        }

    thresholds = load_deadzone_thresholds(deadzone_file)
    report = audit_startup_ownership(
        expert_actions=expert_actions,
        policy_collections=loaded_collections,
        thresholds=thresholds,
        sample_hz=float(sample_hz),
    )
    scorer_path = Path(__file__).resolve().parents[1] / "policies" / (
        "startup_ownership.py"
    )
    source_manifest = {
        "schema_version": 1,
        "contract": "startup_ownership_source_manifest_v1",
        "dataset_dir": str(dataset),
        "split": str(split_file),
        "split_sha256": sha256_file(split_file),
        "deadzone_json": str(deadzone_file),
        "deadzone_json_sha256": sha256_file(deadzone_file),
        "sample_hz": float(sample_hz),
        "episode_ids": episode_ids,
        "expert_hdf5": expert_sources,
        "policy_collections": collection_sources,
        "audit_implementation": [
            {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            {
                "path": str(scorer_path),
                "sha256": sha256_file(scorer_path),
            },
        ],
        "source_hdf5_modified": False,
        "policy_inference_performed": False,
        "first_expert_onset_used_at_deployment": False,
    }
    report["inputs"] = source_manifest

    output.mkdir(parents=True, exist_ok=True)
    source_manifest_path = output / "source_manifest.json"
    source_manifest_path.write_text(
        json.dumps(source_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    report_path = output / "startup_ownership_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_rows_csv(output / "expert_episode_rows.csv", report["expert_episode_rows"])
    _write_rows_csv(output / "model_episode_rows.csv", report["model_episode_rows"])
    model_aggregate_rows = [
        {"model": model, **values}
        for model, values in report["aggregate"]["models"].items()
    ]
    _write_rows_csv(output / "model_aggregate.csv", model_aggregate_rows)
    return {
        "report": report_path,
        "report_sha256": sha256_file(report_path),
        "source_manifest": source_manifest_path,
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "expert_rows_csv": output / "expert_episode_rows.csv",
        "model_rows_csv": output / "model_episode_rows.csv",
        "model_aggregate_csv": output / "model_aggregate.csv",
    }


def _parse_policy_collection(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            f"--policy-collection must be MODEL=DIR, got {value!r}"
        )
    model, raw_path = value.split("=", 1)
    if not model.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError(
            f"--policy-collection must be MODEL=DIR, got {value!r}"
        )
    return model.strip(), Path(raw_path).expanduser()


def _validate_timebase(
    time_s: np.ndarray,
    *,
    expected_steps: int,
    sample_hz: float,
    source: str,
) -> None:
    if not np.isfinite(sample_hz) or sample_hz <= 0.0:
        raise ValueError("sample_hz must be finite and positive")
    if time_s.shape != (expected_steps,) or not np.isfinite(time_s).all():
        raise ValueError(f"{source} time_s must be finite shape ({expected_steps},)")
    if expected_steps > 1:
        expected_dt = 1.0 / float(sample_hz)
        if not np.allclose(
            np.diff(time_s), expected_dt, rtol=1.0e-6, atol=1.0e-9
        ):
            raise ValueError(f"{source} time_s does not match sample_hz={sample_hz}")


def _write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (dict, list))
                    else value
                    for key, value in row.items()
                }
            )


def _read_mapping_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return payload


def _required_file(value: str | Path, name: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    return path


def _required_directory(value: str | Path, name: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    return path


if __name__ == "__main__":
    main()
