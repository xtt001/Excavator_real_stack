"""Evaluate whole policy activation sequences against training expert tasks."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from testbed.data.expert_intent_events import (
    MANIFEST_FILENAME,
    SEALED_TEST_EPISODE_IDS,
    sha256_file,
)
from testbed.data.expert_intent_events import (
    SCHEMA_VERSION as EVENT_SCHEMA_VERSION,
)
from testbed.policies.deadzone_eval import load_deadzone_thresholds, parse_eval_spec
from testbed.policies.task_sequence_compatibility import (
    SCHEMA_VERSION,
    evaluate_task_sequence_compatibility,
)

REPORT_FILENAME = "task_sequence_compatibility_report.json"
SOURCE_MANIFEST_FILENAME = "task_sequence_compatibility_source_manifest.json"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m testbed.cli.evaluate_task_sequence_compatibility",
        description=(
            "Compare complete deadzone-effective direction-change sequences with "
            "the training expert cohort, calibrated by held-out experts."
        ),
    )
    parser.add_argument("--eval", action="append", required=True, metavar="MODEL=DIR")
    parser.add_argument("--event-dir", type=Path, required=True)
    parser.add_argument("--deadzone-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    eval_dirs: dict[str, Path] = {}
    for value in args.eval:
        spec = parse_eval_spec(value)
        if spec.model in eval_dirs:
            parser.error(f"duplicate --eval model label: {spec.model}")
        eval_dirs[spec.model] = spec.eval_dir
    result = run_task_sequence_compatibility_evaluation(
        eval_dirs=eval_dirs,
        event_dir=args.event_dir,
        deadzone_json=args.deadzone_json,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


def run_task_sequence_compatibility_evaluation(
    *,
    eval_dirs: Mapping[str, str | Path],
    event_dir: str | Path,
    deadzone_json: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Load immutable reference artifacts and write one multi-model report."""

    if not eval_dirs:
        raise ValueError("at least one eval directory is required")
    events_root = _required_directory(event_dir, "event_dir")
    event_manifest_path = _required_file(
        events_root / MANIFEST_FILENAME, "event manifest"
    )
    deadzone_path = _required_file(deadzone_json, "deadzone_json")
    event_manifest = _read_json(event_manifest_path)
    if event_manifest.get("schema_version") != EVENT_SCHEMA_VERSION:
        raise ValueError("unsupported expert event manifest schema")
    thresholds = load_deadzone_thresholds(deadzone_path)
    if event_manifest.get("thresholds") != thresholds:
        raise ValueError("deadzone thresholds differ from event manifest")
    if event_manifest.get("threshold_source_sha256") != sha256_file(deadzone_path):
        raise ValueError("deadzone source hash differs from event manifest")

    train_ids = [int(value) for value in event_manifest.get("train_ids", [])]
    validation_ids = [int(value) for value in event_manifest.get("validation_ids", [])]
    if not train_ids or not validation_ids:
        raise ValueError("event manifest train/validation IDs must be nonempty")
    if set(train_ids) & set(validation_ids):
        raise ValueError("event manifest train/validation IDs overlap")
    sealed = sorted((set(train_ids) | set(validation_ids)) & SEALED_TEST_EPISODE_IDS)
    if sealed:
        raise ValueError(f"sealed/test episode IDs are forbidden: {sealed}")

    episodes = event_manifest.get("episodes")
    if not isinstance(episodes, list):
        raise ValueError("event manifest episodes must be a list")
    episode_source_by_id = {
        int(item["episode_id"]): item for item in episodes if isinstance(item, Mapping)
    }
    if set(episode_source_by_id) != set(train_ids) | set(validation_ids):
        raise ValueError("event manifest episode sources do not match split IDs")

    training_actions: dict[int, np.ndarray] = {}
    training_sources = []
    for episode_id in train_ids:
        source = episode_source_by_id[episode_id]
        path = _required_file(source["path"], f"training episode {episode_id}")
        actual_hash = sha256_file(path)
        if actual_hash != source.get("sha256"):
            raise ValueError(f"training episode {episode_id} hash changed")
        with h5py.File(path, "r") as handle:
            actions = _action_array(handle["/action"][()], name=str(path))
        training_actions[episode_id] = actions
        training_sources.append(
            {
                "episode_id": episode_id,
                "path": str(path),
                "sha256": actual_hash,
                "steps": int(actions.shape[0]),
            }
        )

    reference_expert: dict[int, np.ndarray] | None = None
    model_actions: dict[str, dict[int, np.ndarray]] = {}
    eval_sources: dict[str, Any] = {}
    for raw_model, raw_path in eval_dirs.items():
        model = str(raw_model).strip()
        if not model or model in model_actions:
            raise ValueError(f"invalid or duplicate model label: {raw_model!r}")
        expert, policy, source = _load_saved_open_loop(
            model=model,
            eval_dir=_required_directory(raw_path, f"eval directory {model}"),
            expected_ids=validation_ids,
        )
        if reference_expert is None:
            reference_expert = expert
        else:
            for episode_id in validation_ids:
                if not np.array_equal(reference_expert[episode_id], expert[episode_id]):
                    raise ValueError(
                        f"validation expert actions differ across models: {model}, "
                        f"episode {episode_id}"
                    )
        model_actions[model] = policy
        eval_sources[model] = source
    assert reference_expert is not None

    reports = {
        model: evaluate_task_sequence_compatibility(
            model=model,
            training_expert_actions=training_actions,
            validation_expert_actions=reference_expert,
            policy_actions=policy,
            thresholds=thresholds,
        )
        for model, policy in model_actions.items()
    }
    first_report = next(iter(reports.values()))
    report_payload = {
        "schema_version": SCHEMA_VERSION,
        "capability_boundaries": first_report["capability_boundaries"],
        "training_reference": first_report["training_reference"],
        "validation_expert_calibration": first_report["validation_expert_calibration"],
        "counterfactual_controls": first_report["counterfactual_controls"],
        "models": {
            model: {
                "model_summary": report["model_summary"],
                "cohort_comparison": report["cohort_comparison"],
                "rows": report["rows"],
            }
            for model, report in reports.items()
        },
        "source_manifest": SOURCE_MANIFEST_FILENAME,
    }
    source_manifest = {
        "schema_version": SCHEMA_VERSION,
        "event_dir": str(events_root),
        "event_manifest": str(event_manifest_path),
        "event_manifest_sha256": sha256_file(event_manifest_path),
        "deadzone_json": str(deadzone_path),
        "deadzone_json_sha256": sha256_file(deadzone_path),
        "thresholds": thresholds,
        "train_ids": train_ids,
        "validation_ids": validation_ids,
        "training_sources": training_sources,
        "models": eval_sources,
        "source_hdf5_read": True,
        "policy_inference_performed": False,
        "model_command_changed": False,
        "sealed_test_data_read": False,
        "implementation": [
            {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            {
                "path": str(
                    Path(__file__).resolve().parents[1]
                    / "policies"
                    / "task_sequence_compatibility.py"
                ),
                "sha256": sha256_file(
                    Path(__file__).resolve().parents[1]
                    / "policies"
                    / "task_sequence_compatibility.py"
                ),
            },
        ],
    }

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(output / SOURCE_MANIFEST_FILENAME, source_manifest)
    _write_json_atomic(output / REPORT_FILENAME, report_payload)
    return {
        "report": str(output / REPORT_FILENAME),
        "report_sha256": sha256_file(output / REPORT_FILENAME),
        "models": list(reports),
        "train_episodes": len(train_ids),
        "validation_episodes": len(validation_ids),
    }


def _load_saved_open_loop(
    *,
    model: str,
    eval_dir: Path,
    expected_ids: list[int],
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[str, Any]]:
    summary_path = _required_file(
        eval_dir / "collection_summary.json", f"{model} collection summary"
    )
    summary = _read_json(summary_path)
    summary_ids = [_episode_id(value) for value in summary.get("episode_ids", [])]
    if set(summary_ids) != set(expected_ids) or len(summary_ids) != len(expected_ids):
        raise ValueError(f"{model} collection IDs do not match validation IDs")

    expert: dict[int, np.ndarray] = {}
    policy: dict[int, np.ndarray] = {}
    sources = []
    for episode_id in expected_ids:
        path = _required_file(
            eval_dir / "episodes" / f"episode_{episode_id}" / "actions.npz",
            f"{model}/episode_{episode_id} actions",
        )
        with np.load(path, allow_pickle=False) as payload:
            if "expert_action" not in payload or "policy_action" not in payload:
                raise KeyError(f"{path} lacks expert_action or policy_action")
            stored_expert = _action_array(
                payload["expert_action"], name=f"{model}/{episode_id} expert"
            )
            stored_policy = _action_array(
                payload["policy_action"], name=f"{model}/{episode_id} policy"
            )
        if stored_expert.shape != stored_policy.shape:
            raise ValueError(f"{model}/{episode_id} expert/policy shapes differ")
        expert[episode_id] = stored_expert
        policy[episode_id] = stored_policy
        sources.append(
            {
                "episode_id": episode_id,
                "path": str(path),
                "sha256": sha256_file(path),
                "steps": int(stored_expert.shape[0]),
            }
        )
    return (
        expert,
        policy,
        {
            "eval_dir": str(eval_dir),
            "collection_summary": str(summary_path),
            "collection_summary_sha256": sha256_file(summary_path),
            "actions": sources,
        },
    )


def _action_array(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 4 or array.shape[0] == 0:
        raise ValueError(f"{name} must have nonempty shape (T, 4)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def _episode_id(value: Any) -> int:
    text = str(value)
    if text.startswith("episode_"):
        text = text.removeprefix("episode_")
    return int(text)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _required_file(path: str | Path, name: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{name} does not exist: {resolved}")
    return resolved


def _required_directory(path: str | Path, name: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"{name} does not exist: {resolved}")
    return resolved


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


if __name__ == "__main__":
    main()
