"""Evaluate saved validation actions against ExpertIntentEvent semantics."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from testbed.data.expert_intent_events import (
    EVENTS_FILENAME,
    MANIFEST_FILENAME,
    SEALED_TEST_EPISODE_IDS,
    sha256_file,
)
from testbed.data.expert_intent_events import (
    SCHEMA_VERSION as EVENT_SCHEMA_VERSION,
)
from testbed.policies.deadzone_eval import (
    load_deadzone_thresholds,
    parse_eval_spec,
)
from testbed.policies.intent_eval import (
    INFERENCE_SOURCE,
    SCHEMA_VERSION,
    evaluate_open_loop_intent,
)

REPORT_FILENAME = "expert_intent_eval_report.json"
ROWS_FILENAME = "expert_intent_eval_rows.jsonl"
ROWS_CSV_FILENAME = "expert_intent_eval_rows.csv"
SOURCE_MANIFEST_FILENAME = "expert_intent_eval_source_manifest.json"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m testbed.cli.evaluate_expert_intent",
        description=(
            "Threshold saved teacher-forced open-loop continuous outputs and "
            "compare them with explicit ExpertIntentEvent direction sets."
        ),
    )
    parser.add_argument("--eval", action="append", required=True, metavar="MODEL=DIR")
    parser.add_argument("--event-dir", type=Path, required=True)
    parser.add_argument("--deadzone-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="validation", choices=("validation",))
    parser.add_argument("--sampling-hz", type=float, default=20.0)
    args = parser.parse_args(argv)

    eval_dirs: dict[str, Path] = {}
    for value in args.eval:
        spec = parse_eval_spec(value)
        if spec.model in eval_dirs:
            parser.error(f"duplicate --eval model label: {spec.model}")
        eval_dirs[spec.model] = spec.eval_dir
    result = run_expert_intent_evaluation(
        eval_dirs=eval_dirs,
        event_dir=args.event_dir,
        deadzone_json=args.deadzone_json,
        output_dir=args.output_dir,
        split=args.split,
        sampling_hz=args.sampling_hz,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


def run_expert_intent_evaluation(
    *,
    eval_dirs: Mapping[str, str | Path],
    event_dir: str | Path,
    deadzone_json: str | Path,
    output_dir: str | Path,
    split: str,
    sampling_hz: float = 20.0,
) -> dict[str, Any]:
    """Validate immutable inputs, score models, and atomically write artifacts."""

    if split != "validation":
        raise ValueError("only the validation split is permitted")
    hz = float(sampling_hz)
    if not np.isfinite(hz) or hz <= 0.0:
        raise ValueError("sampling_hz must be finite and positive")
    if not eval_dirs:
        raise ValueError("at least one eval directory is required")
    events_root = _required_directory(event_dir, "event_dir")
    manifest_path = _required_file(events_root / MANIFEST_FILENAME, "event manifest")
    events_path = _required_file(events_root / EVENTS_FILENAME, "event JSONL")
    deadzone_path = _required_file(deadzone_json, "deadzone_json")
    manifest = _read_json_mapping(manifest_path)
    if manifest.get("schema_version") != EVENT_SCHEMA_VERSION:
        raise ValueError("unsupported ExpertIntentEvent manifest schema")
    expected_event_hash = manifest.get("artifacts", {}).get(EVENTS_FILENAME)
    if expected_event_hash != sha256_file(events_path):
        raise ValueError("ExpertIntentEvent JSONL hash does not match manifest")
    thresholds = load_deadzone_thresholds(deadzone_path)
    if _normalized_thresholds(manifest.get("thresholds")) != thresholds:
        raise ValueError("deadzone thresholds differ from ExpertIntentEvent manifest")
    if manifest.get("threshold_source_sha256") != sha256_file(deadzone_path):
        raise ValueError("deadzone source hash differs from ExpertIntentEvent manifest")

    expected_ids = [int(value) for value in manifest.get("validation_ids", [])]
    if not expected_ids or len(expected_ids) != len(set(expected_ids)):
        raise ValueError("event manifest validation_ids must be unique and nonempty")
    _reject_sealed_ids(expected_ids, context="validation IDs")
    _reject_sealed_source_paths(manifest)
    events = _read_validation_events(events_path)
    event_ids = {int(event["episode_id"]) for event in events}
    if event_ids != set(expected_ids):
        raise ValueError(
            "validation event episode IDs do not exactly match manifest validation IDs"
        )

    reference_expert: dict[int, np.ndarray] | None = None
    sources: dict[str, Any] = {}
    reports: dict[str, Any] = {}
    all_rows: list[dict[str, Any]] = []
    for raw_model, raw_dir in eval_dirs.items():
        model = str(raw_model).strip()
        if not model or model in reports:
            raise ValueError(f"invalid or duplicate model label: {raw_model!r}")
        eval_root = _required_directory(raw_dir, f"eval directory {model}")
        expert, policy, model_sources = _load_eval_actions(
            model=model,
            eval_dir=eval_root,
            expected_ids=expected_ids,
        )
        if reference_expert is None:
            reference_expert = expert
        else:
            for episode_id in expected_ids:
                if not np.array_equal(reference_expert[episode_id], expert[episode_id]):
                    raise ValueError(
                        "expert_action differs across models for "
                        f"episode_{episode_id}: {model}"
                    )
        _validate_event_expert_alignment(
            events=events,
            expert_actions=expert,
            thresholds=thresholds,
        )
        report = evaluate_open_loop_intent(
            model=model,
            events=events,
            policy_actions=policy,
            thresholds=thresholds,
            sampling_hz=hz,
        )
        reports[model] = {key: value for key, value in report.items() if key != "rows"}
        all_rows.extend(report["rows"])
        sources[model] = model_sources

    source_manifest = {
        "schema_version": SCHEMA_VERSION,
        "inference_source": INFERENCE_SOURCE,
        "split": split,
        "sampling_hz": hz,
        "algorithm_semantics": {
            "event_windows": "event-level direction-set comparison",
            "expert_direction_onset_early": (
                "neutral expert timing disagreement; not an unsafe or premature claim"
            ),
            "startup_readiness": (
                "first deadzone-effective policy output from step 0 only; later "
                "teacher-forced outputs do not affect readiness"
            ),
            "recording_wait": "not treated as expert idle ground truth",
            "startup_axis_requirement": "none",
            "startup_expert_match": (
                "exact, overlap, local-support, and opposite results are descriptive "
                "expert-data similarity only"
            ),
            "startup_gate_policy": (
                "no startup expert-match metric is a promotion or safety gate"
            ),
        },
        "event_dir": str(events_root),
        "event_manifest": str(manifest_path),
        "event_manifest_sha256": sha256_file(manifest_path),
        "event_jsonl": str(events_path),
        "event_jsonl_sha256": sha256_file(events_path),
        "deadzone_json": str(deadzone_path),
        "deadzone_json_sha256": sha256_file(deadzone_path),
        "thresholds": thresholds,
        "validation_ids": expected_ids,
        "models": sources,
        "evaluation_implementation": [
            {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            {
                "path": str(
                    Path(__file__).resolve().parents[1] / "policies" / "intent_eval.py"
                ),
                "sha256": sha256_file(
                    Path(__file__).resolve().parents[1] / "policies" / "intent_eval.py"
                ),
            },
        ],
        "source_hdf5_read": False,
        "policy_inference_performed": False,
        "model_command_changed": False,
        "sealed_test_data_read": False,
    }
    report_payload = {
        "schema_version": SCHEMA_VERSION,
        "inference_source": INFERENCE_SOURCE,
        "split": split,
        "sampling_hz": hz,
        "capability_boundaries": next(iter(reports.values()))["capability_boundaries"],
        "event_count": len(events),
        "episode_count": len(expected_ids),
        "models": reports,
        "source_manifest": SOURCE_MANIFEST_FILENAME,
    }
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(output / SOURCE_MANIFEST_FILENAME, source_manifest)
    _write_jsonl_atomic(output / ROWS_FILENAME, all_rows)
    _write_rows_csv_atomic(output / ROWS_CSV_FILENAME, all_rows)
    _write_json_atomic(output / REPORT_FILENAME, report_payload)
    return {
        "report": str(output / REPORT_FILENAME),
        "report_sha256": sha256_file(output / REPORT_FILENAME),
        "event_rows": len(all_rows),
        "models": list(reports),
    }


def _load_eval_actions(
    *,
    model: str,
    eval_dir: Path,
    expected_ids: list[int],
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[str, Any]]:
    summary_path = _required_file(
        eval_dir / "collection_summary.json", f"{model} collection summary"
    )
    summary = _read_json_mapping(summary_path)
    summary_ids = [
        _parse_episode_label(value) for value in summary.get("episode_ids", [])
    ]
    if len(summary_ids) != len(set(summary_ids)) or set(summary_ids) != set(
        expected_ids
    ):
        raise ValueError(
            f"{model} collection episode IDs do not exactly match validation"
        )
    actual_paths = sorted((eval_dir / "episodes").glob("episode_*/actions.npz"))
    actual_ids = [_parse_episode_label(path.parent.name) for path in actual_paths]
    if len(actual_ids) != len(set(actual_ids)) or set(actual_ids) != set(expected_ids):
        raise ValueError(f"{model} actions episode IDs do not exactly match validation")
    _reject_sealed_ids(actual_ids, context=f"{model} actions")

    expert: dict[int, np.ndarray] = {}
    policy: dict[int, np.ndarray] = {}
    action_sources = []
    for episode_id in expected_ids:
        path = _required_file(
            eval_dir / "episodes" / f"episode_{episode_id}" / "actions.npz",
            f"{model}/episode_{episode_id} actions",
        )
        with np.load(path, allow_pickle=False) as payload:
            missing = [
                key for key in ("expert_action", "policy_action") if key not in payload
            ]
            if missing:
                raise KeyError(f"{path} is missing array(s): {', '.join(missing)}")
            stored_expert = _action_array(
                payload["expert_action"], name=f"{model}/episode_{episode_id} expert"
            )
            stored_policy = _action_array(
                payload["policy_action"], name=f"{model}/episode_{episode_id} policy"
            )
        if stored_policy.shape != stored_expert.shape:
            raise ValueError(
                f"{model}/episode_{episode_id} expert/policy shapes differ"
            )
        expert[episode_id] = stored_expert
        policy[episode_id] = stored_policy
        action_sources.append(
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
            "actions": action_sources,
        },
    )


def _validate_event_expert_alignment(
    *,
    events: list[dict[str, Any]],
    expert_actions: Mapping[int, np.ndarray],
    thresholds: Mapping[str, Mapping[str, float]],
) -> None:
    masks = {
        episode_id: _effective_labels_by_step(action, thresholds)
        for episode_id, action in expert_actions.items()
    }
    for event in events:
        episode_id = int(event["episode_id"])
        labels = masks[episode_id]
        onset = int(event["onset_step"])
        requested = int(event["support_horizon_requested_ticks"])
        expected_end = min(len(labels), onset + requested)
        if not 0 <= onset < len(labels):
            raise ValueError(f"{event['event_id']} onset is out of actions bounds")
        if int(event["support_end_step_exclusive"]) != expected_end:
            raise ValueError(f"{event['event_id']} support window is out of alignment")
        checks = (
            ("anchor_intent", 0, 0),
            ("immediate_intent_0_1", 0, 1),
            ("near_intent_2_5", 2, 5),
            ("near_intent_6_10", 6, 10),
            ("single_demo_event_support_directions", 0, requested - 1),
        )
        for field, start, stop in checks:
            observed = _union_labels(labels, onset + start, onset + stop + 1)
            if observed != set(event[field]):
                raise ValueError(
                    f"expert_action does not match {event['event_id']} field {field}"
                )


def _effective_labels_by_step(
    action: np.ndarray,
    thresholds: Mapping[str, Mapping[str, float]],
) -> list[set[str]]:
    from testbed.policies.deadzone_eval import AXIS_NAMES, effective_direction_mask

    mask = effective_direction_mask(action, dict(thresholds))
    result = []
    for step in mask:
        labels = set()
        for index, axis in enumerate(AXIS_NAMES):
            if step[index, 0]:
                labels.add(f"{axis}+")
            if step[index, 1]:
                labels.add(f"{axis}-")
        result.append(labels)
    return result


def _union_labels(labels: list[set[str]], start: int, stop: int) -> set[str]:
    if start >= len(labels):
        return set()
    result: set[str] = set()
    for value in labels[start : min(stop, len(labels))]:
        result.update(value)
    return result


def _read_validation_events(path: Path) -> list[dict[str, Any]]:
    events = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"event line {line_number} is not a mapping")
            if payload.get("split") == "validation":
                events.append(payload)
    if not events:
        raise ValueError("ExpertIntentEvent JSONL has no validation events")
    ids = [str(event.get("event_id")) for event in events]
    if len(ids) != len(set(ids)):
        raise ValueError("validation event IDs are not unique")
    return events


def _reject_sealed_source_paths(manifest: Mapping[str, Any]) -> None:
    validation_ids = set(int(value) for value in manifest.get("validation_ids", []))
    for episode in manifest.get("episodes", []):
        if int(episode.get("episode_id", -1)) not in validation_ids:
            continue
        match = re.search(r"episode_(\d+)\.hdf5$", str(episode.get("path", "")))
        if match and int(match.group(1)) in SEALED_TEST_EPISODE_IDS:
            raise ValueError(
                f"validation sidecar references sealed source ID {match.group(1)}"
            )


def _reject_sealed_ids(values: list[int], *, context: str) -> None:
    forbidden = sorted(set(values) & SEALED_TEST_EPISODE_IDS)
    if forbidden:
        raise ValueError(f"{context} contains sealed/test episode IDs: {forbidden}")


def _parse_episode_label(value: Any) -> int:
    match = re.fullmatch(r"episode_(\d+)", str(value))
    if match is None:
        raise ValueError(f"invalid episode label: {value!r}")
    return int(match.group(1))


def _normalized_thresholds(value: Any) -> dict[str, dict[str, float]]:
    if not isinstance(value, Mapping):
        raise ValueError("event manifest thresholds must be a mapping")
    return {
        str(axis): {str(sign): float(threshold) for sign, threshold in values.items()}
        for axis, values in value.items()
        if isinstance(values, Mapping)
    }


def _action_array(value: Any, *, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.ndim != 2 or result.shape[1] != 4 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite (T, 4) array")
    return result


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


def _read_json_mapping(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a mapping")
    return payload


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    _write_text_atomic(
        path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    _write_text_atomic(
        path,
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for row in rows
        ),
    )


def _write_rows_csv_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = (
        "model",
        "event_id",
        "episode_id",
        "event_index",
        "onset_step",
        "window",
        "window_complete",
        "demonstrated_directions",
        "predicted_directions",
        "matched_demonstrated_directions",
        "outside_demonstrated_window_directions",
        "single_demo_later_supported_directions",
        "single_demo_direction_onset_later_directions",
        "outside_single_demo_event_support_directions",
        "opposite_to_single_demo_anchor_directions",
        "single_demo_direction_recall",
        "single_demo_exact_set",
    )
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        output = {field: row[field] for field in fields}
        for field in fields[7:15]:
            output[field] = "|".join(output[field])
        writer.writerow(output)
    _write_text_atomic(path, stream.getvalue())


def _write_text_atomic(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


if __name__ == "__main__":
    main()
