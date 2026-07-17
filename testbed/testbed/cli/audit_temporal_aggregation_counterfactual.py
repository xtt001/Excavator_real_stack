"""Rescore saved state-hold aggregation alternatives with immutable provenance."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from testbed.policies.action_start_distribution import (
    FORBIDDEN_HELDOUT,
    sha256_file,
)
from testbed.policies.deadzone_eval import load_deadzone_thresholds
from testbed.policies.temporal_aggregation_counterfactual import (
    MODES,
    evaluate_temporal_aggregation_counterfactual,
)

_BUNDLE_ARTIFACTS = (
    "policy_best.ckpt",
    "dataset_stats.pkl",
    "resolved_config.yaml",
    "run_metadata.json",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m testbed.cli.audit_temporal_aggregation_counterfactual",
        description=(
            "Rescore legacy/newest/recency commands from complete, "
            "assist-disabled state-hold traces."
        ),
    )
    parser.add_argument(
        "--trace",
        action="append",
        required=True,
        metavar="MODEL=STATE_HOLD_ANCHORS_JSONL",
    )
    parser.add_argument("--deadzone-json", type=Path, required=True)
    parser.add_argument("--expected-anchors", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    traces = dict(_parse_trace(value) for value in args.trace)
    if len(traces) != len(args.trace):
        raise SystemExit("trace model labels must be unique")
    result = run_temporal_aggregation_counterfactual_audit(
        trace_paths=traces,
        deadzone_json=args.deadzone_json,
        expected_anchors=int(args.expected_anchors),
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                key: str(value) if isinstance(value, Path) else value
                for key, value in result.items()
            },
            ensure_ascii=False,
        )
    )


def run_temporal_aggregation_counterfactual_audit(
    *,
    trace_paths: Mapping[str, str | Path],
    deadzone_json: str | Path,
    expected_anchors: int,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Validate trace provenance, rescore commands, and write review artifacts."""

    if expected_anchors <= 0:
        raise ValueError("expected_anchors must be positive")
    if not trace_paths:
        raise ValueError("trace_paths must not be empty")
    deadzone_path = _required_file(deadzone_json, "deadzone_json")
    deadzone_sha256 = sha256_file(deadzone_path)
    thresholds = load_deadzone_thresholds(deadzone_path)

    rows_by_model: dict[str, list[dict[str, Any]]] = {}
    source_models: dict[str, dict[str, Any]] = {}
    for raw_model, raw_trace_path in trace_paths.items():
        model = str(raw_model).strip()
        if not model or model in rows_by_model:
            raise ValueError(f"invalid or duplicate model label: {raw_model!r}")
        trace_path = _required_file(raw_trace_path, f"trace {model}")
        rows = _read_jsonl(trace_path)
        if len(rows) != expected_anchors:
            raise ValueError(
                f"trace {model!r} has {len(rows)} rows; expected {expected_anchors}"
            )
        episode_ids = sorted(
            {str(row.get("episode_id", "")) for row in rows},
            key=_episode_number,
        )
        forbidden = sorted(
            {
                number
                for number in map(_episode_number, episode_ids)
                if number in FORBIDDEN_HELDOUT
            }
        )
        if forbidden:
            raise ValueError(f"held-out episodes are forbidden: {forbidden}")
        rows_by_model[model] = rows
        source_models[model] = _validate_source_run(
            trace_path=trace_path,
            trace_rows=rows,
            episode_ids=episode_ids,
            expected_anchors=expected_anchors,
            deadzone_sha256=deadzone_sha256,
        )

    report = evaluate_temporal_aggregation_counterfactual(
        rows_by_model=rows_by_model,
        thresholds=thresholds,
    )
    scorer_path = (
        Path(__file__).resolve().parents[1]
        / "policies"
        / "temporal_aggregation_counterfactual.py"
    )
    safety_path = (
        Path(__file__).resolve().parents[1] / "policies" / "state_hold_demo_relation.py"
    )
    source_manifest = {
        "schema_version": 1,
        "contract": "temporal_aggregation_counterfactual_source_manifest_v1",
        "expected_anchors_per_model": expected_anchors,
        "deadzone_json": str(deadzone_path),
        "deadzone_json_sha256": deadzone_sha256,
        "deadzone_action": thresholds,
        "models": source_models,
        "audit_implementation": [
            _file_record(Path(__file__).resolve()),
            _file_record(scorer_path),
            _file_record(safety_path),
        ],
        "heldout_episode_ids": sorted(FORBIDDEN_HELDOUT),
        "heldout_evaluated": False,
        "source_jsonl_modified": False,
        "policy_inference_performed": False,
        "alternatives_controlled_observations": False,
        "mechanical_assist_estimated": False,
    }
    report["inputs"] = source_manifest

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "source_manifest.json"
    report_path = output / "temporal_aggregation_counterfactual_report.json"
    per_anchor_jsonl_path = output / "per_anchor_comparison.jsonl"
    per_anchor_csv_path = output / "per_anchor_comparison.csv"
    mode_aggregate_csv_path = output / "mode_aggregate.csv"
    pairwise_aggregate_csv_path = output / "pairwise_aggregate.csv"

    _write_json(manifest_path, source_manifest)
    _write_json(report_path, report)
    _write_jsonl(per_anchor_jsonl_path, report["per_anchor"])
    _write_csv(per_anchor_csv_path, report["per_anchor"])
    _write_mode_aggregate_csv(mode_aggregate_csv_path, report["aggregate"])
    _write_pairwise_aggregate_csv(pairwise_aggregate_csv_path, report["aggregate"])

    paths = {
        "report": report_path,
        "source_manifest": manifest_path,
        "per_anchor_jsonl": per_anchor_jsonl_path,
        "per_anchor_csv": per_anchor_csv_path,
        "mode_aggregate_csv": mode_aggregate_csv_path,
        "pairwise_aggregate_csv": pairwise_aggregate_csv_path,
    }
    return {
        key: value
        for name, path in paths.items()
        for key, value in (
            (name, path),
            (f"{name}_sha256", sha256_file(path)),
        )
    }


def _validate_source_run(
    *,
    trace_path: Path,
    trace_rows: Sequence[Mapping[str, Any]],
    episode_ids: list[str],
    expected_anchors: int,
    deadzone_sha256: str,
) -> dict[str, Any]:
    run_summary_path = _required_file(
        trace_path.parent.parent / "run_summary.json", "run_summary"
    )
    run_summary = _read_json_mapping(run_summary_path)
    checks = {
        "pipeline_mode": "raw",
        "assist_mode": "disabled",
        "trace_full_horizon_after_reproduction": True,
        "temporal_aggregation_decomposition": True,
    }
    for key, expected in checks.items():
        if run_summary.get(key) != expected:
            raise ValueError(
                f"run summary {key!r} must be {expected!r}: {run_summary_path}"
            )
    if sorted(map(str, run_summary.get("episode_ids", [])), key=_episode_number) != (
        episode_ids
    ):
        raise ValueError("run summary episode ids do not match trace rows")
    horizons = {int(row.get("hold_horizon_steps", -1)) for row in trace_rows}
    if horizons != {int(run_summary.get("hold_horizon_steps", -1))}:
        raise ValueError("run summary hold horizon does not match trace rows")

    provenance = run_summary.get("deadzone_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("run summary is missing deadzone provenance")
    if provenance.get("action_domain") != "direct_policy_output":
        raise ValueError("source action domain must be direct_policy_output")
    if provenance.get("policy_action_scale") != [1.0, 1.0, 1.0, 1.0]:
        raise ValueError("source policy action scale must be identity")
    if bool(provenance.get("legacy_raw_scaled_deadzone_reused")):
        raise ValueError("legacy raw-scaled deadzone must not be reused")

    reports = run_summary.get("reports")
    if not isinstance(reports, list) or len(reports) != 1:
        raise ValueError("run summary must contain one assist-disabled report")
    source_report = reports[0]
    if (
        source_report.get("mode") != "assist_disabled"
        or source_report.get("pipeline_mode") != "raw"
        or source_report.get("assist_enabled") is not False
        or int(source_report.get("anchor_rows", -1)) != expected_anchors
    ):
        raise ValueError("run report does not match required raw/assist-disabled trace")
    report_paths = source_report.get("paths")
    if (
        not isinstance(report_paths, Mapping)
        or Path(str(report_paths.get("rows_jsonl", ""))).resolve() != trace_path
    ):
        raise ValueError("run report rows_jsonl does not match requested trace")

    resolved_deadzone_path = _required_file(
        run_summary.get("resolved_direct_output_deadzone"),
        "resolved_direct_output_deadzone",
    )
    resolved_deadzone_sha256 = sha256_file(resolved_deadzone_path)
    if resolved_deadzone_sha256 != deadzone_sha256:
        raise ValueError("source run deadzone differs from supplied deadzone")

    config_path = _required_file(run_summary.get("config_path"), "source config")
    bundle_dir = _required_directory(
        run_summary.get("action_bundle_dir"), "action bundle"
    )
    bundle_artifacts = {
        name: _file_record(_required_file(bundle_dir / name, f"bundle {name}"))
        for name in _BUNDLE_ARTIFACTS
    }
    return {
        "candidate_id": str(run_summary.get("candidate_id", "")),
        "trace": _file_record(trace_path),
        "trace_rows": len(trace_rows),
        "episode_ids": episode_ids,
        "hold_horizon_steps": next(iter(horizons)),
        "run_summary": _file_record(run_summary_path),
        "source_config": _file_record(config_path),
        "action_bundle_dir": str(bundle_dir),
        "bundle_artifacts": bundle_artifacts,
        "resolved_direct_output_deadzone": _file_record(resolved_deadzone_path),
        "pipeline_mode": "raw",
        "assist_enabled": False,
        "temporal_aggregation_decomposition_complete_required": True,
    }


def _parse_trace(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"--trace must be MODEL=PATH, got: {value}")
    model, raw_path = value.split("=", 1)
    model = model.strip()
    if not model or not raw_path.strip():
        raise ValueError(f"invalid --trace value: {value}")
    return model, Path(raw_path).expanduser().resolve()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            raise ValueError(f"blank JSONL line at {path}:{line_number}")
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"JSONL row must be an object at {path}:{line_number}")
        rows.append(payload)
    if not rows:
        raise ValueError(f"JSONL trace must not be empty: {path}")
    return rows


def _read_json_mapping(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON source must be an object: {path}")
    return payload


def _required_file(value: str | Path | None, label: str) -> Path:
    if value is None:
        raise FileNotFoundError(f"{label} path is missing")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} file does not exist: {path}")
    return path


def _required_directory(value: str | Path | None, label: str) -> Path:
    if value is None:
        raise FileNotFoundError(f"{label} path is missing")
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"{label} directory does not exist: {path}")
    return path


def _file_record(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256_file(path)}


def _episode_number(value: str) -> int:
    try:
        return int(str(value).split("_")[-1])
    except ValueError as exc:
        raise ValueError(f"invalid episode id: {value!r}") from exc


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _write_mode_aggregate_csv(
    path: Path, aggregate: Mapping[str, Mapping[str, Any]]
) -> None:
    rows = [
        {"model": model, "mode": mode, **model_values["modes"][mode]}
        for model, model_values in aggregate.items()
        for mode in MODES
    ]
    _write_csv(path, rows)


def _write_pairwise_aggregate_csv(
    path: Path, aggregate: Mapping[str, Mapping[str, Any]]
) -> None:
    rows = [
        {"model": model, "comparison": comparison, **values}
        for model, model_values in aggregate.items()
        for comparison, values in model_values["comparisons"].items()
    ]
    _write_csv(path, rows)


def _csv_value(value: Any) -> Any:
    if isinstance(value, (Mapping, list, tuple)):
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    return value


if __name__ == "__main__":
    main()


__all__ = ["run_temporal_aggregation_counterfactual_audit"]
