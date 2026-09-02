"""Retrospective candidate decision using only raw-expert-supported gates."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from pathlib import Path
from typing import Any

from testbed.tasks.real_transition import sha256_file, write_immutable_text

SCHEMA = "real_transition_task_state_v2_expert_aligned_redecision_v1"


def main() -> None:
    parser = argparse.ArgumentParser(prog="tb-task-state-v2-expert-redecision")
    parser.add_argument("--expert-reference", type=Path, required=True)
    parser.add_argument("--parent-result", type=Path, required=True)
    parser.add_argument(
        "--acceptance-result", type=Path, action="append", required=True
    )
    parser.add_argument("--max-uncommitted-failures", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = decide(
        expert_reference=args.expert_reference,
        parent_result=args.parent_result,
        acceptance_results=args.acceptance_result,
        max_uncommitted_failures=int(args.max_uncommitted_failures),
        output_dir=args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def decide(
    *,
    expert_reference: Path | str,
    parent_result: Path | str,
    acceptance_results: list[Path] | tuple[Path, ...],
    max_uncommitted_failures: int = 1,
    output_dir: Path | str,
) -> dict[str, Any]:
    expert_path = Path(expert_reference).resolve()
    parent_path = Path(parent_result).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite expert redecision: {output}")
    expert = _json(expert_path)
    parent = _json(parent_path)
    probe_sha = str(expert["probe_manifest"]["sha256"])
    if str(parent.get("probe_manifest_sha256")) != probe_sha:
        raise ValueError("parent result used a different probe")
    counts = expert["expert_aligned_gate_counts"]
    bta = counts["b_to_a"]
    uncommitted = counts["uncommitted_heldout"]
    allowed_failures = int(max_uncommitted_failures)
    if allowed_failures < 0 or allowed_failures >= int(uncommitted["population"]):
        raise ValueError(
            "max_uncommitted_failures must be non-negative and below the "
            "heldout population"
        )
    other = counts["other_transitions"]
    parent_all = parent["summary"]["heldout_all"]
    gates = [
        _gate("bta_no_direct_shortcut", "heldout_b_to_a.direct_shortcut_rate", "<=", bta["direct_shortcut_max"] / bta["population"]),
        _gate("bta_correct_start_motion", "heldout_b_to_a.work_start_correct_motion_rate", ">=", bta["correct_start_motion_min"] / bta["population"]),
        _gate("bta_tool_liveness", "heldout_b_to_a.work_start_tool_liveness_rate", ">=", bta["tool_liveness_min"] / bta["population"]),
        _gate("bta_positive_excursion", "heldout_b_to_a.outbound_positive_swing_rate", ">=", bta["positive_excursion_min"] / bta["population"]),
        _gate("bta_bucket_liveness", "heldout_b_to_a.bucket_tool_liveness_rate", ">=", bta["bucket_liveness_min"] / bta["population"]),
        _gate("bta_return", "heldout_b_to_a.return_negative_swing_rate", ">=", bta["return_negative_min"] / bta["population"]),
        _gate("bta_return_crossing", "heldout_b_to_a.return_ready_crossing_negative_swing_rate", ">=", bta["return_ready_crossing_min"] / bta["population"]),
        _gate("bta_ordered_proxy", "heldout_b_to_a.ordered_action_proxy_rate", ">=", bta["ordered_proxy_min"] / bta["population"]),
        _gate("uncommitted_no_negative", "uncommitted_boundary_state.no_negative_swing_rate", ">=", (uncommitted["population"] - allowed_failures) / uncommitted["population"]),
        _gate("other_no_shortcut", "heldout_other.direct_shortcut_rate", "<=", 0.0),
        _gate("other_start_motion", "heldout_other.work_start_correct_motion_rate", ">=", other["correct_start_motion_min"] / other["population"]),
        _gate("other_positive_excursion", "heldout_other.outbound_positive_swing_rate", ">=", 0.75),
        _gate("other_bucket_liveness", "heldout_other.bucket_tool_liveness_rate", ">=", 0.90),
        _gate("other_return", "heldout_other.return_negative_swing_rate", ">=", 0.70),
        _gate("other_ordered_proxy", "heldout_other.ordered_action_proxy_rate", ">=", 0.50),
        _gate("heldout_mae", "heldout_all.heldout_aggregated_action_mae", "<=", float(parent_all["heldout_aggregated_action_mae"]) * 1.15),
        _gate("heldout_sign", "heldout_all.heldout_effective_sign_agreement", ">=", float(parent_all["heldout_effective_sign_agreement"]) - 0.05),
        _gate("factual_ready_no_shortcut", "b_to_a_ready_pair_factual_qvel.work_no_negative_rate", ">=", 7 / 8),
        _gate("factual_ready_liveness", "b_to_a_ready_pair_factual_qvel.work_any_axis_effective_rate", ">=", 6 / 8),
    ]
    candidate_entries: dict[str, dict[str, Any]] = {}
    source_acceptances = []
    for acceptance_path_raw in acceptance_results:
        acceptance_path = Path(acceptance_path_raw).resolve()
        acceptance = _json(acceptance_path)
        label = acceptance_path.parent.name
        source_acceptances.append(
            {"path": str(acceptance_path), "sha256": sha256_file(acceptance_path)}
        )
        for candidate, spec_raw in acceptance["candidate_summaries"].items():
            name = f"{label}:{candidate}"
            spec = dict(spec_raw)
            result_path = Path(str(spec["result"])).resolve()
            result = _json(result_path)
            if str(result.get("probe_manifest_sha256")) != probe_sha:
                raise ValueError(f"candidate {name} used a different probe")
            candidate_entries[name] = {
                "result": str(result_path),
                "result_sha256": sha256_file(result_path),
                "checkpoint": str(result["checkpoint"]),
                "checkpoint_sha256": str(result["checkpoint_sha256"]),
                "summary": result["summary"],
            }
    gate_rows = []
    for name, candidate in candidate_entries.items():
        failed = []
        for gate in gates:
            value = _metric(candidate["summary"], gate["metric"])
            passed = (
                value >= gate["threshold"]
                if gate["operator"] == ">="
                else value <= gate["threshold"]
            )
            gate_rows.append(
                {
                    "candidate": name,
                    **gate,
                    "value": value,
                    "passed": bool(passed),
                }
            )
            if not passed:
                failed.append(gate["name"])
        candidate["passes_all_expert_aligned_gates"] = not failed
        candidate["failed_gates"] = failed
    passing = [
        name
        for name, value in candidate_entries.items()
        if value["passes_all_expert_aligned_gates"]
    ]
    ranked = sorted(
        passing, key=lambda name: (*_rank(candidate_entries[name]), name)
    )
    selected = None if not ranked else ranked[0]
    payload = {
        "schema": SCHEMA,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": (
            "RETROSPECTIVE_NO_OFFLINE_CANDIDATE"
            if selected is None
            else "RETROSPECTIVE_EXPERT_ALIGNED_CANDIDATE"
        ),
        "selected_candidate": selected,
        "expert_reference": {"path": str(expert_path), "sha256": sha256_file(expert_path)},
        "parent_reference": {"path": str(parent_path), "sha256": sha256_file(parent_path)},
        "source_acceptance_results": source_acceptances,
        "candidate_count": len(candidate_entries),
        "gates": gates,
        "diagnostic_only_removed_from_hard_gates": expert[
            "diagnostic_only_metrics"
        ],
        "candidate_summaries": candidate_entries,
        "passing_candidates_ranked": ranked,
        "uncommitted_tolerance": {
            "population": int(uncommitted["population"]),
            "expert_failure_count": 0,
            "allowed_failure_count": allowed_failures,
            "minimum_pass_count": int(uncommitted["population"])
            - allowed_failures,
            "requirement_change_source": (
                "user-authorized post-result tolerance change"
                if allowed_failures > 1
                else "expert-aligned one-cycle tolerance"
            ),
        },
        "frozen_after_candidate_results_exist": True,
        "threshold_source": (
            "raw expert counts plus the previously fixed parent MAE/sign "
            "non-inferiority rule; candidate metrics were not used"
        ),
        "interpretation": (
            "Unsupported zero-qvel and intervention-delta gates remain "
            "diagnostic only. The pre-commit failure tolerance is recorded "
            "explicitly and never interpreted as expert performance."
        ),
        "physical_evidence": False,
    }
    output.mkdir(parents=True)
    result_path = write_immutable_text(
        output / "expert_aligned_redecision.json",
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    _write_csv(output / "expert_aligned_gate_results.csv", gate_rows)
    sums = [
        f"{sha256_file(path)}  {path.name}"
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "SHA256SUMS.txt"
    ]
    write_immutable_text(output / "SHA256SUMS.txt", "\n".join(sums) + "\n")
    return {
        "result": str(result_path),
        "result_sha256": sha256_file(result_path),
        "status": payload["status"],
        "selected_candidate": selected,
        "candidate_count": len(candidate_entries),
    }


def _rank(candidate: dict[str, Any]) -> tuple[float, ...]:
    summary = candidate["summary"]
    return (
        _metric(summary, "heldout_b_to_a.direct_shortcut_rate"),
        -_metric(summary, "uncommitted_boundary_state.no_negative_swing_rate"),
        -_metric(summary, "heldout_b_to_a.ordered_action_proxy_rate"),
        -_metric(summary, "heldout_b_to_a.work_start_correct_motion_rate"),
        _metric(summary, "heldout_all.heldout_aggregated_action_mae"),
    )


def _gate(name: str, metric: str, operator: str, threshold: float) -> dict[str, Any]:
    return {
        "name": name,
        "metric": metric,
        "operator": operator,
        "threshold": float(threshold),
    }


def _metric(summary: dict[str, Any], path: str) -> float:
    value: Any = summary
    for key in path.split("."):
        value = value[key]
    return float(value)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0])
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


if __name__ == "__main__":
    main()
