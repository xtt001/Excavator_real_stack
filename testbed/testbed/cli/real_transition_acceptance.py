"""Apply the frozen pre-field gates to planner and state-hold reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(prog="tb-real-transition-acceptance")
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--planner-report", type=Path, required=True)
    parser.add_argument("--state-hold-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"output already exists: {args.output}")
    result = evaluate_acceptance(
        contract=json.loads(args.contract.read_text(encoding="utf-8")),
        bundle_dir=args.bundle_dir.resolve(),
        planner_report=json.loads(args.planner_report.read_text(encoding="utf-8")),
        state_hold_report=json.loads(
            args.state_hold_report.read_text(encoding="utf-8")
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "status": result["status"]}))


def evaluate_acceptance(
    *,
    contract: dict[str, Any],
    bundle_dir: Path,
    planner_report: dict[str, Any],
    state_hold_report: dict[str, Any],
) -> dict[str, Any]:
    contract_schema = str(contract.get("schema", ""))
    if contract_schema not in {
        "real_transition_target_release_acceptance_contract_v1",
        "real_transition_target_release_acceptance_contract_v2",
    }:
        raise ValueError("acceptance contract schema is invalid")
    planner_bundle = _planner_bundle(planner_report, bundle_dir)
    authoritative_mode = str(contract["authoritative_runtime_mode"])
    mode = next(
        (
            value
            for value in planner_bundle["modes"]
            if value["mode"] == authoritative_mode
        ),
        None,
    )
    if mode is None:
        raise ValueError(
            f"planner report has no authoritative mode {authoritative_mode!r}"
        )
    state_bundle = Path(str(state_hold_report["bundle_dir"])).resolve()
    if state_bundle != bundle_dir:
        raise ValueError(
            f"state-hold bundle mismatch: expected {bundle_dir}, got {state_bundle}"
        )

    planner_gates = dict(contract["planner_open_loop_gates"])
    hold_gates = dict(contract["state_hold_gates"])
    checks: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {"planner": {}, "state_hold": {}}
    for split in ("validation", "locked_test"):
        cycles = [
            cycle
            for run in mode["runs"]
            if run["split"] == split
            for cycle in run["cycles"]
        ]
        split_metrics = _planner_split_metrics(cycles)
        metrics["planner"][split] = split_metrics
        _check_max(
            checks,
            f"{split}.safe_action_mae",
            split_metrics["safe_action_mae"],
            planner_gates["safe_action_mae_max"],
        )
        _check_min(
            checks,
            f"{split}.policy_action_sign_agreement_rate",
            split_metrics["policy_action_sign_agreement_rate"],
            planner_gates["policy_action_sign_agreement_rate_min"],
        )
        _check_min(
            checks,
            f"{split}.supported_target_release_pair_hit_rate",
            split_metrics["supported_target_release_pair_hit_rate"],
            planner_gates["supported_target_release_pair_hit_rate_min"],
        )
        _check_min(
            checks,
            f"{split}.B_release_idle_rate",
            split_metrics["B_release_idle_rate"],
            planner_gates["B_release_idle_rate_min"],
        )
        _check_min(
            checks,
            f"{split}.dig_positive_effective_rate",
            split_metrics["dig_positive_effective_rate"],
            planner_gates[f"{split}_dig_positive_effective_rate_min"],
        )
        _check_min(
            checks,
            f"{split}.return_negative_effective_rate",
            split_metrics["return_negative_effective_rate"],
            planner_gates[f"{split}_return_negative_effective_rate_min"],
        )
        _check_max(
            checks,
            f"{split}.policy_error_count",
            split_metrics["policy_error_count"],
            planner_gates["policy_error_count_max"],
        )
        _check_max(
            checks,
            f"{split}.guard_trigger_count",
            split_metrics["guard_trigger_count"],
            planner_gates["guard_trigger_count_max"],
        )
        _check_equal(
            checks,
            f"{split}.all_policy_actions_finite",
            split_metrics["all_policy_actions_finite"],
            planner_gates["all_policy_actions_finite"],
        )
        _check_equal(
            checks,
            f"{split}.all_safe_actions_finite",
            split_metrics["all_safe_actions_finite"],
            planner_gates["all_safe_actions_finite"],
        )

        overall = _state_summary(state_hold_report, split, "overall")
        startup = _state_summary(state_hold_report, split, "startup")
        metrics["state_hold"][split] = {
            "overall": overall,
            "startup": startup,
        }
        if contract_schema.endswith("_v1"):
            _check_min(
                checks,
                f"{split}.state_hold.same_direction_within_5_rate",
                overall["same_direction_within_5_rate"],
                hold_gates["same_direction_within_5_rate_min"],
            )
            overall_min = hold_gates["same_direction_within_20_rate_min"]
            startup_min = hold_gates[
                "startup_same_direction_within_20_rate_min"
            ]
            wrong_max = hold_gates["query0_wrong_effective_count_max"]
        else:
            overall_min = hold_gates[
                f"{split}_same_direction_within_20_rate_min"
            ]
            startup_min = hold_gates[
                f"{split}_startup_same_direction_within_20_rate_min"
            ]
            wrong_max = hold_gates[
                f"{split}_query0_wrong_effective_count_max"
            ]
        _check_min(
            checks,
            f"{split}.state_hold.same_direction_within_20_rate",
            overall["same_direction_within_20_rate"],
            overall_min,
        )
        _check_min(
            checks,
            f"{split}.state_hold.startup_within_20_rate",
            startup["same_direction_within_20_rate"],
            startup_min,
        )
        _check_max(
            checks,
            f"{split}.state_hold.query0_wrong_effective_count",
            overall["query0_wrong_effective_count"],
            wrong_max,
        )
    return {
        "schema": "real_transition_target_release_acceptance_result_v1",
        "bundle_dir": str(bundle_dir),
        "status": "PASS" if all(check["passed"] for check in checks) else "FAIL",
        "authoritative_runtime_mode": authoritative_mode,
        "metrics": metrics,
        "checks": checks,
        "failed_checks": [check["name"] for check in checks if not check["passed"]],
        "evidence_boundary": contract["evidence_boundary"],
    }


def _planner_bundle(report: dict[str, Any], bundle: Path) -> dict[str, Any]:
    for value in report.get("reports", []):
        if Path(str(value["bundle_dir"])).resolve() == bundle:
            return value
    raise ValueError(f"planner report has no bundle {bundle}")


def _planner_split_metrics(cycles: list[dict[str, Any]]) -> dict[str, Any]:
    if not cycles:
        raise ValueError("planner split contains no cycles")
    probes = [cycle["supported_target_release_probe"] for cycle in cycles]
    probe_count = sum(int(probe["sample_count"]) for probe in probes)
    if probe_count <= 0:
        raise ValueError("planner split contains no supported target-release probes")
    return {
        "cycle_count": len(cycles),
        "reference_complete_cycle_count": sum(
            cycle["status"] == "REFERENCE_CYCLE_COMPLETE" for cycle in cycles
        ),
        "safe_action_mae": _mean(cycles, "safe_action_mae"),
        "policy_action_sign_agreement_rate": _weighted_phase(
            cycles, "policy_action_sign_agreement_rate", "steps"
        ),
        "dig_positive_effective_rate": _weighted_phase(
            cycles, "policy_dig_positive_effective_rate", "dig"
        ),
        "return_negative_effective_rate": _weighted_phase(
            cycles, "policy_return_negative_effective_rate", "return_approach"
        ),
        "release_idle_rate": _weighted_phase(
            cycles, "policy_release_idle_rate", "release"
        ),
        "supported_target_release_pair_hit_rate": float(
            sum(int(probe["pair_hit_count"]) for probe in probes) / probe_count
        ),
        "B_release_idle_rate": float(
            sum(sum(int(sign) == 0 for sign in probe["B_signs"]) for probe in probes)
            / probe_count
        ),
        "policy_error_count": sum(int(cycle["policy_error_count"]) for cycle in cycles),
        "guard_trigger_count": sum(int(cycle["guard_trigger_count"]) for cycle in cycles),
        "all_policy_actions_finite": all(
            bool(cycle["all_policy_actions_finite"]) for cycle in cycles
        ),
        "all_safe_actions_finite": all(
            bool(cycle["all_safe_actions_finite"]) for cycle in cycles
        ),
    }


def _weighted_phase(
    cycles: list[dict[str, Any]], metric: str, weight: str
) -> float:
    weights = [
        int(cycle["phase_steps"][weight]) if weight in cycle["phase_steps"] else int(cycle[weight])
        for cycle in cycles
    ]
    denominator = sum(weights)
    if denominator <= 0:
        return 0.0
    return float(
        sum(float(cycle[metric]) * value for cycle, value in zip(cycles, weights, strict=True))
        / denominator
    )


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return float(sum(float(row[key]) for row in rows) / len(rows))


def _state_summary(
    report: dict[str, Any], split: str, group: str
) -> dict[str, Any]:
    for value in report["summary"]:
        if value["split"] == split and value["group"] == group:
            return value
    raise ValueError(f"state-hold report lacks {split}/{group} summary")


def _check_min(
    checks: list[dict[str, Any]], name: str, actual: float, threshold: float
) -> None:
    checks.append(
        {
            "name": name,
            "operator": ">=",
            "actual": actual,
            "threshold": threshold,
            "passed": bool(actual >= threshold),
        }
    )


def _check_max(
    checks: list[dict[str, Any]], name: str, actual: float, threshold: float
) -> None:
    checks.append(
        {
            "name": name,
            "operator": "<=",
            "actual": actual,
            "threshold": threshold,
            "passed": bool(actual <= threshold),
        }
    )


def _check_equal(
    checks: list[dict[str, Any]], name: str, actual: Any, expected: Any
) -> None:
    checks.append(
        {
            "name": name,
            "operator": "==",
            "actual": actual,
            "threshold": expected,
            "passed": bool(actual == expected),
        }
    )


if __name__ == "__main__":
    main()
