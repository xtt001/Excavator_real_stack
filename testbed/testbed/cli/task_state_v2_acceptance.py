"""Freeze and apply task-state-v2 offline acceptance without threshold drift."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from pathlib import Path
from typing import Any

import yaml

from testbed.tasks.real_transition import sha256_file, write_immutable_text

CONTRACT_SCHEMA_V1 = "real_transition_task_state_v2_acceptance_contract_v1"
CONTRACT_SCHEMA_V2 = "real_transition_task_state_v2_acceptance_contract_v2"
CONTRACT_SCHEMA_V3 = "real_transition_task_state_v2_acceptance_contract_v3"
CONTRACT_SCHEMA_V4 = "real_transition_task_state_v2_acceptance_contract_v4"
CONTRACT_SCHEMA = "real_transition_task_state_v2_acceptance_contract_v5"
RESULT_SCHEMA = "real_transition_task_state_v2_acceptance_result_v1"


def main() -> None:
    parser = argparse.ArgumentParser(prog="tb-task-state-v2-acceptance")
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--probe-manifest", type=Path, required=True)
    freeze.add_argument("--baseline-result", type=Path, required=True)
    freeze.add_argument("--parent-result", type=Path, required=True)
    freeze.add_argument("--training-config", type=Path, required=True)
    freeze.add_argument("--smoke-dir", type=Path, required=True)
    freeze.add_argument(
        "--contract-version",
        choices=("v1", "v2", "v3", "v4", "v5"),
        default="v5",
    )
    freeze.add_argument("--output", type=Path, required=True)
    decide = subparsers.add_parser("decide")
    decide.add_argument("--contract", type=Path, required=True)
    decide.add_argument(
        "--candidate-result",
        action="append",
        required=True,
        help="NAME=/absolute/path/to/probe_result.json",
    )
    decide.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        result = freeze_contract(
            probe_manifest=args.probe_manifest,
            baseline_result=args.baseline_result,
            parent_result=args.parent_result,
            training_config=args.training_config,
            smoke_dir=args.smoke_dir,
            output_path=args.output,
            contract_version=str(args.contract_version),
        )
    else:
        candidates = {}
        for raw in args.candidate_result:
            name, separator, path = str(raw).partition("=")
            if not separator or not name or not path:
                raise ValueError("--candidate-result must be NAME=/absolute/path")
            if name in candidates:
                raise ValueError(f"duplicate candidate name {name!r}")
            candidates[name] = Path(path)
        result = decide_candidates(
            contract_path=args.contract,
            candidate_results=candidates,
            output_dir=args.output_dir,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def freeze_contract(
    *,
    probe_manifest: Path | str,
    baseline_result: Path | str,
    parent_result: Path | str,
    training_config: Path | str,
    smoke_dir: Path | str,
    output_path: Path | str,
    contract_version: str = "v2",
) -> dict[str, Any]:
    probe_path = Path(probe_manifest).resolve()
    baseline_path = Path(baseline_result).resolve()
    parent_path = Path(parent_result).resolve()
    config_path = Path(training_config).resolve()
    smoke = Path(smoke_dir).resolve()
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen acceptance: {output}")
    probe = _json(probe_path)
    baseline = _json(baseline_path)
    parent = _json(parent_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("training config must be a YAML mapping")
    probe_sha = sha256_file(probe_path)
    for name, result in (("baseline", baseline), ("parent", parent)):
        if str(result.get("probe_manifest_sha256")) != probe_sha:
            raise ValueError(f"{name} result used a different probe")
    baseline_shortcut = float(
        baseline["summary"]["heldout_b_to_a"]["direct_shortcut_rate"]
    )
    parent_shortcut = float(
        parent["summary"]["heldout_b_to_a"]["direct_shortcut_rate"]
    )
    if baseline_shortcut != 0.75 or parent_shortcut != 0.0:
        raise ValueError(
            "control results do not match the pre-training shortcut reproduction"
        )
    train = dict(config.get("train", {}))
    policy = dict(config.get("policy", {}))
    expected_epochs = {"v1": 100, "v2": 100, "v3": 21, "v4": 16, "v5": 16}[
        contract_version
    ]
    if int(train.get("num_epochs", -1)) != expected_epochs or int(
        train.get("seed", -1)
    ) != 0:
        raise ValueError(
            f"frozen training budget must be {expected_epochs} epochs with seed 0"
        )
    if policy.get("low_dim_keys") != [
        "qpos",
        "qvel",
        "real_transition_task_state_v2",
    ]:
        raise ValueError("training config does not use the task_state_v2 contract")
    smoke_metadata = _json(smoke / "run_metadata.json")
    if smoke_metadata.get("status") != "completed":
        raise ValueError("task_state_v2 smoke is not complete")
    gates_v1 = [
        _gate("bta_no_direct_shortcut", "heldout_b_to_a.direct_shortcut_rate", "<=", 0.0),
        _gate("bta_correct_start_motion", "heldout_b_to_a.work_start_correct_motion_rate", ">=", 0.75),
        _gate("bta_tool_liveness", "heldout_b_to_a.work_start_tool_liveness_rate", ">=", 0.75),
        _gate("bta_positive_excursion", "heldout_b_to_a.outbound_positive_swing_rate", ">=", 0.75),
        _gate("bta_bucket_liveness", "heldout_b_to_a.bucket_tool_liveness_rate", ">=", 0.875),
        _gate("bta_return", "heldout_b_to_a.return_negative_swing_rate", ">=", 0.875),
        _gate("bta_ordered_proxy", "heldout_b_to_a.ordered_action_proxy_rate", ">=", 0.625),
        _gate("other_no_shortcut", "heldout_other.direct_shortcut_rate", "<=", 0.0),
        _gate("other_start_motion", "heldout_other.work_start_correct_motion_rate", ">=", 0.75),
        _gate("other_positive_excursion", "heldout_other.outbound_positive_swing_rate", ">=", 0.75),
        _gate("other_bucket_liveness", "heldout_other.bucket_tool_liveness_rate", ">=", 0.90),
        _gate("other_return", "heldout_other.return_negative_swing_rate", ">=", 0.70),
        _gate("other_ordered_proxy", "heldout_other.ordered_action_proxy_rate", ">=", 0.50),
        _gate("heldout_mae", "heldout_all.heldout_aggregated_action_mae", "<=", 0.145),
        _gate("heldout_sign", "heldout_all.heldout_effective_sign_agreement", ">=", 0.70),
        _gate("task_factual_work_no_shortcut", "b_to_a_ready_pair_factual_qvel.work_no_negative_rate", ">=", 0.875),
        _gate("task_factual_work_liveness", "b_to_a_ready_pair_factual_qvel.work_any_axis_effective_rate", ">=", 0.75),
        _gate("task_factual_return", "b_to_a_ready_pair_factual_qvel.return_negative_rate", ">=", 0.875),
        _gate("task_factual_pair", "b_to_a_ready_pair_factual_qvel.task_pair_hit_rate", ">=", 0.75),
        _gate("zero_qvel_work_no_shortcut", "b_to_a_ready_pair_zero_qvel.work_no_negative_rate", ">=", 0.875),
        _gate("zero_qvel_work_liveness", "b_to_a_ready_pair_zero_qvel.work_any_axis_effective_rate", ">=", 0.625),
        _gate("zero_qvel_return", "b_to_a_ready_pair_zero_qvel.return_negative_rate", ">=", 0.875),
        _gate("zero_qvel_pair", "b_to_a_ready_pair_zero_qvel.task_pair_hit_rate", ">=", 0.625),
    ]
    if contract_version == "v1":
        gates = gates_v1
        schema = CONTRACT_SCHEMA_V1
        interpretation = (
            "State-hold invariance under qvel-to-zero is deliberately not a gate. "
            "The replacement tests ask whether a stopped B-ready state still "
            "avoids the A shortcut and produces useful work, and whether an "
            "explicit return state can initiate return from the same observation."
        )
        selection_rule = [
            "discard every checkpoint that fails any frozen gate",
            "lowest heldout_b_to_a.direct_shortcut_rate",
            "highest b_to_a_ready_pair_factual_qvel.task_pair_hit_rate",
            "highest b_to_a_ready_pair_zero_qvel.task_pair_hit_rate",
            "highest heldout_b_to_a.ordered_action_proxy_rate",
            "highest heldout_other.ordered_action_proxy_rate",
            "lowest heldout_all.heldout_aggregated_action_mae",
            "earlier checkpoint grid order as final tie-break",
        ]
    elif contract_version in {"v2", "v3", "v4", "v5"}:
        schema = {
            "v2": CONTRACT_SCHEMA_V2,
            "v3": CONTRACT_SCHEMA_V3,
            "v4": CONTRACT_SCHEMA_V4,
            "v5": CONTRACT_SCHEMA,
        }[contract_version]
        parent_all = parent["summary"]["heldout_all"]
        gates = [
            _gate("bta_no_direct_shortcut", "heldout_b_to_a.direct_shortcut_rate", "<=", 0.0),
            _gate("bta_correct_start_motion", "heldout_b_to_a.work_start_correct_motion_rate", ">=", 0.75),
            _gate("bta_tool_liveness", "heldout_b_to_a.work_start_tool_liveness_rate", ">=", 0.75),
            _gate("bta_positive_excursion", "heldout_b_to_a.outbound_positive_swing_rate", ">=", 0.75),
            _gate("bta_bucket_liveness", "heldout_b_to_a.bucket_tool_liveness_rate", ">=", 0.875),
            _gate("bta_return", "heldout_b_to_a.return_negative_swing_rate", ">=", 0.875),
            _gate("bta_return_crossing", "heldout_b_to_a.return_ready_crossing_negative_swing_rate", ">=", 0.875),
            _gate("bta_ordered_proxy", "heldout_b_to_a.ordered_action_proxy_rate", ">=", 0.625),
            _gate("uncommitted_boundary_no_return", "uncommitted_boundary_state.no_negative_swing_rate", ">=", 0.95),
            _gate("other_no_shortcut", "heldout_other.direct_shortcut_rate", "<=", 0.0),
            _gate("other_start_motion", "heldout_other.work_start_correct_motion_rate", ">=", 0.75),
            _gate("other_positive_excursion", "heldout_other.outbound_positive_swing_rate", ">=", 0.75),
            _gate("other_bucket_liveness", "heldout_other.bucket_tool_liveness_rate", ">=", 0.90),
            _gate("other_return", "heldout_other.return_negative_swing_rate", ">=", 0.70),
            _gate("other_ordered_proxy", "heldout_other.ordered_action_proxy_rate", ">=", 0.50),
            _gate("heldout_mae", "heldout_all.heldout_aggregated_action_mae", "<=", float(parent_all["heldout_aggregated_action_mae"]) * 1.15),
            _gate("heldout_sign", "heldout_all.heldout_effective_sign_agreement", ">=", float(parent_all["heldout_effective_sign_agreement"]) - 0.05),
            _gate("task_factual_work_no_shortcut", "b_to_a_ready_pair_factual_qvel.work_no_negative_rate", ">=", 0.875),
            _gate("task_factual_work_liveness", "b_to_a_ready_pair_factual_qvel.work_any_axis_effective_rate", ">=", 0.75),
            _gate("task_factual_changes_raw_chunk", "b_to_a_ready_pair_factual_qvel.task_raw_chunk_l1_delta_mean", ">=", 0.02),
            _gate("zero_qvel_work_no_shortcut", "b_to_a_ready_pair_zero_qvel.work_no_negative_rate", ">=", 0.875),
            _gate("zero_qvel_work_liveness", "b_to_a_ready_pair_zero_qvel.work_any_axis_effective_rate", ">=", 0.50),
            _gate("zero_qvel_task_changes_raw_chunk", "b_to_a_ready_pair_zero_qvel.task_raw_chunk_l1_delta_mean", ">=", 0.02),
            _gate("qvel_changes_work_raw_chunk", "b_to_a_ready_qvel_sensitivity.work_raw_chunk_l1_delta_mean", ">=", 0.005),
            _gate("qvel_changes_return_raw_chunk", "b_to_a_ready_qvel_sensitivity.return_raw_chunk_l1_delta_mean", ">=", 0.005),
        ]
        interpretation = (
            "The return token is not required to manufacture an immediate return "
            "from the contradictory stopped-ready observation. Token authority is "
            "tested as a raw-chunk intervention, while correct return is gated on "
            "recorded return observations, including the factual pass through B. "
            "qvel authority is tested by paired raw-chunk sensitivity plus useful "
            "work at factual and zero-speed ready states."
        )
        selection_rule = [
            "discard every checkpoint that fails any frozen gate",
            "lowest heldout_b_to_a.direct_shortcut_rate",
            "highest heldout_b_to_a.ordered_action_proxy_rate",
            "highest heldout_b_to_a.work_start_correct_motion_rate",
            "highest heldout_b_to_a.return_ready_crossing_negative_swing_rate",
            "highest heldout_other.ordered_action_proxy_rate",
            "lowest heldout_all.heldout_aggregated_action_mae",
            "earlier checkpoint grid order as final tie-break",
        ]
    else:
        raise ValueError("contract_version must be v1, v2, v3, v4 or v5")
    non_gating_diagnostics = [
        "tail_all_axes_idle_rate",
        "boundary_all_axes_idle",
        "field_normal_synthetic_non_gating",
        "field_abnormal_synthetic_non_gating",
    ]
    if contract_version == "v1":
        non_gating_diagnostics.append("task_raw_chunk_l1_delta_mean")
        smoke_parameter_check = (
            "346 equal-shape tensors unchanged; both expanded projections "
            "retain source columns and zero new columns; all six low-head "
            "tensors update during the second smoke epoch"
        )
    else:
        non_gating_diagnostics.extend(
            [
                "counterfactual_ready_return_negative_rate",
                "counterfactual_ready_task_pair_hit_rate",
            ]
        )
        smoke_parameter_check = (
            "source qpos and qvel low-head columns are semantically remapped; "
            "old target-side maps only to gated_next_target; old always-on "
            "goal_active is folded into the first-layer bias; new event columns "
            "start at zero; the frozen visual residual remains qpos-only"
        )
    loss_contract = (
        "standard ACT L1 plus the existing deadzone same-direction loss; "
        "no new task contrast loss"
        if contract_version not in {"v3", "v4", "v5"}
        else (
            "standard ACT L1 plus the existing deadzone same-direction loss "
            "and one factual uncommitted-state negative-swing guard with "
            + (
                "weight 1.0"
                if contract_version == "v3"
                else "weight 1.0 and guard margin 0.06"
                if contract_version == "v4"
                else "weight 1.0 and per-sample worst-query reduction"
            )
        )
    )
    payload = {
        "schema": schema,
        "path": str(output),
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "frozen_before_long_training": True,
        "probe": {"path": str(probe_path), "sha256": probe_sha},
        "controls": {
            "accepted_epoch199": {
                "path": str(baseline_path),
                "sha256": sha256_file(baseline_path),
                "b_to_a_shortcut_rate": baseline_shortcut,
            },
            "direct_parent_epoch259": {
                "path": str(parent_path),
                "sha256": sha256_file(parent_path),
                "b_to_a_shortcut_rate": parent_shortcut,
            },
        },
        "training": {
            "config": str(config_path),
            "config_sha256": sha256_file(config_path),
            "seed": 0,
            "epochs": expected_epochs,
            "learning_rate": float(train["lr"]),
            "batch_size": int(train["batch_size"]),
            "source_checkpoint": str(
                train.get("resume_ckpt", train.get("warm_start_ckpt"))
            ),
            "source_checkpoint_sha256": sha256_file(
                Path(str(train.get("resume_ckpt", train.get("warm_start_ckpt"))))
            ),
            "source_mode": (
                "resume_with_reset_best"
                if train.get("resume_ckpt")
                else str(train["warm_start_mode"])
            ),
            "loss_contract": loss_contract,
            "checkpoint_grid": (
                [
                    "epoch_1",
                    "epoch_4",
                    "epoch_9",
                    "epoch_14",
                    "epoch_19",
                    "training_last_epoch20",
                    "training_best",
                ]
                if contract_version == "v3"
                else [
                    "epoch_6",
                    "epoch_8",
                    "epoch_11",
                    "epoch_14",
                    "training_last_epoch15",
                    "training_best",
                ]
                if contract_version in {"v4", "v5"}
                else [
                    "epoch_0",
                    "epoch_19",
                    "epoch_39",
                    "epoch_59",
                    "epoch_79",
                    "epoch_99",
                    "training_best",
                ]
            ),
        },
        "smoke": {
            "directory": str(smoke),
            "resolved_config_sha256": sha256_file(smoke / "resolved_config.yaml"),
            "dataset_stats_sha256": sha256_file(smoke / "dataset_stats.pkl"),
            "run_metadata_sha256": sha256_file(smoke / "run_metadata.json"),
            "status": "completed",
            "frozen_residual_tensor_check": smoke_parameter_check,
        },
        "gates": gates,
        "non_gating_diagnostics": non_gating_diagnostics,
        "selection_rule": selection_rule,
        "interpretation": interpretation,
        "maximum_promotion": "NEW_OFFLINE_CANDIDATE_REQUIRES_SHADOW_ZERO_AND_CONTROLLED_FIELD_VALIDATION",
        "no_threshold_relaxation_after_results": True,
        "evidence_boundary": probe["evidence_boundary"],
    }
    written = write_immutable_text(
        output,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return {"contract": str(written), "sha256": sha256_file(written)}


def decide_candidates(
    *,
    contract_path: Path | str,
    candidate_results: dict[str, Path],
    output_dir: Path | str,
) -> dict[str, Any]:
    contract_file = Path(contract_path).resolve()
    contract = _json(contract_file)
    if contract.get("schema") not in {
        CONTRACT_SCHEMA_V1,
        CONTRACT_SCHEMA_V2,
        CONTRACT_SCHEMA_V3,
        CONTRACT_SCHEMA_V4,
        CONTRACT_SCHEMA,
    }:
        raise ValueError("task_state_v2 acceptance contract schema mismatch")
    if sha256_file(Path(contract["probe"]["path"])) != contract["probe"]["sha256"]:
        raise ValueError("frozen probe changed")
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite acceptance output: {output}")
    rows = []
    summaries = {}
    for grid_index, (name, path_raw) in enumerate(candidate_results.items()):
        path = Path(path_raw).resolve()
        result = _json(path)
        if str(result.get("probe_manifest_sha256")) != str(
            contract["probe"]["sha256"]
        ):
            raise ValueError(f"candidate {name} used a different probe")
        if result.get("interface") != "task_state_v2":
            raise ValueError(f"candidate {name} does not use task_state_v2")
        gates = []
        for gate in contract["gates"]:
            value = _metric(result["summary"], str(gate["metric"]))
            threshold = float(gate["threshold"])
            operator = str(gate["operator"])
            passed = value >= threshold if operator == ">=" else value <= threshold
            gates.append(
                {
                    "candidate": name,
                    "gate": str(gate["name"]),
                    "metric": str(gate["metric"]),
                    "operator": operator,
                    "threshold": threshold,
                    "value": value,
                    "passed": bool(passed),
                }
            )
        passed_all = all(row["passed"] for row in gates)
        rows.extend(gates)
        summaries[name] = {
            "result": str(path),
            "result_sha256": sha256_file(path),
            "checkpoint": result["checkpoint"],
            "checkpoint_sha256": result["checkpoint_sha256"],
            "passes_all_gates": passed_all,
            "failed_gates": [row["gate"] for row in gates if not row["passed"]],
            "summary": result["summary"],
            "grid_index": grid_index,
        }
    passing = [name for name, value in summaries.items() if value["passes_all_gates"]]
    ranked = sorted(
        passing, key=lambda name: _rank_key(summaries[name], contract=contract)
    )
    selected = None if not ranked else ranked[0]
    payload = {
        "schema": RESULT_SCHEMA,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "contract": str(contract_file),
        "contract_sha256": sha256_file(contract_file),
        "candidate_summaries": summaries,
        "passing_candidates_ranked": ranked,
        "selected_candidate": selected,
        "status": (
            "NO_OFFLINE_CANDIDATE"
            if selected is None
            else "NEW_OFFLINE_CANDIDATE_REQUIRES_SHADOW_ZERO"
        ),
        "maximum_promotion": contract["maximum_promotion"],
        "evidence_boundary": contract["evidence_boundary"],
    }
    output.mkdir(parents=True)
    result_path = write_immutable_text(
        output / "acceptance_result.json",
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    _write_csv(output / "gate_results.csv", rows)
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
    }


def _rank_key(
    value: dict[str, Any], *, contract: dict[str, Any]
) -> tuple[float, ...]:
    summary = value["summary"]
    if contract.get("schema") in {
        CONTRACT_SCHEMA_V2,
        CONTRACT_SCHEMA_V3,
        CONTRACT_SCHEMA_V4,
        CONTRACT_SCHEMA,
    }:
        return (
            _metric(summary, "heldout_b_to_a.direct_shortcut_rate"),
            -_metric(summary, "heldout_b_to_a.ordered_action_proxy_rate"),
            -_metric(summary, "heldout_b_to_a.work_start_correct_motion_rate"),
            -_metric(
                summary,
                "heldout_b_to_a.return_ready_crossing_negative_swing_rate",
            ),
            -_metric(summary, "heldout_other.ordered_action_proxy_rate"),
            _metric(summary, "heldout_all.heldout_aggregated_action_mae"),
            float(value["grid_index"]),
        )
    return (
        _metric(summary, "heldout_b_to_a.direct_shortcut_rate"),
        -_metric(summary, "b_to_a_ready_pair_factual_qvel.task_pair_hit_rate"),
        -_metric(summary, "b_to_a_ready_pair_zero_qvel.task_pair_hit_rate"),
        -_metric(summary, "heldout_b_to_a.ordered_action_proxy_rate"),
        -_metric(summary, "heldout_other.ordered_action_proxy_rate"),
        _metric(summary, "heldout_all.heldout_aggregated_action_mae"),
        float(value["grid_index"]),
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
    fields = sorted({key for row in rows for key in row})
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
