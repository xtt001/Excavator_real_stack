"""Consolidate frozen task-state-v2 experiment results and plot trade-offs."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from pathlib import Path
from typing import Any

from testbed.tasks.real_transition import sha256_file, write_immutable_text


def main() -> None:
    parser = argparse.ArgumentParser(prog="tb-task-state-v2-summary")
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = build_summary(
        experiment_root=args.experiment_root, output_dir=args.output_dir
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def build_summary(
    *, experiment_root: Path | str, output_dir: Path | str
) -> dict[str, Any]:
    root = Path(experiment_root).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite summary output: {output}")
    acceptance_paths = [
        root / f"acceptance_v{version}" / "acceptance_result.json"
        for version in (1, 2, 3, 4, 5)
    ]
    result_paths = {
        "baseline_epoch199": root / "baseline_epoch199_probe_v2" / "probe_result.json",
        "parent_epoch259": root / "parent_epoch259_probe_v2" / "probe_result.json",
        "semantic_epoch0": root / "semantic_warm_epoch0_probe_v2" / "probe_result.json",
        "semantic_best": root / "semantic_warm_training_best_probe_v2" / "probe_result.json",
        "mean_guard_best": root / "uncommitted_guard_training_best_probe_v2" / "probe_result.json",
        "margin_guard_best": root / "uncommitted_margin_training_best_probe_v2" / "probe_result.json",
        "worst_query_best": root / "worst_query_guard_training_best_probe_v2" / "probe_result.json",
    }
    acceptances = [_json(path) for path in acceptance_paths]
    rows = []
    for version, payload in enumerate(acceptances, start=1):
        for candidate, spec in payload["candidate_summaries"].items():
            rows.append(
                _row(
                    experiment=f"acceptance_v{version}",
                    candidate=str(candidate),
                    result=dict(spec),
                )
            )
    controls = {}
    for name, path in result_paths.items():
        result = _json(path)
        controls[name] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "summary": result["summary"],
        }
    output.mkdir(parents=True)
    table_path = output / "checkpoint_comparison.csv"
    _write_csv(table_path, rows)
    payload = {
        "schema": "real_transition_task_state_v2_experiment_summary_v1",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "experiment_root": str(root),
        "acceptance_results": [
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "status": acceptance["status"],
                "selected_candidate": acceptance["selected_candidate"],
            }
            for path, acceptance in zip(acceptance_paths, acceptances, strict=True)
        ],
        "representative_results": controls,
        "checkpoint_row_count": len(rows),
        "conclusion": {
            "accepted_candidate": None,
            "accepted_epoch199_b_to_a_shortcut_rate": 0.75,
            "all_task_state_candidate_shortcut_rate": 0.0,
            "best_uncommitted_no_negative_rate": 27 / 29,
            "required_uncommitted_no_negative_rate": 0.95,
            "persistent_uncommitted_fail_episode_ids": [52, 59],
            "tradeoff": (
                "uncommitted suppression improved from 21/29 to 27/29, but "
                "no checkpoint simultaneously preserved the frozen B-to-A and "
                "zero-qvel start-liveness gates"
            ),
        },
        "evidence_boundary": (
            "Recorded state/history-conditioned open-loop replay only. No "
            "policy-driven future observations, hydraulic response, soil effect, "
            "or physical closed-loop validation."
        ),
    }
    summary_path = write_immutable_text(
        output / "experiment_summary.json",
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    figure_path = output / "behaviour_tradeoff.png"
    _plot(controls, figure_path)
    sums = [
        f"{sha256_file(path)}  {path.name}"
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "SHA256SUMS.txt"
    ]
    write_immutable_text(output / "SHA256SUMS.txt", "\n".join(sums) + "\n")
    return {
        "summary": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "table": str(table_path),
        "figure": str(figure_path),
    }


def _row(*, experiment: str, candidate: str, result: dict[str, Any]) -> dict[str, Any]:
    summary = result["summary"]
    bta = summary["heldout_b_to_a"]
    other = summary["heldout_other"]
    boundary = summary.get("uncommitted_boundary_state", {})
    factual = summary["b_to_a_ready_pair_factual_qvel"]
    zero = summary["b_to_a_ready_pair_zero_qvel"]
    return {
        "experiment": experiment,
        "candidate": candidate,
        "passes_all_gates": bool(result["passes_all_gates"]),
        "failed_gates": ";".join(result["failed_gates"]),
        "bta_direct_shortcut_rate": bta["direct_shortcut_rate"],
        "bta_start_correct_motion_rate": bta["work_start_correct_motion_rate"],
        "bta_ordered_action_proxy_rate": bta["ordered_action_proxy_rate"],
        "bta_return_crossing_negative_rate": bta.get(
            "return_ready_crossing_negative_swing_rate"
        ),
        "other_start_correct_motion_rate": other["work_start_correct_motion_rate"],
        "uncommitted_no_negative_rate": boundary.get("no_negative_swing_rate"),
        "factual_ready_liveness_rate": factual["work_any_axis_effective_rate"],
        "zero_qvel_ready_liveness_rate": zero["work_any_axis_effective_rate"],
        "heldout_action_mae": summary["heldout_all"][
            "heldout_aggregated_action_mae"
        ],
        "heldout_effective_sign_agreement": summary["heldout_all"][
            "heldout_effective_sign_agreement"
        ],
        "checkpoint": result["checkpoint"],
        "checkpoint_sha256": result["checkpoint_sha256"],
        "probe_result": result["result"],
        "probe_result_sha256": result["result_sha256"],
    }


def _plot(results: dict[str, dict[str, Any]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [
        "baseline_epoch199",
        "parent_epoch259",
        "semantic_epoch0",
        "mean_guard_best",
        "margin_guard_best",
        "worst_query_best",
    ]
    labels = ["E199", "Parent259", "Semantic E0", "Mean guard", "Margin", "Worst-query"]
    shortcut = [
        results[name]["summary"]["heldout_b_to_a"]["direct_shortcut_rate"]
        for name in names
    ]
    boundary = [
        results[name]["summary"]["uncommitted_boundary_state"][
            "no_negative_swing_rate"
        ]
        for name in names
    ]
    liveness = [
        results[name]["summary"]["heldout_b_to_a"][
            "work_start_correct_motion_rate"
        ]
        for name in names
    ]
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), constrained_layout=True)
    series = (
        (shortcut, "B→A direct shortcut", 0.0, "lower is better"),
        (boundary, "Uncommitted no-return", 0.95, "higher is better"),
        (liveness, "B→A correct start motion", 0.75, "higher is better"),
    )
    for axis, (values, title, threshold, note) in zip(axes, series, strict=True):
        bars = axis.bar(range(len(labels)), values, color="#4c78a8")
        axis.axhline(threshold, color="#e45756", linestyle="--", linewidth=1.5)
        axis.set_ylim(0.0, 1.05)
        axis.set_title(title)
        axis.set_ylabel("rate")
        axis.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
        axis.text(0.02, 0.98, note, transform=axis.transAxes, va="top", fontsize=9)
        for bar, value in zip(bars, values, strict=True):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                min(1.02, float(value) + 0.025),
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Task-state-v2 offline behaviour trade-off")
    figure.savefig(path, dpi=180)
    plt.close(figure)


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
