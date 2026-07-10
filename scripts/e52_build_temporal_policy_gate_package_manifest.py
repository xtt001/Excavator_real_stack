#!/usr/bin/env python3
"""Build a package manifest for the E51 causal temporal gate candidate."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.e36_build_policy_gate_package_manifest import build_artifact_entry, verify_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-id", default="E52")
    parser.add_argument("--base-package-manifest", type=Path, required=True)
    parser.add_argument("--temporal-direction-dir", type=Path, required=True)
    parser.add_argument("--combined-candidate-dir", type=Path, required=True)
    parser.add_argument("--runtime-smoke-dir", type=Path, required=True)
    parser.add_argument("--runtime-deadzone-gate-dir", type=Path, required=True)
    parser.add_argument("--runtime-window-eval-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    base_manifest = _read_json(args.base_package_manifest)
    runtime_summary = _read_json(args.runtime_smoke_dir / "full_act_temporal_gate_smoke_summary.json")
    selected = dict(runtime_summary.get("selected_gates", {}))
    selected["phase_gate"] = str(selected.get("phase_gate", ""))
    selected["gohome_gate"] = str(selected.get("gohome_gate", ""))
    selected["temporal_direction_gate"] = (
        f"tdir_t{int(round(float(selected.get('direction_threshold', 0.0)) * 100)):02d}"
        f"_s{int(round(float(selected.get('direction_inactive_scale', 0.0)) * 100)):02d}"
    )

    artifacts = _base_artifacts(base_manifest) + _temporal_artifacts(args)
    evidence = {
        "base_package": _compact_base_package(base_manifest),
        "combined_candidate": _read_json(args.combined_candidate_dir / "combined_candidate_summary.json"),
        "runtime_gate_smoke": _compact_runtime_summary(runtime_summary),
        "startup_tail_gate": _read_json(args.runtime_deadzone_gate_dir / "tail_stability_summary.json"),
        "window_eval": _read_json(args.runtime_window_eval_dir / "deadzone_window_summary.json"),
    }
    manifest = build_temporal_manifest(
        candidate_id=str(args.candidate_id),
        selected_gates=selected,
        artifacts=artifacts,
        evidence=evidence,
    )
    verify_report = verify_manifest(manifest)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_dir / "candidate_package_manifest.json", manifest)
    _write_json(args.output_dir / "candidate_package_verify.json", verify_report)
    (args.output_dir / "field_smoke_checklist.md").write_text(
        build_field_smoke_checklist(manifest, verify_report),
        encoding="utf-8",
    )
    if not verify_report["ok"]:
        raise SystemExit(f"candidate package verification failed: {verify_report['errors']}")
    print(f"Temporal candidate package manifest: {args.output_dir / 'candidate_package_manifest.json'}")
    print(f"Temporal candidate package verification: {args.output_dir / 'candidate_package_verify.json'}")


def build_temporal_manifest(
    *,
    candidate_id: str,
    selected_gates: dict[str, Any],
    artifacts: list[tuple[str, Path]] | list[dict[str, Any]],
    evidence: dict[str, Any],
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for item in artifacts:
        if isinstance(item, dict):
            entries.append(dict(item))
        else:
            name, path = item
            entries.append(build_artifact_entry(str(name), Path(path)))
    return {
        "schema_version": 1,
        "candidate_id": str(candidate_id),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "offline package for ACT policy plus phase, causal temporal direction, and gohome request gates",
        "selected_gates": dict(selected_gates),
        "artifacts": entries,
        "evidence": dict(evidence),
    }


def build_field_smoke_checklist(manifest: dict[str, Any], verify_report: dict[str, Any]) -> str:
    gates = dict(manifest.get("selected_gates", {}))
    evidence = dict(manifest.get("evidence", {}))
    runtime = dict(evidence.get("runtime_gate_smoke", {}))
    startup_tail = dict(evidence.get("startup_tail_gate", {}))
    startup_rows = list(startup_tail.get("startup_aggregate", []))
    tail_rows = list(startup_tail.get("tail_aggregate", []))
    startup = dict(startup_rows[0]) if startup_rows else {}
    tail = dict(tail_rows[0]) if tail_rows else {}
    return "\n".join(
        [
            "# E52 Field Smoke Checklist",
            "",
            f"- Candidate: `{manifest.get('candidate_id', '')}`",
            f"- Manifest verification: `{verify_report.get('ok', False)}`",
            f"- Base phase gate: `{gates.get('phase_gate', '')}`",
            f"- Temporal direction gate: `{gates.get('temporal_direction_gate', '')}`",
            f"- Gohome gate: `{gates.get('gohome_gate', '')}`",
            f"- Full-ACT temporal action MAE: `{runtime.get('temporal_direction_action_mae', '')}`",
            f"- Startup effective / same / extra: `{startup.get('mean_policy_any_effective_pct', '')}` / `{startup.get('mean_same_axis_dir_pct_of_expert_effective', '')}` / `{startup.get('mean_extra_or_wrong_pct_of_policy_effective', '')}`",
            f"- Tail effective frames: `{tail.get('total_policy_effective_frames', '')}`",
            f"- Gohome pre-tail FP episodes: `{runtime.get('gohome_pre_tail_false_positive_episodes', '')}`",
            f"- ACT p95 latency ms: `{runtime.get('act_p95_ms', '')}`",
            f"- Gate p95 latency ms: `{runtime.get('gate_p95_ms', '')}`",
            "",
            "## Before Any Live Motion",
            "",
            "- Verify all artifact SHA-256 values against `candidate_package_manifest.json`.",
            "- Load the ACT checkpoint with its matching `dataset_stats.pkl` and `resolved_config.yaml`.",
            "- Load the causal temporal direction model and reject models whose context offsets include future frames.",
            "- Keep logs split into raw policy action, phase-gated action, snapped action, temporal-direction action, safe action, and commanded action.",
            "- Treat gohome output as a conservative request signal only.",
            "- Abort if tail stop crosses effective deadzones or if gohome fires before the tail/cycle-complete region.",
            "",
        ]
    )


def _base_artifacts(base_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    keep = {
        "action_policy_best",
        "action_dataset_stats",
        "action_resolved_config",
        "action_run_metadata",
        "phase_gate_model",
        "phase_gate_metadata",
        "tail_candidate_model",
        "tail_candidate_metadata",
        "gohome_eligibility_model",
        "gohome_eligibility_metadata",
    }
    return [dict(item) for item in base_manifest.get("artifacts", []) if item.get("name") in keep]


def _temporal_artifacts(args: argparse.Namespace) -> list[tuple[str, Path]]:
    return [
        ("temporal_direction_model", args.temporal_direction_dir / "temporal_direction_gate_model.pt"),
        ("temporal_direction_metadata", args.temporal_direction_dir / "temporal_direction_gate_model_metadata.json"),
        ("temporal_direction_fold_summary", args.temporal_direction_dir / "fold_summary.json"),
        ("temporal_direction_scan", args.temporal_direction_dir / "temporal_direction_gate_scan.csv"),
        ("combined_candidate_summary", args.combined_candidate_dir / "combined_candidate_summary.json"),
        ("runtime_temporal_smoke_summary", args.runtime_smoke_dir / "full_act_temporal_gate_smoke_summary.json"),
        ("runtime_gohome_events", args.runtime_smoke_dir / "full_act_gohome_events.csv"),
        (
            "runtime_temporal_replay_summary",
            args.runtime_smoke_dir / "temporal_direction_action_replay" / "collection_summary.json",
        ),
        (
            "runtime_temporal_replay_metrics",
            args.runtime_smoke_dir / "temporal_direction_action_replay" / "episode_metrics.csv",
        ),
        (
            "runtime_temporal_deadzone_startup",
            args.runtime_deadzone_gate_dir / "startup_first_expert_effective_40_aggregate.csv",
        ),
        ("runtime_temporal_deadzone_tail", args.runtime_deadzone_gate_dir / "tail_stability_summary.csv"),
        ("runtime_temporal_window_aggregate", args.runtime_window_eval_dir / "deadzone_window_aggregate.csv"),
        ("runtime_temporal_window_summary", args.runtime_window_eval_dir / "deadzone_window_summary.json"),
    ]


def _compact_base_package(base_manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": base_manifest.get("candidate_id", ""),
        "selected_gates": base_manifest.get("selected_gates", {}),
        "artifact_count": len(list(base_manifest.get("artifacts", []))),
    }


def _compact_runtime_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "episodes": summary.get("episodes", ""),
        "raw_action_mae": summary.get("raw_action_mae", ""),
        "phase_gated_action_mae": summary.get("phase_gated_action_mae", ""),
        "snap_action_mae": summary.get("snap_action_mae", ""),
        "temporal_direction_action_mae": summary.get("temporal_direction_action_mae", ""),
        "temporal_direction_action_rmse": summary.get("temporal_direction_action_rmse", ""),
        "gohome_event_recall": summary.get("gohome_event_recall", ""),
        "gohome_pre_tail_false_positive_episodes": summary.get("gohome_pre_tail_false_positive_episodes", ""),
        "gohome_pre_tail_active_frames": summary.get("gohome_pre_tail_active_frames", ""),
        "act_p95_ms": summary.get("act_p95_ms", ""),
        "gate_p95_ms": summary.get("gate_p95_ms", ""),
        "temporal_context_offsets": summary.get("temporal_context_offsets", []),
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
