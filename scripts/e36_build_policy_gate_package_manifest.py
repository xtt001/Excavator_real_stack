#!/usr/bin/env python3
"""Build and verify the E36 policy+gate candidate package manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-id", default="E36")
    parser.add_argument("--action-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--phase-gate-dir", type=Path, required=True)
    parser.add_argument("--action-gate-replay-dir", type=Path, required=True)
    parser.add_argument("--action-deadzone-gate-dir", type=Path, required=True)
    parser.add_argument("--gohome-eligibility-dir", type=Path, required=True)
    parser.add_argument("--two-stage-gohome-dir", type=Path, required=True)
    parser.add_argument("--combined-candidate-dir", type=Path, required=True)
    parser.add_argument("--runtime-smoke-dir", type=Path, required=True)
    parser.add_argument("--runtime-deadzone-gate-dir", type=Path, required=True)
    parser.add_argument("--phase-gate-name", default="simple_0.15_s0.50")
    parser.add_argument("--gohome-gate-name", default="learned_tail_t0.97_tc10_e0.80_ec3")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    artifacts = _candidate_artifacts(args)
    evidence = _candidate_evidence(args)
    manifest = build_manifest(
        candidate_id=str(args.candidate_id),
        selected_gates={
            "phase_gate": str(args.phase_gate_name),
            "gohome_gate": str(args.gohome_gate_name),
        },
        artifacts=artifacts,
        evidence=evidence,
    )
    verify_report = verify_manifest(manifest)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "candidate_package_manifest.json"
    verify_path = args.output_dir / "candidate_package_verify.json"
    checklist_path = args.output_dir / "field_smoke_checklist.md"
    _write_json(manifest_path, manifest)
    _write_json(verify_path, verify_report)
    checklist_path.write_text(build_field_smoke_checklist(manifest, verify_report), encoding="utf-8")
    if not verify_report["ok"]:
        raise SystemExit(f"candidate package verification failed: {verify_report['errors']}")
    print(f"Candidate package manifest: {manifest_path}")
    print(f"Candidate package verification: {verify_path}")
    print(f"Field smoke checklist: {checklist_path}")


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def build_artifact_entry(name: str, path: Path) -> dict[str, Any]:
    artifact_path = Path(path)
    stat = artifact_path.stat()
    return {
        "name": str(name),
        "path": str(artifact_path),
        "size_bytes": int(stat.st_size),
        "sha256": sha256_file(artifact_path),
    }


def build_manifest(
    *,
    candidate_id: str,
    selected_gates: dict[str, str],
    artifacts: list[dict[str, Any]],
    evidence: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "candidate_id": str(candidate_id),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "offline candidate package for action policy plus phase and gohome request gates",
        "selected_gates": dict(selected_gates),
        "artifacts": list(artifacts),
        "evidence": dict(evidence),
    }


def verify_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    selected_gates = dict(manifest.get("selected_gates", {}))
    for key in ("phase_gate", "gohome_gate"):
        if not str(selected_gates.get(key, "")).strip():
            errors.append(f"missing selected gate {key}")

    checked_artifacts = 0
    for artifact in list(manifest.get("artifacts", [])):
        name = str(artifact.get("name", ""))
        path = Path(str(artifact.get("path", "")))
        if not path.exists():
            errors.append(f"missing artifact {name}")
            continue
        checked_artifacts += 1
        actual_size = path.stat().st_size
        expected_size = int(artifact.get("size_bytes", -1))
        if actual_size != expected_size:
            errors.append(f"size mismatch for {name}")
        actual_sha = sha256_file(path)
        expected_sha = str(artifact.get("sha256", ""))
        if actual_sha != expected_sha:
            errors.append(f"sha256 mismatch for {name}")

    return {
        "ok": len(errors) == 0,
        "candidate_id": str(manifest.get("candidate_id", "")),
        "checked_artifacts": checked_artifacts,
        "declared_artifacts": len(list(manifest.get("artifacts", []))),
        "errors": errors,
    }


def build_field_smoke_checklist(manifest: dict[str, Any], verify_report: dict[str, Any]) -> str:
    gates = dict(manifest.get("selected_gates", {}))
    evidence = dict(manifest.get("evidence", {}))
    e35 = dict(evidence.get("runtime_gate_smoke", {}))
    latency = dict(e35.get("latency_summary", {}))
    gohome = dict(e35.get("gohome_event_summary", {}))
    action = dict(e35.get("action_global_metrics", {}).get("overall", {}))
    lines = [
        "# E36 Field Smoke Checklist",
        "",
        f"- Candidate: `{manifest.get('candidate_id', '')}`",
        f"- Manifest verification: `{verify_report.get('ok', False)}`",
        f"- Phase gate: `{gates.get('phase_gate', '')}`",
        f"- Gohome gate: `{gates.get('gohome_gate', '')}`",
        f"- Action MAE: `{action.get('mae', '')}`",
        f"- Gohome event recall: `{gohome.get('event_recall', '')}`",
        f"- Gohome pre-tail FP episodes: `{gohome.get('pre_tail_false_positive_episodes', '')}`",
        f"- Gate CPU p95 latency ms: `{latency.get('p95_ms', '')}`",
        "",
        "## Before Any Live Motion",
        "",
        "- Verify all artifact SHA-256 values against `candidate_package_manifest.json`.",
        "- Load the ACT checkpoint together with its matching `dataset_stats.pkl` and `resolved_config.yaml`.",
        "- Keep policy output logging split into raw policy action, phase-gated action, safe action, and commanded action.",
        "- Treat gohome output as a conservative request signal only; do not imitate gohome automation actions.",
        "- Abort if the policy moves during the tail stop window or if gohome request fires before the tail/cycle-complete region.",
        "",
    ]
    return "\n".join(lines)


def _candidate_artifacts(args: argparse.Namespace) -> list[dict[str, Any]]:
    artifact_specs = [
        ("action_policy_best", args.action_checkpoint_dir / "policy_best.ckpt"),
        ("action_dataset_stats", args.action_checkpoint_dir / "dataset_stats.pkl"),
        ("action_resolved_config", args.action_checkpoint_dir / "resolved_config.yaml"),
        ("action_run_metadata", args.action_checkpoint_dir / "run_metadata.json"),
        ("phase_gate_model", args.phase_gate_dir / "phase_gate_model.pt"),
        ("phase_gate_metadata", args.phase_gate_dir / "phase_gate_model_metadata.json"),
        ("phase_gate_fold_summary", args.phase_gate_dir / "fold_summary.json"),
        ("phase_gate_scan", args.phase_gate_dir / "threshold_scan.csv"),
        ("phase_gate_replay_summary", args.action_gate_replay_dir / "collection_summary.json"),
        ("phase_gate_replay_metrics", args.action_gate_replay_dir / "episode_metrics.csv"),
        ("phase_deadzone_summary", args.action_deadzone_gate_dir / "deadzone_window_summary.json"),
        ("phase_deadzone_startup", args.action_deadzone_gate_dir / "startup_first_expert_effective_40_aggregate.csv"),
        ("phase_deadzone_tail", args.action_deadzone_gate_dir / "tail_stability_summary.csv"),
        ("gohome_eligibility_model", args.gohome_eligibility_dir / "gohome_eligibility_model.pt"),
        ("gohome_eligibility_metadata", args.gohome_eligibility_dir / "gohome_eligibility_model_metadata.json"),
        ("gohome_eligibility_gate_summary", args.gohome_eligibility_dir / "gate_summary.json"),
        ("tail_candidate_model", args.two_stage_gohome_dir / "tail_candidate_model.pt"),
        ("tail_candidate_metadata", args.two_stage_gohome_dir / "tail_candidate_model_metadata.json"),
        ("two_stage_gohome_gate_summary", args.two_stage_gohome_dir / "gate_summary.json"),
        ("two_stage_gohome_events", args.two_stage_gohome_dir / f"{args.gohome_gate_name}_events.csv"),
        ("combined_candidate_summary", args.combined_candidate_dir / "combined_candidate_summary.json"),
        ("runtime_gate_smoke_summary", args.runtime_smoke_dir / "runtime_gate_smoke_summary.json"),
        ("runtime_gohome_events", args.runtime_smoke_dir / "final_gohome_events.csv"),
        ("runtime_phase_deadzone_startup", args.runtime_deadzone_gate_dir / "startup_first_expert_effective_40_aggregate.csv"),
        ("runtime_phase_deadzone_tail", args.runtime_deadzone_gate_dir / "tail_stability_summary.csv"),
    ]
    return [build_artifact_entry(name, path) for name, path in artifact_specs]


def _candidate_evidence(args: argparse.Namespace) -> dict[str, Any]:
    runtime_smoke = _read_json(args.runtime_smoke_dir / "runtime_gate_smoke_summary.json")
    return {
        "combined_candidate": _read_json(args.combined_candidate_dir / "combined_candidate_summary.json"),
        "runtime_gate_smoke": _compact_runtime_smoke_summary(runtime_smoke),
        "phase_gate_summary": _read_json(args.phase_gate_dir / args.phase_gate_name / "gate_summary.json"),
        "two_stage_gohome_summary": _read_json(args.two_stage_gohome_dir / "gate_summary.json"),
    }


def _compact_runtime_smoke_summary(summary: dict[str, Any]) -> dict[str, Any]:
    action_aggregate = dict(summary.get("action_aggregate", {}))
    return {
        "phase_gate_name": summary.get("phase_gate_name", ""),
        "gohome_gate_name": summary.get("gohome_gate_name", ""),
        "episodes": summary.get("episodes", ""),
        "action_global_metrics": action_aggregate.get("global_metrics", {}),
        "gohome_event_summary": summary.get("gohome_event_summary", {}),
        "latency_summary": summary.get("latency_summary", {}),
        "artifact_paths": summary.get("artifact_paths", {}),
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
