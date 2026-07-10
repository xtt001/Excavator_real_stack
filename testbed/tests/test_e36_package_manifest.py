import json
from pathlib import Path

from scripts.e36_build_policy_gate_package_manifest import (
    build_artifact_entry,
    build_field_smoke_checklist,
    build_manifest,
    sha256_file,
    verify_manifest,
)


def test_sha256_file_hashes_file_content(tmp_path: Path) -> None:
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"policy artifact")

    assert sha256_file(path) == "1c122c364ffab148fd84474ea2e8fca676eb868e4738b5a61010736865bf669f"


def test_build_artifact_entry_records_size_and_hash(tmp_path: Path) -> None:
    path = tmp_path / "artifact.txt"
    path.write_text("gate\n", encoding="utf-8")

    entry = build_artifact_entry("phase_gate_model", path)

    assert entry["name"] == "phase_gate_model"
    assert entry["path"] == str(path)
    assert entry["size_bytes"] == 5
    assert len(entry["sha256"]) == 64


def test_verify_manifest_fails_missing_artifact(tmp_path: Path) -> None:
    manifest = build_manifest(
        candidate_id="E36",
        selected_gates={
            "phase_gate": "simple_0.15_s0.50",
            "gohome_gate": "learned_tail_t0.97_tc10_e0.80_ec3",
        },
        artifacts=[{"name": "missing", "path": str(tmp_path / "missing.bin"), "size_bytes": 1, "sha256": "0"}],
        evidence={},
    )

    report = verify_manifest(manifest)

    assert report["ok"] is False
    assert "missing artifact missing" in report["errors"]


def test_verify_manifest_fails_sha_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"current")
    entry = build_artifact_entry("policy_best", path)
    entry["sha256"] = "0" * 64
    manifest = build_manifest(
        candidate_id="E36",
        selected_gates={
            "phase_gate": "simple_0.15_s0.50",
            "gohome_gate": "learned_tail_t0.97_tc10_e0.80_ec3",
        },
        artifacts=[entry],
        evidence={},
    )

    report = verify_manifest(manifest)

    assert report["ok"] is False
    assert "sha256 mismatch for policy_best" in report["errors"]


def test_verify_manifest_requires_selected_gate_names(tmp_path: Path) -> None:
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"current")
    manifest = build_manifest(
        candidate_id="E36",
        selected_gates={"phase_gate": "", "gohome_gate": ""},
        artifacts=[build_artifact_entry("policy_best", path)],
        evidence={},
    )

    report = verify_manifest(manifest)

    assert report["ok"] is False
    assert "missing selected gate phase_gate" in report["errors"]
    assert "missing selected gate gohome_gate" in report["errors"]


def test_verify_manifest_round_trips_json(tmp_path: Path) -> None:
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"current")
    manifest = build_manifest(
        candidate_id="E36",
        selected_gates={
            "phase_gate": "simple_0.15_s0.50",
            "gohome_gate": "learned_tail_t0.97_tc10_e0.80_ec3",
        },
        artifacts=[build_artifact_entry("policy_best", path)],
        evidence={"runtime_smoke": {"ok": True}},
    )
    loaded = json.loads(json.dumps(manifest))

    report = verify_manifest(loaded)

    assert report["ok"] is True
    assert report["checked_artifacts"] == 1


def test_field_smoke_checklist_reads_runtime_global_metrics() -> None:
    manifest = build_manifest(
        candidate_id="E36",
        selected_gates={
            "phase_gate": "simple_0.15_s0.50",
            "gohome_gate": "learned_tail_t0.97_tc10_e0.80_ec3",
        },
        artifacts=[],
        evidence={
            "runtime_gate_smoke": {
                "action_global_metrics": {"overall": {"mae": 0.0414}},
                "gohome_event_summary": {
                    "event_recall": 0.9583,
                    "pre_tail_false_positive_episodes": 0,
                },
                "latency_summary": {"p95_ms": 0.035},
            }
        },
    )

    checklist = build_field_smoke_checklist(manifest, {"ok": True})

    assert "- Action MAE: `0.0414`" in checklist
    assert "- Gohome event recall: `0.9583`" in checklist
