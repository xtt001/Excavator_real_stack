from pathlib import Path

from scripts.e52_build_temporal_policy_gate_package_manifest import build_temporal_manifest
from scripts.e36_build_policy_gate_package_manifest import verify_manifest


def test_build_temporal_manifest_includes_temporal_artifacts(tmp_path: Path) -> None:
    temporal_model = tmp_path / "temporal_direction_gate_model.pt"
    temporal_metadata = tmp_path / "temporal_direction_gate_model_metadata.json"
    runtime_summary = tmp_path / "full_act_temporal_gate_smoke_summary.json"
    for path in (temporal_model, temporal_metadata, runtime_summary):
        path.write_text(path.name, encoding="utf-8")

    manifest = build_temporal_manifest(
        candidate_id="E52",
        selected_gates={
            "phase_gate": "simple_0.15_s0.50",
            "gohome_gate": "learned_tail_t0.97_tc10_e0.80_ec3",
            "temporal_direction_gate": "tdir_t50_s75",
        },
        artifacts=[
            ("temporal_direction_model", temporal_model),
            ("temporal_direction_metadata", temporal_metadata),
            ("runtime_temporal_smoke_summary", runtime_summary),
        ],
        evidence={"runtime_gate_smoke": {"episodes": 24}},
    )

    names = {artifact["name"] for artifact in manifest["artifacts"]}
    assert {"temporal_direction_model", "temporal_direction_metadata", "runtime_temporal_smoke_summary"} <= names
    assert verify_manifest(manifest)["ok"] is True
