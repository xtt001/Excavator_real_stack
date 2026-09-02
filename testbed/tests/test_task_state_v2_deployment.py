from __future__ import annotations

import hashlib
import json
from pathlib import Path

import scripts.build_real_transition_task_state_v2_bundle as builder
import scripts.verify_real_transition_task_state_v2_runtime as verifier
import yaml

from testbed.config_loader import load_yaml_config
from testbed.tasks.home_side_contract import build_rule_ready_contract

REPO_ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_task_state_v2_bundle_and_static_runtime_preflight(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "offline_candidate"
    source.mkdir()
    checkpoint = source / "policy_candidate.ckpt"
    checkpoint.write_bytes(b"task-state-v2-test-checkpoint")
    checkpoint_sha = _sha256(checkpoint)
    monkeypatch.setattr(builder, "CHECKPOINT_SHA256", checkpoint_sha)
    monkeypatch.setattr(verifier, "CHECKPOINT_SHA256", checkpoint_sha)

    _write_json(
        source / "accepted_model.json",
        {
            "status": "OFFLINE_CANDIDATE_ONLY",
            "checkpoint": "policy_candidate.ckpt",
            "checkpoint_sha256": checkpoint_sha,
            "offline_metrics": {},
        },
    )
    (source / "dataset_stats.pkl").write_bytes(b"stats")
    (source / "resolved_config.yaml").write_text(
        yaml.safe_dump(
            {
                "task": {"camera_names": verifier.CAMERAS},
                "policy": {
                    "low_dim_keys": verifier.LOW_DIM_KEYS,
                    "act_params": {"state_dim": 13, "chunk_size": 20},
                },
                "train": {
                    "state_visual_residual": {"enabled": True},
                    "task_state_v2_adherence_loss": {"enabled": True},
                    "deadzone_loss": {"enabled": True},
                },
            }
        ),
        encoding="utf-8",
    )
    (source / "run_metadata.json").write_text("{}\n", encoding="utf-8")
    for name in (
        "acceptance_result.json",
        "expert_reference.json",
        "frozen_probe_manifest.json",
        "task_state_manifest.json",
        "bundle_manifest.json",
    ):
        (source / name).write_text("{}\n", encoding="utf-8")
    _write_json(source / "evaluation/probe_result.json", {"status": "PASS"})
    hashes = {
        path.relative_to(source).as_posix(): _sha256(path)
        for path in sorted(source.rglob("*"))
        if path.is_file()
    }
    (source / "SHA256SUMS").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(hashes.items())),
        encoding="utf-8",
    )

    contracts = tmp_path / "contracts"
    _write_json(contracts / "ready_contract.json", build_rule_ready_contract())
    _write_json(
        contracts / "target_release_contract_v2.json",
        {
            "schema": "real_transition_target_release_contract_v1",
            "decision_region": {
                "train_A_endpoint_range_rad": [-0.38, -0.09],
                "swing_qpos_range_rad": [0.11, 0.39],
            },
        },
    )
    _write_json(contracts / "direct_policy_output_mechanical_deadzone.json", {})

    bundle = tmp_path / "policy_bundles/real_transition_task_state_v2_allow2"
    result = builder.build_bundle(
        source=source, contract_source=contracts, output=bundle
    )
    config = load_yaml_config(
        REPO_ROOT
        / "testbed/testbed/configs/policy_real_transition_task_state_v2_allow2.yaml"
    )
    config["teleop"]["policy"]["bundle_dir"] = str(bundle)
    runtime_config = tmp_path / "runtime.yaml"
    runtime_config.write_text(yaml.safe_dump(config), encoding="utf-8")

    preflight = verifier.verify_runtime(
        config_path=runtime_config,
        bundle=bundle,
        expected_output_mode="shadow_zero",
    )

    assert result["status"] == "PASS"
    assert preflight["status"] == "PASS"
    assert preflight["task_state_owner"] == "planner_plus_explicit_operator_mark"
    assert (bundle / "policy_accepted.ckpt").read_bytes() == checkpoint.read_bytes()
