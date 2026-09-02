"""Package a selected task-state-v2 checkpoint as an offline-only candidate."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
from pathlib import Path
from typing import Any

import yaml

from testbed.tasks.real_transition import sha256_file, write_immutable_text


def main() -> None:
    parser = argparse.ArgumentParser(prog="tb-task-state-v2-package")
    parser.add_argument("--redecision", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = package_candidate(
        redecision_path=args.redecision, output_dir=args.output_dir
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def package_candidate(
    *, redecision_path: Path | str, output_dir: Path | str
) -> dict[str, Any]:
    decision_path = Path(redecision_path).resolve()
    decision = _json(decision_path)
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite candidate package: {output}")
    selected = decision.get("selected_candidate")
    if decision.get("status") != "RETROSPECTIVE_EXPERT_ALIGNED_CANDIDATE" or not selected:
        raise ValueError("redecision did not select an offline candidate")
    selected_spec = dict(decision["candidate_summaries"][selected])
    checkpoint = Path(str(selected_spec["checkpoint"])).resolve()
    if sha256_file(checkpoint) != str(selected_spec["checkpoint_sha256"]):
        raise ValueError("selected checkpoint SHA-256 mismatch")
    import torch

    checkpoint_payload = torch.load(checkpoint, map_location="cpu")
    checkpoint_epoch = int(checkpoint_payload["epoch"])
    run_dir = checkpoint.parent
    source_files = {
        "policy_candidate.ckpt": checkpoint,
        "dataset_stats.pkl": run_dir / "dataset_stats.pkl",
        "resolved_config.yaml": run_dir / "resolved_config.yaml",
        "run_metadata.json": run_dir / "run_metadata.json",
    }
    for destination, source in source_files.items():
        if not source.is_file():
            raise FileNotFoundError(f"missing {destination} source: {source}")
    metadata = _json(run_dir / "run_metadata.json")
    if metadata.get("status") != "completed":
        raise ValueError("selected training run is not complete")
    resolved = yaml.safe_load((run_dir / "resolved_config.yaml").read_text())
    expected_keys = ["qpos", "qvel", "real_transition_task_state_v2"]
    if resolved["policy"]["low_dim_keys"] != expected_keys:
        raise ValueError("selected model low_dim_keys mismatch")
    if int(resolved["policy"]["act_params"]["state_dim"]) != 13:
        raise ValueError("selected model state_dim mismatch")
    evaluation_result = Path(str(selected_spec["result"])).resolve()
    if sha256_file(evaluation_result) != str(selected_spec["result_sha256"]):
        raise ValueError("selected evaluation result SHA-256 mismatch")
    expert_reference = Path(str(decision["expert_reference"]["path"])).resolve()
    expert = _json(expert_reference)
    probe = Path(str(expert["probe_manifest"]["path"])).resolve()
    task_manifest = Path(str(expert["task_state_manifest"]["path"])).resolve()

    output.mkdir(parents=True)
    for destination, source in source_files.items():
        shutil.copy2(source, output / destination)
    shutil.copy2(task_manifest, output / "task_state_manifest.json")
    shutil.copy2(probe, output / "frozen_probe_manifest.json")
    shutil.copy2(expert_reference, output / "expert_reference.json")
    shutil.copy2(decision_path, output / "acceptance_result.json")
    evaluation_output = output / "evaluation"
    evaluation_output.mkdir()
    for source in sorted(evaluation_result.parent.iterdir()):
        if source.is_file():
            shutil.copy2(source, evaluation_output / source.name)

    candidate = {
        "schema": "real_transition_task_state_v2_offline_candidate_v1",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "OFFLINE_CANDIDATE_ONLY",
        "selected_candidate": str(selected),
        "checkpoint": "policy_candidate.ckpt",
        "checkpoint_epoch": checkpoint_epoch,
        "checkpoint_sha256": sha256_file(output / "policy_candidate.ckpt"),
        "low_dim_keys": expected_keys,
        "state_dim": 13,
        "uncommitted_tolerance": decision["uncommitted_tolerance"],
        "offline_metrics": selected_spec["summary"],
        "selection": {
            "candidate_population": int(decision["candidate_count"]),
            "passing_candidate_count": len(decision["passing_candidates_ranked"]),
            "ranking": decision["passing_candidates_ranked"],
            "requirement_change_timing": "post-result user-authorized",
        },
        "runtime": {
            "task_state_owner_implemented": False,
            "shadow_zero_required": True,
            "controlled_motion_authorized": False,
            "field_ready": False,
        },
        "evidence_boundary": (
            "Recorded-observation open-loop replay only. This package does not "
            "prove policy-driven future state, hydraulic response, soil effect, "
            "or physical closed-loop completion."
        ),
    }
    accepted_path = write_immutable_text(
        output / "accepted_model.json",
        json.dumps(candidate, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    checklist = """# Task-state-v2 shadow_zero checklist

- Verify `sha256sum -c SHA256SUMS` from this directory.
- Confirm the loaded keys are exactly `qpos`, `qvel`, `real_transition_task_state_v2` and state dimension is 13.
- Do not enable motion until planner/runtime owns and logs current side, dig target, dig complete, return commit and next target.
- Confirm next target is zero-gated before return commit and exposed only after commit.
- Reset ACT temporal aggregation on every task-state bit transition.
- Run with `output_mode: shadow_zero`; verify nonzero policy diagnostics still produce zero returned, safe and commanded actions.
- Replay one A→B, one B→A and one same-side cycle and compare task-state epochs with policy reset logs.
- Treat the two documented abnormal field hybrids as non-gating because their actual field images are absent.
- This checklist does not authorize controlled motion. Create a separate reviewed field contract before any actuator command.
"""
    write_immutable_text(output / "SHADOW_ZERO_CHECKLIST.md", checklist)
    manifest = {
        "schema": "real_transition_task_state_v2_candidate_bundle_manifest_v1",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_run": str(run_dir),
        "source_checkpoint": str(checkpoint),
        "source_redecision": str(decision_path),
        "files": {},
    }
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name not in {"SHA256SUMS", "bundle_manifest.json"}:
            manifest["files"][str(path.relative_to(output))] = sha256_file(path)
    manifest_path = write_immutable_text(
        output / "bundle_manifest.json",
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    sums = [
        f"{sha256_file(path)}  {path.relative_to(output)}"
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "SHA256SUMS"
    ]
    write_immutable_text(output / "SHA256SUMS", "\n".join(sums) + "\n")
    return {
        "bundle": str(output),
        "accepted_model": str(accepted_path),
        "bundle_manifest": str(manifest_path),
        "checkpoint_sha256": candidate["checkpoint_sha256"],
        "status": candidate["status"],
    }


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


if __name__ == "__main__":
    main()
