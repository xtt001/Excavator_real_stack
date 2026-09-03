#!/usr/bin/env python3
"""Build a portable runtime bundle for the task-state-v2 offline candidate."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import yaml


DEFAULT_SOURCE = Path(
    "/data/pingfan/Excavator_real_stack_data/runs/"
    "real_transition_v2_0_1_task_state_v2_v1/offline_candidate_allow2_v1"
)
DEFAULT_CONTRACT_SOURCE = Path(
    "policy_bundles/real_transition_target_release_v2/contracts"
)
DEFAULT_OUTPUT = Path("policy_bundles/real_transition_task_state_v2_allow2")
DEFAULT_AUTO_PROGRESS_CONTRACT = Path(
    "/data/pingfan/Excavator_real_stack_data/runs/"
    "real_transition_v2_0_1_task_state_v2_v1/auto_progress_contract_v1/"
    "task_state_auto_progress_contract.json"
)
DEFAULT_AUTO_PROGRESS_REPLAY = Path(
    "/data/pingfan/Excavator_real_stack_data/runs/"
    "real_transition_v2_0_1_task_state_v2_v1/auto_progress_replay_v1"
)
EXPECTED_LOW_DIM_KEYS = ["qpos", "qvel", "real_transition_task_state_v2"]
CHECKPOINT_SHA256 = "e57bd59f07650f674f58eb9dfdaae2c06ead22b903922039cb2e6400daacaa4b"

SOURCE_FILES = {
    "policy_candidate.ckpt": "policy_accepted.ckpt",
    "dataset_stats.pkl": "dataset_stats.pkl",
    "resolved_config.yaml": "resolved_config.yaml",
    "run_metadata.json": "run_metadata.json",
    "acceptance_result.json": "evaluation/acceptance_result.json",
    "expert_reference.json": "evaluation/expert_reference.json",
    "frozen_probe_manifest.json": "evaluation/frozen_probe_manifest.json",
    "task_state_manifest.json": "manifest/task_state_manifest.json",
    "bundle_manifest.json": "manifest/offline_candidate_bundle_manifest.json",
}
CONTRACT_FILES = (
    "ready_contract.json",
    "target_release_contract_v2.json",
    "direct_policy_output_mechanical_deadzone.json",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--contract-source", type=Path, default=DEFAULT_CONTRACT_SOURCE)
    parser.add_argument(
        "--auto-progress-contract",
        type=Path,
        default=DEFAULT_AUTO_PROGRESS_CONTRACT,
    )
    parser.add_argument(
        "--auto-progress-replay",
        type=Path,
        default=DEFAULT_AUTO_PROGRESS_REPLAY,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build_bundle(
        source=args.source.resolve(),
        contract_source=args.contract_source.resolve(),
        auto_progress_contract=args.auto_progress_contract.resolve(),
        auto_progress_replay=args.auto_progress_replay.resolve(),
        output=args.output.resolve(),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def build_bundle(
    *,
    source: Path,
    contract_source: Path,
    auto_progress_contract: Path,
    auto_progress_replay: Path,
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite runtime bundle: {output}")
    _verify_source(source)
    source_accepted = _json(source / "accepted_model.json")
    if source_accepted.get("status") != "OFFLINE_CANDIDATE_ONLY":
        raise ValueError("source must retain OFFLINE_CANDIDATE_ONLY status")
    if source_accepted.get("checkpoint") != "policy_candidate.ckpt":
        raise ValueError("source checkpoint name is invalid")
    if source_accepted.get("checkpoint_sha256") != CHECKPOINT_SHA256:
        raise ValueError(
            "source checkpoint identity is not the selected allow-2 candidate"
        )

    resolved = yaml.safe_load((source / "resolved_config.yaml").read_text()) or {}
    low_dim_keys = list((resolved.get("policy", {}) or {}).get("low_dim_keys", ()))
    state_dim = int(
        ((resolved.get("policy", {}) or {}).get("act_params", {}) or {}).get(
            "state_dim", -1
        )
    )
    if low_dim_keys != EXPECTED_LOW_DIM_KEYS or state_dim != 13:
        raise ValueError(
            "source model must use qpos4+qvel4+task-state-v2(5), state_dim=13"
        )

    missing = [
        source / source_name
        for source_name in SOURCE_FILES
        if not (source / source_name).is_file()
    ]
    missing.extend(
        contract_source / name
        for name in CONTRACT_FILES
        if not (contract_source / name).is_file()
    )
    if not auto_progress_contract.is_file():
        missing.append(auto_progress_contract)
    if not (auto_progress_replay / "auto_progress_replay.json").is_file():
        missing.append(auto_progress_replay / "auto_progress_replay.json")
    if missing:
        raise FileNotFoundError(
            "runtime bundle source is incomplete: "
            + ", ".join(str(path) for path in missing)
        )
    _verify_sha256(auto_progress_replay, sums_name="SHA256SUMS.txt")

    output.mkdir(parents=True)
    for source_name, destination_name in SOURCE_FILES.items():
        _copy_verified(source / source_name, output / destination_name)
    for path in sorted((source / "evaluation").iterdir()):
        if path.is_file():
            _copy_verified(path, output / "evaluation" / path.name)
    for name in CONTRACT_FILES:
        _copy_verified(contract_source / name, output / "contracts" / name)
    _copy_verified(
        auto_progress_contract,
        output / "contracts/task_state_auto_progress_contract.json",
    )
    for path in sorted(auto_progress_replay.iterdir()):
        if path.is_file():
            _copy_verified(
                path,
                output / "evaluation/automatic_progress" / path.name,
            )

    git_commit = _git_head()
    generated_at = dt.datetime.now(dt.timezone.utc).isoformat()
    runtime_contract = {
        "schema": "real_transition_task_state_v2_runtime_contract_v1",
        "generated_at": generated_at,
        "model_input": {
            "low_dim_keys": EXPECTED_LOW_DIM_KEYS,
            "state_dim": 13,
            "task_state_layout": [
                "current_side_code",
                "dig_target_code",
                "work_complete",
                "return_commit",
                "gated_next_target_code",
            ],
        },
        "owner": {
            "type": "planner_plus_automatic_causal_progress",
            "automatic_causal_progress": True,
            "future_observation_used": False,
            "operator_mark_required": False,
            "goal_commit": (
                "current_side=dig_target=planner current side; work_complete=0; "
                "return_commit=0; next target hidden"
            ),
            "work_complete": (
                "after measured boom/bucket liveness, confirmed positive swing "
                "excursion, sustained effective positive bucket action and its "
                "causal release window; reset ACT temporal state"
            ),
            "return_commit": (
                "after work_complete and a causal all-axis mechanically idle policy "
                "window; expose planner next target and reset ACT temporal state"
            ),
            "cycle_ready": "planner closes goal and next goal commit resets task state",
        },
        "control_path": [
            "remote operator events",
            "scripted cycle and task-state owner",
            "automatic causal task-progress detector",
            "ACT raw chunk and temporal aggregation",
            "policy action scaling (identity)",
            "deadzone assist (disabled)",
            "measured-state swing landing",
            "ActionGuard",
            "50 Hz real action pump",
            "bridge low-level controller",
        ],
        "safety": {
            "default_output_mode": "shadow_zero",
            "control_requires_per_run_confirmation": True,
            "missing_progress_evidence": "remain uncommitted until review/timeout",
            "script_fault_or_completion": "latched zero output",
            "shutdown": "zero command",
        },
        "qualification": {
            "stationary_shadow": (
                "requires WORK to remain unchanged with no liveness, pending event, "
                "or applied event while returned/safe/commanded actions stay zero"
            ),
            "automatic_progress": (
                "requires the full ordered task-state sequence only in recorded-state "
                "replay or controlled physical motion"
            ),
            "stationary_shadow_can_prove_full_progress": False,
        },
        "evidence_boundary": (
            "Software and bundle contract only. No hydraulic response, soil effect, "
            "or physical closed-loop cycle is asserted."
        ),
    }
    _write_json(output / "contracts/task_state_runtime_contract.json", runtime_contract)

    deployed_accepted = dict(source_accepted)
    deployed_accepted.update(
        {
            "schema": "real_transition_task_state_v2_runtime_candidate_v1",
            "generated_at": generated_at,
            "status": "OFFLINE_CANDIDATE_ONLY",
            "checkpoint": "policy_accepted.ckpt",
            "runtime_packaging_git_commit": git_commit,
            "source_offline_candidate": str(source),
            "runtime": {
                "task_state_owner_implemented": True,
                "task_state_owner": "automatic_causal_policy_state",
                "control_path_implemented": True,
                "shadow_zero_required_before_control": True,
                "controlled_motion_authorized_by_bundle": False,
                "field_ready": False,
            },
        }
    )
    _write_json(output / "accepted_model.json", deployed_accepted)
    (output / "SOURCE_COMMIT.txt").write_text(git_commit + "\n", encoding="utf-8")
    (output / "DEPLOYMENT_README.md").write_text(
        _deployment_readme(git_commit), encoding="utf-8"
    )
    (output / "SHADOW_ZERO_CHECKLIST.md").write_text(
        _shadow_zero_checklist(), encoding="utf-8"
    )
    (output / "CONTROLLED_CYCLE_CHECKLIST.md").write_text(
        _controlled_cycle_checklist(), encoding="utf-8"
    )

    files = _bundle_files(output)
    manifest = {
        "schema": "real_transition_task_state_v2_runtime_bundle_v1",
        "generated_at": generated_at,
        "status": "OFFLINE_CANDIDATE_ONLY",
        "source_bundle": str(source),
        "runtime_packaging_git_commit": git_commit,
        "checkpoint": "policy_accepted.ckpt",
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "files": files,
        "evidence_boundary": runtime_contract["evidence_boundary"],
    }
    _write_json(output / "runtime_bundle_manifest.json", manifest)
    hashes = {
        path.relative_to(output).as_posix(): _sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "SHA256SUMS"
    }
    (output / "SHA256SUMS").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(hashes.items())),
        encoding="utf-8",
    )
    _verify_sha256(output)
    return {
        "status": "PASS",
        "output": str(output),
        "checkpoint_sha256": _sha256(output / "policy_accepted.ckpt"),
        "runtime_packaging_git_commit": git_commit,
        "file_count": len(hashes),
        "sha256sums": str(output / "SHA256SUMS"),
        "evidence_boundary": runtime_contract["evidence_boundary"],
    }


def _verify_source(source: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(f"offline candidate does not exist: {source}")
    _verify_sha256(source)


def _verify_sha256(directory: Path, *, sums_name: str = "SHA256SUMS") -> None:
    result = subprocess.run(
        ["sha256sum", "-c", str(sums_name)],
        cwd=directory,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"SHA256SUMS verification failed in {directory}:\n"
            + result.stdout
            + result.stderr
        )


def _copy_verified(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".tmp-{os.getpid()}")
    shutil.copy2(source, temporary)
    if _sha256(source) != _sha256(temporary):
        raise RuntimeError(f"copy hash mismatch: {source} -> {destination}")
    os.replace(temporary, destination)


def _bundle_files(output: Path) -> dict[str, dict[str, Any]]:
    return {
        path.relative_to(output).as_posix(): {
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(output.rglob("*"))
        if path.is_file()
        and path.name not in {"SHA256SUMS", "runtime_bundle_manifest.json"}
    }


def _deployment_readme(git_commit: str) -> str:
    return f"""# Task-state-v2 runtime candidate

This bundle remains an offline candidate. It includes the actual-control
software path but does not claim a physical closed-loop result.

Required code commit: `{git_commit}`

1. Run `MODE=shadow DRY_RUN=YES scripts/run_real_transition_task_state_v2_policy.sh`.
2. Run stationary shadow_zero at A and B. It must remain in WORK with no task
   transition while returned/safe/commanded actions remain zero.
3. Do not require WORK_COMPLETE or RETURN_COMMITTED in stationary shadow: physical
   boom/bucket liveness is intentionally absent.
4. Review the frozen recorded-state automatic progress replay, then use a finite
   single-cycle script for the first controlled test.
5. Physical button 7 arms/stops policy. Normal cycle progress requires no mark
   button. Only after the script, stationary shadow log, automatic progress
   contract, and motion boundary have been reviewed may control be confirmed.

The runner starts the existing `slave_real_stack.sh` control stack. Its control
chain is ACT -> landing -> ActionGuard -> real action pump -> low-level bridge.
"""


def _shadow_zero_checklist() -> str:
    return """# Task-state-v2 shadow_zero checklist

- Verify `sha256sum -c SHA256SUMS` inside the bundle.
- Verify qpos, raw qvel, four camera inputs, and the five-value task token in logs.
- Keep the machine stationary in one supported A/B ready region for at least 20
  logged policy steps.
- Confirm task state remains WORK and its vector remains valid and planner-aligned.
- Confirm no boom/bucket work liveness, bucket-effective latch, pending event,
  WORK_COMPLETE, RETURN_COMMITTED, or cycle advance is reported.
- Confirm a nonzero raw policy action still yields zero returned and commanded action.
- Confirm script fault, completion, manual toggle, and shutdown all yield zero output.
- Repeat stationary shadow at A and B before any controlled motion.
- Do not describe shadow or offline replay as hydraulic or physical closed-loop proof.
"""


def _controlled_cycle_checklist() -> str:
    return """# Task-state-v2 controlled single-cycle checklist

- Use a reviewed finite single-cycle script; start with B-to-A for the known failure.
- Confirm physical button 7 stops policy and the physical emergency path is ready.
- Confirm measured boom and bucket qpos liveness before automatic task progress.
- Confirm positive swing excursion, sustained effective bucket action, and bucket
  release occur in order before WORK_COMPLETE.
- Confirm the all-axis policy action-idle window occurs before RETURN_COMMITTED.
- Confirm each task-state change resets ACT before the next inference.
- Confirm post-commit negative return, landing, target ready, script completion, and
  final zero command from raw policy_action through safe/commanded action logs.
- Stop on missing, reordered, or prematurely applied progress evidence.
- One controlled cycle is physical evidence for that run only, not production proof.
"""


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: MappingLike) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


MappingLike = dict[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout.strip()


if __name__ == "__main__":
    main()
