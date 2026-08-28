#!/usr/bin/env python3
"""Build the portable field bundle for the accepted target-release ACT model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path


DEFAULT_SOURCE = Path(
    "/data/pingfan/Excavator_real_stack_data/runs/"
    "real_transition_v2_0_1_ctx04_target_release_v2_w0p503_final300"
)
DEFAULT_OUTPUT = Path("policy_bundles/real_transition_target_release_v2")

FILES = {
    "policy_accepted.ckpt": "policy_accepted.ckpt",
    "dataset_stats.pkl": "dataset_stats.pkl",
    "resolved_config.yaml": "resolved_config.yaml",
    "run_metadata.json": "run_metadata.json",
    "accepted_model.json": "accepted_model.json",
    "FIELD_CHECKLIST.md": "FIELD_CHECKLIST.md",
    "contracts/ready_contract.json": "contracts/ready_contract.json",
    "contracts/direct_policy_output_mechanical_deadzone.json": (
        "contracts/direct_policy_output_mechanical_deadzone.json"
    ),
    "contracts/target_release_contract_v2.json": (
        "contracts/target_release_contract_v2.json"
    ),
    "contracts/acceptance_contract_v2.json": "contracts/acceptance_contract_v2.json",
    "evaluation/acceptance_result.json": "evaluation/acceptance_result.json",
    "evaluation/planner_open_loop_per_goal_reset.json": (
        "evaluation/planner_open_loop_per_goal_reset.json"
    ),
    "evaluation/state_hold_transition.json": "evaluation/state_hold_transition.json",
    "evaluation/state_hold_n5_reference.json": (
        "evaluation/state_hold_n5_reference.json"
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = build_bundle(
        source=args.source.resolve(),
        output=args.output.resolve(),
        overwrite=bool(args.overwrite),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def build_bundle(*, source: Path, output: Path, overwrite: bool) -> dict:
    accepted_path = source / "accepted_model.json"
    if not accepted_path.is_file():
        raise FileNotFoundError(
            f"accepted model manifest does not exist: {accepted_path}"
        )
    accepted = json.loads(accepted_path.read_text(encoding="utf-8"))
    if accepted.get("status") != "OFFLINE_ACCEPTED_FIELD_CANDIDATE":
        raise ValueError("source is not an OFFLINE_ACCEPTED_FIELD_CANDIDATE")
    if accepted.get("checkpoint") != "policy_accepted.ckpt":
        raise ValueError("accepted manifest checkpoint must be policy_accepted.ckpt")

    missing = [
        str(source / source_rel)
        for source_rel in FILES
        if not (source / source_rel).is_file()
    ]
    if missing:
        raise FileNotFoundError("source bundle is incomplete: " + ", ".join(missing))
    _verify_source_sha256(source)

    existing = [
        output / destination_rel
        for destination_rel in FILES.values()
        if (output / destination_rel).exists()
    ]
    if existing and not overwrite:
        raise FileExistsError(
            "output bundle already contains files; pass --overwrite after review: "
            + ", ".join(str(path) for path in existing)
        )
    output.mkdir(parents=True, exist_ok=True)

    manifest_files: dict[str, dict[str, object]] = {}
    for source_rel, destination_rel in FILES.items():
        source_path = source / source_rel
        destination = output / destination_rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        _copy_atomic(source_path, destination)
        source_hash = _sha256(source_path)
        destination_hash = _sha256(destination)
        if source_hash != destination_hash:
            raise RuntimeError(f"bundle copy hash mismatch: {destination}")
        manifest_files[destination_rel] = {
            "source": str(source_path),
            "size_bytes": destination.stat().st_size,
            "sha256": destination_hash,
        }

    manifest = {
        "schema": "real_transition_target_release_runtime_bundle_v1",
        "status": "OFFLINE_ACCEPTED_FIELD_CANDIDATE",
        "source_bundle": str(source),
        "runtime_packaging_git_commit": _git_head(),
        "checkpoint": "policy_accepted.ckpt",
        "checkpoint_sha256": str(accepted.get("checkpoint_sha256", "")),
        "files": manifest_files,
        "evidence_boundary": accepted.get("evidence_boundary", ""),
    }
    manifest_path = output / "runtime_bundle_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    hashes = {
        **{name: str(value["sha256"]) for name, value in manifest_files.items()},
        "runtime_bundle_manifest.json": _sha256(manifest_path),
    }
    sha_path = output / "SHA256SUMS"
    sha_path.write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(hashes.items())),
        encoding="utf-8",
    )
    return {
        "status": "PASS",
        "output": str(output),
        "checkpoint_sha256": hashes["policy_accepted.ckpt"],
        "file_count": len(hashes),
        "sha256sums": str(sha_path),
    }


def _verify_source_sha256(source: Path) -> None:
    result = subprocess.run(
        ["sha256sum", "-c", "SHA256SUMS"],
        cwd=source,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "source SHA256SUMS verification failed:\n" + result.stdout + result.stderr
        )


def _copy_atomic(source: Path, destination: Path) -> None:
    temporary = destination.with_name(destination.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
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
