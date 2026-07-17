"""Build a portable, hash-pinned ACT runtime bundle."""

from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
from pathlib import Path
from typing import Any

from testbed.policies.act.deployment_preflight import (
    BUNDLE_MANIFEST_FILENAME,
    BUNDLE_SCHEMA_VERSION,
    REQUIRED_BUNDLE_FILES,
    read_yaml_mapping,
    sha256_file,
)


def package_act_runtime_bundle(
    *,
    source_dir: str | Path,
    output_dir: str | Path,
    candidate_id: str,
) -> dict[str, Any]:
    """Copy the minimum runtime files and write a portable identity manifest."""

    source = Path(source_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    identity = str(candidate_id).strip()
    if not identity:
        raise ValueError("candidate_id must not be empty")
    missing = [source / name for name in REQUIRED_BUNDLE_FILES if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "source bundle is missing required file(s): "
            + ", ".join(str(path) for path in missing)
        )
    run_metadata = json.loads((source / "run_metadata.json").read_text(encoding="utf-8"))
    if not isinstance(run_metadata, dict) or run_metadata.get("status") != "completed":
        raise ValueError("run_metadata.json must declare status=completed")
    resolved = read_yaml_mapping(source / "resolved_config.yaml")

    output.mkdir(parents=True, exist_ok=True)
    unexpected = [path for path in output.iterdir() if path.name not in (*REQUIRED_BUNDLE_FILES, BUNDLE_MANIFEST_FILENAME)]
    if unexpected:
        raise ValueError(
            "output bundle contains unexpected file(s): "
            + ", ".join(str(path) for path in unexpected)
        )

    file_records: list[dict[str, Any]] = []
    for name in REQUIRED_BUNDLE_FILES:
        source_path = source / name
        target_path = output / name
        source_sha = sha256_file(source_path)
        if target_path.exists() and sha256_file(target_path) != source_sha:
            raise ValueError(f"refusing to overwrite different bundle file: {target_path}")
        if not target_path.exists():
            temporary = output / f".{name}.tmp"
            try:
                shutil.copy2(source_path, temporary)
                if sha256_file(temporary) != source_sha:
                    raise OSError(f"copied file failed SHA-256 verification: {name}")
                os.replace(temporary, target_path)
            finally:
                if temporary.exists():
                    temporary.unlink()
        file_records.append(
            {
                "name": name,
                "size_bytes": target_path.stat().st_size,
                "sha256": source_sha,
            }
        )

    manifest = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "candidate_id": identity,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source_dir": str(source),
        "task_name": str(resolved.get("task", {}).get("task_name", "")),
        "camera_names": list(resolved.get("task", {}).get("camera_names", [])),
        "low_dim_keys": list(resolved.get("policy", {}).get("low_dim_keys", [])),
        "files": file_records,
    }
    manifest_path = output / BUNDLE_MANIFEST_FILENAME
    temporary_manifest = output / f".{BUNDLE_MANIFEST_FILENAME}.tmp"
    try:
        temporary_manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_manifest, manifest_path)
    finally:
        if temporary_manifest.exists():
            temporary_manifest.unlink()
    return {
        "ok": True,
        "bundle_dir": str(output),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "candidate_id": identity,
        "files": file_records,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    args = parser.parse_args(argv)
    report = package_act_runtime_bundle(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        candidate_id=args.candidate_id,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
