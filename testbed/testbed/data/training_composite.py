"""Build immutable, provenance-rich episode views for policy training.

The ACT loader currently consumes one directory containing ``episode_N.hdf5``
files.  This module composes multiple already-QC'd datasets without copying or
rewriting HDF5 payloads: every output episode is an absolute symbolic link and
the manifest records its source episode, role, and SHA-256.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import yaml


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_mapping(path: str | Path) -> dict[str, Any]:
    """Load a JSON/YAML mapping and reject ambiguous top-level values."""

    source = Path(path).expanduser().resolve()
    payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in {source}")
    return payload


def build_training_composite(
    spec_path: str | Path,
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Build one training-only symlink view from a declarative source spec.

    Existing output directories are never overwritten.  This makes the view
    disposable and keeps all source HDF5 files immutable.
    """

    spec_file = Path(spec_path).expanduser().resolve()
    spec = load_mapping(spec_file)
    if int(spec.get("schema_version", 0)) != 1:
        raise ValueError("training composite spec schema_version must be 1")

    raw_target = output_dir if output_dir is not None else spec.get("output_dir")
    if not raw_target:
        raise ValueError("training composite spec requires output_dir")
    target = Path(raw_target).expanduser().resolve()
    if target.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing training composite: {target}"
        )

    raw_sources = spec.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError("training composite spec requires non-empty sources")
    forbidden = {int(value) for value in spec.get("forbidden_source_episode_ids", [])}

    records: list[dict[str, Any]] = []
    seen_composite_ids: set[int] = set()
    seen_source_names: set[str] = set()
    source_summaries: list[dict[str, Any]] = []

    for source in raw_sources:
        if not isinstance(source, dict):
            raise ValueError("each training composite source must be a mapping")
        name = str(source.get("name", "")).strip()
        if not name or name in seen_source_names:
            raise ValueError(f"source name must be non-empty and unique: {name!r}")
        seen_source_names.add(name)

        dataset_dir = Path(str(source.get("dataset_dir", ""))).expanduser().resolve()
        split_path = Path(str(source.get("split_path", ""))).expanduser().resolve()
        if not dataset_dir.is_dir():
            raise FileNotFoundError(f"source dataset_dir does not exist: {dataset_dir}")
        if not split_path.is_file():
            raise FileNotFoundError(f"source split_path does not exist: {split_path}")
        split = load_mapping(split_path)
        saved_dataset = str(split.get("dataset_dir", ""))
        if saved_dataset and Path(saved_dataset).expanduser().resolve() != dataset_dir:
            raise ValueError(
                f"source split dataset_dir mismatch for {name}: "
                f"{saved_dataset} != {dataset_dir}"
            )

        roles = [str(role) for role in source.get("include_roles", [])]
        if not roles or any(role not in {"train", "val"} for role in roles):
            raise ValueError(f"source {name} include_roles must contain train/val")
        composite_id = int(source.get("composite_start", -1))
        if composite_id < 0:
            raise ValueError(f"source {name} composite_start must be non-negative")

        source_count = 0
        role_counts: dict[str, int] = {}
        for role in roles:
            source_ids = [int(value) for value in split.get(f"{role}_ids", [])]
            if not source_ids:
                raise ValueError(f"source {name} has no {role}_ids")
            blocked = sorted(set(source_ids) & forbidden)
            if blocked:
                raise ValueError(
                    f"source {name} {role} includes forbidden episode ids: {blocked}"
                )
            role_counts[role] = len(source_ids)
            for source_episode_id in source_ids:
                while composite_id in seen_composite_ids:
                    composite_id += 1
                source_path = dataset_dir / f"episode_{source_episode_id}.hdf5"
                if not source_path.is_file():
                    raise FileNotFoundError(
                        f"source episode does not exist: {source_path}"
                    )
                records.append(
                    {
                        "composite_episode_id": composite_id,
                        "role": role,
                        "source_name": name,
                        "source_episode_id": source_episode_id,
                        "source_path": str(source_path),
                        "source_sha256": sha256_file(source_path),
                    }
                )
                seen_composite_ids.add(composite_id)
                composite_id += 1
                source_count += 1

        source_summaries.append(
            {
                "name": name,
                "dataset_dir": str(dataset_dir),
                "split_path": str(split_path),
                "split_sha256": sha256_file(split_path),
                "included_roles": roles,
                "role_counts": role_counts,
                "episode_count": source_count,
            }
        )

    train_ids = [r["composite_episode_id"] for r in records if r["role"] == "train"]
    val_ids = [r["composite_episode_id"] for r in records if r["role"] == "val"]
    if not train_ids or not val_ids:
        raise ValueError("training composite must contain non-empty train and val roles")
    if set(train_ids) & set(val_ids):
        raise ValueError("training composite train and val ids overlap")

    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f".{target.name}.partial-{os.getpid()}")
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir()
    try:
        for record in records:
            link = partial / f"episode_{record['composite_episode_id']}.hdf5"
            link.symlink_to(record["source_path"])

        final_split_path = target / "train_val_split.yaml"
        split_payload = {
            "schema_version": 1,
            "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "dataset_dir": str(target),
            "requested_num_episodes": len(records),
            "available_episode_ids": sorted(seen_composite_ids),
            "split_seed": None,
            "train_split_ratio": len(train_ids) / len(records),
            "train_ids": train_ids,
            "val_ids": val_ids,
            "reused_existing_split": True,
            "provenance": {
                "contract": "training_composite_explicit_roles_v1",
                "spec_path": str(spec_file),
                "spec_sha256": sha256_file(spec_file),
                "forbidden_source_episode_ids": sorted(forbidden),
            },
        }
        (partial / "train_val_split.yaml").write_text(
            yaml.safe_dump(split_payload, sort_keys=False), encoding="utf-8"
        )

        manifest = {
            "schema_version": 1,
            "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "contract": "training_composite_symlink_view_v1",
            "dataset_dir": str(target),
            "train_ready_episode_ids": sorted(seen_composite_ids),
            "train_ids": train_ids,
            "val_ids": val_ids,
            "test_ids": [],
            "selection_boundary": "train and validation only; no test episodes linked",
            "forbidden_source_episode_ids": sorted(forbidden),
            "spec_path": str(spec_file),
            "spec_sha256": sha256_file(spec_file),
            "split_path": str(final_split_path),
            "sources": source_summaries,
            "episodes": records,
        }
        manifest_path = partial / "train_ready_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        manifest_digest = sha256_file(manifest_path)
        (partial / "train_ready_manifest.sha256").write_text(
            f"{manifest_digest}  train_ready_manifest.json\n", encoding="utf-8"
        )
        os.replace(partial, target)
    except Exception:
        shutil.rmtree(partial, ignore_errors=True)
        raise

    return validate_training_composite(target, verify_hashes=False)


def validate_training_composite(
    dataset_dir: str | Path,
    *,
    verify_hashes: bool = False,
) -> dict[str, Any]:
    """Validate links, split membership, and optionally all source hashes."""

    root = Path(dataset_dir).expanduser().resolve()
    manifest_path = root / "train_ready_manifest.json"
    split_path = root / "train_val_split.yaml"
    manifest = load_mapping(manifest_path)
    split = load_mapping(split_path)
    if Path(str(manifest.get("dataset_dir", ""))).resolve() != root:
        raise ValueError("training composite manifest dataset_dir mismatch")
    if Path(str(split.get("dataset_dir", ""))).resolve() != root:
        raise ValueError("training composite split dataset_dir mismatch")

    records = manifest.get("episodes")
    if not isinstance(records, list) or not records:
        raise ValueError("training composite manifest has no episodes")
    record_ids = {int(record["composite_episode_id"]) for record in records}
    train_ids = {int(value) for value in split.get("train_ids", [])}
    val_ids = {int(value) for value in split.get("val_ids", [])}
    if train_ids & val_ids or train_ids | val_ids != record_ids:
        raise ValueError("training composite split does not exactly cover manifest")

    for record in records:
        composite_id = int(record["composite_episode_id"])
        link = root / f"episode_{composite_id}.hdf5"
        source = Path(str(record["source_path"])).resolve()
        if not link.is_symlink() or link.resolve() != source:
            raise ValueError(f"training composite link mismatch: {link}")
        if verify_hashes and sha256_file(source) != str(record["source_sha256"]):
            raise ValueError(f"training composite source hash mismatch: {source}")

    return {
        "status": "valid",
        "dataset_dir": str(root),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "split_path": str(split_path),
        "split_sha256": sha256_file(split_path),
        "episode_count": len(records),
        "train_count": len(train_ids),
        "val_count": len(val_ids),
        "hashes_verified": bool(verify_hashes),
    }
