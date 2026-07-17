"""Build causal execution-feedback sidecars for an explicit train/val split."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from testbed.data.execution_feedback import (
    ALIGNMENT_MODE,
    COUNTERFACTUAL_MODE,
    COUNTERFACTUAL_VARIANT_COUNT,
    MANIFEST_FILENAME,
    MANIFEST_SCHEMA_VERSION,
    build_episode_execution_feedback,
    load_split_episode_ids,
    sha256_file,
    validate_execution_feedback_manifest,
)


def build_execution_feedback_sidecars(
    *,
    dataset_dir: str | Path,
    split_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Build sidecars only for IDs explicitly named by train_ids and val_ids."""

    dataset = Path(dataset_dir).expanduser().resolve()
    split = Path(split_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if not dataset.is_dir():
        raise FileNotFoundError(f"dataset_dir does not exist: {dataset}")
    if not split.is_file():
        raise FileNotFoundError(f"split_path does not exist: {split}")
    train_ids, val_ids = load_split_episode_ids(split)
    episode_ids = _ordered_union(train_ids, val_ids)
    output.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for episode_id in episode_ids:
        resampled_path = dataset / f"episode_{episode_id}.hdf5"
        if not resampled_path.is_file():
            raise FileNotFoundError(
                f"split episode {episode_id} is missing from dataset_dir: {resampled_path}"
            )
        sidecar_path = output / f"episode_{episode_id}.execution_feedback.npz"
        record = build_episode_execution_feedback(
            episode_id=episode_id,
            resampled_path=resampled_path,
            output_path=sidecar_path,
        )
        memberships: list[str] = []
        if episode_id in train_ids:
            memberships.append("train")
        if episode_id in val_ids:
            memberships.append("val")
        record["split_membership"] = memberships
        records.append(record)

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "alignment_mode": ALIGNMENT_MODE,
        "strict_causal_comparator": "command_send_timestamp_ns < observation_timestamp_ns",
        "reset_semantics": (
            "zero at episode start and train-exclude/source-gap samples; "
            "post-reset commands require send timestamp at or after latest reset"
        ),
        "counterfactual_mode": COUNTERFACTUAL_MODE,
        "counterfactual_variant_count": COUNTERFACTUAL_VARIANT_COUNT,
        "dataset_dir": str(dataset),
        "split_path": str(split),
        "split_sha256": sha256_file(split),
        "output_dir": str(output),
        "train_ids": train_ids,
        "val_ids": val_ids,
        "episode_ids": episode_ids,
        "episodes": records,
    }
    manifest_path = output / MANIFEST_FILENAME
    _write_json_atomic(manifest_path, manifest)
    validate_execution_feedback_manifest(
        manifest_path,
        verify_hashes=False,
        expected_dataset_dir=dataset,
        expected_split_path=split,
    )
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build strict-causal previous-command sidecars for explicit train/val IDs."
        )
    )
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--split-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    manifest = build_execution_feedback_sidecars(
        dataset_dir=args.dataset_dir,
        split_path=args.split_path,
        output_dir=args.output_dir,
    )
    print(
        f"wrote {Path(args.output_dir).resolve() / MANIFEST_FILENAME} "
        f"episodes={len(manifest['episode_ids'])}"
    )


def _ordered_union(first: list[int], second: list[int]) -> list[int]:
    result: list[int] = []
    seen: set[int] = set()
    for episode_id in [*first, *second]:
        if episode_id not in seen:
            seen.add(episode_id)
            result.append(episode_id)
    return result


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


if __name__ == "__main__":
    main()
