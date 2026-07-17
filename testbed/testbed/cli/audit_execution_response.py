"""Build execution-response sidecars from existing train-ready HDF5 episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from testbed.data.execution_response import (
    DEFAULT_RESPONSE_HORIZONS,
    DEFAULT_SUPPORTED_AXES,
    build_execution_response_episode,
    summarize_response_latency,
    write_event_tables,
    write_execution_response_episode,
)

DEFAULT_POSITIVE = (0.661, 0.259, 0.5, 0.408)
DEFAULT_NEGATIVE = (0.721, 0.357, 0.5, 0.508)
DEFAULT_QVEL_NOISE = (0.07044806, 0.02980213, 0.02067453, 0.07763434)
DEFAULT_QVEL_NOISE_PROVENANCE = (
    "train-only stationary centered nine-step expert windows; "
    "max(3*population_std, 0.006) per axis"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit direct-domain command-to-qvel response using existing HDF5 "
            "episodes; source files are never modified."
        )
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--deadzone-json", type=Path, required=True)
    parser.add_argument("--episode-id", action="append", type=int, default=None)
    parser.add_argument(
        "--supported-axis",
        action="append",
        dest="supported_axes",
        default=None,
    )
    parser.add_argument(
        "--response-horizon",
        action="append",
        type=int,
        dest="response_horizons",
        default=None,
    )
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    deadzone_payload = json.loads(args.deadzone_json.read_text(encoding="utf-8"))
    deadzone = deadzone_payload["deadzone_action"]
    positive = [float(deadzone[axis]["pos"]) for axis in ("swing", "boom", "stick", "bucket")]
    negative = [float(deadzone[axis]["neg"]) for axis in ("swing", "boom", "stick", "bucket")]
    supported_axes = tuple(args.supported_axes or DEFAULT_SUPPORTED_AXES)
    horizons = tuple(args.response_horizons or DEFAULT_RESPONSE_HORIZONS)
    episode_ids = args.episode_id
    if episode_ids is None:
        episode_paths = sorted(
            dataset_dir.glob("episode_*.hdf5"),
            key=lambda path: int(path.stem.split("_", 1)[1]),
        )
    else:
        episode_paths = [dataset_dir / f"episode_{episode_id}.hdf5" for episode_id in episode_ids]

    records = []
    event_rows = []
    for episode_path in episode_paths:
        result = build_execution_response_episode(
            resampled_path=episode_path,
            positive_threshold=positive,
            negative_threshold=negative,
            qvel_noise=DEFAULT_QVEL_NOISE,
            supported_axes=supported_axes,
            response_horizons=horizons,
        )
        records.append(
            write_execution_response_episode(
                result,
                output_dir=output_dir,
                positive_threshold=positive,
                negative_threshold=negative,
                qvel_noise=DEFAULT_QVEL_NOISE,
                supported_axes=supported_axes,
                response_horizons=horizons,
                qvel_noise_provenance=DEFAULT_QVEL_NOISE_PROVENANCE,
            )
        )
        event_rows.extend(result.event_rows)

    write_event_tables(
        event_rows,
        jsonl_path=output_dir / "execution_response_events.jsonl",
        csv_path=output_dir / "execution_response_events.csv",
    )
    latency_summary = summarize_response_latency(
        event_rows,
        response_horizons=horizons,
    )
    (output_dir / "execution_response_latency_summary.json").write_text(
        json.dumps(latency_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "label_contract": "direct_command_qvel_response_v1",
        "dataset_dir": str(dataset_dir),
        "deadzone_json": str(args.deadzone_json.expanduser().resolve()),
        "deadzone_json_sha256": _sha256(args.deadzone_json),
        "action_domain": deadzone_payload.get("metadata", {}).get(
            "action_domain", "unknown"
        ),
        "policy_action_scale": deadzone_payload.get("metadata", {}).get(
            "policy_action_scale", []
        ),
        "positive_threshold": positive,
        "negative_threshold": negative,
        "qvel_noise": list(DEFAULT_QVEL_NOISE),
        "qvel_noise_provenance": DEFAULT_QVEL_NOISE_PROVENANCE,
        "supported_axes": list(supported_axes),
        "unsupported_axes": [
            axis
            for axis in ("swing", "boom", "stick", "bucket")
            if axis not in supported_axes
        ],
        "response_horizons": list(horizons),
        "episode_ids": [record["episode_id"] for record in records],
        "episodes": records,
        "event_rows": len(event_rows),
        "latency_summary": str(output_dir / "execution_response_latency_summary.json"),
        "source_episodes_unchanged": True,
    }
    manifest_path = output_dir / "execution_response_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
