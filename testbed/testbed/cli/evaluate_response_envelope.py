"""Build a train-calibrated all-axis response envelope and validate it."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from testbed.data.execution_response import (
    build_execution_response_episode,
    write_event_tables,
)
from testbed.data.execution_response_envelope import (
    AXIS_NAMES,
    DEFAULT_HORIZONS,
    calibrate_response_contract,
    calibration_to_dict,
    evaluate_response_envelope,
    extract_response_events,
    fit_response_envelope,
    sequence_from_execution_response,
    summarize_response_events,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build an all-axis historical command-response envelope from train "
            "episodes and validate support/calibration on a frozen validation split."
        )
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--split-yaml", type=Path, required=True)
    parser.add_argument("--deadzone-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--response-horizon", action="append", type=int, default=None)
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.expanduser().resolve()
    split_path = args.split_yaml.expanduser().resolve()
    deadzone_path = args.deadzone_json.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory must be absent or empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    split = yaml.safe_load(split_path.read_text(encoding="utf-8"))
    train_ids = [int(value) for value in split["train_ids"]]
    validation_ids = [int(value) for value in split["val_ids"]]
    deadzone_payload = json.loads(deadzone_path.read_text(encoding="utf-8"))
    deadzone = deadzone_payload["deadzone_action"]
    positive = [float(deadzone[axis]["pos"]) for axis in AXIS_NAMES]
    negative = [float(deadzone[axis]["neg"]) for axis in AXIS_NAMES]
    horizons = tuple(args.response_horizon or DEFAULT_HORIZONS)

    sequences = []
    source_records = []
    for split_name, episode_ids in (("train", train_ids), ("validation", validation_ids)):
        for dataset_episode_id in episode_ids:
            path = dataset_dir / f"episode_{dataset_episode_id}.hdf5"
            result = build_execution_response_episode(
                resampled_path=path,
                positive_threshold=positive,
                negative_threshold=negative,
                qvel_noise=(0.0, 0.0, 0.0, 0.0),
                supported_axes=AXIS_NAMES,
                response_horizons=(1,),
            )
            sequences.append(
                sequence_from_execution_response(
                    result,
                    dataset_episode_id=dataset_episode_id,
                    split=split_name,
                )
            )
            source_records.append(
                {
                    "dataset_episode_id": dataset_episode_id,
                    "source_episode_id": result.episode_id,
                    "split": split_name,
                    "resampled_path": str(result.resampled_path),
                    "raw_source_path": str(result.raw_source_path),
                }
            )

    train_sequences = [sequence for sequence in sequences if sequence.split == "train"]
    validation_sequences = [
        sequence for sequence in sequences if sequence.split == "validation"
    ]
    calibration = calibrate_response_contract(
        train_sequences,
        positive_threshold=positive,
        negative_threshold=negative,
    )
    train_rows = extract_response_events(
        train_sequences,
        calibration=calibration,
        positive_threshold=positive,
        negative_threshold=negative,
        response_horizons=horizons,
    )
    validation_rows = extract_response_events(
        validation_sequences,
        calibration=calibration,
        positive_threshold=positive,
        negative_threshold=negative,
        response_horizons=horizons,
    )
    envelope = fit_response_envelope(
        train_rows,
        response_horizons=horizons,
    )
    validation = evaluate_response_envelope(validation_rows, envelope)

    _write_json(output_dir / "response_calibration.json", calibration_to_dict(calibration))
    _write_json(output_dir / "train_response_summary.json", summarize_response_events(train_rows, response_horizons=horizons))
    _write_json(output_dir / "validation_response_summary.json", summarize_response_events(validation_rows, response_horizons=horizons))
    _write_json(output_dir / "response_envelope.json", envelope)
    _write_json(output_dir / "validation_envelope_evaluation.json", validation)
    write_event_tables(
        train_rows,
        jsonl_path=output_dir / "train_response_events.jsonl",
        csv_path=output_dir / "train_response_events.csv",
    )
    write_event_tables(
        validation_rows,
        jsonl_path=output_dir / "validation_response_events.jsonl",
        csv_path=output_dir / "validation_response_events.csv",
    )
    manifest = {
        "schema_version": "all_axis_response_envelope_v1",
        "dataset_dir": str(dataset_dir),
        "split_yaml": str(split_path),
        "split_yaml_sha256": _sha256(split_path),
        "deadzone_json": str(deadzone_path),
        "deadzone_json_sha256": _sha256(deadzone_path),
        "positive_threshold": positive,
        "negative_threshold": negative,
        "response_horizons": list(horizons),
        "train_episode_ids": train_ids,
        "validation_episode_ids": validation_ids,
        "sealed_test_read": False,
        "source_records": source_records,
        "artifacts": {
            name: _sha256(output_dir / name)
            for name in (
                "response_calibration.json",
                "train_response_summary.json",
                "validation_response_summary.json",
                "response_envelope.json",
                "validation_envelope_evaluation.json",
                "train_response_events.jsonl",
                "train_response_events.csv",
                "validation_response_events.jsonl",
                "validation_response_events.csv",
            )
        },
        "capability_boundary": (
            "historical operator-command response only; unsent model command "
            "response and closed-loop task success remain unknown"
        ),
    }
    manifest_path = output_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
