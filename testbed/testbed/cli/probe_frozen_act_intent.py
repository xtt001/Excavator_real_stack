"""Run matched frozen ACT decoder-feature intent probes."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import torch
import yaml

from testbed.actions.policy import load_act_policy_from_bundle
from testbed.policies.act.frozen_intent_probe import (
    AXIS_NAMES,
    FeatureCache,
    FrozenIntentFrameDataset,
    anchor_prediction_rows,
    build_cache_identity,
    build_startup_anchor_inventory,
    cache_key,
    class_counts,
    evaluate_intent_predictions,
    extract_frozen_features,
    load_feature_cache,
    predict_linear_probe,
    save_feature_cache,
    save_probe_weights,
    sha256_file,
    train_linear_probe,
    validate_startup_anchor_contract,
    write_compact_plots,
    write_metrics_csv,
    write_rows_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m testbed.cli.probe_frozen_act_intent",
        description=(
            "Capture frozen query-0 ACT decoder features and train fixed linear "
            "per-axis neg/idle/pos probes."
        ),
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--split-path", type=Path, required=True)
    parser.add_argument(
        "--bundle",
        action="append",
        required=True,
        help="Matched model specification NAME=/absolute/bundle/path.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--probe-epochs", type=int, default=50)
    parser.add_argument("--probe-batch-size", type=int, default=1024)
    parser.add_argument("--probe-learning-rate", type=float, default=3e-3)
    parser.add_argument("--probe-weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--deadzone-positive", type=float, nargs=4, required=True
    )
    parser.add_argument(
        "--deadzone-negative", type=float, nargs=4, required=True
    )
    parser.add_argument(
        "--max-frames-per-split",
        type=int,
        default=None,
        help="Fixed plumbing-smoke limit; omit for the formal natural distribution.",
    )
    args = parser.parse_args()
    run_probe(args)


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "run_state.json", {"status": "in_progress"})
    started = time.perf_counter()
    split = _read_yaml(args.split_path)
    train_ids = [int(value) for value in split["train_ids"]]
    validation_ids = [int(value) for value in split["val_ids"]]
    if set(train_ids) & set(validation_ids):
        raise ValueError("train and validation episode IDs overlap")
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    split_dataset = Path(str(split.get("dataset_dir", ""))).expanduser().resolve()
    if split_dataset != dataset_dir:
        raise ValueError(
            f"split dataset mismatch: {split_dataset} vs {dataset_dir}"
        )
    bundle_specs = _bundle_specs(args.bundle)
    thresholds = {
        axis: {"pos": float(pos), "neg": float(neg)}
        for axis, pos, neg in zip(
            AXIS_NAMES, args.deadzone_positive, args.deadzone_negative
        )
    }
    episode_paths = {
        str(episode_id): dataset_dir / f"episode_{episode_id}.hdf5"
        for episode_id in [*train_ids, *validation_ids]
    }
    missing = [str(path) for path in episode_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing episode(s): " + ", ".join(missing))
    provenance_started = time.perf_counter()
    episode_sha = {
        episode_id: sha256_file(path)
        for episode_id, path in episode_paths.items()
    }
    provenance_seconds = time.perf_counter() - provenance_started
    split_sha = sha256_file(args.split_path)
    expected_startup_inventory = build_startup_anchor_inventory(
        dataset_dir=dataset_dir,
        episode_ids=validation_ids,
        thresholds=thresholds,
    )
    model_results: dict[str, Any] = {}
    model_metrics: dict[str, Any] = {}
    anchor_rows: list[dict[str, Any]] = []

    for model_name, bundle_dir in bundle_specs.items():
        model_started = time.perf_counter()
        resolved_path = bundle_dir / "resolved_config.yaml"
        checkpoint_path = bundle_dir / "policy_best.ckpt"
        stats_path = bundle_dir / "dataset_stats.pkl"
        for required in (resolved_path, checkpoint_path, stats_path):
            if not required.is_file():
                raise FileNotFoundError(required)
        resolved = _read_yaml(resolved_path)
        camera_names = [str(value) for value in resolved["task"]["camera_names"]]
        low_dim_keys = list(resolved["policy"].get("low_dim_keys", ["qpos"]))
        if low_dim_keys != ["qpos"]:
            raise ValueError(
                f"{model_name} is not the fixed qpos-only comparison: {low_dim_keys}"
            )
        image_transform = str(resolved["train"].get("image_transform", "none"))
        resolved_split = Path(
            str(resolved["train"].get("split_path", ""))
        ).expanduser().resolve()
        resolved_split_payload = _read_yaml(resolved_split)
        if (
            [int(value) for value in resolved_split_payload.get("train_ids", [])]
            != train_ids
            or [
                int(value) for value in resolved_split_payload.get("val_ids", [])
            ]
            != validation_ids
            or Path(
                str(resolved_split_payload.get("dataset_dir", ""))
            ).expanduser().resolve()
            != dataset_dir
        ):
            raise ValueError(
                f"{model_name} resolved split content mismatch: {resolved_split}"
            )
        hash_started = time.perf_counter()
        checkpoint_sha = sha256_file(checkpoint_path)
        resolved_sha = sha256_file(resolved_path)
        stats_sha = sha256_file(stats_path)
        hash_seconds = time.perf_counter() - hash_started
        identity = build_cache_identity(
            model_name=model_name,
            checkpoint_sha256=checkpoint_sha,
            resolved_config_sha256=resolved_sha,
            stats_sha256=stats_sha,
            split_sha256=split_sha,
            camera_names=camera_names,
            image_transform=image_transform,
            train_episode_ids=train_ids,
            validation_episode_ids=validation_ids,
            thresholds=thresholds,
            episode_sha256=episode_sha,
            frame_limit_per_split=args.max_frames_per_split,
        )
        load_started = time.perf_counter()
        adapter = load_act_policy_from_bundle(
            bundle_dir=bundle_dir,
            device=str(args.device),
            temporal_agg=False,
        )
        load_seconds = time.perf_counter() - load_started
        partitions: dict[str, FeatureCache] = {}
        cache_manifests: dict[str, Any] = {}
        try:
            for partition, episode_ids in (
                ("train", train_ids),
                ("validation", validation_ids),
            ):
                partition_identity = {**identity, "partition": partition}
                key = cache_key(partition_identity)
                cache_path = output_dir / "feature_cache" / f"{key}.npz"
                if cache_path.exists() or cache_path.with_suffix(".json").exists():
                    cache = load_feature_cache(
                        cache_path, expected_identity=partition_identity
                    )
                    cache_manifest = dict(cache.metadata)
                    cache_manifest["cache_reused"] = True
                else:
                    dataset = FrozenIntentFrameDataset(
                        dataset_dir=dataset_dir,
                        episode_ids=episode_ids,
                        camera_names=camera_names,
                        thresholds=thresholds,
                        image_transform=image_transform,
                        max_frames=args.max_frames_per_split,
                    )
                    try:
                        cache = extract_frozen_features(
                            adapter=adapter,
                            dataset=dataset,
                            batch_size=int(args.batch_size),
                            num_workers=int(args.num_workers),
                            prefetch_factor=int(args.prefetch_factor),
                        )
                    finally:
                        dataset.close()
                    cache_manifest = save_feature_cache(
                        cache_path,
                        cache=cache,
                        identity=partition_identity,
                    )
                    cache_manifest["cache_reused"] = False
                partitions[partition] = cache
                cache_manifests[partition] = {
                    **cache_manifest,
                    "path": str(cache_path),
                }
        finally:
            del adapter
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        train_cache = partitions["train"]
        validation_cache = partitions["validation"]
        probe_started = time.perf_counter()
        probe = train_linear_probe(
            train_cache.features,
            train_cache.labels,
            epochs=int(args.probe_epochs),
            learning_rate=float(args.probe_learning_rate),
            weight_decay=float(args.probe_weight_decay),
            batch_size=int(args.probe_batch_size),
            seed=int(args.seed),
            device=str(args.device),
        )
        probe_train_seconds = time.perf_counter() - probe_started
        eval_started = time.perf_counter()
        validation_probabilities, _ = predict_linear_probe(
            probe, validation_cache.features
        )
        metrics = evaluate_intent_predictions(
            labels=validation_cache.labels,
            probabilities=validation_probabilities,
            anchor_mask=validation_cache.anchor_mask,
            startup_mask=validation_cache.startup_mask,
            mid_cycle_mask=validation_cache.mid_cycle_mask,
        )
        evaluation_seconds = time.perf_counter() - eval_started
        rows = anchor_prediction_rows(
            model_name=model_name,
            cache=validation_cache,
            probabilities=validation_probabilities,
        )
        startup_rows = [row for row in rows if row["group"] == "startup"]
        if args.max_frames_per_split is None:
            validate_startup_anchor_contract(
                observed_rows=startup_rows,
                expected_inventory=expected_startup_inventory,
            )
        weights_path = output_dir / f"{model_name}_linear_probe.npz"
        weights_sha = save_probe_weights(weights_path, probe=probe)
        model_metrics[model_name] = metrics
        anchor_rows.extend(rows)
        model_results[model_name] = {
            "bundle_dir": str(bundle_dir),
            "camera_names": camera_names,
            "image_transform": image_transform,
            "checkpoint_sha256": checkpoint_sha,
            "resolved_config_sha256": resolved_sha,
            "dataset_stats_sha256": stats_sha,
            "cache_identity": identity,
            "cache": cache_manifests,
            "train_frame_count": int(train_cache.features.shape[0]),
            "validation_frame_count": int(validation_cache.features.shape[0]),
            "train_class_counts": class_counts(train_cache.labels),
            "validation_class_counts": class_counts(validation_cache.labels),
            "linear_probe": {
                "head": "Linear(512, 12), reshaped to 4x3",
                "class_order": ["neg", "idle", "pos"],
                "class_weight_rule": (
                    "train-only inverse frequency N/(K*n_c), normalized to "
                    "mean 1 over present classes; absent classes weight 0"
                ),
                "class_weights": probe.class_weights.tolist(),
                "epochs": probe.epochs,
                "learning_rate": probe.learning_rate,
                "weight_decay": probe.weight_decay,
                "batch_size": int(args.probe_batch_size),
                "seed": probe.seed,
                "final_train_loss": float(probe.train_loss[-1]),
                "weights_path": str(weights_path),
                "weights_sha256": weights_sha,
            },
            "metrics": metrics,
            "startup_anchor_count": len(startup_rows),
            "startup_anchor_contract_verified": (
                args.max_frames_per_split is None
            ),
            "timings_seconds": {
                "artifact_hashing": hash_seconds,
                "model_loading": load_seconds,
                "train_feature_extraction": float(
                    train_cache.metadata.get("extraction", train_cache.metadata).get(
                        "wall_seconds", 0.0
                    )
                ),
                "validation_feature_extraction": float(
                    validation_cache.metadata.get(
                        "extraction", validation_cache.metadata
                    ).get("wall_seconds", 0.0)
                ),
                "probe_training": probe_train_seconds,
                "evaluation": evaluation_seconds,
                "model_total": time.perf_counter() - model_started,
            },
        }

    metrics_path = output_dir / "metrics.csv"
    anchors_path = output_dir / "anchor_predictions.csv"
    write_metrics_csv(metrics_path, model_metrics=model_metrics)
    write_rows_csv(anchors_path, anchor_rows)
    plot_paths = write_compact_plots(output_dir, model_metrics=model_metrics)
    experiment = {
        "schema_version": 1,
        "status": "complete",
        "capability_question": (
            "Are frozen inference-time ACT query-0 decoder features linearly "
            "separable for per-axis executable neg/idle/pos intent?"
        ),
        "runtime_effect": "none",
        "future_action_leakage": "none: model forward actions=None, zero inference latent",
        "dataset_dir": str(dataset_dir),
        "split_path": str(Path(args.split_path).expanduser().resolve()),
        "split_sha256": split_sha,
        "train_episode_ids": train_ids,
        "validation_episode_ids": validation_ids,
        "train_episode_count": len(train_ids),
        "validation_episode_count": len(validation_ids),
        "deadzone_thresholds": thresholds,
        "validation_startup_anchor_contract": {
            "source": "validation episode actions plus direct-output deadzones",
            "anchor_count": len(expected_startup_inventory),
            "inventory": expected_startup_inventory,
            "verified_for_all_models": args.max_frames_per_split is None,
        },
        "episode_sha256": episode_sha,
        "formal_natural_distribution": args.max_frames_per_split is None,
        "frame_limit_per_split": args.max_frames_per_split,
        "protocol": {
            "feature": "ACT decoder hidden state at query 0 before action_head",
            "latent": "inference zero latent",
            "actions_argument": None,
            "probe_family": ["linear"],
            "validation_tuning": "none; fixed epochs and train-only weights",
            "num_workers": int(args.num_workers),
            "prefetch_factor": int(args.prefetch_factor),
        },
        "models": model_results,
        "timings_seconds": {
            "episode_provenance_hashing": provenance_seconds,
            "total": time.perf_counter() - started,
        },
    }
    experiment_path = output_dir / "experiment.json"
    _write_json(experiment_path, experiment)
    artifact_paths = [
        experiment_path,
        metrics_path,
        anchors_path,
        *plot_paths,
        *sorted((output_dir / "feature_cache").glob("*")),
        *sorted(output_dir.glob("*_linear_probe.npz")),
    ]
    manifest = {
        "status": "complete",
        "artifacts": [
            {
                "path": str(path),
                "size_bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
            for path in artifact_paths
            if path.is_file()
        ],
    }
    _write_json(output_dir / "artifact_manifest.json", manifest)
    _write_json(
        output_dir / "run_state.json",
        {
            "status": "complete",
            "experiment_path": str(experiment_path),
            "experiment_sha256": sha256_file(experiment_path),
            "total_wall_seconds": time.perf_counter() - started,
        },
    )
    return experiment


def _bundle_specs(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for raw in values:
        if "=" not in raw:
            raise ValueError(f"bundle must be NAME=PATH, got {raw!r}")
        name, path_text = raw.split("=", 1)
        name = name.strip()
        if not name or name in result:
            raise ValueError(f"invalid or duplicate model name: {name!r}")
        path = Path(path_text).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(path)
        result[name] = path
    return result


def _read_yaml(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
