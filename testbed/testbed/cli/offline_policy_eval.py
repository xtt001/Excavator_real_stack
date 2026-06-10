"""tb-offline-policy-eval — replay one HDF5 episode through a policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from testbed.policies.offline_eval import (
    aggregate_episode_results,
    choose_default_ckpt,
    default_collection_output_dir,
    default_output_dir,
    episode_path,
    evaluate_episode,
    load_episode_features,
    load_policy_for_episode,
    load_train_ready_episode_ids,
    normalize_episode_id,
    select_representative_episode,
    write_collection_report,
    write_eval_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="tb-offline-policy-eval",
        description="Replay one real-excavator HDF5 episode through an ACT policy.",
    )
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=Path("runs/ckpts/real_excavation_act_20hz_v1"),
        help="Directory containing checkpoint, dataset_stats.pkl and resolved_config.yaml.",
    )
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=None,
        help="Checkpoint path. Defaults to policy_best.ckpt, policy_latest.ckpt, then newest epoch ckpt.",
    )
    parser.add_argument(
        "--resolved-config",
        type=Path,
        default=None,
        help="Resolved config path. Defaults to <bundle-dir>/resolved_config.yaml.",
    )
    parser.add_argument(
        "--stats",
        type=Path,
        default=None,
        help="Dataset stats path. Defaults to <bundle-dir>/dataset_stats.pkl.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Dataset directory. Defaults to task.dataset_dir from resolved_config.yaml.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Train-ready manifest. Defaults to task.train_ready_manifest_path.",
    )
    parser.add_argument(
        "--episode-id",
        default="auto",
        help="Episode id, e.g. 43 or episode_43. Use auto to choose representative train-ready episode.",
    )
    parser.add_argument(
        "--all-train-ready",
        action="store_true",
        help="Evaluate every train-ready episode from the manifest.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Report output directory. Defaults to runs/offline_policy_eval/<timestamp>_<ckpt>_<episode>.",
    )
    parser.add_argument("--device", default=None, help="Torch device override, e.g. cuda or cpu.")
    parser.add_argument(
        "--no-temporal-agg",
        action="store_true",
        help="Disable ACT temporal aggregation during replay.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional replay step cap for smoke tests.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print replay progress every N steps. Use 0 to disable.",
    )
    args = parser.parse_args()

    bundle_dir = Path(args.bundle_dir)
    resolved_config = Path(args.resolved_config) if args.resolved_config else bundle_dir / "resolved_config.yaml"
    stats_path = Path(args.stats) if args.stats else bundle_dir / "dataset_stats.pkl"
    ckpt_path = Path(args.ckpt) if args.ckpt else choose_default_ckpt(bundle_dir)

    resolved = _load_resolved_config(resolved_config)
    task_cfg = dict(resolved.get("task", {}) or {})
    policy_cfg = dict(resolved.get("policy", {}) or {})
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else Path(task_cfg["dataset_dir"])
    manifest = (
        Path(args.manifest)
        if args.manifest
        else Path(task_cfg.get("train_ready_manifest_path", dataset_dir / "qc_full" / "train_ready_manifest.json"))
    )
    camera_names = [str(cam) for cam in task_cfg.get("camera_names", ["fpv"])]

    train_ready_ids = load_train_ready_episode_ids(manifest)
    representative_scores = None
    if args.all_train_ready:
        selected_ids = train_ready_ids
        selected_episode = "all_train_ready"
    elif str(args.episode_id).strip().lower() == "auto":
        features = load_episode_features(dataset_dir, train_ready_ids)
        selected_episode, representative_scores = select_representative_episode(
            train_ready_ids,
            features,
        )
        selected_ids = [selected_episode]
    else:
        selected_episode = normalize_episode_id(args.episode_id)
        selected_ids = [selected_episode]

    selected_paths = [episode_path(dataset_dir, episode_id) for episode_id in selected_ids]
    missing = [path for path in selected_paths if not path.exists()]
    if missing:
        raise SystemExit(f"Episode file does not exist: {missing[0]}")
    episode_len = max(_episode_len(path, max_steps=args.max_steps) for path in selected_paths)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else default_collection_output_dir(
            Path("runs/offline_policy_eval"),
            ckpt_path=ckpt_path,
        )
        if args.all_train_ready
        else default_output_dir(
            Path("runs/offline_policy_eval"),
            episode_id=selected_episode,
            ckpt_path=ckpt_path,
        )
    )

    print(f"Selected episode: {selected_episode}")
    if args.all_train_ready:
        print(f"Episode count: {len(selected_ids)}")
    else:
        print(f"Episode path: {selected_paths[0]}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Output dir: {output_dir}")

    policy = load_policy_for_episode(
        bundle_dir=bundle_dir,
        ckpt_path=ckpt_path,
        resolved_config_path=resolved_config,
        stats_path=stats_path,
        max_episode_len=episode_len,
        temporal_agg=not args.no_temporal_agg,
        device=args.device,
    )
    common_metadata = {
        "selection_mode": (
            "all_train_ready"
            if args.all_train_ready
            else "representative"
            if representative_scores
            else "explicit"
        ),
        "dataset_dir": str(dataset_dir),
        "manifest": str(manifest),
        "bundle_dir": str(bundle_dir),
        "ckpt_path": str(ckpt_path),
        "resolved_config": str(resolved_config),
        "stats_path": str(stats_path),
        "camera_names": camera_names,
        "low_dim_keys": list(policy_cfg.get("low_dim_keys", ["qpos"])),
        "temporal_agg": not args.no_temporal_agg,
        "max_steps": args.max_steps,
    }
    if args.all_train_ready:
        results = []
        episode_root = output_dir / "episodes"
        for idx, episode_id in enumerate(selected_ids, start=1):
            path = episode_path(dataset_dir, episode_id)
            print(f"[{idx}/{len(selected_ids)}] Evaluating {episode_id}: {path}")
            result = evaluate_episode(
                policy=policy,
                episode_file=path,
                camera_names=camera_names,
                max_steps=args.max_steps,
                progress_every=max(0, int(args.progress_every)),
            )
            result["episode_id"] = episode_id
            write_eval_report(
                result=result,
                output_dir=episode_root / episode_id,
                metadata={
                    **common_metadata,
                    "selected_episode_id": episode_id,
                },
            )
            results.append(result)
        aggregate = aggregate_episode_results(results)
        paths = write_collection_report(
            aggregate=aggregate,
            output_dir=output_dir,
            metadata={
                **common_metadata,
                "episode_ids": selected_ids,
            },
        )
        metrics = aggregate["global_metrics"]
        print(f"Collection summary: {paths['collection_summary']}")
        print(f"Episode metrics CSV: {paths['episode_metrics_csv']}")
        print(f"Collection distribution: {paths['distribution_plot']}")
        print(f"Episode p95 plot: {paths['p95_plot']}")
        print(f"Episode MAE plot: {paths['mae_plot']}")
    else:
        result = evaluate_episode(
            policy=policy,
            episode_file=selected_paths[0],
            camera_names=camera_names,
            max_steps=args.max_steps,
            progress_every=max(0, int(args.progress_every)),
        )
        result["episode_id"] = selected_episode
        paths = write_eval_report(
            result=result,
            output_dir=output_dir,
            representative_scores=representative_scores,
            metadata={
                **common_metadata,
                "selected_episode_id": selected_episode,
            },
        )

        summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
        metrics = summary["metrics"]
        print(f"Summary: {paths['summary']}")
        print(f"Action timeseries: {paths['timeseries_plot']}")
        print(f"Action distribution: {paths['distribution_plot']}")
    print(
        "Overall: "
        f"mae={metrics['overall']['mae']:.4f} "
        f"expert_p95_abs={metrics['overall']['expert_p95_abs']:.4f} "
        f"policy_p95_abs={metrics['overall']['policy_p95_abs']:.4f} "
        f"policy_max_abs={metrics['overall']['policy_max_abs']:.4f}"
    )


def _load_resolved_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Resolved config does not exist: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _episode_len(path: Path, *, max_steps: int | None) -> int:
    import h5py

    with h5py.File(path, "r") as f:
        length = int(f["action"].shape[0])
    if max_steps is not None:
        return min(length, int(max_steps))
    return length


if __name__ == "__main__":
    main()
