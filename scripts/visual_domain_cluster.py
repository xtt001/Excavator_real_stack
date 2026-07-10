#!/usr/bin/env python3
"""Cluster train-ready HDF5 episodes into visual texture domains."""

from __future__ import annotations

import argparse
from pathlib import Path

from testbed.data.visual_domain import VisualDomainConfig, run_visual_domain_clustering


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cameras", nargs="+", default=["video4", "video5", "video6", "video7"])
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--max-frames-per-episode", type=int, default=24)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--feature-size", type=int, default=96)
    parser.add_argument("--contact-sheet-per-cluster", type=int, default=30)
    args = parser.parse_args()

    summary = run_visual_domain_clustering(
        VisualDomainConfig(
            dataset_dir=args.dataset_dir,
            manifest_path=args.manifest,
            output_dir=args.output_dir,
            camera_names=tuple(args.cameras),
            k=args.k,
            max_frames_per_episode=args.max_frames_per_episode,
            seed=args.seed,
            feature_size=args.feature_size,
            contact_sheet_per_cluster=args.contact_sheet_per_cluster,
        )
    )
    print(f"wrote {summary['sample_count']} samples from {summary['episode_count']} episodes")
    print(f"summary: {args.output_dir / 'cluster_summary.json'}")


if __name__ == "__main__":
    main()
