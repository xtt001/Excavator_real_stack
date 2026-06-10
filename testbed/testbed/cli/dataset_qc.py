"""tb-dataset-qc — inspect recorded HDF5 episodes and emit QC reports."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="tb-dataset-qc",
        description="Run QC checks and plots for a recorded HDF5 dataset.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Directory containing episode_*.hdf5 files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for QC artifacts. Defaults to <dataset_dir>/qc.",
    )
    parser.add_argument(
        "--profile",
        choices=["real"],
        default="real",
        help="QC profile. This branch only supports real excavator datasets.",
    )
    parser.add_argument(
        "--mode",
        choices=["quick", "full"],
        default="quick",
        help="quick for live QC, full for plots and offline training QC.",
    )
    parser.add_argument(
        "--reference-episodes",
        default="26-46",
        help="Reference episode ids/ranges for training QC, e.g. 26-46 or 24,26-46.",
    )
    parser.add_argument(
        "--no-training-qc",
        action="store_true",
        help="Disable training-focused QC additions.",
    )
    args = parser.parse_args()

    from testbed.data.qc import run_dataset_qc
    from testbed.data.bucket_repair import parse_episode_spec

    result = run_dataset_qc(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        profile=args.profile,
        mode=args.mode,
        training_qc=not args.no_training_qc,
        reference_episode_ids=parse_episode_spec(args.reference_episodes),
    )
    print(f"Dataset QC summary written to {result['summary_path']}")
    print(f"Per-episode QC CSV written to {result['episodes_csv_path']}")
    training_qc = result.get("training_qc")
    if training_qc:
        print(f"Training QC summary written to {training_qc['summary_path']}")
        print(f"Train-ready manifest written to {training_qc['manifest_path']}")


if __name__ == "__main__":
    main()
