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
    args = parser.parse_args()

    from testbed.data.qc import run_dataset_qc

    result = run_dataset_qc(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        profile=args.profile,
    )
    print(f"Dataset QC summary written to {result['summary_path']}")
    print(f"Per-episode QC CSV written to {result['episodes_csv_path']}")


if __name__ == "__main__":
    main()
