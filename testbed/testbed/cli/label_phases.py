"""tb-label-phases - generate offline coarse phase labels for HDF5 episodes."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from testbed.data.hdf5_io import list_episodes
from testbed.data.phase_labeler import (
    PhaseLabelConfig,
    label_episode_phases,
    write_phase_labels,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="tb-label-phases",
        description="Generate coarse DIG/SWING/DUMP/RETURN/GO_HOME labels without modifying HDF5.",
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--phase-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--indices", type=int, nargs="*", default=None)
    args = parser.parse_args()

    raw_cfg = _load_yaml(args.phase_config)
    cfg = PhaseLabelConfig.from_mapping(raw_cfg.get("phase_labeling", raw_cfg))
    output_dir = args.output_dir or args.dataset_dir / "phase_labels"
    index_set = None if args.indices is None else set(int(i) for i in args.indices)

    count = 0
    for path in list_episodes(args.dataset_dir):
        if index_set is not None and _episode_index(path) not in index_set:
            continue
        result = label_episode_phases(path, config=cfg)
        json_path, csv_path = write_phase_labels(result, output_dir)
        print(f"{path.name}: {json_path} {csv_path}")
        count += 1
    print(f"Generated phase labels for {count} episode(s).")


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"phase config must decode to a mapping: {path}")
    return data


def _episode_index(path: Path) -> int:
    return int(path.stem.split("_", 1)[1])


if __name__ == "__main__":
    main()
