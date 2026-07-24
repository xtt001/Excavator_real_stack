"""Build semantically cropped cycle datasets from reviewed cut annotations."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from testbed.data.bucket_repair import parse_episode_spec
from testbed.data.hdf5_io import episode_id_from_path, list_episodes


ANNOTATION_FIELDS = [
    "episode_id",
    "source_path",
    "source_steps",
    "cut_step",
    "cut_label",
    "review_status",
    "notes",
]
REVIEWED_STATUSES = frozenset({"manual_reviewed", "accepted", "approved"})


@dataclass(frozen=True)
class SemanticCycleCut:
    episode_id: int
    cut_step: int
    cut_label: str
    review_status: str
    notes: str = ""


def write_semantic_cycle_annotation_template(
    *,
    input_dir: str | Path,
    output_csv: str | Path,
    episode_ids: list[int] | None = None,
) -> dict[str, Any]:
    input_path = Path(input_dir)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    episodes = _select_episode_paths(input_path, episode_ids)
    rows: list[dict[str, str]] = []
    for path in episodes:
        episode_id = episode_id_from_path(path)
        with h5py.File(path, "r") as f:
            source_steps = int(f["action"].shape[0])
        rows.append(
            {
                "episode_id": str(episode_id),
                "source_path": str(path),
                "source_steps": str(source_steps),
                "cut_step": "",
                "cut_label": "",
                "review_status": "needs_manual_review",
                "notes": "",
            }
        )

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ANNOTATION_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return {
        "schema_version": 1,
        "input_dir": str(input_path),
        "annotation_csv": str(output_path),
        "episodes": [
            {"episode_id": int(row["episode_id"]), "source_steps": int(row["source_steps"])}
            for row in rows
        ],
    }


def build_semantic_cycle_dataset(
    *,
    input_dir: str | Path,
    output_dir: str | Path,
    annotation_csv: str | Path,
    episode_ids: list[int] | None = None,
    allow_unreviewed: bool = False,
) -> dict[str, Any]:
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    cuts = load_semantic_cycle_cuts(
        annotation_csv,
        allow_unreviewed=allow_unreviewed,
    )
    selected_paths = _select_episode_paths(input_path, episode_ids)
    rows: list[dict[str, Any]] = []
    for src in selected_paths:
        episode_id = episode_id_from_path(src)
        if episode_id not in cuts:
            raise ValueError(f"Missing semantic cut annotation for episode_{episode_id}")
        dst = output_path / src.name
        rows.append(
            crop_semantic_cycle_episode(
                input_path=src,
                output_path=dst,
                cut=cuts[episode_id],
            )
        )

    summary = {
        "schema_version": 1,
        "input_dir": str(input_path),
        "output_dir": str(output_path),
        "annotation_csv": str(Path(annotation_csv)),
        "allow_unreviewed": bool(allow_unreviewed),
        "episodes": rows,
    }
    with (output_path / "semantic_cycle_crop_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def load_semantic_cycle_cuts(
    annotation_csv: str | Path,
    *,
    allow_unreviewed: bool = False,
) -> dict[int, SemanticCycleCut]:
    cuts: dict[int, SemanticCycleCut] = {}
    with Path(annotation_csv).open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            episode_id = int(_required(row, "episode_id"))
            status = _required(row, "review_status")
            if not allow_unreviewed and status not in REVIEWED_STATUSES:
                raise ValueError(
                    f"episode_{episode_id} semantic cut is not reviewed: {status}"
                )
            cut_step_raw = _required(row, "cut_step")
            cut_step = int(cut_step_raw)
            cut_label = _required(row, "cut_label")
            cuts[episode_id] = SemanticCycleCut(
                episode_id=episode_id,
                cut_step=cut_step,
                cut_label=cut_label,
                review_status=status,
                notes=str(row.get("notes", "") or ""),
            )
    return cuts


def crop_semantic_cycle_episode(
    *,
    input_path: str | Path,
    output_path: str | Path,
    cut: SemanticCycleCut,
) -> dict[str, Any]:
    src = Path(input_path)
    dst = Path(output_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(src, "r") as in_f:
        source_steps = int(in_f["action"].shape[0])
        cut_step = _validated_cut_step(cut, source_steps=source_steps)
        if dst.exists():
            dst.unlink()
        with h5py.File(dst, "w") as out_f:
            _copy_attrs(in_f, out_f)
            _copy_node(in_f, out_f, source_steps=source_steps, cut_step=cut_step)
            _write_semantic_cycle_metadata(
                out_f,
                cut=cut,
                source_path=src,
                source_steps=source_steps,
                output_steps=cut_step,
            )
    return {
        "episode_id": int(cut.episode_id),
        "input_path": str(src),
        "output_path": str(dst),
        "source_steps": int(source_steps),
        "output_steps": int(cut_step),
        "cut_step": int(cut_step),
        "cut_label": cut.cut_label,
        "review_status": cut.review_status,
    }


def _copy_node(
    in_group: h5py.Group,
    out_group: h5py.Group,
    *,
    source_steps: int,
    cut_step: int,
) -> None:
    for name, obj in in_group.items():
        if isinstance(obj, h5py.Group):
            child = out_group.create_group(name)
            _copy_attrs(obj, child)
            _copy_node(obj, child, source_steps=source_steps, cut_step=cut_step)
        elif isinstance(obj, h5py.Dataset):
            data = obj[:cut_step] if obj.shape and obj.shape[0] == source_steps else obj[()]
            out = out_group.create_dataset(name, data=data, dtype=obj.dtype)
            _copy_attrs(obj, out)


def _write_semantic_cycle_metadata(
    f: h5py.File,
    *,
    cut: SemanticCycleCut,
    source_path: Path,
    source_steps: int,
    output_steps: int,
) -> None:
    meta = f.require_group("metadata")
    meta.attrs["semantic_cycle_cut"] = 1
    meta.attrs["semantic_cycle_cut_step"] = int(output_steps)
    meta.attrs["semantic_cycle_source_steps"] = int(source_steps)
    meta.attrs["semantic_cycle_source_dataset_path"] = str(source_path)
    meta.attrs["semantic_cycle_cut_label"] = cut.cut_label
    meta.attrs["semantic_cycle_review_status"] = cut.review_status
    meta.attrs["semantic_cycle_notes"] = cut.notes
    meta.attrs["semantic_cycle_policy_scope"] = "dig_carry_dump_before_return"
    meta.attrs["n_steps"] = int(output_steps)
    diag = f.require_group("diagnostics")
    if "source_cycle_keep_mask" in diag:
        del diag["source_cycle_keep_mask"]
    diag.create_dataset(
        "source_cycle_keep_mask",
        data=np.ones(int(output_steps), dtype=np.uint8),
    )


def _select_episode_paths(input_dir: Path, episode_ids: list[int] | None) -> list[Path]:
    paths_by_id = {episode_id_from_path(path): path for path in list_episodes(input_dir)}
    if episode_ids is None:
        return [paths_by_id[idx] for idx in sorted(paths_by_id)]
    selected: list[Path] = []
    for episode_id in episode_ids:
        if int(episode_id) not in paths_by_id:
            raise FileNotFoundError(input_dir / f"episode_{episode_id}.hdf5")
        selected.append(paths_by_id[int(episode_id)])
    return selected


def _validated_cut_step(cut: SemanticCycleCut, *, source_steps: int) -> int:
    cut_step = int(cut.cut_step)
    if cut_step <= 0 or cut_step > int(source_steps):
        raise ValueError(
            f"episode_{cut.episode_id} cut_step must be within 1..{source_steps}, got {cut_step}"
        )
    return cut_step


def _copy_attrs(src: h5py.AttributeManager | h5py.Group | h5py.Dataset, dst: h5py.Group | h5py.Dataset | h5py.File) -> None:
    attrs = src.attrs if hasattr(src, "attrs") else src
    for key, value in attrs.items():
        dst.attrs[key] = value


def _required(row: dict[str, str | None], key: str) -> str:
    value = row.get(key)
    if value is None or str(value).strip() == "":
        raise ValueError(f"Missing required semantic cycle annotation field: {key}")
    return str(value).strip()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build semantically cropped real-excavator cycle datasets."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    template = subparsers.add_parser("template")
    template.add_argument("--input-dir", type=Path, required=True)
    template.add_argument("--output-csv", type=Path, required=True)
    template.add_argument("--episodes", default=None)

    build = subparsers.add_parser("build")
    build.add_argument("--input-dir", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--annotation-csv", type=Path, required=True)
    build.add_argument("--episodes", default=None)
    build.add_argument("--allow-unreviewed", action="store_true")

    args = parser.parse_args(argv)
    episode_ids = parse_episode_spec(args.episodes) if args.episodes else None
    if args.command == "template":
        summary = write_semantic_cycle_annotation_template(
            input_dir=args.input_dir,
            output_csv=args.output_csv,
            episode_ids=episode_ids,
        )
        print(json.dumps(summary, indent=2))
    elif args.command == "build":
        summary = build_semantic_cycle_dataset(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            annotation_csv=args.annotation_csv,
            episode_ids=episode_ids,
            allow_unreviewed=bool(args.allow_unreviewed),
        )
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
