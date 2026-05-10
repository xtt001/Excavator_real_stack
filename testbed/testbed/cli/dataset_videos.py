"""
tb-dataset-videos — Export stored FPV images from HDF5 episodes as MP4 videos.

Offline utility: reads images directly from HDF5 files.

Usage
-----
    # All episodes in a directory
    tb-dataset-videos data/real_teleop_v1/

    # Single episode
    tb-dataset-videos data/real_teleop_v1/episode_0.hdf5

    # Custom output directory and camera
    tb-dataset-videos data/real_teleop_v1/ -o runs/videos/v1 --camera fpv

    # Specific episodes by index
    tb-dataset-videos data/real_teleop_v1/ --indices 0 3 5 10
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def _episode_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"episode_(\d+)", path.stem)
    return (int(match.group(1)), path.stem) if match else (999999, path.stem)


def _resolve_episodes(path: Path, indices: list[int] | None) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Not a file or directory: {path}")

    episodes = sorted(path.glob("episode_*.hdf5"), key=_episode_sort_key)
    if not episodes:
        episodes = sorted(path.glob("*.hdf5"), key=_episode_sort_key)

    if indices is not None:
        index_set = set(indices)
        episodes = [
            ep for ep in episodes
            if _parse_episode_index(ep) in index_set
        ]
    return episodes


def _parse_episode_index(path: Path) -> int:
    match = re.search(r"episode_(\d+)", path.stem)
    return int(match.group(1)) if match else -1


def _save_video(frames: list[np.ndarray], path: Path, fps: int = 50) -> None:
    try:
        import cv2
    except ImportError:
        log.error("opencv-python not installed — cannot save video.")
        return

    if not frames:
        return

    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    for frame in frames:
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if frame.shape[-1] == 3 else frame
        writer.write(bgr)
    writer.release()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    parser = argparse.ArgumentParser(
        prog="tb-dataset-videos",
        description="Export stored images from HDF5 episodes as MP4 videos.",
    )
    parser.add_argument("path", type=Path,
                        help="Path to a single episode .hdf5 or a directory of episodes.")
    parser.add_argument("-o", "--output-dir", type=Path, default=None,
                        help="Output directory for videos. Defaults to <dataset_dir>/videos/.")
    parser.add_argument("--camera", type=str, default="fpv",
                        help="Camera name to extract (default: fpv).")
    parser.add_argument("--fps", type=int, default=None,
                        help="Video FPS. If omitted, reads control_hz from episode metadata.")
    parser.add_argument("--indices", type=int, nargs="+", default=None,
                        help="Only export specific episode indices (e.g. --indices 0 3 5).")
    args = parser.parse_args()

    episodes = _resolve_episodes(args.path, args.indices)
    if not episodes:
        log.error("No .hdf5 episode files found at %s", args.path)
        return

    if args.output_dir is not None:
        output_dir = args.output_dir
    elif args.path.is_dir():
        output_dir = args.path / "videos"
    else:
        output_dir = args.path.parent / "videos"
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Exporting %d episode(s) → %s  (camera=%s)", len(episodes), output_dir, args.camera)

    from testbed.data.hdf5_io import read_episode

    exported = 0
    skipped = 0
    for ep_idx, ep_path in enumerate(episodes):
        ep = read_episode(ep_path)
        images = ep.get("images", {})
        cam_images = images.get(args.camera)

        if cam_images is None or len(cam_images) == 0:
            log.warning(
                "[%d/%d] %s — no '%s' images found, skipping.",
                ep_idx + 1, len(episodes), ep_path.name, args.camera,
            )
            skipped += 1
            continue

        meta = ep.get("metadata", {})
        fps = args.fps or int(meta.get("control_hz", 50))

        out_path = output_dir / f"{ep_path.stem}.mp4"
        frames = [cam_images[t] for t in range(len(cam_images))]
        _save_video(frames, out_path, fps=fps)

        log.info(
            "[%d/%d] %s → %s  (%d frames, %dx%d, %d fps)",
            ep_idx + 1, len(episodes), ep_path.name, out_path.name,
            len(frames), frames[0].shape[1], frames[0].shape[0], fps,
        )
        exported += 1

    log.info("Done. exported=%d  skipped=%d  output_dir=%s", exported, skipped, output_dir)


if __name__ == "__main__":
    main()
