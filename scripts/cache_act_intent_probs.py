#!/usr/bin/env python3
"""Cache ACT auxiliary intent probabilities for offline replay inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import yaml

from testbed.data.dataset import _read_camera_image
from testbed.data.image_transforms import IMAGE_TRANSFORM_CHOICES, build_image_transform
from testbed.policies.offline_eval import (
    choose_default_ckpt,
    episode_path,
    load_policy_for_episode,
    load_train_ready_episode_ids,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, default=None)
    parser.add_argument("--resolved-config", type=Path, default=None)
    parser.add_argument("--stats", type=Path, default=None)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-transform", choices=IMAGE_TRANSFORM_CHOICES, default="none")
    parser.add_argument("--device", default=None)
    parser.add_argument("--progress-every", type=int, default=1)
    args = parser.parse_args()

    bundle_dir = Path(args.bundle_dir)
    resolved_config = Path(args.resolved_config) if args.resolved_config else bundle_dir / "resolved_config.yaml"
    stats_path = Path(args.stats) if args.stats else bundle_dir / "dataset_stats.pkl"
    ckpt_path = Path(args.ckpt) if args.ckpt else choose_default_ckpt(bundle_dir)
    resolved = _load_yaml(resolved_config)
    task_cfg = dict(resolved.get("task", {}) or {})
    camera_names = [str(cam) for cam in task_cfg.get("camera_names", ["fpv"])]
    episode_ids = load_train_ready_episode_ids(args.manifest)
    episode_paths = [episode_path(args.dataset_dir, episode_id) for episode_id in episode_ids]
    missing = [path for path in episode_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(missing[0])
    max_episode_len = max(_episode_len(path) for path in episode_paths)
    policy = load_policy_for_episode(
        bundle_dir=bundle_dir,
        ckpt_path=ckpt_path,
        resolved_config_path=resolved_config,
        stats_path=stats_path,
        max_episode_len=max_episode_len,
        temporal_agg=True,
        device=args.device,
    )

    transform = build_image_transform(str(args.image_transform))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    intent_dir = output_dir / "intent_probs"
    intent_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, (episode_id, path) in enumerate(zip(episode_ids, episode_paths), start=1):
        if int(args.progress_every) > 0 and (index == 1 or index % int(args.progress_every) == 0):
            print(f"[{index}/{len(episode_ids)}] caching {episode_id}: {path}")
        probs = _cache_episode_intent_probs(
            policy=policy,
            episode_path=path,
            camera_names=camera_names,
            image_transform=transform,
        )
        np.savez_compressed(intent_dir / f"{episode_id}.npz", intent_prob=probs)
        rows.append({"episode_id": episode_id, "steps": int(probs.shape[0])})

    summary = {
        "bundle_dir": str(bundle_dir),
        "ckpt_path": str(ckpt_path),
        "resolved_config": str(resolved_config),
        "stats_path": str(stats_path),
        "dataset_dir": str(args.dataset_dir),
        "manifest": str(args.manifest),
        "image_transform": str(args.image_transform),
        "camera_names": camera_names,
        "episodes": rows,
        "output_dir": str(output_dir),
        "intent_prob_dir": str(intent_dir),
    }
    (output_dir / "intent_prob_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Intent prob dir: {intent_dir}")
    print(f"Summary: {output_dir / 'intent_prob_summary.json'}")


def _cache_episode_intent_probs(
    *,
    policy: Any,
    episode_path: Path,
    camera_names: list[str],
    image_transform: Any,
) -> np.ndarray:
    with h5py.File(episode_path, "r") as f:
        qpos = np.asarray(f["observations/qpos"][()], dtype=np.float32)
        qvel = np.asarray(f["observations/qvel"][()], dtype=np.float32)
        action = np.asarray(f["action"][()], dtype=np.float32)
        step_count = int(min(qpos.shape[0], qvel.shape[0], action.shape[0]))
        probs = np.zeros((step_count, 8), dtype=np.float32)
        for step in range(step_count):
            obs: dict[str, Any] = {
                "qpos": qpos[step],
                "qvel": qvel[step],
            }
            for camera_name in camera_names:
                image = _read_camera_image(f, camera_name, step)
                if image_transform is not None:
                    image = image_transform(image)
                obs[f"image_{camera_name}"] = image
            probs[step] = _predict_intent_prob_query0(policy, obs)
    return probs


def _predict_intent_prob_query0(policy: Any, obs: dict[str, Any]) -> np.ndarray:
    proprio = policy._build_proprio(obs)
    proprio = (proprio - policy._proprio_mean) / policy._proprio_std
    image = _image_tensor_for_policy(policy, obs)
    if policy._model.training:
        policy._model.eval()
    with torch.inference_mode():
        _, _, _, intent_logits = policy._unpack_model_output(
            policy._model(proprio, image, None)
        )
    if intent_logits is None:
        raise ValueError("loaded ACT policy has no intent logits")
    prob = torch.sigmoid(intent_logits[0, 0]).detach().cpu().numpy()
    return np.asarray(prob, dtype=np.float32)


def _image_tensor_for_policy(policy: Any, obs: dict[str, Any]) -> torch.Tensor:
    cam_images: list[np.ndarray] = []
    for cam in policy.camera_names:
        raw = np.asarray(obs[f"image_{cam}"])
        image = np.asarray(raw, dtype=np.float32)
        if image.ndim != 3:
            raise ValueError(f"image_{cam} must be rank-3, got {image.shape}")
        if image.shape[0] == 3:
            pass
        elif image.shape[-1] == 3:
            image = np.transpose(image, (2, 0, 1))
            if raw.dtype == np.uint8 or image.max() > 1.0:
                image = image / 255.0
        else:
            raise ValueError(f"image_{cam} must have 3 channels, got {image.shape}")
        cam_images.append(image)
    stacked = np.ascontiguousarray(np.stack(cam_images, axis=0))
    tensor = torch.from_numpy(stacked).float().to(policy.device).unsqueeze(0)
    return policy._normalize(tensor)


def _episode_len(path: Path) -> int:
    with h5py.File(path, "r") as f:
        return int(min(f["observations/qpos"].shape[0], f["observations/qvel"].shape[0], f["action"].shape[0]))


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


if __name__ == "__main__":
    main()
