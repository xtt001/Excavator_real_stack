"""Measure continuous ACT response to a same-state condition flip.

This is deliberately separate from the auxiliary condition classifier audit.
The classifier can be correct while the continuous action proposal moves in the
wrong direction or remains below the mechanical deadzone.  The probe uses the
initial observation of each manifest validation episode, evaluates condition
codes +1 and -1 with fresh policy state, and reports the direct query-0 action
difference.  It is an offline diagnostic, not a plant-effect claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml

from testbed.actions.policy import load_act_policy_from_bundle
from testbed.data.dataset import _read_camera_image

CAMERAS = ("video4", "video5", "video6", "video7")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="tb-condition-weight-sensitivity",
        description=(
            "Probe direct action response to a same-state +1/-1 condition flip."
        ),
    )
    parser.add_argument("--bundle-dir", type=Path, action="append", required=True)
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--cycle-manifest", type=Path, default=None)
    parser.add_argument("--train-ready-manifest", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    reports = []
    for bundle_dir in args.bundle_dir:
        reports.append(
            probe_bundle(
                bundle_dir=bundle_dir,
                dataset_dir=args.dataset_dir,
                cycle_manifest=args.cycle_manifest,
                train_ready_manifest=args.train_ready_manifest,
                device=str(args.device),
            )
        )
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"output already exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "schema": "condition_action_sensitivity_v1",
                "reports": reports,
                "boundary": (
                    "The probe compares direct query-0 actions at the same "
                    "initial qpos and images with condition code +1 versus -1. "
                    "It does not simulate hydraulics or establish field effect."
                ),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "reports": reports}, ensure_ascii=False))


def probe_bundle(
    *,
    bundle_dir: Path,
    dataset_dir: Path | None,
    cycle_manifest: Path | None,
    train_ready_manifest: Path | None,
    device: str,
) -> dict[str, Any]:
    bundle_dir = bundle_dir.resolve()
    resolved = yaml.safe_load(
        (bundle_dir / "resolved_config.yaml").read_text(encoding="utf-8")
    ) or {}
    task_cfg = dict(resolved.get("task", {}) or {})
    episodes_dir = (
        dataset_dir
        if dataset_dir is not None
        else Path(str(task_cfg["dataset_dir"]))
    )
    cycle_path = cycle_manifest or episodes_dir.parent / "cycle_manifest.jsonl"
    ready_path = train_ready_manifest or Path(
        str(task_cfg["train_ready_manifest_path"])
    )
    ready_ids = {
        int(str(value).split("_")[-1])
        for value in json.loads(ready_path.read_text(encoding="utf-8"))[
            "train_ready_episode_ids"
        ]
    }
    rows = [
        json.loads(line)
        for line in cycle_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    validation_rows = sorted(
        (
            row
            for row in rows
            if row.get("split") == "validation"
            and int(row["episode_id"]) in ready_ids
        ),
        key=lambda row: int(row["episode_id"]),
    )
    policy = load_act_policy_from_bundle(
        bundle_dir=bundle_dir,
        ckpt_path=bundle_dir / "policy_best.ckpt",
        device=device,
        temporal_agg=False,
    )
    deltas: list[float] = []
    action_l2: list[float] = []
    finite = True
    per_episode: list[dict[str, Any]] = []
    try:
        for row in validation_rows:
            episode_id = int(row["episode_id"])
            episode_path = episodes_dir / f"episode_{episode_id}.hdf5"
            with h5py.File(episode_path, "r") as handle:
                qpos = np.asarray(handle["observations/qpos"][0], dtype=np.float32)
                images = {
                    f"image_{camera}": _read_camera_image(handle, camera, 0)
                    for camera in CAMERAS
                }
            base = {"qpos": qpos, **images}

            actions: dict[str, np.ndarray] = {}
            for code_name, code in (("positive", 1.0), ("negative", -1.0)):
                policy.reset()
                observation = {
                    **base,
                    "real_transition_condition_v1": np.asarray(
                        [code, 1.0], dtype=np.float32
                    ),
                }
                actions[code_name] = np.asarray(policy.predict(observation), dtype=np.float32)
            positive = actions["positive"]
            negative = actions["negative"]
            delta = positive - negative
            is_finite = bool(np.isfinite(positive).all() and np.isfinite(negative).all())
            finite = finite and is_finite
            deltas.append(float(delta[0]))
            action_l2.append(float(np.linalg.norm(delta)))
            current_side, target_side = str(row["transition_type"]).split("->", 1)
            per_episode.append(
                {
                    "episode_id": episode_id,
                    "transition": f"{current_side}->{target_side}",
                    "cross_target": current_side != target_side,
                    "finite": is_finite,
                    "positive_condition_swing": float(positive[0]),
                    "negative_condition_swing": float(negative[0]),
                    "swing_delta_positive_minus_negative": float(delta[0]),
                    "action_l2_delta": float(np.linalg.norm(delta)),
                }
            )
    finally:
        close = getattr(policy, "close", None)
        if callable(close):
            close()

    cross = [item for item in per_episode if item["cross_target"]]
    same = [item for item in per_episode if not item["cross_target"]]
    cross_deltas = [item["swing_delta_positive_minus_negative"] for item in cross]
    same_deltas = [item["swing_delta_positive_minus_negative"] for item in same]
    return {
        "bundle_dir": str(bundle_dir),
        "checkpoint": str(bundle_dir / "policy_best.ckpt"),
        "condition_action_weight": float(
            (resolved.get("train", {}).get("condition_action_loss", {}) or {}).get(
                "weight", 0.0
            )
        ),
        "validation_episode_ids": [int(row["episode_id"]) for row in validation_rows],
        "finite": finite,
        "action_l2_abs_quantiles": _quantiles(action_l2),
        "swing_delta_quantiles": _quantiles(deltas),
        "swing_delta_mean": float(np.mean(deltas)) if deltas else 0.0,
        "positive_minus_negative_count": int(sum(value > 0 for value in deltas)),
        "cross_target_count": len(cross),
        "cross_target_positive_minus_negative_count": int(
            sum(value > 0 for value in cross_deltas)
        ),
        "cross_target_swing_delta_quantiles": _quantiles(cross_deltas),
        "same_target_swing_delta_abs_quantiles": _quantiles(
            np.abs(same_deltas).tolist()
        ),
        "per_episode": per_episode,
    }


def _quantiles(values: list[float]) -> list[float]:
    if not values:
        return []
    return [float(value) for value in np.quantile(np.asarray(values), [0, 0.5, 0.95, 1])]


if __name__ == "__main__":
    main()
