"""Offline coarse phase labeling for one-bucket real excavator episodes."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np


PHASES = ("DIG", "SWING_TO_DUMP", "DUMP", "RETURN_NEAR_HOME", "GO_HOME", "END")


@dataclass(frozen=True)
class PhaseLabelConfig:
    home_pose_rad: np.ndarray | None = None
    near_home_tolerance_rad: np.ndarray | None = None
    dig_swing_range: tuple[float, float] | None = None
    dump_swing_range: tuple[float, float] | None = None
    swing_velocity_threshold_rad_s: float = 0.02
    bucket_velocity_threshold_rad_s: float = 0.02
    dwell_steps: int = 5

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any] | None) -> "PhaseLabelConfig":
        raw = dict(cfg or {})
        return cls(
            home_pose_rad=_optional_vec4(raw.get("home_pose_rad")),
            near_home_tolerance_rad=_optional_vec4(
                raw.get("near_home_tolerance_rad", raw.get("near_home_tolerance"))
            ),
            dig_swing_range=_optional_range(raw.get("dig_swing_range")),
            dump_swing_range=_optional_range(raw.get("dump_swing_range")),
            swing_velocity_threshold_rad_s=float(
                raw.get("swing_velocity_threshold_rad_s", 0.02)
            ),
            bucket_velocity_threshold_rad_s=float(
                raw.get("bucket_velocity_threshold_rad_s", 0.02)
            ),
            dwell_steps=max(1, int(raw.get("dwell_steps", 5))),
        )


def label_episode_phases(
    episode_path: str | Path,
    *,
    config: PhaseLabelConfig,
) -> dict[str, Any]:
    path = Path(episode_path)
    with h5py.File(path, "r") as f:
        qpos = np.asarray(f["observations/qpos"][()], dtype=np.float32)
        qvel = np.asarray(f["observations/qvel"][()], dtype=np.float32)
        diagnostics = f.get("diagnostics")
        go_home_running = (
            np.asarray(diagnostics["go_home_running"][()], dtype=np.int32)
            if diagnostics is not None and "go_home_running" in diagnostics
            else np.zeros(qpos.shape[0], dtype=np.int32)
        )
    labels = _label_arrays(qpos=qpos, qvel=qvel, go_home_running=go_home_running, cfg=config)
    transitions = _transitions(labels)
    return {
        "episode_path": str(path),
        "episode_id": path.stem,
        "phases": labels,
        "transitions": transitions,
        "phase_counts": {phase: int(labels.count(phase)) for phase in PHASES},
    }


def write_phase_labels(result: dict[str, Any], output_dir: str | Path) -> tuple[Path, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    episode_id = str(result["episode_id"])
    json_path = out_dir / f"{episode_id}.json"
    csv_path = out_dir / f"{episode_id}.csv"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "phase"])
        writer.writeheader()
        for step, phase in enumerate(result["phases"]):
            writer.writerow({"step": step, "phase": phase})
    return json_path, csv_path


def _label_arrays(
    *,
    qpos: np.ndarray,
    qvel: np.ndarray,
    go_home_running: np.ndarray,
    cfg: PhaseLabelConfig,
) -> list[str]:
    if qpos.ndim != 2 or qpos.shape[1] != 4:
        raise ValueError(f"qpos must have shape (T,4), got {qpos.shape}")
    if qvel.ndim != 2 or qvel.shape != qpos.shape:
        raise ValueError(f"qvel must have same shape as qpos, got {qvel.shape}")
    labels: list[str] = []
    state = "DIG"
    counters: dict[str, int] = {}
    previous_home_distance: float | None = None
    dump_seen = False
    dump_center = _range_center(cfg.dump_swing_range)
    dig_center = _range_center(cfg.dig_swing_range)

    for i in range(qpos.shape[0]):
        swing = float(qpos[i, 0])
        swing_vel = float(qvel[i, 0])
        bucket_vel = float(qvel[i, 3])
        home_distance = _home_distance(qpos[i], cfg)
        if int(go_home_running[i]) != 0 and state != "END":
            state = "GO_HOME"

        if state == "DIG":
            toward_dump = (
                dump_center is None
                or np.sign(dump_center - swing) == np.sign(swing_vel)
            )
            if _dwell(
                counters,
                "leave_dig",
                _outside(swing, cfg.dig_swing_range)
                and abs(swing_vel) >= cfg.swing_velocity_threshold_rad_s
                and bool(toward_dump),
                cfg.dwell_steps,
            ):
                state = "SWING_TO_DUMP"

        elif state == "SWING_TO_DUMP":
            if _dwell(
                counters,
                "enter_dump",
                _inside(swing, cfg.dump_swing_range)
                and abs(swing_vel) <= max(cfg.swing_velocity_threshold_rad_s * 2.0, 1e-6)
                and abs(bucket_vel) >= cfg.bucket_velocity_threshold_rad_s,
                cfg.dwell_steps,
            ):
                state = "DUMP"
                dump_seen = True

        elif state == "DUMP":
            toward_home = (
                dig_center is None
                or np.sign(dig_center - swing) == np.sign(swing_vel)
            )
            bucket_quiet = abs(bucket_vel) < cfg.bucket_velocity_threshold_rad_s
            if _dwell(
                counters,
                "return_home",
                dump_seen
                and bucket_quiet
                and (
                    (abs(swing_vel) >= cfg.swing_velocity_threshold_rad_s and bool(toward_home))
                    or (
                        previous_home_distance is not None
                        and home_distance is not None
                        and home_distance < previous_home_distance
                    )
                ),
                cfg.dwell_steps,
            ):
                state = "RETURN_NEAR_HOME"

        elif state == "RETURN_NEAR_HOME":
            if int(go_home_running[i]) != 0:
                state = "GO_HOME"
            elif _dwell(
                counters,
                "near_home",
                _near_home(qpos[i], cfg),
                cfg.dwell_steps,
            ):
                state = "GO_HOME"

        labels.append(state)
        previous_home_distance = home_distance

    if labels and labels[-1] == "GO_HOME":
        labels[-1] = "END"
    return labels


def _dwell(counters: dict[str, int], key: str, condition: bool, steps: int) -> bool:
    counters[key] = counters.get(key, 0) + 1 if condition else 0
    return counters[key] >= steps


def _inside(value: float, rng: tuple[float, float] | None) -> bool:
    if rng is None:
        return False
    return min(rng) <= value <= max(rng)


def _outside(value: float, rng: tuple[float, float] | None) -> bool:
    if rng is None:
        return False
    return not _inside(value, rng)


def _range_center(rng: tuple[float, float] | None) -> float | None:
    if rng is None:
        return None
    return 0.5 * (float(rng[0]) + float(rng[1]))


def _home_distance(qpos: np.ndarray, cfg: PhaseLabelConfig) -> float | None:
    if cfg.home_pose_rad is None:
        return None
    return float(np.linalg.norm(np.asarray(qpos, dtype=np.float32) - cfg.home_pose_rad))


def _near_home(qpos: np.ndarray, cfg: PhaseLabelConfig) -> bool:
    if cfg.home_pose_rad is None or cfg.near_home_tolerance_rad is None:
        return False
    return bool(np.all(np.abs(np.asarray(qpos, dtype=np.float32) - cfg.home_pose_rad) <= cfg.near_home_tolerance_rad))


def _transitions(labels: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    prev: str | None = None
    for step, phase in enumerate(labels):
        if phase != prev:
            out.append({"step": int(step), "phase": phase})
            prev = phase
    return out


def _optional_vec4(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        value = [float(value)] * 4
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.shape != (4,):
        raise ValueError(f"expected 4 values, got {arr.shape}")
    return arr


def _optional_range(value: Any) -> tuple[float, float] | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.shape != (2,):
        raise ValueError(f"expected range with 2 values, got {arr.shape}")
    return float(arr[0]), float(arr[1])
