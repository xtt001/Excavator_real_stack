#!/usr/bin/env python3
"""Generate versioned automatic L/C/R achieved-dig-sector sidecars.

Historical recordings do not contain a requested dig-sector command.  This
tool therefore writes hindsight ``actual_dig_sector`` labels only.  A record is
automatic-training-usable only when swing qpos and both eye-camera background
displacements independently produce the same non-boundary sector.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import cv2
import matplotlib
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
TESTBED_ROOT = REPO_ROOT / "testbed"
if str(TESTBED_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTBED_ROOT))

from testbed.data.dig_sector_annotation import (  # noqa: E402
    ALL_CAMERAS,
    CONTRACT_ID,
    ENTRY_METHOD,
    EYE_CAMERAS,
    FUSION_METHOD,
    VISION_METHOD,
    AnnotationConfig,
    CameraRegistration,
    EpisodeSignals,
    classify_qpos_sector,
    detect_entry_step,
    home_score,
    intersect_candidates,
    jsonable,
    load_episode,
    load_manifest_episode_names,
    register_eye_pair,
    robust_home_reference,
    wrapped_delta,
)

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


DEFAULT_DATASET = Path(
    "/data/pingfan/Excavator_real_stack_data/"
    "pro_real_teleop_20260713_20hz_v1"
)
DEFAULT_MANIFEST = DEFAULT_DATASET / "qc_full/train_ready_manifest.json"
DEFAULT_OUTPUT = REPO_ROOT / "docs/evidence/dig_sector_3x1_auto_v0_1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--calibration-id",
        default=AnnotationConfig.calibration_id,
        help="fixed eye-camera mounting and threshold calibration identifier",
    )
    return parser.parse_args()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    payload = "".join(
        json.dumps(record, ensure_ascii=False, default=jsonable) + "\n"
        for record in records
    )
    atomic_write_text(path, payload)


def source_block(episode: EpisodeSignals) -> dict[str, Any]:
    stat = episode.path.stat()
    return {
        "domain": "real",
        "dataset_path": str(episode.path.resolve()),
        "episode_name": episode.episode_name,
        "episode_id": episode.episode_id,
        "step_count": int(episode.qpos.shape[0]),
        "camera_names": list(ALL_CAMERAS),
        "source_size_bytes": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
    }


def empty_qpos_evidence(
    episode: EpisodeSignals,
    initial_score: float,
    config: AnnotationConfig,
) -> dict[str, Any]:
    return {
        "initial_qpos_rad": episode.initial_qpos_rad.tolist(),
        "initial_home_score": initial_score,
        "initial_home_score_threshold": config.home_outlier_score,
        "relative_swing_rad": None,
        "entry_window_swing_span_rad": None,
        "cumulative_swing_action_to_entry": None,
        "sector_thresholds_rad": {
            "center_accept_abs_max": config.qpos_center_max_abs_rad,
            "side_accept_abs_min": config.qpos_side_min_abs_rad,
            "boundary_band_abs": [
                config.qpos_center_max_abs_rad,
                config.qpos_side_min_abs_rad,
            ],
        },
        "sector_label": None,
        "candidate_sector_labels": [],
        "boundary_flag": False,
    }


def empty_vision_evidence(
    config: AnnotationConfig,
    status: str = "not_run",
) -> dict[str, Any]:
    return {
        "method": VISION_METHOD,
        "status": status,
        "calibration_id": config.calibration_id,
        "reference_step": 0,
        "image_resolution": [config.image_width, config.image_height],
        "static_background_mask": {
            "top_rows": config.static_mask_top_rows,
            "left_end_col": config.static_mask_left_end_col,
            "right_start_col": config.static_mask_right_start_col,
        },
        "sector_thresholds_px": {
            "center_accept_abs_max": config.vision_center_max_abs_px,
            "side_accept_abs_min": config.vision_side_min_abs_px,
            "boundary_band_abs": [
                config.vision_center_max_abs_px,
                config.vision_side_min_abs_px,
            ],
        },
        "cameras": [],
        "median_horizontal_displacement_px": None,
        "sector_label": None,
        "candidate_sector_labels": [],
        "boundary_flag": False,
    }


def base_record(
    episode: EpisodeSignals,
    initial_score: float,
    config: AnnotationConfig,
    generated_at: str,
) -> dict[str, Any]:
    return {
        "contract_id": CONTRACT_ID,
        "annotation_id": (
            f"real:{episode.episode_name}:actual_dig_sector_3x1:v0_1"
        ),
        "generated_at": generated_at,
        "source": source_block(episode),
        "semantics": {
            "label_kind": "hindsight_achieved_sector",
            "coordinate_frame": (
                "operator-left/center/right relative to the initial recorded pose"
            ),
            "positive_relative_swing": "L",
            "negative_relative_swing": "R",
            "metric_equal_width_grid_claimed": False,
        },
        "command": {
            "source": "unknown_not_recorded",
            "dig_sector": None,
        },
        "dig_entry": {
            "representative_step": None,
            "window_start_step": None,
            "window_end_step": None,
            "method": ENTRY_METHOD,
            "parameters": {
                "moving_median_window_steps": 5,
                "bucket_rise_rad": config.bucket_rise_rad,
                "bucket_rise_hold_rad": config.bucket_rise_hold_rad,
                "bucket_rise_hold_steps": config.bucket_rise_hold_steps,
                "bucket_rise_lookahead_steps": (
                    config.bucket_rise_lookahead_steps
                ),
            },
            "semantic_limit": (
                "bucket-motion timing proxy; not measured soil contact"
            ),
        },
        "qpos_evidence": empty_qpos_evidence(
            episode,
            initial_score,
            config,
        ),
        "vision_evidence": empty_vision_evidence(config),
        "outcome": {
            "actual_dig_sector": None,
            "candidate_sector_labels": [],
            "boundary_flag": False,
            "method": FUSION_METHOD,
        },
        "verification": {
            "privilege_used_for_annotation": False,
            "metric_ground_truth_available": False,
            "manual_review_performed": False,
        },
        "quality": {
            "status": "rejected",
            "confidence": "low",
            "auto_usable": False,
            "review_required": True,
            "reason_codes": [],
        },
    }


def vision_block(
    registrations: Sequence[CameraRegistration],
    config: AnnotationConfig,
) -> dict[str, Any]:
    successful = [item for item in registrations if item.status == "ok"]
    if len(successful) == len(EYE_CAMERAS):
        status = "ok"
    elif successful:
        status = "partial"
    else:
        status = "failed"

    groups = [item.candidate_sector_labels for item in successful]
    candidates = intersect_candidates(groups)
    labels = {
        item.sector_label
        for item in successful
        if item.sector_label is not None
    }
    fused_label = (
        next(iter(labels))
        if len(labels) == 1
        and len(successful) == len(EYE_CAMERAS)
        and all(not item.boundary_flag for item in successful)
        else None
    )
    horizontal = [
        float(item.horizontal_displacement_px)
        for item in successful
        if item.horizontal_displacement_px is not None
    ]
    return {
        **empty_vision_evidence(config, status=status),
        "cameras": [item.to_dict() for item in registrations],
        "median_horizontal_displacement_px": (
            float(np.median(horizontal)) if horizontal else None
        ),
        "sector_label": fused_label,
        "candidate_sector_labels": list(candidates),
        "boundary_flag": any(item.boundary_flag for item in successful),
    }


def annotate_episode(
    episode: EpisodeSignals,
    *,
    home_center: np.ndarray,
    home_scale: np.ndarray,
    config: AnnotationConfig,
    generated_at: str,
) -> dict[str, Any]:
    initial_score = home_score(
        episode.initial_qpos_rad,
        home_center,
        home_scale,
    )
    record = base_record(episode, initial_score, config, generated_at)
    reasons: list[str] = []

    if initial_score >= config.home_outlier_score:
        reasons.append("initial_pose_outside_home_cluster")
        record["quality"]["reason_codes"] = reasons
        return record

    entry = detect_entry_step(
        episode.qpos,
        float(episode.initial_qpos_rad[3]),
        config,
    )
    if entry is None:
        reasons.append("no_sustained_bucket_rise")
        record["quality"]["reason_codes"] = reasons
        return record

    window_start = max(0, entry - config.entry_window_radius_steps)
    window_end = min(
        episode.qpos.shape[0] - 1,
        entry + config.entry_window_radius_steps,
    )
    local_start = max(0, entry - 2)
    local_end = min(episode.qpos.shape[0], entry + 3)
    relative_swing = float(
        np.median(
            wrapped_delta(
                episode.qpos[local_start:local_end, 0],
                float(episode.initial_qpos_rad[0]),
            )
        )
    )
    window_swing = np.unwrap(
        wrapped_delta(
            episode.qpos[window_start : window_end + 1, 0],
            float(episode.initial_qpos_rad[0]),
        )
    )
    qpos_sector = classify_qpos_sector(relative_swing, config)
    record["dig_entry"].update(
        {
            "representative_step": entry,
            "window_start_step": window_start,
            "window_end_step": window_end,
        }
    )
    record["qpos_evidence"] = {
        "initial_qpos_rad": episode.initial_qpos_rad.tolist(),
        "initial_home_score": initial_score,
        "initial_home_score_threshold": config.home_outlier_score,
        "relative_swing_rad": relative_swing,
        "entry_window_swing_span_rad": float(np.ptp(window_swing)),
        "cumulative_swing_action_to_entry": float(
            np.sum(episode.action[: entry + 1, 0])
        ),
        "sector_thresholds_rad": {
            "center_accept_abs_max": config.qpos_center_max_abs_rad,
            "side_accept_abs_min": config.qpos_side_min_abs_rad,
            "boundary_band_abs": [
                config.qpos_center_max_abs_rad,
                config.qpos_side_min_abs_rad,
            ],
        },
        "sector_label": qpos_sector.label,
        "candidate_sector_labels": list(qpos_sector.candidate_labels),
        "boundary_flag": qpos_sector.boundary,
    }

    registrations = register_eye_pair(episode.path, entry, config)
    visual = vision_block(registrations, config)
    record["vision_evidence"] = visual

    successful = [
        item for item in registrations if item.status == "ok"
    ]
    all_groups: list[Sequence[str]] = [qpos_sector.candidate_labels]
    all_groups.extend(item.candidate_sector_labels for item in successful)
    fused_candidates = intersect_candidates(all_groups)

    clear_three_sensor_agreement = (
        qpos_sector.label is not None
        and not qpos_sector.boundary
        and len(successful) == len(EYE_CAMERAS)
        and all(
            item.sector_label == qpos_sector.label
            and not item.boundary_flag
            for item in successful
        )
    )
    if clear_three_sensor_agreement:
        label = str(qpos_sector.label)
        record["outcome"] = {
            "actual_dig_sector": label,
            "candidate_sector_labels": [label],
            "boundary_flag": False,
            "method": FUSION_METHOD,
        }
        record["quality"] = {
            "status": "accepted",
            "confidence": "high",
            "auto_usable": True,
            "review_required": False,
            "reason_codes": [
                "qpos_and_both_eye_cameras_agree",
            ],
        }
        return record

    if len(successful) < len(EYE_CAMERAS):
        reasons.append("incomplete_eye_pair_registration")
    if qpos_sector.boundary or any(
        item.boundary_flag for item in successful
    ):
        reasons.append("sensor_boundary_requires_review")
    if fused_candidates:
        reasons.append("sensor_candidates_overlap_but_not_fully_accepted")
        record["outcome"] = {
            "actual_dig_sector": None,
            "candidate_sector_labels": list(fused_candidates),
            "boundary_flag": (
                qpos_sector.boundary
                or any(item.boundary_flag for item in successful)
            ),
            "method": FUSION_METHOD,
        }
        record["quality"] = {
            "status": "provisional",
            "confidence": "medium",
            "auto_usable": False,
            "review_required": True,
            "reason_codes": reasons,
        }
        return record

    reasons.append(
        "qpos_eye_sector_conflict"
        if successful
        else "eye_registration_unavailable"
    )
    record["quality"]["reason_codes"] = reasons
    return record


def quantiles(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {}
    return {
        key: float(np.quantile(array, level))
        for key, level in (
            ("min", 0.0),
            ("p05", 0.05),
            ("p50", 0.50),
            ("p95", 0.95),
            ("max", 1.0),
        )
    }


def build_summary(
    records: Sequence[dict[str, Any]],
    *,
    config: AnnotationConfig,
    dataset_dir: Path,
    manifest: Path,
    home_center: np.ndarray,
    home_scale: np.ndarray,
    generated_at: str,
) -> dict[str, Any]:
    status_counts = Counter(record["quality"]["status"] for record in records)
    accepted = [
        record for record in records if record["quality"]["auto_usable"]
    ]
    label_counts = Counter(
        record["outcome"]["actual_dig_sector"] for record in accepted
    )
    reason_counts = Counter(
        reason
        for record in records
        for reason in record["quality"]["reason_codes"]
    )
    entry_records = [
        record
        for record in records
        if record["dig_entry"]["representative_step"] is not None
    ]
    eye_results = [
        camera
        for record in entry_records
        for camera in record["vision_evidence"]["cameras"]
    ]
    paired = [
        record
        for record in entry_records
        if record["vision_evidence"]["median_horizontal_displacement_px"]
        is not None
        and record["qpos_evidence"]["relative_swing_rad"] is not None
    ]
    qpos_values = np.asarray(
        [
            record["qpos_evidence"]["relative_swing_rad"]
            for record in paired
        ],
        dtype=np.float64,
    )
    vision_values = np.asarray(
        [
            record["vision_evidence"]["median_horizontal_displacement_px"]
            for record in paired
        ],
        dtype=np.float64,
    )
    correlation = (
        float(np.corrcoef(qpos_values, vision_values)[0, 1])
        if qpos_values.size >= 2
        else None
    )
    accepted_center_abs = [
        abs(record["vision_evidence"]["median_horizontal_displacement_px"])
        for record in accepted
        if record["outcome"]["actual_dig_sector"] == "C"
    ]
    accepted_side_abs = [
        abs(record["vision_evidence"]["median_horizontal_displacement_px"])
        for record in accepted
        if record["outcome"]["actual_dig_sector"] in {"L", "R"}
    ]
    return {
        "contract_id": CONTRACT_ID,
        "generated_at": generated_at,
        "dataset_dir": str(dataset_dir.resolve()),
        "manifest": str(manifest.resolve()),
        "source_hdf5_modified": False,
        "record_count": len(records),
        "status_counts": dict(sorted(status_counts.items())),
        "accepted_label_counts": {
            label: int(label_counts.get(label, 0))
            for label in ("L", "C", "R")
        },
        "automatic_coverage": (
            len(accepted) / len(records) if records else None
        ),
        "entry_proxy_found_count": len(entry_records),
        "eye_registration": {
            "attempted_camera_count": len(eye_results),
            "successful_camera_count": sum(
                result["status"] == "ok" for result in eye_results
            ),
            "good_match_count_quantiles": quantiles(
                [float(result["good_match_count"]) for result in eye_results]
            ),
            "robust_match_count_quantiles": quantiles(
                [
                    float(result["robust_match_count"])
                    for result in eye_results
                ]
            ),
        },
        "cross_sensor_checks": {
            "paired_episode_count": len(paired),
            "qpos_vs_eye_displacement_pearson": correlation,
            "three_sensor_clear_agreement_count": len(accepted),
            "three_sensor_clear_agreement_rate_among_entry_records": (
                len(accepted) / len(entry_records)
                if entry_records
                else None
            ),
            "accepted_center_max_abs_eye_displacement_px": (
                max(accepted_center_abs) if accepted_center_abs else None
            ),
            "accepted_side_min_abs_eye_displacement_px": (
                min(accepted_side_abs) if accepted_side_abs else None
            ),
        },
        "initial_pose_gate": {
            "robust_center_qpos_rad": home_center,
            "robust_scale_qpos_rad": home_scale,
            "outlier_score_threshold": config.home_outlier_score,
        },
        "config": asdict(config),
        "reason_counts": dict(sorted(reason_counts.items())),
        "review_queue": [
            {
                "annotation_id": record["annotation_id"],
                "episode_id": record["source"]["episode_id"],
                "status": record["quality"]["status"],
                "reason_codes": record["quality"]["reason_codes"],
            }
            for record in records
            if record["quality"]["review_required"]
        ],
        "capability_boundary": [
            (
                "Labels are achieved sectors relative to the initial pose, "
                "not physical equal-width sandbox thirds."
            ),
            (
                "The entry step is a sustained bucket-motion proxy, not "
                "measured soil contact."
            ),
            (
                "Historical command remains unknown; hindsight outcomes do "
                "not prove instruction following."
            ),
            (
                "The visual calibration is valid only while the eye-camera "
                "mounting and image preprocessing remain unchanged."
            ),
        ],
    }


def write_scatter_plot(
    path: Path,
    records: Sequence[dict[str, Any]],
    config: AnnotationConfig,
) -> None:
    colors = {"L": "#d95f02", "C": "#1b9e77", "R": "#7570b3"}
    figure, axis = plt.subplots(figsize=(9, 6), constrained_layout=True)
    for record in records:
        qpos = record["qpos_evidence"]["relative_swing_rad"]
        displacement = record["vision_evidence"][
            "median_horizontal_displacement_px"
        ]
        label = record["outcome"]["actual_dig_sector"]
        if qpos is None or displacement is None:
            continue
        axis.scatter(
            qpos,
            displacement,
            color=colors.get(label, "#777777"),
            s=28,
            alpha=0.8,
        )
    for value in (
        -config.qpos_side_min_abs_rad,
        -config.qpos_center_max_abs_rad,
        config.qpos_center_max_abs_rad,
        config.qpos_side_min_abs_rad,
    ):
        axis.axvline(value, color="#999999", linestyle="--", linewidth=0.8)
    for value in (
        -config.vision_side_min_abs_px,
        -config.vision_center_max_abs_px,
        config.vision_center_max_abs_px,
        config.vision_side_min_abs_px,
    ):
        axis.axhline(value, color="#999999", linestyle=":", linewidth=0.8)
    axis.set_xlabel("relative swing qpos at entry proxy (rad)")
    axis.set_ylabel("median eye-background horizontal displacement (px)")
    axis.set_title("Automatic 3x1 label: independent qpos and eye evidence")
    axis.grid(alpha=0.2)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def write_report(
    path: Path,
    summary: dict[str, Any],
) -> None:
    counts = summary["status_counts"]
    labels = summary["accepted_label_counts"]
    cross = summary["cross_sensor_checks"]
    review = summary["review_queue"]
    text = f"""# 左/中/右实际下铲扇区自动标注 v0.1

## 可直接使用的结论

当前批次共 {summary['record_count']} 条轨迹，其中
{counts.get('accepted', 0)} 条通过 swing qpos 与两路 eye 相机的三传感器一致性门槛，
可作为自动生成的 hindsight `actual_dig_sector`；其余
{counts.get('provisional', 0) + counts.get('rejected', 0)} 条进入审查队列，不会静默写入训练标签。

自动接收标签分布为：左 {labels['L']}、中 {labels['C']}、右 {labels['R']}。
分布是否均衡不影响标注流程是否成立；训练覆盖应在标注之后单独处理。

## 自动接收规则

一条记录只有同时满足以下条件，`quality.auto_usable` 才为 `true`：

1. 开始姿态属于当前 home 聚类；
2. 能检测到持续的 bucket qpos 上升；
3. 下铲代理时刻的相对 swing qpos 不在边界带；
4. video4 和 video5 的静态背景位移都能可靠配准；
5. qpos、video4、video5 给出同一个非边界 L/C/R 扇区。

任一证据缺失、落入边界或发生冲突，结果都会转为 provisional/rejected，并写入
`review_queue.jsonl`。

## 独立证据是否一致

![qpos 与 eye 背景位移](qpos_vs_eye_displacement.png)

可配对样本数为 {cross['paired_episode_count']}，相对 swing 与 eye 背景水平位移的
Pearson 相关系数为 {cross['qpos_vs_eye_displacement_pearson']:.6f}。负相关来自相机
坐标关系：机身向左回转时，静态背景在画面中向左移动。

自动接收的中间样本最大绝对背景位移为
{cross['accepted_center_max_abs_eye_displacement_px']:.3f} px，侧向样本最小绝对位移为
{cross['accepted_side_min_abs_eye_displacement_px']:.3f} px。v0.1 在两者之间保留
10–15 px 的拒绝/复核边界带。

## 文件语义

- `annotations.jsonl`：全量、版本化 sidecar；源 HDF5 不被修改；
- `review_queue.jsonl`：仅包含未自动接收的完整记录；
- `summary.json`：覆盖率、阈值、配准质量和能力边界；
- `qpos_vs_eye_displacement.png`：跨传感器一致性证据。

历史数据没有录制“要求去左/中/右”的 command，因此 command 始终保持
`unknown_not_recorded`。自动标注得到的是最终实际到达扇区，可用于 hindsight
relabeling；它不能用于证明策略遵从了一个历史上不存在的指令。

## 当前审查队列

待审查记录数：{len(review)}。

```json
{json.dumps(review, ensure_ascii=False, indent=2)}
```

## 能力边界

- L/C/R 是相对开始姿态的回转扇区，不是沙箱物理宽度严格三等分；
- bucket 事件只定位明确卷收开始，不声称测得真实入土接触；
- 相机固定方式、分辨率或裁剪变化后，必须换 calibration id 并重新标定像素阈值；
- 本报告来自离线 HDF5 replay，不是真机闭环执行。
"""
    atomic_write_text(path, text)


def main() -> int:
    args = parse_args()
    config = AnnotationConfig(calibration_id=args.calibration_id)
    config.validate()
    cv2.setNumThreads(1)
    cv2.setRNGSeed(0)

    names = load_manifest_episode_names(args.manifest)
    episodes = [
        load_episode(
            args.dataset_dir,
            name,
            config.initial_window_steps,
        )
        for name in names
    ]
    home_center, home_scale = robust_home_reference(episodes)
    generated_at = datetime.now(timezone.utc).isoformat()
    records = [
        annotate_episode(
            episode,
            home_center=home_center,
            home_scale=home_scale,
            config=config,
            generated_at=generated_at,
        )
        for episode in episodes
    ]
    summary = build_summary(
        records,
        config=config,
        dataset_dir=args.dataset_dir,
        manifest=args.manifest,
        home_center=home_center,
        home_scale=home_scale,
        generated_at=generated_at,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "annotations.jsonl", records)
    write_jsonl(
        args.output_dir / "review_queue.jsonl",
        [
            record
            for record in records
            if record["quality"]["review_required"]
        ],
    )
    atomic_write_text(
        args.output_dir / "summary.json",
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
            default=jsonable,
        )
        + "\n",
    )
    write_scatter_plot(
        args.output_dir / "qpos_vs_eye_displacement.png",
        records,
        config,
    )
    write_report(args.output_dir / "report.md", summary)
    print(
        json.dumps(
            {
                "records": len(records),
                "status_counts": summary["status_counts"],
                "accepted_label_counts": summary["accepted_label_counts"],
                "automatic_coverage": summary["automatic_coverage"],
                "review_queue_count": len(summary["review_queue"]),
                "output_dir": str(args.output_dir.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
