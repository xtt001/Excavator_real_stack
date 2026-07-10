#!/usr/bin/env python3
"""Summarize one E52 receiver control trace without commanding hardware."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ACTION_KEYS = (
    "policy_action",
    "policy_scaled_action",
    "policy_phase_gated_action",
    "policy_snap_action",
    "policy_temporal_direction_action",
    "policy_returned_action",
    "safe_action",
    "commanded_action",
)
REQUIRED_TRACE_KEYS = (
    "qpos",
    "qvel",
    "policy_action",
    "policy_intent_probabilities",
    "policy_phase_gated_action",
    "policy_snap_action",
    "temporal_direction_gate_probabilities",
    "policy_temporal_direction_action",
    "policy_returned_action",
    "safe_action",
    "commanded_action",
    "controller_ack",
    "receiver_health_ok",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--preflight-report", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    paths = summarize_trace(
        run_dir=args.run_dir,
        config_path=args.config,
        bundle_dir=args.bundle_dir,
        preflight_report=args.preflight_report,
        output_dir=args.output_dir,
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


def summarize_trace(
    *,
    run_dir: Path,
    config_path: Path,
    bundle_dir: Path,
    preflight_report: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Path]:
    run = Path(run_dir)
    steps = _read_jsonl(run / "steps.jsonl")
    if not steps:
        raise ValueError(f"trace contains no steps: {run}")
    metadata = _read_json(run / "metadata.json")
    receiver_summary = _read_json(run / "summary.json")
    termination = _read_json(run / "termination.json")
    config = _read_yaml(config_path)
    bundle = Path(bundle_dir)
    manifest_path = bundle / "candidate_package_manifest.json"
    candidate_manifest = _read_json(manifest_path)
    preflight = _read_json(preflight_report) if preflight_report is not None else None

    output = Path(output_dir) if output_dir is not None else run / "e52_trace_analysis"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "run_manifest": output / "run_manifest.json",
        "trace_summary": output / "trace_summary.json",
        "trace_context": output / "trace_context.json",
        "trace_timeline": output / "trace_timeline.png",
    }

    trace_summary = _build_trace_summary(
        steps,
        receiver_summary=receiver_summary,
        termination=termination,
        metadata=metadata,
    )
    run_manifest = {
        "schema_version": 1,
        "claim_boundary": "trace_completeness_and_observed_execution_only",
        "run_dir": str(run.resolve()),
        "repository": _git_provenance(Path(__file__).resolve().parents[1]),
        "config": {
            "path": str(Path(config_path).resolve()),
            "sha256": _sha256(Path(config_path)),
            "policy_output_mode_in_file": str(
                config.get("teleop", {}).get("policy", {}).get("output_mode", "")
            ),
            "record_config_sha256": hashlib.sha256(
                str(metadata.get("record_config_yaml", "")).encode("utf-8")
            ).hexdigest(),
            "record_policy_output_mode": str(
                _record_config(metadata)
                .get("teleop", {})
                .get("policy", {})
                .get("output_mode", "")
            ),
            "observed_policy_output_modes": sorted(
                {str(step.get("policy_output_mode", "")) for step in steps}
            ),
        },
        "bundle": {
            "path": str(bundle.resolve()),
            "candidate_manifest": str(manifest_path.resolve()),
            "candidate_manifest_sha256": _sha256(manifest_path),
            "candidate_id": str(candidate_manifest.get("candidate_id", "")),
            "declared_artifacts": [
                {
                    "name": str(item.get("name", "")),
                    "sha256": str(item.get("sha256", "")),
                    "source_path": str(item.get("path", "")),
                }
                for item in candidate_manifest.get("artifacts", [])
                if isinstance(item, dict)
            ],
        },
        "preflight_report": preflight,
        "receiver_metadata": metadata,
    }
    context = _trace_context(steps, trace_summary)
    _write_json(paths["run_manifest"], run_manifest)
    _write_json(paths["trace_summary"], trace_summary)
    _write_json(paths["trace_context"], context)
    _plot_trace(steps, paths["trace_timeline"])
    return paths


def _build_trace_summary(
    steps: list[dict[str, Any]],
    *,
    receiver_summary: dict[str, Any],
    termination: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    missing_counts = {
        key: sum(1 for step in steps if key not in step or step.get(key) is None)
        for key in REQUIRED_TRACE_KEYS
    }
    malformed_counts = {
        key: sum(1 for step in steps if _vec(step.get(key), width=4) is None)
        for key in ACTION_KEYS
    }
    actions = {
        key: np.stack(
            [_required_vec(step.get(key), width=4, name=key) for step in steps]
        )
        for key in ACTION_KEYS
        if not malformed_counts[key]
    }
    qpos = np.stack(
        [_required_vec(step.get("qpos"), width=4, name="qpos") for step in steps]
    )
    qvel = np.stack(
        [_required_vec(step.get("qvel"), width=4, name="qvel") for step in steps]
    )
    qpos_delta = qpos[-1] - qpos[0]
    qpos_delta[0] = (qpos_delta[0] + np.pi) % (2.0 * np.pi) - np.pi
    action_metrics = {
        key: {
            "mean": values.mean(axis=0).tolist(),
            "mean_abs": np.abs(values).mean(axis=0).tolist(),
            "max_abs": np.abs(values).max(axis=0).tolist(),
            "nonzero_steps": int(
                np.count_nonzero(np.any(np.abs(values) > 1.0e-6, axis=1))
            ),
        }
        for key, values in actions.items()
    }
    modifications = {}
    for name, before, after in (
        ("phase", "policy_scaled_action", "policy_phase_gated_action"),
        ("snap", "policy_phase_gated_action", "policy_snap_action"),
        ("temporal", "policy_snap_action", "policy_temporal_direction_action"),
        ("guard_or_backend", "policy_returned_action", "commanded_action"),
    ):
        if before in actions and after in actions:
            delta = np.abs(actions[after] - actions[before])
            modifications[name] = {
                "changed_steps": int(np.count_nonzero(np.any(delta > 1.0e-7, axis=1))),
                "mean_abs_delta": delta.mean(axis=0).tolist(),
                "max_abs_delta": delta.max(axis=0).tolist(),
            }
    wall = [int(step.get("wall_time_ns", 0) or 0) for step in steps]
    duration_s = (wall[-1] - wall[0]) * 1.0e-9 if wall[-1] > wall[0] else 0.0
    errors = []
    for key, count in missing_counts.items():
        if count:
            errors.append(f"missing {key} on {count} step(s)")
    for key, count in malformed_counts.items():
        if count:
            errors.append(f"malformed {key} on {count} step(s)")
    policy_error_steps = sum(1 for step in steps if str(step.get("policy_error", "")))
    health_bad_steps = sum(
        1 for step in steps if not bool(step.get("receiver_health_ok", 0))
    )
    ack_bad_steps = sum(1 for step in steps if not bool(step.get("controller_ack", 0)))
    if policy_error_steps:
        errors.append(f"policy_error on {policy_error_steps} step(s)")
    if not termination.get("zero_command_confirmed", False):
        errors.append("terminal zero command is not confirmed")
    image_steps = [
        int(step.get("local_step", 0)) for step in steps if step.get("fpv_image_path")
    ]
    return {
        "schema_version": 1,
        "claim_boundary": "trace_completeness_and_observed_execution_only",
        "trace_complete": not errors,
        "trace_errors": errors,
        "steps": len(steps),
        "duration_s": duration_s,
        "effective_hz": (len(steps) - 1) / duration_s if duration_s > 0.0 else None,
        "receiver_stop_reason": str(receiver_summary.get("stop_reason", "")),
        "termination": termination,
        "missing_counts": missing_counts,
        "malformed_action_counts": malformed_counts,
        "policy_error_steps": policy_error_steps,
        "receiver_health_bad_steps": health_bad_steps,
        "controller_ack_bad_steps": ack_bad_steps,
        "guard_triggered_steps": sum(
            1 for step in steps if bool(step.get("guard_triggered", 0))
        ),
        "gohome_requested_steps": sum(
            1 for step in steps if bool(step.get("go_home_requested", 0))
        ),
        "action_metrics": action_metrics,
        "gate_modifications": modifications,
        "state": {
            "qpos_start": qpos[0].tolist(),
            "qpos_end": qpos[-1].tolist(),
            "qpos_delta_shortest_swing": qpos_delta.tolist(),
            "qpos_range": (qpos.max(axis=0) - qpos.min(axis=0)).tolist(),
            "qvel_peak_abs": np.abs(qvel).max(axis=0).tolist(),
        },
        "stick": {
            "policy_max_abs": _axis_max(actions.get("policy_action"), 2),
            "e52_max_abs": _axis_max(
                actions.get("policy_temporal_direction_action"), 2
            ),
            "commanded_max_abs": _axis_max(actions.get("commanded_action"), 2),
        },
        "image_capture": {
            "captured": len(image_steps),
            "steps": image_steps,
            "metadata": metadata.get("image_capture", {}),
        },
    }


def _trace_context(
    steps: list[dict[str, Any]], summary: dict[str, Any]
) -> dict[str, Any]:
    anomaly_indices = [
        index
        for index, step in enumerate(steps)
        if str(step.get("policy_error", ""))
        or not bool(step.get("receiver_health_ok", 0))
        or not bool(step.get("controller_ack", 0))
        or bool(step.get("guard_triggered", 0))
    ]
    anchor = anomaly_indices[0] if anomaly_indices else len(steps) - 1
    start = max(0, anchor - 10)
    end = min(len(steps), anchor + 11)
    keys = (
        "wall_time_ns",
        "local_step",
        "qpos",
        "qvel",
        "policy_action",
        "policy_phase_gated_action",
        "policy_snap_action",
        "policy_temporal_direction_action",
        "safe_action",
        "commanded_action",
        "raw_low_level_command",
        "phase_gate_prob",
        "temporal_direction_gate_probabilities",
        "go_home_requested",
        "guard_triggered",
        "guard_reasons",
        "receiver_health_ok",
        "receiver_health_error_code",
        "controller_ack",
        "controller_fault_code",
        "fpv_image_path",
    )
    return {
        "schema_version": 1,
        "anchor_kind": "first_anomaly" if anomaly_indices else "last_step",
        "anchor_index": anchor,
        "window_start": start,
        "window_end_exclusive": end,
        "trace_complete": summary["trace_complete"],
        "rows": [
            {key: steps[index].get(key) for key in keys} for index in range(start, end)
        ],
    }


def _plot_trace(steps: list[dict[str, Any]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t0 = int(steps[0].get("wall_time_ns", 0) or 0)
    time_s = np.asarray(
        [(int(step.get("wall_time_ns", t0) or t0) - t0) * 1.0e-9 for step in steps]
    )
    qpos = np.stack(
        [_required_vec(step.get("qpos"), width=4, name="qpos") for step in steps]
    )
    qvel = np.stack(
        [_required_vec(step.get("qvel"), width=4, name="qvel") for step in steps]
    )
    fig, axes = plt.subplots(4, 1, figsize=(16, 13), sharex=True)
    labels = ("swing", "boom", "stick", "bucket")
    for key in (
        "policy_action",
        "policy_temporal_direction_action",
        "commanded_action",
    ):
        values = [_vec(step.get(key), width=4) for step in steps]
        if any(value is None for value in values):
            continue
        array = np.stack(values)
        for axis in range(4):
            axes[0].plot(
                time_s, array[:, axis], label=f"{key}:{labels[axis]}", alpha=0.75
            )
    for axis in range(4):
        axes[1].plot(time_s, qpos[:, axis], label=labels[axis])
        axes[2].plot(time_s, qvel[:, axis], label=labels[axis])
    axes[3].plot(
        time_s,
        [float(step.get("phase_gate_prob", 0.0) or 0.0) for step in steps],
        label="phase",
    )
    axes[3].plot(
        time_s,
        [float(step.get("gohome_request_probability", 0.0) or 0.0) for step in steps],
        label="gohome",
    )
    axes[0].set_ylabel("Action")
    axes[1].set_ylabel("qpos (rad)")
    axes[2].set_ylabel("qvel (rad/s)")
    axes[3].set_ylabel("Probability")
    axes[3].set_xlabel("Time (s)")
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend(ncol=4, fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _git_provenance(repo: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=repo, capture_output=True, text=True, check=False
        )
        return result.stdout.strip() if result.returncode == 0 else ""

    status = run("status", "--short")
    return {
        "path": str(repo.resolve()),
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(status),
        "status_short": status.splitlines(),
    }


def _axis_max(values: np.ndarray | None, axis: int) -> float | None:
    return float(np.abs(values[:, axis]).max()) if values is not None else None


def _vec(value: Any, *, width: int) -> np.ndarray | None:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if array.shape != (width,) or not np.all(np.isfinite(array)):
        return None
    return array


def _required_vec(value: Any, *, width: int, name: str) -> np.ndarray:
    array = _vec(value, width=width)
    if array is None:
        raise ValueError(f"{name} must be a finite vector with shape ({width},)")
    return array


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} must be a JSON object")
        rows.append(value)
    return rows


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not Path(path).is_file():
        raise FileNotFoundError(path)
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return value


def _record_config(metadata: dict[str, Any]) -> dict[str, Any]:
    value = yaml.safe_load(str(metadata.get("record_config_yaml", ""))) or {}
    return value if isinstance(value, dict) else {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
