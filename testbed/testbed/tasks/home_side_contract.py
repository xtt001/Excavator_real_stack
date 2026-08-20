"""Field calibration resolver for the v2.0.1 A/B home-side contract."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from testbed.tasks.real_transition import (
    TransitionContractError,
    sha256_file,
    write_immutable_text,
)

CALIBRATION_INPUT_SCHEMA = "real_transition_home_calibration_samples_v1"
HOME_SIDE_CONTRACT_SCHEMA = "real_home_side_contract_v1"
READY_RULE_CONTRACT_SCHEMA = "real_transition_ready_rule_contract_v2"
BASELINE_ID = "real_transition_historical_baseline_20260813"

READY_RULE_DEFAULTS = {
    "home_swing_qpos_rad": 0.000690,
    "physical_left_qpos_sign": -1,
    "home_tolerance_rad": 0.05,
    "clean_ready_min_abs_delta_rad": 0.08,
    "safe_swing_qpos_range_rad": [-0.3892, 0.4189],
    "stable_window_s": 0.5,
    "swing_qvel_abs_max_rad_s": 0.015,
    "max_sample_gap_s": 0.15,
}

READY_BASELINE = {
    "dwell_s": 0.5,
    "commanded_action_abs_max": 0.05,
    "qvel_abs_max": [0.015, 0.015, 0.020, 0.020],
    "qpos_peak_to_peak_max": [0.005, 0.005, 0.005, 0.005],
    "visual_confirmation_required": True,
}


def build_rule_ready_contract() -> dict[str, Any]:
    """Build the field-confirmed v2 rule contract without pose calibration files."""

    contract: dict[str, Any] = {
        "schema": READY_RULE_CONTRACT_SCHEMA,
        "context_version": "v2.0.1-field-ready-rule-20260817",
        "swing_axis": {
            "axis_index": 0,
            **READY_RULE_DEFAULTS,
            "side_mapping": {
                "A": "physical_left_of_home",
                "B": "physical_right_of_home",
            },
        },
        "ready_requirements": {
            "clean_side_required": True,
            "swing_stable_window_required": True,
            "bucket_clear_confirmation_required": True,
            "operator_confirmation_required": True,
            "non_swing_qpos_policy": "record_only_unbounded",
            "non_swing_qvel_policy": "record_only_not_a_ready_gate",
        },
        "threshold_evidence": {
            "field_date": "2026-08-17",
            "hdf5_episode_ids": ["episode_111", "episode_112", "episode_113"],
            "natural_stop_rule": (
                "accept only when every swing qvel sample in the latest 0.5 s "
                "window is <= 0.015 rad/s"
            ),
            "episode_113_sha256": (
                "4e41aebeff36118a78ab488c83b4cf0f8db02289205ed51b83a8eb9186d753b5"
            ),
        },
        "contract_sha256_scope": "canonical_json_without_contract_sha256",
    }
    contract["contract_sha256"] = _contract_digest(contract)
    return contract


def write_rule_ready_contract(output_path: Path | str) -> dict[str, Any]:
    contract = build_rule_ready_contract()
    output = write_immutable_text(
        output_path,
        json.dumps(
            contract,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    return {
        "status": "PASS",
        "output": str(output),
        "sha256": sha256_file(output),
        "contract_sha256": contract["contract_sha256"],
    }


def validate_rule_ready_contract(contract: Mapping[str, Any]) -> None:
    if contract.get("schema") != READY_RULE_CONTRACT_SCHEMA:
        raise TransitionContractError("ready rule contract schema mismatch")
    if contract.get("contract_sha256") != _contract_digest(contract):
        raise TransitionContractError("ready rule contract digest mismatch")
    swing = contract.get("swing_axis", {})
    if not isinstance(swing, Mapping):
        raise TransitionContractError("ready rule swing_axis must be an object")
    if int(swing.get("axis_index", -1)) != 0:
        raise TransitionContractError("ready rule swing axis index must be 0")
    for key, expected in READY_RULE_DEFAULTS.items():
        actual = swing.get(key)
        if isinstance(expected, list):
            if list(actual or ()) != expected:
                raise TransitionContractError(f"ready rule {key} is not field-frozen")
        elif not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12):
            raise TransitionContractError(f"ready rule {key} is not field-frozen")
    if swing.get("side_mapping") != {
        "A": "physical_left_of_home",
        "B": "physical_right_of_home",
    }:
        raise TransitionContractError("ready rule A/B side mapping is invalid")
    requirements = contract.get("ready_requirements", {})
    required_true = (
        "clean_side_required",
        "swing_stable_window_required",
        "bucket_clear_confirmation_required",
        "operator_confirmation_required",
    )
    if not isinstance(requirements, Mapping) or any(
        requirements.get(name) is not True for name in required_true
    ):
        raise TransitionContractError("ready rule confirmations/gates must be enabled")
    if requirements.get("non_swing_qpos_policy") != "record_only_unbounded":
        raise TransitionContractError("non-swing qpos must remain unbounded")
    if requirements.get("non_swing_qvel_policy") != "record_only_not_a_ready_gate":
        raise TransitionContractError("non-swing qvel must remain record-only")


def classify_ready_swing_qpos(
    contract: Mapping[str, Any],
    swing_qpos_rad: float,
) -> str:
    """Classify one current swing qpos as home/transition/A/B/outside-safe."""

    validate_rule_ready_contract(contract)
    value = float(swing_qpos_rad)
    if not math.isfinite(value):
        raise TransitionContractError("swing qpos must be finite")
    swing = contract["swing_axis"]
    lower, upper = [float(item) for item in swing["safe_swing_qpos_range_rad"]]
    if value < lower or value > upper:
        return "outside_safe_range"
    delta = _shortest_angle(value - float(swing["home_swing_qpos_rad"]))
    abs_delta = abs(delta)
    if abs_delta <= float(swing["home_tolerance_rad"]):
        return "home"
    if abs_delta < float(swing["clean_ready_min_abs_delta_rad"]):
        return "transition"
    left_sign = int(swing["physical_left_qpos_sign"])
    return "A" if delta * left_sign > 0.0 else "B"


def build_home_side_contract(
    calibration: Mapping[str, Any],
    *,
    source_base_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Resolve home, deadband, and demonstrated A/B support from field windows."""

    if calibration.get("schema") != CALIBRATION_INPUT_SCHEMA:
        raise TransitionContractError(
            f"calibration schema must be {CALIBRATION_INPUT_SCHEMA!r}"
        )
    context_version = _required_text(calibration, "context_version")
    resolved_by = _required_text(calibration, "resolved_by")
    resolved_at = _required_text(calibration, "resolved_at")
    physical_left_sign = int(calibration.get("physical_left_qpos_sign", 0))
    if physical_left_sign not in {-1, 1}:
        raise TransitionContractError(
            "physical_left_qpos_sign must be explicitly confirmed as -1 or +1"
        )

    home_reference = calibration.get("home_reference", {})
    if not isinstance(home_reference, Mapping):
        raise TransitionContractError("home_reference must be an object")
    source_config_raw = _required_text(home_reference, "source_config")
    source_config = Path(source_config_raw).expanduser()
    if not source_config.is_absolute() and source_base_dir is not None:
        source_config = Path(source_base_dir) / source_config
    source_config = source_config.resolve()
    if not source_config.is_file():
        raise TransitionContractError(
            f"home_reference.source_config does not exist: {source_config}"
        )
    home_pose = _vector4(home_reference.get("home_pose_rad"), "home_pose_rad")
    source_value_path = _required_text(home_reference, "source_value_path")
    try:
        source_payload = yaml.safe_load(source_config.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise TransitionContractError(
            f"cannot read home source config {source_config}: {exc}"
        ) from exc
    source_home_pose = _vector4(
        _resolve_mapping_path(source_payload, source_value_path),
        f"source_config:{source_value_path}",
    )
    if not np.allclose(source_home_pose, home_pose, rtol=0.0, atol=1e-9):
        raise TransitionContractError(
            "home_reference.home_pose_rad does not match its source_config value"
        )

    ready_candidate = resolve_home_calibration_ready_candidate(calibration)
    samples_value = calibration.get("samples", ())
    if not isinstance(samples_value, Sequence) or isinstance(
        samples_value, (str, bytes)
    ):
        raise TransitionContractError("calibration samples must be a list")
    samples = [
        validate_home_calibration_sample(
            sample,
            ready_candidate=ready_candidate,
        )
        for sample in samples_value
    ]
    ids = [sample["reference_id"] for sample in samples]
    if len(ids) != len(set(ids)):
        raise TransitionContractError("calibration reference_id values must be unique")
    counts = Counter(sample["side"] for sample in samples)
    for side in ("home", "A", "B"):
        if counts[side] < 10:
            raise TransitionContractError(
                f"calibration requires at least 10 accepted {side} windows, found {counts[side]}"
            )

    home_samples = [sample for sample in samples if sample["side"] == "home"]
    home_swing_means = np.asarray(
        [sample["qpos_window_mean"][0] for sample in home_samples],
        dtype=np.float64,
    )
    medoid_index = _circular_medoid_index(home_swing_means)
    home_swing_axis = float(home_swing_means[medoid_index])
    center_errors = np.asarray(
        [_shortest_angle(value - home_swing_axis) for value in home_swing_means],
        dtype=np.float64,
    )
    max_repeat_error = float(np.max(np.abs(center_errors)))
    repeat_p95 = float(np.percentile(np.abs(center_errors), 95))
    deadband = max(0.05, _ceil_to_0_01(max_repeat_error + 0.01))
    clean_threshold = deadband + max(0.03, repeat_p95)

    side_samples: dict[str, list[dict[str, Any]]] = {
        side: [sample for sample in samples if sample["side"] == side]
        for side in ("A", "B")
    }
    side_contracts: list[dict[str, Any]] = []
    side_statistics: dict[str, Any] = {}
    for side in ("A", "B"):
        expected_sign = -1 if side == "A" else 1
        values = side_samples[side]
        coordinates = np.asarray(
            [
                -physical_left_sign
                * _shortest_angle(sample["qpos_window_mean"][0] - home_swing_axis)
                for sample in values
            ],
            dtype=np.float64,
        )
        margins = expected_sign * coordinates
        if np.any(margins <= deadband):
            failing = [
                values[index]["reference_id"]
                for index in np.flatnonzero(margins <= deadband)
            ]
            raise TransitionContractError(
                f"side {side} has ambiguous or wrong-sign references: {failing}"
            )
        if float(np.min(margins)) < clean_threshold:
            raise TransitionContractError(
                f"side {side} nearest demonstrated margin {float(np.min(margins)):.6f} "
                f"is below clean threshold {clean_threshold:.6f}"
            )
        qpos_means = np.stack([sample["qpos_window_mean"] for sample in values])
        qvel_rows = np.concatenate([sample["qvel_rows"] for sample in values], axis=0)
        side_contracts.append(
            {
                "side_id": side,
                "physical_role": "left_of_home" if side == "A" else "right_of_home",
                "condition_code": expected_sign,
                "demonstrated_side_coordinate_support_rad": [
                    float(np.min(coordinates)),
                    float(np.max(coordinates)),
                ],
                "demonstrated_qpos_support": _axis_ranges(qpos_means),
                "ready_qvel_support": _axis_ranges(qvel_rows),
                "ready_reference_ids": [sample["reference_id"] for sample in values],
                "visual_reference_ids": sorted(
                    {
                        reference
                        for sample in values
                        for reference in sample["visual_reference_ids"]
                    }
                ),
            }
        )
        side_statistics[side] = {
            "count": len(values),
            "side_coordinate_min_rad": float(np.min(coordinates)),
            "side_coordinate_max_rad": float(np.max(coordinates)),
            "nearest_clean_margin_rad": float(np.min(margins)),
            "nearest_reference_distance_method": "four_axis_scaled_distance_plus_visual_review",
        }

    observed_statistics = {
        "accepted_window_counts": dict(sorted(counts.items())),
        "home_swing_window_means_rad": home_swing_means.tolist(),
        "home_medoid_reference_id": home_samples[medoid_index]["reference_id"],
        "center_repeat_error_rad": center_errors.tolist(),
        "center_repeat_abs_max_rad": max_repeat_error,
        "center_repeat_abs_p95_rad": repeat_p95,
        "sides": side_statistics,
    }
    field_overrides = list(calibration.get("field_overrides", ()) or ())
    contract: dict[str, Any] = {
        "schema": HOME_SIDE_CONTRACT_SCHEMA,
        "context_version": context_version,
        "home_reference": {
            "source_config": str(source_config),
            "source_sha256": sha256_file(source_config),
            "source_value_path": source_value_path,
            "home_pose_rad": home_pose.tolist(),
            "home_swing_axis_rad": home_swing_axis,
            "physical_left_qpos_sign": physical_left_sign,
            "classification_deadband_rad": deadband,
            "clean_endpoint_min_abs_side_coordinate_rad": clean_threshold,
        },
        "sides": side_contracts,
        "ready_candidate": ready_candidate,
        "calibration_source_ids": ids,
        "parameter_resolution": {
            "baseline_id": str(calibration.get("baseline_id", BASELINE_ID)),
            "observed_statistics": observed_statistics,
            "field_overrides": field_overrides,
            "resolved_at": resolved_at,
            "resolved_by": resolved_by,
        },
        "contract_sha256_scope": "canonical_json_without_contract_sha256",
    }
    contract["contract_sha256"] = _contract_digest(contract)
    return contract


def write_home_side_contract(
    *,
    calibration_path: Path | str,
    output_path: Path | str,
) -> dict[str, Any]:
    calibration_path = Path(calibration_path).resolve()
    try:
        value = json.loads(calibration_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TransitionContractError(
            f"cannot read calibration input {calibration_path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise TransitionContractError("calibration input root must be an object")
    contract = build_home_side_contract(
        value,
        source_base_dir=calibration_path.parent,
    )
    encoded = (
        json.dumps(
            contract,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    output = write_immutable_text(output_path, encoded)
    return {
        "status": "PASS",
        "output": str(output),
        "sha256": sha256_file(output),
        "contract_sha256": contract["contract_sha256"],
        "home_swing_axis_rad": contract["home_reference"]["home_swing_axis_rad"],
        "classification_deadband_rad": contract["home_reference"][
            "classification_deadband_rad"
        ],
        "clean_endpoint_min_abs_side_coordinate_rad": contract["home_reference"][
            "clean_endpoint_min_abs_side_coordinate_rad"
        ],
    }


def validate_home_side_contract(contract: Mapping[str, Any]) -> None:
    if contract.get("schema") != HOME_SIDE_CONTRACT_SCHEMA:
        raise TransitionContractError("home-side contract schema mismatch")
    expected = _contract_digest(contract)
    if contract.get("contract_sha256") != expected:
        raise TransitionContractError("home-side contract digest mismatch")
    home = contract.get("home_reference", {})
    deadband = float(home.get("classification_deadband_rad", 0.0))
    clean = float(home.get("clean_endpoint_min_abs_side_coordinate_rad", 0.0))
    if deadband < 0.05 or clean < deadband + 0.03 - 1e-12:
        raise TransitionContractError("home-side thresholds are below the v2.0.1 floor")
    sides = {str(side.get("side_id")): side for side in contract.get("sides", ())}
    if set(sides) != {"A", "B"}:
        raise TransitionContractError("home-side contract must contain A and B")
    for side, expected_code in (("A", -1), ("B", 1)):
        value = sides[side]
        if int(value.get("condition_code", 0)) != expected_code:
            raise TransitionContractError(f"side {side} condition code is invalid")
        support = value.get("demonstrated_side_coordinate_support_rad", ())
        if len(support) != 2:
            raise TransitionContractError(f"side {side} support must have two bounds")
        nearest_margin = -float(support[1]) if side == "A" else float(support[0])
        if nearest_margin < clean:
            raise TransitionContractError(
                f"side {side} has no clean demonstrated support"
            )


def resolve_home_calibration_ready_candidate(
    calibration: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve the ready-window thresholds used by field sample capture."""

    raw = dict(READY_BASELINE)
    override = calibration.get("ready_candidate", {})
    if override:
        if not isinstance(override, Mapping):
            raise TransitionContractError("ready_candidate must be an object")
        raw.update(dict(override))
    resolved = {
        "dwell_s": float(raw["dwell_s"]),
        "commanded_action_abs_max": float(raw["commanded_action_abs_max"]),
        "qvel_abs_max": _positive_vector4(raw["qvel_abs_max"], "qvel_abs_max").tolist(),
        "qpos_peak_to_peak_max": _positive_vector4(
            raw["qpos_peak_to_peak_max"], "qpos_peak_to_peak_max"
        ).tolist(),
        "visual_confirmation_required": bool(raw["visual_confirmation_required"]),
    }
    if resolved["dwell_s"] <= 0 or resolved["commanded_action_abs_max"] < 0:
        raise TransitionContractError(
            "ready candidate dwell/action thresholds are invalid"
        )
    relaxed = []
    if resolved["dwell_s"] < float(READY_BASELINE["dwell_s"]):
        relaxed.append("dwell_s")
    if resolved["commanded_action_abs_max"] > float(
        READY_BASELINE["commanded_action_abs_max"]
    ):
        relaxed.append("commanded_action_abs_max")
    if np.any(
        np.asarray(resolved["qvel_abs_max"])
        > np.asarray(READY_BASELINE["qvel_abs_max"])
    ):
        relaxed.append("qvel_abs_max")
    if np.any(
        np.asarray(resolved["qpos_peak_to_peak_max"])
        > np.asarray(READY_BASELINE["qpos_peak_to_peak_max"])
    ):
        relaxed.append("qpos_peak_to_peak_max")
    if not resolved["visual_confirmation_required"]:
        relaxed.append("visual_confirmation_required")
    if relaxed:
        approvals = calibration.get("field_overrides", ()) or ()
        approved = {
            str(item.get("parameter"))
            for item in approvals
            if isinstance(item, Mapping)
            and str(item.get("reason", "")).strip()
            and str(item.get("approved_by", "")).strip()
            and str(item.get("evidence_artifact", "")).strip()
            and str(item.get("evidence_sha256", "")).strip()
        }
        missing = sorted(set(relaxed) - approved)
        if missing:
            raise TransitionContractError(
                "relaxed ready thresholds require approved evidence entries for: "
                + ", ".join(missing)
            )
    return resolved


def validate_home_calibration_sample(
    value: Any,
    *,
    ready_candidate: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one accepted field window and return its derived statistics."""

    if not isinstance(value, Mapping):
        raise TransitionContractError("each calibration sample must be an object")
    reference_id = _required_text(value, "reference_id")
    side = str(value.get("side", ""))
    if side not in {"home", "A", "B"}:
        raise TransitionContractError(
            f"sample {reference_id} side must be home, A, or B"
        )
    if not bool(value.get("accepted", False)):
        raise TransitionContractError(
            f"sample {reference_id} is not accepted; omit rejected windows from freeze input"
        )
    dwell = float(value.get("stable_duration_s", 0.0))
    if dwell < float(ready_candidate["dwell_s"]):
        raise TransitionContractError(
            f"sample {reference_id} stable duration {dwell:.3f}s is too short"
        )
    if bool(ready_candidate["visual_confirmation_required"]) and not bool(
        value.get("visual_confirmed", False)
    ):
        raise TransitionContractError(
            f"sample {reference_id} lacks required visual confirmation"
        )
    action_abs_max = float(value.get("commanded_action_abs_max", math.inf))
    if action_abs_max > float(ready_candidate["commanded_action_abs_max"]):
        raise TransitionContractError(
            f"sample {reference_id} action max {action_abs_max:.6f} exceeds ready threshold"
        )
    qpos = _matrix4(value.get("qpos_samples_rad"), "qpos_samples_rad", reference_id)
    qvel = _matrix4(value.get("qvel_samples_rad_s"), "qvel_samples_rad_s", reference_id)
    if qpos.shape != qvel.shape:
        raise TransitionContractError(
            f"sample {reference_id} qpos/qvel row counts differ"
        )
    qvel_abs_max = np.max(np.abs(qvel), axis=0)
    qvel_limit = np.asarray(ready_candidate["qvel_abs_max"], dtype=np.float64)
    if np.any(qvel_abs_max > qvel_limit):
        raise TransitionContractError(
            f"sample {reference_id} qvel exceeds ready threshold: "
            f"observed={qvel_abs_max.tolist()} limit={qvel_limit.tolist()}"
        )
    qpos_peak_to_peak = np.ptp(qpos, axis=0)
    qpos_limit = np.asarray(ready_candidate["qpos_peak_to_peak_max"], dtype=np.float64)
    if np.any(qpos_peak_to_peak > qpos_limit):
        raise TransitionContractError(
            f"sample {reference_id} qpos peak-to-peak exceeds ready threshold: "
            f"observed={qpos_peak_to_peak.tolist()} limit={qpos_limit.tolist()}"
        )
    qpos_mean = np.mean(qpos, axis=0)
    qpos_mean[0] = _circular_mean(qpos[:, 0])
    visual_ids = [str(item) for item in value.get("visual_reference_ids", ())]
    if bool(ready_candidate["visual_confirmation_required"]) and not visual_ids:
        raise TransitionContractError(
            f"sample {reference_id} has no visual_reference_ids"
        )
    return {
        "reference_id": reference_id,
        "side": side,
        "qpos_window_mean": qpos_mean,
        "qvel_rows": qvel,
        "visual_reference_ids": visual_ids,
    }


def _circular_mean(values: np.ndarray) -> float:
    return float(
        math.atan2(float(np.mean(np.sin(values))), float(np.mean(np.cos(values))))
    )


def _circular_medoid_index(values: np.ndarray) -> int:
    distances = [
        sum(abs(_shortest_angle(float(candidate - other))) for other in values)
        for candidate in values
    ]
    return int(np.argmin(np.asarray(distances, dtype=np.float64)))


def _shortest_angle(value: float) -> float:
    return float(math.atan2(math.sin(float(value)), math.cos(float(value))))


def _ceil_to_0_01(value: float) -> float:
    return math.ceil((float(value) - 1e-12) * 100.0) / 100.0


def _axis_ranges(values: np.ndarray) -> list[list[float]]:
    return [
        [float(np.min(values[:, axis])), float(np.max(values[:, axis]))]
        for axis in range(4)
    ]


def _vector4(value: Any, name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise TransitionContractError(f"{name} must be numeric") from exc
    if array.shape != (4,) or not np.all(np.isfinite(array)):
        raise TransitionContractError(f"{name} must contain four finite values")
    return array


def _positive_vector4(value: Any, name: str) -> np.ndarray:
    array = _vector4(value, name)
    if np.any(array <= 0):
        raise TransitionContractError(f"{name} values must be positive")
    return array


def _matrix4(value: Any, name: str, reference_id: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TransitionContractError(
            f"sample {reference_id} {name} must be numeric"
        ) from exc
    if array.ndim != 2 or array.shape[1] != 4 or array.shape[0] < 2:
        raise TransitionContractError(
            f"sample {reference_id} {name} must have shape (N,4), N>=2"
        )
    if not np.all(np.isfinite(array)):
        raise TransitionContractError(f"sample {reference_id} {name} is not finite")
    return array


def _required_text(value: Mapping[str, Any], field: str) -> str:
    text = str(value.get(field, "")).strip()
    if not text:
        raise TransitionContractError(f"{field} is required")
    return text


def _resolve_mapping_path(value: Any, path: str) -> Any:
    current = value
    for token in path.split("."):
        if not isinstance(current, Mapping) or token not in current:
            raise TransitionContractError(
                f"source config does not contain home value path {path!r}"
            )
        current = current[token]
    return current


def _contract_digest(contract: Mapping[str, Any]) -> str:
    payload = dict(contract)
    payload.pop("contract_sha256", None)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
