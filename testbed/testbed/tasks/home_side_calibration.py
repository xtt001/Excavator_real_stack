"""Read-only field capture for the v2.0.1 home/A/B calibration input."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from testbed.backends.real.bridge_socket import JsonTcpBridgeClient
from testbed.backends.real.state import RealStateSamples
from testbed.tasks.home_side_contract import (
    BASELINE_ID,
    CALIBRATION_INPUT_SCHEMA,
    READY_BASELINE,
    resolve_home_calibration_ready_candidate,
    validate_home_calibration_sample,
)
from testbed.tasks.real_transition import (
    TransitionContractError,
    sha256_file,
    write_immutable_text,
)

DEFAULT_CALIBRATION_CAMERAS = ("video4", "video5", "video6", "video7")
CALIBRATION_IMAGE_MEAN_MIN = 5.0
_REFERENCE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def initialise_home_calibration(
    *,
    output_path: Path | str,
    context_version: str,
    resolved_by: str,
    physical_left_qpos_sign: int,
    source_config: Path | str,
    source_value_path: str,
    expected_cameras: Sequence[str] = DEFAULT_CALIBRATION_CAMERAS,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    """Create a portable, append-only field calibration input and config snapshot."""

    output = Path(output_path).expanduser().resolve()
    if output.exists():
        raise TransitionContractError(
            f"calibration output already exists; refusing to overwrite: {output}"
        )
    context = _required_text(context_version, "context_version")
    operator = _required_text(resolved_by, "resolved_by")
    sign = int(physical_left_qpos_sign)
    if sign not in {-1, 1}:
        raise TransitionContractError("physical_left_qpos_sign must be -1 or +1")
    value_path = _required_text(source_value_path, "source_value_path")
    cameras = _normalise_cameras(expected_cameras)

    source = Path(source_config).expanduser().resolve()
    if not source.is_file():
        raise TransitionContractError(f"home source config does not exist: {source}")
    try:
        source_text = source.read_text(encoding="utf-8")
        source_payload = yaml.safe_load(source_text)
    except (OSError, yaml.YAMLError) as exc:
        raise TransitionContractError(
            f"cannot read home source config {source}: {exc}"
        ) from exc
    home_pose = _vector4(
        _resolve_mapping_path(source_payload, value_path),
        f"source_config:{value_path}",
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    snapshot = output.parent / f"{output.stem}.home_reference_source.yaml"
    write_immutable_text(snapshot, source_text)
    timestamp = created_at_utc or _utc_now()
    calibration = {
        "schema": CALIBRATION_INPUT_SCHEMA,
        "context_version": context,
        "physical_left_qpos_sign": sign,
        "resolved_by": operator,
        "resolved_at": timestamp,
        "baseline_id": BASELINE_ID,
        "home_reference": {
            "source_config": snapshot.name,
            "source_value_path": value_path,
            "home_pose_rad": home_pose.tolist(),
            "source_original_path": str(source),
            "source_original_sha256": sha256_file(source),
        },
        "ready_candidate": _jsonable(dict(READY_BASELINE)),
        "collection": {
            "mode": "read_only_gateway_receiver_absent",
            "started_at": timestamp,
            "expected_cameras": cameras,
            "software_command_abs_max_evidence": (
                "receiver_port_closed_and_operator_confirmation"
            ),
        },
        "field_overrides": [],
        "samples": [],
    }
    write_immutable_text(output, _pretty_json(calibration))
    return {
        "status": "PASS",
        "calibration": str(output),
        "calibration_sha256": sha256_file(output),
        "home_source_snapshot": str(snapshot),
        "home_source_snapshot_sha256": sha256_file(snapshot),
        "home_pose_rad": home_pose.tolist(),
        "expected_cameras": cameras,
        "accepted_window_counts": {"home": 0, "A": 0, "B": 0},
    }


def capture_home_calibration_window(
    *,
    calibration_path: Path | str,
    side: str,
    reference_id: str,
    confirm_visual: bool,
    confirm_no_software_action_source: bool,
    host: str = "127.0.0.1",
    port: int = 8765,
    receiver_port: int = 8770,
    duration_s: float = 0.5,
    rate_hz: float = 20.0,
    timeout_s: float = 2.0,
    jpeg_quality: int = 90,
    client_factory: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    """Capture and append one accepted stable window without sending actions."""

    calibration_file = Path(calibration_path).expanduser().resolve()
    calibration_bytes = _read_bytes(calibration_file)
    calibration = _decode_json_object(calibration_bytes, calibration_file)
    if calibration.get("schema") != CALIBRATION_INPUT_SCHEMA:
        raise TransitionContractError(
            f"calibration schema must be {CALIBRATION_INPUT_SCHEMA!r}"
        )
    if side not in {"home", "A", "B"}:
        raise TransitionContractError("side must be home, A, or B")
    ref_id = _validate_reference_id(reference_id)
    samples = calibration.get("samples", [])
    if not isinstance(samples, list):
        raise TransitionContractError("calibration samples must be a list")
    if any(
        isinstance(item, Mapping) and str(item.get("reference_id", "")) == ref_id
        for item in samples
    ):
        raise TransitionContractError(f"reference_id already exists: {ref_id}")
    if not confirm_visual:
        raise TransitionContractError(
            "visual confirmation is required before accepting a calibration window"
        )
    if not confirm_no_software_action_source:
        raise TransitionContractError(
            "operator must confirm that no software action source is active"
        )
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise TransitionContractError(
            "calibration capture must run on the slave and use a loopback gateway"
        )
    if not 1 <= int(port) <= 65535 or not 1 <= int(receiver_port) <= 65535:
        raise TransitionContractError("gateway and receiver ports must be in [1, 65535]")
    if _tcp_port_is_listening(int(receiver_port)):
        raise TransitionContractError(
            f"receiver port {int(receiver_port)} is listening; stop the receiver "
            "before read-only calibration"
        )
    duration = float(duration_s)
    rate = float(rate_hz)
    timeout = float(timeout_s)
    if not all(math.isfinite(value) for value in (duration, rate, timeout)):
        raise TransitionContractError("duration_s, rate_hz, and timeout_s must be finite")
    if duration < float(READY_BASELINE["dwell_s"]):
        raise TransitionContractError(
            f"duration_s must be at least {READY_BASELINE['dwell_s']:.1f}s"
        )
    if rate <= 0 or timeout <= 0:
        raise TransitionContractError("rate_hz and timeout_s must be positive")
    if not 1 <= int(jpeg_quality) <= 100:
        raise TransitionContractError("jpeg_quality must be in [1, 100]")

    collection = calibration.get("collection", {})
    if not isinstance(collection, Mapping):
        raise TransitionContractError("calibration collection must be an object")
    expected_cameras = _normalise_cameras(
        collection.get("expected_cameras", DEFAULT_CALIBRATION_CAMERAS)
    )
    ready_candidate = resolve_home_calibration_ready_candidate(calibration)

    if client_factory is None:
        client_factory = lambda: JsonTcpBridgeClient(
            host=host,
            port=int(port),
            timeout_s=timeout,
            connect_on_init=True,
        )
    try:
        captured = _capture_window(
            client_factory=client_factory,
            duration_s=duration,
            rate_hz=rate,
            expected_cameras=expected_cameras,
        )
    except TransitionContractError:
        raise
    except Exception as exc:
        raise TransitionContractError(f"gateway capture failed: {exc}") from exc

    visual_rel_paths = [
        str(Path("calibration_visuals") / ref_id / f"{camera}.jpg")
        for camera in expected_cameras
    ]
    sample = {
        "reference_id": ref_id,
        "side": side,
        "accepted": True,
        "stable_duration_s": captured["stable_duration_s"],
        "visual_confirmed": True,
        "visual_reference_ids": visual_rel_paths,
        "commanded_action_abs_max": 0.0,
        "qpos_samples_rad": captured["qpos_samples_rad"],
        "qvel_samples_rad_s": captured["qvel_samples_rad_s"],
        "capture_provenance": {
            "captured_at": _utc_now(),
            "gateway_host": host,
            "gateway_port": int(port),
            "receiver_port_checked_closed": int(receiver_port),
            "software_action_source": "operator_confirmed_inactive",
            "sample_rate_hz_requested": rate,
            "joint_timestamps_ns": captured["joint_timestamps_ns"],
            "image_timestamps_ns": captured["image_timestamps_ns"],
            "image_means": captured["latest_image_means"],
            "status_samples": captured["status_samples"],
        },
    }
    derived = validate_home_calibration_sample(
        sample,
        ready_candidate=ready_candidate,
    )

    final_visual_dir = calibration_file.parent / "calibration_visuals" / ref_id
    if final_visual_dir.exists():
        raise TransitionContractError(
            f"visual reference directory already exists: {final_visual_dir}"
        )
    temp_visual_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{ref_id}.",
            dir=str(final_visual_dir.parent.parent),
        )
    )
    json_temp: Path | None = None
    moved_visual_dir = False
    try:
        for camera in expected_cameras:
            _write_rgb_jpeg(
                temp_visual_dir / f"{camera}.jpg",
                captured["latest_images"][camera],
                quality=int(jpeg_quality),
            )
        if hashlib.sha256(_read_bytes(calibration_file)).digest() != hashlib.sha256(
            calibration_bytes
        ).digest():
            raise TransitionContractError(
                "calibration changed during capture; refusing a concurrent append"
            )
        updated = dict(calibration)
        updated_samples = list(samples)
        updated_samples.append(sample)
        updated["samples"] = updated_samples
        updated["resolved_at"] = _utc_now()
        encoded = _pretty_json(updated).encode("utf-8")
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{calibration_file.name}.",
            dir=str(calibration_file.parent),
            delete=False,
        ) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
            json_temp = Path(handle.name)
        final_visual_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temp_visual_dir, final_visual_dir)
        moved_visual_dir = True
        os.replace(json_temp, calibration_file)
        json_temp = None
    except Exception:
        if moved_visual_dir and final_visual_dir.is_dir():
            shutil.rmtree(final_visual_dir)
        raise
    finally:
        if temp_visual_dir.is_dir():
            shutil.rmtree(temp_visual_dir)
        if json_temp is not None and json_temp.exists():
            json_temp.unlink()

    counts = Counter(str(item["side"]) for item in updated_samples)
    qpos = np.asarray(sample["qpos_samples_rad"], dtype=np.float64)
    qvel = np.asarray(sample["qvel_samples_rad_s"], dtype=np.float64)
    return {
        "status": "PASS",
        "calibration": str(calibration_file),
        "calibration_sha256": sha256_file(calibration_file),
        "reference_id": ref_id,
        "side": side,
        "accepted_window_counts": {
            name: int(counts.get(name, 0)) for name in ("home", "A", "B")
        },
        "stable_duration_s": sample["stable_duration_s"],
        "qpos_window_mean_rad": _jsonable(derived["qpos_window_mean"]),
        "qpos_peak_to_peak_rad": np.ptp(qpos, axis=0).tolist(),
        "qvel_abs_max_rad_s": np.max(np.abs(qvel), axis=0).tolist(),
        "image_means": captured["latest_image_means"],
        "visual_reference_ids": visual_rel_paths,
    }


def _capture_window(
    *,
    client_factory: Callable[[], Any],
    duration_s: float,
    rate_hz: float,
    expected_cameras: Sequence[str],
) -> dict[str, Any]:
    client = client_factory()
    qpos_rows: list[list[float]] = []
    qvel_rows: list[list[float]] = []
    status_rows: list[Any] = []
    joint_timestamps: list[int] = []
    image_timestamps: dict[str, list[int]] = {
        camera: [] for camera in expected_cameras
    }
    latest_images: dict[str, np.ndarray] = {}
    latest_image_means: dict[str, float] = {}
    period_s = 1.0 / float(rate_hz)
    first_sample_time: float | None = None
    next_sample_time: float | None = None
    step_id = 0
    stable_duration_s = 0.0
    try:
        while True:
            now = time.monotonic()
            if next_sample_time is not None and now < next_sample_time:
                time.sleep(next_sample_time - now)
            state: RealStateSamples = client.read_state(step_id=step_id)
            sampled_at = time.monotonic()
            if first_sample_time is None:
                first_sample_time = sampled_at
                next_sample_time = sampled_at
            payload = dict(state.joint.payload)
            qpos = _state_vector4(payload.get("qpos"), "qpos")
            qvel = _state_vector4(payload.get("qvel"), "qvel")
            missing = [
                camera for camera in expected_cameras if camera not in state.images
            ]
            if missing:
                raise TransitionContractError(
                    "gateway state is missing calibration cameras: "
                    + ", ".join(missing)
                )
            for camera in expected_cameras:
                image_sample = state.images[camera]
                image = np.asarray(image_sample.payload, dtype=np.uint8)
                if image.ndim != 3 or image.shape[-1] not in {1, 3, 4}:
                    raise TransitionContractError(
                        f"camera {camera} has invalid image shape {image.shape}"
                    )
                image_mean = float(np.mean(image))
                if image_mean <= CALIBRATION_IMAGE_MEAN_MIN:
                    raise TransitionContractError(
                        f"camera {camera} mean {image_mean:.3f} is at or below "
                        f"the black-frame threshold {CALIBRATION_IMAGE_MEAN_MIN:.1f}"
                    )
                latest_images[camera] = np.ascontiguousarray(image.copy())
                latest_image_means[camera] = image_mean
                image_timestamps[camera].append(int(image_sample.timestamp_ns))
            qpos_rows.append(qpos.tolist())
            qvel_rows.append(qvel.tolist())
            status_rows.append(_jsonable(payload.get("status")))
            joint_timestamps.append(int(state.joint.timestamp_ns))
            elapsed = sampled_at - first_sample_time
            if elapsed >= float(duration_s) and len(qpos_rows) >= 2:
                stable_duration_s = elapsed
                break
            step_id += 1
            assert next_sample_time is not None
            next_sample_time += period_s
    finally:
        force_close = getattr(client, "force_close", None)
        try:
            if callable(force_close):
                force_close()
            else:
                close = getattr(client, "close", None)
                if callable(close):
                    close()
        except Exception:
            pass

    assert first_sample_time is not None
    frozen = [
        camera
        for camera, timestamps in image_timestamps.items()
        if len(set(timestamps)) < 2
    ]
    if frozen:
        raise TransitionContractError(
            "camera timestamps did not advance during the stable window: "
            + ", ".join(frozen)
        )
    return {
        "stable_duration_s": float(stable_duration_s),
        "qpos_samples_rad": qpos_rows,
        "qvel_samples_rad_s": qvel_rows,
        "status_samples": status_rows,
        "joint_timestamps_ns": joint_timestamps,
        "image_timestamps_ns": image_timestamps,
        "latest_images": latest_images,
        "latest_image_means": latest_image_means,
    }


def _write_rgb_jpeg(path: Path, image: np.ndarray, *, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image

        pil_image = image[..., 0] if image.shape[-1] == 1 else image
        Image.fromarray(pil_image).save(path, format="JPEG", quality=int(quality))
        return
    except ImportError:
        pass
    import cv2

    if image.shape[-1] == 3:
        encoded_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif image.shape[-1] == 4:
        encoded_image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    else:
        encoded_image = image
    ok, buffer = cv2.imencode(
        ".jpg",
        encoded_image,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
    )
    if not ok:
        raise TransitionContractError(f"failed to encode calibration image {path.name}")
    path.write_bytes(buffer.tobytes())


def _tcp_port_is_listening(port: int) -> bool:
    target = int(port)
    inspected = False
    for proc_path in (Path("/proc/net/tcp"), Path("/proc/net/tcp6")):
        try:
            lines = proc_path.read_text(encoding="ascii").splitlines()[1:]
        except OSError:
            continue
        inspected = True
        for line in lines:
            fields = line.split()
            if len(fields) < 4 or fields[3] != "0A":
                continue
            try:
                local_port = int(fields[1].rsplit(":", 1)[1], 16)
            except (IndexError, ValueError):
                continue
            if local_port == target:
                return True
    if not inspected:
        raise TransitionContractError(
            "cannot inspect local TCP listeners; refusing read-only calibration"
        )
    return False


def _normalise_cameras(value: Sequence[str] | Any) -> list[str]:
    if isinstance(value, str):
        cameras = [item.strip() for item in value.split(",") if item.strip()]
    elif isinstance(value, Sequence):
        cameras = [str(item).strip() for item in value if str(item).strip()]
    else:
        cameras = []
    if not cameras or len(cameras) != len(set(cameras)):
        raise TransitionContractError(
            "expected_cameras must contain unique, non-empty names"
        )
    invalid = [camera for camera in cameras if not _REFERENCE_ID_RE.fullmatch(camera)]
    if invalid:
        raise TransitionContractError(
            "expected_cameras contains unsafe names: " + ", ".join(invalid)
        )
    return cameras


def _validate_reference_id(value: str) -> str:
    reference_id = str(value).strip()
    if not _REFERENCE_ID_RE.fullmatch(reference_id):
        raise TransitionContractError(
            "reference_id must use 1-64 letters, digits, dot, underscore, or hyphen"
        )
    return reference_id


def _required_text(value: str, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise TransitionContractError(f"{name} is required")
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


def _vector4(value: Any, name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise TransitionContractError(f"{name} must be numeric") from exc
    if array.shape != (4,) or not np.all(np.isfinite(array)):
        raise TransitionContractError(f"{name} must contain four finite values")
    return array


def _state_vector4(value: Any, name: str) -> np.ndarray:
    return _vector4(value, f"gateway state {name}")


def _read_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise TransitionContractError(f"cannot read calibration {path}: {exc}") from exc


def _decode_json_object(payload: bytes, path: Path) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TransitionContractError(f"invalid calibration JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TransitionContractError("calibration input root must be an object")
    return value


def _pretty_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        raise TransitionContractError("calibration payload contains non-finite values")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)
