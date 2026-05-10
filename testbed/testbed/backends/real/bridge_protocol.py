"""JSON-line protocol helpers for external real-machine bridge clients.

This protocol is a development-friendly control/state boundary. It is not the
final low-latency video transport; camera frames are encoded here only so the
socket client can be tested end to end without ROS, CAN, or hardware.
"""

from __future__ import annotations

import base64
import json
from typing import Any, Mapping

import numpy as np

from testbed.backends.real.control import ControlResult
from testbed.backends.real.state import RealStateSamples
from testbed.backends.real.sync import TimestampedSample


BRIDGE_PROTOCOL_VERSION = 1


class BridgeProtocolError(RuntimeError):
    """Raised when a bridge message is malformed or unsupported."""


def encode_frame(message: Mapping[str, Any]) -> bytes:
    """Encode one JSON message as a newline-delimited UTF-8 frame."""

    payload = {
        "version": int(message.get("version", BRIDGE_PROTOCOL_VERSION)),
        **{str(k): _jsonable(v) for k, v in message.items() if k != "version"},
    }
    return json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8") + b"\n"


def decode_frame(frame: bytes | str) -> dict[str, Any]:
    """Decode and validate one newline-delimited JSON frame."""

    text = frame.decode("utf-8") if isinstance(frame, bytes) else str(frame)
    try:
        message = json.loads(text.strip())
    except json.JSONDecodeError as exc:
        raise BridgeProtocolError(f"invalid bridge JSON frame: {exc}") from exc
    if not isinstance(message, dict):
        raise BridgeProtocolError("bridge frame must decode to a JSON object")
    version = int(message.get("version", -1))
    if version != BRIDGE_PROTOCOL_VERSION:
        raise BridgeProtocolError(
            f"unsupported bridge protocol version {version}; expected {BRIDGE_PROTOCOL_VERSION}"
        )
    if "type" not in message:
        raise BridgeProtocolError("bridge frame missing message type")
    return message


def request_message(message_type: str, payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {
        "version": BRIDGE_PROTOCOL_VERSION,
        "type": str(message_type),
        "payload": dict(payload or {}),
    }


def response_message(
    message_type: str,
    payload: Mapping[str, Any] | None = None,
    *,
    ok: bool = True,
    error: str = "",
) -> dict[str, Any]:
    return {
        "version": BRIDGE_PROTOCOL_VERSION,
        "type": str(message_type),
        "ok": bool(ok),
        "error": str(error),
        "payload": dict(payload or {}),
    }


def control_result_to_payload(result: ControlResult) -> dict[str, Any]:
    return {
        "ack": bool(result.ack),
        "fault_code": str(result.fault_code),
        "controller_timestamp_ns": int(result.controller_timestamp_ns),
        "commanded_action": _array_to_list(result.commanded_action, dtype=np.float32),
        "raw_low_level_command": (
            None
            if result.raw_low_level_command is None
            else _array_to_list(result.raw_low_level_command, dtype=np.float32)
        ),
    }


def control_result_from_payload(payload: Mapping[str, Any]) -> ControlResult:
    return ControlResult(
        ack=bool(payload.get("ack", False)),
        fault_code=str(payload.get("fault_code", "")),
        controller_timestamp_ns=int(payload.get("controller_timestamp_ns", 0)),
        commanded_action=np.asarray(payload.get("commanded_action", []), dtype=np.float32),
        raw_low_level_command=(
            None
            if payload.get("raw_low_level_command") is None
            else np.asarray(payload.get("raw_low_level_command"), dtype=np.float32)
        ),
    )


def state_samples_to_payload(samples: RealStateSamples) -> dict[str, Any]:
    return {
        "joint": _sample_to_payload(samples.joint, image=False),
        "images": {
            str(name): _sample_to_payload(sample, image=True)
            for name, sample in dict(samples.images).items()
        },
    }


def state_samples_from_payload(payload: Mapping[str, Any]) -> RealStateSamples:
    if "joint" not in payload:
        raise BridgeProtocolError("state payload missing joint sample")
    images_raw = payload.get("images", {})
    if not isinstance(images_raw, Mapping):
        raise BridgeProtocolError("state payload images must be a mapping")
    return RealStateSamples(
        joint=_sample_from_payload(payload["joint"], image=False),
        images={
            str(name): _sample_from_payload(sample_payload, image=True)
            for name, sample_payload in images_raw.items()
        },
    )


def _sample_to_payload(sample: TimestampedSample, *, image: bool) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "timestamp_ns": int(sample.timestamp_ns),
        "source": str(sample.source),
        "receive_time_ns": sample.receive_time_ns,
    }
    if image:
        payload["payload"] = _image_to_payload(sample.payload)
    else:
        payload["payload"] = _mapping_to_payload(sample.payload)
    return payload


def _sample_from_payload(payload: Mapping[str, Any], *, image: bool) -> TimestampedSample:
    if not isinstance(payload, Mapping):
        raise BridgeProtocolError("sample payload must be a mapping")
    sample_payload = payload.get("payload", {})
    return TimestampedSample(
        timestamp_ns=int(payload.get("timestamp_ns", 0)),
        payload=(
            _image_from_payload(sample_payload)
            if image
            else _mapping_from_payload(sample_payload)
        ),
        source=str(payload.get("source", "")),
        receive_time_ns=(
            None
            if payload.get("receive_time_ns") is None
            else int(payload.get("receive_time_ns"))
        ),
    )


def _image_to_payload(value: Any) -> dict[str, Any]:
    arr = np.asarray(value, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[-1] not in (1, 3, 4):
        raise BridgeProtocolError(f"image payload must have shape HxWxC, got {arr.shape}")
    return {
        "encoding": "raw_uint8",
        "shape": [int(v) for v in arr.shape],
        "data_b64": base64.b64encode(arr.tobytes()).decode("ascii"),
    }


def _image_from_payload(payload: Any) -> np.ndarray:
    if not isinstance(payload, Mapping):
        raise BridgeProtocolError("image payload must be a mapping")
    if payload.get("encoding") != "raw_uint8":
        raise BridgeProtocolError("only raw_uint8 image payloads are supported")
    shape = tuple(int(v) for v in payload.get("shape", ()))
    data = base64.b64decode(str(payload.get("data_b64", "")).encode("ascii"))
    arr = np.frombuffer(data, dtype=np.uint8)
    try:
        return arr.reshape(shape).copy()
    except ValueError as exc:
        raise BridgeProtocolError(f"image payload shape does not match data length: {shape}") from exc


def _mapping_to_payload(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise BridgeProtocolError("joint sample payload must be a mapping")
    return {str(k): _jsonable(v) for k, v in value.items()}


def _mapping_from_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise BridgeProtocolError("joint sample payload must be a mapping")
    result: dict[str, Any] = {}
    for key, value in payload.items():
        if key in {"qpos", "qvel", "qacc", "motor_rpm", "plan_rpm", "env_state"}:
            result[str(key)] = np.asarray(value, dtype=np.float32)
        elif key == "status":
            result[str(key)] = np.asarray(value, dtype=np.int32)
        else:
            result[str(key)] = value
    return result


def _array_to_list(value: Any, *, dtype: Any) -> list[Any]:
    return np.asarray(value, dtype=dtype).reshape(-1).tolist()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value
