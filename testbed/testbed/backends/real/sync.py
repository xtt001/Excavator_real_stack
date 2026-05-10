"""Timestamp helpers for real-machine observation alignment.

The ROS/CAN bridge is expected to provide timestamped joint-state and camera
samples. This module stays pure Python so data alignment can be tested on a
developer machine before hardware-facing code exists.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from testbed.backends.real.contracts import observation_from_real_vectors


DEFAULT_SYNC_SLOP_NS = 40_000_000


@dataclass(frozen=True)
class TimestampedSample:
    """One sensor payload with a monotonic or system-clock nanosecond stamp."""

    timestamp_ns: int
    payload: Any
    source: str = ""
    receive_time_ns: int | None = None

    def __post_init__(self) -> None:
        if int(self.timestamp_ns) <= 0:
            raise ValueError("timestamp_ns must be a positive integer")
        if self.receive_time_ns is not None and int(self.receive_time_ns) <= 0:
            raise ValueError("receive_time_ns must be positive when provided")


@dataclass(frozen=True)
class SyncResult:
    """Output from building a synchronized real-excavator observation."""

    observation: dict[str, Any]
    joint_sample: TimestampedSample
    image_samples: Mapping[str, TimestampedSample]
    sync_timestamp_ns: int
    max_skew_ns: int
    warnings: tuple[str, ...] = field(default_factory=tuple)


class TimestampedBuffer:
    """Small in-memory buffer for sensor samples keyed by timestamp."""

    def __init__(self, *, maxlen: int = 256, max_age_ns: int | None = None) -> None:
        if maxlen <= 0:
            raise ValueError("maxlen must be positive")
        self._samples: deque[TimestampedSample] = deque(maxlen=int(maxlen))
        self._max_age_ns = None if max_age_ns is None else int(max_age_ns)

    def add(
        self,
        sample: TimestampedSample | Any,
        *,
        timestamp_ns: int | None = None,
        source: str = "",
        receive_time_ns: int | None = None,
    ) -> TimestampedSample:
        if isinstance(sample, TimestampedSample):
            item = sample
        else:
            item = TimestampedSample(
                timestamp_ns=int(timestamp_ns or timestamp_ns_now()),
                payload=sample,
                source=str(source),
                receive_time_ns=receive_time_ns,
            )
        self._samples.append(item)
        if len(self._samples) > 1 and self._samples[-2].timestamp_ns > item.timestamp_ns:
            self._samples = deque(
                sorted(self._samples, key=lambda s: s.timestamp_ns),
                maxlen=self._samples.maxlen,
            )
        self._drop_expired(now_ns=item.timestamp_ns)
        return item

    def latest(self) -> TimestampedSample | None:
        return self._samples[-1] if self._samples else None

    def latest_before(self, timestamp_ns: int) -> TimestampedSample | None:
        target = int(timestamp_ns)
        best: TimestampedSample | None = None
        for sample in self._samples:
            if sample.timestamp_ns <= target:
                best = sample
            else:
                break
        return best

    def nearest(
        self,
        timestamp_ns: int,
        *,
        max_slop_ns: int | None = None,
        prefer_past: bool = True,
    ) -> TimestampedSample | None:
        if not self._samples:
            return None
        target = int(timestamp_ns)
        best = min(
            self._samples,
            key=lambda sample: (
                abs(sample.timestamp_ns - target),
                0 if (sample.timestamp_ns <= target) == bool(prefer_past) else 1,
            ),
        )
        if max_slop_ns is not None and abs(best.timestamp_ns - target) > int(max_slop_ns):
            return None
        return best

    def __len__(self) -> int:
        return len(self._samples)

    def _drop_expired(self, *, now_ns: int) -> None:
        if self._max_age_ns is None:
            return
        cutoff = int(now_ns) - self._max_age_ns
        while self._samples and self._samples[0].timestamp_ns < cutoff:
            self._samples.popleft()


class SynchronizedObservationBuilder:
    """Build observations from timestamped joint and camera samples."""

    def __init__(
        self,
        *,
        max_slop_ns: int = DEFAULT_SYNC_SLOP_NS,
        prefer_past: bool = True,
    ) -> None:
        if max_slop_ns <= 0:
            raise ValueError("max_slop_ns must be positive")
        self.max_slop_ns = int(max_slop_ns)
        self.prefer_past = bool(prefer_past)

    def build(
        self,
        *,
        joint_sample: TimestampedSample,
        image_samples: Mapping[str, TimestampedSample],
        step_id: int = 0,
        action_timestamp_ns: int | None = None,
    ) -> SyncResult:
        if not image_samples:
            raise ValueError("at least one image sample is required")

        joint_payload = _payload_mapping(joint_sample.payload)
        image_payloads = {
            str(name): _image_payload(sample.payload)
            for name, sample in image_samples.items()
        }
        image_timestamps = {
            str(name): int(sample.timestamp_ns)
            for name, sample in image_samples.items()
        }
        sync_timestamp_ns = int(joint_sample.timestamp_ns)
        skews = [abs(ts_ns - sync_timestamp_ns) for ts_ns in image_timestamps.values()]
        if action_timestamp_ns is not None:
            skews.append(abs(int(action_timestamp_ns) - sync_timestamp_ns))
        max_skew_ns = int(max(skews) if skews else 0)
        warnings_list = [
            f"{name}_skew_exceeds_slop"
            for name, ts_ns in image_timestamps.items()
            if abs(ts_ns - sync_timestamp_ns) > self.max_slop_ns
        ]
        if (
            action_timestamp_ns is not None
            and abs(int(action_timestamp_ns) - sync_timestamp_ns) > self.max_slop_ns
        ):
            warnings_list.append("action_skew_exceeds_slop")
        warnings = tuple(warnings_list)

        observation = observation_from_real_vectors(
            qpos=_required_payload_value(joint_payload, "qpos"),
            qvel=_required_payload_value(joint_payload, "qvel"),
            step_id=int(step_id),
            sensor_timestamp_ns=sync_timestamp_ns,
            joint_timestamp_ns=int(joint_sample.timestamp_ns),
            image_timestamp_ns=image_timestamps,
            sync_timestamp_ns=sync_timestamp_ns,
            sync_max_skew_ns=max_skew_ns,
            sync_warnings=warnings,
            images=image_payloads,
            env_state=joint_payload.get("env_state"),
            status=joint_payload.get("status"),
            motor_rpm=joint_payload.get("motor_rpm"),
            plan_rpm=joint_payload.get("plan_rpm"),
        )
        if action_timestamp_ns is not None:
            observation["action_timestamp_ns"] = int(action_timestamp_ns)
        return SyncResult(
            observation=observation,
            joint_sample=joint_sample,
            image_samples=dict(image_samples),
            sync_timestamp_ns=sync_timestamp_ns,
            max_skew_ns=max_skew_ns,
            warnings=warnings,
        )

    def from_buffers(
        self,
        *,
        joint_buffer: TimestampedBuffer,
        image_buffers: Mapping[str, TimestampedBuffer],
        target_timestamp_ns: int | None = None,
        step_id: int = 0,
        action_timestamp_ns: int | None = None,
    ) -> SyncResult:
        target = int(target_timestamp_ns or action_timestamp_ns or timestamp_ns_now())
        joint_sample = joint_buffer.nearest(
            target,
            max_slop_ns=self.max_slop_ns,
            prefer_past=self.prefer_past,
        )
        if joint_sample is None:
            raise RuntimeError("no joint sample is close enough to the target timestamp")

        image_samples: dict[str, TimestampedSample] = {}
        for name, buffer in image_buffers.items():
            sample = buffer.nearest(
                joint_sample.timestamp_ns,
                max_slop_ns=self.max_slop_ns,
                prefer_past=self.prefer_past,
            )
            if sample is None:
                raise RuntimeError(
                    f"no image sample for camera {name!r} is close enough to the joint timestamp"
                )
            image_samples[str(name)] = sample

        return self.build(
            joint_sample=joint_sample,
            image_samples=image_samples,
            step_id=step_id,
            action_timestamp_ns=action_timestamp_ns,
        )


def timestamp_ns_now() -> int:
    return int(time.time_ns())


def _payload_mapping(payload: Any) -> Mapping[str, Any]:
    if isinstance(payload, Mapping):
        return payload
    raise TypeError("joint payload must be a mapping with at least qpos and qvel")


def _required_payload_value(payload: Mapping[str, Any], key: str) -> Any:
    if key not in payload:
        raise KeyError(f"joint payload missing required key {key!r}")
    return payload[key]


def _image_payload(payload: Any) -> np.ndarray:
    if isinstance(payload, Mapping):
        if "image" in payload:
            return np.asarray(payload["image"], dtype=np.uint8)
        if "frame" in payload:
            return np.asarray(payload["frame"], dtype=np.uint8)
    return np.asarray(payload, dtype=np.uint8)
