"""Shared camera-name selection helpers for real HDF5/QC code paths."""

from __future__ import annotations

from typing import Any

from testbed.data.schema import ATTR_CAMERA_NAMES


def camera_names_from_metadata(metadata: dict[str, Any] | None) -> list[str]:
    raw = (metadata or {}).get(ATTR_CAMERA_NAMES)
    if raw is None:
        return []
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="ignore")
    if isinstance(raw, str):
        return [item.strip() for item in raw.split(",") if item.strip()]
    if isinstance(raw, (list, tuple)):
        return [str(item).strip() for item in raw if str(item).strip()]
    return [str(raw).strip()] if str(raw).strip() else []


def select_primary_camera(
    *,
    metadata: dict[str, Any] | None,
    raw_group: Any | None = None,
    encoded_group: Any | None = None,
    default: str = "fpv",
) -> str:
    available = _available_camera_names(raw_group=raw_group, encoded_group=encoded_group)
    for camera_name in camera_names_from_metadata(metadata):
        if camera_name in available:
            return camera_name
    default = str(default or "fpv")
    if default in available:
        return default
    if "fpv" in available:
        return "fpv"
    if available:
        return sorted(available)[0]
    return default


def sanitize_camera_key(value: Any) -> str:
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in str(value))


def _available_camera_names(*, raw_group: Any | None, encoded_group: Any | None) -> set[str]:
    names: set[str] = set()
    for group in (raw_group, encoded_group):
        if group is None:
            continue
        try:
            names.update(str(key) for key in group.keys())
        except Exception:
            continue
    return names
