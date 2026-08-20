"""Shared YAML configuration loading with recursive ``extends`` support."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_yaml_config(
    path: Path | str,
    *,
    _stack: tuple[Path, ...] = (),
) -> dict[str, Any]:
    """Load a YAML mapping and recursively deep-merge its base configuration."""

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - runtime dependency guard.
        raise RuntimeError("PyYAML is required to load excavator configuration") from exc

    resolved = Path(path).expanduser().resolve()
    if resolved in _stack:
        chain = " -> ".join(str(item) for item in (*_stack, resolved))
        raise ValueError(f"config extends cycle: {chain}")
    with resolved.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError(f"config must decode to a mapping: {resolved}")

    overlay = dict(value)
    base_value = overlay.pop("extends", None)
    if base_value is None:
        return overlay
    if not isinstance(base_value, str) or not base_value.strip():
        raise ValueError(f"config extends must be a non-empty path string: {resolved}")
    base_path = Path(base_value).expanduser()
    if not base_path.is_absolute():
        base_path = resolved.parent / base_path
    base = load_yaml_config(base_path, _stack=(*_stack, resolved))
    return deep_merge_config(base, overlay)


def deep_merge_config(
    base: dict[str, Any], overlay: dict[str, Any]
) -> dict[str, Any]:
    """Merge nested mappings while replacing scalar and sequence leaves."""

    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge_config(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged
