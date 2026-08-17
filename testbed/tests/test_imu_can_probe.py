from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "imu_can_probe.py"
SPEC = importlib.util.spec_from_file_location("imu_can_probe", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_daoyuan_chain_four_device_summary(monkeypatch) -> None:
    monkeypatch.setattr(MODULE, "_interface_is_up", lambda _interface: True)
    monkeypatch.setattr(
        MODULE,
        "_capture_ids",
        lambda _interface, _duration: [0x121, 0x122, 0x123, 0x124, 0x121],
    )

    result = MODULE._summarize_interface("can5", 3.0)

    assert result["detected_protocols"] == ["daoyuan_chain"]
    assert result["daoyuan_chain_frames"] == 5
    assert result["missing_daoyuan_ids"] == []
    assert result["daoyuan_chain_ids"]["0x121"] == {
        "joint": "boom",
        "frames": 2,
    }


def test_legacy_highspeed_protocol_remains_supported(monkeypatch) -> None:
    monkeypatch.setattr(MODULE, "_interface_is_up", lambda _interface: True)
    legacy_ids = [0x200, 0x201, 0x202, 0x203]
    monkeypatch.setattr(
        MODULE,
        "_capture_ids",
        lambda _interface, _duration: legacy_ids,
    )

    result = MODULE._summarize_interface("can3", 3.0)

    assert result["detected_protocols"] == ["highspeed_ch1"]
    assert result["missing_raw_addr_0_to_3"] == []
    assert result["missing_daoyuan_ids"] == ["0x121", "0x122", "0x123", "0x124"]
