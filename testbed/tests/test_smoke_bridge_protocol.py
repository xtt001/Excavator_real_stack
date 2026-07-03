from __future__ import annotations

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_SCRIPT = REPO_ROOT / "scripts" / "smoke_bridge_protocol.py"


def _load_smoke_module():
    spec = importlib.util.spec_from_file_location("smoke_bridge_protocol", SMOKE_SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_smoke_bridge_protocol_defaults_to_legacy_fpv_camera() -> None:
    module = _load_smoke_module()

    assert module._parse_expected_cameras(None) == ["fpv"]
    assert module._parse_expected_cameras("") == ["fpv"]


def test_smoke_bridge_protocol_accepts_configured_gmsl_cameras() -> None:
    module = _load_smoke_module()
    state = {
        "ok": True,
        "payload": {
            "joint": {"qpos": [0.0, 0.0, 0.0, 0.0]},
            "images": {
                "video4": {},
                "video5": {},
                "video6": {},
                "video7": {},
            },
        },
    }

    module._expect_read_state_contract(
        state,
        expected_cameras=["video4", "video5", "video6", "video7"],
    )
