from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "slave_real_stack.sh"


def test_no_camera_stack_configures_gateway_without_camera_source() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert 'prepare_start "${no_camera}"' in text
    assert 'export EXCAVATOR_NO_CAMERA="${no_camera}"' in text
    assert 'export EXCAVATOR_CAMERA_SOURCE=none' in text
    assert 'camera disabled: read_state images are empty' in text


def test_run_uses_dashboard_log_view_by_default() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert 'LOG_VIEW="${EXCAVATOR_LOG_VIEW:-dashboard}"' in text
    assert "dashboard_logs()" in text
    assert 'EXCAVATOR_LOG_VIEW=dashboard' in text
    assert '# dashboard | plain' in text
    assert 'tail -n 80 -f "${dir}"/*.log' in text
