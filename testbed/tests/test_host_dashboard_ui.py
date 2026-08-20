from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from PyQt5 import QtCore, QtWidgets

from testbed.cli.host_dashboard import (
    HostDashboard,
    _load_config,
    _parse_args,
)


class _LayoutOnlyDashboard(HostDashboard):
    def __init__(self) -> None:
        QtWidgets.QMainWindow.__init__(self)
        self.args = SimpleNamespace(video_topic="/test/video4/compressed")
        self.config = {}
        self.video_pixmap = None
        self._build_ui()

    def closeEvent(self, event) -> None:
        event.accept()


def test_always_on_top_checkbox_toggles_window_flag() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = _LayoutOnlyDashboard()
    window.show()
    app.processEvents()

    assert window.latency_text.text() == "等待状态"
    assert "status11" in window.machine_status_label.text()
    assert window.badges["sender"].title == "SENDER AGE"
    assert window.badges["receiver"].title == "RECEIVER AGE"

    assert window.always_on_top_checkbox.text() == "保持窗口最前"
    assert not bool(window.windowFlags() & QtCore.Qt.WindowStaysOnTopHint)

    window.always_on_top_checkbox.click()
    app.processEvents()
    assert bool(window.windowFlags() & QtCore.Qt.WindowStaysOnTopHint)
    assert window.isVisible()

    window.always_on_top_checkbox.click()
    app.processEvents()
    assert not bool(window.windowFlags() & QtCore.Qt.WindowStaysOnTopHint)
    assert window.isVisible()
    window.close()


def test_status11_is_visible_in_top_machine_status_bar() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = _LayoutOnlyDashboard()
    window.latest_status = {
        "sender": {
            "status11": [1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
            "action": [0.0, 0.0, 0.0, 0.0],
        }
    }
    window.record_hz = 50.0
    window.max_steps = 15000
    window._update_status_panels()
    text = window.machine_status_label.text()
    assert "点火=ON" in text
    assert "遥控=ON" in text
    assert "先导=ON" in text
    assert "status11=[1, 0, 0, 0, 1, 1" in text
    window.close()
    app.processEvents()


def test_always_on_top_startup_option(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["host-dashboard", "--always-on-top"],
    )
    assert _parse_args().always_on_top is True


def test_v2_panel_exposes_only_the_valid_next_event() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = _LayoutOnlyDashboard()

    window._update_transition_panel(
        {
            "phase": "idle",
            "receiver_mode": "armed",
            "receiver_health_ok": True,
            "session_id": "rt01",
            "next_run_id": "b01_r01",
            "next_run_ordinal": 1,
            "ready_state": {},
        }
    )
    assert window.transition_primary_button.isEnabled()
    assert window.transition_primary_command == "start-run"
    assert window.workface_reset_id.text() == "wf_001"
    assert window.workface_action.text() == "fresh_strip"

    window.workface_action.setText("restore")
    window._update_transition_panel(
        {
            "phase": "idle",
            "receiver_mode": "armed",
            "receiver_health_ok": True,
            "session_id": "rt01",
            "next_run_id": "b02_r01",
            "next_run_ordinal": 2,
            "ready_state": {},
        }
    )
    assert window.workface_reset_id.text() == "wf_002"
    assert window.workface_action.text() == "fresh_strip"

    ready_state = {
        "actual_side": "A",
        "window_complete": True,
        "swing_stable": True,
        "clean_side_window": True,
        "blockers": [],
    }
    window._update_transition_panel(
        {
            "phase": "new",
            "receiver_mode": "armed",
            "receiver_health_ok": True,
            "run_id": "b01_r01",
            "initial_side": "A",
            "next_target_side": "B",
            "completed_cycles": 0,
            "planned_cycle_count": 4,
            "ready_state": ready_state,
        }
    )
    assert window.transition_primary_button.isEnabled()
    assert window.transition_primary_command == "initial-ready"
    assert "A" in window.transition_primary_button.text()

    window._update_transition_panel(
        {
            "phase": "ready",
            "receiver_mode": "recording",
            "receiver_health_ok": True,
            "run_id": "b01_r01",
            "initial_side": "A",
            "next_target_side": "B",
            "completed_cycles": 0,
            "planned_cycle_count": 4,
            "ready_state": ready_state,
        }
    )
    assert window.transition_primary_button.isEnabled()
    assert window.transition_primary_command == "commit-goal"
    assert "B" in window.transition_primary_button.text()
    window.close()
    app.processEvents()


def test_dashboard_resolves_v2_extended_config() -> None:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "testbed"
        / "configs"
        / "teleop_real_transition_v2_0_1.yaml"
    )
    config = _load_config(config_path)
    assert config["record_hz"] == 50.0
    assert config["max_steps"] == 15000
    assert config["home"]["enabled"] == 1
    assert config["field_context_defaults"] == {
        "workface_reset_id_prefix": "wf_",
        "workface_action": "fresh_strip",
    }
