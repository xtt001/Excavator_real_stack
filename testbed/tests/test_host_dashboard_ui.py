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
    RED,
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
        self.last_event_key = None
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


def test_v2_panel_is_read_only_and_explains_the_next_mark_action() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = _LayoutOnlyDashboard()

    window._update_transition_panel(
        {
            "phase": "idle",
            "receiver_mode": "armed",
            "receiver_health_ok": True,
            "mark_next_action": "arm-session",
            "automatic_wait_reason": "session_not_armed",
            "next_initial_side": "A",
            "next_field_context": {
                "workface_reset_id": "wf_001",
                "workface_action": "fresh_strip",
            },
            "camera_sync_state": {"ready": True},
            "ready_state": {
                "window_duration_s": 0.32,
                "window_required_s": 0.5,
            },
        }
    )
    assert "NEXT INITIAL=A" in window.transition_state.text()
    assert "稳定窗=0.32/0.50s" in window.transition_state.text()
    assert "下一条 INITIAL=A" in window.transition_instruction.text()
    assert "按一次左手柄物理按钮 2" in window.transition_instruction.text()
    assert "ARM 自动录制" in window.transition_instruction.text()

    window._update_transition_panel(
        {
            "phase": "goal_committed",
            "receiver_mode": "recording",
            "receiver_health_ok": True,
            "run_id": "b01_r01",
            "next_target_side": "B",
            "mark_next_action": "automatic",
            "automatic_wait_reason": "waiting_target_ready",
            "cycle_excursion_observed": True,
            "last_automatic_event": {
                "event_type": "cycle_excursion_observed",
                "cycle_index": 0,
                "event_step_id": 42,
            },
            "ready_state": {
                "window_duration_s": 0.5,
                "window_required_s": 0.5,
                "window_complete": True,
            },
        }
    )
    assert "无需按键" not in window.transition_instruction.text()
    assert "停稳后自动结束" in window.transition_instruction.text()
    assert "excursion=YES" in window.transition_state.text()
    assert "cycle_excursion_observed" in window.transition_context.text()
    assert "step=42" in window.transition_context.text()

    window._update_transition_panel(
        {
            "phase": "complete",
            "receiver_mode": "recording",
            "receiver_health_ok": True,
            "run_id": "b01_r01",
            "mark_next_action": "automatic",
            "automatic_wait_reason": "saving_run",
            "ready_state": {},
        }
    )
    assert "禁止关闭、重启或拔盘" in window.transition_instruction.text()
    assert RED in window.transition_state.styleSheet()

    window._update_transition_panel(
        {
            "phase": "idle",
            "receiver_mode": "armed",
            "receiver_health_ok": True,
            "session_id": "rt01",
            "next_run_id": "b01_r01",
            "next_run_ordinal": 1,
            "mark_next_action": "start-run",
            "next_field_context": {
                "workface_reset_id": "wf_001",
                "workface_action": "fresh_strip",
            },
            "camera_sync_state": {
                "ready": True,
                "observed_max_skew_ms": 0.04,
            },
            "ready_state": {},
        }
    )
    assert "按左手柄物理按钮 2" in window.transition_instruction.text()
    assert "wf_001 / fresh_strip" in window.transition_context.text()
    assert "PASS" in window.transition_context.text()
    assert not hasattr(window, "transition_primary_button")
    assert not hasattr(window, "workface_reset_id")

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
            "mark_next_action": "initial-ready",
            "field_context": {
                "workface_reset_id": "wf_001",
                "workface_action": "fresh_strip",
            },
            "ready_state": ready_state,
        }
    )
    assert "INITIAL READY" in window.transition_instruction.text()
    assert "A侧" in window.transition_instruction.text()

    window._update_transition_panel(
        {
            "phase": "goal_committed",
            "receiver_mode": "recording",
            "receiver_health_ok": True,
            "run_id": "b01_r01",
            "initial_side": "A",
            "next_target_side": "B",
            "completed_cycles": 0,
            "planned_cycle_count": 4,
            "mark_next_action": "dump-end",
            "ready_state": ready_state,
        }
    )
    assert "目标 B 已按预定序列自动提交" in window.transition_instruction.text()
    assert "DUMP END" in window.transition_instruction.text()
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
    assert config["automatic_annotation"] == {
        "enabled": True,
        "activity_action_abs_min": 0.05,
        "require_inter_run_activity": True,
    }
