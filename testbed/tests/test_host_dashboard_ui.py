from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from PyQt5 import QtCore, QtWidgets

from testbed.cli.host_dashboard import (
    REC_ORANGE,
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
            "current_cycle_start_side": "A",
            "next_planned_target_side": "B",
            "next_planned_cycle_count": 5,
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
    assert "当前起始点=A" in window.transition_state.text()
    assert "下次目标位置=B" in window.transition_state.text()
    assert "cycle=1/5" in window.transition_state.text()
    assert "稳定窗=0.32/0.50s" in window.transition_state.text()
    assert "下一条 INITIAL=A" in window.transition_instruction.text()
    assert "按一次左手柄物理按钮 2" in window.transition_instruction.text()
    assert "ARM 自动录制" in window.transition_instruction.text()
    assert "color:#f2f5f7" in window.transition_instruction.styleSheet()
    assert "background:#11181d" in window.transition_instruction.styleSheet()

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
    assert REC_ORANGE in window.transition_state.styleSheet()

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
            "current_cycle_start_side": "A",
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
    assert "cycle=1/4" in window.transition_state.text()

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
    assert "● REC 正在录制" in window.transition_state.text()
    assert "左手柄物理按钮 4" in window.transition_instruction.text()
    assert "当前起始点=A" in window.transition_state.text()
    assert "下次目标位置=B" in window.transition_state.text()
    assert "当前位置=A" in window.transition_state.text()
    assert REC_ORANGE in window.transition_state.styleSheet()
    assert "cycle=1/4" in window.transition_state.text()
    window.close()
    app.processEvents()


def test_v2_record_count_and_compact_details_use_transition_status() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = _LayoutOnlyDashboard()
    window.record_hz = 50.0
    window.max_steps = 15000
    window.latest_status = {
        "receiver": {
            "receiver_mode": "armed",
            "episode_idx": 1,
            "record_steps": 0,
            "saved": 1,
            "scripted_cycle_enabled": 1,
            "scripted_cycle_auto_armed": 1,
            "planner_selected_initial_side": "A",
            "planner_script_id": "four-cycle-left-start-field-v1",
            "planner_cycle_index": 0,
            "planner_planned_cycle_count": 4,
            "planner_target_side": "B",
            "health": {
                "imu": {
                    "online": [1, 1, 1, 1],
                    "valid_attitude": [1, 1, 1, 1],
                    "host_rx_age_ms": [7.0, 8.0, 9.0, 10.0],
                    "packet_loss_count": [0, 0, 0, 0],
                }
            },
            "storage": {
                "recorded_count": 0,
                "last_recorded_episode_idx": -1,
            },
            "real_transition": {
                "phase": "idle",
                "sealed_run_count": 1,
                "planned_run_count": 24,
                "next_initial_side": "A",
                "next_planned_cycle_count": 5,
                "last_automatic_event": {
                    "run_id": "b01_r01",
                    "event_type": "target_ready_mark",
                },
                "ready_state": {
                    "swing_qpos_current_rad": -0.008,
                    "non_swing_qpos_current_rad": [-0.137, -0.509, 0.239],
                },
            },
        }
    }
    window._update_status_panels()
    assert window.prominent_cards["recorded"].value.text() == "1 条"
    assert "已封存 1/24" in window.prominent_cards["recorded"].detail.text()
    assert "b01_r01" in window.prominent_cards["recorded"].detail.text()
    assert window.imu_cards[0].state.text() == "-0.008 rad"
    assert window.imu_cards[3].state.text() == "+0.239 rad"
    assert "ONLINE · age=7.0 ms" in window.imu_cards[0].detail.text()
    assert "loss=" not in window.imu_cards[0].detail.text()
    assert "four-cycle-left-start-field-v1" in window.control_text.text()
    assert "起点=A" in window.control_text.text()
    assert "cycle=1/4" in window.control_text.text()
    assert "target=B" in window.control_text.text()
    assert "ARM=YES" in window.control_text.text()

    window.latest_status["receiver"]["recording"] = 1
    window.latest_status["receiver"]["receiver_mode"] = "recording"
    window._update_status_panels()
    assert window.prominent_cards["recording"].value.text() == "● REC 正在录制"
    assert "物理4号键取消" in window.prominent_cards["recording"].detail.text()
    assert "4px solid" in window.video_label.styleSheet()
    assert REC_ORANGE in window.video_label.styleSheet()

    transition = window.latest_status["receiver"].pop("real_transition")
    window._update_status_panels()
    assert "已 ARM，等待稳定区位后自动开始" in window.transition_state.text()
    assert "当前起始点=A" in window.transition_state.text()
    assert "下次目标位置=B" in window.transition_state.text()
    assert "按钮 7" in window.transition_instruction.text()
    window.latest_status["receiver"]["real_transition"] = transition

    for label in (
        window.record_text,
        window.save_text,
        window.storage_text,
        window.latency_text,
        window.control_text,
    ):
        label.setText("第一行\n第二行\n第三行")
    window.resize(1600, 990)
    window.show()
    app.processEvents()
    for label in (
        window.record_text,
        window.save_text,
        window.storage_text,
        window.latency_text,
        window.control_text,
    ):
        assert label.isVisible()
        assert label.height() >= label.sizeHint().height()
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


def test_saving_state_replaces_stale_rec_and_previous_save_result() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = _LayoutOnlyDashboard()
    window.record_hz = 50.0
    window.max_steps = 15000
    now_ns = time.time_ns()
    window.last_status_rx_ns = now_ns
    window.video_stats = {
        "receive_time_ns": now_ns,
        "stamp_ns": now_ns,
    }
    window.latest_status = {
        "receiver_available": 1,
        "receiver_receive_time_ns": now_ns - 38_000_000_000,
        "receiver": {
            "receiver_mode": "saving",
            "recording": 0,
            "episode_idx": 4,
            "record_steps": 8230,
            "saved": 3,
            "save": {
                "state": "writing",
                "episode_idx": 4,
                "steps": 8230,
                "started_ns": now_ns - 38_000_000_000,
                "finished_ns": 0,
            },
            "health": {
                "ok": 1,
                "bridge_snapshot_age_ms": 14.0,
                "controller_ack": 1,
            },
            "real_transition": {
                "phase": "complete",
                "receiver_mode": "saving",
                "automatic_wait_reason": "saving_run",
                "run_id": "b01_r04",
                "planned_cycle_count": 5,
                "completed_cycles": 5,
                "ready_state": {},
            },
        },
    }

    window._update_status_panels()
    window._refresh_dynamic_state()

    assert window.prominent_cards["recording"].value.text() == "录制已结束"
    assert "正在写盘" in window.prominent_cards["recording"].detail.text()
    assert window.prominent_cards["saving"].value.text() == "正在保存数据"
    assert "episode 4" in window.prominent_cards["saving"].detail.text()
    assert "正在保存数据（录制已结束）" in window.transition_state.text()
    assert "REC 正在录制" not in window.transition_state.text()
    assert "RECEIVER AGE  SAVING" == window.badges["receiver"].text()
    assert "从端 receiver 状态中断" not in window.alert_label.text()
    assert "请勿关闭或拔盘" in window.alert_label.text()

    window.latest_status["receiver"]["receiver_mode"] = "armed"
    window.latest_status["receiver"]["save"] = {
        "state": "success",
        "episode_idx": 4,
        "steps": 8230,
        "file_size_bytes": 1_000_000,
    }
    window._update_status_panels()
    assert window.prominent_cards["saving"].value.text() == "上次保存完成"
    assert "上次 episode 4" in window.prominent_cards["saving"].detail.text()
    window.close()
    app.processEvents()
