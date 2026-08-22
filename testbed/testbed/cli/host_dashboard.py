"""Read-only host operations dashboard for the real excavator stack."""

from __future__ import annotations

import argparse
import math
import os
import signal
import socket
import threading
import time
from pathlib import Path
from typing import Any

# GNOME/Mutter intentionally ignores client-requested always-on-top for native
# Wayland windows.  Run this operations dashboard through XWayland so the
# standard _NET_WM_STATE_ABOVE hint is enforceable.  Explicit test/headless
# platform choices (for example offscreen) remain untouched.
if (
    os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland"
    and os.environ.get("DISPLAY")
):
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

from PyQt5 import QtCore, QtGui, QtWidgets

from testbed.host_status import (
    DEFAULT_HOST_STATUS_HOST,
    DEFAULT_HOST_STATUS_PORT,
    HostStatusProtocolError,
    decode_host_status,
)
from testbed.config_loader import load_yaml_config


AXIS_NAMES = ("回转", "动臂", "斗杆", "铲斗")
GREEN = "#2ecc71"
AMBER = "#f5b041"
RED = "#e74c3c"
GRAY = "#7f8c8d"
BLUE = "#3498db"


class DashboardSignals(QtCore.QObject):
    status_received = QtCore.pyqtSignal(object)
    status_error = QtCore.pyqtSignal(str)
    video_error = QtCore.pyqtSignal(str)


class StatusReceiverThread(threading.Thread):
    def __init__(
        self,
        signals: DashboardSignals,
        *,
        host: str,
        port: int,
    ) -> None:
        super().__init__(name="host-dashboard-status", daemon=True)
        self.signals = signals
        self.host = str(host)
        self.port = int(port)
        self.stop_event = threading.Event()
        self.socket: socket.socket | None = None

    def stop(self) -> None:
        self.stop_event.set()
        sock = self.socket
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass

    def run(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket = sock
        try:
            sock.bind((self.host, self.port))
            sock.settimeout(0.2)
        except OSError as exc:
            self.signals.status_error.emit(
                f"状态端口 {self.host}:{self.port} 无法监听：{exc}"
            )
            return
        while not self.stop_event.is_set():
            try:
                frame, _addr = sock.recvfrom(65_535)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                payload = decode_host_status(frame)
            except HostStatusProtocolError as exc:
                self.signals.status_error.emit(f"状态包解析失败：{exc}")
                continue
            payload["_dashboard_receive_time_ns"] = time.time_ns()
            self.signals.status_received.emit(payload)


class VideoPipelineThread(QtCore.QThread):
    """ROS receive + latest-only JPEG decode, independent of the Qt GUI thread."""

    def __init__(self, signals: DashboardSignals, *, topic: str) -> None:
        super().__init__()
        self.setObjectName("host-dashboard-video")
        self.signals = signals
        self.topic = str(topic)
        self.stop_event = threading.Event()
        self._latest_lock = threading.Lock()
        self._latest_frame: tuple[int, Any, dict[str, Any]] | None = None

    def stop(self) -> None:
        self.stop_event.set()

    def latest_frame(self) -> tuple[int, Any, dict[str, Any]] | None:
        """Return the newest decoded frame without queueing older GUI work."""
        with self._latest_lock:
            return self._latest_frame

    def run(self) -> None:
        try:
            import rclpy

            from excavator_bridge_gateway.host_fpv_low_latency_viewer import (
                LatestCompressedViewer,
                LatestJpegDecoder,
                _spin,
            )
        except Exception as exc:
            self.signals.video_error.emit(f"视频运行环境不可用：{exc}")
            return

        while not self.stop_event.is_set():
            node = None
            decoder = None
            spin_thread = None
            try:
                rclpy.init()
                node = LatestCompressedViewer(
                    self.topic,
                    node_name="host_integrated_dashboard_video",
                )
                spin_thread = threading.Thread(
                    target=_spin,
                    args=(node,),
                    name="host-dashboard-ros-spin",
                    daemon=True,
                )
                spin_thread.start()
                decoder = LatestJpegDecoder(node, scale=1.0)
                decoder.start()
                last_seq = 0
                last_stats_s = time.monotonic()
                last_received = 0
                last_decoded = 0
                while not self.stop_event.is_set() and rclpy.ok():
                    latest = decoder.latest()
                    if latest is not None and int(latest[0]) != last_seq:
                        seq, stamp_ns, frame, decode_ms = latest
                        last_seq = int(seq)
                        now_s = time.monotonic()
                        received = node.received_count()
                        decoded, skipped = decoder.stats()
                        dt = max(now_s - last_stats_s, 1e-6)
                        stats = {
                            "seq": int(seq),
                            "stamp_ns": int(stamp_ns),
                            "decode_ms": float(decode_ms),
                            "receive_hz": float((received - last_received) / dt),
                            "decode_hz": float((decoded - last_decoded) / dt),
                            "skipped": int(skipped),
                            "receive_time_ns": int(time.time_ns()),
                        }
                        with self._latest_lock:
                            self._latest_frame = (int(seq), frame, stats)
                        if now_s - last_stats_s >= 1.0:
                            last_stats_s = now_s
                            last_received = received
                            last_decoded = decoded
                    self.stop_event.wait(0.004)
            except Exception as exc:
                if not self.stop_event.is_set():
                    self.signals.video_error.emit(
                        "视频线程异常，将在 2 秒后重试："
                        f"{type(exc).__name__}: {exc}"
                    )
            finally:
                if decoder is not None:
                    decoder.stop()
                if node is not None:
                    node.destroy_node()
                try:
                    if rclpy.ok():
                        rclpy.shutdown()
                except Exception:
                    pass
                if spin_thread is not None:
                    spin_thread.join(timeout=1.0)
            self.stop_event.wait(2.0)


class StatusBadge(QtWidgets.QLabel):
    def __init__(self, title: str) -> None:
        super().__init__()
        self.title = str(title)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setMinimumWidth(125)
        self.set_state("WAIT", GRAY)

    def set_state(self, value: str, color: str) -> None:
        self.setText(f"{self.title}  {value}")
        self.setStyleSheet(
            "QLabel {"
            f"background:{color}; color:#101820; border-radius:6px; "
            "padding:7px 10px; font-weight:700;"
            "}"
        )


class ProminentStateCard(QtWidgets.QFrame):
    def __init__(self, title: str) -> None:
        super().__init__()
        self.setMinimumHeight(80)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Fixed,
        )
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(1)
        self.title = QtWidgets.QLabel(str(title))
        self.title.setStyleSheet("font-size:13px; font-weight:700;")
        self.value = QtWidgets.QLabel("等待状态")
        self.value.setAlignment(QtCore.Qt.AlignCenter)
        self.value.setStyleSheet("font-size:25px; font-weight:800;")
        self.detail = QtWidgets.QLabel("-")
        self.detail.setAlignment(QtCore.Qt.AlignCenter)
        self.detail.setStyleSheet("font-size:11px; color:#bdc3c7;")
        layout.addWidget(self.title)
        layout.addWidget(self.value, stretch=1)
        layout.addWidget(self.detail)
        self.set_state("等待状态", "-", GRAY)

    def set_state(self, value: str, detail: str, color: str) -> None:
        self.value.setText(str(value))
        self.value.setStyleSheet(
            f"font-size:25px; font-weight:800; color:{color};"
        )
        self.detail.setText(str(detail))
        self.setStyleSheet(
            "QFrame {"
            f"border:3px solid {color}; border-radius:9px; background:#202830;"
            "} QLabel { border:none; }"
        )


class ImuCard(QtWidgets.QFrame):
    def __init__(self, index: int) -> None:
        super().__init__()
        self.index = int(index)
        self.setMinimumHeight(90)
        self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        layout = QtWidgets.QVBoxLayout(self)
        self.title = QtWidgets.QLabel(f"IMU {index}")
        self.title.setAlignment(QtCore.Qt.AlignCenter)
        self.title.setStyleSheet("font-weight:800; font-size:16px;")
        self.state = QtWidgets.QLabel("等待状态")
        self.state.setAlignment(QtCore.Qt.AlignCenter)
        self.state.setStyleSheet("font-weight:800; font-size:19px;")
        self.detail = QtWidgets.QLabel("age=-  loss=-")
        self.detail.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.title)
        layout.addWidget(self.state)
        layout.addWidget(self.detail)
        self.update_state(None, None, None, None)

    def update_state(
        self,
        online: Any,
        attitude: Any,
        age_ms: Any,
        loss: Any,
    ) -> None:
        if online is None:
            color = GRAY
            state = "WAIT"
        elif not bool(online):
            color = RED
            state = "OFFLINE"
        elif not bool(attitude):
            color = AMBER
            state = "姿态无效"
        elif _float(age_ms, -1.0) < 0 or _float(age_ms, -1.0) > 100.0:
            color = AMBER
            state = "数据陈旧"
        else:
            color = GREEN
            state = "ONLINE"
        self.state.setText(state)
        self.state.setStyleSheet(
            f"font-weight:800; font-size:19px; color:{color};"
        )
        age_text = "-" if age_ms is None else f"{_float(age_ms, -1.0):.1f} ms"
        loss_text = "-" if loss is None else str(_int(loss, 0))
        self.detail.setText(f"age={age_text}  loss={loss_text}")
        self.setStyleSheet(
            "QFrame {"
            f"border:2px solid {color}; border-radius:7px; background:#202830;"
            "} QLabel { border:none; }"
        )


class HostDashboard(QtWidgets.QMainWindow):
    def __init__(self, args: argparse.Namespace, config: dict[str, Any]) -> None:
        super().__init__()
        self.args = args
        self.config = config
        self.latest_status: dict[str, Any] = {}
        self.last_status_rx_ns = 0
        self.video_stats: dict[str, Any] = {}
        self.video_pixmap: QtGui.QPixmap | None = None
        self.last_video_seq = 0
        self.last_event_key: tuple[Any, ...] | None = None
        self.last_status_error = ""
        self.last_video_error = ""
        self.record_hz = float(config.get("record_hz", 0.0) or 0.0)
        self.max_steps = int(config.get("max_steps", 0) or 0)

        self.setWindowTitle("Excavator 主端现场状态")
        self.resize(1500, 900)
        self.setMinimumSize(1100, 700)
        self._build_ui()
        self.always_on_top_checkbox.setChecked(
            bool(getattr(args, "always_on_top", False))
        )

        self.signals = DashboardSignals()
        self.signals.status_received.connect(self._on_status)
        self.signals.status_error.connect(self._on_status_error)
        self.signals.video_error.connect(self._on_video_error)
        self.status_thread = StatusReceiverThread(
            self.signals,
            host=str(args.status_host),
            port=int(args.status_port),
        )
        self.video_thread = VideoPipelineThread(
            self.signals,
            topic=str(args.video_topic),
        )
        self.status_thread.start()
        self.video_thread.start()

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._refresh_dynamic_state)
        self.timer.start(100)
        self.video_timer = QtCore.QTimer(self)
        self.video_timer.timeout.connect(self._consume_latest_video_frame)
        self.video_timer.start(33)

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        header = QtWidgets.QHBoxLayout()
        self.badges = {
            name: StatusBadge(title)
            for name, title in (
                ("sender", "SENDER AGE"),
                ("receiver", "RECEIVER AGE"),
                ("bridge", "BRIDGE AGE"),
                ("camera", "VIDEO4 AGE"),
                ("control", "CONTROL"),
            )
        }
        for badge in self.badges.values():
            header.addWidget(badge)
        header.addStretch(1)
        self.always_on_top_checkbox = QtWidgets.QCheckBox("保持窗口最前")
        self.always_on_top_checkbox.setToolTip(
            "勾选后状态窗口始终显示在其他普通窗口上方"
        )
        self.always_on_top_checkbox.setStyleSheet(
            "font-size:13px; font-weight:700; padding:5px;"
        )
        self.always_on_top_checkbox.toggled.connect(
            self._set_always_on_top
        )
        header.addWidget(self.always_on_top_checkbox)
        self.clock_label = QtWidgets.QLabel("--:--:--")
        self.clock_label.setStyleSheet("font-size:18px; font-weight:700;")
        header.addWidget(self.clock_label)
        root.addLayout(header)

        self.alert_label = QtWidgets.QLabel("等待主端 sender 状态……")
        self.alert_label.setWordWrap(True)
        self.alert_label.setStyleSheet(
            f"background:{GRAY}; color:white; padding:7px; border-radius:5px;"
        )
        top_state_row = QtWidgets.QHBoxLayout()
        top_state_row.setSpacing(8)
        top_state_row.addWidget(self.alert_label, stretch=3)
        self.machine_status_label = QtWidgets.QLabel(
            "机器状态：等待 sender / status11"
        )
        self.machine_status_label.setAlignment(QtCore.Qt.AlignCenter)
        self.machine_status_label.setTextInteractionFlags(
            QtCore.Qt.TextSelectableByMouse
        )
        self.machine_status_label.setStyleSheet(
            f"background:#11181d; color:{GRAY}; padding:7px; "
            "border:1px solid #46535e; border-radius:5px; "
            "font-family:monospace; font-size:13px; font-weight:800;"
        )
        top_state_row.addWidget(self.machine_status_label, stretch=5)
        root.addLayout(top_state_row)

        transition_group = QtWidgets.QGroupBox(
            "v2.0.1 连续录制（session 只 ARM 一次，边界自动，GUI 只读）"
        )
        transition_layout = QtWidgets.QVBoxLayout(transition_group)
        transition_layout.setSpacing(5)
        self.transition_state = QtWidgets.QLabel("等待 v2 receiver 状态")
        self.transition_state.setWordWrap(True)
        self.transition_state.setStyleSheet(
            "font-size:16px; font-weight:800; padding:5px; "
            "background:#11181d; border-radius:5px;"
        )
        transition_layout.addWidget(self.transition_state)

        status_row = QtWidgets.QHBoxLayout()
        self.transition_instruction = QtWidgets.QLabel(
            "等待 receiver 自动流程状态"
        )
        self.transition_instruction.setWordWrap(True)
        self.transition_instruction.setStyleSheet(
            "font-size:15px; font-weight:800; padding:6px; "
            f"color:{BLUE}; background:#17222a; border-radius:5px;"
        )
        status_row.addWidget(self.transition_instruction, stretch=3)
        self.transition_context = QtWidgets.QLabel("自动工作面：-")
        self.transition_context.setWordWrap(True)
        self.transition_context.setTextInteractionFlags(
            QtCore.Qt.TextSelectableByMouse
        )
        self.transition_context.setStyleSheet(
            "font-family:monospace; font-size:13px; padding:6px; "
            "background:#11181d; border-radius:5px;"
        )
        status_row.addWidget(self.transition_context, stretch=2)
        transition_layout.addLayout(status_row)
        root.addWidget(transition_group)
        self._update_transition_panel({})

        summary = QtWidgets.QHBoxLayout()
        summary.setSpacing(8)
        self.prominent_cards = {
            name: ProminentStateCard(title)
            for name, title in (
                ("recorded", "已录制条数"),
                ("recording", "录制状态"),
                ("saving", "保存状态"),
                ("gohome", "回位状态"),
            )
        }
        for card in self.prominent_cards.values():
            summary.addWidget(card)
        root.addLayout(summary)

        critical = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        critical.setMinimumHeight(145)
        root.addWidget(critical)

        imu_group = QtWidgets.QGroupBox("四路 IMU 健康（必须全部 ONLINE）")
        imu_layout = QtWidgets.QHBoxLayout(imu_group)
        imu_layout.setSpacing(7)
        self.imu_cards = [ImuCard(index) for index in range(4)]
        for card in self.imu_cards:
            imu_layout.addWidget(card)
        critical.addWidget(imu_group)

        home_group = QtWidgets.QGroupBox("HOME 标定（实际运行配置优先）")
        home_layout = QtWidgets.QVBoxLayout(home_group)
        home_layout.setSpacing(4)
        self.home_state = QtWidgets.QLabel("等待 HOME 配置")
        self.home_state.setAlignment(QtCore.Qt.AlignCenter)
        self.home_state.setStyleSheet("font-size:18px; font-weight:800;")
        home_layout.addWidget(self.home_state)
        home_axes = QtWidgets.QHBoxLayout()
        home_axes.setSpacing(5)
        self.home_axis_labels: list[QtWidgets.QLabel] = []
        for axis_name in AXIS_NAMES:
            label = QtWidgets.QLabel(f"{axis_name}\n-")
            label.setAlignment(QtCore.Qt.AlignCenter)
            label.setMinimumHeight(52)
            label.setStyleSheet(
                "background:#11181d; border:1px solid #46535e; "
                "border-radius:5px; padding:4px; font-family:monospace; "
                "font-size:11px;"
            )
            self.home_axis_labels.append(label)
            home_axes.addWidget(label)
        home_layout.addLayout(home_axes)
        self.home_detail = QtWidgets.QLabel("配置来源：-")
        self.home_detail.setWordWrap(True)
        self.home_detail.setStyleSheet("font-size:11px; color:#bdc3c7;")
        home_layout.addWidget(self.home_detail)
        critical.addWidget(home_group)
        critical.setStretchFactor(0, 5)
        critical.setStretchFactor(1, 4)
        self._update_home_panel({})

        middle = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        root.addWidget(middle, stretch=7)

        video_group = QtWidgets.QGroupBox("video4 / eye_left 实时预览")
        video_layout = QtWidgets.QVBoxLayout(video_group)
        self.video_label = QtWidgets.QLabel("等待 video4 图像")
        self.video_label.setAlignment(QtCore.Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 360)
        self.video_label.setStyleSheet("background:#05080a; color:#7f8c8d;")
        self.video_info = QtWidgets.QLabel("topic: " + str(self.args.video_topic))
        video_layout.addWidget(self.video_label, stretch=1)
        video_layout.addWidget(self.video_info)
        middle.addWidget(video_group)

        side = QtWidgets.QWidget()
        side_layout = QtWidgets.QVBoxLayout(side)
        self.record_text = self._info_box("录制状态")
        self.save_text = self._info_box("保存状态")
        self.storage_text = self._info_box("存储状态")
        self.latency_text = self._info_box("实时延迟")
        self.control_text = self._info_box("控制与动作（只读）")
        for group, widget in (
            ("录制", self.record_text),
            ("HDF5 保存", self.save_text),
            ("外置盘", self.storage_text),
            ("当前动作年龄", self.latency_text),
            ("控制链路", self.control_text),
        ):
            box = QtWidgets.QGroupBox(group)
            layout = QtWidgets.QVBoxLayout(box)
            layout.addWidget(widget)
            side_layout.addWidget(box)
        side_layout.addStretch(1)
        middle.addWidget(side)
        middle.setStretchFactor(0, 7)
        middle.setStretchFactor(1, 3)

        lower = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        root.addWidget(lower, stretch=2)

        qc_group = QtWidgets.QGroupBox("在线 QC")
        qc_layout = QtWidgets.QVBoxLayout(qc_group)
        self.qc_text = self._info_box("QC")
        qc_layout.addWidget(self.qc_text)
        lower.addWidget(qc_group)

        event_group = QtWidgets.QGroupBox("最近事件 / 告警")
        event_layout = QtWidgets.QVBoxLayout(event_group)
        self.event_log = QtWidgets.QPlainTextEdit()
        self.event_log.setReadOnly(True)
        self.event_log.document().setMaximumBlockCount(100)
        event_layout.addWidget(self.event_log)
        lower.addWidget(event_group)
        lower.setStretchFactor(0, 3)
        lower.setStretchFactor(1, 5)

        self.setStyleSheet(
            "QMainWindow, QWidget { background:#151b20; color:#ecf0f1; }"
            "QGroupBox { border:1px solid #46535e; border-radius:6px; "
            "margin-top:10px; padding-top:8px; font-weight:700; }"
            "QGroupBox::title { subcontrol-origin:margin; left:10px; padding:0 4px; }"
            "QPlainTextEdit { background:#0d1216; border:1px solid #46535e; }"
        )

    @staticmethod
    def _info_box(_name: str) -> QtWidgets.QLabel:
        label = QtWidgets.QLabel("等待状态")
        label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        label.setWordWrap(True)
        label.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        label.setStyleSheet("font-family:monospace; font-size:13px;")
        return label

    def _update_home_panel(self, receiver_home: dict[str, Any]) -> None:
        home = dict(receiver_home or self.config.get("home", {}) or {})
        if not home or not bool(home.get("available", 0)):
            self.home_state.setText("HOME 配置不可用")
            self.home_state.setStyleSheet(
                f"font-size:18px; font-weight:800; color:{GRAY};"
            )
            for index, label in enumerate(self.home_axis_labels):
                label.setText(f"{AXIS_NAMES[index]}\n-")
            self.home_detail.setText("配置来源：-")
            return

        enabled = bool(home.get("enabled", 0))
        consistent = _int(home.get("runtime_phase_consistent"), -1)
        if not enabled:
            headline, color = "HOME 未启用", RED
        elif consistent == 0:
            headline, color = "HOME 已启用 / runtime 与 phase 不一致", AMBER
        elif consistent == 1:
            headline, color = "HOME 已启用 / runtime 与 phase 一致", GREEN
        else:
            headline, color = "HOME 已启用", GREEN
        self.home_state.setText(headline)
        self.home_state.setStyleSheet(
            f"font-size:18px; font-weight:800; color:{color};"
        )

        pose = _vector(home.get("home_pose_rad"), 4)
        success = _vector(home.get("success_tolerance_rad"), 4)
        for index, label in enumerate(self.home_axis_labels):
            if pose is None:
                value_text = "-"
            else:
                rad = _float(pose[index], 0.0)
                value_text = f"{rad:+.3f}rad / {math.degrees(rad):+.1f}°"
                if success is not None:
                    tolerance_deg = abs(
                        math.degrees(_float(success[index], 0.0))
                    )
                    value_text += f"\ntol ±{tolerance_deg:.1f}°"
            label.setText(f"{AXIS_NAMES[index]}\n{value_text}")

        source = str(home.get("source_config", "") or "-")
        timeout = _format_float(home.get("timeout_s"))
        dwell = _format_float(home.get("dwell_s"))
        self.home_detail.setText(
            f"成功容差显示在各轴下方  |  timeout={timeout}s  dwell={dwell}s\n"
            f"配置来源：{source}"
        )

    def _update_transition_panel(self, transition: dict[str, Any]) -> None:
        transition = dict(transition or {})
        phase = str(transition.get("phase", "") or "")
        ready = dict(transition.get("ready_state", {}) or {})
        blockers = [str(item) for item in ready.get("blockers", []) or []]
        actual_side = str(ready.get("actual_side", "unknown") or "unknown")
        initial_side = str(transition.get("initial_side", "-") or "-")
        next_initial_side = str(
            transition.get("next_initial_side", initial_side) or initial_side
        )
        next_initial_text = next_initial_side if phase in {"idle", "disabled"} else "-"
        target_side = str(transition.get("next_target_side", "-") or "-")
        completed = _int(transition.get("completed_cycles"), 0)
        planned = _int(transition.get("planned_cycle_count"), 0)
        stable = bool(ready.get("swing_stable", False))
        window_complete = bool(ready.get("window_complete", False))
        window_duration = _float(ready.get("window_duration_s"), 0.0)
        window_required = _float(ready.get("window_required_s"), 0.5)
        window_progress = min(window_duration, window_required)
        excursion = bool(transition.get("cycle_excursion_observed", False))
        run_id = str(transition.get("run_id", "-") or "-")
        receiver_mode = str(transition.get("receiver_mode", "-") or "-")
        health_ok = bool(transition.get("receiver_health_ok", False))

        if not transition:
            self.transition_state.setText(
                "等待 v2 receiver 状态；普通 v1 receiver 不提供该面板数据"
            )
            color = GRAY
        else:
            blocker_text = ",".join(blockers) if blockers else "无"
            armed_text = _yes_no(transition.get("session_armed", False))
            auto_wait = str(
                transition.get("automatic_wait_reason", "") or "-"
            )
            self.transition_state.setText(
                f"ARM={armed_text}  |  RUN {run_id}  |  phase={phase}  |  "
                f"cycle={completed}/{planned}  |  "
                f"NEXT INITIAL={next_initial_text}  |  initial={initial_side}  |  "
                f"TARGET={target_side}  |  "
                f"实际侧={actual_side}  |  swing稳定={_yes_no(stable)}  |  "
                f"稳定窗={window_progress:.2f}/{window_required:.2f}s"
                f"({_yes_no(window_complete)})  |  excursion={_yes_no(excursion)}  |  "
                f"blockers={blocker_text}  |  "
                f"auto={auto_wait}"
            )
            saving = auto_wait == "saving_run" or phase in {"complete", "cycles_complete"}
            color = RED if saving else (GREEN if health_ok and not blockers else AMBER)
        self.transition_state.setStyleSheet(
            "font-size:16px; font-weight:800; padding:5px; "
            f"background:#11181d; border:2px solid {color}; border-radius:5px;"
        )

        mark_action = str(transition.get("mark_next_action", "") or "")
        automatic_wait = str(
            transition.get("automatic_wait_reason", "") or ""
        )
        instruction = {
            "arm-session": (
                f"下一条 INITIAL={next_initial_side}；准备好整个 session 后，"
                "按一次左手柄物理按钮 2：ARM 自动录制"
            ),
            "automatic": {
                "waiting_inter_run_activity": (
                    f"已 ARM：自然调整工作面/姿态后回到下一条 INITIAL={next_initial_side}，"
                    "程序自动开始"
                ),
                "waiting_initial_ready": (
                    f"下一条 INITIAL={next_initial_side}：自然到达该侧并停稳；无需按键"
                ),
                f"waiting_initial_side_{next_initial_side}": (
                    f"下一条 INITIAL={next_initial_side}：当前侧不匹配，自然移动到该侧并停稳"
                ),
                "waiting_operator_action_neutral": (
                    f"下一条 INITIAL={next_initial_side}：摇杆回中后自动开始；无需按键"
                ),
                "camera_sync": "等待四相机同步门禁；不要开始动作",
                "receiver_health": "等待 receiver 健康恢复；不要开始动作",
                "waiting_cycle_excursion": (
                    f"TARGET={target_side}：正常操作；等待 swing 形成有效行程"
                ),
                "confirming_cycle_excursion": (
                    f"TARGET={target_side}：正在确认 swing 有效行程；无需按键"
                ),
                "waiting_target_ready": (
                    f"TARGET={target_side}：回到目标侧并停稳后自动结束本 cycle"
                ),
                "saving_run": "本 run 自动完成，正在写盘；禁止关闭、重启或拔盘",
                "recorder_start_requested": "INITIAL READY 已自动确认，正在开始录制",
            }.get(
                automatic_wait,
                f"已 ARM：自动流程等待 {automatic_wait or '状态更新'}；无需标注按键",
            ),
            "start-run": "按左手柄物理按钮 2（MARK）：开始下一条 RUN",
            "initial-ready": (
                f"机器位于 {initial_side}侧、swing 稳定且铲斗离土后，"
                "按 MARK 确认 INITIAL READY"
            ),
            "dump-end": (
                f"目标 {target_side} 已按预定序列自动提交；"
                "完成倒料后按 MARK 标记 DUMP END"
            ),
            "target-ready": (
                f"到达 {target_side}侧、swing 稳定且铲斗离土后，"
                "按 MARK 确认 TARGET READY"
            ),
        }.get(mark_action, "等待 receiver 自动流程状态更新")
        mark_error = str(transition.get("last_mark_error", "") or "")
        if mark_error:
            instruction += f"  |  上次 MARK 未接受：{mark_error}"
        self.transition_instruction.setText(instruction)
        self.transition_instruction.setStyleSheet(
            "font-size:15px; font-weight:800; padding:6px; "
            f"color:{RED if mark_error else BLUE}; "
            "background:#17222a; border-radius:5px;"
        )

        context = dict(transition.get("field_context", {}) or {})
        if phase == "idle":
            context = dict(transition.get("next_field_context", {}) or {})
        reset_id = str(context.get("workface_reset_id", "-") or "-")
        action = str(context.get("workface_action", "-") or "-")
        camera = dict(transition.get("camera_sync_state", {}) or {})
        camera_gate = "PASS" if camera.get("ready", False) else "WAIT"
        camera_skew = camera.get("observed_max_skew_ms")
        skew_text = "-" if camera_skew is None else f"{_float(camera_skew, 0.0):.2f}ms"
        last_auto = dict(transition.get("last_automatic_event", {}) or {})
        last_type = str(last_auto.get("event_type", "-") or "-")
        last_cycle = last_auto.get("cycle_index")
        last_step = last_auto.get("event_step_id")
        last_detail = (
            f"{last_type}  cycle={last_cycle if last_cycle is not None else '-'}  "
            f"step={last_step if last_step is not None else '-'}"
        )
        self.transition_context.setText(
            f"自动工作面: {reset_id} / {action}\n"
            f"录制前相机同步: {camera_gate}  max_skew={skew_text}\n"
            f"最近自动事件: {last_detail}"
        )

    @QtCore.pyqtSlot(object)
    def _on_status(self, status: object) -> None:
        self.latest_status = dict(status or {})
        self.last_status_rx_ns = time.time_ns()
        self._update_status_panels()

    @QtCore.pyqtSlot(str)
    def _on_status_error(self, message: str) -> None:
        if message != self.last_status_error:
            self.last_status_error = message
            self._append_event("STATUS ERROR", message)

    def _consume_latest_video_frame(self) -> None:
        latest = self.video_thread.latest_frame()
        if latest is None or int(latest[0]) == self.last_video_seq:
            return
        seq, frame, stats = latest
        self.last_video_seq = int(seq)
        image = frame
        if image is None or not hasattr(image, "shape"):
            return
        height, width = image.shape[:2]
        bytes_per_line = int(image.strides[0])
        qimage = QtGui.QImage(
            image.data,
            width,
            height,
            bytes_per_line,
            QtGui.QImage.Format_BGR888,
        ).copy()
        self.video_pixmap = QtGui.QPixmap.fromImage(qimage)
        self.video_stats = dict(stats or {})
        self._render_video_pixmap()

    @QtCore.pyqtSlot(str)
    def _on_video_error(self, message: str) -> None:
        if message != self.last_video_error:
            self.last_video_error = message
            self._append_event("VIDEO ERROR", message)

    def _render_video_pixmap(self) -> None:
        if self.video_pixmap is None:
            return
        self.video_label.setPixmap(
            self.video_pixmap.scaled(
                self.video_label.size(),
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.FastTransformation,
            )
        )

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._render_video_pixmap()

    def _update_status_panels(self) -> None:
        status = self.latest_status
        sender = dict(status.get("sender", {}) or {})
        receiver = dict(status.get("receiver", {}) or {})
        health = dict(receiver.get("health", {}) or {})
        save = dict(receiver.get("save", {}) or {})
        storage = dict(receiver.get("storage", {}) or {})
        qc = dict(receiver.get("online_qc", {}) or {})
        latency = dict(receiver.get("latency", {}) or {})
        transition = dict(receiver.get("real_transition", {}) or {})

        mode = str(receiver.get("receiver_mode", "WAIT") or "WAIT").upper()
        episode = _int(receiver.get("episode_idx"), -1)
        steps = _int(receiver.get("record_steps"), 0)
        saved = _int(receiver.get("saved"), 0)
        duration = steps / self.record_hz if self.record_hz > 0 else None
        progress = (
            f"{100.0 * steps / self.max_steps:.1f}%"
            if self.max_steps > 0
            else "-"
        )
        duration_text = "-" if duration is None else f"{duration:.1f} s"
        save_state_key = str(save.get("state", "idle") or "idle").lower()
        save_state = save_state_key.upper()
        has_receiver_status = bool(receiver)
        recorded_count_available = storage.get("recorded_count") is not None
        recorded_count = _int(storage.get("recorded_count"), 0)
        recorded_detail = (
            f"本次进程保存 {saved} 条 / 最近 episode "
            f"{storage.get('last_recorded_episode_idx', '-')}"
            if recorded_count_available
            else f"当前 receiver 尚未提供磁盘总数 / 本次保存 {saved} 条"
        )
        self.prominent_cards["recorded"].set_state(
            (
                f"{recorded_count} 条"
                if recorded_count_available
                else "待重启统计"
            ) if has_receiver_status else "等待状态",
            recorded_detail if has_receiver_status else "等待 receiver",
            (
                BLUE if recorded_count_available else AMBER
            ) if has_receiver_status else GRAY,
        )

        recording_active = bool(receiver.get("recording", 0))
        self.prominent_cards["recording"].set_state(
            "正在录制" if recording_active else (
                "未在录制" if has_receiver_status else "等待状态"
            ),
            (
                f"episode {episode} / {steps} steps / {duration_text}"
                if has_receiver_status
                else "等待 receiver"
            ),
            RED if recording_active else GRAY,
        )

        if save_state_key == "writing" or mode == "SAVING":
            save_headline, save_color = "正在保存", AMBER
        elif save_state_key == "failed":
            save_headline, save_color = "保存失败", RED
        elif save_state_key == "success":
            save_headline, save_color = "保存完成", GREEN
        elif has_receiver_status:
            save_headline, save_color = "等待保存", GRAY
        else:
            save_headline, save_color = "等待状态", GRAY
        self.prominent_cards["saving"].set_state(
            save_headline,
            f"episode {save.get('episode_idx', '-')} / {_fmt_bytes(save.get('file_size_bytes'))}",
            save_color,
        )

        go_home_result = str(receiver.get("go_home_result", "") or "")
        go_home_running = bool(receiver.get("go_home_running", 0)) or mode == "GO_HOME"
        if go_home_running:
            go_home_headline, go_home_color = "正在回位", BLUE
        elif go_home_result.startswith("failed:"):
            go_home_headline, go_home_color = "回位失败", RED
        elif go_home_result == "done":
            go_home_headline, go_home_color = "回位完成", GREEN
        elif has_receiver_status:
            go_home_headline, go_home_color = "未在回位", GRAY
        else:
            go_home_headline, go_home_color = "等待状态", GRAY
        self.prominent_cards["gohome"].set_state(
            go_home_headline,
            go_home_result or f"receiver mode={mode}",
            go_home_color,
        )

        self.record_text.setText(
            f"mode       {mode}\n"
            f"episode    {episode}\n"
            f"steps      {steps} / {self.max_steps or '-'} ({progress})\n"
            f"duration   {duration_text}\n"
            f"saved      {saved}\n"
            f"go_home    {receiver.get('go_home_result') or '-'}"
        )

        save_path = str(save.get("path", "") or "-")
        save_size = _fmt_bytes(save.get("file_size_bytes"))
        save_elapsed = _duration_from_ns(save.get("started_ns"), save.get("finished_ns"))
        self.save_text.setText(
            f"state      {save_state}\n"
            f"episode    {save.get('episode_idx', '-')}\n"
            f"steps      {save.get('steps', '-')}\n"
            f"elapsed    {save_elapsed}\n"
            f"size       {save_size}\n"
            f"error      {save.get('error_code') or '-'}\n"
            f"final_qc   {save.get('qc_status') or '-'}\n"
            f"path       {save_path}"
        )

        free = _fmt_bytes(storage.get("free_bytes"))
        total = _fmt_bytes(storage.get("total_bytes"))
        free_fraction = 100.0 * _float(storage.get("free_fraction"), 0.0)
        self.storage_text.setText(
            f"mounted    {_yes_no(storage.get('mounted'))}\n"
            f"mount      {storage.get('mount_point') or '-'}\n"
            f"free       {free} / {total} ({free_fraction:.1f}%)\n"
            f"dataset    {storage.get('path') or '-'}"
        )

        action_age = _optional_finite_float(
            latency.get("action_age_ms", latency.get("policy_action_age_ms"))
        )
        action_age_text = "--" if action_age is None else f"{action_age:.1f} ms"
        self.latency_text.setText(
            f"action_age  {action_age_text}\n"
            f"source      {latency.get('action_source') or '-'}\n"
            "定义        receiver 状态时刻 - 当前动作采样时刻"
        )

        raw_action = _vector(receiver.get("policy_action"), 4)
        safe_action = _vector(receiver.get("policy_assisted_action"), 4)
        commanded = _vector(receiver.get("commanded_action"), 4)
        sender_action = _vector(sender.get("action"), 4)
        status11 = sender.get("status11")
        status_values = (
            [int(value) for value in status11]
            if isinstance(status11, (list, tuple))
            else []
        )
        ignition = status_values[0] if len(status_values) > 0 else None
        remote_mode = status_values[4] if len(status_values) > 4 else None
        pilot = status_values[5] if len(status_values) > 5 else None
        if status_values:
            self.machine_status_label.setText(
                "机器状态  "
                f"点火={_on_off(ignition)}  遥控={_on_off(remote_mode)}  "
                f"先导={_on_off(pilot)}  status11={status_values}"
            )
            machine_color = GREEN if ignition and remote_mode and pilot else AMBER
        else:
            self.machine_status_label.setText("机器状态：等待 sender / status11")
            machine_color = GRAY
        self.machine_status_label.setStyleSheet(
            f"background:#11181d; color:{machine_color}; padding:7px; "
            f"border:2px solid {machine_color}; border-radius:5px; "
            "font-family:monospace; font-size:13px; font-weight:800;"
        )
        self.control_text.setText(
            f"mode       {receiver.get('control_mode') or 'manual'}\n"
            f"sender     {_format_action(sender_action)}\n"
            f"policy     {_format_action(raw_action)}\n"
            f"safe       {_format_action(safe_action)}\n"
            f"commanded  {_format_action(commanded)}\n"
            f"status11   {sender.get('status11') or '-'}"
        )

        imu = dict(health.get("imu", {}) or {})
        online = _vector(imu.get("online"), 4)
        attitude = _vector(imu.get("valid_attitude"), 4)
        ages = _vector(imu.get("host_rx_age_ms"), 4)
        losses = _vector(imu.get("packet_loss_count"), 4)
        for index, card in enumerate(self.imu_cards):
            card.update_state(
                _item(online, index),
                _item(attitude, index),
                _item(ages, index),
                _item(losses, index),
            )
        self._update_home_panel(dict(receiver.get("home", {}) or {}))
        self._update_transition_panel(transition)

        warnings = qc.get("warning_codes") or []
        self.qc_text.setText(
            f"status          {qc.get('status') or 'WAIT'}\n"
            f"error           {qc.get('error_code') or '-'}\n"
            f"warnings        {','.join(str(x) for x in warnings) or '-'}\n"
            f"train_exclude   {qc.get('train_exclude', '-')}\n"
            f"healthy_steps   {qc.get('healthy_steps', '-')}\n"
            f"healthy_frac    {_format_fraction(qc.get('healthy_fraction'))}\n"
            f"masked_steps    {qc.get('train_exclude_steps', '-')}\n"
            f"fpv_brightness  {_format_float(qc.get('fpv_brightness'))}\n"
            f"fpv_contrast    {_format_float(qc.get('fpv_contrast'))}"
        )

        event_key = (
            mode,
            episode,
            saved,
            save_state,
            str(save.get("path", "")),
            str(health.get("error_code", "")),
            str(qc.get("status", "")),
            str(qc.get("error_code", "")),
            str(receiver.get("message", "")),
        )
        if self.last_event_key != event_key:
            self.last_event_key = event_key
            message = str(receiver.get("message", "") or "-")
            self._append_event(
                "STATE",
                f"mode={mode} ep={episode} steps={steps} save={save_state} "
                f"health={health.get('error_code') or 'OK'} "
                f"qc={qc.get('status') or '-'} msg={message}",
            )

    def _refresh_dynamic_state(self) -> None:
        now_ns = time.time_ns()
        self.clock_label.setText(time.strftime("%F %T"))
        sender_age_ms = _age_ms(now_ns, self.last_status_rx_ns)
        receiver_age_ms = _age_ms(
            now_ns,
            self.latest_status.get("receiver_receive_time_ns", 0),
        )
        receiver = dict(self.latest_status.get("receiver", {}) or {})
        health = dict(receiver.get("health", {}) or {})
        sender_ok = sender_age_ms >= 0 and sender_age_ms <= 500.0
        receiver_ok = (
            bool(self.latest_status.get("receiver_available"))
            and receiver_age_ms >= 0
            and receiver_age_ms <= 500.0
        )
        bridge_age = _float(health.get("bridge_snapshot_age_ms"), -1.0)
        bridge_ok = receiver_ok and bridge_age >= 0 and bridge_age <= 200.0
        video_rx_age = _age_ms(now_ns, self.video_stats.get("receive_time_ns", 0))
        video_ok = video_rx_age >= 0 and video_rx_age <= 500.0
        control_ok = receiver_ok and bool(health.get("controller_ack", 0))

        self._set_badge("sender", sender_ok, sender_age_ms)
        self._set_badge("receiver", receiver_ok, receiver_age_ms)
        self._set_badge("bridge", bridge_ok, bridge_age)
        self._set_badge("camera", video_ok, video_rx_age)
        if control_ok:
            self.badges["control"].set_state("ACK", GREEN)
        elif receiver_ok:
            self.badges["control"].set_state("FAULT", RED)
        else:
            self.badges["control"].set_state("WAIT", GRAY)

        stamp_ns = _int(self.video_stats.get("stamp_ns"), 0)
        frame_age = _age_ms(now_ns, stamp_ns)
        self.video_info.setText(
            f"topic={self.args.video_topic}  "
            f"recv={_float(self.video_stats.get('receive_hz'), 0.0):.1f} Hz  "
            f"decode={_float(self.video_stats.get('decode_ms'), 0.0):.2f} ms  "
            f"frame_age={_format_ms(frame_age)}  "
            f"skipped={_int(self.video_stats.get('skipped'), 0)}"
        )

        alerts: list[str] = []
        if not sender_ok:
            alerts.append("主端 sender 状态中断")
        if not receiver_ok:
            alerts.append("从端 receiver 状态中断")
        if receiver_ok and not bool(health.get("ok", 0)):
            alerts.append("健康门禁：" + str(health.get("error_code") or "unknown"))
        if not video_ok:
            alerts.append("video4 无新帧")
        storage = dict(receiver.get("storage", {}) or {})
        if storage.get("available") and not storage.get("mounted"):
            alerts.append("外置 USB 未挂载")
        save = dict(receiver.get("save", {}) or {})
        if save.get("state") == "failed":
            alerts.append("最近 HDF5 保存失败")
        qc = dict(receiver.get("online_qc", {}) or {})
        if qc.get("status") == "FAIL_EPISODE":
            alerts.append("在线 QC 失败：" + str(qc.get("error_code") or "unknown"))
        if alerts:
            self.alert_label.setText("  |  ".join(alerts))
            self.alert_label.setStyleSheet(
                f"background:{RED}; color:white; padding:7px; border-radius:5px; font-weight:700;"
            )
        else:
            self.alert_label.setText(
                "所有关键链路正常（GUI 只读；v2 正常 run 内无需标注按键）"
            )
            self.alert_label.setStyleSheet(
                f"background:{GREEN}; color:#101820; padding:7px; border-radius:5px; font-weight:700;"
            )

    def _set_badge(self, name: str, ok: bool, age_ms: float | None) -> None:
        if ok:
            suffix = "OK" if age_ms is None else f"{age_ms:.0f}ms"
            self.badges[name].set_state(suffix, GREEN)
        else:
            suffix = "WAIT" if age_ms is None or age_ms < 0 else f"{age_ms:.0f}ms"
            self.badges[name].set_state(suffix, RED if age_ms is not None else GRAY)

    def _append_event(self, category: str, message: str) -> None:
        self.event_log.appendPlainText(
            f"{time.strftime('%H:%M:%S')} [{category}] {message}"
        )

    @QtCore.pyqtSlot(bool)
    def _set_always_on_top(self, enabled: bool) -> None:
        was_visible = self.isVisible()
        self.setWindowFlag(
            QtCore.Qt.WindowStaysOnTopHint,
            bool(enabled),
        )
        # Changing a top-level window flag hides the native window. Restore it
        # immediately when this action came from the visible checkbox.
        if was_visible:
            self.show()
            if enabled:
                self.raise_()
                self.activateWindow()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.timer.stop()
        self.video_timer.stop()
        self.status_thread.stop()
        self.video_thread.stop()
        self.status_thread.join(timeout=1.0)
        self.video_thread.wait(2_000)
        event.accept()


def _load_config(path: Path) -> dict[str, Any]:
    try:
        data = load_yaml_config(path)
    except Exception:
        return {
            "record_hz": 0.0,
            "max_steps": 0,
            "home": {
                "available": 0,
                "enabled": 0,
                "source_config": str(path),
            },
        }
    task = dict(data.get("task", {}) or {})
    transition = dict(data.get("real_transition", {}) or {})
    teleop = dict(data.get("teleop", {}) or {})
    recording = dict(teleop.get("recording", {}) or {})
    go_home = dict(recording.get("go_home", {}) or {})
    phase = dict(data.get("phase_labeling", {}) or {})
    runtime_pose = _vector(go_home.get("home_pose_rad"), 4)
    phase_pose = _vector(phase.get("home_pose_rad"), 4)
    consistent = -1
    if runtime_pose is not None and phase_pose is not None:
        consistent = int(
            all(
                abs(_float(runtime_pose[index], 0.0) - _float(phase_pose[index], 0.0))
                <= 1.0e-6
                for index in range(4)
            )
        )
    return {
        "record_hz": float(task.get("record_hz", task.get("control_hz", 0.0)) or 0.0),
        "max_steps": int(task.get("max_steps", 0) or 0),
        "field_context_defaults": dict(
            transition.get("field_context_defaults", {}) or {}
        ),
        "automatic_annotation": dict(
            transition.get("automatic_annotation", {}) or {}
        ),
        "home": {
            "available": 1,
            "enabled": int(bool(go_home.get("enabled", False))),
            "home_pose_rad": runtime_pose,
            "phase_home_pose_rad": phase_pose,
            "success_tolerance_rad": _vector(
                go_home.get("success_tolerance_rad"), 4
            ),
            "near_tolerance_rad": _vector(
                go_home.get("near_tolerance_rad"), 4
            ),
            "center_tolerance_rad": _vector(
                go_home.get("center_tolerance_rad"), 4
            ),
            "timeout_s": go_home.get("timeout_s"),
            "dwell_s": go_home.get("dwell_s"),
            "runtime_phase_consistent": consistent,
            "source_config": str(path.resolve()),
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only host operations dashboard for the real stack."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("testbed/testbed/configs/teleop_real_v1.yaml"),
    )
    parser.add_argument("--status-host", default=DEFAULT_HOST_STATUS_HOST)
    parser.add_argument("--status-port", type=int, default=DEFAULT_HOST_STATUS_PORT)
    parser.add_argument(
        "--video-topic",
        default="/excavator/eye/video4/image_raw/compressed",
    )
    parser.add_argument(
        "--always-on-top",
        action="store_true",
        help="Start with the dashboard window kept above ordinary windows.",
    )
    return parser.parse_args()


def _vector(value: Any, width: int) -> list[Any] | None:
    if not isinstance(value, (list, tuple)) or len(value) != width:
        return None
    return list(value)


def _item(values: list[Any] | None, index: int) -> Any:
    return None if values is None or index >= len(values) else values[index]


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _optional_finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) and result >= 0.0 else None


def _int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _age_ms(now_ns: int, timestamp_ns: Any) -> float:
    timestamp = _int(timestamp_ns, 0)
    if timestamp <= 0:
        return -1.0
    return max(0.0, (int(now_ns) - timestamp) / 1_000_000.0)


def _fmt_bytes(value: Any) -> str:
    amount = _float(value, -1.0)
    if amount < 0:
        return "-"
    for suffix in ("B", "KiB", "MiB", "GiB", "TiB"):
        if amount < 1024.0 or suffix == "TiB":
            return f"{amount:.1f} {suffix}"
        amount /= 1024.0
    return "-"


def _duration_from_ns(start_ns: Any, finish_ns: Any) -> str:
    start = _int(start_ns, 0)
    finish = _int(finish_ns, 0)
    if start <= 0:
        return "-"
    end = finish if finish > 0 else time.time_ns()
    return f"{max(0.0, (end - start) / 1_000_000_000.0):.2f} s"


def _format_action(values: list[Any] | None) -> str:
    if values is None:
        return "-"
    return "[" + ",".join(f"{_float(value, 0.0):+.3f}" for value in values) + "]"


def _yes_no(value: Any) -> str:
    return "YES" if bool(value) else "NO"


def _on_off(value: Any) -> str:
    return "ON" if bool(value) else "OFF"


def _format_fraction(value: Any) -> str:
    if value is None:
        return "-"
    return f"{100.0 * _float(value, 0.0):.1f}%"


def _format_float(value: Any) -> str:
    return "-" if value is None else f"{_float(value, 0.0):.3f}"


def _format_ms(value: float) -> str:
    return "-" if value < 0 else f"{value:.1f} ms"


def main() -> None:
    args = _parse_args()
    app = QtWidgets.QApplication([])
    app.setApplicationName("Excavator Host Dashboard")
    window = HostDashboard(args, _load_config(args.config))
    signal.signal(signal.SIGINT, lambda *_args: app.quit())
    signal.signal(signal.SIGTERM, lambda *_args: app.quit())
    signal_timer = QtCore.QTimer()
    signal_timer.timeout.connect(lambda: None)
    signal_timer.start(200)
    window.show()
    raise SystemExit(app.exec_())


if __name__ == "__main__":
    main()
