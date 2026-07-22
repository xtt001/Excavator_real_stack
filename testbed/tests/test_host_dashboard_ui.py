from __future__ import annotations

import os
import sys
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from PyQt5 import QtCore, QtWidgets

from testbed.cli.host_dashboard import HostDashboard, _parse_args


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


def test_always_on_top_startup_option(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["host-dashboard", "--always-on-top"],
    )
    assert _parse_args().always_on_top is True
