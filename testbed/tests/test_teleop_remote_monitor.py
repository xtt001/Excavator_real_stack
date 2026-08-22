from __future__ import annotations

import io
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from testbed.cli.teleop_remote import _RemoteTeleopMonitor


def test_monitor_draws_and_counts_task_mark() -> None:
    monitor = _RemoteTeleopMonitor(
        enabled=True,
        target="192.168.100.1:8770",
        input_device="joystick",
        rate_hz=50.0,
        source_id="host_joystick",
        min_interval_s=0.02,
    )
    monitor._record_events(  # noqa: SLF001 - focused monitor regression test.
        seq=1,
        now_s=1.0,
        event_flags={"mark": True},
    )
    output = io.StringIO()
    with patch("sys.stdout", output):
        monitor._draw(  # noqa: SLF001 - exercises the crashing render path.
            seq=1,
            action=np.zeros(4, dtype=np.float32),
            info=SimpleNamespace(source_id="host_joystick", latency_ms=0.0),
            extras={"mark_requested": True, "status11": [0] * 11},
            now_s=1.0,
            receiver_status=None,
        )

    assert monitor._event_counts["mark"] == 1  # noqa: SLF001
    assert "TASK ARM/MARK:" in output.getvalue()
    assert "count=1" in output.getvalue()
