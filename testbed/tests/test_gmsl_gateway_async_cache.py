from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[2]
TESTBED_PATH = str(REPO_ROOT / "testbed")
ROS2_BRIDGE_PATH = str(REPO_ROOT / "ros2_bridge")
if TESTBED_PATH not in sys.path:
    sys.path.insert(0, TESTBED_PATH)
if ROS2_BRIDGE_PATH not in sys.path:
    sys.path.insert(0, ROS2_BRIDGE_PATH)

from excavator_bridge_gateway.gateway_server import (  # noqa: E402
    BridgeGateway,
    _select_nearest_gmsl_frames,
)


def _frame(sequence: int, timestamp_ms: float, *, now_ns: int) -> SimpleNamespace:
    return SimpleNamespace(
        sequence=sequence,
        v4l2_timestamp_ns=int(timestamp_ms * 1_000_000),
        receive_time_ns=now_ns,
        timestamp_ns=now_ns,
        width=2,
        height=2,
        rgb=bytes(12),
        v4l2_flags=0,
    )


class GmslGatewayAsyncCacheTests(unittest.TestCase):
    def test_selects_nearest_cached_frames_without_requiring_valid_skew(self) -> None:
        now_ns = time.time_ns()
        histories = {
            "video4": [_frame(1, 66.0, now_ns=now_ns), _frame(2, 99.0, now_ns=now_ns)],
            "video5": [_frame(1, 66.0, now_ns=now_ns), _frame(2, 99.0, now_ns=now_ns)],
            "video6": [_frame(1, 76.0, now_ns=now_ns), _frame(2, 109.0, now_ns=now_ns)],
            "video7": [_frame(1, 50.0, now_ns=now_ns), _frame(2, 83.0, now_ns=now_ns)],
        }

        selected, group = _select_nearest_gmsl_frames(
            histories,
            max_stale_ms=1000,
            max_group_skew_ms=5.0,
            now_ns=now_ns,
        )

        self.assertEqual(group["group_target_v4l2_timestamp_ns"], 83_000_000)
        self.assertEqual(group["group_skew_ms"], 23.0)
        self.assertEqual(group["group_valid"], 0)
        self.assertEqual(group["group_source"], "gmsl_v4l2_timestamp_nearest_cache")
        self.assertEqual(
            {key: frame.sequence for key, frame in selected.items()},
            {"video4": 2, "video5": 2, "video6": 1, "video7": 2},
        )

    def test_gateway_group_read_delegates_to_cache_without_sleeping(self) -> None:
        expected_frames = {"video4": object()}
        expected_group = {"group_valid": 0}

        class Cache:
            def nearest_group(self, *, max_stale_ms, max_group_skew_ms):
                self.args = (max_stale_ms, max_group_skew_ms)
                return expected_frames, expected_group

        cache = Cache()
        gateway = BridgeGateway.__new__(BridgeGateway)
        gateway.gmsl_readers = {"video4": object()}
        gateway.gmsl_frame_cache = cache
        gateway.fpv_max_stale_ms = 1000
        gateway.gmsl_max_group_skew_ms = 5.0

        started = time.monotonic()
        frames, group = gateway._read_gmsl_timestamp_group()
        elapsed_s = time.monotonic() - started

        self.assertIs(frames, expected_frames)
        self.assertIs(group, expected_group)
        self.assertEqual(cache.args, (1000, 5.0))
        self.assertLess(elapsed_s, 0.01)


if __name__ == "__main__":
    unittest.main()
