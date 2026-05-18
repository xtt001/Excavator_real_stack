"""Fixed-rate action pump for real-machine velocity commands.

The recorder may run slower than the low-level hydraulic controller because it
also reads observations, decodes images, and buffers data.  This pump keeps the
latest safe action flowing at a fixed rate so a slow recording step does not
starve the bridge watchdog.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Mapping

import numpy as np

from testbed.backends.real.contracts import as_real_action
from testbed.backends.real.control import ACTION_DIM, ControlResult, LowLevelController

log = logging.getLogger(__name__)


class RealActionPump:
    """Repeat the latest normalized velocity action on a background thread."""

    def __init__(
        self,
        controller: LowLevelController,
        *,
        hz: float = 50.0,
        initial_action: np.ndarray | None = None,
        send_immediately_on_update: bool = True,
        zero_on_stop: bool = True,
        error_log_interval_s: float = 1.0,
    ) -> None:
        if hz <= 0:
            raise ValueError("action pump hz must be positive")
        self._controller = controller
        self.hz = float(hz)
        self.period_s = 1.0 / self.hz
        self.send_immediately_on_update = bool(send_immediately_on_update)
        self.zero_on_stop = bool(zero_on_stop)
        self.error_log_interval_s = float(error_log_interval_s)
        self._action = (
            np.zeros(ACTION_DIM, dtype=np.float32)
            if initial_action is None
            else as_real_action(initial_action, clip=True)
        )
        self._state: Mapping[str, Any] | None = None
        self._latest_result = _local_result(
            action=self._action,
            ack=True,
            fault_code="init",
        )
        self._latest_error = ""
        self._last_error_log_s = 0.0
        self._stop = threading.Event()
        self._schedule_changed = threading.Event()
        self._state_lock = threading.RLock()
        self._send_lock = threading.RLock()
        self._next_send_s = time.perf_counter()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._schedule_changed.clear()
        with self._state_lock:
            self._next_send_s = time.perf_counter()
        self._thread = threading.Thread(
            target=self._run,
            name="real-action-pump",
            daemon=True,
        )
        self._thread.start()

    def update_action(
        self,
        action: np.ndarray,
        *,
        state: Mapping[str, Any] | None = None,
    ) -> ControlResult:
        with self._state_lock:
            self._action = as_real_action(action, clip=True)
            self._state = dict(state) if state is not None else None
        if self.send_immediately_on_update:
            result = self._send_once()
            with self._state_lock:
                self._next_send_s = time.perf_counter() + self.period_s
            self._schedule_changed.set()
            return result
        return self.latest_result

    def apply_status_toggle_mask(self, toggle_mask: int) -> bool:
        try:
            return self._controller.apply_status_toggle_mask(toggle_mask)
        except Exception as exc:
            self._log_send_error(exc, prefix="status toggle failed")
            return False

    @property
    def latest_result(self) -> ControlResult:
        with self._state_lock:
            return self._latest_result

    @property
    def latest_error(self) -> str:
        with self._state_lock:
            return self._latest_error

    def stop(self, *, close_controller: bool = True) -> None:
        self._stop.set()
        self._schedule_changed.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(1.0, self.period_s * 4.0))
        self._thread = None
        if self.zero_on_stop:
            with self._state_lock:
                self._action = np.zeros(ACTION_DIM, dtype=np.float32)
                self._state = None
            self._send_once()
        if close_controller:
            self._controller.close()

    def _run(self) -> None:
        while not self._stop.is_set():
            with self._state_lock:
                remaining_s = self._next_send_s - time.perf_counter()
            if remaining_s > 0.0:
                self._schedule_changed.wait(min(remaining_s, self.period_s))
                self._schedule_changed.clear()
                continue
            self._send_once()
            with self._state_lock:
                self._next_send_s = time.perf_counter() + self.period_s

    def _send_once(self) -> ControlResult:
        with self._state_lock:
            action = self._action.copy()
            state = dict(self._state) if self._state is not None else None
        try:
            with self._send_lock:
                result = self._controller.send(action, state=state)
        except Exception as exc:
            self._log_send_error(exc, prefix="action send failed")
            result = _local_result(
                action=action,
                ack=False,
                fault_code=str(exc),
            )
        with self._state_lock:
            self._latest_result = result
            self._latest_error = "" if result.ack else str(result.fault_code)
        return result

    def _log_send_error(self, exc: Exception, *, prefix: str) -> None:
        now_s = time.monotonic()
        if now_s - self._last_error_log_s >= self.error_log_interval_s:
            log.warning("%s: %s", prefix, exc)
            self._last_error_log_s = now_s


def _local_result(
    *,
    action: np.ndarray,
    ack: bool,
    fault_code: str,
) -> ControlResult:
    commanded = as_real_action(action, clip=True)
    return ControlResult(
        ack=bool(ack),
        fault_code=str(fault_code),
        controller_timestamp_ns=time.time_ns(),
        commanded_action=commanded,
        raw_low_level_command=commanded.copy(),
    )
