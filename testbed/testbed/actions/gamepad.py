"""
JoystickActionSource — maps a gamepad to the real excavator action vector.

Action vector layout (length 4):
    [swing_speed_cmd, boom_speed_cmd, stick_speed_cmd, bucket_speed_cmd]

Status buttons follow control/python/excavator_api_tcp_server.py on one joystick:
    button0~10 rising edge -> toggle_mask bit0~10 -> excavator_api status bits
    button11 rising edge -> group_switch (telemetry only in v1 four-axis path)

Requires: pygame (pip install pygame)
"""

from __future__ import annotations

import logging
import time
from typing import Sequence

import numpy as np

from testbed.actions.base import ActionInfo, ActionSource
from testbed.actions.smoothing import ActionResponseSmoother
from testbed.backends.real.contracts import (
    STATUS_TOGGLE_BIT_COUNT,
    apply_status_toggle_mask_to_status11,
)

log = logging.getLogger(__name__)

IDX_SWING = 0
IDX_BOOM = 1
IDX_STICK = 2
IDX_BUCKET = 3
ACTION_DIM = 4
STATUS_BUTTON_SLOTS = 12


def remap_axis_deadzone(value: float, deadzone: float) -> float:
    """|v|<=dz 为 0； (dz,1] 线性映射到 (0,1]。"""
    dz = float(deadzone)
    if dz < 0.0 or dz >= 1.0:
        raise ValueError("deadzone must be in [0, 1)")
    v = float(np.clip(value, -1.0, 1.0))
    magnitude = abs(v)
    if magnitude <= dz:
        return 0.0
    out = (magnitude - dz) / (1.0 - dz)
    return float(np.sign(v) * out)


class JoystickActionSource(ActionSource):
    """Reads a gamepad via pygame: 4D action + discrete status toggle_mask."""

    def __init__(
        self,
        joystick_id: int = 0,
        joystick_ids: Sequence[int] | None = None,
        axis_map: Sequence[int] = (0, 1, 3, 4),
        invert: Sequence[bool] = (False, True, False, True),
        deadzone: float | Sequence[float] = 0.05,
        scale: float | Sequence[float] = 1.0,
        clip: float = 1.0,
        reset_button: int | None = None,
        discard_button: int | None = None,
        quit_button: int | None = None,
        button_joystick_ids: Sequence[int] | None = None,
        response_profile: dict | None = None,
        default_dt: float = 0.02,
        *,
        status_button_device: int = 0,
        status_buttons_enabled: bool = True,
        status_button_count: int = STATUS_TOGGLE_BIT_COUNT,
        group_switch_button: int | None = 11,
    ) -> None:
        import pygame

        self._pygame = pygame
        self.joystick_id = joystick_id
        self.axis_map = list(axis_map)
        self.invert = list(invert)
        if joystick_ids is None:
            self._joystick_ids = [int(joystick_id)] * ACTION_DIM
        else:
            self._joystick_ids = [int(device_id) for device_id in joystick_ids]
            if len(self._joystick_ids) != ACTION_DIM:
                raise ValueError(f"joystick_ids must have length {ACTION_DIM}")

        if isinstance(deadzone, (int, float)):
            self._deadzone = [float(deadzone)] * ACTION_DIM
        else:
            self._deadzone = [float(d) for d in deadzone]

        if isinstance(scale, (int, float)):
            self._scale = [float(scale)] * ACTION_DIM
        else:
            self._scale = [float(s) for s in scale]

        self._clip = float(clip)
        self._joystick = None
        self._joysticks: dict[int, object] = {}
        self._smoothing_dt = float(default_dt)
        self._button_joystick_ids = (
            sorted(set(int(device_id) for device_id in button_joystick_ids))
            if button_joystick_ids is not None
            else sorted(set(self._joystick_ids))
        )
        self._reset_button = self._normalize_button(reset_button, name="reset_button")
        self._discard_button = self._normalize_button(discard_button, name="discard_button")
        self._quit_button = self._normalize_button(quit_button, name="quit_button")
        self._button_states: dict[tuple[int, int], bool] = {}

        self._status_button_device = int(status_button_device)
        self._status_buttons_enabled = bool(status_buttons_enabled)
        self._status_button_count = int(status_button_count)
        if not (0 < self._status_button_count <= STATUS_TOGGLE_BIT_COUNT):
            raise ValueError(
                f"status_button_count must be in 1..{STATUS_TOGGLE_BIT_COUNT}"
            )
        self._group_switch_button = self._normalize_button(
            group_switch_button, name="group_switch_button"
        )
        self._status11: list[int] = [0] * STATUS_TOGGLE_BIT_COUNT
        self._status_prev_buttons = [0] * STATUS_BUTTON_SLOTS
        self._use_rear_group = False

        self._response_profile_cfg = dict(response_profile or {})
        self._response_profile_use_measured_dt = bool(
            self._response_profile_cfg.get("use_measured_dt", False)
        )
        self._smoother = self._build_smoother(default_dt=default_dt)

        self._init_pygame()

    def reset(self) -> None:
        self._pygame.event.pump()
        if self._smoother is not None:
            self._smoother.reset()
        self._button_states.clear()
        self._status11 = [0] * STATUS_TOGGLE_BIT_COUNT
        self._status_prev_buttons = [0] * STATUS_BUTTON_SLOTS
        self._use_rear_group = False

    def next_action(self, obs: dict) -> tuple[np.ndarray, ActionInfo]:
        self._pygame.event.pump()
        action = np.zeros(ACTION_DIM, dtype=np.float32)

        if self._joystick is None:
            log.warning("No joystick initialised — returning zeros.")
            return action, ActionInfo(source_type="teleop", source_id="joystick_missing")

        t0 = time.perf_counter()
        raw_action = np.zeros(ACTION_DIM, dtype=np.float32)
        for out_idx, (device_id, axis_idx) in enumerate(
            zip(self._joystick_ids, self.axis_map)
        ):
            joystick = self._joysticks.get(device_id)
            if joystick is None:
                continue
            raw = float(joystick.get_axis(axis_idx))
            if self.invert[out_idx]:
                raw = -raw
            raw_action[out_idx] = remap_axis_deadzone(raw, self._deadzone[out_idx])

        scale = np.asarray(self._scale, dtype=np.float32)
        if self._smoother is not None:
            action = self._smoother.apply(
                raw_action,
                delta_time=None
                if self._response_profile_use_measured_dt
                else self._smoothing_dt,
            )
            action = np.clip(action * scale, -self._clip, self._clip).astype(
                np.float32, copy=False
            )
        else:
            action = np.clip(raw_action * scale, -self._clip, self._clip).astype(
                np.float32, copy=False
            )

        toggle_mask = self._poll_status_toggle_mask()
        latency_ms = (time.perf_counter() - t0) * 1000.0
        device_names = [
            self._joysticks[device_id].get_name()
            for device_id in sorted(set(self._joystick_ids))
            if device_id in self._joysticks
        ]
        info = ActionInfo(
            source_type="teleop",
            source_id="joystick:" + "|".join(
                f"{device_id}:{name}"
                for device_id, name in zip(sorted(set(self._joystick_ids)), device_names)
            ),
            latency_ms=latency_ms,
            extras={
                "toggle_mask": int(toggle_mask),
                "status11": list(self._status11),
                "use_rear_group": bool(self._use_rear_group),
                "reset_requested": self._button_edge(self._reset_button),
                "discard_requested": self._button_edge(self._discard_button),
                "quit_requested": self._button_edge(self._quit_button),
            },
        )
        return action, info

    def close(self) -> None:
        try:
            for joystick in self._joysticks.values():
                joystick.quit()
            self._pygame.joystick.quit()
        except Exception:
            pass

    def _poll_status_toggle_mask(self) -> int:
        """Rising-edge mask for status bits (excavator_api_tcp_server semantics)."""

        if not self._status_buttons_enabled:
            return 0

        joystick = self._joysticks.get(self._status_button_device)
        if joystick is None:
            return 0

        cur = [
            1
            if (i < joystick.get_numbuttons() and joystick.get_button(i))
            else 0
            for i in range(STATUS_BUTTON_SLOTS)
        ]
        toggle_mask = 0
        for bit in range(self._status_button_count):
            if self._status_prev_buttons[bit] == 0 and cur[bit] == 1:
                toggle_mask |= 1 << bit

        if toggle_mask:
            apply_status_toggle_mask_to_status11(self._status11, toggle_mask)

        if self._group_switch_button is not None:
            gb = int(self._group_switch_button)
            if (
                gb < STATUS_BUTTON_SLOTS
                and self._status_prev_buttons[gb] == 0
                and cur[gb] == 1
            ):
                self._use_rear_group = not self._use_rear_group

        self._status_prev_buttons = cur
        return int(toggle_mask)

    def _init_pygame(self) -> None:
        pg = self._pygame
        if not pg.get_init():
            pg.init()
        if not pg.joystick.get_init():
            pg.joystick.init()

        n = pg.joystick.get_count()
        if n == 0:
            log.error("No joystick/gamepad detected by pygame.")
            return

        requested_ids = sorted(
            set(self._joystick_ids) | {self._status_button_device}
        )
        for requested_id in requested_ids:
            actual_id = requested_id
            if actual_id >= n:
                log.warning(
                    "Requested joystick_id=%d but only %d found. Using 0.",
                    actual_id,
                    n,
                )
                actual_id = 0

            joystick = pg.joystick.Joystick(actual_id)
            joystick.init()
            self._joysticks[requested_id] = joystick
            log.info(
                "Joystick ready: [%d -> %d] %s axes=%d buttons=%d",
                requested_id,
                actual_id,
                joystick.get_name(),
                joystick.get_numaxes(),
                joystick.get_numbuttons(),
            )

        self._joystick = self._joysticks.get(self._joystick_ids[0])
        if self._smoother is not None:
            log.info("Joystick response smoothing enabled.")
        if self._status_buttons_enabled:
            log.info(
                "Status buttons on device %d: bits 0..%d, group_switch=%s",
                self._status_button_device,
                self._status_button_count - 1,
                self._group_switch_button,
            )

    @classmethod
    def from_config(cls, cfg: dict, *, default_dt: float = 0.02) -> "JoystickActionSource":
        group_switch = cfg.get("group_switch_button", 11)
        return cls(
            joystick_id=cfg.get("joystick_id", 0),
            joystick_ids=cfg.get("joystick_ids"),
            axis_map=cfg.get("axis_map", [0, 1, 3, 4]),
            invert=cfg.get("invert", [False, True, False, True]),
            deadzone=cfg.get("deadzone", 0.05),
            scale=cfg.get("scale", 1.0),
            clip=cfg.get("clip", 1.0),
            reset_button=cfg.get("reset_button"),
            discard_button=cfg.get("discard_button"),
            quit_button=cfg.get("quit_button"),
            button_joystick_ids=cfg.get("button_joystick_ids"),
            response_profile=cfg.get("response_profile"),
            default_dt=default_dt,
            status_button_device=int(cfg.get("status_button_device", 0)),
            status_buttons_enabled=bool(cfg.get("status_buttons_enabled", True)),
            status_button_count=int(
                cfg.get("status_button_count", STATUS_TOGGLE_BIT_COUNT)
            ),
            group_switch_button=None if group_switch is None else int(group_switch),
        )

    def _build_smoother(self, *, default_dt: float) -> ActionResponseSmoother | None:
        cfg = self._response_profile_cfg
        if not bool(cfg.get("enabled", False)):
            return None

        return ActionResponseSmoother(
            deadzone=cfg.get("deadzone", self._deadzone),
            attack_rate=cfg.get("attack_rate", 4.0),
            release_rate=cfg.get("release_rate", 6.0),
            recenter_rate=cfg.get("recenter_rate", 7.0),
            exponent=cfg.get("exponent", 1.0),
            default_dt=default_dt,
        )

    @staticmethod
    def _normalize_button(button: int | None, *, name: str) -> int | None:
        if button is None:
            return None
        button = int(button)
        if button < 0:
            raise ValueError(f"{name} must be >= 0")
        return button

    def _button_edge(self, button_idx: int | None) -> bool:
        if button_idx is None:
            return False

        triggered = False
        for device_id in self._button_joystick_ids:
            joystick = self._joysticks.get(device_id)
            if joystick is None or button_idx >= joystick.get_numbuttons():
                continue

            key = (device_id, button_idx)
            is_pressed = bool(joystick.get_button(button_idx))
            was_pressed = self._button_states.get(key, False)
            self._button_states[key] = is_pressed
            if is_pressed and not was_pressed:
                triggered = True

        return triggered
