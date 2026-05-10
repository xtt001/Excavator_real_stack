"""
JoystickActionSource — maps a gamepad to the real excavator action vector.

Action vector layout (length 4):
    [swing_speed_cmd, boom_speed_cmd, stick_speed_cmd, bucket_speed_cmd]

All commands are normalized to [-1, 1] post-deadzone and post-scale.

Default axis mapping (Xbox / generic dual-stick layout):
    axis 0  → swing    (left stick X)
    axis 1  → boom     (left stick Y, inverted)
    axis 3  → stick    (right stick X)
    axis 4  → bucket   (right stick Y, inverted)

Override via the `axis_map` and `invert` config keys.
All tunable values live in teleop_real_v1.yaml; defaults only provide a
reasonable controller layout.

Requires: pygame (pip install pygame)
"""

from __future__ import annotations

import logging
import time
from typing import Sequence

import numpy as np

from testbed.actions.base import ActionInfo, ActionSource
from testbed.actions.smoothing import ActionResponseSmoother

log = logging.getLogger(__name__)

# Real action indices, matching [swing, boom, stick, bucket].
IDX_SWING  = 0
IDX_BOOM   = 1
IDX_STICK  = 2
IDX_BUCKET = 3
ACTION_DIM = 4


class JoystickActionSource(ActionSource):
    """
    Reads a gamepad via pygame and produces normalized 4-axis commands.

    Parameters
    ----------
    joystick_id  pygame joystick index (0 = first connected gamepad).
    axis_map     List of 4 pygame axis indices mapped to
                 [swing, boom, stick, bucket] in that order.
    invert       List of 4 booleans; True = negate that axis.
    deadzone     Per-axis deadzone threshold (applied before scale).
                 Scalar applies to all axes; list overrides per-axis.
    scale        Per-axis scale applied after deadzone.
                 Scalar or list of 4.
    clip         Hard clip limit applied last (default 1.0).
    """

    def __init__(
        self,
        joystick_id: int = 0,
        joystick_ids: Sequence[int] | None = None,
        axis_map:    Sequence[int]  = (0, 1, 3, 4),
        invert:      Sequence[bool] = (False, True, False, True),
        deadzone:    float | Sequence[float] = 0.05,
        scale:       float | Sequence[float] = 1.0,
        clip:        float = 1.0,
        reset_button: int | None = None,
        discard_button: int | None = None,
        quit_button: int | None = None,
        button_joystick_ids: Sequence[int] | None = None,
        response_profile: dict | None = None,
        default_dt: float = 0.02,
    ) -> None:
        import pygame  # lazy import — only needed when joystick is used

        self._pygame = pygame
        self.joystick_id = joystick_id
        self.axis_map    = list(axis_map)
        self.invert      = list(invert)
        if joystick_ids is None:
            self._joystick_ids = [int(joystick_id)] * ACTION_DIM
        else:
            self._joystick_ids = [int(device_id) for device_id in joystick_ids]
            if len(self._joystick_ids) != ACTION_DIM:
                raise ValueError(f"joystick_ids must have length {ACTION_DIM}")

        # Broadcast scalar → per-axis list
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
        self._response_profile_cfg = dict(response_profile or {})
        self._response_profile_use_measured_dt = bool(
            self._response_profile_cfg.get("use_measured_dt", False)
        )
        self._smoother = self._build_smoother(default_dt=default_dt)

        self._init_pygame()

    # ── ActionSource interface ────────────────────────────────────────────────

    def reset(self) -> None:
        """Re-pump the event queue; no state to reset for a joystick."""
        self._pygame.event.pump()
        if self._smoother is not None:
            self._smoother.reset()
        self._button_states.clear()

    def next_action(self, obs: dict) -> tuple[np.ndarray, ActionInfo]:
        """Read gamepad axes and return a (4,) action vector."""
        self._pygame.event.pump()
        action = np.zeros(ACTION_DIM, dtype=np.float32)

        if self._joystick is None:
            log.warning("No joystick initialised — returning zeros.")
            return action, ActionInfo(source_type="teleop", source_id="joystick_missing")

        t0 = time.perf_counter()
        raw_action = np.zeros(ACTION_DIM, dtype=np.float32)
        for out_idx, (device_id, axis_idx) in enumerate(zip(self._joystick_ids, self.axis_map)):
            joystick = self._joysticks.get(device_id)
            if joystick is None:
                log.warning("Configured joystick_id=%d is not available — returning zeros for that axis.", device_id)
                continue
            raw = float(joystick.get_axis(axis_idx))
            if self.invert[out_idx]:
                raw = -raw
            raw_action[out_idx] = np.clip(raw, -1.0, 1.0)

        if self._smoother is not None:
            action = self._smoother.apply(
                raw_action,
                delta_time=None if self._response_profile_use_measured_dt else self._smoothing_dt,
            )
            action = np.clip(
                action * np.asarray(self._scale, dtype=np.float32),
                -self._clip,
                self._clip,
            ).astype(np.float32, copy=False)
        else:
            for out_idx, raw in enumerate(raw_action):
                dz = self._deadzone[out_idx]
                if abs(raw) < dz:
                    raw = 0.0
                else:
                    raw = (raw - dz * np.sign(raw)) / (1.0 - dz)
                action[out_idx] = np.clip(raw * self._scale[out_idx], -self._clip, self._clip)

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
                "reset_requested": self._button_edge(self._reset_button),
                "discard_requested": self._button_edge(self._discard_button),
                "quit_requested": self._button_edge(self._quit_button),
            },
        )
        return action, info

    def close(self) -> None:
        """Quit pygame joystick subsystem."""
        try:
            for joystick in self._joysticks.values():
                joystick.quit()
            self._pygame.joystick.quit()
        except Exception:
            pass

    # ── Init helpers ──────────────────────────────────────────────────────────

    def _init_pygame(self) -> None:
        pg = self._pygame
        if not pg.get_init():
            pg.init()
        if not pg.joystick.get_init():
            pg.joystick.init()

        n = pg.joystick.get_count()
        if n == 0:
            log.error(
                "No joystick/gamepad detected by pygame. "
                "Connect a gamepad and retry."
            )
            return

        requested_ids = sorted(set(self._joystick_ids))
        for requested_id in requested_ids:
            actual_id = requested_id
            if actual_id >= n:
                log.warning(
                    "Requested joystick_id=%d but only %d found. Using 0 for that binding.",
                    actual_id, n,
                )
                actual_id = 0

            joystick = pg.joystick.Joystick(actual_id)
            joystick.init()
            self._joysticks[requested_id] = joystick
            log.info(
                "Joystick ready: [%d -> %d] %s  axes=%d",
                requested_id,
                actual_id,
                joystick.get_name(),
                joystick.get_numaxes(),
            )

        self._joystick = self._joysticks.get(self._joystick_ids[0])
        if self._smoother is not None:
            log.info("Joystick response smoothing enabled.")

    @classmethod
    def from_config(cls, cfg: dict, *, default_dt: float = 0.02) -> "JoystickActionSource":
        """Construct from a flat config dict."""
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
