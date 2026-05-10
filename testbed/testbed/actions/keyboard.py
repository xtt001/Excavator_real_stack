"""
KeyboardActionSource — keyboard fallback for the real excavator action vector.

Uses pygame keyboard events.  Much less precise than a joystick but
useful for debugging and CI environments without hardware.

Default key mapping (WASD + arrows):
    W / S   → boom    +/-
    A / D   → swing   -/+
    ↑ / ↓   → stick   +/-
    ← / →   → bucket  -/+

Hold key = continuous speed command at `key_speed` magnitude.
All values are normalized to [-1, 1].

Requires: pygame
"""

from __future__ import annotations

import logging
import time

import numpy as np

from testbed.actions.base import ActionInfo, ActionSource

log = logging.getLogger(__name__)

# Real action indices, matching [swing, boom, stick, bucket].
IDX_SWING  = 0
IDX_BOOM   = 1
IDX_STICK  = 2
IDX_BUCKET = 3
ACTION_DIM = 4


class KeyboardActionSource(ActionSource):
    """
    WASD + arrow keys -> normalized 4-axis commands.

    Parameters
    ----------
    key_speed    Magnitude when a key is held (default 0.5, range 0..1).
    window_title Optional pygame window title (set to "" to skip display).
    """

    def __init__(self, key_speed: float = 0.5, window_title: str = "Testbed Teleop") -> None:
        import pygame  # lazy import

        self._pygame    = pygame
        self.key_speed  = float(key_speed)
        self._win_title = window_title
        self._init_pygame()

    # ── ActionSource interface ────────────────────────────────────────────────

    def reset(self) -> None:
        self._pygame.event.pump()

    def next_action(self, obs: dict) -> tuple[np.ndarray, ActionInfo]:
        pg = self._pygame
        pg.event.pump()

        keys  = pg.key.get_pressed()
        spd   = self.key_speed
        action = np.zeros(ACTION_DIM, dtype=np.float32)

        t0 = time.perf_counter()

        # swing: A(-) / D(+)
        if keys[pg.K_a]:
            action[IDX_SWING] -= spd
        if keys[pg.K_d]:
            action[IDX_SWING] += spd

        # boom: W(+) / S(-)
        if keys[pg.K_w]:
            action[IDX_BOOM] += spd
        if keys[pg.K_s]:
            action[IDX_BOOM] -= spd

        # stick: UP(+) / DOWN(-)
        if keys[pg.K_UP]:
            action[IDX_STICK] += spd
        if keys[pg.K_DOWN]:
            action[IDX_STICK] -= spd

        # bucket: RIGHT(+) / LEFT(-)
        if keys[pg.K_RIGHT]:
            action[IDX_BUCKET] += spd
        if keys[pg.K_LEFT]:
            action[IDX_BUCKET] -= spd

        action = np.clip(action, -1.0, 1.0)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        info = ActionInfo(
            source_type="teleop",
            source_id="keyboard",
            latency_ms=latency_ms,
        )
        return action, info

    def close(self) -> None:
        try:
            self._pygame.quit()
        except Exception:
            pass

    # ── Init helper ───────────────────────────────────────────────────────────

    def _init_pygame(self) -> None:
        pg = self._pygame
        if not pg.get_init():
            pg.init()
        if self._win_title:
            pg.display.set_mode((400, 120))
            pg.display.set_caption(self._win_title)
        log.info("KeyboardActionSource ready (WASD+arrows).")

    @classmethod
    def from_config(cls, cfg: dict) -> "KeyboardActionSource":
        return cls(
            key_speed=cfg.get("key_speed", 0.5),
            window_title=cfg.get("window_title", "Testbed Teleop"),
        )
