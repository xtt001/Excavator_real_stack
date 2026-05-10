"""
Small pygame joystick probe for Python-side teleop bring-up.

Examples
--------
    python scripts/gamepad_probe.py
    python scripts/gamepad_probe.py --watch --joystick-id 0
"""

from __future__ import annotations

import argparse
import time


def _list_devices(pg) -> None:
    count = pg.joystick.get_count()
    print(f"pygame joysticks detected: {count}")
    for joystick_id in range(count):
        joystick = pg.joystick.Joystick(joystick_id)
        joystick.init()
        print(
            f"[{joystick_id}] name={joystick.get_name()!r} "
            f"axes={joystick.get_numaxes()} buttons={joystick.get_numbuttons()} "
            f"hats={joystick.get_numhats()}"
        )


def _watch_device(pg, joystick_id: int, interval_s: float) -> None:
    joystick = pg.joystick.Joystick(joystick_id)
    joystick.init()
    print(
        f"Watching joystick [{joystick_id}] {joystick.get_name()!r}. "
        "Press Ctrl+C to stop."
    )
    while True:
        pg.event.pump()
        axes = [round(float(joystick.get_axis(i)), 4) for i in range(joystick.get_numaxes())]
        buttons = [int(joystick.get_button(i)) for i in range(joystick.get_numbuttons())]
        hats = [tuple(joystick.get_hat(i)) for i in range(joystick.get_numhats())]
        print(f"axes={axes} buttons={buttons} hats={hats}")
        time.sleep(interval_s)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--joystick-id", type=int, default=0)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=0.1)
    args = parser.parse_args()

    import pygame

    pygame.init()
    pygame.joystick.init()
    try:
        _list_devices(pygame)
        if args.watch:
            if pygame.joystick.get_count() <= args.joystick_id:
                raise SystemExit(
                    f"joystick_id={args.joystick_id} not available; "
                    f"found {pygame.joystick.get_count()} device(s)"
                )
            _watch_device(pygame, args.joystick_id, args.interval)
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
