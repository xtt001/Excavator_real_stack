"""CLI entrypoint for the local JSON/TCP bridge mock server."""

from __future__ import annotations

import argparse
import logging
import signal

from testbed.backends.real.bridge_server import JsonTcpBridgeMockServer


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        prog="tb-bridge-mock-server",
        description="Run a local JSON/TCP real-excavator bridge mock server.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--image-width", type=int, default=160)
    parser.add_argument("--image-height", type=int, default=120)
    parser.add_argument("--velocity-scale-rad-s", type=float, default=0.5)
    parser.add_argument("--one-shot", action="store_true")
    args = parser.parse_args()

    server = JsonTcpBridgeMockServer(
        host=args.host,
        port=args.port,
        dt=args.dt,
        image_width=args.image_width,
        image_height=args.image_height,
        velocity_scale_rad_s=args.velocity_scale_rad_s,
        one_shot=bool(args.one_shot),
    )

    def _stop(_sig, _frame) -> None:
        server.shutdown()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    server.serve_forever()


if __name__ == "__main__":
    main()
