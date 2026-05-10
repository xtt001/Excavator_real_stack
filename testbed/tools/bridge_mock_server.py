#!/usr/bin/env python3
"""Run the local JSON/TCP bridge mock server."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from testbed.cli.bridge_mock_server import main


if __name__ == "__main__":
    main()
