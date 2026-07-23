"""Verify a portable ACT bundle against a no-motion runtime configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from testbed.policies.act.deployment_preflight import (
    BUNDLE_MANIFEST_FILENAME,
    verify_shadow_deployment,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest_path = args.bundle_dir / BUNDLE_MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report = verify_shadow_deployment(
        config_path=args.config,
        bundle_dir=args.bundle_dir,
        manifest=manifest,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
