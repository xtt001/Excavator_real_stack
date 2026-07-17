"""Resolve bundle hashes and validate a field-supplied short-rollout contract."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from testbed.policies.act.deployment_preflight import read_yaml_mapping, sha256_file
from testbed.policies.short_horizon_rollout import ShortHorizonRolloutContract


def preflight_short_horizon_contract(
    *,
    config_path: str | Path,
    bundle_dir: str | Path,
) -> dict[str, Any]:
    """Return one validated immutable contract without enabling runtime motion."""

    payload = read_yaml_mapping(config_path)
    bundle = Path(bundle_dir).expanduser().resolve()
    expected_hashes = {
        "checkpoint_sha256": sha256_file(bundle / "policy_best.ckpt"),
        "resolved_config_sha256": sha256_file(bundle / "resolved_config.yaml"),
    }
    for key, expected in expected_hashes.items():
        configured = payload.get(key)
        if configured in (None, ""):
            payload[key] = expected
        elif str(configured) != expected:
            raise ValueError(f"{key} does not match the verified bundle")
    contract = ShortHorizonRolloutContract.from_mapping(payload)
    return {
        **payload,
        "test_id": contract.test_id,
        "checkpoint_sha256": contract.checkpoint_sha256,
        "resolved_config_sha256": contract.resolved_config_sha256,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = preflight_short_horizon_contract(
        config_path=args.config,
        bundle_dir=args.bundle_dir,
    )
    output = args.output_json.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(json.dumps({"ok": True, "output_json": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
