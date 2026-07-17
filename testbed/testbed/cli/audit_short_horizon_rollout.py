"""Audit a recorded bounded short-horizon rollout trace."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from testbed.policies.short_horizon_rollout import (
    SCHEMA_VERSION,
    ShortHorizonRolloutContract,
    ShortHorizonRolloutStep,
    evaluate_short_horizon_rollout,
)

REPORT_FILENAME = "short_horizon_rollout_report.json"
SOURCE_MANIFEST_FILENAME = "short_horizon_rollout_source_manifest.json"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m testbed.cli.audit_short_horizon_rollout",
        description=(
            "Validate rollout authority, action-chain logging, and causal links "
            "without sending or rewriting commands."
        ),
    )
    parser.add_argument("--trace-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_short_horizon_rollout_audit(
        trace_json=args.trace_json,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


def run_short_horizon_rollout_audit(
    *,
    trace_json: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Parse one immutable trace and atomically write its capability report."""

    source = Path(trace_json).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("trace JSON root must be an object")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"trace schema_version must be {SCHEMA_VERSION!r}")
    raw_contract = payload.get("contract")
    if not isinstance(raw_contract, Mapping):
        raise ValueError("trace contract must be an object")
    raw_steps = payload.get("steps")
    if not isinstance(raw_steps, list):
        raise ValueError("trace steps must be a list")

    contract = ShortHorizonRolloutContract.from_mapping(raw_contract)
    steps = [
        ShortHorizonRolloutStep.from_mapping(step)
        for step in raw_steps
        if isinstance(step, Mapping)
    ]
    if len(steps) != len(raw_steps):
        raise ValueError("every trace step must be an object")
    report = evaluate_short_horizon_rollout(
        contract=contract,
        steps=steps,
        termination_reason=str(payload.get("termination_reason", "")),
    )
    report_payload = {
        **report,
        "contract": dict(raw_contract),
        "source_manifest": SOURCE_MANIFEST_FILENAME,
    }
    implementation = Path(__file__).resolve()
    owner = (
        implementation.parents[1] / "policies" / "short_horizon_rollout.py"
    )
    source_manifest = {
        "schema_version": SCHEMA_VERSION,
        "trace_json": str(source),
        "trace_json_sha256": _sha256(source),
        "policy_inference_performed": False,
        "command_sent_by_auditor": False,
        "command_rewritten_by_auditor": False,
        "implementation": [
            {"path": str(implementation), "sha256": _sha256(implementation)},
            {"path": str(owner), "sha256": _sha256(owner)},
        ],
    }
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(output / SOURCE_MANIFEST_FILENAME, source_manifest)
    _write_json_atomic(output / REPORT_FILENAME, report_payload)
    return {
        "report": str(output / REPORT_FILENAME),
        "report_sha256": _sha256(output / REPORT_FILENAME),
        "trace_integrity_valid": bool(report["trace_integrity_valid"]),
        "contract_compliant": bool(report["contract_compliant"]),
        "self_generated_state_evidence": report["causal_state_progression"][
            "self_generated_state_evidence"
        ],
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


if __name__ == "__main__":
    main()
