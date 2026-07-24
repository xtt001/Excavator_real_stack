"""Small immutable-artifact helpers used by the SimVerify M0 pipeline."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from testbed.simverify.contracts import sha256_file


def write_json(path: str | Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    target = Path(path)
    text = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    )
    _write_new_text(target, text + "\n")
    return artifact_identity(target)


def write_jsonl(
    path: str | Path,
    records: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    target = Path(path)
    text = "".join(
        json.dumps(
            record,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
        for record in records
    )
    _write_new_text(target, text)
    return artifact_identity(target)


def artifact_identity(path: str | Path) -> dict[str, Any]:
    target = Path(path).resolve(strict=True)
    return {
        "path": str(target),
        "size_bytes": int(target.stat().st_size),
        "sha256": sha256_file(target),
    }


def write_checksums(
    root: str | Path,
    identities: Iterable[Mapping[str, Any]],
    *,
    path: str | Path,
) -> dict[str, Any]:
    base = Path(root).resolve(strict=True)
    rows: list[tuple[str, str]] = []
    for identity in identities:
        target = Path(str(identity["path"])).resolve(strict=True)
        try:
            relative = target.relative_to(base)
        except ValueError as exc:
            raise ValueError(f"artifact lies outside export root: {target}") from exc
        if relative.parts and relative.parts[0] == "oracle_audit":
            raise ValueError("main checksums must not reference oracle_audit")
        rows.append((str(relative), str(identity["sha256"])))
    rows.sort(key=lambda item: item[0])
    if len({relative for relative, _sha in rows}) != len(rows):
        raise ValueError("duplicate artifact in checksum inventory")
    text = "".join(f"{sha}  {relative}\n" for relative, sha in rows)
    _write_new_text(Path(path), text)
    return artifact_identity(path)


def verify_checksums(root: str | Path, checksum_path: str | Path) -> dict[str, Any]:
    base = Path(root).resolve(strict=True)
    checksum_file = Path(checksum_path).resolve(strict=True)
    failures: list[dict[str, str]] = []
    verified = 0
    for raw_line in checksum_file.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        try:
            expected, relative = raw_line.split("  ", 1)
        except ValueError as exc:
            raise ValueError(f"invalid checksum line: {raw_line!r}") from exc
        target = (base / relative).resolve(strict=True)
        try:
            target.relative_to(base)
        except ValueError as exc:
            raise ValueError(f"checksum path escapes export root: {relative}") from exc
        actual = sha256_file(target)
        verified += 1
        if actual != expected:
            failures.append(
                {
                    "path": relative,
                    "expected": expected,
                    "actual": actual,
                }
            )
    return {
        "schema": "sha256_verification_v1",
        "checksum_path": str(checksum_file),
        "verified_file_count": int(verified),
        "failures": failures,
        "ok": not failures,
    }


def _write_new_text(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(f"immutable artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            raise FileExistsError(
                f"immutable artifact appeared during write: {path}"
            ) from None
    finally:
        if temporary.exists():
            temporary.unlink()
