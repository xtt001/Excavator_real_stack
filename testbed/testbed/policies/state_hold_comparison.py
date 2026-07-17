"""Compare two single-demo-target state-hold reports."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

AXIS_NAMES = ("swing", "boom", "stick", "bucket")
ANCHOR_GROUPS = ("startup", "mid_cycle")
STATE_HOLD_STATUSES = ("demo_target_reproduced", "demo_target_not_reproduced")
CLASSIFICATIONS = (
    "both_reproduced_demo_target",
    "reference_reproduced_candidate_not_reproduced",
    "reference_not_reproduced_candidate_reproduced",
    "both_not_reproduced_demo_target",
)

_IDENTITY_FIELDS = (
    "episode_id",
    "anchor_step",
    "axis_index",
    "axis",
    "direction",
)
_MATCHED_INVARIANT_FIELDS = (
    "anchor_group",
    "deadzone_threshold",
    "expert_action",
    "hold_horizon_steps",
)
_OUTPUT_NAMES = {
    "rows_csv": "state_hold_comparison_rows.csv",
    "rows_json": "state_hold_comparison_rows.json",
    "summary": "state_hold_comparison_summary.json",
}


def load_state_hold_rows(path: str | Path) -> list[dict[str, Any]]:
    """Load and strictly validate one ``state_hold_anchors.jsonl`` file."""

    source = Path(path)
    rows: list[dict[str, Any]] = []
    seen: dict[tuple[Any, ...], int] = {}
    with source.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                raise ValueError(f"{source}:{line_number}: blank JSONL line")
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{source}:{line_number}: invalid JSON: {exc.msg}"
                ) from exc
            if not isinstance(parsed, dict):
                raise ValueError(f"{source}:{line_number}: row must be an object")
            row = dict(parsed)
            _validate_row(row, location=f"{source}:{line_number}")
            key = _anchor_key(row)
            if key in seen:
                raise ValueError(
                    f"{source}:{line_number}: duplicate anchor key {key!r}; "
                    f"first seen on line {seen[key]}"
                )
            seen[key] = line_number
            rows.append(row)
    if not rows:
        raise ValueError(f"{source}: state-hold anchor file is empty")
    return rows


def compare_state_hold_rows(
    *,
    reference_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    reference_label: str,
    candidate_label: str,
) -> list[dict[str, Any]]:
    """Match two anchor sets and label every liveness transition."""

    reference_label, candidate_label = _validate_labels(
        reference_label, candidate_label
    )
    reference = _validated_row_map(reference_rows, source="reference rows")
    candidate = _validated_row_map(candidate_rows, source="candidate rows")
    reference_keys = set(reference)
    candidate_keys = set(candidate)
    if reference_keys != candidate_keys:
        missing_candidate = sorted(reference_keys - candidate_keys)
        missing_reference = sorted(candidate_keys - reference_keys)
        raise ValueError(
            "anchor key sets differ: "
            f"missing from candidate={missing_candidate!r}; "
            f"missing from reference={missing_reference!r}"
        )

    rows: list[dict[str, Any]] = []
    for key in sorted(reference_keys):
        reference_row = reference[key]
        candidate_row = candidate[key]
        for field in _MATCHED_INVARIANT_FIELDS:
            if reference_row[field] != candidate_row[field]:
                raise ValueError(
                    f"anchor {key!r} invariant {field!r} differs: "
                    f"reference={reference_row[field]!r}, "
                    f"candidate={candidate_row[field]!r}"
                )

        reference_reproduced = (
            reference_row["state_hold_status"] == "demo_target_reproduced"
        )
        candidate_reproduced = (
            candidate_row["state_hold_status"] == "demo_target_reproduced"
        )
        classification = _classification(
            reference_reproduced=reference_reproduced,
            candidate_reproduced=candidate_reproduced,
        )
        rows.append(
            {
                **{field: reference_row[field] for field in _IDENTITY_FIELDS},
                **{field: reference_row[field] for field in _MATCHED_INVARIANT_FIELDS},
                "reference_label": reference_label,
                "candidate_label": candidate_label,
                "reference_state_hold_status": reference_row["state_hold_status"],
                "reference_demo_target_not_reproduced": reference_row[
                    "state_hold_demo_target_not_reproduced"
                ],
                "reference_demo_target_reproduction_delay_ticks": reference_row[
                    "state_hold_demo_target_reproduction_delay_ticks"
                ],
                "reference_demo_target_reproduction_hidden_by_teacher_forcing": reference_row[
                    "demo_target_reproduction_hidden_by_teacher_forcing"
                ],
                "candidate_state_hold_status": candidate_row["state_hold_status"],
                "candidate_demo_target_not_reproduced": candidate_row[
                    "state_hold_demo_target_not_reproduced"
                ],
                "candidate_demo_target_reproduction_delay_ticks": candidate_row[
                    "state_hold_demo_target_reproduction_delay_ticks"
                ],
                "candidate_demo_target_reproduction_hidden_by_teacher_forcing": candidate_row[
                    "demo_target_reproduction_hidden_by_teacher_forcing"
                ],
                "classification": classification,
                "candidate_changed_reproduction_to_nonreproduction": (
                    classification == "reference_reproduced_candidate_not_reproduced"
                ),
                "candidate_changed_nonreproduction_to_reproduction": (
                    classification == "reference_not_reproduced_candidate_reproduced"
                ),
            }
        )
    return rows


def aggregate_state_hold_comparison(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate comparison rows for overall, startup, and mid-cycle anchors."""

    summaries: list[dict[str, Any]] = []
    grouped = (
        ("overall", list(rows)),
        ("startup", [row for row in rows if row["anchor_group"] == "startup"]),
        (
            "mid_cycle",
            [row for row in rows if row["anchor_group"] == "mid_cycle"],
        ),
    )
    for group_name, group_rows in grouped:
        total = len(group_rows)
        counts = {
            classification: sum(
                row["classification"] == classification for row in group_rows
            )
            for classification in CLASSIFICATIONS
        }
        summaries.append(
            {
                "group": group_name,
                "anchors_total": total,
                "reference_demo_target_reproduced_anchors": sum(
                    row["reference_state_hold_status"] == "demo_target_reproduced"
                    for row in group_rows
                ),
                "reference_demo_target_not_reproduced_anchors": sum(
                    bool(row["reference_demo_target_not_reproduced"])
                    for row in group_rows
                ),
                "reference_demo_target_reproduction_hidden_by_teacher_forcing_anchors": sum(
                    bool(
                        row[
                            "reference_demo_target_reproduction_hidden_by_teacher_forcing"
                        ]
                    )
                    for row in group_rows
                ),
                "candidate_demo_target_reproduced_anchors": sum(
                    row["candidate_state_hold_status"] == "demo_target_reproduced"
                    for row in group_rows
                ),
                "candidate_demo_target_not_reproduced_anchors": sum(
                    bool(row["candidate_demo_target_not_reproduced"])
                    for row in group_rows
                ),
                "candidate_demo_target_reproduction_hidden_by_teacher_forcing_anchors": sum(
                    bool(
                        row[
                            "candidate_demo_target_reproduction_hidden_by_teacher_forcing"
                        ]
                    )
                    for row in group_rows
                ),
                "both_reproduced_demo_target_anchors": counts[
                    "both_reproduced_demo_target"
                ],
                "reference_reproduced_candidate_not_reproduced_anchors": counts[
                    "reference_reproduced_candidate_not_reproduced"
                ],
                "reference_not_reproduced_candidate_reproduced_anchors": counts[
                    "reference_not_reproduced_candidate_reproduced"
                ],
                "both_not_reproduced_demo_target_anchors": counts[
                    "both_not_reproduced_demo_target"
                ],
                "candidate_demo_target_reproduction_rate": _rate(
                    sum(
                        row["candidate_state_hold_status"] == "demo_target_reproduced"
                        for row in group_rows
                    ),
                    total,
                ),
            }
        )
    return summaries


def compare_state_hold_files(
    *,
    reference_path: str | Path,
    candidate_path: str | Path,
    reference_label: str,
    candidate_label: str,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Compare two files and write non-overwriting review artifacts."""

    reference_source = Path(reference_path).resolve()
    candidate_source = Path(candidate_path).resolve()
    reference_rows = load_state_hold_rows(reference_source)
    candidate_rows = load_state_hold_rows(candidate_source)
    rows = compare_state_hold_rows(
        reference_rows=reference_rows,
        candidate_rows=candidate_rows,
        reference_label=reference_label,
        candidate_label=candidate_label,
    )
    reference_label, candidate_label = _validate_labels(
        reference_label, candidate_label
    )
    provenance = {
        "reference": {
            "label": reference_label,
            "source_path": str(reference_source),
            "sha256": _sha256(reference_source),
            "anchor_count": len(reference_rows),
        },
        "candidate": {
            "label": candidate_label,
            "source_path": str(candidate_source),
            "sha256": _sha256(candidate_source),
            "anchor_count": len(candidate_rows),
        },
    }
    return write_state_hold_comparison(
        output_dir=output_dir,
        rows=rows,
        provenance=provenance,
    )


def write_state_hold_comparison(
    *,
    output_dir: str | Path,
    rows: Sequence[Mapping[str, Any]],
    provenance: Mapping[str, Any],
) -> dict[str, Path]:
    """Write CSV and JSON artifacts, refusing to replace any target file."""

    output = Path(output_dir)
    paths = {key: output / name for key, name in _OUTPUT_NAMES.items()}
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite state-hold comparison artifact(s): "
            + ", ".join(existing)
        )

    safe_rows = [dict(row) for row in rows]
    aggregates = aggregate_state_hold_comparison(safe_rows)
    classification_counts = {
        classification: sum(
            row["classification"] == classification for row in safe_rows
        )
        for classification in CLASSIFICATIONS
    }
    rows_document = {
        "schema_version": 2,
        "comparison": dict(provenance),
        "rows": safe_rows,
    }
    summary_document = {
        "schema_version": 2,
        "comparison": dict(provenance),
        "capability_boundaries": {
            "comparison_target": "single_demo_axis_direction_reproduction",
            "correctness_estimable": False,
            "task_support_estimable": False,
            "physical_validity_estimable": False,
        },
        "classification_counts": classification_counts,
        "aggregate": aggregates,
    }

    output.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in safe_rows for key in row))
    with paths["rows_csv"].open("x", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(safe_rows)
    with paths["rows_json"].open("x", encoding="utf-8") as file:
        file.write(json.dumps(rows_document, indent=2, ensure_ascii=False) + "\n")
    with paths["summary"].open("x", encoding="utf-8") as file:
        file.write(json.dumps(summary_document, indent=2, ensure_ascii=False) + "\n")
    return paths


def _validated_row_map(
    rows: Sequence[Mapping[str, Any]], *, source: str
) -> dict[tuple[Any, ...], dict[str, Any]]:
    if not rows:
        raise ValueError(f"{source}: anchor rows must not be empty")
    result: dict[tuple[Any, ...], dict[str, Any]] = {}
    for index, original in enumerate(rows):
        if not isinstance(original, Mapping):
            raise ValueError(f"{source}[{index}]: row must be a mapping")
        row = dict(original)
        _validate_row(row, location=f"{source}[{index}]")
        key = _anchor_key(row)
        if key in result:
            raise ValueError(f"{source}[{index}]: duplicate anchor key {key!r}")
        result[key] = row
    return result


def _validate_row(row: Mapping[str, Any], *, location: str) -> None:
    required = (
        set(_IDENTITY_FIELDS)
        | set(_MATCHED_INVARIANT_FIELDS)
        | {
            "state_hold_status",
            "state_hold_demo_target_not_reproduced",
            "state_hold_demo_target_reproduction_delay_ticks",
            "demo_target_reproduction_hidden_by_teacher_forcing",
            "teacher_forced_status",
        }
    )
    missing = sorted(required - set(row))
    if missing:
        raise ValueError(f"{location}: missing required fields {missing!r}")

    episode_id = row["episode_id"]
    if not isinstance(episode_id, str) or not episode_id.strip():
        raise ValueError(f"{location}: episode_id must be a non-empty string")
    anchor_step = _strict_int(
        row["anchor_step"], field="anchor_step", location=location
    )
    if anchor_step < 0:
        raise ValueError(f"{location}: anchor_step must be non-negative")
    axis_index = _strict_int(row["axis_index"], field="axis_index", location=location)
    if axis_index not in range(len(AXIS_NAMES)):
        raise ValueError(f"{location}: axis_index must be in [0, 3]")
    if row["axis"] != AXIS_NAMES[axis_index]:
        raise ValueError(
            f"{location}: axis {row['axis']!r} does not match "
            f"axis_index {axis_index} ({AXIS_NAMES[axis_index]!r})"
        )
    direction = row["direction"]
    if direction not in ("pos", "neg"):
        raise ValueError(f"{location}: direction must be 'pos' or 'neg'")
    if row["anchor_group"] not in ANCHOR_GROUPS:
        raise ValueError(f"{location}: anchor_group must be one of {ANCHOR_GROUPS!r}")
    threshold = _finite_float(
        row["deadzone_threshold"], field="deadzone_threshold", location=location
    )
    if threshold < 0.0:
        raise ValueError(f"{location}: deadzone_threshold must be non-negative")
    expert_action = _finite_float(
        row["expert_action"], field="expert_action", location=location
    )
    if direction == "pos" and expert_action < threshold:
        raise ValueError(
            f"{location}: positive expert_action does not cross its deadzone"
        )
    if direction == "neg" and expert_action > -threshold:
        raise ValueError(
            f"{location}: negative expert_action does not cross its deadzone"
        )
    horizon = _strict_int(
        row["hold_horizon_steps"], field="hold_horizon_steps", location=location
    )
    if horizon <= 0:
        raise ValueError(f"{location}: hold_horizon_steps must be positive")

    status = row["state_hold_status"]
    if status not in STATE_HOLD_STATUSES:
        raise ValueError(
            f"{location}: state_hold_status must be one of {STATE_HOLD_STATUSES!r}"
        )
    not_reproduced = row["state_hold_demo_target_not_reproduced"]
    if not isinstance(not_reproduced, bool):
        raise ValueError(
            f"{location}: state_hold_demo_target_not_reproduced must be boolean"
        )
    expected_not_reproduced = status == "demo_target_not_reproduced"
    if not_reproduced != expected_not_reproduced:
        raise ValueError(
            f"{location}: state_hold_demo_target_not_reproduced is inconsistent "
            "with state_hold_status"
        )
    delay = row["state_hold_demo_target_reproduction_delay_ticks"]
    if status == "demo_target_reproduced":
        delay_value = _strict_int(
            delay,
            field="state_hold_demo_target_reproduction_delay_ticks",
            location=location,
        )
        if not 0 <= delay_value < horizon:
            raise ValueError(
                f"{location}: reproduction delay must be in [0, hold_horizon_steps)"
            )
    elif delay is not None:
        raise ValueError(f"{location}: non-reproduced demo target delay must be null")

    hidden = row["demo_target_reproduction_hidden_by_teacher_forcing"]
    if not isinstance(hidden, bool):
        raise ValueError(
            f"{location}: demo_target_reproduction_hidden_by_teacher_forcing "
            "must be boolean"
        )
    teacher_status = row["teacher_forced_status"]
    if teacher_status not in STATE_HOLD_STATUSES:
        raise ValueError(
            f"{location}: teacher_forced_status must be one of {STATE_HOLD_STATUSES!r}"
        )
    expected_hidden = (
        teacher_status == "demo_target_reproduced" and expected_not_reproduced
    )
    if hidden != expected_hidden:
        raise ValueError(
            f"{location}: demo target reproduction hidden-by-teacher-forcing "
            "is inconsistent with statuses"
        )


def _anchor_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(row[field] for field in _IDENTITY_FIELDS)


def _classification(*, reference_reproduced: bool, candidate_reproduced: bool) -> str:
    if reference_reproduced and candidate_reproduced:
        return "both_reproduced_demo_target"
    if reference_reproduced:
        return "reference_reproduced_candidate_not_reproduced"
    if candidate_reproduced:
        return "reference_not_reproduced_candidate_reproduced"
    return "both_not_reproduced_demo_target"


def _validate_labels(reference_label: str, candidate_label: str) -> tuple[str, str]:
    reference = str(reference_label).strip()
    candidate = str(candidate_label).strip()
    if not reference or not candidate:
        raise ValueError("reference and candidate labels must be non-empty")
    if reference == candidate:
        raise ValueError("reference and candidate labels must be distinct")
    return reference, candidate


def _strict_int(value: Any, *, field: str, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{location}: {field} must be an integer")
    return value


def _finite_float(value: Any, *, field: str, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{location}: {field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{location}: {field} must be finite")
    return result


def _rate(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
