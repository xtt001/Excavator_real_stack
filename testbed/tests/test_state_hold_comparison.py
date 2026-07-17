from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from testbed.policies.state_hold_comparison import (
    aggregate_state_hold_comparison,
    compare_state_hold_files,
    compare_state_hold_rows,
    load_state_hold_rows,
)


def _row(
    *,
    step: int,
    axis_index: int,
    direction: str = "pos",
    group: str = "mid_cycle",
    status: str = "demo_target_reproduced",
    hidden: bool = False,
) -> dict[str, Any]:
    threshold = 0.2
    return {
        "episode_id": "episode_101",
        "anchor_step": step,
        "anchor_group": group,
        "axis_index": axis_index,
        "axis": ("swing", "boom", "stick", "bucket")[axis_index],
        "direction": direction,
        "deadzone_threshold": threshold,
        "expert_action": 0.5 if direction == "pos" else -0.5,
        "hold_horizon_steps": 8,
        "teacher_forced_status": "demo_target_reproduced" if hidden else status,
        "state_hold_status": status,
        "state_hold_demo_target_not_reproduced": (
            status == "demo_target_not_reproduced"
        ),
        "state_hold_demo_target_reproduction_delay_ticks": (
            2 if status == "demo_target_reproduced" else None
        ),
        "demo_target_reproduction_hidden_by_teacher_forcing": hidden,
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_comparison_classifies_all_transitions_and_aggregates_groups() -> None:
    reference = [
        _row(step=0, axis_index=0, group="startup", status="demo_target_reproduced"),
        _row(step=2, axis_index=1, status="demo_target_reproduced"),
        _row(step=4, axis_index=2, status="demo_target_not_reproduced", hidden=True),
        _row(step=6, axis_index=3, status="demo_target_not_reproduced"),
    ]
    candidate = [
        _row(step=0, axis_index=0, group="startup", status="demo_target_reproduced"),
        _row(step=2, axis_index=1, status="demo_target_not_reproduced", hidden=True),
        _row(step=4, axis_index=2, status="demo_target_reproduced"),
        _row(step=6, axis_index=3, status="demo_target_not_reproduced"),
    ]

    rows = compare_state_hold_rows(
        reference_rows=reference,
        candidate_rows=candidate,
        reference_label="baseline",
        candidate_label="e52",
    )

    assert [row["classification"] for row in rows] == [
        "both_reproduced_demo_target",
        "reference_reproduced_candidate_not_reproduced",
        "reference_not_reproduced_candidate_reproduced",
        "both_not_reproduced_demo_target",
    ]
    aggregate = aggregate_state_hold_comparison(rows)
    assert [row["group"] for row in aggregate] == [
        "overall",
        "startup",
        "mid_cycle",
    ]
    overall = aggregate[0]
    assert overall["candidate_demo_target_reproduced_anchors"] == 2
    assert overall["candidate_demo_target_not_reproduced_anchors"] == 2
    assert overall[
        "candidate_demo_target_reproduction_hidden_by_teacher_forcing_anchors"
    ] == 1
    assert overall["reference_reproduced_candidate_not_reproduced_anchors"] == 1
    assert overall["reference_not_reproduced_candidate_reproduced_anchors"] == 1
    assert aggregate[1]["anchors_total"] == 1
    assert aggregate[2]["anchors_total"] == 3


def test_loader_rejects_duplicate_anchor_key(tmp_path: Path) -> None:
    source = tmp_path / "state_hold_anchors.jsonl"
    duplicate = _row(step=0, axis_index=0, group="startup")
    _write_jsonl(source, [duplicate, duplicate])

    with pytest.raises(ValueError, match="duplicate anchor key"):
        load_state_hold_rows(source)


def test_comparison_rejects_anchor_key_set_drift() -> None:
    reference = [_row(step=0, axis_index=0, group="startup")]
    candidate = [_row(step=1, axis_index=0, group="startup")]

    with pytest.raises(ValueError, match="anchor key sets differ"):
        compare_state_hold_rows(
            reference_rows=reference,
            candidate_rows=candidate,
            reference_label="reference",
            candidate_label="candidate",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("anchor_group", "startup"),
        ("deadzone_threshold", 0.25),
        ("expert_action", 0.6),
        ("hold_horizon_steps", 9),
    ],
)
def test_comparison_rejects_matched_anchor_invariant_drift(
    field: str, value: Any
) -> None:
    reference = _row(step=2, axis_index=1)
    candidate = _row(step=2, axis_index=1)
    candidate[field] = value

    with pytest.raises(ValueError, match=f"invariant {field!r} differs"):
        compare_state_hold_rows(
            reference_rows=[reference],
            candidate_rows=[candidate],
            reference_label="reference",
            candidate_label="candidate",
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("axis", "stick", "does not match axis_index"),
        (
            "state_hold_demo_target_not_reproduced",
            True,
            "inconsistent with state_hold_status",
        ),
        (
            "state_hold_demo_target_reproduction_delay_ticks",
            8,
            "reproduction delay must be",
        ),
        (
            "demo_target_reproduction_hidden_by_teacher_forcing",
            True,
            "inconsistent with statuses",
        ),
    ],
)
def test_loader_rejects_internal_invariant_violations(
    tmp_path: Path, field: str, value: Any, message: str
) -> None:
    source = tmp_path / "state_hold_anchors.jsonl"
    row = _row(step=0, axis_index=0, group="startup")
    row[field] = value
    _write_jsonl(source, [row])

    with pytest.raises(ValueError, match=message):
        load_state_hold_rows(source)


def test_file_comparison_writes_labeled_provenance_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    reference_path = tmp_path / "reference.jsonl"
    candidate_path = tmp_path / "candidate.jsonl"
    _write_jsonl(
        reference_path,
        [_row(step=0, axis_index=0, group="startup", status="demo_target_reproduced")],
    )
    _write_jsonl(
        candidate_path,
        [_row(step=0, axis_index=0, group="startup", status="demo_target_not_reproduced")],
    )
    output_dir = tmp_path / "comparison"

    paths = compare_state_hold_files(
        reference_path=reference_path,
        candidate_path=candidate_path,
        reference_label="baseline",
        candidate_label="e52",
        output_dir=output_dir,
    )

    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    rows_document = json.loads(paths["rows_json"].read_text(encoding="utf-8"))
    with paths["rows_csv"].open(encoding="utf-8", newline="") as file:
        csv_rows = list(csv.DictReader(file))
    assert summary["comparison"]["reference"]["label"] == "baseline"
    assert summary["comparison"]["candidate"]["label"] == "e52"
    assert summary["comparison"]["reference"]["source_path"] == str(
        reference_path.resolve()
    )
    assert (
        summary["comparison"]["reference"]["sha256"]
        == hashlib.sha256(reference_path.read_bytes()).hexdigest()
    )
    assert summary["classification_counts"] == {
        "both_reproduced_demo_target": 0,
        "reference_reproduced_candidate_not_reproduced": 1,
        "reference_not_reproduced_candidate_reproduced": 0,
        "both_not_reproduced_demo_target": 0,
    }
    assert rows_document["rows"][0]["reference_label"] == "baseline"
    assert csv_rows[0]["candidate_label"] == "e52"

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        compare_state_hold_files(
            reference_path=reference_path,
            candidate_path=candidate_path,
            reference_label="baseline",
            candidate_label="e52",
            output_dir=output_dir,
        )
