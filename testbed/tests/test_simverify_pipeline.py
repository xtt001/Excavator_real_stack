from __future__ import annotations

import pytest

from testbed.simverify.pipeline import (
    _annotation_bootstrap_gate_report,
    _require_bootstrap_stability,
)


def _gate_inputs(
    *,
    sector_boundary_width: float,
) -> tuple[dict[str, object], ...]:
    numeric = {
        "dump_release": {
            "swing_cluster_centers": [0.0, 1.0],
        }
    }
    numeric_bootstrap = {
        "requested_samples": 100,
        "failed_samples": 0,
        "dump_swing_threshold": {
            "p02_5": 0.45,
            "p97_5": 0.55,
        },
    }
    sector = {
        "cluster_centers_low_to_high": [0.0, 1.0, 2.0],
    }
    sector_bootstrap = {
        "boundaries": {
            "p02_5": [0.5, 1.5],
            "p97_5": [
                0.5 + sector_boundary_width,
                1.5 + sector_boundary_width,
            ],
        }
    }
    return numeric, numeric_bootstrap, sector, sector_bootstrap


def test_annotation_gate_report_preserves_the_frozen_failure_boundary() -> None:
    inputs = _gate_inputs(sector_boundary_width=0.3)

    with pytest.raises(RuntimeError, match="sector boundary bootstrap"):
        _require_bootstrap_stability(*inputs)
    report = _annotation_bootstrap_gate_report(*inputs)

    assert report["passed"] is False
    assert report["failure_reason"] == "sector boundary bootstrap is unstable"
    assert report["m1_import_smoke_authorized"] is False
    assert report["criteria"][
        "maximum_ci95_width_fraction_of_cluster_gap"
    ] == pytest.approx(0.25)
    assert report["sector_thresholds"][
        "ci95_width_to_minimum_cluster_gap"
    ] == pytest.approx([0.3, 0.3])


def test_annotation_gate_report_matches_a_passing_gate() -> None:
    inputs = _gate_inputs(sector_boundary_width=0.2)

    _require_bootstrap_stability(*inputs)
    report = _annotation_bootstrap_gate_report(*inputs)

    assert report["passed"] is True
    assert report["failure_reason"] is None
    assert report["m1_import_smoke_authorized"] is True
