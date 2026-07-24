from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from testbed.simverify.annotations import EpisodeSignals
from testbed.simverify.event_selector import match_event_interval
from testbed.simverify.pipeline import (
    _annotation_bootstrap_gate_report,
    _extract_cycle_features,
    _rebuild_sector_evidence_after_visual_selection,
    _require_bootstrap_stability,
    run_m0_pipeline,
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
        "requested_samples": 100,
        "failed_samples": 0,
        "boundaries": {
            "p02_5": [0.5, 1.5],
            "p97_5": [
                0.5 + sector_boundary_width,
                1.5 + sector_boundary_width,
            ],
        },
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


def test_m0_rejects_non_frozen_jpeg_quality_before_source_access(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="frozen.*95"):
        run_m0_pipeline(
            source_root=tmp_path / "missing_source",
            output_root=tmp_path / "missing_output",
            repo_root=tmp_path / "missing_repo",
            jpeg_quality=90,
        )


class _RecordingExtractor:
    def __init__(self) -> None:
        self.eye_calls: list[list[int]] = []
        self.stick_calls: list[list[int]] = []

    def extract_hdf5_eye_pair(
        self,
        _path: Path,
        indices: list[int],
    ) -> np.ndarray:
        self.eye_calls.append(list(indices))
        return np.asarray(
            [[float(index + 1), 1.0] for index in indices],
            dtype=np.float32,
        )

    def extract_hdf5_stick_pair(
        self,
        _path: Path,
        indices: list[int],
    ) -> np.ndarray:
        self.stick_calls.append(list(indices))
        return np.asarray(
            [[1.0, float(index + 1)] for index in indices],
            dtype=np.float32,
        )


def test_cycle_feature_extraction_reads_every_unique_interval_frame() -> None:
    extractor = _RecordingExtractor()
    cycles = {
        7: [
            {
                "observable_events": {
                    "ready_start": {
                        "interval": [2, 5],
                        "representative_step": 3,
                    },
                    "dig_entry_proxy": {
                        "interval": [4, 7],
                        "representative_step": 4,
                    },
                    "ready_end": None,
                }
            }
        ]
    }

    cache = _extract_cycle_features(
        extractor,
        {7: Path("/does/not/need/to/exist.hdf5")},
        cycles,
        episode_ids=[7],
        episode_lengths={7: 8},
        chunk_rows=64,
    )

    assert extractor.eye_calls == [[1, 2, 3, 4, 5, 6, 7]]
    assert extractor.stick_calls == [[1, 2, 3, 4, 5, 6, 7]]
    assert sorted(step for episode, step in cache if episode == 7) == [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
    ]


def _selector(
    *,
    offset_low: int = -2,
    offset_high: int = 2,
) -> dict[str, object]:
    phases = (
        "ready",
        "dig_entry_proxy",
        "carry_transition_proxy",
        "dump_start_proxy",
        "dump_end_proxy",
    )
    prototypes = {
        phase: {
            "eye": np.asarray([1.0, 0.0], dtype=np.float32),
            "stick": np.asarray([1.0, 0.0], dtype=np.float32),
        }
        for phase in phases
    }
    prototypes["ready"] = {
        "eye": np.asarray([0.0, 1.0], dtype=np.float32),
        "stick": np.asarray([0.0, 1.0], dtype=np.float32),
    }
    return {
        "prototypes": prototypes,
        "support_thresholds": {phase: {"eye": 0.8, "stick": 0.8} for phase in phases},
        "change_thresholds": {
            "ready": {"stick": 0.5},
            "dig_entry_proxy": {"eye": 0.0, "stick": 0.0},
            "carry_transition_proxy": {"stick": 0.0},
            "dump_start_proxy": {"eye": 0.0, "stick": 0.0},
            "dump_end_proxy": {"eye": 0.0, "stick": 0.0},
        },
        "offset_bounds": {
            phase: {
                "minimum_signed_offset_steps": offset_low,
                "maximum_signed_offset_steps": offset_high,
            }
            for phase in phases
        },
    }


def test_interval_match_uses_separate_roles_and_entering_halo() -> None:
    features = {
        (7, 9): {
            "eye": np.asarray([0.0, 1.0]),
            "stick": np.asarray([0.0, 1.0]),
        },
        (7, 10): {
            "eye": np.asarray([1.0, 0.0]),
            "stick": np.asarray([1.0, 0.0]),
        },
        (7, 11): {
            "eye": np.asarray([1.0, 0.0]),
            "stick": np.asarray([1.0, 0.0]),
        },
        (7, 12): {
            "eye": np.asarray([1.0, 0.0]),
            "stick": np.asarray([1.0, 0.0]),
        },
    }
    event = {
        "event_key": "episode_7:cycle_0:dig_entry_proxy",
        "episode_id": 7,
        "phase": "dig_entry_proxy",
        "interval": [10, 13],
        "numeric_representative_step": 10,
    }

    selector = _selector()
    selector["prototypes"]["dig_entry_proxy"] = {
        "eye": np.asarray([0.9, 0.4358899], dtype=np.float32),
        "stick": np.asarray([0.9, 0.4358899], dtype=np.float32),
    }
    match = match_event_interval(
        event,
        features=features,
        selector=selector,
    )

    assert match["status"] == "confirmed"
    assert match["representative_step"] == 10
    assert match["selected"]["role_metrics"]["eye"]["prediction"] == (
        "carry_transition_proxy"
    )
    assert match["selected"]["role_metrics"]["stick"][
        "expected_similarity"
    ] == pytest.approx(0.9)


def test_interval_match_treats_top1_as_diagnostic_only() -> None:
    features = {
        (7, 9): {
            "eye": np.asarray([1.0, 0.0]),
            "stick": np.asarray([1.0, 0.0]),
        },
        (7, 10): {
            "eye": np.asarray([1.0, 0.0]),
            "stick": np.asarray([1.0, 0.0]),
        },
        (7, 11): {
            "eye": np.asarray([1.0, 0.0]),
            "stick": np.asarray([1.0, 0.0]),
        },
    }
    event = {
        "event_key": "episode_7:cycle_0:dig_entry_proxy",
        "episode_id": 7,
        "phase": "dig_entry_proxy",
        "interval": [10, 11],
        "numeric_representative_step": 10,
    }

    selector = _selector()
    selector["prototypes"]["dig_entry_proxy"] = {
        "eye": np.asarray([0.9, 0.4358899], dtype=np.float32),
        "stick": np.asarray([0.9, 0.4358899], dtype=np.float32),
    }
    match = match_event_interval(
        event,
        features=features,
        selector=selector,
    )

    assert match["status"] == "confirmed"
    assert match["acceptance_rule"].endswith("top1_and_margin_diagnostic_only")


def test_rebuild_sector_evidence_refreshes_adjacent_next_observation() -> None:
    qpos = np.zeros((30, 4), dtype=np.float32)
    qpos[4, 0] = 0.2
    qpos[14, 0] = 0.8
    signals = {
        7: EpisodeSignals(
            episode_id=7,
            step_id=np.arange(30, dtype=np.int64),
            qpos=qpos,
            qvel=np.zeros((30, 4), dtype=np.float32),
            action=np.zeros((30, 4), dtype=np.float32),
            dt=0.02,
        )
    }

    def cycle(cycle_id: int, representative: int) -> dict[str, object]:
        return {
            "cycle_id": cycle_id,
            "observable_events": {
                "dig_entry_proxy": {
                    "interval": [representative - 1, representative + 2],
                    "numeric_representative_step": representative - 1,
                    "representative_step": representative,
                    "visual_interval_selection": {"status": "confirmed"},
                }
            },
            "sector_validity": {
                "current": {
                    "valid": True,
                    "source_cycle_id": cycle_id,
                    "reason_codes": [],
                },
                "next": None,
            },
            "sector_observations": {"current": None, "next": None},
            "numeric_sector_evidence": {
                "current_swing_qpos": None,
                "next_swing_qpos": None,
            },
            "verification": {
                "visual_event_order_valid": True,
                "visual_current_sector_order_valid": True,
            },
        }

    cycles = {7: [cycle(0, 4), cycle(1, 14)]}

    _rebuild_sector_evidence_after_visual_selection(
        cycles,
        signals,
        episode_ids=[7],
    )

    assert cycles[7][0]["numeric_sector_evidence"][
        "current_swing_qpos"
    ] == pytest.approx(0.2)
    assert cycles[7][0]["numeric_sector_evidence"]["next_swing_qpos"] == pytest.approx(
        0.8
    )
    assert cycles[7][0]["sector_observations"]["next"]["representative_step"] == 14
