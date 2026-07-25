from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from testbed.simverify.annotations import EpisodeSignals
from testbed.simverify.event_selector import match_event_interval
from testbed.simverify.pipeline import (
    _annotation_bootstrap_gate_report,
    _annotation_bootstrap_gate_report_v2,
    _calibrate_interval_confirmation_reliability,
    _extract_cycle_features,
    _rebuild_sector_evidence_after_visual_selection,
    _require_bootstrap_stability,
    _require_sector_visual_identifiability,
    _resample_gate_report_v2,
    _transition_support_gate_report_v2,
    _wilson_lower_bound,
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


def test_v2_boundary_gate_uses_center_separation_not_arbitrary_gap_ratio() -> None:
    numeric = {
        "dump_release": {
            "swing_cluster_centers": [0.0, 1.0],
        }
    }
    numeric_bootstrap = {
        "requested_samples": 256,
        "failed_samples": 0,
        "dump_swing_cluster_centers": {
            "p02_5": [-0.1, 0.9],
            "p97_5": [0.1, 1.1],
        },
        "dump_swing_threshold": {
            "p02_5": 0.3,
            "p97_5": 0.7,
        },
    }
    sector = {
        "cluster_centers_low_to_high": [0.0, 1.0, 2.0],
    }
    sector_bootstrap = {
        "requested_samples": 256,
        "failed_samples": 0,
        "cluster_centers": {
            "p02_5": [-0.1, 0.9, 1.9],
            "p97_5": [0.1, 1.1, 2.1],
        },
        "boundaries": {
            "p02_5": [0.3, 1.3],
            "p97_5": [0.7, 1.7],
        },
    }

    report = _annotation_bootstrap_gate_report_v2(
        numeric,
        numeric_bootstrap,
        sector,
        sector_bootstrap,
    )

    assert report["passed"] is True
    assert report["m1_import_smoke_authorized"] is True
    assert report["diagnostic_only_not_gate"][
        "dump_ci95_width_to_cluster_gap"
    ] == pytest.approx(0.4)
    assert report["dump_boundary_separation"]["passed"] is True


def test_v2_boundary_gate_rejects_ci_that_reaches_a_cluster_center() -> None:
    numeric = {
        "dump_release": {
            "swing_cluster_centers": [0.0, 1.0],
        }
    }
    numeric_bootstrap = {
        "requested_samples": 256,
        "failed_samples": 0,
        "dump_swing_cluster_centers": {
            "p02_5": [-0.1, 0.9],
            "p97_5": [0.1, 1.1],
        },
        "dump_swing_threshold": {
            "p02_5": 0.05,
            "p97_5": 0.7,
        },
    }
    sector = {
        "cluster_centers_low_to_high": [0.0, 1.0, 2.0],
    }
    sector_bootstrap = {
        "requested_samples": 256,
        "failed_samples": 0,
        "cluster_centers": {
            "p02_5": [-0.1, 0.9, 1.9],
            "p97_5": [0.1, 1.1, 2.1],
        },
        "boundaries": {
            "p02_5": [0.3, 1.3],
            "p97_5": [0.7, 1.7],
        },
    }

    report = _annotation_bootstrap_gate_report_v2(
        numeric,
        numeric_bootstrap,
        sector,
        sector_bootstrap,
    )

    assert report["passed"] is False
    assert (
        "dump_boundary_ci_not_strictly_between_adjacent_center_cis"
        in report["failure_reasons"]
    )


def test_interval_reliability_threshold_is_derived_from_3x3_support() -> None:
    sequence = ("left", "left", "center", "left", "right", "center",
                "center", "right", "right", "left")
    qpos = {"left": -1.0, "center": 0.0, "right": 1.0}
    cycles: dict[int, list[dict[str, object]]] = {}
    point_cycles: dict[str, dict[str, object]] = {}
    point_events: dict[str, dict[str, object]] = {}
    outer_events: dict[str, dict[str, object]] = {}
    event_names = (
        "ready_start",
        "dig_entry_proxy",
        "carry_transition_proxy",
        "dump_start_proxy",
        "dump_end_proxy",
        "ready_end",
    )
    for episode_id in (1, 2):
        cycles[episode_id] = []
        for cycle_id, sector in enumerate(sequence):
            cycles[episode_id].append(
                {
                    "cycle_id": cycle_id,
                    "sector_validity": {
                        "current": {"valid": True},
                    },
                    "numeric_sector_evidence": {
                        "current_swing_qpos": qpos[sector],
                    },
                }
            )
            keys: dict[str, str] = {}
            for event_name in event_names:
                key = f"episode_{episode_id}:cycle_{cycle_id}:{event_name}"
                keys[event_name] = key
                point_events[key] = {"status": "confirmed"}
                outer_events[key] = {
                    "confirmation_frequency": 224 / 256,
                }
            point_cycles[f"episode_{episode_id}:cycle_{cycle_id}"] = {
                "episode_id": episode_id,
                "cycle_id": cycle_id,
                "event_keys": keys,
            }
    contract = _calibrate_interval_confirmation_reliability(
        {"events": point_events, "cycles": point_cycles},
        {
            "successful_samples": 256,
            "selection_stability": outer_events,
        },
        cycles,
        train_ids=[1],
        validation_ids=[2],
    )

    assert contract["passed"] is True
    assert contract["minimum_interval_confirmation_frequency"] == pytest.approx(
        0.875
    )
    assert contract["transition_support_at_selected_threshold"]["train"][
        "nonzero_transition_count"
    ] == 9
    assert contract["transition_support_at_selected_threshold"]["validation"][
        "nonzero_transition_count"
    ] == 9
    assert contract["confirmation_replicates"][
        "one_sided_95pct_wilson_lower_bound"
    ] == pytest.approx(_wilson_lower_bound(224, 256))
    assert contract["confirmation_replicates"][
        "one_sided_95pct_wilson_lower_bound"
    ] > 0.5


def test_final_transition_support_gate_requires_9x9_and_locked_test() -> None:
    full_matrix = {
        current: {next_sector: 1 for next_sector in ("left", "center", "right")}
        for current in ("left", "center", "right")
    }
    inventory = {
        "splits": {
            split_name: {
                "status": "computed",
                "nonzero_transition_count": 9,
                "transition_matrix": full_matrix,
                "continuity_errors": [],
            }
            for split_name in ("train", "validation")
        }
    }
    inventory["splits"]["held_out_test"] = {"status": "locked_unread"}

    report = _transition_support_gate_report_v2(inventory)

    assert report["passed"] is True
    assert report["held_out_test_status"] == "locked_unread"

    inventory["splits"]["validation"]["transition_matrix"]["left"]["right"] = 0
    inventory["splits"]["validation"]["nonzero_transition_count"] = 8
    rejected = _transition_support_gate_report_v2(inventory)
    assert rejected["passed"] is False
    assert (
        "validation_does_not_retain_all_3x3_transitions"
        in rejected["failure_reasons"]
    )


def test_sector_visual_gate_uses_balanced_null_and_zero_failures() -> None:
    calibration = {
        "sector": {
            "validation_accuracy": 0.95,
            "validation_balanced_accuracy": 0.84,
            "permutation_null_p95_accuracy": 0.80,
            "permutation_null_p95_balanced_accuracy": 0.40,
            "source_episode_bootstrap": {
                "requested_samples": 1024,
                "successful_samples": 1024,
                "failed_samples": 0,
                "validation_accuracy": {"p02_5": 0.88},
                "validation_balanced_accuracy": {"p02_5": 0.72},
            },
        }
    }

    _require_sector_visual_identifiability(calibration)

    calibration["sector"]["source_episode_bootstrap"]["failed_samples"] = 1
    with pytest.raises(RuntimeError, match="not_fully_computable"):
        _require_sector_visual_identifiability(calibration)


def test_resample_gate_rejects_a_durable_missed_segment() -> None:
    qc = {
        "source_time_basis": "step_id_times_metadata_dt",
        "wall_clock_step_ns_used": False,
        "action_label_offset_s": 0.0,
        "same_source_row_all_fields": True,
        "episode_count": 24,
        "valid_action_sign_segment_count": 11292,
        "preserved_action_sign_segment_count": 11237,
        "missing_action_sign_segment_count": 55,
        "durable_min_duration_s": 0.05,
        "durable_missing_segment_count": 0,
        "all_missing_segments_shorter_than_50ms": True,
        "max_preserved_onset_delay_s": 0.04,
    }

    assert _resample_gate_report_v2(qc)["passed"] is True

    qc["durable_missing_segment_count"] = 1
    qc["all_missing_segments_shorter_than_50ms"] = False
    rejected = _resample_gate_report_v2(qc)
    assert rejected["passed"] is False
    assert "durable_action_segment_missing" in rejected["failure_reasons"]


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
