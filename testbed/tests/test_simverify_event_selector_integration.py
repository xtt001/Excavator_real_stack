from __future__ import annotations

import copy
import json

import numpy as np

from testbed.simverify.annotations import EpisodeSignals
from testbed.simverify.event_selector import (
    EVENT_NAMES,
    EVENT_PHASE,
    EVENT_PHASES,
    apply_event_selections,
    assess_point_selection_stability,
    bootstrap_event_selected_sector,
    event_selector_gate_report,
    fit_event_null_control,
    fit_event_selector,
    match_event_interval,
    public_selector,
    refit_outer_sector_with_stability_mask,
    select_event_corpus,
)


def _synthetic_calibration_corpus() -> tuple[
    dict[int, list[dict[str, object]]],
    dict[tuple[int, int], dict[str, np.ndarray]],
    dict[int, EpisodeSignals],
    list[int],
    list[int],
]:
    cycles: dict[int, list[dict[str, object]]] = {}
    features: dict[tuple[int, int], dict[str, np.ndarray]] = {}
    signals: dict[int, EpisodeSignals] = {}
    phase_index = {phase: index for index, phase in enumerate(EVENT_PHASES)}
    event_steps = (10, 20, 30, 40, 50, 60)

    for episode_id in range(1, 19):
        events: dict[str, dict[str, object]] = {}
        for event_name, step in zip(EVENT_NAMES, event_steps, strict=True):
            phase = EVENT_PHASE[event_name]
            events[event_name] = {
                "interval": [step - 1, step + 2],
                "representative_step": step,
            }
            own = np.eye(len(EVENT_PHASES), dtype=np.float32)[phase_index[phase]]
            other = np.eye(len(EVENT_PHASES), dtype=np.float32)[
                (phase_index[phase] + 1) % len(EVENT_PHASES)
            ]
            for source_step, vector in (
                (step - 2, other),
                (step - 1, other),
                (step, own),
                (step + 1, own),
                (step + 2, other),
            ):
                features[(episode_id, source_step)] = {
                    "eye": vector.copy(),
                    "stick": vector.copy(),
                }

        cycles[episode_id] = [
            {
                "cycle_id": 0,
                "source_steps": [event_steps[0], event_steps[-1]],
                "observable_events": events,
                "quality": {
                    "status": "accepted",
                    "review_required": False,
                    "reason_codes": [],
                },
                "verification": {},
            }
        ]
        qpos = np.zeros((72, 4), dtype=np.float32)
        qpos[:, 0] = (-1.0, 0.0, 1.0)[(episode_id - 1) % 3]
        signals[episode_id] = EpisodeSignals(
            episode_id=episode_id,
            step_id=np.arange(72, dtype=np.int64),
            qpos=qpos,
            qvel=np.zeros_like(qpos),
            action=np.zeros_like(qpos),
            dt=0.02,
        )

    return cycles, features, signals, list(range(1, 10)), list(range(10, 19))


def test_complete_event_selector_fit_gate_apply_path_is_json_safe() -> None:
    cycles, features, signals, train_ids, validation_ids = (
        _synthetic_calibration_corpus()
    )
    calibration_ids = train_ids + validation_ids

    selector = fit_event_selector(
        cycles,
        features,
        train_draw=train_ids,
        validation_draw=validation_ids,
    )
    null_control = fit_event_null_control(
        selector,
        cycles,
        features,
        validation_ids=validation_ids,
        replicates=20,
        seed=7,
    )
    selections = select_event_corpus(
        selector,
        cycles,
        features,
        episode_ids=calibration_ids,
    )
    outer_bootstrap = bootstrap_event_selected_sector(
        cycles,
        signals,
        features,
        train_ids=train_ids,
        validation_ids=validation_ids,
        point_selector=selector,
        point_selections=selections,
        samples=20,
        seed=7,
    )
    stability = assess_point_selection_stability(
        selector,
        selections,
        null_control,
        outer_bootstrap,
    )
    refit_outer_sector_with_stability_mask(outer_bootstrap, stability)
    gate = event_selector_gate_report(
        selector,
        null_control,
        outer_bootstrap,
        selections,
        stability,
    )

    assert gate["passed"] is True
    assert gate["failure_reasons"] == []
    assert outer_bootstrap["successful_samples"] == 20
    assert outer_bootstrap["sector"] is not None
    assert (
        outer_bootstrap["sector"]["unit"]
        == "source_episode_full_selector_refit_with_frozen_point_stability_mask"
    )
    assert not any(key.startswith("_") for key in outer_bootstrap)
    assert all(
        row["passed"]
        for key, row in stability["events"].items()
        if selections["events"][key]["status"] == "confirmed"
    )

    unstable_cycles = copy.deepcopy(cycles)
    unstable_selections = copy.deepcopy(selections)
    unstable_assessment = copy.deepcopy(stability)
    unstable_key = next(
        key
        for key, row in unstable_selections["events"].items()
        if row["phase"] == "dig_entry_proxy"
    )
    unstable_assessment["events"][unstable_key]["passed"] = False
    unstable_assessment["events"][unstable_key]["reason_codes"] = [
        "synthetic_reselection_failure"
    ]

    selector_sha256 = "0" * 64
    apply_event_selections(
        cycles,
        selections,
        stability=outer_bootstrap["selection_stability"],
        stability_assessment=stability,
        selector=selector,
        selector_sha256=selector_sha256,
        episode_ids=calibration_ids,
    )

    assert selections["summary"]["event_order_valid_cycle_count"] == len(
        calibration_ids
    )
    for episode_id in calibration_ids:
        cycle = cycles[episode_id][0]
        assert cycle["quality"]["status"] == "accepted"
        assert cycle["verification"]["visual_event_order_valid"] is True
        for event in cycle["observable_events"].values():
            selection = event["visual_interval_selection"]
            assert selection["status"] == "confirmed"
            assert selection["selector_sha256"] == selector_sha256
            assert selection["confidence"]["joint"] > 0.0

    apply_event_selections(
        unstable_cycles,
        unstable_selections,
        stability=outer_bootstrap["selection_stability"],
        stability_assessment=unstable_assessment,
        selector=selector,
        selector_sha256=selector_sha256,
        episode_ids=calibration_ids,
    )
    downgraded = unstable_selections["events"][unstable_key]
    assert downgraded["status"] == "ambiguous"
    assert downgraded["representative_step"] is None
    assert downgraded["signed_offset_steps"] is None
    assert downgraded["absolute_offset_steps"] is None
    assert "synthetic_reselection_failure" in downgraded["reason_codes"]
    downgraded_cycle = unstable_cycles[int(unstable_key.split(":", 1)[0][8:])][0]
    assert downgraded_cycle["quality"]["status"] == "ambiguous"
    assert downgraded_cycle["verification"]["visual_event_order_valid"] is False

    public = public_selector(selector)
    assert "_calibration_samples" not in public
    assert public["prototypes"]["ready"]["stick"] == {
        "npz_key": "event_stick_ready",
        "dimension": len(EVENT_PHASES),
    }
    encoded = json.dumps(
        {
            "selector": public,
            "null_control": null_control,
            "selections": selections,
            "outer_bootstrap": outer_bootstrap,
            "stability": stability,
            "gate": gate,
        },
        allow_nan=False,
        sort_keys=True,
    )
    assert encoded


def test_off_offset_neighbor_cannot_suppress_eligible_local_peak() -> None:
    selector_vector = np.asarray([1.0, 0.0], dtype=np.float32)
    selector = {
        "prototypes": {
            phase: {
                "eye": selector_vector,
                "stick": selector_vector,
            }
            for phase in EVENT_PHASES
        },
        "support_thresholds": {
            phase: {"eye": -1.0, "stick": -1.0} for phase in EVENT_PHASES
        },
        "change_thresholds": {
            "ready": {"stick": 2.0},
            "dig_entry_proxy": {"eye": 0.0, "stick": 0.0},
            "carry_transition_proxy": {"stick": 0.0},
            "dump_start_proxy": {"eye": 0.0, "stick": 0.0},
            "dump_end_proxy": {"eye": 0.0, "stick": 0.0},
        },
        "offset_bounds": {
            phase: {
                "minimum_signed_offset_steps": 0,
                "maximum_signed_offset_steps": 2,
            }
            for phase in EVENT_PHASES
        },
    }
    angle_deltas = (
        np.pi / 2.0,
        float(np.arccos(0.2)),
        float(np.arccos(0.8)),
        float(np.arccos(0.3)),
    )
    angles = [0.0]
    for delta in angle_deltas:
        angles.append(angles[-1] + delta)
    features = {
        (7, source_step): {
            "eye": np.asarray(
                [np.cos(angle), np.sin(angle)],
                dtype=np.float32,
            ),
            "stick": np.asarray(
                [np.cos(angle), np.sin(angle)],
                dtype=np.float32,
            ),
        }
        for source_step, angle in zip(range(8, 13), angles, strict=True)
    }
    row = {
        "event_key": "episode_7:cycle_0:dig_entry_proxy",
        "episode_id": 7,
        "phase": "dig_entry_proxy",
        "interval": [9, 13],
        "numeric_representative_step": 10,
    }

    selection = match_event_interval(
        row,
        features,
        selector=selector,
    )

    assert selection["status"] == "confirmed"
    assert selection["eligible_candidate_count"] == 3
    assert selection["representative_step"] == 10
    assert selection["selected"]["signed_offset_steps"] == 0


def test_gate_report_fails_closed_when_outer_selector_summary_is_missing() -> None:
    selector = {
        "validation": {
            "interval_length_steps": {phase: {"p50": 3.0} for phase in EVENT_PHASES}
        }
    }
    outer_bootstrap = {
        "requested_samples": 20,
        "successful_samples": 0,
        "failed_samples": 20,
        "failure_reasons": {"synthetic_failure": 20},
        "event_selector": {},
    }
    stability = {
        "summary": {
            "by_phase": {phase: {"stable_fraction": 0.0} for phase in EVENT_PHASES}
        }
    }

    report = event_selector_gate_report(
        selector,
        {"coverage_p95": {phase: 0.0 for phase in EVENT_PHASES}},
        outer_bootstrap,
        {"events": {}},
        stability,
    )

    assert report["passed"] is False
    assert "event_selector_outer_bootstrap_failure_rate" in report["failure_reasons"]
    for phase in EVENT_PHASES:
        assert (
            f"{phase}_selector_coverage_bootstrap_missing" in report["failure_reasons"]
        )


def _outer_sector_candidate(
    name: str,
    *,
    swing_qpos: float,
    dig_step: int = 20,
    carry_step: int = 30,
) -> dict[str, object]:
    return {
        "event_keys": [f"{name}:{event_name}" for event_name in EVENT_NAMES],
        "event_steps": [10, dig_step, carry_step, 40, 50, 60],
        "swing_qpos": swing_qpos,
    }


def _stability_for_candidates(
    candidates: list[dict[str, object]],
    *,
    unstable_keys: set[str] | None = None,
) -> dict[str, object]:
    unstable = set() if unstable_keys is None else unstable_keys
    return {
        "events": {
            str(key): {"passed": str(key) not in unstable}
            for candidate in candidates
            for key in candidate["event_keys"]
        }
    }


def _outer_with_candidates(
    candidates: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "requested_samples": 1,
        "successful_samples": 1,
        "failed_samples": 0,
        "seed": 7,
        "failure_reasons": {},
        "event_selector": {},
        "selection_stability": {},
        "_sector_candidates_by_selector_successful_replicate": [candidates],
        "sector": None,
    }


def test_sector_refit_masks_all_events_before_rechecking_local_order() -> None:
    left = _outer_sector_candidate("left", swing_qpos=-1.0)
    center = _outer_sector_candidate("center", swing_qpos=0.0)
    right = _outer_sector_candidate("right", swing_qpos=1.0)
    conflict = _outer_sector_candidate(
        "conflict",
        swing_qpos=0.2,
        dig_step=30,
        carry_step=20,
    )
    # The duplicate left row represents a repeated source-episode draw slot.
    candidates = [left, left, center, right, conflict]
    conflict_carry_key = str(
        conflict["event_keys"][EVENT_NAMES.index("carry_transition_proxy")]
    )

    restored = _outer_with_candidates(copy.deepcopy(candidates))
    refit_outer_sector_with_stability_mask(
        restored,
        _stability_for_candidates(
            candidates,
            unstable_keys={conflict_carry_key},
        ),
    )
    assert restored["sector"]["successful_samples"] == 1
    assert restored["sector"]["selected_train_cycle_count"]["median"] == 5.0

    rejected = _outer_with_candidates(copy.deepcopy(candidates))
    refit_outer_sector_with_stability_mask(
        rejected,
        _stability_for_candidates(candidates),
    )
    assert rejected["sector"]["successful_samples"] == 1
    assert rejected["sector"]["selected_train_cycle_count"]["median"] == 4.0
