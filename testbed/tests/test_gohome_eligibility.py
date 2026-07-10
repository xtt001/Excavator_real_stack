import numpy as np

from testbed.policies.gohome_eligibility import (
    aggregate_gohome_event_rows,
    consecutive_active_mask,
    gated_active_mask,
    gohome_event_metrics,
    gohome_event_metrics_from_active_mask,
)


def test_consecutive_active_mask_only_fires_after_required_run() -> None:
    probs = np.asarray([0.7, 0.1, 0.8, 0.9, 0.2, 0.95, 0.96, 0.97], dtype=np.float32)

    mask = consecutive_active_mask(probs, threshold=0.75, consecutive_steps=2)

    np.testing.assert_array_equal(mask, [False, False, False, True, False, False, True, True])


def test_gohome_event_metrics_reports_late_detection_without_early_false_positive() -> None:
    labels = np.asarray([False, False, True, True, True, False], dtype=bool)
    loss_mask = np.ones(labels.shape[0], dtype=bool)
    probs = np.asarray([0.1, 0.2, 0.8, 0.9, 0.7, 0.1], dtype=np.float32)

    row = gohome_event_metrics(
        episode_id="episode_1",
        probability=probs,
        eligible_label=labels,
        loss_mask=loss_mask,
        threshold=0.75,
        consecutive_steps=2,
    )

    assert row["detected"] == 1
    assert row["early_false_positive"] == 0
    assert row["eligible_start"] == 2
    assert row["eligible_end"] == 4
    assert row["first_active_step"] == 3
    assert row["detection_delay_steps"] == 1
    assert row["steps_before_t_go"] == 1


def test_gohome_event_metrics_reports_early_false_positive_before_eligible_window() -> None:
    labels = np.asarray([False, False, False, True, True], dtype=bool)
    loss_mask = np.ones(labels.shape[0], dtype=bool)
    tail_idle_mask = np.asarray([False, False, True, True, True], dtype=bool)
    probs = np.asarray([0.9, 0.9, 0.1, 0.9, 0.9], dtype=np.float32)

    row = gohome_event_metrics(
        episode_id="episode_2",
        probability=probs,
        eligible_label=labels,
        loss_mask=loss_mask,
        tail_idle_mask=tail_idle_mask,
        threshold=0.8,
        consecutive_steps=2,
    )

    assert row["detected"] == 1
    assert row["early_false_positive"] == 1
    assert row["pre_tail_false_positive"] == 1
    assert row["dwell_early_active_frames"] == 0
    assert row["first_active_step"] == 1
    assert row["first_eligible_active_step"] == 4
    assert row["first_early_active_step"] == 1


def test_gohome_event_metrics_separates_dwell_early_from_pre_tail_false_positive() -> None:
    labels = np.asarray([False, False, False, False, True, True], dtype=bool)
    loss_mask = np.ones(labels.shape[0], dtype=bool)
    tail_idle_mask = np.asarray([False, False, True, True, True, True], dtype=bool)
    probs = np.asarray([0.1, 0.1, 0.9, 0.9, 0.9, 0.1], dtype=np.float32)

    row = gohome_event_metrics(
        episode_id="episode_3",
        probability=probs,
        eligible_label=labels,
        loss_mask=loss_mask,
        tail_idle_mask=tail_idle_mask,
        threshold=0.8,
        consecutive_steps=2,
    )

    assert row["detected"] == 1
    assert row["early_false_positive"] == 1
    assert row["pre_tail_false_positive"] == 0
    assert row["dwell_early_active_frames"] == 1
    assert row["first_early_active_step"] == 3
    assert row["first_eligible_active_step"] == 4


def test_gated_active_mask_removes_pre_candidate_eligibility_trigger() -> None:
    candidate_active = np.asarray([False, False, True, True, True], dtype=bool)
    eligibility_prob = np.asarray([0.9, 0.9, 0.9, 0.9, 0.1], dtype=np.float32)

    active = gated_active_mask(
        candidate_active=candidate_active,
        eligibility_probability=eligibility_prob,
        eligibility_threshold=0.8,
        eligibility_consecutive_steps=2,
    )

    np.testing.assert_array_equal(active, [False, False, True, True, False])


def test_gohome_event_metrics_from_active_mask_accepts_two_stage_gate() -> None:
    labels = np.asarray([False, False, False, True, True], dtype=bool)
    loss_mask = np.ones(labels.shape[0], dtype=bool)
    tail_idle_mask = np.asarray([False, False, True, True, True], dtype=bool)
    active_mask = np.asarray([False, False, True, True, False], dtype=bool)

    row = gohome_event_metrics_from_active_mask(
        episode_id="episode_4",
        active_mask=active_mask,
        eligible_label=labels,
        loss_mask=loss_mask,
        tail_idle_mask=tail_idle_mask,
        gate="oracle_tail+eligibility",
    )

    assert row["detected"] == 1
    assert row["early_false_positive"] == 1
    assert row["pre_tail_false_positive"] == 0
    assert row["dwell_early_active_frames"] == 1
    assert row["first_active_step"] == 2
    assert row["first_eligible_active_step"] == 3
    assert row["gate"] == "oracle_tail+eligibility"


def test_aggregate_gohome_event_rows_summarizes_event_safety_and_recall() -> None:
    rows = [
        {
            "detected": 1,
            "early_false_positive": 0,
            "pre_tail_false_positive": 0,
            "early_active_frames": 0,
            "pre_tail_active_frames": 0,
            "dwell_early_active_frames": 0,
            "eligible_active_frames": 8,
            "detection_delay_steps": 1,
            "steps_before_t_go": 4,
        },
        {
            "detected": 0,
            "early_false_positive": 1,
            "pre_tail_false_positive": 1,
            "early_active_frames": 3,
            "pre_tail_active_frames": 1,
            "dwell_early_active_frames": 2,
            "eligible_active_frames": 0,
            "detection_delay_steps": "",
            "steps_before_t_go": "",
        },
    ]

    agg = aggregate_gohome_event_rows(rows)

    assert agg["episodes"] == 2
    assert agg["event_recall"] == 0.5
    assert agg["early_false_positive_episode_rate"] == 0.5
    assert agg["pre_tail_false_positive_episode_rate"] == 0.5
    assert agg["early_active_frames"] == 3
    assert agg["pre_tail_active_frames"] == 1
    assert agg["dwell_early_active_frames"] == 2
    assert agg["eligible_active_frames"] == 8
    assert agg["mean_detection_delay_steps"] == 1.0
    assert agg["mean_steps_before_t_go"] == 4.0
