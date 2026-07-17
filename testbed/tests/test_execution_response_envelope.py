import numpy as np

from testbed.data.execution_response_envelope import (
    ResponseSequence,
    calibrate_response_contract,
    evaluate_response_envelope,
    extract_response_events,
    fit_response_envelope,
)

POSITIVE = [0.5, 0.5, 0.5, 0.5]
NEGATIVE = [0.5, 0.5, 0.5, 0.5]


def _sequence(*, episode: int, split: str, stick_response_sign: int = -1) -> ResponseSequence:
    n_steps = 80
    command = np.zeros((n_steps, 4), dtype=np.float32)
    qpos = np.zeros((n_steps, 4), dtype=np.float32)
    qvel = np.zeros((n_steps, 4), dtype=np.float32)
    valid = np.ones(n_steps, dtype=bool)
    for axis in range(4):
        start = 15 + axis * 12
        command[start : start + 8, axis] = 0.7
        response_sign = stick_response_sign if axis == 2 else 1
        qvel[start + 1 : start + 8, axis] = 0.2 * response_sign
        qpos[start + 1 :, axis] += np.cumsum(qvel[start + 1 :, axis])
    return ResponseSequence(
        dataset_episode_id=episode,
        source_episode_id=episode,
        split=split,
        command=command,
        qpos=qpos,
        qvel=qvel,
        valid_mask=valid,
    )


def test_calibration_learns_inverted_stick_qvel_sign_from_train_only() -> None:
    calibration = calibrate_response_contract(
        [_sequence(episode=1, split="train")],
        positive_threshold=POSITIVE,
        negative_threshold=NEGATIVE,
        stationary_window_ticks=5,
        response_offset_ticks=2,
    )
    assert calibration.qvel_direction_sign == (1, 1, -1, 1)
    assert all(value >= 0.006 for value in calibration.qvel_noise)
    assert all(value > 0.9 for value in calibration.direction_agreement_by_axis)


def test_response_events_separate_from_rest_and_use_physical_stick_sign() -> None:
    sequence = _sequence(episode=1, split="train")
    calibration = calibrate_response_contract(
        [sequence],
        positive_threshold=POSITIVE,
        negative_threshold=NEGATIVE,
        stationary_window_ticks=5,
        response_offset_ticks=2,
    )
    rows = extract_response_events(
        [sequence],
        calibration=calibration,
        positive_threshold=POSITIVE,
        negative_threshold=NEGATIVE,
        response_horizons=(1, 5),
    )
    assert len(rows) == 4
    assert all(row["eligible_from_rest"] for row in rows)
    stick = next(row for row in rows if row["axis"] == "stick")
    assert stick["response_1t"] == 1
    assert stick["opposite_motion_1t"] == 0


def test_envelope_marks_missing_cells_unknown_instead_of_failure() -> None:
    train_sequence = _sequence(episode=1, split="train")
    calibration = calibrate_response_contract(
        [train_sequence],
        positive_threshold=POSITIVE,
        negative_threshold=NEGATIVE,
        stationary_window_ticks=5,
        response_offset_ticks=2,
    )
    train_rows = extract_response_events(
        [train_sequence],
        calibration=calibration,
        positive_threshold=POSITIVE,
        negative_threshold=NEGATIVE,
        response_horizons=(1, 5),
    )
    envelope = fit_response_envelope(
        train_rows,
        response_horizons=(1, 5),
        minimum_supported_events=2,
        minimum_weak_events=1,
    )
    validation = _sequence(episode=2, split="validation")
    validation.command[15:23, 0] = 0.95
    validation_rows = extract_response_events(
        [validation],
        calibration=calibration,
        positive_threshold=POSITIVE,
        negative_threshold=NEGATIVE,
        response_horizons=(1, 5),
    )
    report = evaluate_response_envelope(validation_rows, envelope)
    assert report["support_status_counts"]["out_of_support"] >= 1
    out_of_support = [
        row
        for row in report["events"]
        if row["envelope_support_status"] == "out_of_support"
    ]
    assert out_of_support
    assert out_of_support[0]["predicted_response_probability_5t"] is None


def test_envelope_hides_probability_for_underpopulated_existing_cell() -> None:
    train_sequence = _sequence(episode=1, split="train")
    calibration = calibrate_response_contract(
        [train_sequence],
        positive_threshold=POSITIVE,
        negative_threshold=NEGATIVE,
        stationary_window_ticks=5,
        response_offset_ticks=2,
    )
    train_rows = extract_response_events(
        [train_sequence],
        calibration=calibration,
        positive_threshold=POSITIVE,
        negative_threshold=NEGATIVE,
        response_horizons=(1,),
    )
    envelope = fit_response_envelope(
        train_rows,
        response_horizons=(1,),
        minimum_supported_events=3,
        minimum_weak_events=2,
    )
    validation_sequence = _sequence(episode=2, split="validation")
    validation_rows = extract_response_events(
        [validation_sequence],
        calibration=calibration,
        positive_threshold=POSITIVE,
        negative_threshold=NEGATIVE,
        response_horizons=(1,),
    )
    report = evaluate_response_envelope(validation_rows, envelope)
    assert report["support_status_counts"] == {"out_of_support": 4}
    assert all(
        row["predicted_response_probability_1t"] is None
        for row in report["events"]
    )
