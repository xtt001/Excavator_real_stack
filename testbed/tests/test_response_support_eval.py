import numpy as np

from testbed.data.execution_response_envelope import (
    ResponseSequence,
    calibrate_response_contract,
    extract_response_events,
    fit_response_envelope,
)
from testbed.policies.response_support_eval import evaluate_policy_response_support


def test_policy_response_support_keeps_intent_and_physical_support_separate() -> None:
    command = np.zeros((40, 4), dtype=np.float32)
    qpos = np.zeros((40, 4), dtype=np.float32)
    qvel = np.zeros((40, 4), dtype=np.float32)
    for axis in range(4):
        start = 10 + axis * 5
        command[start : start + 4, axis] = 0.7
        qvel[start + 1 : start + 4, axis] = -0.2 if axis == 2 else 0.2
    sequence = ResponseSequence(
        dataset_episode_id=1,
        source_episode_id=1,
        split="train",
        command=command,
        qpos=qpos,
        qvel=qvel,
        valid_mask=np.ones(40, dtype=bool),
    )
    calibration = calibrate_response_contract(
        [sequence],
        positive_threshold=[0.5] * 4,
        negative_threshold=[0.5] * 4,
        stationary_window_ticks=3,
        response_offset_ticks=1,
    )
    train_rows = extract_response_events(
        [sequence],
        calibration=calibration,
        positive_threshold=[0.5] * 4,
        negative_threshold=[0.5] * 4,
        response_horizons=(1,),
        minimum_sustain_ticks=3,
    )
    envelope = fit_response_envelope(
        train_rows,
        response_horizons=(1,),
        minimum_supported_events=1,
        minimum_weak_events=1,
    )
    policy = np.zeros((40, 4), dtype=np.float32)
    policy[10, 0] = 0.7
    event = {
        "event_id": "episode_2:event_0000:step_10",
        "episode_id": 2,
        "event_index": 0,
        "onset_step": 10,
        "anchor_intent": ["boom+"],
        "immediate_intent_0_1": ["boom+"],
        "near_intent_2_5": ["boom+"],
        "near_intent_6_10": ["boom+"],
        "single_demo_event_support_directions": ["boom+"],
        "qpos_at_onset": {"swing": 0.0, "boom": 0.0, "stick": 0.0, "bucket": 0.0},
    }
    report = evaluate_policy_response_support(
        model="test",
        events=[event],
        policy_actions={2: policy},
        positive_threshold=[0.5] * 4,
        negative_threshold=[0.5] * 4,
        envelope=envelope,
        training_supported_directions=["swing+", "boom+", "stick+", "bucket+"],
    )
    command_row = report["command_rows"][0]
    assert command_row["single_demo_action_relation"] == "training_dataset_only_match"
    assert command_row["historical_response_evidence"] == (
        "sufficient_similar_condition_evidence"
    )
    assert report["first_event"]["within_single_demo_event_actions"] == 0
    assert report["first_event"]["within_single_demo_episode_actions"] == 0
    assert report["first_event"]["outside_single_demo_episode_command_count"] == 1
    assert report["first_event"]["scoring_rule"] == {
        "single_demo_relation_axis": "single_demo_action_relation",
        "response_coverage_axis": "historical_response_evidence",
        "insufficient_response_data_counts_as_action_error": False,
        "single_combined_score": False,
    }

    policy[10] = 0.0
    policy[10, 1] = 0.95
    sparse_report = evaluate_policy_response_support(
        model="test",
        events=[event],
        policy_actions={2: policy},
        positive_threshold=[0.5] * 4,
        negative_threshold=[0.5] * 4,
        envelope=envelope,
        training_supported_directions=["swing+", "boom+", "stick+", "bucket+"],
    )
    sparse_command = sparse_report["command_rows"][0]
    assert sparse_command["single_demo_action_relation"] == "current_frame_match"
    assert sparse_command["historical_response_evidence"] == (
        "insufficient_similar_condition_data"
    )
    assert (
        sparse_report["first_event"]["outside_single_demo_episode_command_count"]
        == 0
    )
