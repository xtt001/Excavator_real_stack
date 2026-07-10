from scripts.e34_combine_policy_gohome_candidate import (
    build_combined_summary,
    select_model_row,
)


def test_select_model_row_requires_exact_model_match() -> None:
    rows = [
        {"model": "e16", "value": "old"},
        {"model": "e22b", "value": "selected"},
    ]

    assert select_model_row(rows, "e22b") == {"model": "e22b", "value": "selected"}


def test_build_combined_summary_requires_action_tail_and_gohome_safety() -> None:
    action_gate = {
        "gate_name": "simple_0.15_s0.50",
        "scan_row": {
            "mae": 0.041,
            "rmse": 0.086,
            "startup_policy_any_effective_pct": 65.0,
            "startup_same_dir_pct": 78.0,
            "startup_extra_or_wrong_pct": 5.0,
            "main_policy_any_effective_pct": 95.0,
            "main_same_dir_pct": 94.0,
            "main_extra_or_wrong_pct": 1.0,
        },
    }
    startup_row = {
        "mean_policy_any_effective_pct": "65.0",
        "mean_same_axis_dir_pct_of_expert_effective": "78.0",
        "mean_extra_or_wrong_pct_of_policy_effective": "5.0",
    }
    tail_row = {
        "total_policy_effective_frames": "0",
        "tail_policy_any_effective_rate": "0.0",
        "mean_policy_p95_max_abs": "0.062",
        "max_policy_max_abs": "0.29",
    }
    gohome_gate = {
        "gate": "learned_tail_t0.97_tc10_e0.80_ec3",
        "scan_row": {
            "event_recall": 0.9583333333333334,
            "pre_tail_false_positive_episodes": 0,
            "pre_tail_active_frames": 0,
            "mean_detection_delay_steps": 2.87,
            "mean_steps_before_t_go": 12.3,
        },
    }

    summary = build_combined_summary(
        candidate_id="E34",
        action_model="e22b",
        action_gate_summary=action_gate,
        startup_row=startup_row,
        tail_row=tail_row,
        gohome_gate_summary=gohome_gate,
        artifact_paths={"action": "/tmp/action", "gohome": "/tmp/gohome"},
    )

    assert summary["candidate_id"] == "E34"
    assert summary["action_gate"] == "simple_0.15_s0.50"
    assert summary["gohome_gate"] == "learned_tail_t0.97_tc10_e0.80_ec3"
    assert summary["tail_stop_pass"] is True
    assert summary["gohome_pre_tail_pass"] is True
    assert summary["combined_offline_gate_pass"] is True


def test_build_combined_summary_fails_when_gohome_requests_early() -> None:
    action_gate = {"gate_name": "action", "scan_row": {"mae": 0.1}}
    startup_row = {}
    tail_row = {"total_policy_effective_frames": "0", "tail_policy_any_effective_rate": "0.0"}
    gohome_gate = {
        "gate": "unsafe",
        "scan_row": {
            "event_recall": 1.0,
            "pre_tail_false_positive_episodes": 1,
            "pre_tail_active_frames": 1,
        },
    }

    summary = build_combined_summary(
        candidate_id="E34",
        action_model="e22b",
        action_gate_summary=action_gate,
        startup_row=startup_row,
        tail_row=tail_row,
        gohome_gate_summary=gohome_gate,
        artifact_paths={},
    )

    assert summary["tail_stop_pass"] is True
    assert summary["gohome_pre_tail_pass"] is False
    assert summary["combined_offline_gate_pass"] is False
