from __future__ import annotations

import json

import numpy as np

from testbed.cli.simulate_counterfactual_response import (
    run_counterfactual_response_simulation,
)
from testbed.policies.counterfactual_response import (
    aggregate_counterfactual_results,
    build_response_profile,
    classify_trace_effects,
    simulate_anchor,
)


def test_profile_and_trace_separate_command_from_response_limits() -> None:
    profile = build_response_profile(
        [
            {
                "episode_id": 1,
                "axis": "boom",
                "direction": "pos",
                "response_1t": 1,
                "response_2t": 1,
                "response_4t": 1,
            },
            {
                "episode_id": 2,
                "axis": "boom",
                "direction": "pos",
                "response_1t": 0,
                "response_2t": 1,
                "response_4t": 1,
            },
        ],
        horizons=(1, 2, 4),
    )
    assert profile["boom:pos"].response_probability(1) == 0.5
    assert profile["boom:pos"].median_first_response_ticks == 1.5

    action_trace = np.zeros((3, 4), dtype=np.float32)
    action_trace[:, 1] = 0.3
    effective, direction = classify_trace_effects(
        action_trace,
        positive_threshold=(0.661, 0.259, 0.5, 0.408),
        negative_threshold=(0.721, 0.357, 0.5, 0.508),
    )
    assert effective[:, 1].all()
    assert np.all(direction[:, 1] == 1)

    row = {
        "episode_id": "episode_1",
        "anchor_step": 10,
        "anchor_group": "mid_cycle",
        "axis": "bucket",
        "direction": "pos",
        "state_hold_status": "demo_target_not_reproduced",
        "state_hold_demo_target_not_reproduced": True,
        "demo_target_reproduction_hidden_by_teacher_forcing": False,
        "state_hold_action_trace": action_trace.tolist(),
    }
    result = simulate_anchor(
        row,
        profiles=profile,
        positive_threshold=(0.661, 0.259, 0.5, 0.408),
        negative_threshold=(0.721, 0.357, 0.5, 0.508),
    )
    assert result.command_limited
    assert not result.response_limited
    assert not result.optimistic_instant_response


def test_instant_response_upper_bound_recovers_effective_target() -> None:
    profile = build_response_profile(
        [
            {
                "episode_id": 1,
                "axis": "boom",
                "direction": "pos",
                "response_1t": 1,
            }
        ],
        horizons=(1,),
    )
    row = {
        "episode_id": "episode_1",
        "anchor_step": 10,
        "anchor_group": "mid_cycle",
        "axis": "boom",
        "direction": "pos",
        "state_hold_status": "demo_target_not_reproduced",
        "state_hold_demo_target_not_reproduced": True,
        "state_hold_action_trace": [[0.0, 0.3, 0.0, 0.0]],
    }
    result = simulate_anchor(
        row,
        profiles=profile,
        positive_threshold=(0.661, 0.259, 0.5, 0.408),
        negative_threshold=(0.721, 0.357, 0.5, 0.508),
    )
    aggregate = aggregate_counterfactual_results([result])
    assert result.optimistic_instant_response
    assert result.response_limited
    assert aggregate["optimistic_demo_target_reproduction_gain"] == 1


def test_cli_uses_train_events_and_rejects_heldout_state_hold(tmp_path) -> None:
    response_dir = tmp_path / "response"
    response_dir.mkdir()
    manifest = {
        "label_contract": "direct_command_qvel_response_v1",
        "action_domain": "direct_policy_output",
        "policy_action_scale": [1, 1, 1, 1],
        "positive_threshold": [0.661, 0.259, 0.5, 0.408],
        "negative_threshold": [0.721, 0.357, 0.5, 0.508],
        "supported_axes": ["swing", "boom", "bucket"],
        "response_horizons": [1],
        "qvel_noise_provenance": "synthetic",
    }
    (response_dir / "execution_response_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    (response_dir / "execution_response_events.jsonl").write_text(
        json.dumps(
            {
                "episode_id": 1,
                "axis": "boom",
                "direction": "pos",
                "response_1t": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    split = tmp_path / "split.yaml"
    split.write_text(
        "train_ids: [1]\nval_ids: [2]\n",
        encoding="utf-8",
    )
    trace = {
        "episode_id": "episode_1",
        "anchor_step": 1,
        "anchor_group": "startup",
        "axis": "boom",
        "direction": "pos",
        "state_hold_status": "demo_target_reproduced",
        "state_hold_demo_target_not_reproduced": False,
        "demo_target_reproduction_hidden_by_teacher_forcing": False,
        "state_hold_action_trace": [[0.0, 0.3, 0.0, 0.0]],
    }
    trace_path = tmp_path / "anchors.jsonl"
    trace_path.write_text(json.dumps(trace) + "\n", encoding="utf-8")
    report = run_counterfactual_response_simulation(
        response_dir=response_dir,
        split_path=split,
        state_hold_specs=[f"synthetic={trace_path}"],
        output_dir=tmp_path / "out",
    )
    assert report["heldout_evaluated"] is False
    assert report["train_event_rows"] == 1
    assert report["runs"]["synthetic"]["observed_demo_target_reproduced"] == 1
