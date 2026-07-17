from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from testbed.cli.evaluate_expert_intent import run_expert_intent_evaluation
from testbed.data.expert_intent_events import (
    EVENTS_FILENAME,
    MANIFEST_FILENAME,
    sha256_file,
)
from testbed.data.expert_intent_events import (
    SCHEMA_VERSION as EVENT_SCHEMA_VERSION,
)
from testbed.policies.intent_eval import (
    INFERENCE_SOURCE,
    evaluate_open_loop_intent,
)

THRESHOLDS = {
    "swing": {"pos": 0.6, "neg": 0.7},
    "boom": {"pos": 0.25, "neg": 0.35},
    "stick": {"pos": 0.5, "neg": 0.5},
    "bucket": {"pos": 0.4, "neg": 0.5},
}


def test_distinguishes_onset_early_unsupported_and_opposite_with_scopes() -> None:
    events = [
        _event(
            episode_id=1,
            event_index=0,
            onset=0,
            anchor=["stick+"],
            immediate=["stick+"],
            near_2_5=["boom-"],
            near_6_10=["bucket+"],
            supported=["boom-", "stick+", "bucket+"],
            total_steps=12,
        ),
        _event(
            episode_id=1,
            event_index=1,
            onset=2,
            anchor=["boom-"],
            immediate=["boom-"],
            near_2_5=[],
            near_6_10=[],
            supported=["boom-"],
            total_steps=12,
        ),
    ]
    policy = np.zeros((12, 4), dtype=np.float32)
    policy[0, 2] = 0.6  # exact anchor stick+
    policy[1, 1] = -0.4  # earlier than the expert's later-supported onset
    policy[2, 1] = 0.3  # boom+, opposite to event 1 anchor boom-
    policy[3, 0] = 0.65  # swing+, unsupported for event 0

    report = evaluate_open_loop_intent(
        model="candidate",
        events=events,
        policy_actions={1: policy},
        thresholds=THRESHOLDS,
    )

    rows = {(row["event_index"], row["window"]): row for row in report["rows"]}
    assert rows[(0, "immediate_0_1")]["single_demo_later_supported_directions"] == [
        "boom-"
    ]
    assert rows[(0, "immediate_0_1")][
        "single_demo_direction_onset_later_directions"
    ] == ["boom-"]
    assert rows[(0, "near_2_5")]["outside_single_demo_event_support_directions"] == [
        "swing+",
        "boom+",
    ]
    assert rows[(1, "immediate_0_1")]["opposite_to_single_demo_anchor_directions"] == [
        "boom+"
    ]
    first = report["aggregates"]["first_event"]["anchor_current"]
    all_events = report["aggregates"]["all_events"]["anchor_current"]
    assert first["single_demo_exact_set_rate"] == 1.0
    assert all_events["single_demo_exact_set_rate"] == 0.5
    assert "exact_set_rate" not in first
    assert "unsupported_directions" not in rows[(0, "near_2_5")]
    assert report["inference_source"] == INFERENCE_SOURCE
    assert report["capability_boundaries"]["correctness_estimable"] is False


def test_previously_active_direction_is_not_expert_direction_onset_early() -> None:
    event = _event(
        episode_id=1,
        event_index=0,
        onset=0,
        anchor=["stick+"],
        immediate=["stick+"],
        near_2_5=[],
        near_6_10=[],
        supported=["stick+"],
        total_steps=12,
        direction_delays={"stick+": 0},
    )
    event["direction_details"][0]["release_step_exclusive"] = 2
    policy = np.zeros((12, 4), dtype=np.float32)
    policy[6, 2] = 0.6

    report = evaluate_open_loop_intent(
        model="candidate",
        events=[event],
        policy_actions={1: policy},
        thresholds=THRESHOLDS,
    )

    row = next(row for row in report["rows"] if row["window"] == "near_6_10")
    assert row["single_demo_later_supported_directions"] == ["stick+"]
    assert row["single_demo_direction_onset_later_directions"] == []
    aggregate = report["aggregates"]["first_event"]["near_6_10"]
    assert aggregate["single_demo_later_supported_rate"] == 1.0
    assert aggregate["single_demo_direction_onset_later_rate"] == 0.0


def test_startup_readiness_uses_only_first_effective_candidate() -> None:
    events = [
        _event(1, 0, 3, ["stick+"], ["stick+"], [], [], ["stick+"], 12),
        _event(
            2,
            0,
            3,
            ["stick+"],
            ["stick+"],
            ["boom-"],
            [],
            ["boom-", "stick+"],
            12,
            direction_delays={"boom-": 2, "stick+": 0},
        ),
        _event(3, 0, 3, ["stick+"], ["stick+"], [], [], ["stick+"], 12),
        _event(4, 0, 3, ["stick+"], ["stick+"], [], [], ["stick+"], 12),
        _event(5, 0, 3, ["stick+"], ["stick+"], [], [], ["stick+"], 12),
    ]
    t0_anchor_then_unsupported = np.zeros((12, 4), dtype=np.float32)
    t0_anchor_then_unsupported[0, 2] = 0.6
    t0_anchor_then_unsupported[1, 0] = 0.65
    later_supported_early = np.zeros((12, 4), dtype=np.float32)
    later_supported_early[0, 1] = -0.4
    later_supported_early[3, 1] = -0.4
    post_expert = np.zeros((12, 4), dtype=np.float32)
    post_expert[4, 2] = 0.6
    none = np.zeros((12, 4), dtype=np.float32)
    unsupported_opposite = np.zeros((12, 4), dtype=np.float32)
    unsupported_opposite[0, 0] = 0.65
    unsupported_opposite[0, 2] = -0.6

    report = evaluate_open_loop_intent(
        model="candidate",
        events=events,
        policy_actions={
            1: t0_anchor_then_unsupported,
            2: later_supported_early,
            3: post_expert,
            4: none,
            5: unsupported_opposite,
        },
        thresholds=THRESHOLDS,
        sampling_hz=20.0,
    )

    startup = report["startup_readiness"]
    rows = {row["episode_id"]: row for row in startup["episode_rows"]}
    assert rows[1]["first_effective_step"] == 0
    assert rows[1]["first_direction_set"] == ["stick+"]
    assert rows[1]["single_demo_exact_anchor"] is True
    assert "exact_expert_anchor" not in rows[1]
    assert rows[1]["has_outside_single_demo_local_support_direction"] is False
    assert rows[1]["relative_to_single_demo_onset_seconds"] == -0.15

    assert rows[2]["intersects_single_demo_anchor"] is False
    assert rows[2]["within_single_demo_local_support"] is True
    assert rows[2]["has_outside_single_demo_local_support_direction"] is False
    assert rows[2]["single_demo_direction_onset_later_directions"] == ["boom-"]
    anchor_row = next(
        row
        for row in report["rows"]
        if row["episode_id"] == 2 and row["window"] == "anchor_current"
    )
    assert anchor_row["single_demo_direction_onset_later_directions"] == ["boom-"]

    assert rows[3]["status"] == "post_single_demo_trajectory_not_initial_readiness"
    assert rows[3]["included_in_pre_or_at_demo_similarity_summary"] is False
    assert rows[4]["status"] == "none"
    assert rows[4]["first_effective_step"] is None
    assert rows[5]["outside_single_demo_local_support_directions"] == [
        "swing+",
        "stick-",
    ]
    assert rows[5]["opposite_to_single_demo_anchor_directions"] == ["stick-"]

    metrics = startup["first_candidate_single_demo_similarity"]
    assert metrics["pre_or_at_single_demo_onset_candidate_count"] == 3
    assert metrics["anchor_overlap_count"] == 1
    assert metrics["exact_anchor_count"] == 1
    assert metrics["within_local_support_count"] == 2
    assert metrics["outside_local_support_count"] == 1
    assert metrics["opposite_to_anchor_count"] == 1
    assert (
        startup["first_effective_before_or_at_single_demo_onset_rate_of_episodes"]
        == 0.6
    )
    assert startup["post_single_demo_trajectory_not_initial_readiness_count"] == 1
    assert startup["none_count"] == 1
    assert startup["timing_distribution"]["count"] == 4
    capability = report["capability_boundaries"]["startup_readiness_proxy"]
    assert capability["no_required_startup_axis"] is True
    assert "not promotion or safety gates" in capability["gate_policy"]


@pytest.mark.parametrize(
    ("details", "error"),
    [
        ([], "do not cover single-demo support"),
        (
            [
                {"direction": "stick+", "onset_delay_ticks": 0},
                {"direction": "stick+", "onset_delay_ticks": 0},
            ],
            "duplicate direction detail",
        ),
        (
            [{"direction": "stick+", "onset_delay_ticks": 0.5}],
            "unusable onset delay",
        ),
        (
            [{"direction": "swing+", "onset_delay_ticks": 0}],
            "outside single-demo support",
        ),
    ],
)
def test_requires_complete_unique_direction_details(details, error) -> None:
    event = _event(1, 0, 0, ["stick+"], ["stick+"], [], [], ["stick+"], 12)
    event["direction_details"] = details
    with pytest.raises(ValueError, match=error):
        evaluate_open_loop_intent(
            model="candidate",
            events=[event],
            policy_actions={1: np.zeros((12, 4), dtype=np.float32)},
            thresholds=THRESHOLDS,
        )


def test_episode_macro_does_not_equal_event_micro_when_episode_sizes_differ() -> None:
    events = [
        _event(1, 0, 0, ["stick+"], ["stick+"], [], [], ["stick+"], 12),
        _event(1, 1, 1, ["stick+"], ["stick+"], [], [], ["stick+"], 12),
        _event(2, 0, 0, ["stick+"], ["stick+"], [], [], ["stick+"], 12),
    ]
    episode_1 = np.zeros((12, 4), dtype=np.float32)
    episode_1[:2, 2] = 0.6
    episode_2 = np.zeros((12, 4), dtype=np.float32)

    report = evaluate_open_loop_intent(
        model="candidate",
        events=events,
        policy_actions={1: episode_1, 2: episode_2},
        thresholds=THRESHOLDS,
    )
    aggregate = report["aggregates"]["all_events"]["anchor_current"]
    assert aggregate["single_demo_direction_recall"] == pytest.approx(2 / 3)
    assert aggregate["episode_macro"]["single_demo_direction_recall"] == 0.5
    assert aggregate["clustered_non_independent"] is True


def test_cli_rejects_nonvalidation_or_sealed_source_before_eval_read(tmp_path) -> None:
    with pytest.raises(ValueError, match="sampling_hz must be finite and positive"):
        run_expert_intent_evaluation(
            eval_dirs={"model": tmp_path / "missing"},
            event_dir=tmp_path / "missing",
            deadzone_json=tmp_path / "missing.json",
            output_dir=tmp_path / "output",
            split="validation",
            sampling_hz=0.0,
        )
    with pytest.raises(ValueError, match="only the validation split"):
        run_expert_intent_evaluation(
            eval_dirs={"model": tmp_path / "missing"},
            event_dir=tmp_path / "missing",
            deadzone_json=tmp_path / "missing.json",
            output_dir=tmp_path / "output",
            split="test",
        )

    event_dir, deadzone = _write_event_artifact(
        tmp_path,
        [_event(10120, 0, 0, ["stick+"], ["stick+"], [], [], ["stick+"], 12)],
        source_episode_id=156,
    )
    with pytest.raises(ValueError, match="sealed source ID 156"):
        run_expert_intent_evaluation(
            eval_dirs={"model": tmp_path / "not-read"},
            event_dir=event_dir,
            deadzone_json=deadzone,
            output_dir=tmp_path / "output",
            split="validation",
        )


def test_rejects_episode_set_mismatch(tmp_path) -> None:
    events = [
        _event(10120, 0, 0, ["stick+"], ["stick+"], [], [], ["stick+"], 12),
        _event(10121, 0, 0, ["stick+"], ["stick+"], [], [], ["stick+"], 12),
    ]
    event_dir, deadzone = _write_event_artifact(tmp_path, events)
    eval_dir = _write_eval_dir(tmp_path / "eval", {10120: _expert_action()})

    with pytest.raises(ValueError, match="do not exactly match validation"):
        run_expert_intent_evaluation(
            eval_dirs={"model": eval_dir},
            event_dir=event_dir,
            deadzone_json=deadzone,
            output_dir=tmp_path / "output",
            split="validation",
        )


def test_rejects_cross_model_expert_mismatch(tmp_path) -> None:
    events = [_event(10120, 0, 0, ["stick+"], ["stick+"], [], [], ["stick+"], 12)]
    event_dir, deadzone = _write_event_artifact(tmp_path, events)
    first = _write_eval_dir(tmp_path / "first", {10120: _expert_action()})
    changed = _expert_action()
    changed[5, 0] = 0.1
    second = _write_eval_dir(tmp_path / "second", {10120: changed})

    with pytest.raises(ValueError, match="expert_action differs across models"):
        run_expert_intent_evaluation(
            eval_dirs={"first": first, "second": second},
            event_dir=event_dir,
            deadzone_json=deadzone,
            output_dir=tmp_path / "output",
            split="validation",
        )


def test_success_writes_machine_readable_provenance(tmp_path) -> None:
    events = [_event(10120, 0, 0, ["stick+"], ["stick+"], [], [], ["stick+"], 12)]
    event_dir, deadzone = _write_event_artifact(tmp_path, events)
    eval_dir = _write_eval_dir(tmp_path / "eval", {10120: _expert_action()})
    output = tmp_path / "output"

    result = run_expert_intent_evaluation(
        eval_dirs={"model": eval_dir},
        event_dir=event_dir,
        deadzone_json=deadzone,
        output_dir=output,
        split="validation",
    )

    report = json.loads(Path(result["report"]).read_text())
    source = json.loads(
        (output / "expert_intent_eval_source_manifest.json").read_text()
    )
    assert report["inference_source"] == INFERENCE_SOURCE
    assert source["source_hdf5_read"] is False
    assert source["policy_inference_performed"] is False
    assert source["model_command_changed"] is False
    assert source["sampling_hz"] == 20.0
    assert report["sampling_hz"] == 20.0
    assert report["schema_version"] == "single_demo_open_loop_similarity_v4"
    assert source["algorithm_semantics"]["startup_axis_requirement"] == "none"
    assert (
        "promotion or safety gate"
        in source["algorithm_semantics"]["startup_gate_policy"]
    )
    assert result["event_rows"] == 4


def _event(
    episode_id: int,
    event_index: int,
    onset: int,
    anchor: list[str],
    immediate: list[str],
    near_2_5: list[str],
    near_6_10: list[str],
    supported: list[str],
    total_steps: int,
    direction_delays: dict[str, int] | None = None,
) -> dict[str, object]:
    end = min(total_steps, onset + 11)
    inferred_delays = {}
    for direction in supported:
        if direction in anchor:
            inferred_delays[direction] = 0
        elif direction in immediate:
            inferred_delays[direction] = 1
        elif direction in near_2_5:
            inferred_delays[direction] = 2
        elif direction in near_6_10:
            inferred_delays[direction] = 6
        else:
            inferred_delays[direction] = 10
    inferred_delays.update(direction_delays or {})
    return {
        "schema_version": EVENT_SCHEMA_VERSION,
        "event_id": f"episode_{episode_id}:event_{event_index:04d}:step_{onset}",
        "episode_id": episode_id,
        "split": "validation",
        "event_index": event_index,
        "onset_step": onset,
        "support_end_step_exclusive": end,
        "support_horizon_requested_ticks": 11,
        "support_horizon_observed_ticks": end - onset,
        "anchor_intent": anchor,
        "immediate_intent_0_1": immediate,
        "near_intent_2_5": near_2_5,
        "near_intent_6_10": near_6_10,
        "single_demo_event_support_directions": supported,
        "direction_details": [
            {
                "direction": direction,
                "onset_delay_ticks": inferred_delays[direction],
            }
            for direction in supported
        ],
    }


def _expert_action() -> np.ndarray:
    action = np.zeros((12, 4), dtype=np.float32)
    action[0, 2] = 0.6
    return action


def _write_event_artifact(
    tmp_path: Path,
    events: list[dict[str, object]],
    *,
    source_episode_id: int = 135,
) -> tuple[Path, Path]:
    event_dir = tmp_path / "events"
    event_dir.mkdir()
    event_path = event_dir / EVENTS_FILENAME
    event_path.write_text(
        "".join(json.dumps(event, sort_keys=True) + "\n" for event in events),
        encoding="utf-8",
    )
    deadzone = tmp_path / "deadzone.json"
    deadzone.write_text(json.dumps({"deadzone_action": THRESHOLDS}), encoding="utf-8")
    validation_ids = sorted({int(event["episode_id"]) for event in events})
    manifest = {
        "schema_version": EVENT_SCHEMA_VERSION,
        "validation_ids": validation_ids,
        "thresholds": THRESHOLDS,
        "threshold_source_sha256": sha256_file(deadzone),
        "artifacts": {EVENTS_FILENAME: sha256_file(event_path)},
        "episodes": [
            {
                "episode_id": episode_id,
                "split": "validation",
                "path": f"/immutable/episode_{source_episode_id}.hdf5",
            }
            for episode_id in validation_ids
        ],
    }
    (event_dir / MANIFEST_FILENAME).write_text(json.dumps(manifest), encoding="utf-8")
    return event_dir, deadzone


def _write_eval_dir(root: Path, expert_actions: dict[int, np.ndarray]) -> Path:
    root.mkdir()
    episode_labels = [f"episode_{episode_id}" for episode_id in expert_actions]
    (root / "collection_summary.json").write_text(
        json.dumps({"episode_ids": episode_labels}), encoding="utf-8"
    )
    for episode_id, expert in expert_actions.items():
        episode_dir = root / "episodes" / f"episode_{episode_id}"
        episode_dir.mkdir(parents=True)
        np.savez(
            episode_dir / "actions.npz",
            expert_action=expert,
            policy_action=expert.copy(),
        )
    return root
