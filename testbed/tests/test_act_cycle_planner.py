from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from testbed.tasks.act_cycle_planner import (
    ABCyclePlanner,
    CyclePlannerError,
    ScriptCyclePlanner,
    SideMatchedScriptCyclePlanner,
    parse_side_pattern,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_parse_side_pattern_accepts_operator_separators() -> None:
    assert parse_side_pattern("A->B->B->A") == ("A", "B", "B", "A")


def test_non_looping_abbababa_manifest_has_expected_transitions() -> None:
    planner = ABCyclePlanner("ABBABABA", loop=False)

    manifest = planner.manifest()

    assert [row["transition"] for row in manifest["cycles"]] == [
        "A->B",
        "B->B",
        "B->A",
        "A->B",
        "B->A",
        "A->B",
        "B->A",
    ]
    assert [row["target_side_code"] for row in manifest["cycles"]] == [
        1,
        1,
        -1,
        1,
        -1,
        1,
        -1,
    ]
    assert manifest["action_owner"] == "ACT"
    assert manifest["policy_input_boundary"] == [
        "real_transition_condition_v1"
    ]


def test_looping_planner_requires_ready_before_advancing() -> None:
    planner = ABCyclePlanner("ABBABABA", loop=True, max_cycles=9)
    first = planner.commit_goal()

    assert first.transition == "A->B"
    np.testing.assert_allclose(
        planner.apply_condition({"qpos": [0.0]})[
            "real_transition_condition_v1"
        ],
        [1.0, 1.0],
    )
    with pytest.raises(CyclePlannerError, match="already committed"):
        planner.commit_goal()
    with pytest.raises(CyclePlannerError, match="does not match"):
        planner.mark_target_ready("A")
    assert planner.committed_goal == first

    second = planner.mark_target_ready("B")
    assert second is not None
    assert second.transition == "B->B"
    assert second.goal_epoch == 2

    manifest = planner.manifest()
    assert manifest["cycles"][7]["transition"] == "A->A"


def test_planner_marks_finite_plan_done() -> None:
    planner = ABCyclePlanner("AB", loop=False)
    planner.commit_goal()

    assert planner.mark_target_ready("B") is None
    assert planner.done is True
    with pytest.raises(CyclePlannerError, match="no remaining goal"):
        planner.commit_goal()


def test_manifest_write_refuses_accidental_overwrite(tmp_path) -> None:
    path = tmp_path / "planner.json"
    planner = ABCyclePlanner("AB", loop=False)
    planner.write_manifest(path)
    with pytest.raises(FileExistsError):
        planner.write_manifest(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema"] == "act_cycle_planner_v1"


def test_script_planner_accepts_variable_length_and_order() -> None:
    planner = ScriptCyclePlanner.from_mapping(
        {
            "schema": "act_cycle_script_v1",
            "script_id": "strip_early",
            "initial_side": "A",
            "steps": [
                {"step_id": "dig_b1", "target_side": "B"},
                {"step_id": "dig_a1", "target_side": "A", "label": "short"},
                "A",
                {"target": "B", "metadata": {"workface": "wf_02"}},
            ],
            "loop": True,
        }
    )

    manifest = planner.manifest()

    assert manifest["planner_type"] == "script"
    assert [row["transition"] for row in manifest["cycles"]] == [
        "A->B",
        "B->A",
        "A->A",
        "A->B",
    ]
    assert manifest["cycles"][1]["label"] == "short"
    assert manifest["cycles"][3]["metadata"] == {"workface": "wf_02"}


def test_script_planner_loops_explicit_targets_without_extra_wrap() -> None:
    planner = ScriptCyclePlanner(
        initial_side="A",
        steps=[
            # A->B then B->A on every repeated pass.
            {"target_side": "B"},
            {"target_side": "A"},
        ],
        loop=True,
        max_cycles=4,
    )

    assert [row["transition"] for row in planner.manifest()["cycles"]] == [
        "A->B",
        "B->A",
        "A->B",
        "B->A",
    ]


def test_script_planner_loads_yaml_file(tmp_path) -> None:
    path = tmp_path / "cycle_script.yaml"
    path.write_text(
        "schema: act_cycle_script_v1\n"
        "initial_side: B\n"
        "steps:\n"
        "  - target_side: A\n"
        "    label: first\n",
        encoding="utf-8",
    )

    planner = ScriptCyclePlanner.from_script(path)

    assert planner.manifest()["initial_side"] == "B"
    assert planner.manifest()["cycles"][0]["transition"] == "B->A"
    assert planner.manifest()["cycles"][0]["label"] == "first"


@pytest.mark.parametrize(
    ("filename", "initial_side", "transition", "target_side"),
    [
        (
            "real_transition_single_cycle_right_to_left_v1.json",
            "B",
            "B->A",
            "A",
        ),
        (
            "real_transition_single_cycle_left_to_right_v1.json",
            "A",
            "A->B",
            "B",
        ),
    ],
)
def test_field_single_cycle_scripts_are_finite_and_directionally_explicit(
    filename: str,
    initial_side: str,
    transition: str,
    target_side: str,
) -> None:
    script_path = REPO_ROOT / "testbed/testbed/configs" / filename

    planner = ScriptCyclePlanner.from_script(script_path, loop=False)
    manifest = planner.manifest()

    assert manifest["script"]["initial_side"] == initial_side
    assert manifest["script"]["loop"] is False
    assert len(manifest["cycles"]) == 1
    assert manifest["cycles"][0]["transition"] == transition
    assert manifest["cycles"][0]["target_side"] == target_side


def test_side_matched_field_scripts_select_from_observed_initial_side() -> None:
    config_dir = REPO_ROOT / "testbed/testbed/configs"
    planner = SideMatchedScriptCyclePlanner.from_script_paths(
        {
            "A": config_dir / "real_transition_four_cycle_left_start_v1.json",
            "B": config_dir / "real_transition_four_cycle_right_start_v1.json",
        },
        loop=False,
    )

    assert planner.selected_initial_side is None
    assert planner.available_initial_sides == ("A", "B")
    with pytest.raises(CyclePlannerError, match="has not been selected"):
        planner.commit_goal()

    planner.select_initial_side("A")
    manifest = planner.manifest()
    assert manifest["selected_initial_side"] == "A"
    assert [row["transition"] for row in manifest["cycles"]] == [
        "A->B",
        "B->A",
        "A->B",
        "B->A",
    ]
    assert planner.commit_goal().transition == "A->B"


def test_side_matched_planner_reset_allows_new_side_selection() -> None:
    planner = SideMatchedScriptCyclePlanner(
        {
            "A": ScriptCyclePlanner(initial_side="A", steps=["B"]),
            "B": ScriptCyclePlanner(initial_side="B", steps=["A"]),
        }
    )
    planner.select_initial_side("B")
    planner.commit_goal()

    planner.reset()
    planner.select_initial_side("A")

    assert planner.commit_goal().transition == "A->B"
