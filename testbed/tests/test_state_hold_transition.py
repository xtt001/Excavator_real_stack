from __future__ import annotations

import json

import numpy as np
import pytest

from testbed.data.state_hold_transition import (
    anchor_transition_direction_mask,
    compute_transition_direction_mask,
    intersect_transition_starts,
    resolve_state_hold_transition_config,
    sample_state_hold_start,
)


def _thresholds() -> dict[str, dict[str, float]]:
    return {
        "swing": {"pos": 0.5, "neg": 0.5},
        "boom": {"pos": 0.4, "neg": 0.4},
        "stick": {"pos": 0.3, "neg": 0.3},
        "bucket": {"pos": 0.2, "neg": 0.2},
    }


def test_disabled_config_is_the_backward_compatible_default() -> None:
    expected = {
        "enabled": False,
        "thresholds": {},
        "probability": 0.0,
        "hold_horizon_steps": 1,
        "append_samples_per_episode": 0,
    }

    assert resolve_state_hold_transition_config(None) == expected
    assert resolve_state_hold_transition_config({"enabled": False}) == expected


def test_enabled_config_resolves_threshold_json_to_canonical_keys(tmp_path) -> None:
    path = tmp_path / "direct_deadzone.json"
    path.write_text(
        json.dumps({"schema_version": 1, "deadzone_action": _thresholds()}),
        encoding="utf-8",
    )

    resolved = resolve_state_hold_transition_config(
        {
            "enabled": True,
            "threshold_json": path,
            "probability": 0.75,
            "hold_horizon_steps": 20,
        }
    )

    assert set(resolved) == {
        "enabled",
        "thresholds",
        "probability",
        "hold_horizon_steps",
        "append_samples_per_episode",
    }
    assert resolved == {
        "enabled": True,
        "thresholds": _thresholds(),
        "probability": 0.75,
        "hold_horizon_steps": 20,
        "append_samples_per_episode": 0,
    }


@pytest.mark.parametrize(
    "override, error",
    [
        ({}, "exactly one"),
        ({"thresholds": _thresholds(), "threshold_json": "also.json"}, "exactly one"),
        ({"thresholds": _thresholds(), "probability": -0.01}, "probability"),
        ({"thresholds": _thresholds(), "probability": 1.01}, "probability"),
        ({"thresholds": _thresholds(), "probability": np.nan}, "probability"),
        ({"thresholds": _thresholds(), "hold_horizon_steps": 0}, "positive"),
        ({"thresholds": _thresholds(), "hold_horizon_steps": 1.5}, "integer"),
        ({"thresholds": {"swing": {"pos": 0.5, "neg": 0.5}}}, "boom"),
    ],
)
def test_enabled_config_rejects_invalid_values(
    override: dict[str, object],
    error: str,
) -> None:
    config: dict[str, object] = {"enabled": True}
    config.update(override)

    with pytest.raises(ValueError, match=error):
        resolve_state_hold_transition_config(config)


def test_transition_mask_tracks_positive_negative_and_direction_switches() -> None:
    actions = np.asarray(
        [
            [0.00, 0.00, 0.00, 0.00],
            [0.60, 0.00, 0.00, 0.00],
            [0.70, 0.00, 0.00, 0.00],
            [-0.60, 0.00, 0.00, 0.00],
            [0.00, 0.00, 0.00, -0.30],
        ],
        dtype=np.float32,
    )

    transitions = compute_transition_direction_mask(
        actions=actions,
        thresholds=_thresholds(),
    )

    assert transitions.shape == (5, 4, 2)
    assert transitions.dtype == np.bool_
    assert np.flatnonzero(transitions[:, 0, 0]).tolist() == [1]
    assert np.flatnonzero(transitions[:, 0, 1]).tolist() == [3]
    assert np.flatnonzero(transitions[:, 3, 1]).tolist() == [4]
    assert transitions.sum() == 3


def test_forced_stop_masks_break_effective_runs_and_create_fresh_transitions() -> None:
    actions = np.tile(
        np.asarray([[0.60, 0.00, 0.00, 0.00]], dtype=np.float32),
        (6, 1),
    )
    action_loss_mask = np.asarray([False, True, True, True, True, True])
    tail_idle_mask = np.asarray([False, False, True, False, False, False])
    owner_automation = np.asarray([False, False, False, False, True, False])

    transitions = compute_transition_direction_mask(
        actions=actions,
        thresholds=_thresholds(),
        action_loss_mask=action_loss_mask,
        tail_idle_mask=tail_idle_mask,
        owner_automation=owner_automation,
    )

    assert np.flatnonzero(transitions[:, 0, 0]).tolist() == [1, 3, 5]
    assert not transitions[[0, 2, 4]].any()


def test_transition_starts_intersect_valid_starts_and_full_hold_horizon() -> None:
    transition_mask = np.zeros((7, 4, 2), dtype=bool)
    transition_mask[1, 0, 0] = True
    transition_mask[2, 2, 0] = True
    transition_mask[4, 1, 1] = True
    transition_mask[6, 3, 0] = True

    starts = intersect_transition_starts(
        transition_mask,
        np.asarray([0, 1, 3, 4, 6], dtype=np.int64),
        total_steps=7,
        hold_horizon_steps=3,
    )

    assert starts.tolist() == [1, 4]


def test_probability_zero_samples_all_valid_starts(monkeypatch) -> None:
    selected_pool: list[int] = []

    def choose(values: np.ndarray) -> int:
        selected_pool.extend(values.tolist())
        return int(values[-1])

    monkeypatch.setattr(np.random, "choice", choose)

    selected = sample_state_hold_start(
        valid_starts=np.asarray([0, 2, 4], dtype=np.int64),
        transition_starts=np.asarray([2], dtype=np.int64),
        probability=0.0,
    )

    assert selected == 4
    assert selected_pool == [0, 2, 4]


def test_probability_one_samples_only_transition_starts(monkeypatch) -> None:
    selected_pool: list[int] = []

    def choose(values: np.ndarray) -> int:
        selected_pool.extend(values.tolist())
        return int(values[0])

    monkeypatch.setattr(np.random, "choice", choose)

    selected = sample_state_hold_start(
        valid_starts=np.asarray([0, 2, 4], dtype=np.int64),
        transition_starts=np.asarray([2, 4], dtype=np.int64),
        probability=1.0,
    )

    assert selected == 2
    assert selected_pool == [2, 4]


def test_no_transitions_falls_back_to_uniform_valid_sampling(monkeypatch) -> None:
    monkeypatch.setattr(np.random, "choice", lambda values: int(values[1]))

    selected = sample_state_hold_start(
        valid_starts=np.asarray([1, 3], dtype=np.int64),
        transition_starts=np.asarray([], dtype=np.int64),
        probability=1.0,
    )

    assert selected == 3


def test_sampling_is_reproducible_with_a_seeded_numpy_rng() -> None:
    kwargs = {
        "valid_starts": np.asarray([0, 1, 2, 3], dtype=np.int64),
        "transition_starts": np.asarray([1, 3], dtype=np.int64),
        "probability": 0.5,
    }

    first_rng = np.random.default_rng(731)
    second_rng = np.random.default_rng(731)
    first = [sample_state_hold_start(**kwargs, rng=first_rng) for _ in range(20)]
    second = [sample_state_hold_start(**kwargs, rng=second_rng) for _ in range(20)]

    assert first == second


def test_anchor_direction_mask_returns_selected_start_copy() -> None:
    transitions = np.zeros((3, 4, 2), dtype=bool)
    transitions[1, 0, 0] = True
    transitions[1, 3, 1] = True

    anchor = anchor_transition_direction_mask(transitions, 1)
    anchor[0, 0] = False

    assert anchor.shape == (4, 2)
    assert anchor[3, 1]
    assert transitions[1, 0, 0]
