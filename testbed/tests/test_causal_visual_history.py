from __future__ import annotations

import numpy as np
import pytest

from testbed.data.causal_visual_history import (
    CausalVisualHistory,
    causal_window_indices,
)


def _images(*, cameras: tuple[str, ...] = ("video4",)) -> dict[str, np.ndarray]:
    return {
        camera: np.full((2, 3, 1), index, dtype=np.uint8)
        for index, camera in enumerate(cameras, start=1)
    }


def _single(value: int, *, camera: str = "video4") -> dict[str, np.ndarray]:
    return {camera: np.full((2, 3, 1), value, dtype=np.uint8)}


@pytest.mark.parametrize(
    ("target_step", "expected"),
    ((0, [0, 0, 0, 0]), (1, [0, 0, 0, 1]), (3, [0, 1, 2, 3]), (6, [3, 4, 5, 6])),
)
def test_causal_window_indices_pads_only_with_episode_start(
    target_step: int, expected: list[int]
) -> None:
    indices = causal_window_indices(7, target_step, 4)

    np.testing.assert_array_equal(indices, expected)
    assert np.all(indices <= target_step)
    assert indices.shape == (4,)


def test_causal_window_indices_rejects_out_of_episode_or_invalid_lengths() -> None:
    with pytest.raises(ValueError, match="target_step must be less"):
        causal_window_indices(3, 3, 2)
    with pytest.raises(ValueError, match="target_step must be non-negative"):
        causal_window_indices(3, -1, 2)
    with pytest.raises(ValueError, match="history_length must be positive"):
        causal_window_indices(3, 1, 0)


def test_first_frame_is_padded_but_only_observed_frame_is_valid() -> None:
    history = CausalVisualHistory(["video4"], history_length=3)

    snapshot = history.append(_single(7), {"video4": 100})

    np.testing.assert_array_equal(snapshot.images["video4"][:, 0, 0, 0], [7, 7, 7])
    np.testing.assert_array_equal(snapshot.timestamps_ns["video4"], [100, 100, 100])
    np.testing.assert_array_equal(snapshot.valid_mask["video4"], [False, False, True])
    np.testing.assert_array_equal(snapshot.age_steps["video4"], [-1, -1, 0])
    assert snapshot.accepted["video4"]
    assert not snapshot.duplicate_timestamp["video4"]


def test_history_is_causal_and_oldest_to_newest() -> None:
    history = CausalVisualHistory(["video4"], history_length=3)

    first = history.append(_single(1), {"video4": 10})
    second = history.append(_single(2), {"video4": 20})
    third = history.append(_single(3), {"video4": 30})
    fourth = history.append(_single(4), {"video4": 40})

    np.testing.assert_array_equal(first.images["video4"][:, 0, 0, 0], [1, 1, 1])
    np.testing.assert_array_equal(second.images["video4"][:, 0, 0, 0], [1, 1, 2])
    np.testing.assert_array_equal(third.images["video4"][:, 0, 0, 0], [1, 2, 3])
    np.testing.assert_array_equal(fourth.images["video4"][:, 0, 0, 0], [2, 3, 4])
    np.testing.assert_array_equal(fourth.timestamps_ns["video4"], [20, 30, 40])
    np.testing.assert_array_equal(fourth.age_steps["video4"], [2, 1, 0])


def test_duplicate_timestamp_is_ignored_and_reported() -> None:
    history = CausalVisualHistory(["video4"], history_length=2)
    history.append(_single(1), {"video4": 10})

    duplicate = history.append(_single(99), {"video4": 10})

    np.testing.assert_array_equal(duplicate.images["video4"][:, 0, 0, 0], [1, 1])
    assert not duplicate.accepted["video4"]
    assert duplicate.duplicate_timestamp["video4"]


def test_older_timestamp_is_rejected_without_partial_multi_camera_update() -> None:
    cameras = ("video4", "video5")
    history = CausalVisualHistory(cameras, history_length=2)
    history.append(_images(cameras=cameras), {"video4": 10, "video5": 10})

    with pytest.raises(ValueError, match="video5.*monotonic"):
        history.append(
            {
                "video4": _single(2)["video4"],
                "video5": _single(3, camera="video5")["video5"],
            },
            {"video4": 20, "video5": 9},
        )

    snapshot = history.snapshot()
    np.testing.assert_array_equal(snapshot.timestamps_ns["video4"], [10, 10])
    np.testing.assert_array_equal(snapshot.timestamps_ns["video5"], [10, 10])


def test_multi_camera_metadata_and_shape_are_kept_separately() -> None:
    cameras = ("video4", "video5")
    history = CausalVisualHistory(cameras, history_length=2)

    snapshot = history.append(
        {
            "video4": np.zeros((2, 3, 1), dtype=np.uint8),
            "video5": np.ones((4, 5, 3), dtype=np.uint8),
        },
        {"video4": np.int64(100), "video5": np.int64(101)},
    )

    assert snapshot.camera_names == cameras
    assert snapshot.images["video4"].shape == (2, 2, 3, 1)
    assert snapshot.images["video5"].shape == (2, 4, 5, 3)
    np.testing.assert_array_equal(snapshot.timestamps_ns["video4"], [100, 100])
    np.testing.assert_array_equal(snapshot.timestamps_ns["video5"], [101, 101])


def test_reset_requires_new_frames_and_restarts_padding() -> None:
    history = CausalVisualHistory(["video4"], history_length=2)
    history.append(_single(1), {"video4": 10})
    history.append(_single(2), {"video4": 20})
    history.reset()

    with pytest.raises(RuntimeError, match="before every configured camera"):
        history.snapshot()
    restarted = history.append(_single(9), {"video4": 30})
    np.testing.assert_array_equal(restarted.valid_mask["video4"], [False, True])
    np.testing.assert_array_equal(restarted.images["video4"][:, 0, 0, 0], [9, 9])


def test_retained_and_returned_arrays_do_not_alias_caller_buffers() -> None:
    history = CausalVisualHistory(["video4"], history_length=2)
    image = _single(4)["video4"]
    snapshot = history.append({"video4": image}, {"video4": 10})

    image[...] = 99
    np.testing.assert_array_equal(snapshot.images["video4"][:, 0, 0, 0], [4, 4])
    with pytest.raises(ValueError):
        snapshot.images["video4"][0, 0, 0, 0] = 8
