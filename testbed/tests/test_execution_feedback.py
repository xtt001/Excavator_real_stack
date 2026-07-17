from __future__ import annotations

import inspect
import json

import h5py
import numpy as np
import pytest

from testbed.data.execution_feedback import (
    COUNTERFACTUAL_MODE,
    align_causal_previous_commands,
    build_episode_execution_feedback,
    generate_symmetric_weak_command_variants,
    load_execution_feedback_sidecar,
    resolve_execution_feedback_config,
)


def _thresholds() -> dict[str, dict[str, float]]:
    return {
        "swing": {"pos": 0.6, "neg": 0.7},
        "boom": {"pos": 0.2, "neg": 0.3},
        "stick": {"pos": 0.4, "neg": 0.5},
        "bucket": {"pos": 0.3, "neg": 0.45},
    }


def test_alignment_uses_latest_strictly_earlier_raw_command() -> None:
    raw_commands = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    aligned = align_causal_previous_commands(
        observation_timestamp_ns=np.asarray([100, 200, 300], dtype=np.int64),
        raw_commanded_action=raw_commands,
        raw_action_send_timestamp_ns=np.asarray([100, 150, 200, 250], dtype=np.int64),
        train_exclude_mask=np.zeros(3, dtype=bool),
        source_time_gap_exceeds_threshold=np.zeros(3, dtype=bool),
    )

    assert aligned.raw_source_index.tolist() == [-1, 1, 3]
    assert aligned.command_send_timestamp_ns.tolist() == [-1, 150, 250]
    assert aligned.valid_mask.tolist() == [False, True, True]
    np.testing.assert_array_equal(aligned.previous_final_command[0], np.zeros(4))
    np.testing.assert_array_equal(aligned.previous_final_command[1], raw_commands[1])
    np.testing.assert_array_equal(aligned.previous_final_command[2], raw_commands[3])


def test_alignment_resets_across_exclude_and_gap_without_pre_gap_inheritance() -> None:
    raw_commands = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    aligned = align_causal_previous_commands(
        observation_timestamp_ns=np.asarray(
            [100, 200, 300, 400, 500, 600],
            dtype=np.int64,
        ),
        raw_commanded_action=raw_commands,
        raw_action_send_timestamp_ns=np.asarray([150, 250, 350, 550], dtype=np.int64),
        train_exclude_mask=np.asarray([0, 0, 1, 0, 0, 0], dtype=np.uint8),
        source_time_gap_exceeds_threshold=np.asarray(
            [0, 0, 0, 1, 0, 0],
            dtype=np.uint8,
        ),
    )

    assert aligned.reset_mask.tolist() == [True, False, True, True, False, False]
    assert aligned.raw_source_index.tolist() == [-1, 0, -1, -1, -1, 3]
    assert aligned.valid_mask.tolist() == [False, True, False, False, False, True]
    np.testing.assert_array_equal(aligned.previous_final_command[4], np.zeros(4))
    np.testing.assert_array_equal(aligned.previous_final_command[5], raw_commands[3])


@pytest.mark.parametrize(
    "observation_ts, send_ts, error",
    [
        ([100, 100], [50], "strictly increasing"),
        ([100, 200], [150, 140], "nondecreasing"),
    ],
)
def test_alignment_rejects_timestamp_order_errors(
    observation_ts: list[int],
    send_ts: list[int],
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        align_causal_previous_commands(
            observation_timestamp_ns=np.asarray(observation_ts, dtype=np.int64),
            raw_commanded_action=np.zeros((len(send_ts), 4), dtype=np.float32),
            raw_action_send_timestamp_ns=np.asarray(send_ts, dtype=np.int64),
            train_exclude_mask=np.zeros(len(observation_ts), dtype=bool),
            source_time_gap_exceeds_threshold=np.zeros(
                len(observation_ts),
                dtype=bool,
            ),
        )


@pytest.mark.parametrize(
    "commands, error",
    [
        (np.zeros((2, 3), dtype=np.float32), "shape"),
        (
            np.asarray(
                [[0.0, 0.0, 0.0, 0.0], [np.nan, 0.0, 0.0, 0.0]],
                dtype=np.float32,
            ),
            "nonfinite",
        ),
    ],
)
def test_alignment_rejects_command_shape_and_nonfinite_values(
    commands: np.ndarray,
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        align_causal_previous_commands(
            observation_timestamp_ns=np.asarray([100, 200], dtype=np.int64),
            raw_commanded_action=commands,
            raw_action_send_timestamp_ns=np.asarray([50, 150], dtype=np.int64),
            train_exclude_mask=np.zeros(2, dtype=bool),
            source_time_gap_exceeds_threshold=np.zeros(2, dtype=bool),
        )


def test_execution_feedback_config_disabled_and_enabled_contract(tmp_path) -> None:
    disabled = resolve_execution_feedback_config(None)
    assert disabled == {
        "enabled": False,
        "manifest_path": None,
        "base_norm_stats_path": None,
        "counterfactual": {
            "enabled": False,
            "seed": 0,
            "loss_weight": 0.0,
            "thresholds": {},
        },
    }

    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    norm_stats = tmp_path / "dataset_stats.pkl"
    norm_stats.write_bytes(b"stats")
    threshold_json = tmp_path / "direct_deadzone.json"
    threshold_json.write_text(
        json.dumps(
            {
                "deadzone_action": _thresholds(),
                "metadata": {
                    "action_domain": "direct_policy_output",
                    "policy_action_scale": [1.0, 1.0, 1.0, 1.0],
                },
            }
        ),
        encoding="utf-8",
    )

    enabled = resolve_execution_feedback_config(
        {
            "enabled": True,
            "manifest_path": manifest,
            "base_norm_stats_path": norm_stats,
            "counterfactual": {
                "enabled": True,
                "seed": 731,
                "loss_weight": 1.0,
                "threshold_json": threshold_json,
            },
        }
    )

    assert enabled == {
        "enabled": True,
        "manifest_path": str(manifest.resolve()),
        "base_norm_stats_path": str(norm_stats.resolve()),
        "counterfactual": {
            "enabled": True,
            "seed": 731,
            "loss_weight": 1.0,
            "thresholds": _thresholds(),
        },
    }


@pytest.mark.parametrize("loss_weight", [-0.1, np.nan, np.inf])
def test_execution_feedback_config_rejects_invalid_counterfactual_weight(
    tmp_path,
    loss_weight: float,
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    norm_stats = tmp_path / "stats.pkl"
    norm_stats.write_bytes(b"stats")

    with pytest.raises(ValueError, match="loss_weight"):
        resolve_execution_feedback_config(
            {
                "enabled": True,
                "manifest_path": manifest,
                "base_norm_stats_path": norm_stats,
                "counterfactual": {
                    "enabled": True,
                    "seed": 1,
                    "loss_weight": loss_weight,
                    "thresholds": _thresholds(),
                },
            }
        )


def test_counterfactual_pair_is_deterministic_symmetric_and_target_independent() -> None:
    assert "action" not in inspect.signature(
        generate_symmetric_weak_command_variants
    ).parameters
    first = generate_symmetric_weak_command_variants(
        episode_id=73,
        timestep=19,
        seed=731,
        thresholds=_thresholds(),
    )
    second = generate_symmetric_weak_command_variants(
        episode_id=73,
        timestep=19,
        seed=731,
        thresholds=_thresholds(),
    )

    assert first.mode == COUNTERFACTUAL_MODE
    assert first.axis_index == second.axis_index
    assert first.magnitude_fraction == second.magnitude_fraction
    assert 0.0 <= first.magnitude_fraction < 1.0
    np.testing.assert_array_equal(
        first.previous_final_command,
        second.previous_final_command,
    )
    np.testing.assert_array_equal(first.qvel, np.zeros((2, 4), dtype=np.float32))
    nonzero_axes = np.flatnonzero(np.any(first.previous_final_command != 0.0, axis=0))
    assert nonzero_axes.tolist() == [first.axis_index]
    positive = float(first.previous_final_command[0, first.axis_index])
    negative = float(first.previous_final_command[1, first.axis_index])
    axis_thresholds = _thresholds()[first.axis]
    assert 0.0 <= positive < axis_thresholds["pos"]
    assert -axis_thresholds["neg"] < negative <= 0.0
    assert positive / axis_thresholds["pos"] == pytest.approx(
        abs(negative) / axis_thresholds["neg"],
        abs=1e-6,
    )


def test_sidecar_loader_rejects_noncausal_persisted_timestamp(tmp_path) -> None:
    resampled_path, output_path = _write_episode_pair(tmp_path, episode_id=7)
    build_episode_execution_feedback(
        episode_id=7,
        resampled_path=resampled_path,
        output_path=output_path,
    )
    with np.load(output_path, allow_pickle=False) as payload:
        values = {key: np.asarray(payload[key]).copy() for key in payload.files}
    valid_index = int(np.flatnonzero(values["valid_mask"])[0])
    values["command_send_timestamp_ns"][valid_index] = values[
        "observation_timestamp_ns"
    ][valid_index]
    with output_path.open("wb") as file:
        np.savez_compressed(file, **values)

    with pytest.raises(ValueError, match="noncausal"):
        load_execution_feedback_sidecar(output_path)


def _write_episode_pair(tmp_path, *, episode_id: int):
    raw_dir = tmp_path / "raw"
    dataset_dir = tmp_path / "resampled"
    output_dir = tmp_path / "sidecars"
    raw_dir.mkdir(exist_ok=True)
    dataset_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    raw_path = raw_dir / f"episode_{episode_id}.hdf5"
    with h5py.File(raw_path, "w") as raw:
        diagnostics = raw.create_group("diagnostics")
        diagnostics.create_dataset(
            "commanded_action",
            data=np.asarray(
                [
                    [0.1, 0.0, 0.0, 0.0],
                    [0.2, 0.0, 0.0, 0.0],
                    [0.3, 0.0, 0.0, 0.0],
                ],
                dtype=np.float32,
            ),
        )
        diagnostics.create_dataset(
            "action_send_timestamp_ns",
            data=np.asarray([150, 250, 450], dtype=np.int64),
        )
    resampled_path = dataset_dir / f"episode_{episode_id}.hdf5"
    with h5py.File(resampled_path, "w") as resampled:
        metadata = resampled.create_group("metadata")
        metadata.attrs["source_dataset_path"] = str(raw_path)
        diagnostics = resampled.create_group("diagnostics")
        diagnostics.create_dataset(
            "source_observation_timestamp_ns",
            data=np.asarray([100, 200, 300, 400, 500], dtype=np.int64),
        )
        diagnostics.create_dataset(
            "train_exclude_mask",
            data=np.asarray([0, 0, 1, 0, 0], dtype=np.uint8),
        )
        diagnostics.create_dataset(
            "source_time_gap_exceeds_threshold",
            data=np.asarray([0, 0, 0, 1, 0], dtype=np.uint8),
        )
    return (
        resampled_path,
        output_dir / f"episode_{episode_id}.execution_feedback.npz",
    )
