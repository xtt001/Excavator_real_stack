from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

from testbed.data.mock_closed_loop import (
    DataCalibratedMockStateReader,
    H5ImageBank,
    MockClosedLoopProfile,
)


def _write_episode(path: Path, *, target_code: int, endpoint: float) -> None:
    steps = 12
    action = np.zeros((steps, 4), dtype=np.float32)
    action[:, 0] = 0.2
    qvel = np.zeros((steps, 4), dtype=np.float32)
    qvel[:, 0] = 0.1
    qpos = np.zeros((steps, 4), dtype=np.float32)
    qpos[:, 0] = np.linspace(0.0, endpoint, steps, dtype=np.float32)
    condition = np.tile(np.asarray([target_code, 1.0], dtype=np.float32), (steps, 1))
    with h5py.File(path, "w") as handle:
        metadata = handle.create_group("metadata")
        metadata.attrs["dt"] = 0.05
        handle.create_dataset("action", data=action)
        handle.create_dataset("observations/qpos", data=qpos)
        handle.create_dataset("observations/qvel", data=qvel)
        handle.create_dataset(
            "conditions/real_transition_condition_v1", data=condition
        )
        handle.create_dataset(
            "observations/images/video4",
            data=np.zeros((steps, 2, 2, 3), dtype=np.uint8),
        )


def _write_ready_contract(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "swing_axis": {
                    "home_swing_qpos_rad": 0.0,
                    "cycle_excursion_min_abs_delta_rad": 0.08,
                    "swing_qvel_abs_max_rad_s": 0.015,
                    "stable_window_s": 0.5,
                    "safe_swing_qpos_range_rad": [-0.4, 0.4],
                }
            }
        ),
        encoding="utf-8",
    )


def _write_deadzone_thresholds(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "deadzone_action": {
                    "swing": {"pos": 0.5, "neg": 0.5},
                    "boom": {"pos": 0.4, "neg": 0.4},
                    "stick": {"pos": 0.3, "neg": 0.3},
                    "bucket": {"pos": 0.2, "neg": 0.2},
                }
            }
        ),
        encoding="utf-8",
    )


def test_profile_uses_real_data_for_response_and_target_ranges(tmp_path: Path) -> None:
    _write_episode(tmp_path / "episode_0.hdf5", target_code=-1, endpoint=-0.2)
    _write_episode(tmp_path / "episode_1.hdf5", target_code=1, endpoint=0.2)
    contract = tmp_path / "ready_contract.json"
    _write_ready_contract(contract)

    profile = MockClosedLoopProfile.from_dataset(
        dataset_dir=tmp_path,
        episode_ids=[0, 1],
        ready_contract_path=contract,
    )

    assert profile.target_ranges["A"][1] < 0.0
    assert profile.target_ranges["B"][0] > 0.0
    assert profile.target_endpoint_episode_count == 2
    assert profile.qvel_damping.shape == (4, 4)
    assert profile.action_to_qvel_gain.shape == (4, 4)
    assert profile.target_ranges["A"][0] < -0.2 < profile.target_ranges["A"][1]
    assert profile.target_ranges["B"][0] < 0.2 < profile.target_ranges["B"][1]
    assert profile.target_quantile_ranges["A"] == profile.target_ranges["A"]
    assert profile.qpos_support_distance_p99 >= profile.qpos_support_distance_p95
    next_qpos, next_qvel = profile.step(
        qpos=np.zeros(4), qvel=np.zeros(4), action=np.asarray([0.2, 0, 0, 0])
    )
    assert np.isfinite(next_qpos).all()
    assert np.isfinite(next_qvel).all()
    assert profile.target_ready(
        qpos=np.asarray([-0.2, 0, 0, 0]),
        qvel=np.zeros(4),
        target_side="A",
    )


def test_data_calibrated_reader_retrieves_image_by_predicted_state(tmp_path: Path) -> None:
    _write_episode(tmp_path / "episode_0.hdf5", target_code=-1, endpoint=-0.2)
    _write_episode(tmp_path / "episode_1.hdf5", target_code=1, endpoint=0.2)
    contract = tmp_path / "ready_contract.json"
    _write_ready_contract(contract)
    profile = MockClosedLoopProfile.from_dataset(
        dataset_dir=tmp_path,
        episode_ids=[0, 1],
        ready_contract_path=contract,
    )
    bank = H5ImageBank(
        tmp_path / "episode_0.hdf5",
        camera_names=["video4"],
        qpos_state_scale=profile.qpos_state_scale,
        qvel_state_scale=profile.qvel_state_scale,
    )
    reader = DataCalibratedMockStateReader(
        profile=profile,
        image_bank=bank,
        support_bank=H5ImageBank(
            [tmp_path / "episode_0.hdf5", tmp_path / "episode_1.hdf5"],
            camera_names=["video4"],
            qpos_state_scale=profile.qpos_state_scale,
            qvel_state_scale=profile.qvel_state_scale,
        ),
        initial_qpos=np.zeros(4),
        initial_qvel=np.zeros(4),
    )

    samples = reader.read(step_id=0)
    assert samples.images["video4"].payload.shape == (2, 2, 3)
    reader.apply_control_result(
        SimpleNamespace(commanded_action=np.asarray([0.2, 0, 0, 0])), dt=0.05
    )
    assert reader.last_image_index is not None
    assert reader.last_image_distance is not None
    assert reader.last_data_support_distance is not None


def test_profile_reuses_state_hold_deadzone_for_subthreshold_commands(
    tmp_path: Path,
) -> None:
    _write_episode(tmp_path / "episode_0.hdf5", target_code=-1, endpoint=-0.2)
    _write_episode(tmp_path / "episode_1.hdf5", target_code=1, endpoint=0.2)
    contract = tmp_path / "ready_contract.json"
    thresholds = tmp_path / "deadzone.json"
    _write_ready_contract(contract)
    _write_deadzone_thresholds(thresholds)
    profile = MockClosedLoopProfile.from_dataset(
        dataset_dir=tmp_path,
        episode_ids=[0, 1],
        ready_contract_path=contract,
        deadzone_threshold_path=thresholds,
    )

    held_qpos, held_qvel = profile.step(
        qpos=np.zeros(4),
        qvel=np.zeros(4),
        action=np.asarray([0.2, 0.0, 0.0, 0.0]),
    )
    moving_qpos, moving_qvel = profile.step(
        qpos=np.zeros(4),
        qvel=np.zeros(4),
        action=np.asarray([0.6, 0.0, 0.0, 0.0]),
    )

    np.testing.assert_allclose(held_qpos, np.zeros(4))
    np.testing.assert_allclose(held_qvel, np.zeros(4))
    assert abs(float(moving_qpos[0])) > 0.0 or abs(float(moving_qvel[0])) > 0.0
