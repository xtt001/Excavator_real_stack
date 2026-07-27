from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from testbed.simverify.agx_closed_loop_probe import (
    build_cycle_condition,
    policy_source_step,
    run_bounded_closed_loop_probe,
    validate_agx_info,
)


def _jpeg_frame(value: int) -> dict:
    rgb = np.full((288, 512, 3), value, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    assert ok
    return {
        "encoding": "jpeg",
        "shape": (288, 512, 3),
        "row_order": "top_to_bottom",
        "color_space": "rgb",
        "data": encoded.tobytes(),
    }


class _FakeEnvironment:
    def __init__(self) -> None:
        self.step_id = 0
        self.qpos = np.asarray([0.5, 0.3, 0.7, 0.4], dtype=np.float32)
        self.last_action = np.zeros(4, dtype=np.float32)
        self.frames = {
            "stick_up": _jpeg_frame(10),
            "stick_down": _jpeg_frame(20),
            "eye_left": _jpeg_frame(30),
            "eye_right": _jpeg_frame(40),
        }

    def get_info(self) -> dict:
        return {
            "protocol_version": "agx-sim/v2",
            "runtime_build_id": "fake",
            "dt": 0.02,
            "control_hz": 50.0,
            "action_order": [
                "swing_speed_cmd",
                "boom_speed_cmd",
                "stick_speed_cmd",
                "bucket_speed_cmd",
            ],
            "qpos_order": [
                "swing_position_norm",
                "boom_position_norm",
                "stick_position_norm",
                "bucket_position_norm",
            ],
            "qvel_order": [
                "swing_speed",
                "boom_speed",
                "stick_speed",
                "bucket_speed",
            ],
            "camera_names": [
                "stick_up",
                "stick_down",
                "eye_left",
                "eye_right",
            ],
            "supports_images": True,
        }

    def reset(self, *, seed: int) -> dict:
        assert seed == 7
        self.step_id = 0
        self.qpos = np.asarray([0.5, 0.3, 0.7, 0.4], dtype=np.float32)
        self.last_action = np.zeros(4, dtype=np.float32)
        return self._observation()

    def step(self, action: np.ndarray) -> dict:
        self.last_action = np.asarray(action, dtype=np.float32)
        self.step_id += 1
        self.qpos = self.qpos + self.last_action * 0.01
        return self._observation()

    def close(self) -> None:
        return None

    def _observation(self) -> dict:
        return {
            "qpos": self.qpos.copy(),
            "qvel": self.last_action.copy(),
            "encoded_images": self.frames,
            "step_id": self.step_id,
            # Deliberately not tied to 20 ms physics ticks.
            "sim_time_ns": [100, 100, 131, 193, 193, 224][self.step_id],
            "warnings": [],
        }


class _FakePolicy:
    condition_route_diagnostics = {
        "route": "next",
        "route_index": 2,
        "consecutive_pending": 0,
    }

    def __init__(self) -> None:
        self.observations: list[dict] = []

    def reset(self) -> None:
        self.observations.clear()

    def predict(self, observation: dict) -> np.ndarray:
        self.observations.append(observation)
        assert observation["image_video4"].shape == (216, 384, 3)
        assert observation["image_video5"].shape == (216, 384, 3)
        assert observation["cycle_condition_v1"].shape == (6,)
        return np.asarray([0.2, 0.0, 0.0, 0.0], dtype=np.float32)

    def last_raw_action_chunk(self) -> np.ndarray:
        return np.full((20, 4), 0.1, dtype=np.float32)

    def last_raw_action_chunk_direct(self) -> np.ndarray:
        return np.full((20, 4), 0.2, dtype=np.float32)


def test_policy_schedule_uses_first_non_early_50hz_step() -> None:
    assert [policy_source_step(tick, sim_dt=0.02) for tick in range(7)] == [
        0,
        3,
        5,
        8,
        10,
        13,
        15,
    ]


def test_condition_vector_is_current_then_next_one_hot() -> None:
    np.testing.assert_array_equal(
        build_cycle_condition("left", "right"),
        np.asarray([1, 0, 0, 0, 0, 1], dtype=np.float32),
    )
    with pytest.raises(ValueError, match="sectors"):
        build_cycle_condition("unknown", "right")


def test_agx_contract_rejects_wrong_time_step() -> None:
    info = _FakeEnvironment().get_info()
    contract = validate_agx_info(info)
    assert contract["time_basis"] == (
        "applied_step_index_times_get_info_dt"
    )
    info["dt"] = 0.019999999552965164
    contract = validate_agx_info(info)
    assert contract["reported_dt"] == 0.019999999552965164
    assert contract["dt"] == 0.02
    assert [
        policy_source_step(tick, sim_dt=contract["dt"]) for tick in range(3)
    ] == [0, 3, 5]
    info["dt"] = 0.03125
    with pytest.raises(ValueError, match="frozen source dt"):
        validate_agx_info(info)


def test_bounded_probe_records_feedback_without_privilege(tmp_path: Path) -> None:
    output = tmp_path / "probe"
    result = run_bounded_closed_loop_probe(
        policy=_FakePolicy(),
        environment=_FakeEnvironment(),
        output_root=output,
        bundle_contract={
            "baseline_id": "B1.4",
            "condition_input": "cycle_condition_v1_next_sector_only",
        },
        current_git={"branch": "v2.0.0-simVerify", "dirty": False},
        external_provenance={
            "pact": {"git_sha": "pact", "dirty": True},
            "unity": {"git_sha": "unity", "dirty": True},
        },
        current_sector="left",
        next_sector="right",
        seed=7,
        policy_ticks=2,
        save_images=False,
    )
    assert result["status"] == "completed_bounded_diagnostic"
    assert result["task_success_claimed"] is False
    assert result["timing_contract"]["policy_source_steps"] == [0, 3, 5]
    assert result["qpos_delta"][0] == pytest.approx(0.01, abs=2.0e-7)

    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["privilege_policy_input_scan"]["env_state"] is False
    assert manifest["timing_contract"]["unity_sim_time_ns_used_for_scheduling"] is False
    policy_rows = [
        json.loads(line)
        for line in (output / "policy_ticks.jsonl").read_text().splitlines()
    ]
    step_rows = [
        json.loads(line)
        for line in (output / "steps.jsonl").read_text().splitlines()
    ]
    assert [row["source_step_id"] for row in policy_rows] == [0, 3]
    assert [row["step_id"] for row in step_rows] == [0, 1, 2, 3, 4, 5]
    assert "env_state" not in (output / "policy_ticks.jsonl").read_text()
    assert (output / "checksums.sha256").is_file()
