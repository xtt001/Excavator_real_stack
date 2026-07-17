from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from testbed.configs.schema import TrainConfig
from testbed.runtime._train import _resolve_loader_settings


def test_loader_settings_keep_unspecified_configs_single_process() -> None:
    assert _resolve_loader_settings({}) == {
        "num_workers": 0,
        "prefetch_factor": None,
        "persistent_workers": False,
        "pin_memory": True,
    }


def test_loader_settings_enable_prefetch_and_persistence() -> None:
    assert _resolve_loader_settings(
        {
            "num_workers": 4,
            "prefetch_factor": 2,
            "persistent_workers": True,
            "pin_memory": True,
        }
    ) == {
        "num_workers": 4,
        "prefetch_factor": 2,
        "persistent_workers": True,
        "pin_memory": True,
    }


def test_loader_settings_disable_invalid_zero_worker_options() -> None:
    assert _resolve_loader_settings(
        {
            "num_workers": 0,
            "prefetch_factor": 8,
            "persistent_workers": True,
            "pin_memory": False,
        }
    ) == {
        "num_workers": 0,
        "prefetch_factor": None,
        "persistent_workers": False,
        "pin_memory": False,
    }


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"num_workers": -1}, "num_workers"),
        ({"num_workers": 1, "prefetch_factor": 0}, "prefetch_factor"),
    ],
)
def test_loader_settings_reject_invalid_values(config: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _resolve_loader_settings(config)


def test_train_schema_uses_storage_efficient_checkpoint_defaults() -> None:
    fields = TrainConfig.model_fields
    assert fields["save_latest_every"].default == 100
    assert fields["checkpoint_every"].default == 500
    assert fields["checkpoint_keep_last"].default == 3


def test_formal_four_camera_config_enables_worker_pipeline() -> None:
    config_path = (
        Path(__file__).parents[1]
        / "testbed/configs/act_real_gmsl_fourcam_camera_role_qpos_2000.yaml"
    )
    config = yaml.safe_load(config_path.read_text())
    train = config["train"]
    assert train["num_workers"] == 4
    assert train["prefetch_factor"] == 2
    assert train["persistent_workers"] is True
    assert train["pin_memory"] is True
    assert train["save_latest_every"] == 100
    assert train["checkpoint_every"] == 500
    assert train["checkpoint_keep_last"] == 3


def test_g49_models_use_shared_worker_and_checkpoint_contract() -> None:
    config_dir = Path(__file__).parents[1] / "testbed/configs"
    paths = sorted(config_dir.glob("act_real_gmsl_*_g49_n*_2000.yaml"))
    assert {path.name for path in paths} == {
        "act_real_gmsl_eye2_g49_n0_new_continuous_2000.yaml",
        "act_real_gmsl_eye2_g49_n1_new_transition_boundary_2000.yaml",
        "act_real_gmsl_eye2_g49_n2_new_effective_action_2000.yaml",
        "act_real_gmsl_fourcam_g49_n3_new_continuous_2000.yaml",
        "act_real_gmsl_fourcam_g49_n4_new_camera_role_2000.yaml",
        "act_real_gmsl_fourcam_g49_n5_new_camera_role_transition_2000.yaml",
    }
    for path in paths:
        train = yaml.safe_load(path.read_text())["train"]
        assert train["num_workers"] == 4, path.name
        assert train["prefetch_factor"] == 2, path.name
        assert train["persistent_workers"] is True, path.name
        assert train["pin_memory"] is True, path.name
        assert train["save_latest_every"] == 100, path.name
        assert train["checkpoint_every"] == 500, path.name
        assert train["checkpoint_keep_last"] == 3, path.name
