from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "testbed" / "testbed" / "configs"
DATASET_ROOT = Path("/data/pingfan/Excavator_real_stack_data") / (
    "sim_expert_habit_ready_cycle_v2"
)


def _load(baseline: str) -> dict:
    return yaml.safe_load(
        (CONFIG_ROOT / f"simverify_habit_{baseline}_v2.yaml").read_text()
    )


def test_v2_configs_bind_frozen_observable_dataset() -> None:
    expected_sha = {
        "definition_manifest_sha256": (
            "47bc98c1ff31ef20f518c0e0a6caa9032d622dd628ac46ae45111b4a03ef8e87"
        ),
        "dataset_manifest_sha256": (
            "8ca37bf22638b4bbc86292c00e0230c2fe2e236a15d21e0426ad14dc0f2ca149"
        ),
        "scenario_manifest_sha256": (
            "9f41deabf6e7a045f3b7bd5f15320920f9ea3930e2e73911a677f22e2b0ab888"
        ),
        "split_manifest_sha256": (
            "cca16faac80b9437c99a0478bdb7e3a62a694c477935c093ca98131ee4854b73"
        ),
    }
    for baseline in ("b0", "b1", "b2"):
        config = _load(baseline)
        contract = config["experiment_contract"]
        assert {
            key: contract[key] for key in expected_sha
        } == expected_sha
        assert config["task"]["dataset_dir"] == str(DATASET_ROOT / "episodes")
        assert config["task"]["num_episodes"] == 201
        assert config["train"]["split_path"] == str(
            DATASET_ROOT / "derived_split.yaml"
        )
        assert config["train"]["reuse_split"] is True
        assert contract["held_out_test"] == "locked_unread"
        assert contract["closed_loop_claim_allowed"] is False
        assert config["checkpoint_semantics"]["real_control_allowed"] is False


def test_v2_b0_b1_b2_change_only_condition_factor() -> None:
    b0 = _load("b0")
    b1 = _load("b1")
    b2 = _load("b2")

    assert b0["policy"]["low_dim_keys"] == ["qpos", "qvel"]
    assert b1["policy"]["low_dim_keys"] == [
        "qpos",
        "qvel",
        "cycle_condition_v1",
    ]
    assert b2["policy"]["low_dim_keys"] == b1["policy"]["low_dim_keys"]
    assert b0["policy"]["act_params"]["state_dim"] == 8
    assert b1["policy"]["act_params"]["state_dim"] == 14
    assert b2["policy"]["act_params"]["state_dim"] == 14
    assert b1["train"]["condition_shuffle"]["enabled"] is False
    assert b2["train"]["condition_shuffle"] == {
        "enabled": True,
        "key": "cycle_condition_v1",
        "scope": "train_only",
        "seed": 20260727,
        "mode": "next_sector_within_current_committed_only",
        "committed_mask_path": "conditions/target_committed_mask",
    }

    invariant_train_keys = {
        "lr",
        "num_epochs",
        "batch_size",
        "seed",
        "device",
        "num_workers",
        "prefetch_factor",
        "persistent_workers",
        "pin_memory",
        "split_seed",
        "train_split_ratio",
        "reuse_split",
        "split_path",
        "sample_valid_mask_path",
        "norm_stats_train_only",
        "image_transform",
        "val_every",
        "save_latest_every",
        "checkpoint_every",
        "plot_every",
        "amp",
        "amp_dtype",
        "deadzone_loss",
    }
    for key in invariant_train_keys:
        assert b0["train"][key] == b1["train"][key] == b2["train"][key]
    assert b0["policy"]["act_params"]["chunk_size"] == 20
    assert b1["policy"]["act_params"]["chunk_size"] == 20
    assert b2["policy"]["act_params"]["chunk_size"] == 20
