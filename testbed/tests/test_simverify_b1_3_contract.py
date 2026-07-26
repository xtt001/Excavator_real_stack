from __future__ import annotations

from pathlib import Path

import yaml


CONFIG_ROOT = Path(__file__).resolve().parents[1] / "testbed" / "configs"


def _load(name: str) -> dict:
    return yaml.safe_load((CONFIG_ROOT / name).read_text(encoding="utf-8"))


def test_b1_3_keeps_frozen_b1_training_hyperparameters() -> None:
    b1 = _load("simverify_b1_conditioned_v1.yaml")
    b13 = _load("simverify_b1_3_phase_routed_condition_v1.yaml")
    keys = (
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
    )
    assert {key: b13["train"][key] for key in keys} == {
        key: b1["train"][key] for key in keys
    }
    assert b13["task"]["episode_ids"] == b1["task"]["episode_ids"]
    assert b13["task"]["camera_names"] == b1["task"]["camera_names"]
    for key in ("chunk_size", "kl_weight", "hidden_dim", "dim_feedforward"):
        assert b13["policy"]["act_params"][key] == b1["policy"]["act_params"][key]


def test_b1_3_and_b2_3_differ_only_in_label_association_contract() -> None:
    b13 = _load("simverify_b1_3_phase_routed_condition_v1.yaml")
    b23 = _load("simverify_b2_3_phase_routed_shuffled_condition_v1.yaml")
    assert b13["policy"] == b23["policy"]
    assert b13["task"]["episode_ids"] == b23["task"]["episode_ids"]
    assert b13["train"]["condition_shuffle"]["enabled"] is False
    assert b23["train"]["condition_shuffle"] == {
        "enabled": True,
        "key": "cycle_condition_v1",
        "scope": "train_only",
        "seed": 20260725,
    }
    ignored_train = {"condition_shuffle", "ckpt_dir"}
    assert {
        key: value
        for key, value in b13["train"].items()
        if key not in ignored_train
    } == {
        key: value
        for key, value in b23["train"].items()
        if key not in ignored_train
    }


def test_b1_3_is_offline_only_and_heldout_locked() -> None:
    for name in (
        "simverify_b1_3_phase_routed_condition_v1.yaml",
        "simverify_b2_3_phase_routed_shuffled_condition_v1.yaml",
    ):
        config = _load(name)
        assert config["experiment_contract"]["held_out_test"] == "locked_unread"
        assert config["experiment_contract"]["closed_loop_claim_allowed"] is False
        assert config["checkpoint_semantics"]["real_control_allowed"] is False
        assert config["checkpoint_semantics"]["jetson_allowed"] is False

