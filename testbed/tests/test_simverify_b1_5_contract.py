from __future__ import annotations

from pathlib import Path

import yaml

CONFIG_ROOT = Path(__file__).resolve().parents[1] / "testbed" / "configs"


def _load(name: str) -> dict:
    return yaml.safe_load((CONFIG_ROOT / name).read_text(encoding="utf-8"))


def _without(mapping: dict, ignored: set[str]) -> dict:
    return {key: value for key, value in mapping.items() if key not in ignored}


def test_b1_5_changes_only_train_camera_loss_factor_from_b1_4() -> None:
    b14 = _load("simverify_b1_4_next_only_condition_v1.yaml")
    b15 = _load("simverify_b1_5_video7_dropout_v1.yaml")

    assert b15["task"]["episode_ids"] == b14["task"]["episode_ids"]
    assert b15["task"]["camera_names"] == b14["task"]["camera_names"]
    assert b15["policy"] == b14["policy"]
    assert _without(
        b15["train"],
        {"camera_loss_augmentation", "ckpt_dir"},
    ) == _without(b14["train"], {"ckpt_dir"})
    assert b15["train"]["camera_loss_augmentation"] == {
        "enabled": True,
        "scope": "train_only",
        "target_camera": "video7",
        "probability": 0.25,
        "seed": 20260727,
        "mask_rgb": [0, 0, 0],
        "decision_key": [
            "seed",
            "source_episode_id",
            "source_tick",
        ],
    }


def test_b1_5_and_b2_5_are_a_matched_condition_null_pair() -> None:
    b15 = _load("simverify_b1_5_video7_dropout_v1.yaml")
    b25 = _load("simverify_b2_5_video7_dropout_shuffled_v1.yaml")

    assert b15["policy"] == b25["policy"]
    assert b15["task"]["episode_ids"] == b25["task"]["episode_ids"]
    assert (
        b15["train"]["camera_loss_augmentation"]
        == b25["train"]["camera_loss_augmentation"]
    )
    assert b15["train"]["condition_shuffle"]["enabled"] is False
    assert b25["train"]["condition_shuffle"] == {
        "enabled": True,
        "key": "cycle_condition_v1",
        "scope": "train_only",
        "seed": 20260725,
    }
    assert _without(
        b15["train"],
        {"condition_shuffle", "ckpt_dir"},
    ) == _without(
        b25["train"],
        {"condition_shuffle", "ckpt_dir"},
    )


def test_b1_5_pair_remains_offline_and_heldout_locked() -> None:
    for name, baseline_id in (
        ("simverify_b1_5_video7_dropout_v1.yaml", "B1.5"),
        ("simverify_b2_5_video7_dropout_shuffled_v1.yaml", "B2.5"),
    ):
        config = _load(name)
        experiment = config["experiment_contract"]

        assert experiment["baseline_id"] == baseline_id
        assert experiment["stage"] == "M5_revision"
        assert experiment["evidence_scope"] == "recorded-observation/offline"
        assert experiment["held_out_test"] == "locked_unread"
        assert experiment["closed_loop_claim_allowed"] is False
        assert config["checkpoint_semantics"]["real_control_allowed"] is False
        assert config["checkpoint_semantics"]["jetson_allowed"] is False
