from __future__ import annotations

from pathlib import Path

import yaml


CONFIG_ROOT = Path(__file__).resolve().parents[1] / "testbed" / "configs"


def _load(name: str) -> dict:
    return yaml.safe_load((CONFIG_ROOT / name).read_text(encoding="utf-8"))


def test_b1_4_keeps_b1_3_training_and_router_contracts() -> None:
    b13 = _load("simverify_b1_3_phase_routed_condition_v1.yaml")
    b14 = _load("simverify_b1_4_next_only_condition_v1.yaml")
    assert b14["task"]["episode_ids"] == b13["task"]["episode_ids"]
    assert b14["task"]["camera_names"] == b13["task"]["camera_names"]
    ignored = {"ckpt_dir"}
    assert {
        key: value
        for key, value in b14["train"].items()
        if key not in ignored
    } == {
        key: value
        for key, value in b13["train"].items()
        if key not in ignored
    }
    for key in ("chunk_size", "kl_weight", "hidden_dim", "dim_feedforward"):
        assert (
            b14["policy"]["act_params"][key]
            == b13["policy"]["act_params"][key]
        )
    b14_route = b14["policy"]["act_params"]["phase_routed_condition"]
    b13_route = b13["policy"]["act_params"]["phase_routed_condition"]
    for key in (
        "router_manifest_sha256",
        "router_params_sha256",
        "route_assignments_sha256",
        "router_checksums_sha256",
    ):
        assert b14_route[key] == b13_route[key]
    assert b14_route["factor_mode"] == "next_only"
    assert b14_route["current_condition_influence"] == "exact_zero"


def test_b1_4_and_b2_4_differ_only_in_next_label_association() -> None:
    b14 = _load("simverify_b1_4_next_only_condition_v1.yaml")
    b24 = _load("simverify_b2_4_next_only_shuffled_condition_v1.yaml")
    assert b14["policy"] == b24["policy"]
    assert b14["task"]["episode_ids"] == b24["task"]["episode_ids"]
    assert b14["train"]["condition_shuffle"]["enabled"] is False
    assert b24["train"]["condition_shuffle"] == {
        "enabled": True,
        "key": "cycle_condition_v1",
        "scope": "train_only",
        "seed": 20260725,
    }
    ignored_train = {"condition_shuffle", "ckpt_dir"}
    assert {
        key: value
        for key, value in b14["train"].items()
        if key not in ignored_train
    } == {
        key: value
        for key, value in b24["train"].items()
        if key not in ignored_train
    }


def test_b1_4_is_next_only_offline_and_heldout_locked() -> None:
    for name in (
        "simverify_b1_4_next_only_condition_v1.yaml",
        "simverify_b2_4_next_only_shuffled_condition_v1.yaml",
    ):
        config = _load(name)
        experiment = config["experiment_contract"]
        route = config["policy"]["act_params"]["phase_routed_condition"]
        assert experiment["condition_input"] == (
            "cycle_condition_v1_next_sector_only"
        )
        assert experiment["current_sector_semantics"] == (
            "hindsight_annotation_not_policy_input"
        )
        assert experiment["held_out_test"] == "locked_unread"
        assert experiment["closed_loop_claim_allowed"] is False
        assert route["schema"] == (
            "simverify_phase_routed_next_only_condition_v1"
        )
        assert route["factor_mode"] == "next_only"
        assert config["checkpoint_semantics"]["real_control_allowed"] is False
        assert config["checkpoint_semantics"]["jetson_allowed"] is False
