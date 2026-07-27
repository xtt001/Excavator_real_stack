from __future__ import annotations

import json

import numpy as np
import pytest

from testbed.simverify.habit_cycle_eval import (
    _validate_bundle,
    _validate_matched_bundle_contracts,
    condition_swap_metrics,
    delivered_condition_rows,
    sector_condition,
    split_action_metrics,
)


def test_delivered_condition_rows_preserves_causal_gate_and_changes_only_target() -> None:
    recorded = np.zeros((5, 6), dtype=np.float32)
    recorded[2:] = sector_condition("center", "right")
    mask = np.asarray([0, 0, 1, 1, 1], dtype=np.uint8)

    delivered = delivered_condition_rows(
        recorded,
        mask,
        target_override="left",
    )

    np.testing.assert_array_equal(delivered[:2], np.zeros((2, 6)))
    np.testing.assert_array_equal(
        delivered[2:],
        np.repeat(sector_condition("center", "left")[None], 3, axis=0),
    )


def test_delivered_condition_rows_rejects_rearming_or_pre_dump_leakage() -> None:
    recorded = np.zeros((5, 6), dtype=np.float32)
    recorded[2:] = sector_condition("center", "right")
    with pytest.raises(ValueError, match="single false-to-true"):
        delivered_condition_rows(
            recorded,
            np.asarray([0, 0, 1, 0, 1], dtype=np.uint8),
        )
    leaked = recorded.copy()
    leaked[1] = sector_condition("center", "right")
    with pytest.raises(ValueError, match="inactive zeros"):
        delivered_condition_rows(
            leaked,
            np.asarray([0, 0, 1, 1, 1], dtype=np.uint8),
        )


def test_split_action_metrics_keeps_pre_and_post_windows_separate() -> None:
    expert = np.zeros((4, 4), dtype=np.float32)
    policy = np.zeros_like(expert)
    policy[2:] = 2.0
    metrics = split_action_metrics(
        expert,
        policy,
        np.asarray([0, 0, 1, 1], dtype=np.uint8),
    )
    assert metrics["pre_dump"]["overall"]["mae"] == 0.0
    assert metrics["post_commit"]["overall"]["mae"] == 2.0
    assert metrics["full_cycle"]["overall"]["mae"] == 1.0


def test_condition_swap_metrics_reports_phase_localization() -> None:
    base = np.zeros((4, 4), dtype=np.float32)
    alternate = base.copy()
    alternate[2:, 0] = -0.4
    metrics = condition_swap_metrics(
        base,
        alternate,
        np.asarray([0, 0, 1, 1], dtype=np.uint8),
        expected_swing_delta_sign=-1,
    )
    assert metrics["pre_dump_effect_l1"] == 0.0
    assert metrics["post_commit_effect_l1"] == pytest.approx(0.1)
    assert metrics["post_commit_swing_delta_mean"] == pytest.approx(-0.4)
    assert metrics["semantic_direction_correct"] is True
    assert metrics["closed_loop_execution"] is False


def _write_bundle(
    root,
    baseline: str,
    *,
    dataset_sha: str = "dataset-sha",
    split_sha: str = "split-sha",
) -> None:
    condition_input = (
        "absent"
        if baseline == "B0"
        else "cycle_condition_v1_dump_end_gated_low_dim"
    )
    shuffle = {
        "enabled": False,
        "scope": "none",
    }
    if baseline == "B2":
        shuffle = {
            "enabled": True,
            "scope": "train_committed_valid_starts_within_current_sector",
            "pre_commit_rows_unchanged": True,
            "current_sector_unchanged": True,
        }
    root.mkdir()
    (root / "run_metadata.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "experiment_contract": {
                    "experiment_id": f"test-{baseline}",
                    "baseline_id": baseline,
                    "condition_input": condition_input,
                    "dataset_manifest_sha256": dataset_sha,
                    "split_manifest_sha256": split_sha,
                    "held_out_test": "locked_unread",
                    "closed_loop_claim_allowed": False,
                    "condition_shuffle_provenance": shuffle,
                },
                "checkpoint_semantics": {
                    "domain": "sim",
                    "source_action_domain": "actuator_speed_cmd",
                    "real_control_allowed": False,
                    "evidence_scope": "recorded-observation/offline",
                },
            }
        )
    )
    (root / "policy_best.ckpt").write_bytes(b"checkpoint")
    (root / "resolved_config.yaml").write_text("test: true\n")
    stats = b"conditioned-stats" if baseline != "B0" else b"unconditioned-stats"
    (root / "dataset_stats.pkl").write_bytes(stats)


def test_bundle_validation_binds_dataset_split_and_matched_null(tmp_path) -> None:
    bundles = {}
    for baseline in ("B0", "B1", "B2"):
        root = tmp_path / baseline
        _write_bundle(root, baseline)
        bundles[baseline] = _validate_bundle(
            root,
            baseline,
            dataset_manifest_sha256="dataset-sha",
            split_manifest_sha256="split-sha",
        )
    _validate_matched_bundle_contracts(bundles)
    assert bundles["B1"]["dataset_stats_sha256"] == bundles["B2"][
        "dataset_stats_sha256"
    ]


def test_bundle_validation_rejects_cross_dataset_checkpoint(tmp_path) -> None:
    root = tmp_path / "B1"
    _write_bundle(root, "B1", dataset_sha="old-dataset")
    with pytest.raises(ValueError, match="dataset manifest SHA mismatch"):
        _validate_bundle(
            root,
            "B1",
            dataset_manifest_sha256="new-dataset",
            split_manifest_sha256="split-sha",
        )
