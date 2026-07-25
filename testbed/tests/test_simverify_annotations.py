from __future__ import annotations

from copy import deepcopy

import numpy as np

from testbed.simverify.annotations import (
    EpisodeSignals,
    annotate_numeric_cycles,
    classify_sector,
    fit_sector_thresholds,
    fuse_cycle_sectors,
)


def _two_cycle_episode() -> EpisodeSignals:
    count = 120
    qpos = np.zeros((count, 4), dtype=np.float32)
    qvel = np.zeros((count, 4), dtype=np.float32)
    action = np.zeros((count, 4), dtype=np.float32)

    # Each cycle starts in a low-swing ready envelope, enters digging through
    # positive bucket action, carries into the dump sector, and then releases.
    qpos[10:14, 0] = np.asarray([0.1, 0.2, 0.3, 0.4])
    qpos[50:54, 0] = np.asarray([0.2, 0.3, 0.4, 0.45])
    qpos[25:36, 0] = 1.0
    qpos[75:86, 0] = 1.0
    action[10:14, 3] = 1.0
    action[50:54, 3] = 1.0
    action[30:36, 3] = -1.0
    action[80:86, 3] = -1.0
    return EpisodeSignals(
        episode_id=7,
        step_id=np.arange(count, dtype=np.int64),
        qpos=qpos,
        qvel=qvel,
        action=action,
        dt=0.02,
    )


def _numeric_thresholds() -> dict[str, object]:
    return {
        "action_deadzone": 0.1,
        "dump_release": {
            "swing_threshold": 0.5,
            "minimum_release_steps": 3,
            "merge_rule": "merge_until_swing_exits_dump_cluster",
            "gap_duration_threshold": None,
        },
        "ready": {
            "minimum_envelope_steps": 3,
        },
    }


def test_observable_numeric_segmentation_finds_two_ordered_cycles() -> None:
    episode = _two_cycle_episode()

    cycles = annotate_numeric_cycles(episode, _numeric_thresholds())

    assert [cycle["cycle_id"] for cycle in cycles] == [0, 1]
    assert [cycle["source_steps"] for cycle in cycles] == [[0, 36], [36, 86]]
    assert [
        [
            cycle["observable_events"][name]["representative_step"]
            for name in (
                "ready_start",
                "dig_entry_proxy",
                "carry_transition_proxy",
                "dump_start_proxy",
                "dump_end_proxy",
                "ready_end",
            )
        ]
        for cycle in cycles
    ] == [
        [0, 10, 25, 30, 35, 36],
        [36, 50, 75, 80, 85, 86],
    ]
    assert cycles[0]["quality"] == {
        "status": "numeric_candidate",
        "confidence": None,
        "review_required": False,
        "reason_codes": [],
    }
    assert cycles[1]["quality"] == {
        "status": "ambiguous",
        "confidence": None,
        "review_required": True,
        "reason_codes": ["next_dig_entry_not_observable"],
    }
    assert cycles[0]["sector_observations"]["current"]["representative_step"] == 10
    assert cycles[0]["sector_observations"]["next"]["representative_step"] == 50
    assert cycles[1]["sector_observations"]["next"] is None
    assert cycles[0]["observable_events"]["dig_entry_proxy"]["interval"] == [10, 14]
    assert cycles[1]["observable_events"]["dig_entry_proxy"]["interval"] == [50, 54]
    assert np.isclose(
        cycles[0]["numeric_sector_evidence"]["current_swing_qpos"],
        0.1,
    )
    assert np.isclose(
        cycles[0]["sector_observations"]["current"][
            "swing_qpos_at_representative"
        ],
        0.1,
    )
    assert cycles[0]["sector_validity"] == {
        "current": {
            "valid": True,
            "source_cycle_id": 0,
            "reason_codes": [],
        },
        "next": {
            "valid": True,
            "source_cycle_id": 1,
            "reason_codes": [],
        },
    }
    assert cycles[1]["sector_validity"]["current"]["valid"] is True
    assert cycles[1]["sector_validity"]["next"]["valid"] is False
    for cycle in cycles:
        assert cycle["condition_source"] == "hindsight_outcome"
        assert cycle["command"] == {
            "current_sector": "unknown_not_recorded",
            "next_ready_sector": "unknown_not_recorded",
        }
        assert cycle["verification"] == {
            "privilege_used_for_annotation": False,
            "visual_confirmation_complete": False,
        }


def _single_cycle_without_ready_end(*, include_positive_run: bool) -> EpisodeSignals:
    count = 60
    qpos = np.zeros((count, 4), dtype=np.float32)
    qvel = np.zeros((count, 4), dtype=np.float32)
    action = np.zeros((count, 4), dtype=np.float32)
    if include_positive_run:
        qpos[10:14, 0] = np.asarray([0.15, 0.2, 0.3, 0.4])
        action[10:14, 3] = 1.0
    # There is no low-swing envelope after the release. The dump value is
    # intentionally distinct so a sentinel fallback would be visible.
    qpos[25:, 0] = 1.0
    action[30:36, 3] = -1.0
    return EpisodeSignals(
        episode_id=11,
        step_id=np.arange(count, dtype=np.int64),
        qpos=qpos,
        qvel=qvel,
        action=action,
        dt=0.02,
    )


def test_current_dig_and_carry_survive_missing_ready_end() -> None:
    cycle = annotate_numeric_cycles(
        _single_cycle_without_ready_end(include_positive_run=True),
        _numeric_thresholds(),
    )[0]

    assert cycle["observable_events"]["ready_end"] is None
    assert cycle["observable_events"]["dig_entry_proxy"] == {
        "interval": [10, 14],
        "representative_step": 10,
    }
    assert cycle["observable_events"]["carry_transition_proxy"][
        "representative_step"
    ] == 25
    assert cycle["sector_validity"]["current"]["valid"] is True
    assert np.isclose(
        cycle["numeric_sector_evidence"]["current_swing_qpos"],
        0.15,
    )
    assert "ready_end_not_identifiable" in cycle["quality"]["reason_codes"]
    assert "dig_entry_proxy_not_identifiable" not in cycle["quality"]["reason_codes"]


def test_missing_positive_run_has_no_dig_or_dump_sentinel_evidence() -> None:
    cycle = annotate_numeric_cycles(
        _single_cycle_without_ready_end(include_positive_run=False),
        _numeric_thresholds(),
    )[0]

    assert cycle["observable_events"]["dig_entry_proxy"] is None
    assert cycle["observable_events"]["carry_transition_proxy"] is None
    assert cycle["sector_observations"]["current"] is None
    assert cycle["numeric_sector_evidence"]["current_swing_qpos"] is None
    assert cycle["sector_validity"]["current"] == {
        "valid": False,
        "source_cycle_id": 0,
        "reason_codes": ["dig_entry_proxy_not_identifiable"],
    }
    assert "dig_entry_proxy_not_identifiable" in cycle["quality"]["reason_codes"]
    assert cycle["observable_events"]["dump_start_proxy"][
        "representative_step"
    ] == 30


def test_sector_fit_uses_role_local_validity_and_low_swing_means_left() -> None:
    cycles = [
        {
            "numeric_sector_evidence": {"current_swing_qpos": value},
            "sector_validity": {
                "current": {
                    "valid": True,
                    "source_cycle_id": index,
                    "reason_codes": [],
                }
            },
            # These unrelated cycle-level reasons must not discard otherwise
            # valid current-sector evidence.
            "quality": {"reason_codes": ["ready_end_not_identifiable"]},
        }
        for index, value in enumerate((-1.1, -0.9, -0.1, 0.1, 0.9, 1.1))
    ]

    thresholds = fit_sector_thresholds(cycles)

    assert thresholds["labels_low_to_high"] == ["left", "center", "right"]
    assert classify_sector(-1.0, thresholds)[0] == "left"
    assert classify_sector(0.0, thresholds)[0] == "center"
    assert classify_sector(1.0, thresholds)[0] == "right"


def test_qpos_visual_sector_disagreement_is_ambiguous_and_requires_review() -> None:
    cycle = deepcopy(
        annotate_numeric_cycles(_two_cycle_episode(), _numeric_thresholds())[0]
    )
    cycle["numeric_sector_evidence"] = {
        "current_swing_qpos": 0.9,
        "next_swing_qpos": 0.0,
    }
    sector_thresholds = {
        "cluster_centers_low_to_high": [-1.0, 0.0, 1.0],
        "boundaries_low_to_high": [-0.5, 0.5],
        "labels_low_to_high": ["right", "center", "left"],
        "boundary_review_margin": 0.05,
    }
    centroids = {
        "left": np.asarray([1.0, 0.0, 0.0]),
        "center": np.asarray([0.0, 1.0, 0.0]),
        "right": np.asarray([0.0, 0.0, 1.0]),
    }

    fused = fuse_cycle_sectors(
        cycle,
        sector_thresholds=sector_thresholds,
        current_eye_feature=centroids["right"],
        next_eye_feature=centroids["center"],
        visual_centroids=centroids,
        visual_minimum_similarity=0.9,
        visual_minimum_margin=0.2,
    )

    current_evidence = fused["visual_sector_evidence"]["current"]
    assert current_evidence["qpos_label"] == "left"
    assert current_evidence["visual_label"] == "right"
    assert current_evidence["fused_label"] is None
    assert fused["quality"]["status"] == "ambiguous"
    assert fused["quality"]["review_required"] is True
    assert "current_qpos_visual_disagreement" in fused["quality"]["reason_codes"]
    assert fused["policy_condition"]["vector"] is None
    assert fused["outcome"]["actual_current_sector"] is None
    assert fused["verification"]["visual_confirmation_complete"] is True


def test_unique_nearest_centroid_confirms_without_absolute_similarity_cutoff() -> None:
    cycle = deepcopy(
        annotate_numeric_cycles(_two_cycle_episode(), _numeric_thresholds())[0]
    )
    cycle["quality"] = {
        "status": "accepted",
        "review_required": False,
        "reason_codes": [],
    }
    cycle["numeric_sector_evidence"] = {
        "current_swing_qpos": -0.9,
        "next_swing_qpos": 0.0,
    }
    sector_thresholds = {
        "cluster_centers_low_to_high": [-1.0, 0.0, 1.0],
        "boundaries_low_to_high": [-0.5, 0.5],
        "labels_low_to_high": ["left", "center", "right"],
        "boundary_review_margin": 0.05,
    }
    centroids = {
        "left": np.asarray([1.0, 0.0, 0.0]),
        "center": np.asarray([0.8, 0.6, 0.0]),
        "right": np.asarray([0.0, 0.0, 1.0]),
    }

    fused = fuse_cycle_sectors(
        cycle,
        sector_thresholds=sector_thresholds,
        current_eye_feature=np.asarray([0.9, -0.1, 0.4]),
        next_eye_feature=np.asarray([0.7, 0.7, 0.1]),
        visual_centroids=centroids,
        visual_minimum_similarity=0.9999,
        visual_minimum_margin=0.5,
        visual_acceptance_rule="unique_nearest_centroid",
    )

    assert fused["quality"]["status"] == "accepted"
    assert fused["policy_condition"]["current_sector"] == "left"
    assert fused["policy_condition"]["next_ready_sector"] == "center"
    assert (
        fused["visual_sector_evidence"]["current"]["visual_acceptance_rule"]
        == "unique_nearest_centroid"
    )
