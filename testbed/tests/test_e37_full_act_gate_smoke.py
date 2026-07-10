import pytest

from scripts.e37_full_act_gate_smoke import (
    artifact_path_by_name,
    build_compact_summary,
    select_episode_ids,
)


def test_artifact_path_by_name_returns_declared_path() -> None:
    manifest = {
        "artifacts": [
            {"name": "action_policy_best", "path": "/tmp/policy_best.ckpt"},
            {"name": "phase_gate_model", "path": "/tmp/phase_gate_model.pt"},
        ]
    }

    assert artifact_path_by_name(manifest, "phase_gate_model") == "/tmp/phase_gate_model.pt"


def test_artifact_path_by_name_fails_for_missing_artifact() -> None:
    with pytest.raises(KeyError, match="missing artifact"):
        artifact_path_by_name({"artifacts": []}, "phase_gate_model")


def test_select_episode_ids_prefers_explicit_order() -> None:
    selected = select_episode_ids(
        available=["episode_73", "episode_74", "episode_75"],
        requested=["episode_75", "episode_73"],
        max_episodes=1,
    )

    assert selected == ["episode_75", "episode_73"]


def test_select_episode_ids_uses_available_prefix_without_explicit_ids() -> None:
    selected = select_episode_ids(
        available=["episode_73", "episode_74", "episode_75"],
        requested=[],
        max_episodes=2,
    )

    assert selected == ["episode_73", "episode_74"]


def test_build_compact_summary_keeps_runtime_gate_metrics() -> None:
    summary = build_compact_summary(
        candidate_id="E37",
        episode_ids=["episode_73", "episode_79"],
        raw_metrics={"overall": {"mae": 0.05}},
        gated_metrics={"overall": {"mae": 0.04}},
        gohome_summary={"event_recall": 1.0, "pre_tail_false_positive_episodes": 0},
        latency_summary={"act_p95_ms": 12.0, "gate_p95_ms": 0.04},
        artifact_manifest="/tmp/manifest.json",
    )

    assert summary["candidate_id"] == "E37"
    assert summary["episodes"] == 2
    assert summary["raw_action_mae"] == 0.05
    assert summary["phase_gated_action_mae"] == 0.04
    assert summary["gohome_pre_tail_false_positive_episodes"] == 0
