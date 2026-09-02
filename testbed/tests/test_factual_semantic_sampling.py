from __future__ import annotations

import pytest

from testbed.data.factual_semantic_sampling import (
    TIER_NAMES,
    resolve_factual_semantic_sampling_config,
)


def test_disabled_sampling_has_no_extra_tiers() -> None:
    assert resolve_factual_semantic_sampling_config(None) == {
        "enabled": False,
        "scope": "train_only",
        "append_samples_per_episode": 0,
        "manifest_path": None,
        "tier_names": TIER_NAMES,
    }


def test_sampling_rejects_non_train_scope(tmp_path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="scope must be train_only"):
        resolve_factual_semantic_sampling_config(
            {
                "enabled": True,
                "scope": "train_and_validation",
                "manifest_path": str(manifest),
                "append_samples_per_episode": 2,
            }
        )
