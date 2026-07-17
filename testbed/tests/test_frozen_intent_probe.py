from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from torch import nn

from testbed.policies.act.frozen_intent_probe import (
    FeatureCache,
    build_cache_identity,
    build_startup_anchor_inventory,
    cache_key,
    capture_query0_decoder_features,
    evaluate_intent_predictions,
    extract_frozen_features,
    load_feature_cache,
    predict_linear_probe,
    save_feature_cache,
    state_dict_bitwise_sha256,
    ternary_intent_labels,
    train_linear_probe,
    validate_startup_anchor_contract,
)

THRESHOLDS = {
    "swing": {"pos": 0.661, "neg": 0.721},
    "boom": {"pos": 0.259, "neg": 0.357},
    "stick": {"pos": 0.5, "neg": 0.5},
    "bucket": {"pos": 0.408, "neg": 0.508},
}


def _identity() -> dict:
    sha = "a" * 64
    return build_cache_identity(
        model_name="eye2",
        checkpoint_sha256=sha,
        resolved_config_sha256=sha,
        stats_sha256=sha,
        split_sha256=sha,
        camera_names=["video4", "video5"],
        image_transform="none",
        train_episode_ids=[73, 74],
        validation_episode_ids=[75],
        thresholds=THRESHOLDS,
        episode_sha256={"73": sha, "74": sha, "75": sha},
    )


def _cache() -> FeatureCache:
    return FeatureCache(
        features=np.arange(24, dtype=np.float32).reshape(3, 8),
        labels=np.asarray(
            [[0, 1, 1, 2], [1, 2, 1, 0], [2, 0, 1, 1]], dtype=np.int8
        ),
        episode_ids=np.asarray([73, 73, 74], dtype=np.int32),
        steps=np.asarray([0, 1, 0], dtype=np.int32),
        anchor_mask=np.zeros((3, 4), dtype=bool),
        startup_mask=np.zeros((3, 4), dtype=bool),
        mid_cycle_mask=np.zeros((3, 4), dtype=bool),
        metadata={"frozen_weights_bitwise_unchanged": True},
    )


def test_ternary_labels_use_neg_idle_pos_and_asymmetric_thresholds() -> None:
    actions = np.asarray(
        [
            [-0.721, -0.356, -0.5, -0.507],
            [-0.720, 0.259, 0.499, 0.408],
        ],
        dtype=np.float32,
    )
    labels = ternary_intent_labels(actions, THRESHOLDS)
    np.testing.assert_array_equal(labels[0], [0, 1, 0, 1])
    np.testing.assert_array_equal(labels[1], [1, 2, 1, 2])


def test_startup_anchor_contract_is_derived_from_episode_actions(
    tmp_path: Path,
) -> None:
    first = np.asarray(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, -0.5, 0.6, 0.0],
            [0.0, -0.5, 0.6, 0.0],
        ],
        dtype=np.float32,
    )
    second = np.asarray(
        [[0.0, 0.0, 0.0, 0.6], [0.0, 0.0, 0.0, 0.6]],
        dtype=np.float32,
    )
    for episode_id, action in ((120, first), (121, second)):
        with h5py.File(tmp_path / f"episode_{episode_id}.hdf5", "w") as handle:
            handle.create_dataset("action", data=action)

    inventory = build_startup_anchor_inventory(
        dataset_dir=tmp_path,
        episode_ids=[120, 121],
        thresholds=THRESHOLDS,
    )
    assert inventory == [
        {
            "episode_id": 120,
            "step": 1,
            "axis": "boom",
            "true_intent": "neg",
        },
        {
            "episode_id": 120,
            "step": 1,
            "axis": "stick",
            "true_intent": "pos",
        },
        {
            "episode_id": 121,
            "step": 0,
            "axis": "bucket",
            "true_intent": "pos",
        },
    ]
    validate_startup_anchor_contract(
        observed_rows=[{**row, "model": "candidate"} for row in reversed(inventory)],
        expected_inventory=inventory,
    )
    with pytest.raises(ValueError, match="startup anchor contract mismatch"):
        validate_startup_anchor_contract(
            observed_rows=inventory[:-1],
            expected_inventory=inventory,
        )


class _FeatureModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.action_head = nn.Linear(4, 2)
        self.actions_seen = "unset"

    def forward(
        self,
        proprio: torch.Tensor,
        image: torch.Tensor,
        env_state: None,
        actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del image, env_state
        self.actions_seen = actions
        hidden = torch.stack([proprio, proprio + 100.0], dim=1)
        return self.action_head(hidden)


def test_feature_capture_uses_query_zero_and_no_future_actions() -> None:
    model = _FeatureModel()
    proprio = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    before = state_dict_bitwise_sha256(model)
    feature = capture_query0_decoder_features(
        model=model,
        proprio=proprio,
        image=torch.zeros(2, 1, 3, 2, 2),
    )
    after = state_dict_bitwise_sha256(model)
    torch.testing.assert_close(feature, proprio)
    assert model.actions_seen is None
    assert before == after


def test_batched_extraction_preserves_frame_metadata_and_frozen_hash() -> None:
    class Adapter:
        device = torch.device("cpu")
        _model = _FeatureModel()
        _proprio_mean = torch.zeros(4)
        _proprio_std = torch.ones(4)
        _normalize = staticmethod(lambda image: image)

    frames = []
    for index in range(3):
        frames.append(
            {
                "image": torch.zeros(1, 3, 2, 2),
                "qpos": torch.full((4,), float(index)),
                "labels": torch.tensor([1, 1, 1, 1], dtype=torch.int8),
                "anchor_mask": torch.tensor(
                    [index == 1, False, False, False]
                ),
                "startup_mask": torch.tensor(
                    [index == 1, False, False, False]
                ),
                "mid_cycle_mask": torch.zeros(4, dtype=torch.bool),
                "episode_id": torch.tensor(73, dtype=torch.int32),
                "step": torch.tensor(index, dtype=torch.int32),
            }
        )
    cache = extract_frozen_features(
        adapter=Adapter(),
        dataset=frames,  # type: ignore[arg-type]
        batch_size=2,
        num_workers=0,
    )
    np.testing.assert_array_equal(cache.episode_ids, [73, 73, 73])
    np.testing.assert_array_equal(cache.steps, [0, 1, 2])
    np.testing.assert_array_equal(cache.features[:, 0], [0.0, 1.0, 2.0])
    assert cache.startup_mask.sum() == 1
    assert cache.metadata["frozen_weights_bitwise_unchanged"] is True


def test_cache_identity_is_split_order_deterministic_and_sensitive() -> None:
    first = _identity()
    second = _identity()
    assert cache_key(first) == cache_key(second)
    reordered = {**first, "train_episode_ids": [74, 73]}
    assert cache_key(first) != cache_key(reordered)


def test_cache_fails_closed_on_identity_or_payload_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "features.npz"
    identity = _identity()
    save_feature_cache(path, cache=_cache(), identity=identity)
    loaded = load_feature_cache(path, expected_identity=identity)
    np.testing.assert_array_equal(loaded.features, _cache().features)
    with pytest.raises(ValueError, match="identity mismatch"):
        load_feature_cache(path, expected_identity={**identity, "partition": "val"})
    with path.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="payload sha256 mismatch"):
        load_feature_cache(path, expected_identity=identity)


def test_linear_probe_training_is_deterministic_and_uses_train_only_counts() -> None:
    rng = np.random.default_rng(7)
    features = rng.normal(size=(24, 6)).astype(np.float32)
    labels = np.ones((24, 4), dtype=np.int8)
    labels[:4, 0] = 0
    labels[4:8, 0] = 2
    labels[:6, 1] = 0
    labels[6:12, 1] = 2
    first = train_linear_probe(
        features, labels, epochs=3, batch_size=8, seed=3, device="cpu"
    )
    second = train_linear_probe(
        features, labels, epochs=3, batch_size=8, seed=3, device="cpu"
    )
    np.testing.assert_array_equal(first.weight, second.weight)
    np.testing.assert_array_equal(first.bias, second.bias)
    assert first.class_weights[2].tolist() == [0.0, 1.0, 0.0]
    probabilities, predictions = predict_linear_probe(first, features)
    assert probabilities.shape == (24, 4, 3)
    assert predictions.shape == (24, 4)


def test_metrics_separate_idle_false_active_and_opposite_direction() -> None:
    labels = np.asarray(
        [
            [0, 1, 1, 2],
            [2, 1, 1, 0],
            [1, 1, 1, 1],
        ],
        dtype=np.int8,
    )
    predicted = np.asarray(
        [
            [0, 2, 1, 0],
            [1, 1, 1, 0],
            [2, 1, 1, 1],
        ],
        dtype=np.int8,
    )
    probabilities = np.full((3, 4, 3), 0.05, dtype=np.float32)
    for row in range(3):
        for axis in range(4):
            probabilities[row, axis, predicted[row, axis]] = 0.9
    anchor = np.zeros((3, 4), dtype=bool)
    anchor[0, 0] = True
    startup = anchor.copy()
    mid = np.zeros_like(anchor)
    metrics = evaluate_intent_predictions(
        labels=labels,
        probabilities=probabilities,
        anchor_mask=anchor,
        startup_mask=startup,
        mid_cycle_mask=mid,
    )
    swing = metrics["scopes"]["all_frames"]["axes"]["swing"]
    assert swing["active_recall"] == pytest.approx(0.5)
    assert swing["idle_false_active_rate"] == pytest.approx(1.0)
    assert swing["opposite_direction_rate"] == pytest.approx(0.0)
    bucket = metrics["scopes"]["all_frames"]["axes"]["bucket"]
    assert bucket["opposite_direction_rate"] == pytest.approx(0.5)
    stick = metrics["scopes"]["all_frames"]["axes"]["stick"]
    assert stick["estimable"] is False
    assert stick["non_estimable_reason"] == "no_active_labels"
