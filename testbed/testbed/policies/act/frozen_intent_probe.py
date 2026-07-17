"""Frozen ACT decoder-feature probes for per-axis executable intent.

This module owns the complete offline probe capability: frame labels, query-0
decoder feature capture, fail-closed feature caches, the fixed linear probe,
and intent metrics.  It deliberately does not project probe predictions back
into ACT actions or runtime commands.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from testbed.data.dataset import _read_camera_image
from testbed.data.image_transforms import build_image_transform
from testbed.policies.offline_eval import AXIS_NAMES
from testbed.policies.state_hold_demo_target import extract_should_move_anchors

INTENT_CLASS_NAMES = ("neg", "idle", "pos")
INTENT_NEG = 0
INTENT_IDLE = 1
INTENT_POS = 2
LABEL_VERSION = "direct_action_asymmetric_deadzone_ternary_v1"
EXTRACTION_CODE_VERSION = "act_query0_inference_decoder_v1"
FEATURE_LAYER = "action_head_input"
FEATURE_QUERY_INDEX = 0
LATENT_MODE = "inference_zero_actions_none"


@dataclass(frozen=True)
class FeatureCache:
    """Arrays and proof metadata captured from a frozen ACT model."""

    features: np.ndarray
    labels: np.ndarray
    episode_ids: np.ndarray
    steps: np.ndarray
    anchor_mask: np.ndarray
    startup_mask: np.ndarray
    mid_cycle_mask: np.ndarray
    metadata: dict[str, Any]


@dataclass(frozen=True)
class TrainedLinearProbe:
    """A fixed-protocol 3-logit-per-axis linear intent head."""

    weight: np.ndarray
    bias: np.ndarray
    class_weights: np.ndarray
    train_loss: list[float]
    epochs: int
    learning_rate: float
    weight_decay: float
    seed: int


def ternary_intent_labels(
    actions: np.ndarray,
    thresholds: Mapping[str, Mapping[str, float]],
) -> np.ndarray:
    """Map direct actions to axis-major ``neg, idle, pos`` labels."""

    action = np.asarray(actions, dtype=np.float32)
    if action.ndim != 2 or action.shape[1] != len(AXIS_NAMES):
        raise ValueError(
            f"actions must have shape (T, {len(AXIS_NAMES)}), got {action.shape}"
        )
    positive, negative = _threshold_arrays(thresholds)
    labels = np.full(action.shape, INTENT_IDLE, dtype=np.int8)
    labels[action >= positive.reshape(1, -1)] = INTENT_POS
    labels[action <= -negative.reshape(1, -1)] = INTENT_NEG
    return labels


def build_startup_anchor_inventory(
    *,
    dataset_dir: str | Path,
    episode_ids: Sequence[int],
    thresholds: Mapping[str, Mapping[str, float]],
) -> list[dict[str, Any]]:
    """Build the model-independent validation startup contract from actions."""

    root = Path(dataset_dir)
    inventory: list[dict[str, Any]] = []
    for episode_id in [int(value) for value in episode_ids]:
        path = root / f"episode_{episode_id}.hdf5"
        if not path.is_file():
            raise FileNotFoundError(path)
        with h5py.File(path, "r") as handle:
            action = np.asarray(handle["action"][()], dtype=np.float32)
        labels = ternary_intent_labels(action, thresholds)
        for anchor in extract_should_move_anchors(action, dict(thresholds)):
            if anchor.group != "startup":
                continue
            true_intent = "pos" if anchor.direction == "pos" else "neg"
            expected_label = INTENT_POS if anchor.direction == "pos" else INTENT_NEG
            if int(labels[anchor.step, anchor.axis_index]) != expected_label:
                raise AssertionError("startup anchor disagrees with ternary label")
            inventory.append(
                {
                    "episode_id": episode_id,
                    "step": int(anchor.step),
                    "axis": str(anchor.axis),
                    "true_intent": true_intent,
                }
            )
    return sorted(inventory, key=_startup_anchor_key)


def validate_startup_anchor_contract(
    *,
    observed_rows: Sequence[Mapping[str, Any]],
    expected_inventory: Sequence[Mapping[str, Any]],
) -> None:
    """Fail closed when formal probe rows differ from the action-derived contract."""

    observed = sorted(
        [
            {
                "episode_id": int(row["episode_id"]),
                "step": int(row["step"]),
                "axis": str(row["axis"]),
                "true_intent": str(row["true_intent"]),
            }
            for row in observed_rows
        ],
        key=_startup_anchor_key,
    )
    expected = sorted(
        [
            {
                "episode_id": int(row["episode_id"]),
                "step": int(row["step"]),
                "axis": str(row["axis"]),
                "true_intent": str(row["true_intent"]),
            }
            for row in expected_inventory
        ],
        key=_startup_anchor_key,
    )
    if observed != expected:
        raise ValueError(
            "validation startup anchor contract mismatch: "
            f"observed={observed!r}, expected={expected!r}"
        )


def build_cache_identity(
    *,
    model_name: str,
    checkpoint_sha256: str,
    resolved_config_sha256: str,
    stats_sha256: str,
    split_sha256: str,
    camera_names: Sequence[str],
    image_transform: str,
    train_episode_ids: Sequence[int],
    validation_episode_ids: Sequence[int],
    thresholds: Mapping[str, Mapping[str, float]],
    episode_sha256: Mapping[str, str],
    frame_limit_per_split: int | None = None,
) -> dict[str, Any]:
    """Build the complete semantic key for a feature cache."""

    return {
        "schema_version": 1,
        "model_name": str(model_name),
        "checkpoint_sha256": _require_sha256(checkpoint_sha256, "checkpoint"),
        "resolved_config_sha256": _require_sha256(
            resolved_config_sha256, "resolved config"
        ),
        "dataset_stats_sha256": _require_sha256(stats_sha256, "dataset stats"),
        "split_sha256": _require_sha256(split_sha256, "split"),
        "camera_names": [str(value) for value in camera_names],
        "image_transform": str(image_transform),
        "train_episode_ids": [int(value) for value in train_episode_ids],
        "validation_episode_ids": [int(value) for value in validation_episode_ids],
        "frame_limit_per_split": (
            None if frame_limit_per_split is None else int(frame_limit_per_split)
        ),
        "episode_sha256": {
            str(key): _require_sha256(value, f"episode {key}")
            for key, value in sorted(episode_sha256.items())
        },
        "label_version": LABEL_VERSION,
        "deadzone_thresholds": _canonical_thresholds(thresholds),
        "extraction_code_version": EXTRACTION_CODE_VERSION,
        "feature_layer": FEATURE_LAYER,
        "feature_query_index": FEATURE_QUERY_INDEX,
        "latent_mode": LATENT_MODE,
    }


def cache_key(identity: Mapping[str, Any]) -> str:
    """Return a deterministic cache key for an identity payload."""

    encoded = json.dumps(
        dict(identity), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_feature_cache(
    cache_path: str | Path,
    *,
    expected_identity: Mapping[str, Any],
) -> FeatureCache:
    """Load a feature cache only when every identity field matches."""

    path = Path(cache_path)
    manifest_path = path.with_suffix(".json")
    if not path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"feature cache is incomplete: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("identity") != dict(expected_identity):
        raise ValueError(f"feature cache identity mismatch: {path}")
    actual_sha = sha256_file(path)
    if manifest.get("npz_sha256") != actual_sha:
        raise ValueError(f"feature cache payload sha256 mismatch: {path}")
    with np.load(path, allow_pickle=False) as payload:
        cache = FeatureCache(
            features=np.asarray(payload["features"], dtype=np.float32),
            labels=np.asarray(payload["labels"], dtype=np.int8),
            episode_ids=np.asarray(payload["episode_ids"], dtype=np.int32),
            steps=np.asarray(payload["steps"], dtype=np.int32),
            anchor_mask=np.asarray(payload["anchor_mask"], dtype=bool),
            startup_mask=np.asarray(payload["startup_mask"], dtype=bool),
            mid_cycle_mask=np.asarray(payload["mid_cycle_mask"], dtype=bool),
            metadata=dict(manifest),
        )
    _validate_feature_cache(cache)
    return cache


def save_feature_cache(
    cache_path: str | Path,
    *,
    cache: FeatureCache,
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Atomically persist arrays plus their exact semantic identity."""

    _validate_feature_cache(cache)
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez(
            handle,
            features=np.asarray(cache.features, dtype=np.float32),
            labels=np.asarray(cache.labels, dtype=np.int8),
            episode_ids=np.asarray(cache.episode_ids, dtype=np.int32),
            steps=np.asarray(cache.steps, dtype=np.int32),
            anchor_mask=np.asarray(cache.anchor_mask, dtype=bool),
            startup_mask=np.asarray(cache.startup_mask, dtype=bool),
            mid_cycle_mask=np.asarray(cache.mid_cycle_mask, dtype=bool),
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    manifest = {
        "schema_version": 1,
        "complete": True,
        "identity": dict(identity),
        "cache_key": cache_key(identity),
        "npz_sha256": sha256_file(path),
        "frame_count": int(cache.features.shape[0]),
        "feature_dim": int(cache.features.shape[1]),
        "class_counts": class_counts(cache.labels),
        "anchor_count": int(cache.anchor_mask.sum()),
        "startup_anchor_count": int(cache.startup_mask.sum()),
        "mid_cycle_anchor_count": int(cache.mid_cycle_mask.sum()),
        "extraction": dict(cache.metadata),
    }
    _atomic_json(path.with_suffix(".json"), manifest)
    return manifest


class FrozenIntentFrameDataset(Dataset[dict[str, torch.Tensor]]):
    """Deterministic all-frame view over fixed episodes for feature capture."""

    def __init__(
        self,
        *,
        dataset_dir: str | Path,
        episode_ids: Sequence[int],
        camera_names: Sequence[str],
        thresholds: Mapping[str, Mapping[str, float]],
        image_transform: str = "none",
        max_frames: int | None = None,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.episode_ids = [int(value) for value in episode_ids]
        self.camera_names = [str(value) for value in camera_names]
        if not self.episode_ids:
            raise ValueError("episode_ids must not be empty")
        if not self.camera_names:
            raise ValueError("camera_names must not be empty")
        self.image_transform_name = str(image_transform or "none")
        self.image_transform = build_image_transform(self.image_transform_name)
        self._frames: list[tuple[int, int]] = []
        self._qpos: dict[int, np.ndarray] = {}
        self._labels: dict[int, np.ndarray] = {}
        self._anchor: dict[int, np.ndarray] = {}
        self._startup: dict[int, np.ndarray] = {}
        self._mid_cycle: dict[int, np.ndarray] = {}
        self._handles: dict[int, h5py.File] = {}

        for episode_id in self.episode_ids:
            path = self.dataset_dir / f"episode_{episode_id}.hdf5"
            with h5py.File(path, "r") as handle:
                action = np.asarray(handle["action"][()], dtype=np.float32)
                qpos = np.asarray(handle["observations/qpos"][()], dtype=np.float32)
                if qpos.shape != action.shape:
                    raise ValueError(
                        f"episode {episode_id} qpos/action shape mismatch: "
                        f"{qpos.shape} vs {action.shape}"
                    )
                for camera_name in self.camera_names:
                    raw = f"observations/images/{camera_name}"
                    encoded = f"observations/encoded_images/{camera_name}"
                    camera_path = raw if raw in handle else encoded
                    if camera_path not in handle:
                        raise KeyError(
                            f"episode {episode_id} is missing camera {camera_name}"
                        )
                    if int(handle[camera_path].shape[0]) != int(action.shape[0]):
                        raise ValueError(
                            f"episode {episode_id} camera {camera_name} length mismatch"
                        )
            labels = ternary_intent_labels(action, thresholds)
            anchor = np.zeros_like(labels, dtype=bool)
            startup = np.zeros_like(labels, dtype=bool)
            mid_cycle = np.zeros_like(labels, dtype=bool)
            for item in extract_should_move_anchors(action, dict(thresholds)):
                anchor[item.step, item.axis_index] = True
                if item.group == "startup":
                    startup[item.step, item.axis_index] = True
                else:
                    mid_cycle[item.step, item.axis_index] = True
                expected = INTENT_POS if item.direction == "pos" else INTENT_NEG
                if int(labels[item.step, item.axis_index]) != expected:
                    raise AssertionError(
                        "anchor direction disagrees with ternary label"
                    )
            self._qpos[episode_id] = qpos
            self._labels[episode_id] = labels
            self._anchor[episode_id] = anchor
            self._startup[episode_id] = startup
            self._mid_cycle[episode_id] = mid_cycle
            self._frames.extend((episode_id, step) for step in range(len(action)))

        if max_frames is not None:
            if int(max_frames) <= 0:
                raise ValueError("max_frames must be positive")
            self._frames = self._frames[: int(max_frames)]

    def __len__(self) -> int:
        return len(self._frames)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        episode_id, step = self._frames[int(index)]
        handle = self._handle(episode_id)
        images: list[np.ndarray] = []
        for camera_name in self.camera_names:
            image = _read_camera_image(handle, camera_name, step)
            if self.image_transform is not None:
                image = self.image_transform(image)
            images.append(np.asarray(image, dtype=np.uint8))
        stacked = np.stack(images, axis=0)
        image_tensor = (
            torch.from_numpy(stacked)
            .permute(0, 3, 1, 2)
            .contiguous()
            .float()
            .div_(255.0)
        )
        return {
            "image": image_tensor,
            "qpos": torch.from_numpy(self._qpos[episode_id][step]).float(),
            "labels": torch.from_numpy(self._labels[episode_id][step].copy()),
            "anchor_mask": torch.from_numpy(self._anchor[episode_id][step].copy()),
            "startup_mask": torch.from_numpy(self._startup[episode_id][step].copy()),
            "mid_cycle_mask": torch.from_numpy(
                self._mid_cycle[episode_id][step].copy()
            ),
            "episode_id": torch.tensor(episode_id, dtype=torch.int32),
            "step": torch.tensor(step, dtype=torch.int32),
        }

    def _handle(self, episode_id: int) -> h5py.File:
        handle = self._handles.get(int(episode_id))
        if handle is None:
            handle = h5py.File(self.dataset_dir / f"episode_{episode_id}.hdf5", "r")
            self._handles[int(episode_id)] = handle
        return handle

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def __del__(self) -> None:
        self.close()


def capture_query0_decoder_features(
    *,
    model: nn.Module,
    proprio: torch.Tensor,
    image: torch.Tensor,
) -> torch.Tensor:
    """Capture action-head input at query 0 from an inference-only forward."""

    action_head = getattr(model, "action_head", None)
    if not isinstance(action_head, nn.Module):
        raise TypeError("ACT model does not expose an action_head module")
    captured: list[torch.Tensor] = []

    def hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        if len(inputs) != 1 or inputs[0].ndim != 3:
            raise ValueError("action_head input must have shape (B, Q, H)")
        captured.append(inputs[0].detach())

    handle = action_head.register_forward_pre_hook(hook)
    try:
        with torch.inference_mode():
            # Omitting actions is the causal contract: DETRVAE creates a zero
            # latent and cannot encode any future expert-action chunk.
            model(proprio, image, None)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError(
            f"expected one action-head feature capture, observed {len(captured)}"
        )
    hidden = captured[0]
    if hidden.shape[1] <= FEATURE_QUERY_INDEX:
        raise ValueError("ACT decoder did not produce query 0")
    return hidden[:, FEATURE_QUERY_INDEX].clone()


def extract_frozen_features(
    *,
    adapter: Any,
    dataset: FrozenIntentFrameDataset,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int = 2,
) -> FeatureCache:
    """Extract frozen decoder features with the accepted worker path."""

    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    if int(num_workers) < 0:
        raise ValueError("num_workers must be non-negative")
    model = adapter._model
    model.eval()
    before_sha = state_dict_bitwise_sha256(model)
    loader_kwargs: dict[str, Any] = {
        "batch_size": int(batch_size),
        "shuffle": False,
        "num_workers": int(num_workers),
        "pin_memory": True,
    }
    if int(num_workers) > 0:
        loader_kwargs.update(
            {
                "persistent_workers": True,
                "prefetch_factor": int(prefetch_factor),
            }
        )
    loader = DataLoader(dataset, **loader_kwargs)
    device = adapter.device
    features: list[np.ndarray] = []
    arrays: dict[str, list[np.ndarray]] = {
        key: []
        for key in (
            "labels",
            "episode_ids",
            "steps",
            "anchor_mask",
            "startup_mask",
            "mid_cycle_mask",
        )
    }
    started = time.perf_counter()
    for batch in loader:
        proprio = batch["qpos"].to(device, non_blocking=True)
        proprio = (proprio - adapter._proprio_mean) / adapter._proprio_std
        image = batch["image"].to(device, non_blocking=True)
        image = adapter._normalize(image)
        query0 = capture_query0_decoder_features(
            model=model,
            proprio=proprio,
            image=image,
        )
        features.append(query0.cpu().numpy().astype(np.float32, copy=False))
        for key in arrays:
            batch_key = {
                "episode_ids": "episode_id",
                "steps": "step",
            }.get(key, key)
            arrays[key].append(batch[batch_key].cpu().numpy())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    wall_seconds = time.perf_counter() - started
    after_sha = state_dict_bitwise_sha256(model)
    if before_sha != after_sha:
        raise RuntimeError("ACT model parameters changed during frozen extraction")
    return FeatureCache(
        features=np.concatenate(features, axis=0),
        labels=np.concatenate(arrays["labels"], axis=0).astype(np.int8),
        episode_ids=np.concatenate(arrays["episode_ids"], axis=0).astype(np.int32),
        steps=np.concatenate(arrays["steps"], axis=0).astype(np.int32),
        anchor_mask=np.concatenate(arrays["anchor_mask"], axis=0).astype(bool),
        startup_mask=np.concatenate(arrays["startup_mask"], axis=0).astype(bool),
        mid_cycle_mask=np.concatenate(arrays["mid_cycle_mask"], axis=0).astype(bool),
        metadata={
            "wall_seconds": float(wall_seconds),
            "batch_size": int(batch_size),
            "num_workers": int(num_workers),
            "prefetch_factor": (int(prefetch_factor) if int(num_workers) > 0 else None),
            "model_state_sha256_before": before_sha,
            "model_state_sha256_after": after_sha,
            "frozen_weights_bitwise_unchanged": True,
            "actions_argument": None,
            "query_index": FEATURE_QUERY_INDEX,
        },
    )


def class_counts(labels: np.ndarray) -> dict[str, dict[str, int]]:
    """Return exact per-axis ternary counts."""

    label = _validate_labels(labels)
    return {
        axis: {
            name: int(np.count_nonzero(label[:, axis_index] == class_index))
            for class_index, name in enumerate(INTENT_CLASS_NAMES)
        }
        for axis_index, axis in enumerate(AXIS_NAMES)
    }


def train_linear_probe(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    epochs: int = 50,
    learning_rate: float = 3e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 1024,
    seed: int = 0,
    device: str = "cuda",
) -> TrainedLinearProbe:
    """Train the predeclared deterministic class-weighted linear head."""

    x = np.asarray(features, dtype=np.float32)
    y = _validate_labels(labels)
    if x.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError(f"features/labels shape mismatch: {x.shape} vs {y.shape}")
    if int(epochs) <= 0 or int(batch_size) <= 0:
        raise ValueError("epochs and batch_size must be positive")
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    target_device = torch.device(str(device) if torch.cuda.is_available() else "cpu")
    head = nn.Linear(x.shape[1], len(AXIS_NAMES) * len(INTENT_CLASS_NAMES)).to(
        target_device
    )
    weights = _inverse_frequency_class_weights(y)
    weight_tensor = torch.from_numpy(weights).to(target_device)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    x_tensor = torch.from_numpy(x)
    y_tensor = torch.from_numpy(y.astype(np.int64))
    losses: list[float] = []
    generator = torch.Generator().manual_seed(int(seed))
    for _epoch in range(int(epochs)):
        permutation = torch.randperm(x.shape[0], generator=generator)
        total_loss = 0.0
        total_rows = 0
        for start in range(0, x.shape[0], int(batch_size)):
            indices = permutation[start : start + int(batch_size)]
            batch_x = x_tensor[indices].to(target_device, non_blocking=True)
            batch_y = y_tensor[indices].to(target_device, non_blocking=True)
            logits = head(batch_x).reshape(-1, len(AXIS_NAMES), len(INTENT_CLASS_NAMES))
            loss = torch.zeros((), device=target_device)
            for axis_index in range(len(AXIS_NAMES)):
                loss = loss + nn.functional.cross_entropy(
                    logits[:, axis_index],
                    batch_y[:, axis_index],
                    weight=weight_tensor[axis_index],
                )
            loss = loss / len(AXIS_NAMES)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            rows = int(batch_x.shape[0])
            total_loss += float(loss.detach().cpu()) * rows
            total_rows += rows
        losses.append(total_loss / max(1, total_rows))
    return TrainedLinearProbe(
        weight=head.weight.detach().cpu().numpy().astype(np.float32),
        bias=head.bias.detach().cpu().numpy().astype(np.float32),
        class_weights=weights,
        train_loss=losses,
        epochs=int(epochs),
        learning_rate=float(learning_rate),
        weight_decay=float(weight_decay),
        seed=int(seed),
    )


def predict_linear_probe(
    probe: TrainedLinearProbe,
    features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return axis-major probabilities and argmax labels."""

    x = np.asarray(features, dtype=np.float32)
    logits = x @ probe.weight.T + probe.bias.reshape(1, -1)
    logits = logits.reshape(-1, len(AXIS_NAMES), len(INTENT_CLASS_NAMES))
    logits = logits - logits.max(axis=2, keepdims=True)
    exp = np.exp(logits.astype(np.float64))
    probabilities = exp / exp.sum(axis=2, keepdims=True)
    predictions = probabilities.argmax(axis=2).astype(np.int8)
    return probabilities.astype(np.float32), predictions


def evaluate_intent_predictions(
    *,
    labels: np.ndarray,
    probabilities: np.ndarray,
    anchor_mask: np.ndarray,
    startup_mask: np.ndarray,
    mid_cycle_mask: np.ndarray,
) -> dict[str, Any]:
    """Compute all-frame and transition-anchor executable-intent metrics."""

    truth = _validate_labels(labels)
    probs = np.asarray(probabilities, dtype=np.float64)
    expected = (truth.shape[0], len(AXIS_NAMES), len(INTENT_CLASS_NAMES))
    if probs.shape != expected:
        raise ValueError(f"probabilities must have shape {expected}, got {probs.shape}")
    if not np.all(np.isfinite(probs)):
        raise ValueError("probabilities must be finite")
    prediction = probs.argmax(axis=2).astype(np.int8)
    masks = {
        "all_frames": np.ones_like(truth, dtype=bool),
        "transition_anchors": np.asarray(anchor_mask, dtype=bool),
        "startup_anchors": np.asarray(startup_mask, dtype=bool),
        "mid_cycle_anchors": np.asarray(mid_cycle_mask, dtype=bool),
    }
    result = {
        "class_order": list(INTENT_CLASS_NAMES),
        "axis_order": list(AXIS_NAMES),
        "scopes": {},
    }
    for scope_name, scope_mask in masks.items():
        if scope_mask.shape != truth.shape:
            raise ValueError(f"{scope_name} mask shape mismatch")
        axes: dict[str, Any] = {}
        weighted_true: list[np.ndarray] = []
        weighted_pred: list[np.ndarray] = []
        weighted_prob: list[np.ndarray] = []
        for axis_index, axis in enumerate(AXIS_NAMES):
            selected = scope_mask[:, axis_index]
            axis_true = truth[selected, axis_index]
            axis_pred = prediction[selected, axis_index]
            axis_prob = probs[selected, axis_index]
            axes[axis] = _axis_metrics(axis_true, axis_pred, axis_prob)
            if axis_true.size:
                weighted_true.append(axis_true)
                weighted_pred.append(axis_pred)
                weighted_prob.append(axis_prob)
        if weighted_true:
            aggregate = _axis_metrics(
                np.concatenate(weighted_true),
                np.concatenate(weighted_pred),
                np.concatenate(weighted_prob),
            )
        else:
            aggregate = _axis_metrics(
                np.empty(0, dtype=np.int8),
                np.empty(0, dtype=np.int8),
                np.empty((0, 3), dtype=np.float64),
            )
        result["scopes"][scope_name] = {
            "sample_count": int(scope_mask.sum()),
            "aggregate": aggregate,
            "axes": axes,
        }

    true_active = truth != INTENT_IDLE
    predicted_active = prediction != INTENT_IDLE
    double_mask = true_active.sum(axis=1) >= 2
    true_axes_preserved = np.all((prediction == truth) | ~true_active, axis=1)[
        double_mask
    ]
    exact_vector = np.all(prediction == truth, axis=1)[double_mask]
    result["double_axis"] = {
        "frame_count": int(double_mask.sum()),
        "active_axes_preserved_rate": _safe_mean(true_axes_preserved),
        "exact_intent_vector_rate": _safe_mean(exact_vector),
        "predicted_double_active_rate": _safe_mean(
            predicted_active.sum(axis=1)[double_mask] >= 2
        ),
    }
    return result


def save_probe_weights(
    path: str | Path,
    *,
    probe: TrainedLinearProbe,
) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        np.savez(
            handle,
            weight=probe.weight,
            bias=probe.bias,
            class_weights=probe.class_weights,
            train_loss=np.asarray(probe.train_loss, dtype=np.float64),
        )
    return sha256_file(output)


def write_metrics_csv(
    path: str | Path,
    *,
    model_metrics: Mapping[str, Mapping[str, Any]],
) -> None:
    rows: list[dict[str, Any]] = []
    for model_name, metrics in model_metrics.items():
        for scope_name, scope in metrics["scopes"].items():
            for axis, values in scope["axes"].items():
                rows.append(
                    {
                        "model": model_name,
                        "scope": scope_name,
                        "axis": axis,
                        "sample_count": values["sample_count"],
                        "active_count": values["active_count"],
                        "estimable": values["estimable"],
                        "active_recall": values["active_recall"],
                        "idle_false_active_rate": values["idle_false_active_rate"],
                        "opposite_direction_rate": values["opposite_direction_rate"],
                        "macro_f1": values["macro_f1"],
                        "balanced_accuracy": values["balanced_accuracy"],
                        "mean_confidence": values["mean_confidence"],
                        "mean_margin": values["mean_margin"],
                    }
                )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def anchor_prediction_rows(
    *,
    model_name: str,
    cache: FeatureCache,
    probabilities: np.ndarray,
) -> list[dict[str, Any]]:
    """Return one reviewable row per validation transition anchor."""

    probs = np.asarray(probabilities, dtype=np.float64)
    truth = _validate_labels(cache.labels)
    if probs.shape != (truth.shape[0], len(AXIS_NAMES), 3):
        raise ValueError("anchor probabilities shape mismatch")
    prediction = probs.argmax(axis=2)
    rows: list[dict[str, Any]] = []
    for frame_index, axis_index in np.argwhere(cache.anchor_mask):
        true_class = int(truth[frame_index, axis_index])
        predicted_class = int(prediction[frame_index, axis_index])
        group = (
            "startup"
            if bool(cache.startup_mask[frame_index, axis_index])
            else "mid_cycle"
        )
        rows.append(
            {
                "model": str(model_name),
                "episode_id": int(cache.episode_ids[frame_index]),
                "step": int(cache.steps[frame_index]),
                "group": group,
                "axis": AXIS_NAMES[axis_index],
                "true_intent": INTENT_CLASS_NAMES[true_class],
                "predicted_intent": INTENT_CLASS_NAMES[predicted_class],
                "correct": bool(true_class == predicted_class),
                "prob_neg": float(probs[frame_index, axis_index, INTENT_NEG]),
                "prob_idle": float(probs[frame_index, axis_index, INTENT_IDLE]),
                "prob_pos": float(probs[frame_index, axis_index, INTENT_POS]),
                "confidence": float(probs[frame_index, axis_index].max()),
                "margin": float(np.diff(np.sort(probs[frame_index, axis_index]))[-1]),
            }
        )
    return rows


def write_rows_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    materialized = [dict(row) for row in rows]
    fieldnames = list(materialized[0]) if materialized else []
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(materialized)


def write_compact_plots(
    output_dir: str | Path,
    *,
    model_metrics: Mapping[str, Mapping[str, Any]],
) -> list[Path]:
    """Write compact confusion and transition-recall review plots."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model_names = list(model_metrics)
    figure, axes = plt.subplots(
        len(model_names),
        len(AXIS_NAMES),
        figsize=(3.0 * len(AXIS_NAMES), 2.8 * len(model_names)),
        squeeze=False,
    )
    for row_index, model_name in enumerate(model_names):
        scope = model_metrics[model_name]["scopes"]["all_frames"]
        for axis_index, axis_name in enumerate(AXIS_NAMES):
            matrix = np.asarray(
                scope["axes"][axis_name]["confusion_matrix"], dtype=np.int64
            )
            axis = axes[row_index, axis_index]
            axis.imshow(matrix, cmap="Blues")
            for true_index in range(3):
                for predicted_index in range(3):
                    axis.text(
                        predicted_index,
                        true_index,
                        str(int(matrix[true_index, predicted_index])),
                        ha="center",
                        va="center",
                        fontsize=8,
                    )
            axis.set_title(f"{model_name}\n{axis_name}", fontsize=9)
            axis.set_xticks(range(3), INTENT_CLASS_NAMES, rotation=30)
            axis.set_yticks(range(3), INTENT_CLASS_NAMES)
            axis.set_xlabel("predicted")
            axis.set_ylabel("true")
    figure.tight_layout()
    confusion_path = output / "validation_all_frame_confusions.png"
    figure.savefig(confusion_path, dpi=150)
    plt.close(figure)

    scope_names = ("startup_anchors", "mid_cycle_anchors")
    x = np.arange(len(model_names), dtype=np.float64)
    width = 0.35
    figure, axis = plt.subplots(figsize=(max(7.0, 2.2 * len(model_names)), 4.0))
    for offset, scope_name in enumerate(scope_names):
        recalls = []
        for model_name in model_names:
            value = model_metrics[model_name]["scopes"][scope_name]["aggregate"][
                "active_recall"
            ]
            recalls.append(np.nan if value is None else float(value))
        axis.bar(
            x + (offset - 0.5) * width,
            recalls,
            width,
            label=scope_name.replace("_anchors", ""),
        )
    axis.set_xticks(x, model_names, rotation=15)
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("exact active-direction recall")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    anchor_path = output / "validation_anchor_recall.png"
    figure.savefig(anchor_path, dpi=150)
    plt.close(figure)
    return [confusion_path, anchor_path]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_dict_bitwise_sha256(model: nn.Module) -> str:
    """Hash names, tensor metadata, and exact parameter/buffer bytes."""

    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _inverse_frequency_class_weights(labels: np.ndarray) -> np.ndarray:
    """N/(K*n_c), normalized over present classes, derived from train only."""

    label = _validate_labels(labels)
    weights = np.zeros((len(AXIS_NAMES), len(INTENT_CLASS_NAMES)), dtype=np.float32)
    for axis_index in range(len(AXIS_NAMES)):
        counts = np.bincount(
            label[:, axis_index], minlength=len(INTENT_CLASS_NAMES)
        ).astype(np.float64)
        present = counts > 0
        raw = np.zeros_like(counts)
        raw[present] = counts.sum() / (present.sum() * counts[present])
        raw[present] /= raw[present].mean()
        weights[axis_index] = raw.astype(np.float32)
    return weights


def _axis_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, Any]:
    truth = np.asarray(truth, dtype=np.int8).reshape(-1)
    prediction = np.asarray(prediction, dtype=np.int8).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1, 3)
    confusion = np.zeros((3, 3), dtype=np.int64)
    for true_value, predicted_value in zip(truth, prediction):
        confusion[int(true_value), int(predicted_value)] += 1
    active = truth != INTENT_IDLE
    idle = truth == INTENT_IDLE
    opposite = ((truth == INTENT_NEG) & (prediction == INTENT_POS)) | (
        (truth == INTENT_POS) & (prediction == INTENT_NEG)
    )
    recalls: list[float] = []
    f1_scores: list[float] = []
    supported_classes: list[str] = []
    for class_index, class_name in enumerate(INTENT_CLASS_NAMES):
        support = int(confusion[class_index].sum())
        if support == 0:
            continue
        supported_classes.append(class_name)
        true_positive = int(confusion[class_index, class_index])
        false_positive = int(confusion[:, class_index].sum()) - true_positive
        recall = true_positive / support
        precision_denominator = true_positive + false_positive
        precision = (
            true_positive / precision_denominator if precision_denominator else 0.0
        )
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        recalls.append(float(recall))
        f1_scores.append(float(f1))
    sorted_probability = np.sort(probabilities, axis=1) if truth.size else probabilities
    margins = (
        sorted_probability[:, -1] - sorted_probability[:, -2]
        if truth.size
        else np.empty(0)
    )
    confidence = probabilities.max(axis=1) if truth.size else np.empty(0)
    active_count = int(active.sum())
    estimable = bool(active_count > 0)
    return {
        "sample_count": int(truth.size),
        "active_count": active_count,
        "idle_count": int(idle.sum()),
        "estimable": estimable,
        "non_estimable_reason": None if estimable else "no_active_labels",
        "supported_classes": supported_classes,
        "confusion_matrix": confusion.tolist(),
        "active_recall": (
            float(np.mean(prediction[active] == truth[active]))
            if active_count
            else None
        ),
        "idle_false_active_rate": (
            float(np.mean(prediction[idle] != INTENT_IDLE)) if np.any(idle) else None
        ),
        "opposite_direction_rate": (
            float(np.mean(opposite[active])) if active_count else None
        ),
        "macro_f1": float(np.mean(f1_scores)) if f1_scores else None,
        "balanced_accuracy": float(np.mean(recalls)) if recalls else None,
        "mean_confidence": _safe_mean(confidence),
        "mean_margin": _safe_mean(margins),
    }


def _safe_mean(values: np.ndarray) -> float | None:
    arr = np.asarray(values)
    return float(np.mean(arr)) if arr.size else None


def _startup_anchor_key(row: Mapping[str, Any]) -> tuple[int, int, int, str]:
    axis = str(row["axis"])
    if axis not in AXIS_NAMES:
        raise ValueError(f"unknown startup anchor axis: {axis!r}")
    true_intent = str(row["true_intent"])
    if true_intent not in {"neg", "pos"}:
        raise ValueError(
            f"startup anchor true_intent must be neg or pos, got {true_intent!r}"
        )
    return (
        int(row["episode_id"]),
        int(row["step"]),
        AXIS_NAMES.index(axis),
        true_intent,
    )


def _validate_labels(labels: np.ndarray) -> np.ndarray:
    label = np.asarray(labels, dtype=np.int8)
    if label.ndim != 2 or label.shape[1] != len(AXIS_NAMES):
        raise ValueError(
            f"labels must have shape (N, {len(AXIS_NAMES)}), got {label.shape}"
        )
    if label.size and (int(label.min()) < 0 or int(label.max()) >= 3):
        raise ValueError("labels must use neg=0, idle=1, pos=2")
    return label


def _validate_feature_cache(cache: FeatureCache) -> None:
    features = np.asarray(cache.features)
    labels = _validate_labels(cache.labels)
    if features.ndim != 2 or features.shape[0] != labels.shape[0]:
        raise ValueError("feature cache features/labels shape mismatch")
    for name, value, expected_tail in (
        ("episode_ids", cache.episode_ids, ()),
        ("steps", cache.steps, ()),
        ("anchor_mask", cache.anchor_mask, (len(AXIS_NAMES),)),
        ("startup_mask", cache.startup_mask, (len(AXIS_NAMES),)),
        ("mid_cycle_mask", cache.mid_cycle_mask, (len(AXIS_NAMES),)),
    ):
        if np.asarray(value).shape != (features.shape[0], *expected_tail):
            raise ValueError(f"feature cache {name} shape mismatch")
    if not np.all(np.isfinite(features)):
        raise ValueError("feature cache contains non-finite features")


def _threshold_arrays(
    thresholds: Mapping[str, Mapping[str, float]],
) -> tuple[np.ndarray, np.ndarray]:
    canonical = _canonical_thresholds(thresholds)
    positive = np.asarray(
        [canonical[axis]["pos"] for axis in AXIS_NAMES], dtype=np.float32
    )
    negative = np.asarray(
        [canonical[axis]["neg"] for axis in AXIS_NAMES], dtype=np.float32
    )
    return positive, negative


def _canonical_thresholds(
    thresholds: Mapping[str, Mapping[str, float]],
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for axis in AXIS_NAMES:
        raw = thresholds.get(axis)
        if not isinstance(raw, Mapping):
            raise ValueError(f"thresholds missing axis {axis!r}")
        pos = float(raw.get("pos", -1.0))
        neg = float(raw.get("neg", -1.0))
        if not np.isfinite(pos) or not np.isfinite(neg) or pos < 0 or neg < 0:
            raise ValueError(f"invalid asymmetric thresholds for axis {axis}")
        result[axis] = {"pos": pos, "neg": neg}
    return result


def _require_sha256(value: str, name: str) -> str:
    text = str(value)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ValueError(f"{name} sha256 is invalid")
    return text


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
