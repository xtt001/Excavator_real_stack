"""
ACT policy adapter.

Wraps the original ACTPolicy (nn.Module) behind the testbed Policy ABC so
that CLI tools and adapter tests can call policy.predict(obs) without knowing
any ACT internals.

Temporal aggregation
--------------------
When `temporal_agg=True`, actions are chunked and averaged using the
scheme from the original paper: at each step we query the model at
frequency 1, accumulate the chunk into a (T, T+C, Na) tensor, then
select the weighted average of all past predictions for the current step.
Live control can run longer than the resolved training `episode_len`, so the
aggregation tensor grows at runtime instead of treating that length as a hard
limit.
"""

from __future__ import annotations

from contextlib import nullcontext
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from einops import rearrange
import torchvision.transforms as transforms

from testbed.policies.base import Policy, register_policy


def _kl_divergence(mu, logvar):
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))
    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld       = klds.sum(1).mean(0, True)
    dimension_wise  = klds.mean(0)
    mean_kld        = klds.mean(1).mean(0, True)
    return total_kld, dimension_wise, mean_kld


_AXIS_NAMES = ("swing", "boom", "stick", "bucket")
_INFERENCE_PRECISIONS = ("fp32", "fp16")


def _resolve_inference_autocast_dtype(
    value: Any,
    *,
    device: torch.device,
) -> torch.dtype | None:
    precision = str(value or "fp32").strip().lower()
    if precision not in _INFERENCE_PRECISIONS:
        raise ValueError(
            "inference_precision must be one of "
            f"{_INFERENCE_PRECISIONS}, got {value!r}"
        )
    if precision == "fp16":
        if device.type != "cuda":
            raise ValueError(
                "inference_precision='fp16' requires a CUDA device; "
                f"resolved device is {device}."
            )
        return torch.float16
    return None


def _resolve_deadzone_loss_config(raw: Any) -> dict[str, Any]:
    cfg = dict(raw or {})
    enabled = bool(cfg.get("enabled", False))
    if not enabled:
        return {
            "enabled": False,
            "same_dir_promote_weight": 0.0,
            "idle_suppression_weight": 0.0,
            "wrong_effective_weight": 0.0,
            "margin": torch.zeros(4, dtype=torch.float32),
            "pos": torch.zeros(4, dtype=torch.float32),
            "neg": torch.zeros(4, dtype=torch.float32),
            "penalize_wrong_effective": False,
            "idle_denominator": "crossing_axes",
            "wrong_denominator": "crossing_axes",
        }

    thresholds = _load_deadzone_thresholds(cfg)
    margin = _broadcast_axis_values(cfg.get("margin", 0.0), name="deadzone_loss.margin")
    apply_when = str(cfg.get("apply_when", "expert_effective_only"))
    if apply_when != "expert_effective_only":
        raise ValueError(
            "deadzone_loss.apply_when currently supports only 'expert_effective_only'."
        )
    legacy_weight = float(cfg.get("weight", 0.1))
    legacy_wrong_weight = float(cfg.get("wrong_weight", legacy_weight))
    return {
        "enabled": True,
        "same_dir_promote_weight": float(cfg.get("same_dir_promote_weight", legacy_weight)),
        "idle_suppression_weight": float(cfg.get("idle_suppression_weight", legacy_wrong_weight)),
        "wrong_effective_weight": float(cfg.get("wrong_effective_weight", legacy_wrong_weight)),
        "margin": torch.as_tensor(margin, dtype=torch.float32),
        "pos": torch.as_tensor([thresholds[axis]["pos"] for axis in _AXIS_NAMES], dtype=torch.float32),
        "neg": torch.as_tensor([thresholds[axis]["neg"] for axis in _AXIS_NAMES], dtype=torch.float32),
        "penalize_wrong_effective": bool(cfg.get("penalize_wrong_effective", True)),
        "same_dir_window": _resolve_same_dir_window(cfg.get("same_dir_window", "all")),
        "same_dir_window_steps": int(cfg.get("same_dir_window_steps", 1)),
        "idle_denominator": _resolve_denominator(
            cfg.get("idle_denominator", "crossing_axes"),
            name="deadzone_loss.idle_denominator",
            allowed={"crossing_axes", "all_idle_axes"},
        ),
        "wrong_denominator": _resolve_denominator(
            cfg.get("wrong_denominator", "crossing_axes"),
            name="deadzone_loss.wrong_denominator",
            allowed={"crossing_axes", "all_wrong_candidate_axes"},
        ),
    }


def _resolve_intent_loss_config(raw: Any) -> dict[str, Any]:
    cfg = dict(raw or {})
    enabled = bool(cfg.get("enabled", False))
    if not enabled:
        return {
            "enabled": False,
            "weight": 0.0,
            "positive_weight": torch.ones(8, dtype=torch.float32),
            "intent_dim": 0,
            "pos": torch.zeros(4, dtype=torch.float32),
            "neg": torch.zeros(4, dtype=torch.float32),
        }

    thresholds = _load_deadzone_thresholds(cfg)
    positive_weight = _broadcast_intent_values(
        cfg.get("positive_weight", 1.0),
        name="intent_loss.positive_weight",
    )
    return {
        "enabled": True,
        "weight": float(cfg.get("weight", 0.05)),
        "positive_weight": torch.as_tensor(positive_weight, dtype=torch.float32),
        "intent_dim": 8,
        "pos": torch.as_tensor([thresholds[axis]["pos"] for axis in _AXIS_NAMES], dtype=torch.float32),
        "neg": torch.as_tensor([thresholds[axis]["neg"] for axis in _AXIS_NAMES], dtype=torch.float32),
    }


def _resolve_window_deadzone_loss_config(raw: Any) -> dict[str, Any]:
    cfg = dict(raw or {})
    enabled = bool(cfg.get("enabled", False))
    if not enabled:
        return {
            "enabled": False,
            "same_dir_promote_weight": 0.0,
            "stop_suppression_weight": 0.0,
            "wrong_effective_weight": 0.0,
            "margin": torch.zeros(4, dtype=torch.float32),
            "pos": torch.zeros(4, dtype=torch.float32),
            "neg": torch.zeros(4, dtype=torch.float32),
        }
    thresholds = _load_deadzone_thresholds(cfg)
    margin = _broadcast_axis_values(cfg.get("margin", 0.0), name="window_deadzone_loss.margin")
    return {
        "enabled": True,
        "same_dir_promote_weight": float(cfg.get("same_dir_promote_weight", 0.1)),
        "stop_suppression_weight": float(cfg.get("stop_suppression_weight", 0.05)),
        "wrong_effective_weight": float(cfg.get("wrong_effective_weight", 0.05)),
        "margin": torch.as_tensor(margin, dtype=torch.float32),
        "pos": torch.as_tensor([thresholds[axis]["pos"] for axis in _AXIS_NAMES], dtype=torch.float32),
        "neg": torch.as_tensor([thresholds[axis]["neg"] for axis in _AXIS_NAMES], dtype=torch.float32),
    }


def _resolve_temporal_release_loss_config(raw: Any) -> dict[str, Any]:
    cfg = dict(raw or {})
    enabled = bool(cfg.get("enabled", False))
    if not enabled:
        return {
            "enabled": False,
            "weight": 0.0,
            "release_window_steps": 0,
            "pos": torch.zeros(4, dtype=torch.float32),
            "neg": torch.zeros(4, dtype=torch.float32),
        }
    thresholds = _load_deadzone_thresholds(cfg)
    return {
        "enabled": True,
        "weight": float(cfg.get("weight", 0.05)),
        "release_window_steps": int(cfg.get("release_window_steps", 3)),
        "pos": torch.as_tensor([thresholds[axis]["pos"] for axis in _AXIS_NAMES], dtype=torch.float32),
        "neg": torch.as_tensor([thresholds[axis]["neg"] for axis in _AXIS_NAMES], dtype=torch.float32),
    }


def _resolve_same_dir_window(value: Any) -> str:
    result = str(value or "all")
    if result not in {"all", "expert_transition_window"}:
        raise ValueError(
            "deadzone_loss.same_dir_window must be 'all' or 'expert_transition_window'."
        )
    return result


def _resolve_denominator(value: Any, *, name: str, allowed: set[str]) -> str:
    result = str(value or "crossing_axes")
    if result not in allowed:
        formatted = ", ".join(repr(item) for item in sorted(allowed))
        raise ValueError(f"{name} must be one of {formatted}.")
    return result


def _load_deadzone_thresholds(cfg: dict[str, Any]) -> dict[str, dict[str, float]]:
    if "thresholds" in cfg:
        payload = cfg["thresholds"]
    else:
        path_raw = cfg.get("threshold_json")
        if not path_raw:
            raise ValueError("deadzone_loss.enabled requires threshold_json or thresholds.")
        path = Path(str(path_raw))
        if not path.exists():
            raise FileNotFoundError(f"deadzone_loss threshold_json does not exist: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload.get("deadzone_action", payload) if isinstance(payload, dict) else payload
    thresholds: dict[str, dict[str, float]] = {}
    for axis in _AXIS_NAMES:
        axis_raw = raw.get(axis) if isinstance(raw, dict) else None
        if not isinstance(axis_raw, dict):
            raise ValueError(f"deadzone_loss thresholds missing axis {axis!r}.")
        thresholds[axis] = {
            "pos": _threshold_value(axis_raw.get("pos"), axis=axis, direction="pos"),
            "neg": _threshold_value(axis_raw.get("neg"), axis=axis, direction="neg"),
        }
    return thresholds


def _threshold_value(value: Any, *, axis: str, direction: str) -> float:
    if isinstance(value, dict):
        if "threshold_action_abs" in value:
            value = value["threshold_action_abs"]
        elif "value" in value:
            value = value["value"]
        else:
            raise ValueError(f"deadzone_loss threshold for {axis}.{direction} is missing value.")
    result = float(value)
    if result < 0.0:
        raise ValueError(f"deadzone_loss threshold for {axis}.{direction} must be >= 0.")
    return result


def _broadcast_axis_values(value: Any, *, name: str) -> list[float]:
    if isinstance(value, (int, float)):
        return [float(value)] * len(_AXIS_NAMES)
    values = [float(item) for item in value]
    if len(values) != len(_AXIS_NAMES):
        raise ValueError(f"{name} must be a scalar or {len(_AXIS_NAMES)} values.")
    return values


def _broadcast_intent_values(value: Any, *, name: str) -> list[float]:
    if isinstance(value, (int, float)):
        return [float(value)] * (len(_AXIS_NAMES) * 2)
    values = [float(item) for item in value]
    if len(values) == len(_AXIS_NAMES):
        expanded: list[float] = []
        for item in values:
            expanded.extend([item, item])
        return expanded
    if len(values) != len(_AXIS_NAMES) * 2:
        raise ValueError(f"{name} must be a scalar, {len(_AXIS_NAMES)} values, or {len(_AXIS_NAMES) * 2} values.")
    return values


def _expert_transition_window_mask(
    *,
    expert_step_effective: torch.Tensor,
    valid_step: torch.Tensor,
    window_steps: int,
) -> torch.Tensor:
    steps = max(1, int(window_steps))
    effective = expert_step_effective.to(dtype=torch.bool) & valid_step.to(dtype=torch.bool)
    previous = torch.zeros_like(effective)
    previous[:, 1:] = effective[:, :-1]
    transition = effective & ~previous
    mask = transition.clone()
    for offset in range(1, steps):
        shifted = torch.zeros_like(transition)
        shifted[:, offset:] = transition[:, :-offset]
        mask = mask | shifted
    return mask & effective


def _direction_release_window_mask(
    *,
    expert_effective: torch.Tensor,
    valid: torch.Tensor,
    window_steps: int,
) -> torch.Tensor:
    steps = max(1, int(window_steps))
    effective = expert_effective.to(dtype=torch.bool) & valid.to(dtype=torch.bool)
    previous = torch.zeros_like(effective)
    previous[:, 1:] = effective[:, :-1]
    release_start = previous & ~effective & valid.to(dtype=torch.bool)
    mask = release_start.clone()
    for offset in range(1, steps):
        shifted = torch.zeros_like(release_start)
        shifted[:, offset:] = release_start[:, :-offset]
        mask = mask | shifted
    return mask & valid.to(dtype=torch.bool)


def _masked_action_l1(
    *,
    expert: torch.Tensor,
    policy: torch.Tensor,
    valid_mask: torch.Tensor,
    action_loss_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    import torch.nn.functional as F

    all_l1 = F.l1_loss(expert, policy, reduction="none")
    valid = valid_mask.to(dtype=torch.bool).expand_as(all_l1)
    if action_loss_mask is None:
        return (all_l1 * valid.to(all_l1.dtype)).mean()
    action_valid = action_loss_mask.to(device=expert.device, dtype=torch.bool).unsqueeze(-1)
    mask = valid & action_valid
    count = mask.to(all_l1.dtype).sum().clamp_min(1.0)
    return (all_l1 * mask.to(all_l1.dtype)).sum() / count


@register_policy("act")
class ACTAdapter(Policy):
    """
    Runnable ACT policy for inference.

    Parameters
    ----------
    policy_config  Dict passed to build_ACT_model_and_optimizer.
    norm_stats     Dataset normalisation stats (proprio_mean/std, action_mean/std).
    temporal_agg   Use temporal action aggregation (default False).
    device         Torch device string (default "cuda").
    """

    def __init__(
        self,
        policy_config: dict,
        norm_stats: dict[str, np.ndarray],
        temporal_agg: bool = False,
        device: str = "cuda",
        inference_precision: str = "fp32",
    ):
        from testbed.policies.act.detr.main import build_ACT_model_and_optimizer

        self.device       = torch.device(device if torch.cuda.is_available() else "cpu")
        self.norm_stats   = norm_stats
        self.temporal_agg = temporal_agg
        self._inference_autocast_dtype = _resolve_inference_autocast_dtype(
            inference_precision,
            device=self.device,
        )
        self._inference_precision = str(inference_precision or "fp32").strip().lower()
        self.kl_weight    = policy_config.get("kl_weight", 10)
        self._camera_names = list(policy_config.get("camera_names", []))
        self._low_dim_keys = list(policy_config.get("low_dim_keys", ["qpos"]))
        self._deadzone_loss = _resolve_deadzone_loss_config(policy_config.get("deadzone_loss"))
        self._intent_loss = _resolve_intent_loss_config(policy_config.get("intent_loss"))
        self._window_deadzone_loss = _resolve_window_deadzone_loss_config(
            policy_config.get("window_deadzone_loss")
        )
        self._temporal_release_loss = _resolve_temporal_release_loss_config(
            policy_config.get("temporal_release_loss")
        )
        policy_config = dict(policy_config)
        policy_config["intent_dim"] = int(self._intent_loss["intent_dim"])

        model, optimizer = build_ACT_model_and_optimizer(policy_config)
        self._model     = model.to(self.device)
        self._model.eval()
        self._optimizer = optimizer

        # temporal aggregation state
        self._num_queries: int  = policy_config["num_queries"]
        self._t: int            = 0
        self._all_time_actions: torch.Tensor | None = None
        self._temporal_weight_cache: dict[int, torch.Tensor] = {}
        self._max_episode_len = int(policy_config.get("max_episode_len", 400))
        self._last_raw_action_chunk: torch.Tensor | None = None

        self._normalize  = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        self._proprio_mean, self._proprio_std = self._resolve_proprio_norm_stats()
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Called once per inference episode to clear temporal state."""
        self._t = 0
        self._all_time_actions = None
        self._last_raw_action_chunk = None

    @property
    def camera_names(self) -> list[str]:
        return list(self._camera_names)

    @property
    def inference_precision(self) -> str:
        return self._inference_precision

    def last_raw_action_chunk(self) -> np.ndarray:
        """Return the latest normalized ACT chunk for parity diagnostics."""
        if self._last_raw_action_chunk is None:
            raise RuntimeError("no ACT inference has run since reset")
        return self._last_raw_action_chunk.cpu().numpy().copy()

    # ── inference ─────────────────────────────────────────────────────────────

    def predict(self, obs: dict) -> np.ndarray:
        """
        Parameters
        ----------
        obs   dict with keys:
                "qpos"      : (Nq,) float32
                "qvel"      : (Nv,) float32 when configured in low_dim_keys
                "image_<cam>": (C, H, W) float32 [0, 1]   for each camera
              Camera images should be in channel-first format.

        Returns
        -------
        action : (Na,) float32  in *unnormalised* action space.
        """
        action, _ = self._predict_action_and_optional_intent(obs)
        return action

    def predict_action_and_intent(self, obs: dict) -> tuple[np.ndarray, np.ndarray]:
        """Return the executed action and query-0 axis-direction intent probabilities.

        This advances temporal aggregation exactly once, matching :meth:`predict`.
        Intent probabilities are ordered axis-major positive/negative.
        """
        action, intent_probabilities = self._predict_action_and_optional_intent(obs)
        if intent_probabilities is None:
            raise ValueError("loaded ACT policy has no intent logits")
        return action, intent_probabilities

    def _predict_action_and_optional_intent(
        self, obs: dict
    ) -> tuple[np.ndarray, np.ndarray | None]:
        proprio = self._build_proprio(obs)

        # normalise low-dimensional robot state
        proprio = (
            proprio - self._proprio_mean
        ) / self._proprio_std

        # Assemble image tensor in configured camera order. Ignore metadata
        # keys like `image_format` that may appear in live observations.
        cam_images: list[np.ndarray] = []
        for cam in self._camera_names:
            key = f"image_{cam}"
            if key not in obs:
                raise ValueError(
                    f"ACTAdapter.predict(): missing required camera input {key!r}."
                )
            raw_cam_img = np.asarray(obs[key])
            cam_img = np.asarray(raw_cam_img, dtype=np.float32)
            if cam_img.ndim != 3:
                raise ValueError(
                    f"ACTAdapter.predict(): expected {key!r} to be rank-3, got shape {cam_img.shape}."
                )
            # Accept either channel-first float images or raw channel-last RGB.
            if cam_img.shape[0] == 3:
                pass
            elif cam_img.shape[-1] == 3:
                cam_img = np.transpose(cam_img, (2, 0, 1))
                if raw_cam_img.dtype == np.uint8:
                    cam_img = cam_img / 255.0
                elif cam_img.max() > 1.0:
                    cam_img = cam_img / 255.0
            else:
                raise ValueError(
                    f"ACTAdapter.predict(): expected {key!r} to have 3 channels, got shape {cam_img.shape}."
                )
            cam_images.append(cam_img)

        if not cam_images:
            raise ValueError("ACTAdapter.predict(): no camera inputs configured.")

        img = np.ascontiguousarray(np.stack(cam_images, axis=0))  # (n_cams, C, H, W)
        image = torch.from_numpy(img).float().to(self.device).unsqueeze(0)  # (1, n_cams, C, H, W)
        image = self._normalize(image)

        if self._model.training:
            self._model.eval()
        a_hat, intent_logits = self._forward_inference(proprio, image)
        if intent_logits is not None and (
            intent_logits.ndim != 3
            or intent_logits.shape[0] != 1
            or intent_logits.shape[2] != 8
        ):
            raise ValueError(
                "ACT intent logits must have shape (1, num_queries, 8), "
                f"got {tuple(intent_logits.shape)}"
            )

        if self.temporal_agg:
            action = self._aggregate(a_hat)
        else:
            # non-aggregated: execute every num_queries steps
            if self._t % self._num_queries == 0:
                self._cached_actions = a_hat.squeeze(0)    # (C, Na)
            step_in_chunk = self._t % self._num_queries
            action = self._cached_actions[step_in_chunk].cpu().numpy()

        self._t += 1

        # unnormalise
        action = (
            action
            * self.norm_stats["action_std"]
            + self.norm_stats["action_mean"]
        )
        intent_prob = (
            None
            if intent_logits is None
            else torch.sigmoid(intent_logits[0, 0]).detach().cpu().numpy()
        )
        return (
            action.astype(np.float32),
            None
            if intent_prob is None
            else np.asarray(intent_prob, dtype=np.float32),
        )

    def _forward_inference(
        self,
        proprio: torch.Tensor,
        image: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # A few lightweight compatibility tests construct the adapter with
        # ``__new__`` and install only the fields needed for prediction.  Keep
        # those legacy stubs on the default FP32 path instead of requiring the
        # new constructor-owned precision field.
        inference_autocast_dtype = getattr(
            self, "_inference_autocast_dtype", None
        )
        autocast_context = (
            nullcontext()
            if inference_autocast_dtype is None
            else torch.autocast(
                device_type=self.device.type,
                dtype=inference_autocast_dtype,
                enabled=True,
            )
        )
        with torch.inference_mode(), autocast_context:
            a_hat, _, _, intent_logits = self._unpack_model_output(
                self._model(proprio, image, None)
            )

        # Keep temporal aggregation, sigmoid and CPU conversion in FP32.  The
        # reduced precision scope is limited to the neural-network forward.
        a_hat = a_hat.float()
        if intent_logits is not None:
            intent_logits = intent_logits.float()
        self._last_raw_action_chunk = a_hat[0].detach()
        return a_hat, intent_logits

    def _build_proprio(self, obs: dict) -> torch.Tensor:
        parts: list[np.ndarray] = []
        for key in self._low_dim_keys:
            if key not in obs:
                raise ValueError(
                    f"ACTAdapter.predict(): missing required low-dimensional input {key!r}."
                )
            value = np.asarray(obs[key], dtype=np.float32).reshape(-1)
            parts.append(value)
        if not parts:
            raise ValueError("ACTAdapter.predict(): low_dim_keys must not be empty.")
        proprio = np.concatenate(parts, axis=0).astype(np.float32)
        return torch.from_numpy(proprio).float().to(self.device).unsqueeze(0)

    def _resolve_proprio_norm_stats(self) -> tuple[torch.Tensor, torch.Tensor]:
        if "proprio_mean" in self.norm_stats and "proprio_std" in self.norm_stats:
            mean = self.norm_stats["proprio_mean"]
            std = self.norm_stats["proprio_std"]
        elif self._low_dim_keys == ["qpos"]:
            # Backward compatibility for older qpos-only checkpoints.
            mean = self.norm_stats["qpos_mean"]
            std = self.norm_stats["qpos_std"]
        else:
            raise KeyError(
                "dataset_stats.pkl does not contain proprio_mean/proprio_std for "
                f"low_dim_keys={self._low_dim_keys}. Recompute stats by retraining "
                "with the updated data pipeline."
            )
        return (
            torch.from_numpy(np.asarray(mean, dtype=np.float32)).to(self.device),
            torch.from_numpy(np.asarray(std, dtype=np.float32)).to(self.device),
        )

    def _aggregate(self, a_hat: torch.Tensor) -> np.ndarray:
        """
        Temporal aggregation from the ACT paper.
        a_hat shape: (1, C, Na)
        """
        Na = a_hat.shape[-1]

        if self._all_time_actions is None:
            horizon = max(self._max_episode_len, self._t + self._num_queries)
            self._all_time_actions = torch.zeros(
                [horizon, horizon + self._num_queries, Na], device=self.device
            )

        required_rows = self._t + 1
        required_cols = self._t + self._num_queries
        if (
            required_rows > self._all_time_actions.shape[0]
            or required_cols > self._all_time_actions.shape[1]
        ):
            current_rows = self._all_time_actions.shape[0]
            new_rows = max(required_rows, current_rows * 2)
            expanded = torch.zeros(
                [new_rows, new_rows + self._num_queries, Na],
                device=self.device,
            )
            expanded[: self._all_time_actions.shape[0], : self._all_time_actions.shape[1]] = (
                self._all_time_actions
            )
            self._all_time_actions = expanded

        t = self._t
        self._all_time_actions[[t], t : t + self._num_queries] = a_hat

        # weighted average of all past chunks that cover step t
        # NOTE: only rows whose chunk actually covers column t are non-zero;
        #       filter them out exactly as in the original ACT repo to avoid
        #       zero-padding contaminating the weighted mean.
        actions_for_curr_step = self._all_time_actions[:t + 1, t]  # (t+1, Na)
        actions_populated = torch.all(actions_for_curr_step != 0, dim=1)
        actions_for_curr_step = actions_for_curr_step[actions_populated]
        num_actions = int(len(actions_for_curr_step))
        exp_weights = self._temporal_weight_cache.get(num_actions)
        if exp_weights is None:
            k = 0.01
            exp_weights = torch.exp(
                -k * torch.arange(num_actions, dtype=torch.float32, device=self.device)
            )
            exp_weights = (exp_weights / exp_weights.sum()).unsqueeze(1)
            self._temporal_weight_cache[num_actions] = exp_weights
        action = (actions_for_curr_step * exp_weights).sum(0).cpu().numpy()
        return action

    # ── training forward ──────────────────────────────────────────────────────

    def forward_loss(
        self,
        proprio: torch.Tensor,
        image: torch.Tensor,
        actions: torch.Tensor,
        is_pad: torch.Tensor,
        *,
        deadzone_move_mask: torch.Tensor | None = None,
        deadzone_stop_mask: torch.Tensor | None = None,
        deadzone_wrong_mask: torch.Tensor | None = None,
        action_loss_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Training-time forward pass.

        Parameters
        ----------
        proprio (B, Np)
        image   (B, n_cams, C, H, W)   normalised
        actions (B, C, Na)
        is_pad  (B, C)  bool

        Returns
        -------
        {"l1": ..., "kl": ..., "loss": ...}
        """
        image   = self._normalize(image)
        actions = actions[:, : self._model.num_queries]
        is_pad  = is_pad[:,  : self._model.num_queries]
        if action_loss_mask is not None:
            action_loss_mask = action_loss_mask[:, : self._model.num_queries]
        if deadzone_move_mask is not None:
            deadzone_move_mask = deadzone_move_mask[:, : self._model.num_queries]
        if deadzone_stop_mask is not None:
            deadzone_stop_mask = deadzone_stop_mask[:, : self._model.num_queries]
        if deadzone_wrong_mask is not None:
            deadzone_wrong_mask = deadzone_wrong_mask[:, : self._model.num_queries]

        a_hat, _, (mu, logvar), intent_logits = self._unpack_model_output(
            self._model(proprio, image, None, actions, is_pad)
        )
        total_kld, _, _        = _kl_divergence(mu, logvar)

        valid_mask = ~is_pad.unsqueeze(-1)
        l1 = _masked_action_l1(
            expert=actions,
            policy=a_hat,
            valid_mask=valid_mask,
            action_loss_mask=action_loss_mask,
        )
        deadzone_loss_d = self._deadzone_loss_terms(
            expert_normalized=actions,
            policy_normalized=a_hat,
            valid_mask=valid_mask,
        )
        window_deadzone_loss_d = self._window_deadzone_loss_terms(
            expert_normalized=actions,
            policy_normalized=a_hat,
            valid_mask=valid_mask,
            move_mask=deadzone_move_mask,
            stop_mask=deadzone_stop_mask,
            wrong_mask=deadzone_wrong_mask,
        )
        intent_loss_d = self._intent_loss_terms(
            expert_normalized=actions,
            intent_logits=intent_logits,
            valid_mask=valid_mask,
        )
        temporal_release_loss_d = self._temporal_release_loss_terms(
            expert_normalized=actions,
            policy_normalized=a_hat,
            valid_mask=valid_mask,
        )

        return {
            "l1":   l1,
            "kl":   total_kld[0],
            **deadzone_loss_d,
            **window_deadzone_loss_d,
            **intent_loss_d,
            **temporal_release_loss_d,
            "loss": (
                l1
                + total_kld[0] * self.kl_weight
                + deadzone_loss_d["deadzone_loss"]
                + window_deadzone_loss_d["window_deadzone_loss"]
                + intent_loss_d["intent_loss"]
                + temporal_release_loss_d["temporal_release_loss"]
            ),
        }

    @staticmethod
    def _unpack_model_output(output):
        a_hat, is_pad_hat, latent = output[:3]
        intent_logits = output[3] if len(output) > 3 else None
        return a_hat, is_pad_hat, latent, intent_logits

    def _deadzone_loss_terms(
        self,
        *,
        expert_normalized: torch.Tensor,
        policy_normalized: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        cfg = self._deadzone_loss
        zero = policy_normalized.new_zeros(())
        if not cfg["enabled"]:
            return {
                "deadzone_same_dir_loss": zero,
                "deadzone_idle_loss": zero,
                "deadzone_wrong_loss": zero,
                "deadzone_loss": zero,
            }

        action_mean = torch.as_tensor(
            self.norm_stats["action_mean"],
            dtype=policy_normalized.dtype,
            device=policy_normalized.device,
        )
        action_std = torch.as_tensor(
            self.norm_stats["action_std"],
            dtype=policy_normalized.dtype,
            device=policy_normalized.device,
        )
        expert = expert_normalized * action_std + action_mean
        policy = policy_normalized * action_std + action_mean

        pos = cfg["pos"].to(device=policy.device, dtype=policy.dtype)
        neg = cfg["neg"].to(device=policy.device, dtype=policy.dtype)
        margin = cfg["margin"].to(device=policy.device, dtype=policy.dtype)
        valid = valid_mask.to(dtype=torch.bool).expand_as(policy)

        expert_pos = (expert >= pos) & valid
        expert_neg = (expert <= -neg) & valid
        expert_axis_effective = expert_pos | expert_neg
        expert_step_effective = expert_axis_effective.any(dim=-1, keepdim=True)
        if cfg["same_dir_window"] == "expert_transition_window":
            same_dir_step_mask = _expert_transition_window_mask(
                expert_step_effective=expert_step_effective,
                valid_step=valid.any(dim=-1, keepdim=True),
                window_steps=int(cfg["same_dir_window_steps"]),
            )
            expert_pos = expert_pos & same_dir_step_mask
            expert_neg = expert_neg & same_dir_step_mask
        same_pos_shortfall = torch.relu(pos + margin - policy) * expert_pos.to(policy.dtype)
        same_neg_shortfall = torch.relu(policy + neg + margin) * expert_neg.to(policy.dtype)
        same_dir_count = (expert_pos | expert_neg).to(policy.dtype).sum().clamp_min(1.0)
        same_dir_loss = (same_pos_shortfall + same_neg_shortfall).sum() / same_dir_count

        policy_pos_effective = policy >= pos
        policy_neg_effective = policy <= -neg

        idle_axes = valid & ~expert_step_effective
        idle_pos_excess = torch.relu(policy - pos) * (idle_axes & policy_pos_effective).to(policy.dtype)
        idle_neg_excess = torch.relu(-neg - policy) * (idle_axes & policy_neg_effective).to(policy.dtype)
        if cfg["idle_denominator"] == "all_idle_axes":
            idle_count = idle_axes.to(policy.dtype).sum().clamp_min(1.0)
        else:
            idle_count = (
                (idle_axes & (policy_pos_effective | policy_neg_effective))
                .to(policy.dtype)
                .sum()
                .clamp_min(1.0)
            )
        idle_loss = (idle_pos_excess + idle_neg_excess).sum() / idle_count

        wrong_loss = zero
        if cfg["penalize_wrong_effective"]:
            not_expert_pos = (~expert_pos) & valid & expert_step_effective
            not_expert_neg = (~expert_neg) & valid & expert_step_effective
            wrong_pos_mask = not_expert_pos & policy_pos_effective
            wrong_neg_mask = not_expert_neg & policy_neg_effective
            wrong_pos_excess = torch.relu(policy - pos) * wrong_pos_mask.to(policy.dtype)
            wrong_neg_excess = torch.relu(-neg - policy) * wrong_neg_mask.to(policy.dtype)
            if cfg["wrong_denominator"] == "all_wrong_candidate_axes":
                wrong_count = (
                    not_expert_pos.to(policy.dtype).sum()
                    + not_expert_neg.to(policy.dtype).sum()
                ).clamp_min(1.0)
            else:
                wrong_count = (
                    (wrong_pos_mask | wrong_neg_mask)
                    .to(policy.dtype)
                    .sum()
                    .clamp_min(1.0)
                )
            wrong_loss = (wrong_pos_excess + wrong_neg_excess).sum() / wrong_count

        total = (
            float(cfg["same_dir_promote_weight"]) * same_dir_loss
            + float(cfg["idle_suppression_weight"]) * idle_loss
            + float(cfg["wrong_effective_weight"]) * wrong_loss
        )
        return {
            "deadzone_same_dir_loss": same_dir_loss,
            "deadzone_idle_loss": idle_loss,
            "deadzone_wrong_loss": wrong_loss,
            "deadzone_loss": total,
        }

    def _window_deadzone_loss_terms(
        self,
        *,
        expert_normalized: torch.Tensor,
        policy_normalized: torch.Tensor,
        valid_mask: torch.Tensor,
        move_mask: torch.Tensor | None,
        stop_mask: torch.Tensor | None,
        wrong_mask: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        cfg = getattr(self, "_window_deadzone_loss", {"enabled": False})
        zero = policy_normalized.new_zeros(())
        if not cfg.get("enabled", False):
            return {
                "window_deadzone_same_dir_loss": zero,
                "window_deadzone_stop_loss": zero,
                "window_deadzone_wrong_loss": zero,
                "window_deadzone_loss": zero,
            }
        if move_mask is None or stop_mask is None or wrong_mask is None:
            raise ValueError("window deadzone loss requires move, stop, and wrong masks")

        action_mean = torch.as_tensor(
            self.norm_stats["action_mean"],
            dtype=policy_normalized.dtype,
            device=policy_normalized.device,
        )
        action_std = torch.as_tensor(
            self.norm_stats["action_std"],
            dtype=policy_normalized.dtype,
            device=policy_normalized.device,
        )
        policy = policy_normalized * action_std + action_mean

        pos = cfg["pos"].to(device=policy.device, dtype=policy.dtype)
        neg = cfg["neg"].to(device=policy.device, dtype=policy.dtype)
        margin = cfg["margin"].to(device=policy.device, dtype=policy.dtype)
        valid = valid_mask.to(dtype=torch.bool).expand_as(policy)
        valid_dir = torch.stack([valid, valid], dim=-1)

        move = move_mask.to(device=policy.device, dtype=torch.bool) & valid_dir
        move_pos = move[..., 0]
        move_neg = move[..., 1]
        same_pos_shortfall = torch.relu(pos + margin - policy) * move_pos.to(policy.dtype)
        same_neg_shortfall = torch.relu(policy + neg + margin) * move_neg.to(policy.dtype)
        same_count = (move_pos | move_neg).to(policy.dtype).sum().clamp_min(1.0)
        same_dir_loss = (same_pos_shortfall + same_neg_shortfall).sum() / same_count

        policy_pos_effective = policy >= pos
        policy_neg_effective = policy <= -neg
        stop_axes = stop_mask.to(device=policy.device, dtype=torch.bool).unsqueeze(-1) & valid
        stop_pos_mask = stop_axes & policy_pos_effective
        stop_neg_mask = stop_axes & policy_neg_effective
        stop_pos_excess = torch.relu(policy - pos) * stop_pos_mask.to(policy.dtype)
        stop_neg_excess = torch.relu(-neg - policy) * stop_neg_mask.to(policy.dtype)
        stop_count = (stop_pos_mask | stop_neg_mask).to(policy.dtype).sum().clamp_min(1.0)
        stop_loss = (stop_pos_excess + stop_neg_excess).sum() / stop_count

        wrong = wrong_mask.to(device=policy.device, dtype=torch.bool) & valid_dir
        wrong_pos_mask = wrong[..., 0] & policy_pos_effective
        wrong_neg_mask = wrong[..., 1] & policy_neg_effective
        wrong_pos_excess = torch.relu(policy - pos) * wrong_pos_mask.to(policy.dtype)
        wrong_neg_excess = torch.relu(-neg - policy) * wrong_neg_mask.to(policy.dtype)
        wrong_count = (wrong_pos_mask | wrong_neg_mask).to(policy.dtype).sum().clamp_min(1.0)
        wrong_loss = (wrong_pos_excess + wrong_neg_excess).sum() / wrong_count

        total = (
            float(cfg["same_dir_promote_weight"]) * same_dir_loss
            + float(cfg["stop_suppression_weight"]) * stop_loss
            + float(cfg["wrong_effective_weight"]) * wrong_loss
        )
        return {
            "window_deadzone_same_dir_loss": same_dir_loss,
            "window_deadzone_stop_loss": stop_loss,
            "window_deadzone_wrong_loss": wrong_loss,
            "window_deadzone_loss": total,
        }

    def _temporal_release_loss_terms(
        self,
        *,
        expert_normalized: torch.Tensor,
        policy_normalized: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        cfg = getattr(self, "_temporal_release_loss", {"enabled": False})
        zero = policy_normalized.new_zeros(())
        if not cfg.get("enabled", False):
            return {
                "temporal_release_pos_loss": zero,
                "temporal_release_neg_loss": zero,
                "temporal_release_loss": zero,
            }

        action_mean = torch.as_tensor(
            self.norm_stats["action_mean"],
            dtype=policy_normalized.dtype,
            device=policy_normalized.device,
        )
        action_std = torch.as_tensor(
            self.norm_stats["action_std"],
            dtype=policy_normalized.dtype,
            device=policy_normalized.device,
        )
        expert = expert_normalized * action_std + action_mean
        policy = policy_normalized * action_std + action_mean
        pos = cfg["pos"].to(device=policy.device, dtype=policy.dtype)
        neg = cfg["neg"].to(device=policy.device, dtype=policy.dtype)
        valid = valid_mask.to(dtype=torch.bool).expand_as(policy)

        expert_pos = (expert >= pos) & valid
        expert_neg = (expert <= -neg) & valid
        release_pos = _direction_release_window_mask(
            expert_effective=expert_pos,
            valid=valid,
            window_steps=int(cfg["release_window_steps"]),
        )
        release_neg = _direction_release_window_mask(
            expert_effective=expert_neg,
            valid=valid,
            window_steps=int(cfg["release_window_steps"]),
        )
        policy_pos_effective = policy >= pos
        policy_neg_effective = policy <= -neg
        pos_excess = torch.relu(policy - pos) * (release_pos & policy_pos_effective).to(policy.dtype)
        neg_excess = torch.relu(-neg - policy) * (release_neg & policy_neg_effective).to(policy.dtype)
        pos_count = (release_pos & policy_pos_effective).to(policy.dtype).sum().clamp_min(1.0)
        neg_count = (release_neg & policy_neg_effective).to(policy.dtype).sum().clamp_min(1.0)
        pos_loss = pos_excess.sum() / pos_count
        neg_loss = neg_excess.sum() / neg_count
        total = float(cfg["weight"]) * (pos_loss + neg_loss)
        return {
            "temporal_release_pos_loss": pos_loss,
            "temporal_release_neg_loss": neg_loss,
            "temporal_release_loss": total,
        }

    def _intent_loss_terms(
        self,
        *,
        expert_normalized: torch.Tensor,
        intent_logits: torch.Tensor | None,
        valid_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        cfg = self._intent_loss
        zero = expert_normalized.new_zeros(())
        if not cfg["enabled"]:
            return {
                "intent_axis_dir_loss": zero,
                "intent_loss": zero,
            }
        if intent_logits is None:
            raise ValueError("intent_loss.enabled requires model intent logits.")

        action_mean = torch.as_tensor(
            self.norm_stats["action_mean"],
            dtype=expert_normalized.dtype,
            device=expert_normalized.device,
        )
        action_std = torch.as_tensor(
            self.norm_stats["action_std"],
            dtype=expert_normalized.dtype,
            device=expert_normalized.device,
        )
        expert = expert_normalized * action_std + action_mean
        pos = cfg["pos"].to(device=expert.device, dtype=expert.dtype)
        neg = cfg["neg"].to(device=expert.device, dtype=expert.dtype)
        valid = valid_mask.to(dtype=torch.bool).expand_as(expert)
        target = torch.stack(
            [
                ((expert >= pos) & valid).to(intent_logits.dtype),
                ((expert <= -neg) & valid).to(intent_logits.dtype),
            ],
            dim=-1,
        ).reshape(*expert.shape[:-1], 8)
        label_valid = torch.stack([valid, valid], dim=-1).reshape(*expert.shape[:-1], 8).to(
            intent_logits.dtype
        )
        if intent_logits.shape != target.shape:
            raise ValueError(
                f"intent logits must have shape {target.shape}, got {intent_logits.shape}"
            )

        import torch.nn.functional as F

        positive_weight = cfg["positive_weight"].to(
            device=intent_logits.device,
            dtype=intent_logits.dtype,
        )
        bce = F.binary_cross_entropy_with_logits(
            intent_logits,
            target,
            reduction="none",
            pos_weight=positive_weight,
        )
        count = label_valid.sum().clamp_min(1.0)
        axis_dir_loss = (bce * label_valid).sum() / count
        total = float(cfg["weight"]) * axis_dir_loss
        return {
            "intent_axis_dir_loss": axis_dir_loss,
            "intent_loss": total,
        }

    def configure_optimizers(self):
        return self._optimizer

    def state_dict(self):
        return self._model.state_dict()

    def load_state_dict(self, sd, strict: bool = True):
        return self._model.load_state_dict(sd, strict=strict)

    # ── checkpoint helpers ────────────────────────────────────────────────────

    @classmethod
    def from_checkpoint(
        cls,
        ckpt_path: str | Path,
        policy_config: dict,
        norm_stats_path: str | Path,
        temporal_agg: bool = False,
        device: str = "cuda",
        inference_precision: str = "fp32",
    ) -> "ACTAdapter":
        """
        Convenience factory: load an ACT policy from a checkpoint file.

        Parameters
        ----------
        ckpt_path       Path to policy_best.ckpt (or policy_epoch_N.ckpt).
        policy_config   Same config dict used during training.
        norm_stats_path Path to dataset_stats.pkl.
        """
        ckpt_path       = Path(ckpt_path)
        norm_stats_path = Path(norm_stats_path)

        with open(norm_stats_path, "rb") as f:
            norm_stats = pickle.load(f)

        adapter = cls(
            policy_config=policy_config,
            norm_stats=norm_stats,
            temporal_agg=temporal_agg,
            device=device,
            inference_precision=inference_precision,
        )

        raw = torch.load(ckpt_path, map_location="cpu")
        if isinstance(raw, dict) and "model_state_dict" in raw:
            sd = raw["model_state_dict"]
        elif isinstance(raw, dict):
            sd = raw
        else:
            raise ValueError(f"Unsupported checkpoint format: {type(raw)}")

        adapter.load_state_dict(sd)
        adapter._model.to(adapter.device)
        adapter._model.eval()
        return adapter
