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
"""

from __future__ import annotations

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
    ):
        from testbed.policies.act.detr.main import build_ACT_model_and_optimizer

        self.device       = torch.device(device if torch.cuda.is_available() else "cpu")
        self.norm_stats   = norm_stats
        self.temporal_agg = temporal_agg
        self.kl_weight    = policy_config.get("kl_weight", 10)
        self._camera_names = list(policy_config.get("camera_names", []))
        self._low_dim_keys = list(policy_config.get("low_dim_keys", ["qpos"]))

        model, optimizer = build_ACT_model_and_optimizer(policy_config)
        self._model     = model.to(self.device)
        self._optimizer = optimizer

        # temporal aggregation state
        self._num_queries: int  = policy_config["num_queries"]
        self._t: int            = 0
        self._all_time_actions: torch.Tensor | None = None
        self._max_episode_len = int(policy_config.get("max_episode_len", 400))

        self._normalize  = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        self._proprio_mean, self._proprio_std = self._resolve_proprio_norm_stats()

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Called once per inference episode to clear temporal state."""
        self._t = 0
        self._all_time_actions = None

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
            cam_img = np.asarray(obs[key], dtype=np.float32)
            if cam_img.ndim != 3:
                raise ValueError(
                    f"ACTAdapter.predict(): expected {key!r} to be rank-3, got shape {cam_img.shape}."
                )
            # Accept either channel-first float images or raw channel-last RGB.
            if cam_img.shape[0] == 3:
                pass
            elif cam_img.shape[-1] == 3:
                cam_img = np.transpose(cam_img, (2, 0, 1))
                if cam_img.max() > 1.0:
                    cam_img = cam_img / 255.0
            else:
                raise ValueError(
                    f"ACTAdapter.predict(): expected {key!r} to have 3 channels, got shape {cam_img.shape}."
                )
            cam_images.append(cam_img)

        if not cam_images:
            raise ValueError("ACTAdapter.predict(): no camera inputs configured.")

        img = np.stack(cam_images, axis=0)                 # (n_cams, C, H, W)
        image = torch.from_numpy(img).float().to(self.device).unsqueeze(0)  # (1, n_cams, C, H, W)
        image = self._normalize(image)

        self._model.eval()
        with torch.no_grad():
            a_hat, _, _ = self._model(proprio, image, None)   # (1, C, Na)

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
        return action.astype(np.float32)

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

        required_t = self._t + self._num_queries
        if required_t > self._all_time_actions.shape[1]:
            current_t = self._all_time_actions.shape[0]
            new_t = max(required_t, current_t * 2)
            expanded = torch.zeros(
                [new_t, new_t + self._num_queries, Na],
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
        k = 0.01
        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
        exp_weights = exp_weights / exp_weights.sum()
        exp_weights = torch.from_numpy(exp_weights).float().to(self.device).unsqueeze(1)
        action = (actions_for_curr_step * exp_weights).sum(0).cpu().numpy()
        return action

    # ── training forward ──────────────────────────────────────────────────────

    def forward_loss(
        self,
        proprio: torch.Tensor,
        image: torch.Tensor,
        actions: torch.Tensor,
        is_pad: torch.Tensor,
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

        a_hat, _, (mu, logvar) = self._model(proprio, image, None, actions, is_pad)
        total_kld, _, _        = _kl_divergence(mu, logvar)

        import torch.nn.functional as F
        all_l1 = F.l1_loss(actions, a_hat, reduction="none")
        l1     = (all_l1 * ~is_pad.unsqueeze(-1)).mean()

        return {
            "l1":   l1,
            "kl":   total_kld[0],
            "loss": l1 + total_kld[0] * self.kl_weight,
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
