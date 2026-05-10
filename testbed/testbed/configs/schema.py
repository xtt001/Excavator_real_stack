"""Pydantic config schemas for the real-excavator testbed."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class BackendConfig(BaseModel):
    """Hot-pluggable backend selection."""

    name: Literal["real"] = "real"
    dt: float = Field(default=0.02, gt=0, description="Control timestep in seconds.")
    controller_mode: Literal["mock", "noop", "bridge_mock", "bridge_tcp"] = "mock"
    state_reader_mode: Literal["mock", "bridge_mock", "bridge_tcp"] = "mock"
    bridge_host: str = "127.0.0.1"
    bridge_port: int = Field(default=0, ge=0)
    bridge_timeout_s: float = Field(default=1.0, gt=0)


class TaskConfig(BaseModel):
    """Dataset and machine contract for real-excavator recording/training."""

    task_name: str = "real_excavation_teleop_v1"
    equipment_model: str = "real_excavator"
    dataset_dir: str = Field(..., description="Directory where HDF5 episodes are read/written.")
    num_episodes: int = Field(default=30, gt=0)
    episode_len: int = Field(default=1000, gt=0)
    camera_names: list[str] = Field(default_factory=lambda: ["fpv"])
    backend: BackendConfig = Field(default_factory=BackendConfig)


class PolicyConfig(BaseModel):
    """Policy plugin and low-dimensional input selection."""

    name: str = "act"
    params: dict = Field(default_factory=dict)
    low_dim_keys: list[Literal["qpos", "qvel"]] = Field(default_factory=lambda: ["qpos", "qvel"])


class ACTPolicyParams(BaseModel):
    """ACT-specific hyperparameters."""

    lr: float = 1e-5
    lr_backbone: float = 1e-5
    backbone: str = "resnet18"
    enc_layers: int = 4
    dec_layers: int = 7
    nheads: int = 8
    hidden_dim: int = 512
    dim_feedforward: int = 3200
    chunk_size: int = 100
    kl_weight: float = 10.0
    latent_dim: int = 32
    temporal_agg: bool = False


class TrainConfig(BaseModel):
    """Offline training run configuration."""

    task: TaskConfig
    policy: PolicyConfig
    ckpt_dir: str = Field(..., description="Directory to save checkpoints and stats.")
    num_epochs: int = Field(default=2000, gt=0)
    batch_size: int = Field(default=8, gt=0)
    seed: int = 0
    device: str = "cuda"
    num_workers: int = 0
    prefetch_factor: int | None = None
    persistent_workers: bool = False
    pin_memory: bool = True
    split_seed: int | None = None
    train_split_ratio: float = 0.8
    split_path: str | None = None
    reuse_split: bool = True
    val_every: int = 5
    save_latest_every: int = 10
    checkpoint_every: int = 50
    plot_every: int = 50
    amp: bool = True
    amp_dtype: Literal["auto", "bf16", "fp16"] = "auto"
    resume_ckpt: str | None = None
