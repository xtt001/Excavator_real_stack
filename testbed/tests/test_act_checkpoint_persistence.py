from __future__ import annotations

import pickle
from pathlib import Path

import pytest
import torch

from testbed.policies.act.adapter import ACTAdapter
from testbed.policies.act.checkpoint_persistence import (
    ACTCheckpointPersistence,
    atomic_torch_save,
    load_resume_checkpoint,
)
from testbed.policies.act.trainer import ACTTrainer


def _trained_state() -> tuple[dict, dict]:
    model = torch.nn.Linear(64, 64)
    optimizer = torch.optim.AdamW(model.parameters())
    model(torch.ones(2, 64)).sum().backward()
    optimizer.step()
    return model.state_dict(), optimizer.state_dict()


def test_resume_and_inference_checkpoint_schemas_and_sizes(tmp_path: Path) -> None:
    model_state, optimizer_state = _trained_state()
    persistence = ACTCheckpointPersistence(
        tmp_path,
        seed=7,
        periodic_keep_last=3,
    )

    resume_path = persistence.save_resume(
        model_state_dict=model_state,
        optimizer_state_dict=optimizer_state,
        epoch=19,
        min_val_loss=0.25,
        config={"task_name": "test", "seed": 7},
    )
    best_path = persistence.save_best(
        model_state_dict=model_state,
        epoch=17,
        min_val_loss=0.2,
        config={"task_name": "test", "seed": 7},
    )

    resume = load_resume_checkpoint(resume_path)
    inference = torch.load(best_path, map_location="cpu")
    assert resume["checkpoint_kind"] == "resume"
    assert resume["epoch"] == 19
    assert "optimizer_state_dict" in resume
    assert inference["checkpoint_kind"] == "inference"
    assert "optimizer_state_dict" not in inference
    assert best_path.stat().st_size < resume_path.stat().st_size
    assert ACTTrainer._infer_start_epoch(str(resume_path), resume, {}) == 20

    restored_model = torch.nn.Linear(64, 64)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters())
    restored_optimizer.load_state_dict(resume["optimizer_state_dict"])
    assert len(restored_optimizer.state) == len(optimizer_state["state"])


def test_inference_checkpoint_loads_through_adapter_factory(tmp_path: Path) -> None:
    model_state, _ = _trained_state()
    persistence = ACTCheckpointPersistence(
        tmp_path,
        seed=0,
        periodic_keep_last=3,
    )
    checkpoint_path = persistence.save_best(
        model_state_dict=model_state,
        epoch=3,
        min_val_loss=0.1,
        config={},
    )
    stats_path = tmp_path / "dataset_stats.pkl"
    with stats_path.open("wb") as file:
        pickle.dump({}, file)

    class _ModelShell:
        def to(self, _device):
            return self

        def eval(self):
            return self

    class _TinyAdapter(ACTAdapter):
        def __init__(self, **kwargs):
            self.device = torch.device(kwargs["device"])
            self._model = _ModelShell()
            self.loaded_state = None

        def load_state_dict(self, state_dict, strict: bool = True):
            self.loaded_state = state_dict
            return None

    adapter = _TinyAdapter.from_checkpoint(
        ckpt_path=checkpoint_path,
        policy_config={},
        norm_stats_path=stats_path,
        device="cpu",
    )
    assert adapter.loaded_state.keys() == model_state.keys()


def test_inference_checkpoint_is_rejected_for_resume(tmp_path: Path) -> None:
    persistence = ACTCheckpointPersistence(
        tmp_path,
        seed=0,
        periodic_keep_last=3,
    )
    path = persistence.save_best(
        model_state_dict={"weight": torch.ones(1)},
        epoch=0,
        min_val_loss=1.0,
        config={},
    )

    with pytest.raises(ValueError, match="not resume-capable"):
        load_resume_checkpoint(path)


def test_atomic_save_preserves_existing_file_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "target.ckpt"
    target.write_bytes(b"complete-old-checkpoint")

    def _fail_after_partial_write(_payload, file) -> None:
        file.write(b"partial")
        raise RuntimeError("simulated write failure")

    monkeypatch.setattr(
        "testbed.policies.act.checkpoint_persistence.torch.save",
        _fail_after_partial_write,
    )
    with pytest.raises(RuntimeError, match="simulated write failure"):
        atomic_torch_save({"value": 1}, target)

    assert target.read_bytes() == b"complete-old-checkpoint"
    assert not list(tmp_path.glob(".target.ckpt.*.tmp"))


def test_periodic_retention_and_policy_last_hardlink(tmp_path: Path) -> None:
    persistence = ACTCheckpointPersistence(
        tmp_path,
        seed=7,
        periodic_keep_last=2,
    )
    other_seed = tmp_path / "policy_epoch_0_seed_9.ckpt"
    other_seed.write_bytes(b"not-owned")
    unrelated = tmp_path / "candidate.ckpt"
    unrelated.write_bytes(b"not-owned")

    for epoch in range(5):
        persistence.save_periodic(
            model_state_dict={"weight": torch.tensor([epoch])},
            epoch=epoch,
            min_val_loss=float(epoch),
            config={},
        )

    assert sorted(path.name for path in tmp_path.glob("policy_epoch_*_seed_7.ckpt")) == [
        "policy_epoch_3_seed_7.ckpt",
        "policy_epoch_4_seed_7.ckpt",
    ]
    assert other_seed.read_bytes() == b"not-owned"
    assert unrelated.read_bytes() == b"not-owned"

    persistence.save_resume(
        model_state_dict={"weight": torch.ones(1)},
        optimizer_state_dict={},
        epoch=4,
        min_val_loss=0.5,
        config={},
    )
    last_path = persistence.link_last_to_latest()
    assert last_path.stat().st_ino == persistence.latest_path.stat().st_ino
    assert last_path.stat().st_size == persistence.latest_path.stat().st_size


def test_trainer_writes_split_schemas_and_resumes_at_next_epoch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _TinyTrainingAdapter:
        def __init__(self, _policy_config, _norm_stats, *, device):
            self.device = torch.device(device)
            self._model = torch.nn.Linear(1, 1, bias=False).to(self.device)
            self._optimizer = torch.optim.AdamW(self._model.parameters())

        def configure_optimizers(self):
            return self._optimizer

        def state_dict(self):
            return self._model.state_dict()

        def load_state_dict(self, state_dict, strict: bool = True):
            return self._model.load_state_dict(state_dict, strict=strict)

        def forward_loss(self, _proprio, _image, _action, _is_pad, **_extra):
            loss = self._model.weight.square().sum()
            return {"loss": loss, "l1": loss}

    monkeypatch.setattr(
        "testbed.policies.act.trainer.ACTAdapter",
        _TinyTrainingAdapter,
    )
    monkeypatch.setattr(ACTTrainer, "_plot_history", lambda *_args: None)
    with (tmp_path / "dataset_stats.pkl").open("wb") as file:
        pickle.dump({}, file)
    batch = (
        torch.zeros(1, 1),
        torch.zeros(1, 1),
        torch.zeros(1, 1),
        torch.zeros(1, 1, dtype=torch.bool),
    )
    config = {
        "num_epochs": 3,
        "ckpt_dir": str(tmp_path),
        "seed": 0,
        "device": "cpu",
        "val_every": 1,
        "save_latest_every": 2,
        "checkpoint_every": 2,
        "checkpoint_keep_last": 1,
        "plot_every": 10,
        "amp": False,
        "pin_memory": False,
    }

    ACTTrainer(policy_config={}, config=config).fit([batch], [batch])

    latest = load_resume_checkpoint(tmp_path / "policy_latest.ckpt")
    best = torch.load(tmp_path / "policy_best.ckpt", map_location="cpu")
    periodic = torch.load(
        tmp_path / "policy_epoch_1_seed_0.ckpt", map_location="cpu"
    )
    assert latest["checkpoint_kind"] == "resume"
    assert latest["epoch"] == 2
    assert best["checkpoint_kind"] == "inference"
    assert "optimizer_state_dict" not in best
    assert periodic["checkpoint_kind"] == "inference"
    assert "optimizer_state_dict" not in periodic
    assert (tmp_path / "policy_last.ckpt").stat().st_ino == (
        tmp_path / "policy_latest.ckpt"
    ).stat().st_ino

    resumed = dict(config)
    resumed["num_epochs"] = 4
    resumed["resume_ckpt"] = str(tmp_path / "policy_latest.ckpt")
    ACTTrainer(policy_config={}, config=resumed).fit([batch], [batch])
    assert load_resume_checkpoint(tmp_path / "policy_latest.ckpt")["epoch"] == 3
