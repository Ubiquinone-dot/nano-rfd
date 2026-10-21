from dataclasses import dataclass, replace

import pytest
import torch

from nanorfd.base import ExperimentConfig, runtime_setting
from nanorfd.checkpoints import load_checkpoint, restore_checkpoint, save_checkpoint


@dataclass(frozen=True, kw_only=True)
class EntryConfig(ExperimentConfig):
    hidden_dim: int = 4
    sampling_temperature: float = runtime_setting(1.0)


def test_resume_policy_extends_to_new_entries_and_checks_data(tmp_path):
    cfg = EntryConfig(steps=8)
    model = torch.nn.Linear(4, 3)
    ema = torch.nn.Linear(4, 3)
    optimizer = torch.optim.AdamW(model.parameters())
    rng = torch.Generator().manual_seed(7)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, ema, optimizer, 4, rng, 2.0, cfg, "original-data")
    saved_rng = rng.get_state()
    rng.manual_seed(999)
    changed_runtime = replace(cfg, out_dir="elsewhere", max_minutes=1, sampling_temperature=2.0)
    assert (
        restore_checkpoint(path, model, ema, optimizer, rng, 2.0, changed_runtime, "original-data")
        == 4
    )
    assert torch.equal(rng.get_state(), saved_rng)
    for change in ({"hidden_dim": 8}, {"steps": 10}, {"seed": 9}, {"bf16": False}):
        with pytest.raises(ValueError, match="Resume setting changed"):
            restore_checkpoint(
                path, model, ema, optimizer, rng, 2.0, replace(cfg, **change), "original-data"
            )
    for scale, digest in ((3.0, "original-data"), (2.0, "different-data")):
        with pytest.raises(ValueError, match="dataset or normalization changed"):
            restore_checkpoint(path, model, ema, optimizer, rng, scale, cfg, digest)
    state = load_checkpoint(path)
    state["config"].pop("hidden_dim")
    torch.save(state, path)
    with pytest.raises(ValueError, match="Resume setting changed: hidden_dim"):
        restore_checkpoint(path, model, ema, optimizer, rng, 2.0, cfg, "original-data")


def test_checkpoint_version_and_completed_run_checks(tmp_path):
    cfg = EntryConfig(steps=4)
    model = torch.nn.Linear(4, 3)
    optimizer = torch.optim.AdamW(model.parameters())
    rng = torch.Generator()
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, model, optimizer, 4, rng, 2.0, cfg, "data")
    with pytest.raises(ValueError, match="already completed"):
        restore_checkpoint(path, model, model, optimizer, rng, 2.0, cfg, "data")
    torch.save({"version": 999}, path)
    with pytest.raises(ValueError, match="Unsupported checkpoint version"):
        load_checkpoint(path)
