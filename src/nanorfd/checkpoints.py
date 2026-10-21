"""Checkpoint IO and state restoration shared by experiment entries."""

from dataclasses import asdict
from pathlib import Path

import torch


def load_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def save_checkpoint(path, model, ema, optimizer, step, rng, scale, cfg, data_hash):
    state = {
        "version": 1,
        "config": asdict(cfg),
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "rng": rng.get_state(),
        "coordinate_scale": scale,
        "data_sha256": data_hash,
    }
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def restore_checkpoint(path, model, ema, optimizer, rng) -> int:
    """Restore training state and return the completed step."""
    state = load_checkpoint(path)
    model.load_state_dict(state["model"])
    ema.load_state_dict(state["ema"])
    optimizer.load_state_dict(state["optimizer"])
    rng.set_state(state["rng"])
    return state["step"]
