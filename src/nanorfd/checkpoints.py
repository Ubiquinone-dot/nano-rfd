"""Checkpoint IO and state restoration shared by experiment entries."""

from pathlib import Path

import torch

##############################################
# Checkpoint IO
##############################################


def load_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def save_checkpoint(path, model, ema, optimizer, step, cfg):
    state = {
        "version": 1,
        "config": cfg.to_dict(),
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "rng": cfg.rng.get_state(),
        "coordinate_units": "nm",
    }
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


##############################################
# Training state restoration
##############################################


def restore_checkpoint(path, model, ema, optimizer, cfg) -> int:
    """Restore training state and return the completed step."""
    state = load_checkpoint(path)
    model.load_state_dict(state["model"])
    ema.load_state_dict(state["ema"])
    optimizer.load_state_dict(state["optimizer"])
    cfg.rng.set_state(state["rng"])
    return state["step"]
