"""Checkpoint IO and resume checks shared by experiment entries."""

from dataclasses import asdict, fields
from pathlib import Path

import torch


def load_checkpoint(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("version") != 1:
        raise ValueError("Unsupported checkpoint version")
    return state


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


def restore_checkpoint(path, model, ema, optimizer, rng, scale, cfg, data_hash) -> int:
    """Validate a resume before restoring training state; return the completed step."""
    state = load_checkpoint(path)
    for field in fields(cfg):
        if not field.metadata.get("resume_override", False) and (
            field.name not in state["config"]
            or state["config"][field.name] != getattr(cfg, field.name)
        ):
            raise ValueError(f"Resume setting changed: {field.name}; use the saved config")
    if state["data_sha256"] != data_hash or state["coordinate_scale"] != scale:
        raise ValueError("Resume dataset or normalization changed")
    if state["step"] >= cfg.steps:
        raise ValueError("This checkpoint already completed the configured training steps")
    model.load_state_dict(state["model"])
    ema.load_state_dict(state["ema"])
    optimizer.load_state_dict(state["optimizer"])
    rng.set_state(state["rng"])
    return state["step"]
