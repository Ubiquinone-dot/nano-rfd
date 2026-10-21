"""The shared experiment settings and a small CLI for solution-specific configs.

Each entry extends ExperimentConfig with a frozen dataclass and exposes main(config).
An evaluator can use EntryConfig.from_experiment(shared, **solution_settings) to
preserve the same experiment settings across entries.
"""

import argparse
import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import TypeVar, get_type_hints

import torch

##############################################
# Experiment settings and runtime RNG
##############################################


@dataclass(frozen=True, kw_only=True)
class ExperimentConfig:
    command: str = "train"
    data: str = "data/ca.pt"
    out_dir: str = "runs/default"
    manifest: str = ""
    checkpoint: str = ""
    resume: str = ""
    samples: str = ""
    device: str = "auto"
    seed: int = 42
    threads: int = 4
    bf16: bool = True  # CUDA only; CPU/MPS use float32.
    min_length: int = 4
    max_length: int = 128
    steps: int = 10000
    max_minutes: float = 50.0  # Zero disables the training time limit.
    log_every: int = 25
    save_every: int = 250
    sample_steps: int = 32
    length: int = 64
    num_samples: int = 16

    rng: torch.Generator = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        object.__setattr__(self, "rng", torch.Generator().manual_seed(self.seed))

    def to_dict(self):
        """Serialize settings, excluding runtime state such as the RNG."""
        return {f.name: getattr(self, f.name) for f in fields(self) if f.init}

    @classmethod
    def from_experiment(cls, experiment, **solution_settings):
        """Extend an evaluator's settings without allowing solution overrides of shared fields."""
        shared = {f.name: getattr(experiment, f.name) for f in fields(ExperimentConfig) if f.init}
        return cls(**shared, **solution_settings)


##############################################
# Command-line configuration
##############################################


ConfigT = TypeVar("ConfigT", bound=ExperimentConfig)


def parse_config(config_type: type[ConfigT], argv=None) -> ConfigT:
    """Resolve defaults < JSON < CLI into the entry's config. Only primitive fields are supported."""
    parser = argparse.ArgumentParser(description="nanorfd experiment entry")
    parser.add_argument("command", choices=("prepare", "train", "sample", "eval"))
    parser.add_argument("--config", help="JSON configuration overrides")
    hints = get_type_hints(config_type)
    for f in fields(config_type):
        if not f.init or f.name == "command":
            continue
        kind = hints[f.name]
        kwargs = {"action": argparse.BooleanOptionalAction} if kind is bool else {"type": kind}
        parser.add_argument("--" + f.name.replace("_", "-"), default=None, **kwargs)
    args = vars(parser.parse_args(argv))
    config_path = args.pop("config")
    values = json.loads(Path(config_path).read_text()) if config_path else {}
    values.update({key: value for key, value in args.items() if value is not None})
    return config_type(**values)
