"""The shared experiment settings and a small CLI for solution-specific configs.

Each entry extends ExperimentConfig with a frozen dataclass and exposes main(config).
An evaluator can use EntryConfig.from_experiment(shared, **solution_settings) to
preserve the same experiment settings across entries.
"""

import argparse
import json
import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import TypeVar, get_type_hints


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
    eval_every: int = 250
    eval_batches: int = 4
    eval_batch_size: int = 64
    sample_steps: int = 32
    length: int = 64
    num_samples: int = 16

    def validate(self):
        # Inspect the concrete subclass too, including annotations postponed by __future__.
        hints = get_type_hints(type(self))
        for field in fields(self):
            value, expected = getattr(self, field.name), hints[field.name]
            if type(value) is not expected and not (expected is float and type(value) is int):
                raise ValueError(f"{field.name} must be {expected.__name__}")
            if expected is float and not math.isfinite(value):
                raise ValueError(f"{field.name} must be finite")
        if self.command not in ("prepare", "train", "sample", "eval"):
            raise ValueError("command must be prepare, train, sample, or eval")
        positive = (
            "threads",
            "steps",
            "log_every",
            "eval_every",
            "eval_batches",
            "eval_batch_size",
            "num_samples",
        )
        if any(getattr(self, key) <= 0 for key in positive):
            raise ValueError(f"These settings must be positive: {', '.join(positive)}")
        if not 4 <= self.min_length <= self.max_length or self.length < 4:
            raise ValueError("Require 4 <= min_length <= max_length and length >= 4")
        if self.sample_steps < 2 or self.max_minutes < 0:
            raise ValueError("Require sample_steps >= 2 and max_minutes >= 0")

    @classmethod
    def from_experiment(cls, experiment, **solution_settings):
        """Extend an evaluator's settings without allowing solution overrides of shared fields."""
        shared = {field.name: getattr(experiment, field.name) for field in fields(ExperimentConfig)}
        conflicts = shared.keys() & solution_settings.keys()
        if conflicts:
            raise ValueError(
                f"Cannot override shared experiment settings: {', '.join(sorted(conflicts))}"
            )
        config = cls(**shared, **solution_settings)
        config.validate()
        return config


ConfigT = TypeVar("ConfigT", bound=ExperimentConfig)


def parse_config(config_type: type[ConfigT], argv=None) -> ConfigT:
    """Resolve defaults < JSON < CLI into the entry's config. Only primitive fields are supported."""
    parser = argparse.ArgumentParser(description="nanorfd experiment entry")
    parser.add_argument("command", choices=("prepare", "train", "sample", "eval"))
    parser.add_argument("--config", help="JSON configuration overrides")
    hints = get_type_hints(config_type)
    for field in fields(config_type):
        if field.name == "command":
            continue
        kind = hints[field.name]
        kwargs = {"action": argparse.BooleanOptionalAction} if kind is bool else {"type": kind}
        parser.add_argument("--" + field.name.replace("_", "-"), default=None, **kwargs)
    args = vars(parser.parse_args(argv))
    config_path = args.pop("config")
    try:
        values = json.loads(Path(config_path).read_text()) if config_path else {}
        if not isinstance(values, dict):
            raise TypeError("Config JSON must be an object")
        values.update({key: value for key, value in args.items() if value is not None})
        config = config_type(**values)
        config.validate()
        return config
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))
