from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, asdict, dataclass, fields, replace

import pytest

from entries.baseline import main as baseline
from nanorfd import ExperimentConfig
from nanorfd.base import parse_config


@dataclass(frozen=True, kw_only=True)
class OtherConfig(ExperimentConfig):
    hidden_dim: int = 32
    dropout: float = 0.0


def test_shared_config_is_independent_of_solution_settings():
    names = {field.name for field in fields(ExperimentConfig)}
    assert {"data", "seed", "steps", "max_minutes", "sample_steps"} <= names
    assert not {"width", "heads", "lr", "sigma_min", "batch_size"} & names
    shared = ExperimentConfig(seed=7, steps=100, max_minutes=2)
    a = baseline.Config.from_experiment(shared, width=64, heads=4)
    b = OtherConfig.from_experiment(shared, hidden_dim=16)
    for name in names:
        assert getattr(a, name) == getattr(b, name) == getattr(shared, name)
    assert a.width == 64 and b.hidden_dim == 16
    with pytest.raises(FrozenInstanceError):
        a.seed = 999
    with pytest.raises(TypeError, match="seed"):
        baseline.Config.from_experiment(shared, seed=999)
    with pytest.raises(TypeError):
        baseline.Config.from_experiment(shared, nonexistent=1)


def test_cli_resolves_inherited_and_extended_fields(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"command": "train", "seed": 3, "hidden_dim": 64, "dropout": 0.1}))
    config = parse_config(
        OtherConfig, ["sample", "--config", str(path), "--seed", "9", "--no-bf16"]
    )
    assert type(config) is OtherConfig
    assert config.command == "sample" and config.seed == 9
    assert config.hidden_dim == 64 and config.dropout == 0.1 and config.bf16 is False
    assert OtherConfig(**asdict(config)) == config


def test_entry_main_receives_config_object(monkeypatch):
    shared = ExperimentConfig(seed=13, steps=10)
    config = baseline.Config.from_experiment(shared, width=32, heads=4)
    seen = []
    monkeypatch.setattr(baseline, "train", lambda cfg: seen.append(cfg))
    baseline.main(config)
    assert seen == [config] and seen[0] is config
    seen.clear()
    monkeypatch.setattr(baseline, "sample", lambda cfg: seen.append(cfg))
    sampling = replace(config, command="sample")
    baseline.main(sampling)
    assert seen == [sampling]


def test_unknown_fields_fail_at_dataclass_construction(tmp_path):
    path = tmp_path / "unknown.json"
    path.write_text('{"not_a_setting": true}')
    with pytest.raises(TypeError, match="not_a_setting"):
        parse_config(OtherConfig, ["train", "--config", str(path)])
