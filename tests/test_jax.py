"""Scientific parity and lifecycle checks for the optional functional JAX entry."""

import json
from dataclasses import replace
from functools import partial

import numpy as np
import pytest
import torch

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("optax")

from entries.baseline import main as torch_entry
from entries.baseline_jax import main as n


def config(**kwargs):
    return n.Config(
        width=32,
        depth=1,
        heads=4,
        mlp_ratio=2,
        batch_size=2,
        max_length=8,
        length=6,
        steps=4,
        warmup_steps=1,
        save_every=2,
        log_every=2,
        max_minutes=0,
        sample_steps=4,
        num_samples=2,
        device="cpu",
        threads=1,
        **kwargs,
    )


def torch_parameters(model):
    """Translate weights only, so both independently implemented forwards see identical inputs."""

    def linear(layer):
        result = {"w": jnp.asarray(layer.weight.detach().numpy().T)}
        if layer.bias is not None:
            result["b"] = jnp.asarray(layer.bias.detach().numpy())
        return result

    def norm(layer):
        return {
            "scale": jnp.asarray(layer.weight.detach().numpy()),
            "offset": jnp.asarray(layer.bias.detach().numpy()),
        }

    return {
        "input": linear(model.input),
        "noise1": linear(model.noise_embed[0]),
        "noise2": linear(model.noise_embed[2]),
        "length": linear(model.length_embed),
        "blocks": tuple(
            {
                "norm1": norm(b.norm1),
                "norm2": norm(b.norm2),
                "qkv": linear(b.attention.qkv),
                "attention_out": linear(b.attention.out),
                "up": linear(b.mlp[0]),
                "down": linear(b.mlp[2]),
            }
            for b in model.blocks
        ),
        "norm": norm(model.output[0]),
        "output": linear(model.output[1]),
    }


def test_jax_forward_and_gradients_match_torch_and_ignore_padding():
    torch.set_num_threads(1)
    torch.manual_seed(13)
    cfg = config()
    model = torch_entry.Denoiser(torch_entry.Config(**cfg.to_dict()))
    torch.nn.init.normal_(model.output[-1].weight, std=0.1)
    params = torch_parameters(model)
    x = torch.randn(2, 6, 3)
    mask = torch.arange(6)[None, :] < torch.tensor([4, 6])[:, None]
    sigma = torch.tensor([0.1, 3.0])
    expected = torch_entry.denoise(model, x, sigma, mask)
    forward = jax.jit(partial(n.denoise, cfg=cfg))
    actual = forward(
        params, jnp.array(x.numpy()), jnp.array(sigma.numpy()), jnp.array(mask.numpy())
    )
    np.testing.assert_allclose(actual, expected.detach().numpy(), atol=2e-6, rtol=2e-5)
    padded = jnp.pad(jnp.array(x.numpy()), ((0, 0), (0, 2), (0, 0)), constant_values=999)
    padded_mask = jnp.pad(jnp.array(mask.numpy()), ((0, 0), (0, 2)))
    pred = forward(params, padded, jnp.array(sigma.numpy()), padded_mask)
    np.testing.assert_allclose(pred[:, :6], actual, atol=2e-6, rtol=2e-5)
    assert np.count_nonzero(np.asarray(pred[:, 6:])) == 0

    expected.square().sum().backward()
    grads = jax.grad(
        lambda p: jnp.square(
            forward(p, jnp.array(x.numpy()), jnp.array(sigma.numpy()), jnp.array(mask.numpy()))
        ).sum()
    )(params)
    np.testing.assert_allclose(
        grads["output"]["w"], model.output[-1].weight.grad.numpy().T, atol=3e-5, rtol=5e-4
    )
    np.testing.assert_allclose(
        grads["input"]["w"], model.input.weight.grad.numpy().T, atol=3e-5, rtol=5e-4
    )


def test_jax_rotation_and_noise_loss_are_reproducible():
    cfg = config()
    key = jax.random.PRNGKey(4)
    x = jax.random.normal(key, (2, 8, 3))
    mask = jnp.ones((2, 8), dtype=bool)
    rotated = n.CATokenizer.augment(x, mask, key)
    np.testing.assert_allclose(
        jnp.linalg.norm(rotated[:, :, None] - rotated[:, None, :], axis=-1),
        jnp.linalg.norm(x[:, :, None] - x[:, None, :], axis=-1),
        atol=2e-6,
    )
    np.testing.assert_allclose(
        jnp.linalg.det(rotated[:, 1:4] - rotated[:, :1]),
        jnp.linalg.det(x[:, 1:4] - x[:, :1]),
        atol=1e-5,
    )
    params = n.init_model(key, cfg)
    loss_fn = jax.jit(partial(n.diffusion_loss, cfg=cfg))
    a = loss_fn(params, x, mask, key=key)
    b = loss_fn(params, x, mask, key=key)
    assert np.isfinite(float(a)) and float(a) == float(b)


def test_jax_sampler_endpoint_with_oracle(monkeypatch):
    cfg = config()
    target = n.CATokenizer.center(
        jnp.arange(42).reshape(2, 7, 3) / 10, jnp.arange(7)[None, :] < jnp.array([4, 7])[:, None]
    )
    monkeypatch.setattr(n, "denoise", lambda params, x, sigma, mask, cfg, dtype: target)
    samples = jax.jit(partial(n.sample_coordinates, lengths=(4, 7), cfg=cfg))(
        {}, key=jax.random.PRNGKey(2)
    )
    np.testing.assert_allclose(samples[0], target[0, :4], atol=2e-5)
    np.testing.assert_allclose(samples[1], target[1], atol=2e-5)


def test_jax_training_exact_resume_and_shared_evaluation(tmp_path, monkeypatch):
    records = []
    for length in (6, 8):
        i = np.arange(length, dtype=np.float32)
        coords = np.column_stack((2.3 * np.cos(i * 1.74), 2.3 * np.sin(i * 1.74), i * 1.5))
        atoms = torch_entry.CATokenizer.decode(coords)
        # Both entries run the same AtomWorks transform contract on a loaded AtomArray.
        records.append(n.CATokenizer()(atoms, sample_id=str(length)))
    data = tmp_path / "data.pt"
    torch.save({"version": 1, "train": records}, data)
    cfg = config(data=str(data), out_dir=str(tmp_path / "full"))
    n.main(cfg)
    _, reference = n.load_checkpoint(tmp_path / "full/checkpoint.npz")
    original_save = n.save_checkpoint

    class Interrupted(Exception):
        pass

    def stop_at_two(*args):
        original_save(*args)
        if int(args[1]["step"]) == 2:
            raise Interrupted

    monkeypatch.setattr(n, "save_checkpoint", stop_at_two)
    partial_cfg = replace(cfg, out_dir=str(tmp_path / "partial"))
    with pytest.raises(Interrupted):
        n.main(partial_cfg)
    monkeypatch.setattr(n, "save_checkpoint", original_save)
    resumed = replace(partial_cfg, resume=str(tmp_path / "partial/checkpoint.npz"))
    n.main(resumed)
    _, actual = n.load_checkpoint(resumed.resume)
    assert actual.keys() == reference.keys()
    for name in reference:  # Includes Adam moments/counts, EMA, PRNG, and step.
        np.testing.assert_array_equal(actual[name], reference[name], err_msg=name)
    logs = [json.loads(line) for line in (tmp_path / "full/metrics.jsonl").read_text().splitlines()]
    assert all(np.isfinite(row["train_loss"]) and "val_loss" not in row for row in logs)
    initial = n.init_state(cfg, n.make_optimizer(cfg))["params"]
    trained = n.restore_tree(reference, "params", initial)
    assert np.count_nonzero(np.asarray(trained["output"]["w"])) > 0
    assert not np.array_equal(trained["input"]["w"], initial["input"]["w"])

    generation = replace(
        cfg, command="sample", checkpoint=resumed.resume, out_dir=str(tmp_path / "samples")
    )
    report = n.main(generation)
    assert report["num_samples"] == 2
    assert len(list((tmp_path / "samples").glob("*.pdb"))) == 2
    eval_config = torch_entry.Config(
        **replace(
            generation,
            command="eval",
            samples=str(tmp_path / "samples/samples.pt"),
            out_dir=str(tmp_path / "eval"),
        ).to_dict()
    )
    assert torch_entry.main(eval_config) == report


def test_jax_config_rng_and_nanometer_collation():
    cfg = config(seed=9)
    np.testing.assert_array_equal(cfg.rng, jax.random.PRNGKey(9))
    np.testing.assert_array_equal(replace(cfg, seed=10).rng, jax.random.PRNGKey(10))
    assert "rng" not in cfg.to_dict()
    json.dumps(cfg.to_dict())
    coords = torch.arange(18).reshape(6, 3).float()
    records = [{"coords": coords}, {"coords": coords[:4]}]
    x, mask = n.CATokenizer.collate(records, cfg.max_length)
    tx, tm = torch_entry.CATokenizer.collate(records, "cpu")
    np.testing.assert_allclose(x[:, :6], tx.numpy(), rtol=1e-6)
    np.testing.assert_array_equal(mask[:, :6], tm.numpy())
    assert not np.any(np.asarray(mask[:, 6:]))
    assert np.count_nonzero(np.asarray(x[:, 6:])) == 0
