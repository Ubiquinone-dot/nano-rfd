"""Functional JAX baseline: explicit parameter trees, PRNG keys, and optimizer state.

Model layers are plain functions; JIT compiles the loss/gradient/AdamW/EMA update.
AtomWorks preprocessing and the .pt data/sample format match the PyTorch entry.
Torch is used only for these CPU file/tokenizer boundaries, never for the model.
"""

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, replace
from functools import partial
from importlib.metadata import version
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
from atomworks.ml.transforms.base import Compose, Transform
from biotite.structure import AtomArray, filter_amino_acids

from nanorfd.base import ExperimentConfig, parse_config
from nanorfd.evals import evaluate


# This entry only owns model/optimizer/diffusion choices; experiment settings come from base.
@dataclass(frozen=True, kw_only=True)
class Config(ExperimentConfig):
    width: int = 256
    depth: int = 6
    heads: int = 8
    mlp_ratio: int = 4
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    grad_clip: float = 1.0
    ema_decay: float = 0.99
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    log_sigma_mean: float = -1.2
    log_sigma_std: float = 1.2
    rho: float = 7.0


# The tokenizer's AtomWorks pipeline selects a trace, then converts it to tensors.
class SelectCA(Transform):
    def forward(self, data):
        atoms = data["atom_array"]
        atoms = atoms[filter_amino_acids(atoms)]
        chain = data.get("chain") or str(atoms.chain_id[0])
        ca = atoms[(atoms.chain_id == chain) & (atoms.atom_name == "CA") & (atoms.element == "C")]
        return {**data, "atom_array": ca, "chain": chain}


class ExtractCATensor(Transform):
    requires_previous_transforms = ("SelectCA",)

    def forward(self, data):
        ca = data["atom_array"]
        sample = CATokenizer.encode(ca.coord.copy(), f"{data['id']}:{data['chain']}")
        sample["residue_id"] = torch.from_numpy(ca.res_id.copy()).long()
        return {**data, "sample": sample}


class CATokenizer:
    def __init__(self):
        self.pipeline = Compose([SelectCA(), ExtractCATensor()], track_rng_state=False)

    def __call__(self, atom_array, sample_id="sample", chain=None):
        data = {"atom_array": atom_array, "id": str(sample_id), "chain": chain}
        return self.pipeline(data)["sample"]

    def from_file(self, path, chain=None):
        from atomworks.io import parse
        from atomworks.io.config import ParseConfig

        # Use the first model of the asymmetric unit; do not invent missing atoms or assemblies.
        config = ParseConfig.from_preset("minimal", model=1, build_assembly=None)
        atoms = parse(str(path), config=config)["asym_unit"][0]
        return self(atoms, Path(path).resolve(), chain)

    @staticmethod
    def encode(coords, sample_id="sample"):
        coords = torch.as_tensor(coords, dtype=torch.float32).detach().cpu().contiguous()
        if not torch.isfinite(coords).all():
            raise ValueError(f"{sample_id}: NaN/Inf in coordinates")
        return {"id": str(sample_id), "coords": coords}

    @staticmethod
    def decode(coords) -> AtomArray:
        """Return a single-chain CA trace in angstroms; GLY is a placeholder."""
        coords = CATokenizer.encode(coords)["coords"]
        atoms = AtomArray(len(coords))
        atoms.coord = coords.numpy().copy()
        atoms.chain_id[:] = "A"
        atoms.res_id = np.arange(1, len(coords) + 1)
        atoms.res_name[:] = "GLY"
        atoms.atom_name[:] = "CA"
        atoms.element[:] = "C"
        return atoms

    @staticmethod
    def to_pdb(coords, path):
        coords = CATokenizer.encode(coords)["coords"]
        lines = ["REMARK CA TRACE ONLY; GLY IS A PLACEHOLDER, NOT A DESIGNED SEQUENCE"]
        for i, (x, y, z) in enumerate(coords.tolist(), 1):
            lines.append(
                f"ATOM  {i:5d}  CA  GLY A{i:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{0.0:6.2f}           C"
            )
        Path(path).write_text("\n".join([*lines, "TER", "END", ""]))


def file_hash(path):
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def prepare(cfg):
    path = Path(cfg.manifest).resolve()
    manifest = json.loads(path.read_text())
    data = {"version": 1, "manifest_sha256": file_hash(path)}
    tokenizer = CATokenizer()
    for split, entries in manifest.items():
        data[split] = []
        for entry in entries:
            structure_path = path.parent / entry["path"]
            record = tokenizer.from_file(structure_path, entry.get("chain"))
            record["source_sha256"] = file_hash(structure_path)
            data[split].append(record)
    output = Path(cfg.data)
    if output.exists():
        raise ValueError(f"Refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, output)
    print(f"Saved {len(data['train'])} training structures to {output}")


def center(x, mask):
    x = jnp.where(mask[..., None], x, 0)
    mean = x.sum(1, keepdims=True) / jnp.maximum(mask.sum(1)[:, None, None], 1)
    return jnp.where(mask[..., None], x - mean, 0)


def coordinate_scale(records):
    total, count = 0.0, 0
    for record in records:
        x = np.asarray(record["coords"], dtype=np.float64)
        total += ((x - x.mean(0)) ** 2).sum()
        count += x.size
    scale = float(np.sqrt(total / count))
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid training coordinate scale")
    return scale


def make_batch(records, cfg, scale, key):
    # Keep the dataset on the host and transfer only a batch. Fixed L avoids retracing.
    indices = np.asarray(jax.random.randint(key, (cfg.batch_size,), 0, len(records)))
    x = np.zeros((cfg.batch_size, cfg.max_length, 3), dtype=np.float32)
    mask = np.zeros(x.shape[:2], dtype=bool)
    for i, index in enumerate(indices):
        coords = np.asarray(records[index]["coords"])
        x[i, : len(coords)] = coords
        mask[i, : len(coords)] = True
    mask = jnp.asarray(mask)
    return center(jnp.asarray(x), mask) / scale, mask


def rotate(x, key):
    q, r = jnp.linalg.qr(jax.random.normal(key, (len(x), 3, 3)))
    q = q * jnp.sign(jnp.diagonal(r, axis1=-2, axis2=-1))[:, None, :]
    q = q.at[:, :, 0].multiply(jnp.linalg.det(q)[:, None])
    return x @ q


# Layers have explicit init/apply functions and ordinary dicts of float32 parameters.
def init_linear(key, fan_in, fan_out, bias=True):
    weight_key, bias_key = jax.random.split(key)
    bound = fan_in**-0.5  # Same initialization distribution as torch.nn.Linear.
    params = {"w": jax.random.uniform(weight_key, (fan_in, fan_out), minval=-bound, maxval=bound)}
    if bias:
        params["b"] = jax.random.uniform(bias_key, (fan_out,), minval=-bound, maxval=bound)
    return params


def linear(params, x, dtype=jnp.float32):
    y = jnp.matmul(x.astype(dtype), params["w"].astype(dtype), precision=jax.lax.Precision.HIGHEST)
    return y + params["b"].astype(dtype) if "b" in params else y


def init_norm(width):
    return {"scale": jnp.ones(width), "offset": jnp.zeros(width)}


def layer_norm(params, x):
    value = x.astype(jnp.float32)
    value = value - value.mean(-1, keepdims=True)
    value = value * jax.lax.rsqrt(jnp.mean(value**2, axis=-1, keepdims=True) + 1e-5)
    return (value * params["scale"] + params["offset"]).astype(x.dtype)


def init_block(key, cfg):
    qkv_key, out_key, up_key, down_key = jax.random.split(key, 4)
    return {
        "norm1": init_norm(cfg.width),
        "norm2": init_norm(cfg.width),
        "qkv": init_linear(qkv_key, cfg.width, 3 * cfg.width, bias=False),
        "attention_out": init_linear(out_key, cfg.width, cfg.width, bias=False),
        "up": init_linear(up_key, cfg.width, cfg.mlp_ratio * cfg.width),
        "down": init_linear(down_key, cfg.mlp_ratio * cfg.width, cfg.width),
    }


def init_model(key, cfg):
    keys = jax.random.split(key, 4 + cfg.depth)
    return {
        "input": init_linear(keys[0], 3, cfg.width),
        "noise1": init_linear(keys[1], cfg.width, cfg.width),
        "noise2": init_linear(keys[2], cfg.width, cfg.width),
        "length": init_linear(keys[3], 1, cfg.width),
        "blocks": tuple(init_block(k, cfg) for k in keys[4:]),
        "norm": init_norm(cfg.width),
        "output": {"w": jnp.zeros((cfg.width, 3)), "b": jnp.zeros(3)},
    }


def rope(x):
    # x: [B, H, L, D]; rotate adjacent even/odd channels by residue position.
    freqs = 10000.0 ** (-jnp.arange(0, x.shape[-1], 2, dtype=jnp.float32) / x.shape[-1])
    angles = jnp.arange(x.shape[-2])[:, None] * freqs[None, :]
    cos, sin = jnp.cos(angles).astype(x.dtype), jnp.sin(angles).astype(x.dtype)
    even, odd = x[..., 0::2], x[..., 1::2]
    return jnp.stack((even * cos - odd * sin, even * sin + odd * cos), axis=-1).reshape(x.shape)


def attention(params, h, mask, cfg, dtype=jnp.float32):
    b, length, width = h.shape
    qkv = linear(params["qkv"], h, dtype).reshape(b, length, 3, cfg.heads, -1)
    q, k, v = jnp.transpose(qkv, (2, 0, 3, 1, 4))
    # JAX attention accepts [B, L, H, D]; True mask entries permit keys.
    out = jax.nn.dot_product_attention(
        rope(q).swapaxes(1, 2),
        rope(k).swapaxes(1, 2),
        v.swapaxes(1, 2),
        mask=mask[:, None, None, :],
    )
    return linear(params["attention_out"], out.reshape(b, length, width), dtype)


def block(params, h, mask, cfg, dtype=jnp.float32):
    h = h + attention(params, layer_norm(params["norm1"], h), mask, cfg, dtype)
    hidden = linear(params["up"], layer_norm(params["norm2"], h), dtype)
    h = h + linear(params["down"], jax.nn.gelu(hidden, approximate=False), dtype)
    return jnp.where(mask[..., None], h, 0)


def model(params, x, noise_level, mask, cfg, dtype=jnp.float32):
    freqs = jnp.exp(-math.log(10000) * jnp.arange(cfg.width // 2) / (cfg.width // 2))
    angles = noise_level[:, None] * freqs[None, :]
    noise = jnp.concatenate((jnp.sin(angles), jnp.cos(angles)), axis=-1)
    noise = linear(params["noise2"], jax.nn.silu(linear(params["noise1"], noise, dtype)), dtype)
    length = linear(params["length"], mask.sum(-1, keepdims=True) / cfg.max_length, dtype)
    h = linear(params["input"], jnp.where(mask[..., None], x, 0), dtype)
    h = h + (noise + length)[:, None, :]
    for p in params["blocks"]:
        h = block(p, h, mask, cfg, dtype)
    out = linear(params["output"], layer_norm(params["norm"], h), dtype).astype(jnp.float32)
    return jnp.where(mask[..., None], out, 0)


def denoise(params, x, sigma, mask, cfg, dtype=jnp.float32):
    s = sigma[:, None, None]
    c_in = jax.lax.rsqrt(1 + s**2)
    raw = model(params, x * c_in, jnp.log(sigma) / 4, mask, cfg, dtype)
    return center(x / (1 + s**2) + s * c_in * raw, mask)


def diffusion_loss(params, x, mask, cfg, key, dtype=jnp.float32):
    sigma_key, noise_key = jax.random.split(key)
    log_sigma = jax.random.normal(sigma_key, (len(x),)) * cfg.log_sigma_std + cfg.log_sigma_mean
    sigma = jnp.clip(jnp.exp(log_sigma), cfg.sigma_min, cfg.sigma_max)
    noise = center(jax.random.normal(noise_key, x.shape), mask)
    prediction = denoise(params, x + sigma[:, None, None] * noise, sigma, mask, cfg, dtype)
    error = jnp.where(mask[..., None], (prediction - x) ** 2, 0).sum((1, 2)) / (3 * mask.sum(1))
    return (error * (1 + sigma**2) / sigma**2).mean()


def learning_rate(step, cfg):
    warmup = cfg.lr * (step + 1) / max(1, cfg.warmup_steps)
    progress = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps - 1)
    cosine = cfg.lr * (0.1 + 0.9 * 0.5 * (1 + jnp.cos(jnp.pi * progress)))
    return jnp.where(step < cfg.warmup_steps, warmup, cosine)


def make_optimizer(cfg):
    return optax.chain(
        optax.clip_by_global_norm(cfg.grad_clip),
        optax.adamw(lambda step: learning_rate(step, cfg), weight_decay=cfg.weight_decay),
    )


def init_state(cfg, optimizer):
    init_key, key = jax.random.split(jax.random.PRNGKey(cfg.seed))
    params = init_model(init_key, cfg)
    return {
        "params": params,
        "ema": params,
        "optimizer": optimizer.init(params),
        "key": key,
        "step": jnp.array(0, dtype=jnp.int32),
    }


def train_step(state, x, mask, cfg, optimizer, dtype=jnp.float32):
    """Pure update: (state, batch) -> (new state, loss, gradient norm)."""
    key, rotation_key, noise_key = jax.random.split(state["key"], 3)
    x = rotate(x, rotation_key)
    loss, grads = jax.value_and_grad(diffusion_loss)(
        state["params"], x, mask, cfg, noise_key, dtype
    )
    updates, opt_state = optimizer.update(grads, state["optimizer"], state["params"])
    params = optax.apply_updates(state["params"], updates)
    ema = jax.tree.map(
        lambda old, new: old + (1 - cfg.ema_decay) * (new - old), state["ema"], params
    )
    return (
        {
            "params": params,
            "ema": ema,
            "optimizer": opt_state,
            "key": key,
            "step": state["step"] + 1,
        },
        loss,
        jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree.leaves(grads))),
    )


def sample_coordinates(params, lengths, cfg, key, dtype=jnp.float32):
    # lengths is a static tuple when JIT-compiled; scans keep the Euler loop compact.
    mask = jnp.arange(max(lengths))[None, :] < jnp.asarray(lengths)[:, None]
    x = center(jax.random.normal(key, (*mask.shape, 3)), mask) * cfg.sigma_max
    ramp = jnp.linspace(0, 1, cfg.sample_steps)
    schedule = (
        cfg.sigma_max ** (1 / cfg.rho)
        + ramp * (cfg.sigma_min ** (1 / cfg.rho) - cfg.sigma_max ** (1 / cfg.rho))
    ) ** cfg.rho
    schedule = jnp.concatenate((schedule, jnp.zeros(1)))

    def euler(x, sigmas):
        sigma, next_sigma = sigmas
        prediction = denoise(params, x, jnp.full((len(x),), sigma), mask, cfg, dtype)
        x = center(x + (next_sigma - sigma) * (x - prediction) / sigma, mask)
        return x, None

    x, _ = jax.lax.scan(euler, x, (schedule[:-1], schedule[1:]))
    return tuple(coords[:length] for coords, length in zip(x, lengths))


# NPZ holds only numeric leaves and JSON. Rebuild the tree from the initialized state.
def save_checkpoint(path, state, cfg, scale, data_hash):
    metadata = {"config": asdict(cfg), "coordinate_scale": scale, "data_sha256": data_hash}
    arrays = {
        f"{name}/{i}": np.asarray(leaf)
        for name, tree in state.items()
        for i, leaf in enumerate(jax.tree.leaves(tree))
    }
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, metadata=json.dumps(metadata), **arrays)
    temporary.replace(path)


def load_checkpoint(path):
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        arrays = {name: archive[name] for name in archive.files if name != "metadata"}
    return metadata, arrays


def restore_tree(arrays, name, template):
    leaves, structure = jax.tree.flatten(template)
    return jax.tree.unflatten(
        structure, [jnp.asarray(arrays[f"{name}/{i}"]) for i in range(len(leaves))]
    )


def runtime(cfg):
    backend = None if cfg.device == "auto" else cfg.device.replace("cuda", "gpu")
    device = jax.devices(backend)[0]
    dtype = jnp.bfloat16 if cfg.bf16 and device.platform == "gpu" else jnp.float32
    return device, dtype


def train(cfg):
    started = time.monotonic()
    device, dtype = runtime(cfg)
    data_hash = file_hash(cfg.data)
    records = torch.load(cfg.data, map_location="cpu", weights_only=True)["train"]
    scale = coordinate_scale(records)
    optimizer = make_optimizer(cfg)
    state = init_state(cfg, optimizer)
    out = Path(cfg.out_dir)
    if cfg.resume:
        _, arrays = load_checkpoint(cfg.resume)
        state = {name: restore_tree(arrays, name, tree) for name, tree in state.items()}
    elif (out / "config.json").exists() or (out / "checkpoint.npz").exists():
        raise ValueError(f"Run already exists: {out}; use --resume or a new --out-dir")
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2) + "\n")
    metadata = {
        "jax": jax.__version__,
        "optax": optax.__version__,
        "atomworks": version("atomworks"),
        "device": str(device),
        "device_name": device.device_kind,
        "dtype": str(dtype),
        "data_sha256": data_hash,
        "source_sha256": file_hash(__file__),
        "coordinate_scale_angstrom": scale,
        "train_structures": len(records),
        "parameters": sum(p.size for p in jax.tree.leaves(state["params"])),
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata), flush=True)
    step_fn = jax.jit(partial(train_step, cfg=cfg, optimizer=optimizer, dtype=dtype))
    step = int(state["step"])
    while step < cfg.steps:
        batch_key, key = jax.random.split(state["key"])
        x, mask = make_batch(records, cfg, scale, batch_key)
        state, loss, grad_norm = step_fn({**state, "key": key}, x, mask)
        loss, grad_norm = float(loss), float(grad_norm)  # Synchronize before timing/checkpointing.
        if not math.isfinite(loss) or not math.isfinite(grad_norm):
            raise ValueError(f"NaN/Inf loss or gradient at step {step}")
        step += 1
        elapsed = time.monotonic() - started
        timed_out = cfg.max_minutes > 0 and elapsed >= cfg.max_minutes * 60
        save_now = step % cfg.save_every == 0 or step == cfg.steps or timed_out
        if step % cfg.log_every == 0 or save_now:
            row = {
                "step": step,
                "train_loss": loss,
                "lr": float(learning_rate(step - 1, cfg)),
                "elapsed_seconds": elapsed,
            }
            if save_now:
                save_checkpoint(out / "checkpoint.npz", state, cfg, scale, data_hash)
            with (out / "metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(row, allow_nan=False) + "\n")
            print(json.dumps(row), flush=True)
        if timed_out:
            break
    print(f"Checkpoint: {out / 'checkpoint.npz'}", flush=True)


def sample(cfg):
    _, dtype = runtime(cfg)
    metadata, arrays = load_checkpoint(cfg.checkpoint)
    trained = Config(**metadata["config"])
    params = restore_tree(arrays, "ema", init_model(jax.random.PRNGKey(0), trained))
    trained = replace(trained, sample_steps=cfg.sample_steps, rho=cfg.rho)
    sampler = jax.jit(
        partial(sample_coordinates, cfg=trained, dtype=dtype), static_argnames=("lengths",)
    )
    out = Path(cfg.out_dir)
    if (out / "samples.pt").exists():
        raise ValueError(f"Refusing to overwrite samples in {out}")
    out.mkdir(parents=True, exist_ok=True)
    key = jax.random.PRNGKey(cfg.seed)
    records = []
    for start in range(0, cfg.num_samples, cfg.batch_size):
        count = min(cfg.batch_size, cfg.num_samples - start)
        key, sample_key = jax.random.split(key)
        coords = sampler(params, lengths=(cfg.length,) * count, key=sample_key)
        for i, x in enumerate(coords, start):
            record = CATokenizer.encode(
                np.array(x) * metadata["coordinate_scale"], f"sample_{i:04d}"
            )
            records.append(record)
    torch.save(
        {
            "version": 1,
            "samples": records,
            "seed": cfg.seed,
            "checkpoint_sha256": file_hash(cfg.checkpoint),
            "sample_steps": cfg.sample_steps,
            "rho": cfg.rho,
            "batch_size": cfg.batch_size,
        },
        out / "samples.pt",
    )
    for record in records:
        CATokenizer.to_pdb(record["coords"], out / f"{record['id']}.pdb")
    return evaluate({r["id"]: CATokenizer.decode(r["coords"]) for r in records}, out)


def main(config: Config):
    if config.command == "prepare":
        return prepare(config)
    if config.command == "eval":
        records = torch.load(config.samples, map_location="cpu", weights_only=True)["samples"]
        return evaluate({r["id"]: CATokenizer.decode(r["coords"]) for r in records}, config.out_dir)
    device, _ = runtime(config)
    with jax.default_device(device):
        return {"train": train, "sample": sample}[config.command](config)


if __name__ == "__main__":
    main(parse_config(Config))
