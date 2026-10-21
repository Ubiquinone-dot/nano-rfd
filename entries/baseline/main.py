"""C-alpha diffusion in one file: prepare -> train -> sample -> eval.

The model is an ordinary bidirectional Transformer with RoPE. Coordinates are
continuous tokens; there are no pair features, pretrained weights, or varlen kernels.
Run `python entries/baseline/main.py --help` for options and see README.md for the data contract.
"""

import copy
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, replace
from importlib.metadata import version
from itertools import pairwise
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from atomworks.ml.transforms.base import Compose, Transform
from biotite.structure import AtomArray, filter_amino_acids
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from nanorfd.base import ExperimentConfig, parse_config
from nanorfd.checkpoints import load_checkpoint, restore_checkpoint, save_checkpoint
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
    """Center valid coordinates and leave padding exactly zero."""
    x = x.masked_fill(~mask[..., None], 0)
    mean = x.sum(dim=1, keepdim=True) / mask.sum(dim=1)[:, None, None].clamp_min(1)
    return (x - mean).masked_fill(~mask[..., None], 0)


def coordinate_scale(records):
    # One training-set RMS per Cartesian coordinate; never normalize each protein's size.
    total, count = 0.0, 0
    for record in records:
        x = record["coords"].double()
        total += (x - x.mean(0)).square().sum().item()
        count += x.numel()
    scale = math.sqrt(total / count)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid training coordinate scale")
    return scale


def make_batch(records, batch_size, scale, rng, device, augment=True):
    indices = torch.randint(len(records), (batch_size,), generator=rng)
    xs = [records[i]["coords"] for i in indices.tolist()]
    lengths = torch.tensor([len(x) for x in xs])
    x = pad_sequence(xs, batch_first=True)
    mask = torch.arange(x.shape[1])[None, :] < lengths[:, None]
    x = center(x, mask) / scale
    if augment:
        # QR with sign correction gives uniform orthogonal matrices; det=+1 excludes reflections.
        q, r = torch.linalg.qr(torch.randn(batch_size, 3, 3, generator=rng))
        q *= torch.diagonal(r, dim1=-2, dim2=-1).sign()[:, None, :]
        q[:, :, 0] *= torch.linalg.det(q)[:, None]
        x = x @ q
    return x.to(device), mask.to(device)


# Ordinary padded attention. RoPE uses residue order, not spatial coordinates.
class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.heads = cfg.heads
        dim = cfg.width // cfg.heads
        self.register_buffer(
            "freqs", 10000.0 ** (-torch.arange(0, dim, 2).float() / dim), persistent=False
        )
        self.qkv = nn.Linear(cfg.width, 3 * cfg.width, bias=False)
        self.out = nn.Linear(cfg.width, cfg.width, bias=False)

    def rope(self, x):
        angles = torch.arange(x.shape[-2], device=x.device)[:, None] * self.freqs[None, :]
        cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)

    def forward(self, h, mask):
        b, length, width = h.shape
        q, k, v = self.qkv(h).reshape(b, length, 3, self.heads, -1).unbind(2)
        q, k, v = (x.transpose(1, 2) for x in (q, k, v))
        # In SDPA a boolean True means the key is allowed. No causal mask or pair bias.
        out = F.scaled_dot_product_attention(
            self.rope(q), self.rope(k), v, attn_mask=mask[:, None, None, :], dropout_p=0.0
        )
        return self.out(out.transpose(1, 2).reshape(b, length, width))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(cfg.width), nn.LayerNorm(cfg.width)
        self.attention = Attention(cfg)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.width, cfg.mlp_ratio * cfg.width),
            nn.GELU(),
            nn.Linear(cfg.mlp_ratio * cfg.width, cfg.width),
        )

    def forward(self, h, mask):
        h = h + self.attention(self.norm1(h), mask)
        h = h + self.mlp(self.norm2(h))
        return h.masked_fill(~mask[..., None], 0)


class Denoiser(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.max_length = cfg.max_length
        self.register_buffer(
            "noise_freqs",
            torch.exp(-math.log(10000) * torch.arange(cfg.width // 2).float() / (cfg.width // 2)),
            persistent=False,
        )
        self.input = nn.Linear(3, cfg.width)
        self.noise_embed = nn.Sequential(
            nn.Linear(cfg.width, cfg.width), nn.SiLU(), nn.Linear(cfg.width, cfg.width)
        )
        self.length_embed = nn.Linear(1, cfg.width)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.depth)])
        self.output = nn.Sequential(nn.LayerNorm(cfg.width), nn.Linear(cfg.width, 3))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, x, noise_level, mask):
        angles = noise_level[:, None] * self.noise_freqs[None, :]
        noise = self.noise_embed(torch.cat((angles.sin(), angles.cos()), dim=-1))
        length = self.length_embed(mask.sum(-1, keepdim=True).float() / self.max_length)
        h = self.input(x.masked_fill(~mask[..., None], 0)) + (noise + length)[:, None, :]
        for block in self.blocks:
            h = block(h, mask)
        return self.output(h).float().masked_fill(~mask[..., None], 0)


# EDM in units where training-set sigma_data = 1. All geometric arithmetic is float32.
def denoise(model, x, sigma, mask):
    s = sigma[:, None, None]
    c_in = (1 + s.square()).rsqrt()
    raw = model(x * c_in, sigma.log() / 4, mask)
    return center(x / (1 + s.square()) + s * c_in * raw, mask)


def diffusion_loss(model, x, mask, cfg, rng):
    # A dedicated CPU generator owns all stochastic training inputs, including on CUDA.
    # Saving this state makes batch/noise resume exact.
    log_sigma = torch.randn(len(x), generator=rng) * cfg.log_sigma_std + cfg.log_sigma_mean
    sigma = log_sigma.exp().clamp(cfg.sigma_min, cfg.sigma_max).to(x.device)
    noise = center(torch.randn(x.shape, generator=rng).to(x.device), mask)
    prediction = denoise(model, x + sigma[:, None, None] * noise, sigma, mask)
    error = (prediction - x).square().masked_fill(~mask[..., None], 0).sum((1, 2))
    error = error / (3 * mask.sum(1))  # Equal weight per protein, regardless of its length.
    return (error * (1 + sigma.square()) / sigma.square()).mean()


@torch.no_grad()
def sample_coordinates(model, lengths, cfg, rng, device):
    lengths = torch.as_tensor(lengths, device=device)
    mask = torch.arange(int(lengths.max()), device=device)[None, :] < lengths[:, None]
    x = center(torch.randn((*mask.shape, 3), generator=rng).to(device), mask) * cfg.sigma_max
    ramp = torch.linspace(0, 1, cfg.sample_steps, device=device)
    schedule = (
        cfg.sigma_max ** (1 / cfg.rho)
        + ramp * (cfg.sigma_min ** (1 / cfg.rho) - cfg.sigma_max ** (1 / cfg.rho))
    ) ** cfg.rho
    schedule = torch.cat((schedule, schedule.new_zeros(1)))
    for sigma, next_sigma in pairwise(schedule):
        prediction = denoise(model, x, sigma.expand(len(x)), mask)
        x = center(x + (next_sigma - sigma) * (x - prediction) / sigma, mask)
    if not torch.isfinite(x).all():
        raise ValueError("NaN/Inf during sampling")
    return [coords[: int(length)].cpu() for coords, length in zip(x, lengths)]


def runtime(cfg):
    torch.set_num_threads(cfg.threads)
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if cfg.device == "auto"
        else torch.device(cfg.device)
    )
    return device


def autocast(cfg, device):
    return torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda" and cfg.bf16
    )


def learning_rate(step, cfg):
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps - 1)
    return cfg.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def train(cfg):
    started = time.monotonic()
    device = runtime(cfg)
    torch.manual_seed(cfg.seed)
    data_hash = file_hash(cfg.data)
    data = torch.load(cfg.data, map_location="cpu", weights_only=True)
    scale = coordinate_scale(data["train"])
    model = Denoiser(cfg).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    rng = torch.Generator().manual_seed(cfg.seed + 1)
    step = 0
    out = Path(cfg.out_dir)
    if cfg.resume:
        step = restore_checkpoint(cfg.resume, model, ema, optimizer, rng)
    elif (out / "config.json").exists() or (out / "checkpoint.pt").exists():
        raise ValueError(f"Run already exists: {out}; use --resume or a new --out-dir")
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2) + "\n")
    metadata = {
        "torch": str(torch.__version__),
        "atomworks": version("atomworks"),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        "data_sha256": data_hash,
        "source_sha256": file_hash(__file__),
        "coordinate_scale_angstrom": scale,
        "parameters": sum(p.numel() for p in model.parameters()),
        "train_structures": len(data["train"]),
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata), flush=True)
    while step < cfg.steps:
        model.train()
        x, mask = make_batch(data["train"], cfg.batch_size, scale, rng, device)
        optimizer.zero_grad(set_to_none=True)
        lr = learning_rate(step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr
        with autocast(cfg, device):
            loss = diffusion_loss(model, x, mask, cfg, rng)
        if not torch.isfinite(loss):
            raise ValueError(f"NaN/Inf loss at step {step}")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip, error_if_nonfinite=True)
        optimizer.step()
        with torch.no_grad():
            for average, current in zip(ema.parameters(), model.parameters()):
                average.lerp_(current, 1 - cfg.ema_decay)
        step += 1
        elapsed = time.monotonic() - started
        timed_out = cfg.max_minutes > 0 and elapsed >= cfg.max_minutes * 60
        save_now = step % cfg.save_every == 0 or step == cfg.steps or timed_out
        if step % cfg.log_every == 0 or save_now:
            row = {"step": step, "train_loss": loss.item(), "lr": lr, "elapsed_seconds": elapsed}
            if save_now:
                save_checkpoint(
                    out / "checkpoint.pt", model, ema, optimizer, step, rng, scale, cfg, data_hash
                )
            with (out / "metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(row, allow_nan=False) + "\n")
            print(json.dumps(row), flush=True)
        if timed_out:
            break
    print(f"Checkpoint: {out / 'checkpoint.pt'}", flush=True)


def sample(cfg):
    device = runtime(cfg)
    state = load_checkpoint(cfg.checkpoint)
    trained = Config(**state["config"])
    model = Denoiser(trained).to(device).eval()
    model.load_state_dict(state["ema"])
    # The checkpoint owns diffusion units/bounds; inference controls only step count and rho.
    trained = replace(trained, sample_steps=cfg.sample_steps, rho=cfg.rho)
    out = Path(cfg.out_dir)
    if (out / "samples.pt").exists():
        raise ValueError(f"Refusing to overwrite samples in {out}")
    out.mkdir(parents=True, exist_ok=True)
    rng = torch.Generator().manual_seed(cfg.seed)
    records = []
    for start in range(0, cfg.num_samples, cfg.batch_size):
        count = min(cfg.batch_size, cfg.num_samples - start)
        with autocast(cfg, device):
            coords = sample_coordinates(model, [cfg.length] * count, trained, rng, device)
        for i, x in enumerate(coords, start):
            record = CATokenizer.encode(x * state["coordinate_scale"], f"sample_{i:04d}")
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
    evaluate({r["id"]: CATokenizer.decode(r["coords"]) for r in records}, out)


def main(config: Config):
    """Run this entry with a fully resolved, immutable experiment configuration."""
    if config.command == "prepare":
        return prepare(config)
    if config.command == "train":
        return train(config)
    if config.command == "sample":
        return sample(config)
    records = torch.load(config.samples, map_location="cpu", weights_only=True)["samples"]
    return evaluate({r["id"]: CATokenizer.decode(r["coords"]) for r in records}, config.out_dir)


if __name__ == "__main__":
    main(parse_config(Config))
