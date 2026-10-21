"""C-alpha diffusion in one file: prepare -> train -> sample -> eval.

The model is an ordinary bidirectional Transformer with RoPE. Coordinates are
continuous tokens; there are no pair features, pretrained weights, or varlen kernels.
Run `python entries/baseline/main.py --help` for options and see README.md for the data contract.
"""

import copy
import csv
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
from biotite.structure import AtomArray, filter_amino_acids, get_residue_starts
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from nanorfd.base import ExperimentConfig, parse_config


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

    def validate(self):
        super().validate()
        positive = (
            "width",
            "depth",
            "heads",
            "mlp_ratio",
            "batch_size",
            "lr",
            "grad_clip",
            "sigma_min",
            "sigma_max",
            "log_sigma_std",
            "rho",
        )
        if any(getattr(self, key) <= 0 for key in positive):
            raise ValueError(f"These settings must be positive: {', '.join(positive)}")
        if self.width % self.heads or (self.width // self.heads) % 2:
            raise ValueError("width must divide evenly into heads with an even head dimension")
        if not self.sigma_min < self.sigma_max:
            raise ValueError("Require sigma_min < sigma_max")
        if not 0 <= self.ema_decay < 1 or min(self.weight_decay, self.warmup_steps) < 0:
            raise ValueError("Invalid EMA decay, weight decay, or warmup")


# The tokenizer's AtomWorks pipeline selects a trace, then converts it to tensors.
class SelectCA(Transform):
    def forward(self, data):
        atoms = data["atom_array"]
        if not isinstance(atoms, AtomArray):
            raise TypeError("Expected one AtomArray, not an ensemble/AtomArrayStack")
        atoms = atoms[filter_amino_acids(atoms)]
        if data.get("chain") is not None:
            atoms = atoms[atoms.chain_id == data["chain"]]
        chains = np.unique(atoms.chain_id)
        if len(chains) != 1:
            raise ValueError(f"Select one protein chain; found {chains.tolist()}")
        if (
            "chain_iid" in atoms.get_annotation_categories()
            and len(np.unique(atoms.chain_iid)) != 1
        ):
            raise ValueError("Expected one chain instance")
        is_ca = (atoms.atom_name == "CA") & (atoms.element == "C")
        starts = get_residue_starts(atoms)
        if not (np.add.reduceat(is_ca.astype(int), starts) == 1).all():
            raise ValueError("Each amino-acid residue must have exactly one CA atom")
        ca = atoms[is_ca]
        keys = list(zip(ca.res_id.tolist(), ca.ins_code.tolist()))
        for previous, current in pairwise(keys):
            delta = current[0] - previous[0]
            if delta not in (0, 1) or (delta == 0 and current[1] <= previous[1]):
                raise ValueError("Residue numbering gap, duplicate, or nonmonotonic order")
        if "occupancy" in ca.get_annotation_categories() and (
            not np.isfinite(ca.occupancy).all() or (ca.occupancy <= 0).any()
        ):
            raise ValueError("Unresolved CA atom or invalid occupancy")
        return {**data, "atom_array": ca, "chain": str(chains[0])}


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
    def encode(coords, sample_id="sample", check_bonds=True):
        coords = torch.as_tensor(coords, dtype=torch.float32).detach().cpu().contiguous()
        if coords.ndim != 2 or coords.shape[1] != 3 or len(coords) < 4:
            raise ValueError(f"{sample_id}: expected [L, 3] coordinates with L >= 4")
        if not torch.isfinite(coords).all():
            raise ValueError(f"{sample_id}: NaN/Inf in coordinates")
        bonds = (coords[1:] - coords[:-1]).norm(dim=-1)
        if check_bonds and ((bonds < 2.5) | (bonds > 4.5)).any():
            raise ValueError(f"{sample_id}: broken chain or implausible adjacent C-alpha distance")
        return {"id": str(sample_id), "coords": coords}

    @staticmethod
    def to_pdb(coords, path):
        coords = CATokenizer.encode(coords, check_bonds=False)["coords"]
        if (coords.abs() >= 1000).any() or len(coords) > 9999:
            raise ValueError("Coordinates or residue count exceed this minimal PDB writer's limits")
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


def validate_dataset(data, cfg):
    if data.get("version") != 1:
        raise ValueError("Expected dataset version 1")
    seen_ids, seen_coords = set(), set()
    for split in ("train", "val"):
        if not data.get(split):
            raise ValueError(f"Dataset needs a nonempty {split} split")
        for record in data[split]:
            sample = CATokenizer.encode(record["coords"], record["id"])
            if not cfg.min_length <= len(sample["coords"]) <= cfg.max_length:
                raise ValueError(
                    f"{sample['id']}: length outside configured range; no silent cropping"
                )
            # Detect literal duplicates too. Homology-aware splitting is the manifest author's job.
            digest = hashlib.sha256(json.dumps(sample["coords"].tolist()).encode()).hexdigest()
            if sample["id"] in seen_ids or digest in seen_coords:
                raise ValueError(f"Duplicate structure in dataset: {sample['id']}")
            seen_ids.add(sample["id"])
            seen_coords.add(digest)
            record.update(sample)


def prepare(cfg):
    if not cfg.manifest:
        raise ValueError(
            "prepare requires --manifest (JSON with explicit train/val structure entries)"
        )
    path = Path(cfg.manifest).resolve()
    manifest = json.loads(path.read_text())
    if set(manifest) != {"train", "val"}:
        raise ValueError("Manifest must contain exactly train and val lists")
    data = {"version": 1, "manifest_sha256": file_hash(path)}
    tokenizer = CATokenizer()
    for split, entries in manifest.items():
        data[split] = []
        for entry in entries:
            structure_path = path.parent / entry["path"]
            record = tokenizer.from_file(structure_path, entry.get("chain"))
            record["source_sha256"] = file_hash(structure_path)
            data[split].append(record)
    validate_dataset(data, cfg)
    output = Path(cfg.data)
    if output.exists():
        raise ValueError(f"Refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, output)
    print(f"Saved {len(data['train'])} train / {len(data['val'])} val structures to {output}")


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
    # Validation uses its own generator; saving this state makes batch/noise resume exact.
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
    if lengths.ndim != 1 or len(lengths) == 0 or (lengths < 4).any():
        raise ValueError("Sampling requires a nonempty list of lengths >= 4")
    if (lengths > model.max_length).any():
        raise ValueError("Requested length exceeds the trained maximum")
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
    if device.type == "cuda" and cfg.bf16 and not torch.cuda.is_bf16_supported():
        raise ValueError("This CUDA device does not support bf16; use --no-bf16")
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


@torch.no_grad()
def validation_loss(model, records, scale, cfg, device):
    model.eval()
    rng = torch.Generator().manual_seed(cfg.seed + 2)
    losses = []
    for _ in range(cfg.eval_batches):
        x, mask = make_batch(records, cfg.eval_batch_size, scale, rng, device)
        with autocast(cfg, device):
            losses.append(diffusion_loss(model, x, mask, cfg, rng).item())
    if not all(math.isfinite(loss) for loss in losses):
        raise ValueError("NaN/Inf in validation loss")
    return sum(losses) / len(losses)


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
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def train(cfg):
    started = time.monotonic()
    device = runtime(cfg)
    torch.manual_seed(cfg.seed)
    data_hash = file_hash(cfg.data)
    data = torch.load(cfg.data, map_location="cpu", weights_only=True)
    validate_dataset(data, cfg)
    scale = coordinate_scale(data["train"])
    model = Denoiser(cfg).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    rng = torch.Generator().manual_seed(cfg.seed + 1)
    step = 0
    out = Path(cfg.out_dir)
    if cfg.resume:
        state = torch.load(cfg.resume, map_location="cpu", weights_only=True)
        if state.get("version") != 1:
            raise ValueError("Unsupported checkpoint version")
        mutable = {
            "command",
            "data",
            "out_dir",
            "resume",
            "checkpoint",
            "manifest",
            "samples",
            "device",
            "threads",
            "log_every",
            "eval_every",
            "eval_batches",
            "eval_batch_size",
            "max_minutes",
            "length",
            "num_samples",
            "sample_steps",
            "rho",
        }
        for key, value in asdict(cfg).items():
            if key not in mutable and state["config"][key] != value:
                raise ValueError(f"Resume setting changed: {key}; use the saved config")
        if state["data_sha256"] != data_hash or state["coordinate_scale"] != scale:
            raise ValueError("Resume dataset or normalization changed")
        model.load_state_dict(state["model"])
        ema.load_state_dict(state["ema"])
        optimizer.load_state_dict(state["optimizer"])
        rng.set_state(state["rng"])
        step = state["step"]
        if step >= cfg.steps:
            raise ValueError("This checkpoint already completed the configured training steps")
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
        "val_structures": len(data["val"]),
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
        evaluate_now = step % cfg.eval_every == 0 or step == cfg.steps or timed_out
        if step % cfg.log_every == 0 or evaluate_now:
            row = {"step": step, "train_loss": loss.item(), "lr": lr, "elapsed_seconds": elapsed}
            if evaluate_now:
                row["val_loss"] = validation_loss(ema, data["val"], scale, cfg, device)
                save_checkpoint(
                    out / "checkpoint.pt", model, ema, optimizer, step, rng, scale, cfg, data_hash
                )
            with (out / "metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(row, allow_nan=False) + "\n")
            print(json.dumps(row), flush=True)
        if timed_out:
            break
    print(f"Checkpoint: {out / 'checkpoint.pt'}", flush=True)


# Fast C-alpha diagnostics, not an all-atom validity or designability score.
def geometry_metrics(record):
    x = CATokenizer.encode(record["coords"], record["id"], check_bonds=False)["coords"]
    bonds = x[1:] - x[:-1]
    distances = torch.cdist(x, x)
    index = torch.arange(len(x))
    separation = index[None, :] - index[:, None]  # Count each unordered pair once.
    nonlocal_pairs, distant_pairs = separation > 2, separation > 8
    triple = (torch.cross(bonds[:-2], bonds[1:-1], dim=-1) * bonds[2:]).sum(-1)
    norm = bonds[:-2].norm(dim=-1) * bonds[1:-1].norm(dim=-1) * bonds[2:].norm(dim=-1)
    normal_product = torch.cross(bonds[:-2], bonds[1:-1], dim=-1).norm(dim=-1) * torch.cross(
        bonds[1:-1], bonds[2:], dim=-1
    ).norm(dim=-1)
    return {
        "id": record["id"],
        "length": len(x),
        "ca_distance_mean": bonds.norm(dim=-1).mean().item(),
        "ca_distance_mae": (bonds.norm(dim=-1) - 3.8).abs().mean().item(),
        "nonlocal_clash_fraction": ((distances < 3) & nonlocal_pairs).sum().item()
        / max(1, nonlocal_pairs.sum().item()),
        "contact_fraction": ((distances < 8) & distant_pairs).sum().item()
        / max(1, distant_pairs.sum().item()),
        "contact_pair_count": distant_pairs.sum().item(),
        "radius_of_gyration": (x - x.mean(0)).square().sum(-1).mean().sqrt().item(),
        "signed_volume_mean": (triple / norm.clamp_min(1e-8)).mean().item(),
        "degenerate_triplets": (normal_product < 1e-8).sum().item(),
    }


def evaluate(records, out_dir):
    if not records:
        raise ValueError("Cannot evaluate an empty sample set")
    rows = [geometry_metrics(record) for record in records]
    if not all(math.isfinite(value) for row in rows for key, value in row.items() if key != "id"):
        raise ValueError("NaN/Inf in geometry metrics")
    summary = {"num_samples": len(rows)}
    summary.update(
        {key: sum(row[key] for row in rows) / len(rows) for key in rows[0] if key != "id"}
    )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "geometry.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "geometry.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary), flush=True)
    return summary


def sample(cfg):
    if not cfg.checkpoint:
        raise ValueError("sample requires --checkpoint")
    device = runtime(cfg)
    state = torch.load(cfg.checkpoint, map_location="cpu", weights_only=True)
    if state.get("version") != 1:
        raise ValueError("Unsupported checkpoint version")
    trained = Config(**state["config"])
    trained.validate()
    if not trained.min_length <= cfg.length <= trained.max_length:
        raise ValueError("Requested length is outside the checkpoint's configured range")
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
            record = CATokenizer.encode(
                x * state["coordinate_scale"], f"sample_{i:04d}", check_bonds=False
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
    evaluate(records, out)


def main(config: Config):
    """Run this entry with a fully resolved, immutable experiment configuration."""
    config.validate()
    if config.command == "prepare":
        return prepare(config)
    if config.command == "train":
        return train(config)
    if config.command == "sample":
        return sample(config)
    if not config.samples:
        raise ValueError("eval requires --samples (the samples.pt file)")
    records = torch.load(config.samples, map_location="cpu", weights_only=True)["samples"]
    return evaluate(records, config.out_dir)


if __name__ == "__main__":
    main(parse_config(Config))
