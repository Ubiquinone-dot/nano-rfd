"""Assignment: build a CA diffusion model from scratch.

Fill TODOs: AtomWorks extraction -> collation/augmentation -> RoPE/model -> EDM/sampling.
The config owns the seeded RNG. Cached records and exported structures use angstroms;
collation converts to nanometers for all model/diffusion arithmetic.
CLI, training loop, EMA, checkpoint IO, and shared geometry evaluation are supplied.
See entries/baseline/main.py for the worked solution. No validation loss is computed.

Run: python entries/template/main.py --help
"""

import copy
import json
import math
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
from atomworks.ml.transforms.base import Compose, Transform
from biotite.structure import AtomArray
from torch import nn

from nanorfd.base import ExperimentConfig, parse_config
from nanorfd.checkpoints import load_checkpoint, restore_checkpoint, save_checkpoint
from nanorfd.evals import evaluate

##############################################
# Experiment configuration
##############################################


@dataclass(frozen=True, kw_only=True)
class Config(ExperimentConfig):
    width: int = 64
    depth: int = 2
    heads: int = 4
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


##############################################
# AtomWorks tokenizer, collation, and augmentation
##############################################


class SelectCA(Transform):
    def forward(self, data):
        """TODO: Select amino-acid carbon CA atoms from data["atom_array"].
        Use data["chain"] when supplied, otherwise the first protein chain.
        Return the data dict with the selected atom_array and chain name.
        """
        raise NotImplementedError("TODO: SelectCA.forward")


class ExtractCATensor(Transform):
    requires_previous_transforms = ("SelectCA",)

    def forward(self, data):
        """TODO: Convert selected CA atoms to a sample with id, coords, and residue_id.
        coords is float32 [L, 3] in angstroms; residue_id is int64 [L].
        Use CATokenizer.encode() and return {**data, "sample": sample}.
        """
        raise NotImplementedError("TODO: ExtractCATensor.forward")


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
    def collate(records, device):
        """TODO: Pad the selected records and build a boolean mask.
        Records are already tokenized with coords [L, 3] in angstroms.
        Return x [B, Lmax, 3] in nanometers (divide by 10) and mask [B, Lmax]
        on device. Do not sample records, center, or rotate here.
        """
        raise NotImplementedError("TODO: CATokenizer.collate")

    @staticmethod
    def center(x, mask):
        """TODO: Center x [B, L, 3] over mask [B, L] and zero padding.
        Padded coordinates must not affect the mean.
        """
        raise NotImplementedError("TODO: CATokenizer.center")

    @classmethod
    def augment(cls, x, mask, cfg):
        """TODO: Center and apply a proper random rotation per protein.
        Draw rotations using cfg.rng. Keep determinant +1 to preserve
        handedness, and return coordinates in the same units (nm).
        """
        raise NotImplementedError("TODO: CATokenizer.augment")

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


##############################################
# Dataset preparation
##############################################


def prepare(cfg):
    path = Path(cfg.manifest).resolve()
    manifest = json.loads(path.read_text())
    data = {"version": 1}
    tokenizer = CATokenizer()
    for split, entries in manifest.items():
        data[split] = []
        for entry in entries:
            structure_path = path.parent / entry["path"]
            record = tokenizer.from_file(structure_path, entry.get("chain"))
            data[split].append(record)
    output = Path(cfg.data)
    if output.exists():
        raise ValueError(f"Refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, output)
    print(f"Saved {len(data['train'])} training structures to {output}")


##############################################
# Model: RoPE attention and denoiser
##############################################


class Attention(nn.Module):
    def __init__(self, cfg):
        """TODO: Create QKV/output projections and register RoPE frequencies.
        Use cfg.width and cfg.heads; each head dimension must be even.
        """
        super().__init__()
        raise NotImplementedError("TODO: Attention.__init__")

    def rope(self, x):
        """TODO: Apply rotary embeddings to x [B, H, L, D] using residue positions.
        Rotate consecutive even/odd channels; preserve the input shape.
        """
        raise NotImplementedError("TODO: Attention.rope")

    def forward(self, h, mask):
        """TODO: Project h [B, L, width] to multihead Q/K/V, apply RoPE to Q/K.
        Use scaled_dot_product_attention with mask [B, L] on keys.
        No causal mask or pair bias. Project back to [B, L, width].
        """
        raise NotImplementedError("TODO: Attention.forward")


class Block(nn.Module):
    def __init__(self, cfg):
        """TODO: Create pre-attention and pre-MLP layer norms, Attention(cfg),
        and a width -> mlp_ratio * width -> width MLP with GELU.
        """
        super().__init__()
        raise NotImplementedError("TODO: Block.__init__")

    def forward(self, h, mask):
        """TODO: Apply pre-norm attention and MLP, each with a residual connection.
        Zero padded positions in the returned [B, L, width] tensor.
        """
        raise NotImplementedError("TODO: Block.forward")


class Denoiser(nn.Module):
    def __init__(self, cfg):
        """TODO: Embed xyz, log-noise, and normalized chain length into cfg.width.
        Create cfg.depth Transformer blocks and a width -> 3 output head.
        Use sinusoidal noise features and zero-initialize the final projection.
        """
        super().__init__()
        raise NotImplementedError("TODO: Denoiser.__init__")

    def forward(self, x, noise_level, mask):
        """TODO: Combine coordinate/noise/length embeddings and apply the blocks.
        x is [B, L, 3], noise_level is [B], mask is bool [B, L].
        Return float32 [B, L, 3] with zero padding.
        """
        raise NotImplementedError("TODO: Denoiser.forward")


# EDM with sigma_data = 1 nm. All geometric arithmetic is float32.
##############################################
# Diffusion objective and Euler sampler
##############################################


def denoise(model, x, sigma, mask):
    """TODO: Apply EDM preconditioning with sigma_data=1 and sigma [B].
    c_in=1/sqrt(1+sigma^2), c_skip=1/(1+sigma^2),
    c_out=sigma*c_in. Feed x*c_in and log(sigma)/4 to the model.
    Return the centered c_skip*x + c_out*model_output.
    """
    raise NotImplementedError("TODO: denoise")


def diffusion_loss(model, x, mask, cfg):
    # A dedicated CPU generator owns all stochastic training inputs, including on CUDA.
    # Saving this state makes batch/noise resume exact.
    """TODO: Sample log-normal sigma and centered Gaussian noise using cfg.rng.
    Denoise x + sigma*noise. Compute masked xyz MSE per protein,
    weight by (1+sigma^2)/sigma^2, then average over proteins.
    Return one differentiable scalar; padding contributes no loss.
    """
    raise NotImplementedError("TODO: diffusion_loss")


@torch.no_grad()
def sample_coordinates(model, lengths, cfg, device):
    """TODO: Draw centered Gaussian noise using cfg.rng, scaled by cfg.sigma_max.
    Use cfg.sample_steps noise levels spaced by cfg.rho down to sigma_min,
    then append zero. Euler update: x += (next_sigma-sigma)*(x-denoise)/sigma.
    Recenter each step; never call the denoiser at sigma=0.
    Return a list of unpadded CPU tensors [length, 3] in nanometers.
    """
    raise NotImplementedError("TODO: sample_coordinates")


##############################################
# Runtime and training loop
##############################################


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
    data = torch.load(cfg.data, map_location="cpu", weights_only=True)
    model = Denoiser(cfg).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    step = 0
    out = Path(cfg.out_dir)
    if cfg.resume:
        step = restore_checkpoint(cfg.resume, model, ema, optimizer, cfg)
    elif (out / "config.json").exists() or (out / "checkpoint.pt").exists():
        raise ValueError(f"Run already exists: {out}; use --resume or a new --out-dir")
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2) + "\n")
    print(
        f"Device: {device} | Parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True
    )
    tokenizer = CATokenizer()
    while step < cfg.steps:
        model.train()
        indices = torch.randint(len(data["train"]), (cfg.batch_size,), generator=cfg.rng)
        records = [data["train"][i] for i in indices.tolist()]
        x, mask = tokenizer.collate(records, device)
        x = tokenizer.augment(x, mask, cfg)
        optimizer.zero_grad(set_to_none=True)
        lr = learning_rate(step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr
        with autocast(cfg, device):
            loss = diffusion_loss(model, x, mask, cfg)
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
                save_checkpoint(out / "checkpoint.pt", model, ema, optimizer, step, cfg)
            with (out / "metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(row, allow_nan=False) + "\n")
            print(json.dumps(row), flush=True)
        if timed_out:
            break
    print(f"Checkpoint: {out / 'checkpoint.pt'}", flush=True)


##############################################
# Generation and shared evaluation
##############################################


def sample(cfg):
    device = runtime(cfg)
    state = load_checkpoint(cfg.checkpoint)
    trained = Config(**state["config"])
    model = Denoiser(trained).to(device).eval()
    model.load_state_dict(state["ema"])
    # Outputs are in nm; the tokenizer/export boundary uses angstroms.
    trained = replace(trained, seed=cfg.seed, sample_steps=cfg.sample_steps, rho=cfg.rho)
    out = Path(cfg.out_dir)
    if (out / "samples.pt").exists():
        raise ValueError(f"Refusing to overwrite samples in {out}")
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for start in range(0, cfg.num_samples, cfg.batch_size):
        count = min(cfg.batch_size, cfg.num_samples - start)
        with autocast(cfg, device):
            coords = sample_coordinates(model, [cfg.length] * count, trained, device)
        for i, x in enumerate(coords, start):
            record = CATokenizer.encode(x * 10.0, f"sample_{i:04d}")
            records.append(record)
    torch.save(
        {
            "version": 1,
            "samples": records,
            "seed": cfg.seed,
            "sample_steps": cfg.sample_steps,
            "rho": cfg.rho,
            "batch_size": cfg.batch_size,
        },
        out / "samples.pt",
    )
    for record in records:
        CATokenizer.to_pdb(record["coords"], out / f"{record['id']}.pdb")
    evaluate({r["id"]: CATokenizer.decode(r["coords"]) for r in records}, out)


##############################################
# Entry point
##############################################


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
