# nanorfd

A small Cα protein diffusion model in one Python script. The model is a bidirectional
Transformer with RoPE, ordinary padded attention, and an MLP in each block. It predicts
clean coordinates using an EDM objective. There are no pair features or varlen kernels.

`nanorfd.py` contains `ExperimentConfig`, an AtomWorks tokenizer, data batching, the
model, diffusion, training, sampling, and basic geometry evaluation. The default model
has six blocks of width 256. Its generator starts from random weights.

**Status:** working implementation with CPU correctness tests and a tiny synthetic
smoke test. Protein sample quality and the one-GPU/hour target have not been benchmarked.
No protein dataset or trained checkpoint is bundled.

## Setup

```bash
uv sync --python 3.12
uv run python nanorfd.py --help
```

Dependencies are PyTorch and AtomWorks, with NumPy/Biotite supplied by AtomWorks.
The lockfile records the tested versions. `auto` selects CUDA if available, otherwise
CPU; `--device mps` is explicit. CUDA uses bfloat16 autocast by default; use
`--no-bf16` for float32. CPU and MPS run in float32. CUDA/MPS have not been tested here.

## Prepare coordinates with AtomWorks

Make a JSON manifest of local structures with explicit train/validation splits:

```json
{
  "train": [
    {"path": "structures/domain_a.cif", "chain": "A"},
    {"path": "structures/domain_b.pdb", "chain": "A"}
  ],
  "val": [
    {"path": "structures/domain_c.cif", "chain": "B"}
  ]
}
```

Paths resolve relative to the manifest. Choose related-domain/family splits upstream;
the script checks literal duplicates but does not perform homology clustering. Both
splits must be nonempty. Each entry must be a complete, continuous protein chain within
the selected length range. For the intended short-domain benchmark, use 50–128 residues.

```bash
uv run python nanorfd.py prepare --manifest manifest.json --data data/ca.pt --min-length 50
```

The tokenizer uses the actual AtomWorks backend:

```text
AtomWorks parse (minimal preset, first model, asymmetric unit)
    -> AtomArray
    -> Compose([SelectCA(), ExtractCATensor()])
    -> {id, coords: float32[L, 3], residue_id: int64[L]}
```

`SelectCA` selects amino-acid residues and one chain, checks that every residue has
exactly one carbon Cα, and checks residue order and occupancy. `ExtractCATensor`
extracts Å coordinates and rejects nonfinite values or adjacent Cα distances outside
2.5–4.5 Å. The pipeline does not fill missing residues, silently crop, or skip invalid
examples. Missing atoms outside the Cα trace do not affect the model. AtomWorks handles
PDB/mmCIF parsing and alternate locations (`first`); chain names refer to the loaded
array's `chain_id`. The conservative numbering check can reject structures with unusual
author numbering; curate those explicitly upstream.

The same pipeline accepts an already-loaded array:

```python
from nanorfd import CATokenizer

tokenizer = CATokenizer()
sample = tokenizer(atom_array, sample_id="domain_a", chain="A")
sample = tokenizer.from_file("domain_a.cif", chain="A")
```

The prepared `.pt` cache contains only tensors and plain metadata, including source
hashes. Training reads it once, so structure parsing never runs inside an optimizer step.
The cache must fit in CPU RAM. No CCD/PDB mirror is needed for the tested standard-residue
fixtures; AtomWorks may print notices when those optional mirror variables are unset.

## Train

```bash
# Small debugging run, using your prepared data.
uv run python nanorfd.py train --config configs/debug.json --data data/ca.pt --out-dir runs/debug

# Initial reference settings; the runtime/quality tradeoff still needs profiling.
uv run python nanorfd.py train --data data/ca.pt --min-length 50 --out-dir runs/baseline
```

All settings live in one flat `ExperimentConfig`; a JSON file overrides defaults, and
command-line options override JSON. Unknown fields and invalid dimensions fail early.
Use `--steps`, `--batch-size`, `--width`, `--depth`, and `--heads` to shrink experiments.
`--max-minutes 50` is the default soft training limit (checked between steps); final
validation/checkpoint writing can extend it. `--max-minutes 0` disables it.

Training centers each structure and divides by one RMS coordinate scale computed only
from the training set. Proper random rotations preserve handedness. Clean coordinates,
noise, and predictions stay centered; padding is excluded from attention keys and loss.
RoPE encodes residue order, not physical 3D rotation equivariance.

The diffusion section uses `x_sigma = x_clean + sigma * noise`, EDM preconditioning
with normalized `sigma_data=1`, and a weighted clean-coordinate MSE, averaged equally
over proteins. Noise defaults are starting settings, not protein-calibrated optima.
AdamW, warmup/cosine learning rate, gradient clipping, and EMA are explicit in the loop.

Each run writes:

- `config.json`: fully resolved configuration.
- `metadata.json`: source/data hashes, coordinate scale, parameter count, device, and versions.
- `metrics.jsonl`: training loss and fixed-seed EMA validation loss.
- `checkpoint.pt`: model, EMA, optimizer, step, configuration, scale, and training RNG state.

Resume an interrupted run using its saved settings:

```bash
uv run python nanorfd.py train --config runs/baseline/config.json --resume runs/baseline/checkpoint.pt
```

The data hash and training settings must match. Resume continues the original total
step schedule; it does not add `steps` more steps. A dedicated CPU generator controls
batch choice, augmentation, and noise, independently of validation. Exact CPU resume
is tested; bitwise identity across devices or PyTorch versions is not promised.
The time limit applies to each invocation. Existing runs require explicit resume.

## Sample and evaluate

```bash
uv run python nanorfd.py sample --checkpoint runs/baseline/checkpoint.pt --out-dir samples/baseline --length 64 --num-samples 16
uv run python nanorfd.py eval --samples samples/baseline/samples.pt --out-dir samples/baseline
```

Sampling loads EMA weights, architecture, normalization, and diffusion bounds from the
checkpoint. `--sample-steps` (default 32) controls a simple Euler sampler; `--seed` and
`--batch-size` control reproducibility. Use the same batch size when comparing samples.
It writes raw Cα traces as PDB files, a `samples.pt` tensor cache, `geometry.csv`, and
`geometry.json`. GLY names in PDB output are placeholders, not designed sequences.

The geometry report includes adjacent Cα spacing/error, nonlocal Cα overlap fraction
(distance <3 Å, sequence separation >2), radius of gyration, and contact fraction
(distance <8 Å, sequence separation >8). Fractions count unordered pairs. A zero contact
pair count means the contact fraction is not informative. Signed normalized triple
products provide a crude handedness diagnostic; degenerate triplets are counted.
Summary numbers weight samples equally. NaN/Inf samples are errors, never dropped.

These are coarse diagnostics. There is no folding-based designability evaluation,
structural novelty search, or all-atom clash score yet. A low denoising loss or plausible
Cα bond lengths alone do not establish useful protein generation.

## Tests

```bash
uv run pytest -q
uv run ruff check nanorfd.py tests
```

Tests exercise real AtomWorks PDB/mmCIF loading, transform validation, chain selection,
rotation augmentation, RoPE, padding isolation, sampler endpoints, finite gradients,
exact CPU checkpoint resume, and the training/sampling/evaluation workflow. Synthetic
helices in tests are correctness fixtures, not a protein-generation benchmark.

The broader research/teaching proposal and Foundry audit are in [PLAN.md](PLAN.md).
