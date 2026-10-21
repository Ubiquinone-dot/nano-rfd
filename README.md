# nanorfd

A small Cα protein diffusion model with one Python script per solution. The baseline
uses a bidirectional Transformer with RoPE, ordinary padded attention, and an EDM
clean-coordinate objective. There are no pair features or varlen kernels.

The implementation has CPU correctness tests and a tiny synthetic smoke test.
Protein sample quality and the one-GPU/hour target have not been benchmarked.
No dataset or trained checkpoint is bundled.

## Layout

```text
src/nanorfd/
  base.py                 shared ExperimentConfig and CLI parsing
  evals.py                AtomArray geometry metrics and CSV/JSON reports
  checkpoints.py          save/load/restore helpers
entries/baseline/
  main.py                 worked solution
entries/template/
  main.py                 assignment scaffold with TODO functions
entries/baseline_jax/
  main.py                 functional JAX/Optax solution
configs/debug.json        small baseline settings
```

Each entry folder contains only `main.py`. All entries extend the frozen
`ExperimentConfig` and expose `main(config)`. The base config supplies data/run paths,
seed, device/precision, length settings, training budgets, logging/checkpoint cadence,
and sampling settings. Entries add model, optimizer, and diffusion parameters.

```python
from nanorfd import ExperimentConfig
from entries.baseline.main import Config, main

experiment = ExperimentConfig(data="data/ca.pt", out_dir="runs/baseline",
                              seed=42, steps=10000, max_minutes=50)
main(Config.from_experiment(experiment, width=256, depth=6, heads=8))
```

`from_experiment()` passes shared fields directly to the entry dataclass; duplicate
shared overrides and unknown fields fail through Python. There are no `validate()`
methods or custom config type/range checks. JSON overrides defaults; CLI flags override
JSON. Use `dataclasses.replace()` when deliberately constructing a changed config.

## Setup

```bash
uv sync --python 3.12
uv run python entries/baseline/main.py --help
```

Dependencies are PyTorch and AtomWorks, with NumPy/Biotite supplied by AtomWorks.
`uv sync` installs `nanorfd` in editable mode, so entry scripts import the shared package
directly. `auto` selects CUDA if available, otherwise CPU. CUDA uses bfloat16 autocast;
`--no-bf16` selects float32. CPU and explicit `--device mps` use float32.

## Functional JAX baseline

`entries/baseline_jax/main.py` implements the same architecture, EDM objective, EMA,
and Euler sampler using JAX and Optax. Model layers are plain functions: `init_model`
returns a parameter tree, `model` and `denoise` apply it, and `train_step` returns a new
state. There are no model module classes or hidden parameter/RNG stores. State contains
parameters, EMA parameters, Adam state, step, and an explicit PRNG key.

```bash
uv sync --extra jax --python 3.12
uv run --extra jax python entries/baseline_jax/main.py train --config configs/debug.json --data data/ca.pt --out-dir runs/jax
uv run --extra jax python entries/baseline_jax/main.py sample --checkpoint runs/jax/checkpoint.npz --out-dir samples/jax --length 64
uv run --extra jax python entries/baseline_jax/main.py eval --samples samples/jax/samples.pt --out-dir samples/jax
```

The same `prepare` command and AtomWorks transform pipeline work in this entry. Prepared
`.pt` datasets and generated `samples.pt` files are interchangeable between entries.
PyTorch is used only at the CPU tokenizer/file boundary; the model, gradients,
augmentation, and optimizer updates use JAX. Geometry evaluation is the shared module.

The training update is JIT-compiled, while the sampler uses `jax.lax.scan`. Batches are
padded to `max_length` to keep their compiled shape fixed; curate data within that length.
Only the current batch is transferred to the accelerator. Compilation counts toward the
training time budget. CPU uses float32; GPU uses bfloat16 model computations by default
with float32 parameters, optimizer state, and diffusion arithmetic. JAX manages CPU
threading itself; the shared `threads` setting is not applied to XLA.

The `jax` extra installs the CPU-capable packages. For NVIDIA GPU training, install the
matching JAX accelerator packages using the [official JAX installation guide](https://docs.jax.dev/en/latest/installation.html),
then run `.venv/bin/python entries/baseline_jax/main.py ... --device cuda` in that
environment. `--device cpu`, `--device gpu`, and `--device auto` are also supported.
This entry has been tested on CPU; GPU throughput and the one-hour target remain unmeasured.

JAX checkpoints are `checkpoint.npz`, containing numeric tree leaves and JSON metadata.
Resume with the same dataset, configuration, and library versions:

```bash
uv run --extra jax python entries/baseline_jax/main.py train --config runs/jax/config.json --resume runs/jax/checkpoint.npz
```

Model checkpoints are specific to their backend. With matching weights, the denoiser
forward pass and gradients agree with PyTorch within numerical tolerance. The two backends
use different random generators and need not follow identical training trajectories.
Exact interrupted/resumed CPU training is tested independently for both.

## Assignment

Copy `entries/template/main.py` to your own entry folder and fill in its TODOs:

1. **Tokenizer:** select protein CA atoms and extract coordinate tensors through AtomWorks transforms.
2. **Batches:** center coordinates, compute a shared scale, pad/mask, and apply proper random rotations.
3. **Model:** implement RoPE, padded attention, Transformer blocks, and the noise-conditioned denoiser.
4. **Diffusion:** implement EDM preconditioning and the masked training loss.
5. **Sampling:** implement an Euler sampler ending at zero noise.

Each stub documents its inputs, outputs, and mathematical contract, and raises
`NotImplementedError` until completed. Add imports as needed. The scaffold supplies
config/CLI, structure loading, cache/export IO, the AdamW training loop, EMA, checkpoint
resume, and shared geometry evaluation. `entries/baseline/main.py` is the worked solution.
Start with small settings and use the existing correctness tests as reference checks.

```bash
uv run python entries/template/main.py --help
# After completing the TODOs:
uv run python entries/template/main.py train --config configs/debug.json --data data/ca.pt --out-dir runs/my-entry
```

## Prepare coordinates

Create a manifest of local structures. Only `train` is required; additional lists can
be cached for later use, but training does not compute validation loss.

```json
{
  "train": [
    {"path": "structures/domain_a.cif", "chain": "A"},
    {"path": "structures/domain_b.pdb"}
  ]
}
```

Paths resolve relative to the manifest. Curate complete protein chains upstream; the
baseline does not check residue numbering, occupancy, bond lengths, duplicate structures,
or homology relationships. Length settings describe the experiment; there is no automatic
cropping or length filtering. Short domains of 50–128 residues are the intended starting data.

```bash
uv run python entries/baseline/main.py prepare --manifest manifest.json --data data/ca.pt
```

The tokenizer uses AtomWorks to parse the first model of the asymmetric unit, then:

```text
AtomArray -> Compose([SelectCA(), ExtractCATensor()])
          -> {id, coords: float32[L, 3], residue_id: int64[L]}
```

`SelectCA` selects amino-acid carbon Cα atoms, using the requested chain or falling back
to the first protein chain in the array. AtomWorks handles parsing and alternate locations.
The transforms operate directly on the loaded array; there is no custom retry or repair
layer. Missing required data fails in the normal array/transform operations. Nonfinite
coordinates raise an error.

```python
from entries.baseline.main import CATokenizer

tokenizer = CATokenizer()
sample = tokenizer(atom_array, sample_id="domain_a", chain="A")
sample = tokenizer.from_file("domain_a.cif")
```

The `.pt` cache holds tensors and plain metadata, including source hashes. Training loads
it once into CPU RAM; structure parsing does not run inside an optimizer step.

## Train and resume

```bash
uv run python entries/baseline/main.py train --config configs/debug.json --data data/ca.pt --out-dir runs/debug
uv run python entries/baseline/main.py train --data data/ca.pt --out-dir runs/baseline
```

The default model has six blocks of width 256. Training centers each structure and divides
by a single training-set RMS coordinate scale. Proper random rotations preserve handedness.
Padding is masked in attention and loss. RoPE describes residue order, not 3D equivariance.

AdamW, warmup/cosine learning rate, gradient clipping, and EMA are explicit in the loop.
`--max-minutes 50` is the default soft time limit, checked between steps; zero disables it.
`--save-every` controls checkpoints. There is no validation-loss pass or validation dataset
requirement. NaN/Inf losses fail, and existing run outputs require explicit resume.

Runs write resolved `config.json`, provenance `metadata.json`, training `metrics.jsonl`,
and `checkpoint.pt` with model, EMA, optimizer, step, config, normalization, and RNG state.

```bash
uv run python entries/baseline/main.py train --config runs/baseline/config.json --resume runs/baseline/checkpoint.pt
```

Resume restores state with one shared helper call. Use the same dataset and training
settings to continue the original total-step schedule. There is no custom config/hash
compatibility validator; incompatible state-dict shapes fail in PyTorch. A dedicated CPU
RNG controls batch choice, rotations, and noise. Exact CPU resume with unchanged settings
is tested. The time limit applies separately to each invocation.

The former `eval_every` option is now `save_every`; `eval_batches` and `eval_batch_size`
have been removed. Update older saved configs to these fields before reusing them.

## Sample and evaluate

```bash
uv run python entries/baseline/main.py sample --checkpoint runs/baseline/checkpoint.pt --out-dir samples/baseline --length 64 --num-samples 16
uv run python entries/baseline/main.py eval --samples samples/baseline/samples.pt --out-dir samples/baseline
```

Sampling loads EMA weights and model settings from the checkpoint. `--sample-steps`
controls the Euler sampler. Keep seed and batch size fixed for reproducibility. Outputs
include CA-only PDBs, `samples.pt`, and geometry reports. GLY labels are placeholders.

```python
from nanorfd.evals import geometry_metrics, evaluate

metrics = geometry_metrics(atom_array)
summary = evaluate({"sample_0000": atom_array}, out_dir="reports/baseline")
```

Supply single-chain AtomArrays in residue order with coordinates in Å and at least four
Cα atoms. Full-atom arrays are accepted; metrics select protein carbon Cα atoms. Input
curation is the caller's responsibility. Generated tensors become arrays through
`CATokenizer.decode()`. All entries use the same metrics, cutoffs, and report aggregation.

Reports include Cα spacing/error, overlap fraction (distance <3 Å, sequence separation >2),
contact fraction (distance <8 Å, separation >8), radius of gyration, signed normalized
triple products, and degenerate-triplet counts. Pair fractions count unordered pairs;
summary means weight samples equally. Nonfinite values raise instead of being dropped.
These are coarse geometry diagnostics; folding-based designability is not implemented.

## Tests

```bash
uv run --extra jax pytest -q
uv run ruff check src entries tests
```

Tests cover AtomWorks loading, chain selection, rotations, RoPE, padding isolation,
finite gradients, sampler endpoints, exact CPU resume, and shared geometry reports.
The optional JAX tests also check forward/gradient parity with PyTorch; they skip when
JAX/Optax are not installed.
Synthetic helices test correctness rather than protein-generation quality.
The broader research proposal and Foundry audit are in [PLAN.md](PLAN.md).
