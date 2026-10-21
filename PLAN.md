# nanorfd: a small protein diffusion model you can understand and improve

Design and implementation notes, 2026-10-21. The repository started empty. The single-script baseline is now implemented in `entries/baseline/main.py`, with a solution `Config` extending the shared `src/nanorfd/base.py` `ExperimentConfig`, AtomWorks tokenizer transforms, padded attention with RoPE, EDM training, sampling, and basic geometry evaluation. CPU correctness tests cover real PDB/mmCIF loading and exact resume. No protein-quality or one-hour GPU benchmark has been run. The user selected Foundry's `long-context` branch as the structural reference and **one GPU, under an hour** as the target for the first useful training run. Sections below also describe planned research work beyond the current baseline; see README.md for what is runnable now.

## Recommendation

Build one complete, readable reference implementation, then make improving it under a fixed compute budget the challenge.

The project promise should be: **Train a Cα protein diffusion model from scratch in under an hour, understand every component, and improve the quality of its samples.** Treat the runtime and sample-quality claim as a target until a measured reference run supports it. Define “from scratch” as randomly initialized generator weights and an explicit PyTorch implementation; pretrained models may be used in the separate evaluation pipeline.

The assignment has two stages:

1. **Make it work:** implement corruption, masked denoising loss, and sampling; pass correctness checks; reproduce the reference run and explain its failures.
2. **Make it better:** improve generated geometry, distribution coverage, and ultimately designability under the same data and compute limits; submit an ablation and full metrics.

Keep the working solution on the main branch. Supply exercises as a guide identifying a few functions to implement, with an optional teaching tag later. Avoid maintaining two independently evolving implementations.

## Foundry audit: what to borrow

A subagent inspected the requested `long-context` branch at commit `634efb137b314eb4bc41e42e31f7c306e0f853f4`. This branch is a standalone `brm-fused` package, with `src/brm_fused/{tokenizer,data,network,model,runtime,metrics}`, root configuration files, tests, and train/validate/predict entry points. It differs materially from older Foundry monorepo branches.

| Audited component | Useful convention | nanorfd adaptation |
| --- | --- | --- |
| [ExperimentConfig](https://github.com/baker-laboratory/foundry-dev/blob/634efb137b314eb4bc41e42e31f7c306e0f853f4/src/brm_fused/train.py#L72) | Strict declared top-level fields; central construction and serialization | Keep one typed configuration with small nested settings; remove Hydra/Lightning orchestration |
| [Resolved config loading](https://github.com/baker-laboratory/foundry-dev/blob/634efb137b314eb4bc41e42e31f7c306e0f853f4/src/brm_fused/train.py#L176) | Self-contained saved configuration and optional hash checks | Save the exact resolved run settings; validate compatibility on resume |
| [Tokenizer](https://github.com/baker-laboratory/foundry-dev/blob/634efb137b314eb4bc41e42e31f7c306e0f853f4/src/brm_fused/tokenizer/tokenizer.py#L239) | Several input routes produce one representation | File/array inputs share one Cα schema; a length-only generation specification contains no hidden native features |
| [TokenizerSpec](https://github.com/baker-laboratory/foundry-dev/blob/634efb137b314eb4bc41e42e31f7c306e0f853f4/src/brm_fused/tokenizer/tokenizer.py#L62) | Explicit transformation settings and training/validation behavior | Keep deterministic parsing/filter settings separate from stochastic rotation and diffusion |
| [TokenizedStructure](https://github.com/baker-laboratory/foundry-dev/blob/634efb137b314eb4bc41e42e31f7c306e0f853f4/src/brm_fused/tokenizer/context/structure.py#L297) | Explicit tensor shapes, dtypes, and padding semantics | Small sample/batch types, documented masks, units, and provenance |

Two architecture details matter. The current [windowed coordinate autoencoder](https://github.com/baker-laboratory/foundry-dev/blob/634efb137b314eb4bc41e42e31f7c306e0f853f4/src/brm_fused/network/autoencoder.py#L1) explicitly has no equivariance. The [diffusion helpers](https://github.com/baker-laboratory/foundry-dev/blob/634efb137b314eb4bc41e42e31f7c306e0f853f4/src/brm_fused/model/fm.py#L1) implement atom37 flow matching with clean-coordinate prediction and preconditioning adapted to a linear interpolant. The proposed Gaussian EDM baseline below deliberately borrows the interface, not that exact objective; mixing the two schedules/losses would be a bug.

Do not port the atom37 autoencoder, MSA/sequence/context machinery, online packing, chemistry/confidence heads, custom kernels, or cluster launch stack. In particular, tokenizer-injected flow matchers would obscure the few lines the learner needs to understand. Keep corruption, objective, and sampling together in one clearly marked section of `entries/baseline/main.py`.

## Initial task and data

- Unconditional generation given a requested length: one protein domain, one Cα per residue, 50–128 residues. “Unconditional” excludes sequence, fold, motif, and target conditioning; length is supplied.
- Train on complete, contiguous short domains. Random fragments of large proteins are useful for debugging, but should not define a benchmark claiming to generate complete domains.
- Candidate starting corpus: a pinned, cleaned subset of SCOPe domains, following the kind of data preparation used by [Genie](https://github.com/aqlaboratory/genie). The first implementation milestone must establish the available counts and actual download/preparation cost. Do not invent a required dataset size before inspecting the corpus.
- Publish an immutable manifest with source/version, structure and chain/domain identifiers, residue ranges, cluster membership, split, and checksums. Split related domains together, preferably by supplied structural superfamily; explain exactly what the split does and does not exclude. Never split crops from the same parent across train and validation.
- Choose train/validation/test splits before selecting examples or fitting normalization. Reserve the test split for the final report.
- Validate finite coordinates, residue ordering, alternate-location handling, missing Cα atoms, and chain continuity. Log every exclusion with a reason. Raise on NaNs instead of silently filtering them. Do not concatenate across gaps and pretend the endpoints are bonded.
- Cache coordinates and metadata once. Parsing PDB/mmCIF files belongs in preparation, not the training loop. Publish a tiny debug fixture and a reproducible preparation script; distribution of a prepared corpus depends on its source terms.
- Store sequence for provenance/clustering if available, but do not expose it as model conditioning in this benchmark.

## Smallest model worth trying

Use a small, bidirectional coordinate transformer with Gaussian diffusion and clean-coordinate prediction. This is an intentionally simple hypothesis to benchmark, not a claim that it is the most sample-efficient protein architecture.

| Component | Starting choice | Reason |
| --- | --- | --- |
| Representation | Centered Cα coordinates, `[B, L, 3]`, plus a valid-residue mask | Direct, interpretable, easy to export |
| Features | Noisy coordinates, residue position, requested length, noise embedding | No sequence encoder or pretrained features |
| Denoiser | 6 blocks, width 256, 8 attention heads, MLP ratio 4 | Roughly 5M parameters before small embedding/head overhead; confirm from implementation |
| Attention | Ordinary padded bidirectional attention with RoPE on Q/K; no pair bias, varlen, or packing | Short chains keep the basic implementation practical |
| Objective | EDM-style preconditioned clean-coordinate prediction and weighted MSE | One denoising interface supports training and sampling |
| Symmetries | Masked centering and random proper 3D rotations during training | Simple baseline; rotational equivariance is learned rather than guaranteed |
| Optimizer | AdamW, short warmup, gradient clipping, EMA weights | Small, explicit training loop |
| Precision | CUDA bfloat16 autocast; CPU/MPS and geometry/loss reductions in float32 | Speed without needlessly fragile coordinate arithmetic |
| Sampling | A fixed 32-step schedule initially; Euler as the first implementation | Easy to trace; Heun is a later comparison at matched function evaluations |

The diffusion design is informed by [EDM](https://arxiv.org/abs/2206.00364), but protein noise scales must be calibrated to this dataset rather than copied blindly from image experiments.

Important implementation details:

- Compute one global coordinate scale from training data and save it. Do not normalize each structure to unit radius, which would erase physical size variation. Convert back to Å for exported structures and metrics.
- For valid residues, subtract the centroid from both clean coordinates and sampled Gaussian noise. Keep padded entries zero. Use the same centered subspace during sampling.
- Corrupt with `x_sigma = x_0 + sigma * epsilon`. Define the EDM preconditioning, noise distribution, and loss weights in one place, with matching sampler conventions.
- Apply the same proper rotation to all coordinates in an example. Do not augment by reflections, since handedness matters.
- Normalize masked loss per example before averaging across the batch, so length does not silently set example weight. Report denoising error by noise bin as well as an aggregate.
- Compare predicted and clean coordinates in the same augmented frame; do not hide an alignment operation inside the training objective.
- Keep the first loss free of geometric penalty terms. Add bond or local-distance losses only as measured ablations, with clear weights and noise dependence.
- Avoid geometric repair in the baseline sampling path. If a later method adds repair, report both raw and repaired results and include repair in inference cost.

The reference architecture is deliberately ordinary attention with RoPE and an MLP in each residual block. Start with pre-LayerNorm, QKV projections, RoPE on Q/K using residue positions, PyTorch scaled dot-product attention, an output projection, and a two-layer GELU MLP. Add projected noisy coordinates and a broadcast noise/length embedding before the blocks, then project final hidden states to three coordinates. No learned pair state, pair bias, distance features, geometric attention, or adaptive-normalization machinery is required for v0. RoPE encodes sequence position; it does not provide equivariance to physical 3D rotation, so proper-rotation augmentation remains necessary.

Use padded `[B, H, L, head_dim]` Q/K/V tensors and a broadcast key-validity mask. Zero padded outputs after blocks and mask coordinate centering/loss. Every example must contain at least one valid residue. Start with `is_causal=False` and no attention dropout. This is the ordinary PyTorch attention API, not a custom varlen implementation; PyTorch may select an optimized backend when supported. A geometry-aware denoiser is an optional later research comparison, not part of the initial implementation.

## Repository shape and interfaces

Each solution lives in its own entry folder containing only `main.py`. The baseline is `entries/baseline/main.py`, with the tokenizer, model, and training workflow together. Shared `ExperimentConfig` and CLI parsing live in the installable `src/nanorfd/base.py`; each entry extends it with its own frozen `Config` dataclass and exposes `main(config)`.

```text
README.md                 quick start, model explanation, measured reference result
PLAN.md                   this design and unresolved decisions
pyproject.toml            small dependency set and optional evaluation extras
src/nanorfd/base.py        shared ExperimentConfig and CLI parsing
src/nanorfd/__init__.py    public ExperimentConfig export
entries/baseline/main.py   entry Config, AtomWorks tokenizer, prepare/train/sample/eval
configs/debug.json        small model and short training preset
evaluation/designability.py  planned later inverse-fold/refold adapter
tests/                    scientific correctness and resume tests
```

Implemented entry points are `python entries/baseline/main.py prepare`, `train`, `sample`, and `eval`. The preparation command reads an explicit train/validation manifest of local structure files. AtomWorks loads the first model of the asymmetric unit with its minimal parser preset. `Compose([SelectCA(), ExtractCATensor()])` then selects one amino-acid chain, validates its Cα trace, and extracts coordinate tensors. The same pipeline accepts an existing AtomArray. Dataset downloading/clustering and external refolding remain outside the core.

Estimated physical lines, including normal spacing and concise comments, for a readable baseline consuming prepared coordinates:

| Section in entries/baseline/main.py | Estimated lines |
| --- | --- |
| Imports, ExperimentConfig, CLI/config serialization | 60–90 |
| Cα adapter, cache loading, batching, centering, rotations | 80–120 |
| RoPE, attention, transformer blocks, coordinate denoiser | 100–150 |
| Diffusion preconditioning, loss, sampler | 70–110 |
| Training, held-out loss, EMA, logging, save/resume | 150–220 |
| Basic geometry evaluation and Cα PDB export | 70–110 |
| Main dispatch | 20–30 |
| **Total** | **550–830** |

Aim for roughly 700 lines in the working reference. A 300–400-line demonstration is possible with prepared fixed-length inputs and reduced logging/resume/evaluation, but it is a different scope. Robust raw PDB/mmCIF ingestion, download/filter/split manifests, and broader diversity/novelty evaluation could add roughly 250–500 lines, depending on library reuse and data source complexity. Tests and external designability orchestration are additional. These are planning estimates, not measured counts; keep scientific checks rather than golfing code to hit a limit.

The proposed data contract is deliberately small:

```python
CATokenizer()(atom_array, sample_id, chain) -> {id, coords, residue_id}
CATokenizer().from_file(path, chain) -> {id, coords, residue_id}
CATokenizer.to_pdb(coords, path)  # Cα-only placeholder residues
make_batch(...) -> (coords, mask)  # padded [B, L, 3], [B, L]
Denoiser.forward(x_preconditioned, noise_level, mask) -> residual
denoise(model, x_noisy, sigma, mask) -> clean_coordinate_prediction
```

Here “tokenizer” means the mapping from residues/structure records to continuous model inputs. It is not a learned codebook or a discrete structure vocabulary. Parsing and reconstruction live at this boundary; stochastic rotation and noise have explicit owners outside it. Generated PDB residue identities must be clearly marked as placeholders, not designed sequences.

The shared frozen `ExperimentConfig` owns data paths, seeds, device/precision, budgets, length limits, and evaluation settings. Each entry's frozen `Config(ExperimentConfig)` adds model/optimizer/diffusion fields. `Config.from_experiment(shared, **solution_settings)` preserves evaluator-supplied shared values and rejects attempts to override them as solution settings. Validation batch size is independent of the solution's training batch size. Resolve the combined flat configuration once; reject unknown keys and inconsistent dimensions; save it with every run. `main(config)` receives this object, and `train(config)` constructs components explicitly. Local CLI runs use `parse_config(Config)` for JSON and command-line overrides.

Each run saves configuration, code revision (or an explicit uncommitted-source marker), dataset/split hashes, environment and device information, elapsed time, metrics, and a resumable checkpoint containing model, EMA, optimizer, scheduler, step, and RNG/data-order state. Sampling loads architecture and normalization from the checkpoint.

## What “under an hour” means

The reference GPU model and VRAM are still unspecified. Propose one 24 GB CUDA GPU as the initial target class, then publish the actual device used; different GPUs do not constitute a fair fixed-wall-time leaderboard.

- CPU/MPS: correctness and small overfit smoke tests; no promised equivalent one-hour result.
- Debug preset: 16–32 structures and a very small model. Verify overfitting and the complete sample/evaluate path.
- Reference preset: at most 50 minutes from training process startup to saved final checkpoint, including compilation, in-run validation, and checkpoint overhead. Reserve the rest of the hour for the fixed cheap sample/evaluate pass; calibrate this allocation after profiling.
- Data download and preprocessing are a one-time prerequisite, measured and reported separately. Optional designability evaluation also has a separately disclosed budget.
- Run a brief throughput/memory pilot before a full attempt. Start around batch size 64, lower it if required, and select a step budget from measured throughput. Retain a wall-time stop and enough time to save cleanly.
- Record residues/second, optimizer steps, examples seen, peak memory, parameter count, and wall time. `torch.compile` is optional and stays off unless its startup cost pays back within the short run.
- Freeze the data, GPU, total budget, evaluation seeds/lengths, and inference function-evaluation budget for improvement comparisons. Report the number and total compute of tuning runs as well as the final run.

If a useful short-domain baseline does not emerge in the budget, narrow the domain family or revise the denoiser and label that benchmark explicitly. Do not present overfitting a handful of proteins as broad de novo generation.

## Evaluation that stays honest and affordable

Use a dashboard of metrics initially, not a weighted score that can be improved by producing identical helices.

| Level | Measurements | Interpretation |
| --- | --- | --- |
| Correctness | Finite values, padding behavior, export roundtrip, centering, repeatable seeds, checkpoint resume | Is the implementation working? |
| Held-out denoising | Fixed noise-bin losses, compared with zero/identity and untrained-model baselines | Does it generalize at the denoising task? |
| Local geometry | Adjacent Cα distance distribution/error around 3.8 Å, nonlocal Cα overlaps, virtual bond angles and signed torsions | Is the generated trace locally protein-like? |
| Global statistics | Radius of gyration and contact distributions by length, inexpensive Cα secondary-structure proxies | Does it resemble the intended domain distribution? |
| Diversity/novelty | Pairwise aligned Cα RMSD for matched lengths; final structure alignment/clustering and nearest-training similarity | Is it collapsing or memorizing? |
| Optional designability | Cα ProteinMPNN -> fixed folding model -> Cα scRMSD/scTM, folding confidence | Is there sequence/backbone self-consistency? |

For the inexpensive checks, start with 32 fixed-seed samples during development and a final panel of 256 samples spread across lengths 64, 96, and 128. Compare distributions with held-out real domains in matching length bins. Freeze any thresholds using training data or a calibration set before the final evaluation. Give explicit definitions for nonlocal pairs and contact cutoffs. Cα overlaps are a coarse clash proxy, not an all-atom clash score.

Report raw valid/invalid counts and individual sample results; failed samples must not disappear from denominators. Inspect signed torsions/left-handed helix prevalence because distance and compactness metrics miss mirror-image errors. A training-set resampling baseline helps expose metrics that reward memorization.

For an occasional designability check, use a fixed subset selected before looking at geometry, for example 32 backbones and 2 sequences per backbone. Keep inverse-folding checkpoint, temperature, folding checkpoint, seeds, and best-of-K protocol fixed. Report the scRMSD distribution and a preregistered pass fraction (an initial candidate is Cα scRMSD < 2 Å), with failures counted and uncertainty disclosed. A small panel is a development signal, not a stable leaderboard ranking; expand it for a release claim.

[ProteinMPNN provides Cα-only models](https://github.com/dauparas/ProteinMPNN), so this evaluation does not inherently require inventing full backbones first. [The Genie evaluation pipeline](https://github.com/aqlaboratory/insilico_design_pipeline) is a useful reference for self-consistency, structural diversity, novelty, and Cα secondary-structure evaluation. Keep those heavy dependencies outside the core training install. These are computational proxies, not experimental validation.

## Implementation sequence and release gates

1. **Data + contracts:** establish corpus counts and splits; implement tokenizer, cache, config, and Cα export. Verify gap/NaN handling, train-only normalization, and roundtrips.
2. **Tiny end-to-end model:** implement the denoiser and one diffusion objective/sampler. Test masking with unequal lengths, finite loss/gradients, centered noise, sampling endpoints, and overfitting a small real subset.
3. **Measured one-hour reference:** profile, freeze a preset, run from random initialization, sample, evaluate, and publish the complete result including weaknesses. Reference acceptance requires an improvement in generated-sample statistics as well as denoising loss; calibrate numerical targets against real/random baselines before claiming success.
4. **One quality ablation:** choose the dominant observed failure: self-conditioning, a local geometric loss, a different noise distribution, or a compact geometry-aware denoiser. Change one thing and compare at the same total compute and sampling cost.
5. **Teaching/research packaging:** document the derivation and three exercise functions, add a fixed evaluation protocol and results template, and only then establish the improvement challenge.

Before implementation, resolve the exact dataset artifact and reference GPU. Model size, sampler steps, and the time allocation above are starting hypotheses to profile, not measured defaults. Motif scaffolding, multichain conditioning, sequence generation, MSA features, all-atom representations, distributed training, and custom kernels are later projects.
