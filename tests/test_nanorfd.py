import json
from dataclasses import asdict, replace

import numpy as np
import pytest
import torch
from biotite.structure import AtomArray
from biotite.structure.io.pdbx import CIFFile, set_structure

from entries.baseline import main as n


@pytest.fixture(autouse=True)
def cpu_threads():
    torch.set_num_threads(1)


def helix(length=8, radius=2.3):
    i = torch.arange(length).float()
    return torch.stack(
        (radius * torch.cos(i * 1.74), radius * torch.sin(i * 1.74), i * 1.5), dim=-1
    )


def atom_array(length=8):
    atoms = AtomArray(3 * length)
    atoms.coord = (
        (helix(length)[:, None, :] + torch.tensor([[-1.0, 0, 0], [0, 0, 0], [1.0, 0, 0]]))
        .reshape(-1, 3)
        .numpy()
    )
    atoms.chain_id[:] = "A"
    atoms.res_id = np.repeat(np.arange(1, length + 1), 3)
    atoms.res_name[:] = "ALA"
    atoms.atom_name = np.tile(["N", "CA", "C"], length)
    atoms.element = np.tile(["N", "C", "C"], length)
    atoms.set_annotation("occupancy", np.ones(len(atoms)))
    return atoms


def tiny_config(**kwargs):
    return n.Config(
        width=32,
        depth=1,
        heads=4,
        mlp_ratio=2,
        batch_size=2,
        steps=8,
        warmup_steps=2,
        eval_every=4,
        eval_batches=1,
        eval_batch_size=2,
        log_every=4,
        max_minutes=0,
        threads=1,
        device="cpu",
        sample_steps=4,
        length=8,
        num_samples=2,
        **kwargs,
    )


def test_atomworks_pipeline_selects_ca_without_mutating_input():
    atoms = atom_array()
    original = atoms.copy()
    tokenizer = n.CATokenizer()
    sample = tokenizer(atoms, "example", chain="A")
    torch.testing.assert_close(sample["coords"], helix())
    assert sample["id"] == "example:A"
    assert len(atoms) == 24
    np.testing.assert_array_equal(atoms.coord, original.coord)
    assert [type(t).__name__ for t in tokenizer.pipeline.transforms] == [
        "SelectCA",
        "ExtractCATensor",
    ]


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("nan", "NaN/Inf"),
        ("missing_ca", "exactly one CA"),
        ("duplicate_ca", "exactly one CA"),
        ("gap", "numbering gap"),
        ("break", "broken chain"),
        ("unresolved", "Unresolved CA"),
    ],
)
def test_tokenizer_rejects_invalid_traces(mutation, match):
    atoms = atom_array()
    if mutation == "nan":
        atoms.coord[1, 0] = np.nan
    elif mutation == "missing_ca":
        atoms = atoms[np.arange(len(atoms)) != 4]
    elif mutation == "duplicate_ca":
        atoms.atom_name[0], atoms.element[0] = "CA", "C"
    elif mutation == "gap":
        atoms.res_id[12:] += 1
    elif mutation == "break":
        atoms.coord[12:] += 100
    else:
        atoms.occupancy[1] = 0
    with pytest.raises(ValueError, match=match):
        n.CATokenizer()(atoms)


def test_chain_selection_and_calcium_exclusion():
    a, b = atom_array(), atom_array(6)
    b.chain_id[:] = "B"
    with pytest.raises(ValueError, match="Select one protein chain"):
        n.CATokenizer()(a + b)
    torch.testing.assert_close(n.CATokenizer()(a + b, chain="B")["coords"], helix(6))
    ion = AtomArray(1)
    ion.coord[:] = 0
    ion.res_name[:], ion.atom_name[:], ion.element[:] = "CA", "CA", "Ca"
    ion.chain_id[:] = "Z"
    ion.set_annotation("occupancy", np.ones(1))
    torch.testing.assert_close(n.CATokenizer()(a + ion)["coords"], helix())


@pytest.mark.parametrize("extension", ["pdb", "cif"])
def test_real_atomworks_file_loading(tmp_path, extension):
    path = tmp_path / f"trace.{extension}"
    if extension == "pdb":
        n.CATokenizer.to_pdb(helix(), path)
    else:
        cif = CIFFile()
        set_structure(cif, atom_array())
        cif.write(path)
    sample = n.CATokenizer().from_file(path, chain="A")
    torch.testing.assert_close(sample["coords"], helix(), atol=5.1e-4, rtol=0)


def test_prepare_uses_manifest_and_rejects_leakage(tmp_path):
    entries = []
    for i, length in enumerate((7, 8, 9)):
        path = tmp_path / f"trace{i}.pdb"
        n.CATokenizer.to_pdb(helix(length), path)
        entries.append({"path": path.name, "chain": "A"})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"train": entries[:2], "val": entries[2:]}))
    cfg = tiny_config(data=str(tmp_path / "data.pt"), manifest=str(manifest))
    n.prepare(cfg)
    data = torch.load(cfg.data, weights_only=True)
    assert len(data["train"]) == 2 and len(data["val"]) == 1
    data["val"] = [data["train"][0]]
    with pytest.raises(ValueError, match="Duplicate structure"):
        n.validate_dataset(data, cfg)


def test_center_and_augmentation_preserve_distances_and_handedness():
    x = helix()[None]
    mask = torch.ones(1, 8, dtype=torch.bool)
    translated = torch.cat((x + 100, torch.full((1, 3, 3), 999.0)), dim=1)
    padded_mask = torch.cat((mask, torch.zeros(1, 3, dtype=torch.bool)), dim=1)
    actual = n.center(translated, padded_mask)
    torch.testing.assert_close(actual[:, :8], x - x.mean(1, keepdim=True), atol=1e-5, rtol=0)
    assert actual[:, 8:].count_nonzero() == 0
    rotated, _ = n.make_batch([{"coords": x[0]}], 1, 1.0, torch.Generator().manual_seed(1), "cpu")
    torch.testing.assert_close(torch.cdist(rotated, rotated), torch.cdist(x, x))
    original_volume = torch.linalg.det(x[0, 1:4] - x[0, 0])
    rotated_volume = torch.linalg.det(rotated[0, 1:4] - rotated[0, 0])
    torch.testing.assert_close(original_volume, rotated_volume)


def test_rope_preserves_norms_and_relative_position():
    attention = n.Attention(tiny_config())
    q, k = torch.randn(2, 4, 5, 8), torch.randn(2, 4, 5, 8)
    qr, kr = attention.rope(q), attention.rope(k)
    torch.testing.assert_close(q.norm(dim=-1), qr.norm(dim=-1))
    pad = torch.zeros(2, 4, 3, 8)
    shifted_q = attention.rope(torch.cat((pad, q), dim=2))[:, :, 3:]
    shifted_k = attention.rope(torch.cat((pad, k), dim=2))[:, :, 3:]
    torch.testing.assert_close(qr @ kr.transpose(-1, -2), shifted_q @ shifted_k.transpose(-1, -2))


def test_padding_cannot_change_real_predictions_and_loss_has_gradients():
    cfg = tiny_config()
    model = n.Denoiser(cfg)
    # Exercise the complete network instead of only its initially zero output projection.
    torch.nn.init.normal_(model.output[-1].weight, std=0.1)
    x = torch.randn(2, 8, 3)
    mask = torch.ones(2, 8, dtype=torch.bool)
    sigma = torch.tensor([0.1, 3.0])
    prediction = n.denoise(model, x, sigma, mask)
    padded_x = torch.cat((x, torch.randn(2, 5, 3) * 1000), dim=1)
    padded_mask = pad_mask(mask, 5)
    padded_prediction = n.denoise(model, padded_x, sigma, padded_mask)
    torch.testing.assert_close(prediction, padded_prediction[:, :8], atol=1e-6, rtol=1e-5)
    assert padded_prediction[:, 8:].count_nonzero() == 0
    loss = n.diffusion_loss(
        model, n.center(padded_x, padded_mask), padded_mask, cfg, torch.Generator().manual_seed(3)
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def pad_mask(mask, count):
    return torch.cat((mask, torch.zeros(len(mask), count, dtype=torch.bool)), dim=1)


def test_euler_endpoint_and_mask_with_oracle(monkeypatch):
    cfg = tiny_config()
    seen = []
    target = n.center(
        torch.arange(42).reshape(2, 7, 3).float() / 10,
        torch.arange(7)[None, :] < torch.tensor([4, 7])[:, None],
    )

    def oracle(model, x, sigma, mask):
        assert (sigma > 0).all()  # The zero endpoint must not be fed through log(sigma).
        torch.testing.assert_close(x, n.center(x, mask), atol=1e-5, rtol=1e-5)
        seen.append(sigma[0].item())
        return target

    monkeypatch.setattr(n, "denoise", oracle)
    samples = n.sample_coordinates(
        n.Denoiser(cfg), [4, 7], cfg, torch.Generator().manual_seed(2), torch.device("cpu")
    )
    assert len(seen) == cfg.sample_steps
    torch.testing.assert_close(samples[0], target[0, :4], atol=1e-5, rtol=0)
    torch.testing.assert_close(samples[1], target[1], atol=1e-5, rtol=0)


def test_training_resume_and_sample_end_to_end(tmp_path, monkeypatch):
    data_path = tmp_path / "data.pt"
    records = [n.CATokenizer.encode(helix(length), str(length)) for length in (6, 7, 8)]
    torch.save({"version": 1, "train": records[:2], "val": records[2:]}, data_path)
    cfg = tiny_config(data=str(data_path), out_dir=str(tmp_path / "full"))
    n.train(cfg)
    reference = torch.load(tmp_path / "full/checkpoint.pt", weights_only=True)
    original_save = n.save_checkpoint

    class Interrupted(Exception):
        pass

    def stop_at_four(*args):
        original_save(*args)
        if args[4] == 4:
            raise Interrupted

    partial = replace(cfg, out_dir=str(tmp_path / "partial"))
    monkeypatch.setattr(n, "save_checkpoint", stop_at_four)
    with pytest.raises(Interrupted):
        n.train(partial)
    monkeypatch.setattr(n, "save_checkpoint", original_save)
    resumed = replace(partial, resume=str(tmp_path / "partial/checkpoint.pt"))
    n.train(resumed)
    actual = torch.load(resumed.resume, weights_only=True)
    for group in ("model", "ema"):
        for key in reference[group]:
            torch.testing.assert_close(reference[group][key], actual[group][key], rtol=0, atol=0)
    assert torch.equal(reference["rng"], actual["rng"])
    assert actual["step"] == cfg.steps
    generation = replace(cfg, checkpoint=resumed.resume, out_dir=str(tmp_path / "samples"))
    n.sample(generation)
    assert len(list((tmp_path / "samples").glob("*.pdb"))) == 2
    report = json.loads((tmp_path / "samples/geometry.json").read_text())
    assert report["num_samples"] == 2
    with pytest.raises(ValueError, match="Resume setting changed"):
        n.train(replace(resumed, width=64))


def test_geometry_is_rotation_invariant_and_nan_is_an_error(tmp_path):
    x = helix()
    a = n.geometry_metrics({"id": "a", "coords": x})
    rotation = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    b = n.geometry_metrics({"id": "b", "coords": x @ rotation + 10})
    for key in a:
        if key != "id":
            assert a[key] == pytest.approx(b[key], abs=2e-6)
    x[0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN/Inf"):
        n.evaluate([{"id": "bad", "coords": x}], tmp_path)


def test_validation_does_not_depend_on_training_batch_size():
    cfg = replace(tiny_config(), eval_batch_size=3)
    records = [n.CATokenizer.encode(helix(length), str(length)) for length in (6, 8)]
    model = n.Denoiser(cfg)
    first = n.validation_loss(model, records, 1.0, cfg, torch.device("cpu"))
    second = n.validation_loss(model, records, 1.0, replace(cfg, batch_size=7), torch.device("cpu"))
    assert first == second


def test_config_rejects_invalid_shapes_and_unknown_fields():
    with pytest.raises(ValueError, match="head dimension"):
        replace(tiny_config(), width=30).validate()
    with pytest.raises(TypeError):
        n.Config(**{**asdict(tiny_config()), "unknown": 1})
