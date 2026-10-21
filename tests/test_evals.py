import csv
import json

import numpy as np
import pytest
from biotite.structure import AtomArray

from nanorfd.evals import evaluate, geometry_metrics


def trace(length=12):
    i = np.arange(length)
    atoms = AtomArray(length)
    atoms.coord = np.column_stack((2.3 * np.cos(i * 1.74), 2.3 * np.sin(i * 1.74), i * 1.5))
    atoms.chain_id[:] = "A"
    atoms.res_id = i + 1
    atoms.res_name[:] = "GLY"
    atoms.atom_name[:] = "CA"
    atoms.element[:] = "C"
    return atoms


def test_geometry_is_rotation_translation_invariant_and_detects_reflection():
    atoms = trace()
    original = atoms.coord.copy()
    a = geometry_metrics(atoms)
    moved = atoms.copy()
    moved.coord = atoms.coord @ np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]) + 10
    b = geometry_metrics(moved)
    assert a == pytest.approx(b, abs=2e-6)
    moved.coord = atoms.coord * np.array([-1, 1, 1])
    assert geometry_metrics(moved)["signed_volume_mean"] == pytest.approx(-a["signed_volume_mean"])
    np.testing.assert_array_equal(atoms.coord, original)


def test_bad_geometry_is_scored_and_pair_denominators_are_fixed():
    atoms = trace(12)
    atoms.coord[:] = 0
    metrics = geometry_metrics(atoms)
    assert metrics["ca_distance_mae"] == pytest.approx(3.8)
    assert metrics["nonlocal_clash_fraction"] == 1
    assert metrics["contact_fraction"] == 1
    assert metrics["contact_pair_count"] == 6  # Separations 9, 10, 11: 3 + 2 + 1.
    assert metrics["degenerate_triplets"] == 9
    assert metrics["radius_of_gyration"] == metrics["signed_volume_mean"] == 0
    atoms = trace(4)
    atoms.coord = np.column_stack((np.arange(4) * 3.8, np.zeros((4, 2))))
    metrics = geometry_metrics(atoms)
    assert metrics["ca_distance_mae"] == pytest.approx(0, abs=1e-6)
    assert metrics["nonlocal_clash_fraction"] == 0
    assert metrics["contact_pair_count"] == metrics["contact_fraction"] == 0


def test_full_atom_input_selects_carbon_ca():
    ca = trace()
    nitrogen = ca.copy()
    nitrogen.atom_name[:] = "N"
    nitrogen.element[:] = "N"
    # Interleave N/CA to keep atoms grouped by residue.
    atoms = (nitrogen + ca)[np.column_stack((np.arange(12), np.arange(12, 24))).ravel()]
    assert geometry_metrics(atoms) == geometry_metrics(ca)
    with pytest.raises(ValueError, match="exactly one CA"):
        geometry_metrics(atoms[:-1])


def test_shared_reports_weight_samples_equally(tmp_path):
    samples = {"short": trace(4), "long": trace(12)}
    report = evaluate(samples, tmp_path)
    assert report["length"] == 8
    assert report["num_samples"] == 2
    assert report == json.loads((tmp_path / "geometry.json").read_text())
    with (tmp_path / "geometry.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert [row["id"] for row in rows] == ["short", "long"]
    assert report == evaluate(samples)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf])
def test_nonfinite_samples_raise_before_reports_are_written(tmp_path, bad_value):
    bad = trace()
    bad.coord[0, 0] = bad_value
    with pytest.raises(ValueError, match="bad: NaN/Inf"):
        evaluate({"good": trace(), "bad": bad}, tmp_path)
    assert not list(tmp_path.iterdir())


def test_invalid_atom_arrays_are_rejected():
    with pytest.raises(ValueError, match="empty"):
        evaluate({})
    with pytest.raises(TypeError, match="AtomArray"):
        geometry_metrics(np.zeros((4, 3)))
    with pytest.raises(ValueError, match="four CA"):
        geometry_metrics(trace(3))
    multi = trace()
    multi.chain_id[-1] = "B"
    with pytest.raises(ValueError, match="one protein chain"):
        geometry_metrics(multi)
    with pytest.raises(ValueError, match="exactly one CA"):
        geometry_metrics(trace()[[0, 0, 1, 2, 3]])
