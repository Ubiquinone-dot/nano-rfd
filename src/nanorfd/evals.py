"""Fixed CA geometry diagnostics and reporting for all entries.

Inputs are single-chain AtomArrays in residue order, with coordinates in angstroms.
Full-atom arrays are also accepted: each amino-acid residue must have exactly one CA.
These are coarse trace diagnostics, not all-atom validity or designability scores.
"""

import csv
import json
import math
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from biotite.structure import AtomArray, filter_amino_acids, get_residue_starts


def geometry_metrics(atoms: AtomArray) -> dict[str, float | int]:
    """Measure one trace, including bad bond lengths/collapses; reject invalid inputs."""
    if not isinstance(atoms, AtomArray):
        raise TypeError("Expected one AtomArray, not an ensemble/AtomArrayStack")
    protein = atoms[filter_amino_acids(atoms)]
    if len(np.unique(protein.chain_id)) != 1 or (
        "chain_iid" in protein.get_annotation_categories()
        and len(np.unique(protein.chain_iid)) != 1
    ):
        raise ValueError("Expected one protein chain")
    is_ca = (protein.atom_name == "CA") & (protein.element == "C")
    starts = get_residue_starts(protein)
    if not (np.add.reduceat(is_ca.astype(int), starts) == 1).all():
        raise ValueError("Each amino-acid residue must have exactly one CA atom")
    ca = protein[is_ca]
    if len(ca) < 4:
        raise ValueError("Expected at least four CA atoms")
    keys = list(zip(ca.res_id.tolist(), ca.ins_code.tolist()))
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate residues in CA trace")
    x = ca.coord.astype(np.float64)
    if not np.isfinite(x).all():
        raise ValueError("NaN/Inf in CA coordinates")
    bonds = np.diff(x, axis=0)
    lengths = np.linalg.norm(bonds, axis=-1)
    distances = np.linalg.norm(x[:, None] - x[None, :], axis=-1)
    index = np.arange(len(x))
    separation = index[None, :] - index[:, None]  # Count each unordered pair once.
    nonlocal_pairs, distant_pairs = separation > 2, separation > 8
    normals = np.cross(bonds[:-1], bonds[1:])
    triple = (normals[:-1] * bonds[2:]).sum(-1)
    norm = lengths[:-2] * lengths[1:-1] * lengths[2:]
    normal_lengths = np.linalg.norm(normals, axis=-1)
    metrics = {
        "length": len(x),
        "ca_distance_mean": float(lengths.mean()),
        "ca_distance_mae": float(np.abs(lengths - 3.8).mean()),
        "nonlocal_clash_fraction": float(((distances < 3) & nonlocal_pairs).sum())
        / max(1, int(nonlocal_pairs.sum())),
        "contact_fraction": float(((distances < 8) & distant_pairs).sum())
        / max(1, int(distant_pairs.sum())),
        "contact_pair_count": int(distant_pairs.sum()),
        "radius_of_gyration": float(np.sqrt(((x - x.mean(0)) ** 2).sum(-1).mean())),
        "signed_volume_mean": float((triple / np.maximum(norm, 1e-8)).mean()),
        "degenerate_triplets": int((normal_lengths[:-1] * normal_lengths[1:] < 1e-8).sum()),
    }
    if not all(math.isfinite(value) for value in metrics.values()):
        raise ValueError("NaN/Inf in geometry metrics")
    return metrics


def evaluate(samples: Mapping[str, AtomArray], out_dir=None) -> dict[str, float | int]:
    """Score named outputs, average equally over samples, and optionally write reports.

    No samples are dropped. Invalid inputs raise with their sample ID before any
    reports are written. Metric definitions and cutoffs are shared across entries.
    """
    if not samples:
        raise ValueError("Cannot evaluate an empty sample set")
    rows = []
    for sample_id, atoms in samples.items():
        try:
            rows.append({"id": sample_id, **geometry_metrics(atoms)})
        except (TypeError, ValueError) as error:
            raise ValueError(f"{sample_id}: {error}") from error
    summary = {"num_samples": len(rows)}
    summary.update(
        {key: sum(row[key] for row in rows) / len(rows) for key in rows[0] if key != "id"}
    )
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        with (out / "geometry.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (out / "geometry.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, allow_nan=False), flush=True)
    return summary
