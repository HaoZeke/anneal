"""Batched UMA energies and forces match one forward per system.

Two molecules of different sizes, two geometries each. The band forward
is the graph ``UmaPot`` runs when ``batch_max`` is greater than 1. The
test skips when torch, fairchem, or the uma-s-1p1 checkpoint is absent.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("ase")
pytest.importorskip("fairchem.core")

import numpy as np
from ase import Atoms

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import export_uma_aoti as exp  # noqa: E402

COPE_XYZ = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "data"
    / "birkholz_04_cope_ts.xyz"
)


def _checkpoint_present() -> bool:
    roots = [
        Path.home() / ".cache" / "fairchem",
        Path.home() / ".cache" / "huggingface",
    ]
    for key in ("FAIRCHEM_CACHE", "HF_HOME", "HUGGINGFACE_HUB_CACHE"):
        value = os.environ.get(key)
        if value:
            roots.append(Path(value))
    for root in roots:
        if root.is_dir() and any(root.glob("**/uma-s-1p1.pt")):
            return True
    return False


def _load_xyz(path: Path) -> Atoms:
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    n = int(lines[0])
    symbols = []
    pos = []
    for ln in lines[2 : 2 + n]:
        parts = ln.split()
        symbols.append(parts[0])
        pos.append([float(parts[1]), float(parts[2]), float(parts[3])])
    atoms = Atoms(symbols=symbols, positions=pos)
    atoms.info.update({"charge": 0, "spin": 1})
    return atoms


def _perturb(atoms: Atoms) -> Atoms:
    out = atoms.copy()
    pos = np.asarray(out.get_positions(), dtype=np.float64)
    rng = np.random.default_rng(0)
    out.set_positions(pos + rng.normal(scale=0.05, size=pos.shape))
    out.info.update({"charge": 0, "spin": 1})
    return out


def _place(atoms: Atoms, box: float) -> Atoms:
    out = atoms.copy()
    pos = np.asarray(out.get_positions(), dtype=np.float64)
    out.set_positions(pos - pos.mean(axis=0) + box / 2.0)
    out.set_cell([[box, 0.0, 0.0], [0.0, box, 0.0], [0.0, 0.0, box]])
    out.set_pbc(True)
    out.info.update({"charge": 0, "spin": 1})
    return out


def _pack_many(frames, cutoff, device, dtype, max_neighbors):
    pieces = []
    for atoms in frames:
        pos_np = np.asarray(atoms.get_positions(), dtype=np.float64)
        cell_np = np.asarray(atoms.get_cell(), dtype=np.float64)
        edge_index, shifts = exp._vesin_edges(
            pos_np, cell_np, cutoff, max_neighbors=max_neighbors
        )
        pieces.append(
            (
                pos_np,
                np.asarray(atoms.get_atomic_numbers()),
                cell_np,
                edge_index,
                np.asarray(shifts),
                int(atoms.info.get("charge", 0)),
                int(atoms.info.get("spin", 1)),
            )
        )
    n = pieces[0][0].shape[0]
    count = len(pieces)
    pos = torch.tensor(
        np.concatenate([p[0] for p in pieces], axis=0), dtype=dtype, device=device
    )
    atomic_numbers = torch.tensor(
        np.concatenate([p[1] for p in pieces], axis=0),
        dtype=torch.long,
        device=device,
    )
    cell = torch.tensor(
        np.stack([p[2] for p in pieces], axis=0), dtype=dtype, device=device
    )
    pbc = torch.tensor([[True, True, True]] * count, device=device)
    edges = torch.tensor(
        np.concatenate([p[3] + b * n for b, p in enumerate(pieces)], axis=1),
        dtype=torch.long,
        device=device,
    )
    offsets = torch.tensor(
        np.concatenate([p[4] for p in pieces], axis=0), dtype=dtype, device=device
    )
    charge = torch.tensor([p[5] for p in pieces], dtype=torch.long, device=device)
    spin = torch.tensor([p[6] for p in pieces], dtype=torch.long, device=device)
    batch = torch.arange(count, dtype=torch.long, device=device).repeat_interleave(n)
    natoms = torch.tensor([n] * count, dtype=torch.long, device=device)
    nedges = torch.tensor(
        [p[3].shape[1] for p in pieces], dtype=torch.long, device=device
    )
    return (
        pos,
        atomic_numbers,
        cell,
        pbc,
        edges,
        offsets,
        charge,
        spin,
        batch,
        natoms,
        nedges,
    )


def test_batched_forward_matches_per_system_on_two_sizes():
    if not _checkpoint_present():
        pytest.skip("uma-s-1p1 checkpoint is not in the local fairchem cache")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
    torch.set_num_threads(2)

    try:
        pred = exp.load_predictor("uma-s-1p1", "cpu")
    except Exception as exc:  # noqa: BLE001 — missing weights, not a failure
        pytest.skip(f"uma-s-1p1 is not available: {exc}")

    hydra = pred.model
    exp.apply_export_dtype(hydra, "float32")
    backbone = hydra.module.backbone
    backbone.otf_graph = False
    cutoff = float(backbone.cutoff)
    max_neighbors = int(getattr(backbone, "max_neighbors", 300) or 300)
    device = torch.device("cpu")
    dtype = torch.float32
    # Construction bakes the dataset embedding into the backbone, so the
    # per-system reference is the single-system forward of this module.
    band = (
        exp.TensorUmaBatched(hydra, pred_unit=pred, task_name="omol")
        .to(device)
        .eval()
    )

    molecules = {
        "hcn": exp.HCN.copy(),
        "cope": _load_xyz(COPE_XYZ),
    }
    sizes = {name: len(atoms) for name, atoms in molecules.items()}
    assert len(set(sizes.values())) == 2
    assert sizes["hcn"] != sizes["cope"]

    box = 40.0
    for name, base in molecules.items():
        frames = [_place(base, box), _place(_perturb(base), box)]
        n = len(frames[0])
        singles_e = []
        singles_f = []
        with torch.enable_grad():
            for atoms in frames:
                example = exp.pack_example(
                    atoms, cutoff, device, dtype, max_neighbors, n_systems=1
                )
                energy, forces = exp.TensorUma.forward(band, *example)
                singles_e.append(float(energy.detach().cpu()))
                singles_f.append(forces.detach().cpu().numpy().reshape(n, 3))
            batched = _pack_many(frames, cutoff, device, dtype, max_neighbors)
            energy_b, forces_b = band(*batched)
        energy_b = energy_b.detach().cpu().numpy()
        forces_b = forces_b.detach().cpu().numpy().reshape(len(frames), n, 3)
        for i in range(len(frames)):
            assert energy_b[i] == pytest.approx(singles_e[i], abs=1e-4)
            assert forces_b[i] == pytest.approx(singles_f[i], abs=1e-4)

    # The packed export replaces MoLE's per-system segment loop with one
    # GEMM and always uses the batched cell shift. The reference is the
    # unpatched single-system forward. TensorUma.forward writes a dummy
    # nedges, which that shift does not read; the packed call carries
    # the real per-system edge counts.
    cope_frames = [
        _place(molecules["cope"], box),
        _place(_perturb(molecules["cope"]), box),
        _place(molecules["cope"], box),
    ]
    cope_frames[2].positions[0, 0] += 0.15
    n = len(cope_frames[0])
    singles_e = []
    singles_f = []
    with torch.enable_grad():
        for atoms in cope_frames:
            example = exp.pack_example(
                atoms, cutoff, device, dtype, max_neighbors, n_systems=1
            )
            energy, forces = exp.TensorUma.forward(band, *example)
            singles_e.append(float(energy.detach().cpu()))
            singles_f.append(forces.detach().cpu().numpy().reshape(n, 3))
    exp.patch_export_ops()
    with torch.enable_grad():
        for count in (1, 2, 3):
            batched = _pack_many(
                cope_frames[:count], cutoff, device, dtype, max_neighbors
            )
            energy_b, forces_b = band(*batched)
            energy_b = energy_b.detach().cpu().numpy()
            forces_b = forces_b.detach().cpu().numpy().reshape(count, n, 3)
            for i in range(count):
                assert energy_b[i] == pytest.approx(singles_e[i], abs=1e-4)
                assert forces_b[i] == pytest.approx(singles_f[i], abs=1e-4)
