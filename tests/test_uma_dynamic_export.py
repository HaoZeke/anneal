"""One dynamic UMA AOTI package against FAIRChemCalculator.

The exporter writes one float64 package from HCN. That package is then
evaluated on HCN, on HCN with the hydrogen moved past the 6 angstrom
cutoff of nitrogen (6 edges become 4), on ethane before and after its
two methyl groups are pulled apart (56 edges become 24), and on water.
Each result is compared with FAIRChemCalculator on the same checkpoint
in float64: |dE| <= 1e-5 eV and max |dF| <= 1e-4 eV/A.

fairchem's own edge frames carry a random roll, so two FAIRChemCalculator
runs on one geometry already differ by about 1e-7 eV and 3e-6 eV/A in
float64; the bounds sit well above that.

Needs fairchem-core, vesin and the uma-s-1p1 checkpoint at
$RGPOT_UMA_CHECKPOINT, and skips without them. The export compiles the
package, which takes a few minutes on CPU.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
ase = pytest.importorskip("ase")
pytest.importorskip("vesin")
pytest.importorskip("fairchem.core")

from ase import Atoms
from ase.build import molecule

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
import export_uma_aoti as exp

CHECKPOINT = os.environ.get("RGPOT_UMA_CHECKPOINT", "")
pytestmark = pytest.mark.skipif(
    not CHECKPOINT or not Path(CHECKPOINT).is_file(),
    reason="RGPOT_UMA_CHECKPOINT does not name a uma-s-1p1 checkpoint",
)

E_TOL = 1e-5
F_TOL = 1e-4
BOX = 25.0


def _boxed(atoms: Atoms) -> Atoms:
    atoms = atoms.copy()
    atoms.set_cell([BOX, BOX, BOX])
    atoms.set_pbc(True)
    atoms.center()
    atoms.info.update({"charge": 0, "spin": 1})
    return atoms


def _hcn_h_past_n() -> Atoms:
    """H on the C side of the axis, 5.5 A from C and 6.7 A from N."""
    atoms = exp.HCN.copy()
    pos = atoms.get_positions()
    c, n = pos[0], pos[1]
    axis = (c - n) / np.linalg.norm(c - n)
    pos[2] = c + 5.5 * axis
    atoms.set_positions(pos)
    atoms.info.update({"charge": 0, "spin": 1})
    return atoms


def _ethane_pulled_apart() -> Atoms:
    """The two methyl groups 5 A further apart along the C-C bond."""
    atoms = _boxed(molecule("C2H6"))
    pos = atoms.get_positions()
    c0, c1 = pos[0], pos[1]
    axis = (c1 - c0) / np.linalg.norm(c1 - c0)
    side = (pos - c0) @ axis > 0.5 * np.linalg.norm(c1 - c0)
    pos[side] += 5.0 * axis
    atoms.set_positions(pos)
    return atoms


CASES = {
    "hcn": lambda: exp.HCN.copy(),
    "hcn_h_past_n": _hcn_h_past_n,
    "ethane": lambda: _boxed(molecule("C2H6")),
    "ethane_pulled_apart": _ethane_pulled_apart,
    "water": lambda: _boxed(molecule("H2O")),
}


def fairchem_reference(atoms: Atoms):
    from fairchem.core import FAIRChemCalculator
    from fairchem.core.units.mlip_unit import load_predict_unit
    from fairchem.core.units.mlip_unit.api.inference import InferenceSettings

    settings = InferenceSettings(
        tf32=False,
        activation_checkpointing=False,
        merge_mole=True,
        compile=False,
        external_graph_gen=False,
        execution_mode="general",
        base_precision_dtype=torch.float64,
    )
    pred = load_predict_unit(CHECKPOINT, inference_settings=settings, device="cpu")
    pred.model.double()
    probe = atoms.copy()
    probe.calc = FAIRChemCalculator(pred, task_name="omol")
    return float(probe.get_potential_energy()), np.asarray(probe.get_forces())


@pytest.fixture(scope="module")
def package(tmp_path_factory):
    out = tmp_path_factory.mktemp("uma") / "uma-s-1p1-omol-dynamic.pt2"
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "export_uma_aoti.py"),
            "--model",
            "uma-s-1p1",
            "--dtype",
            "float64",
            "--out",
            str(out),
        ],
        cwd=ROOT,
        env=dict(os.environ, RGPOT_UMA_CHECKPOINT=CHECKPOINT),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0 or not out.is_file():
        tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-40:])
        pytest.fail(f"export_uma_aoti.py exited {proc.returncode}\n{tail}")
    pkg = exp.load_aoti(out)
    return pkg, pkg.get_metadata()


def test_package_records_dynamic_bounds(package):
    _pkg, meta = package
    assert meta["export_path"] == "dynamic"
    assert meta["pos_shape"] == "natoms,3"
    assert meta["edge_index_shape"] == "2,nedges"
    assert int(meta["natoms_min"]) <= 3
    assert int(meta["natoms_max"]) >= 8
    assert int(meta["nedges_min"]) <= 2
    assert int(meta["nedges_max"]) >= 56
    assert meta["composition"] == "any"
    for key in ("natoms", "counts", "z_set"):
        assert key not in meta


def test_edge_counts_differ():
    cutoff = 6.0
    counts = {}
    for name, build in CASES.items():
        atoms = build()
        example = exp.pack_example(atoms, cutoff, "cpu", torch.float64, 300)
        counts[name] = int(example[4].shape[1])
    assert counts["hcn"] == 6
    assert counts["hcn_h_past_n"] == 4
    assert counts["ethane"] == 56
    assert counts["ethane_pulled_apart"] < 56


@pytest.mark.parametrize("name", list(CASES))
def test_matches_fairchem_calculator(package, name):
    pkg, meta = package
    atoms = CASES[name]()
    example = exp.pack_example(
        atoms,
        float(meta["cutoff"]),
        "cpu",
        torch.float64,
        int(meta["max_neighbors"]),
    )
    energy, forces = exp.run_aoti(pkg, example)
    energy = float(torch.as_tensor(energy).item())
    forces = torch.as_tensor(forces).detach().cpu().numpy().reshape(-1, 3)
    e_ref, f_ref = fairchem_reference(atoms)
    de = abs(energy - e_ref)
    df = float(np.max(np.abs(forces - f_ref)))
    print(
        f"{name}: natoms={len(atoms)} nedges={int(example[4].shape[1])} "
        f"|dE|={de:.3e} eV max|dF|={df:.3e} eV/A"
    )
    assert de <= E_TOL
    assert df <= F_TOL
