"""librgpot_gromacs through its libcp2k C API, without GROMACS.

The calls are the ones QMMMForceProvider makes each step: create the
force environment from the -qmi input and the PDB file, set positions and
cell in Bohr, evaluate, read the energy in Hartree and the forces in
Hartree/Bohr. Every result is compared with rgpot-eval on the same frame.
A refused input exits the process, so those cases run in a subprocess.

Environment: RGPOT_GROMACS_DROPIN (the library), RGPOT_EVAL (rgpot-eval),
and optionally RGPOT_CLASSICAL_ENGINE with RGPOT_CAPNP_SCHEMA for the
engine route.
"""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dropin_case as dc  # noqa: E402

LIB = os.environ.get("RGPOT_GROMACS_DROPIN", "")
EVAL = os.environ.get("RGPOT_EVAL", "")
ENGINE = os.environ.get("RGPOT_CLASSICAL_ENGINE", "")
SCHEMA = os.environ.get("RGPOT_CAPNP_SCHEMA", "")

pytestmark = pytest.mark.skipif(not (LIB and EVAL),
                                reason="needs RGPOT_GROMACS_DROPIN and RGPOT_EVAL")

PT = ["SuttonChen", "preset", "platinum", "cutoff", "5.0"]
LJ = ["LJ", "u0", "0.0104", "psi", "3.4", "cutoff", "8.0"]


def pdb(symbols, x_nm, qm, charges=None) -> str:
    """The PDB file GROMACS writes (QMMMInputGenerator::generateCP2KPdb)."""
    charges = charges or {}
    lines = []
    for i, (s, row) in enumerate(zip(symbols, x_nm)):
        tag = " QM     1     " if i in qm else " MM     2     "
        xyz = " ".join(f"{10 * v:7.3f}" for v in row)
        lines.append(f"ATOM  {i % 100000:5d}  {s:<3s} {tag}{xyz}  1.00  0.00          "
                     f"{s:<3s} {charges.get(i, 0.0):f}")
    return "\n".join(lines) + "\n"


def write_input(work: Path, rgpot: str, symbols, x_nm, qm=None, charges=None,
                charge: int = 0, multiplicity: int = 1) -> Path:
    work.mkdir(parents=True, exist_ok=True)
    qm = set(range(len(symbols))) if qm is None else set(qm)
    (work / "rgpot.pdb").write_text(pdb(symbols, x_nm, qm, charges))
    inp = work / "rgpot.inp"
    inp.write_text(dc.qmi(rgpot, charge, multiplicity, pdb=str(work / "rgpot.pdb")))
    return inp


class Shim:
    """One force environment, called the way GROMACS calls it."""

    def __init__(self, inp: Path, out: Path):
        self.lib = ctypes.CDLL(LIB)
        d = ctypes.POINTER(ctypes.c_double)
        self.lib.cp2k_set_positions.argtypes = [ctypes.c_int, d, ctypes.c_int]
        self.lib.cp2k_set_cell.argtypes = [ctypes.c_int, d]
        self.lib.cp2k_get_forces.argtypes = [ctypes.c_int, d, ctypes.c_int]
        self.lib.cp2k_get_positions.argtypes = [ctypes.c_int, d, ctypes.c_int]
        self.lib.cp2k_get_potential_energy.argtypes = [ctypes.c_int, d]
        self.env = ctypes.c_int(0)
        self.lib.cp2k_init()
        self.lib.cp2k_create_force_env(ctypes.byref(self.env), str(inp).encode(),
                                       str(out).encode())
        natom = ctypes.c_int(0)
        self.lib.cp2k_get_natom(self.env, ctypes.byref(natom))
        self.natom = natom.value

    @staticmethod
    def ptr(a: np.ndarray):
        return a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    def step(self, x_nm: np.ndarray, box_nm: np.ndarray, layout: str = "2026"):
        """Energy (kJ/mol) and forces (kJ/(mol nm)) as GROMACS computes them."""
        x = np.ascontiguousarray(np.asarray(x_nm, dtype=float) / dc.C_BOHR2NM).reshape(-1)
        box = np.asarray(box_nm, dtype=float) / dc.C_BOHR2NM
        cell = np.ascontiguousarray(box if layout == "2026" else box.T).reshape(-1)
        self.lib.cp2k_set_positions(self.env, self.ptr(x), x.size)
        self.lib.cp2k_set_cell(self.env, self.ptr(cell))
        self.lib.cp2k_calc_energy_force(self.env)
        e = ctypes.c_double(0.0)
        self.lib.cp2k_get_potential_energy(self.env, ctypes.byref(e))
        f = np.zeros(x.size)
        self.lib.cp2k_get_forces(self.env, self.ptr(f), f.size)
        back = np.zeros(x.size)
        self.lib.cp2k_get_positions(self.env, self.ptr(back), back.size)
        assert np.array_equal(back, x), "cp2k_get_positions must return what was set"
        return (e.value * dc.C_HARTREE2KJ * dc.C_AVOGADRO,
                f.reshape(-1, 3) * dc.C_HARTREEBOHR2MD)

    def close(self):
        self.lib.cp2k_destroy_force_env(self.env)
        self.lib.cp2k_finalize()


def reference(spec, symbols, x_nm, box_nm):
    """rgpot-eval on the frame, in kJ/mol and kJ/(mol nm)."""
    e, f = dc.rgpot_eval(EVAL, spec, symbols, dc.wrap(x_nm, box_nm) * 10, box_nm * 10)
    return e * dc.KJ_PER_EV, f * dc.KJ_PER_EV * 10


def assert_close(e, f, e_ref, f_ref, rel=1e-12):
    assert abs(e - e_ref) <= rel * max(abs(e_ref), 1.0), (e, e_ref)
    assert np.max(np.abs(f - f_ref)) <= rel * max(np.max(np.abs(f_ref)), 1.0)


def test_version():
    lib = ctypes.CDLL(LIB)
    buf = ctypes.create_string_buffer(100)
    lib.cp2k_get_version(buf, 100)
    assert buf.value.decode().startswith("rgpot ")


@pytest.mark.parametrize("layout", ["2025", "2026"])
def test_triclinic_platinum_matches_rgpot_eval(tmp_path, layout):
    x, box = dc.fcc(0.3924, 3, jiggle=0.005, tilt=(0.08, 0.05, -0.06))
    symbols = ["Pt"] * len(x)
    shim = Shim(write_input(tmp_path, "    POTENTIAL SuttonChen\n    OPTIONS preset platinum cutoff 5.0",
                            symbols, x), tmp_path / "out.log")
    e, f = shim.step(x, box, layout)
    shim.close()
    assert_close(e, f, *reference(PT, symbols, x, box))
    log = (tmp_path / "out.log").read_text()
    assert "potential SuttonChen" in log and "cell PERIODIC" in log


def test_wrapped_positions_give_the_same_energy(tmp_path):
    x, box = dc.fcc(0.3924, 3, jiggle=0.005, tilt=(0.08, 0.05, -0.06))
    symbols = ["Pt"] * len(x)
    shim = Shim(write_input(tmp_path, "    POTENTIAL SuttonChen\n    OPTIONS preset platinum cutoff 5.0",
                            symbols, x), tmp_path / "out.log")
    e0, f0 = shim.step(x, box)
    shifted = x + np.array([2, -1, 1]) @ box
    e1, f1 = shim.step(shifted, box)
    shim.close()
    assert_close(e1, f1, e0, f0, rel=1e-11)


def test_qm_subset_forces_land_on_qm_atoms(tmp_path):
    rng = np.random.default_rng(7)
    box = np.eye(3) * 2.2
    grid = np.array([[i, j, k] for i in range(4) for j in range(4) for k in range(4)]) * 0.55
    x = dc.wrap(grid + rng.uniform(-0.03, 0.03, grid.shape), box)
    symbols = ["Ar"] * len(x)
    qm = list(range(0, len(x), 2))
    shim = Shim(write_input(tmp_path, "    POTENTIAL LJ\n    OPTIONS u0 0.0104 psi 3.4 cutoff 8.0",
                            symbols, x, qm=qm), tmp_path / "out.log")
    e, f = shim.step(x, box)
    shim.close()
    e_ref, f_ref = reference(LJ, [symbols[i] for i in qm], x[qm], box)
    assert_close(e, f[qm], e_ref, f_ref)
    mm = [i for i in range(len(x)) if i not in qm]
    assert np.all(f[mm] == 0.0)


def test_cell_none_makes_a_split_cluster_whole(tmp_path):
    rng = np.random.default_rng(11)
    box = np.eye(3) * 3.0
    cluster = rng.uniform(-0.45, 0.45, (13, 3))
    keep = [cluster[0]]
    for row in cluster[1:]:
        if all(np.linalg.norm(row - k) > 0.3 for k in keep):
            keep.append(row)
    cluster = np.array(keep)
    symbols = ["Ar"] * len(cluster)
    shim = Shim(write_input(tmp_path, "    POTENTIAL LJ\n    OPTIONS u0 0.0104 psi 3.4 cutoff 8.0\n"
                            "    CELL NONE", symbols, cluster), tmp_path / "out.log")
    # Centred on a box corner, GROMACS hands the atoms over wrapped.
    e0, f0 = shim.step(dc.wrap(cluster, box), box)
    # The next step moves every atom a little; some cross the boundary.
    moved = cluster + 0.01
    e1, f1 = shim.step(dc.wrap(moved, box), box)
    shim.close()
    big = np.eye(3) * 10.0
    for e, f, xs in ((e0, f0, cluster), (e1, f1, moved)):
        e_ref, f_ref = reference(LJ, symbols, xs + 5.0, big)
        assert_close(e, f, e_ref, f_ref, rel=1e-11)


def test_conversion_factors_match_gromacs():
    # GROMACS turns the Hartree back into kJ/mol with c_hartree2Kj * c_avogadro;
    # the library must hand back exactly the eV times rgpot's kJ/mol per eV.
    hartree_per_ev = dc.KJ_PER_EV / (dc.C_HARTREE2KJ * dc.C_AVOGADRO)
    assert abs(hartree_per_ev * dc.C_HARTREE2KJ * dc.C_AVOGADRO / dc.KJ_PER_EV - 1) < 1e-15
    # CODATA 2018: 4.3597447222071e-18 J and 0.529177210903 Angstrom.
    assert abs(dc.C_HARTREE2KJ * 1e3 / 4.3597447222071e-18 - 1) < 1e-12
    assert dc.C_BOHR2NM * 10 == 0.529177210903
    assert abs(dc.KJ_PER_EV - 96.48533212331002) < 1e-12


@pytest.mark.skipif(not (ENGINE and SCHEMA and shutil.which("capnp")),
                    reason="needs RGPOT_CLASSICAL_ENGINE, RGPOT_CAPNP_SCHEMA and capnp")
def test_engine_route_matches_the_engine(tmp_path):
    config = tmp_path / "tersoff.bin"
    config.write_bytes(subprocess.run(["capnp", "encode", SCHEMA, "BuiltinParams"],
                                      input=b'(name = "tersoff")', capture_output=True,
                                      check=True).stdout)
    x, box = dc.diamond(0.5431, 2, jiggle=0.004)
    symbols = ["Si"] * len(x)
    shim = Shim(write_input(tmp_path, f"    ENGINE {ENGINE}\n    CONFIG {config}", symbols, x),
                tmp_path / "out.log")
    e, f = shim.step(x, box)
    shim.close()
    e_ref, f_ref = engine_force(config, symbols, dc.wrap(x, box) * 10, box * 10)
    assert_close(e, f, e_ref * dc.KJ_PER_EV, f_ref * dc.KJ_PER_EV * 10)


def engine_force(config: Path, symbols, x_ang, box_ang):
    """rgpot_engine_force on the engine library itself."""
    lib = ctypes.CDLL(ENGINE)
    lib.rgpot_engine_create.restype = ctypes.c_void_p
    lib.rgpot_engine_create.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_char_p,
                                        ctypes.c_size_t]
    d = ctypes.POINTER(ctypes.c_double)
    lib.rgpot_engine_force.argtypes = [ctypes.c_void_p, ctypes.c_long, d,
                                       ctypes.POINTER(ctypes.c_int), d, d, d, d,
                                       ctypes.c_void_p, ctypes.c_void_p]
    lib.rgpot_engine_destroy.argtypes = [ctypes.c_void_p]
    blob = config.read_bytes()
    why = ctypes.create_string_buffer(512)
    pot = lib.rgpot_engine_create(blob, len(blob), why, 512)
    assert pot, why.value
    x = np.ascontiguousarray(x_ang, dtype=float)
    z = (ctypes.c_int * len(symbols))(*[dc.SYMBOLS.index(s) for s in symbols])
    f = np.zeros_like(x)
    box = np.ascontiguousarray(box_ang, dtype=float)
    e = ctypes.c_double(0.0)
    assert lib.rgpot_engine_force(pot, len(symbols), Shim.ptr(x), z, Shim.ptr(f),
                                  ctypes.byref(e), None, Shim.ptr(box), None, None) == 0
    lib.rgpot_engine_destroy(pot)
    return e.value, f


# --- refused inputs ------------------------------------------------------

def refused(tmp_path: Path, rgpot: str | None, *, charges=None, charge=0, multiplicity=1,
            raw: str | None = None, layout_cell=None) -> str:
    """Runs create and one step in a subprocess; returns its stderr."""
    x, box = dc.fcc(0.3924, 2)
    symbols = ["Pt"] * len(x)
    qm = list(range(len(x) - 2)) if charges else None
    inp = write_input(tmp_path, rgpot or "", symbols, x, qm=qm, charges=charges, charge=charge,
                      multiplicity=multiplicity)
    if raw is not None:
        inp.write_text(raw.replace("PDB_PATH", str(tmp_path / "rgpot.pdb")))
    cell = box if layout_cell is None else layout_cell
    np.save(tmp_path / "x.npy", x)
    np.save(tmp_path / "box.npy", cell)
    done = subprocess.run([sys.executable, __file__, "step", str(inp), str(tmp_path)],
                          capture_output=True, text=True, check=False,
                          env=dict(os.environ, PYTHONPATH=str(Path(__file__).parent)))
    assert done.returncode == 1, (done.returncode, done.stdout, done.stderr)
    assert "rgpot-gromacs: " in done.stderr
    log = (tmp_path / "out.log").read_text() if (tmp_path / "out.log").exists() else ""
    assert done.stderr.split("rgpot-gromacs: ", 1)[1].strip() in log
    return done.stderr


@pytest.mark.parametrize("rgpot, why", [
    ("    POTENTIAL Nope", "unknown rgpot potential Nope"),
    ("    POTENTIAL SuttonChen\n    OPTIONS preset platinum colour blue", "takes no option colour"),
    ("    POTENTIAL SuttonChen\n    OPTIONS preset", "OPTIONS takes key value pairs"),
    ("    POTENTIAL SuttonChen\n    OPTIONS preset copper", "preset is platinum or gold"),
    ("    POTENTIAL LJ\n    ENGINE libnope.so", "exactly one of POTENTIAL and ENGINE"),
    ("    OPTIONS u0 1", "exactly one of POTENTIAL and ENGINE"),
    ("    POTENTIAL LJ\n    CONFIG params.bin", "CONFIG belongs to ENGINE"),
    ("    ENGINE libnope_engine.so", "cannot open ENGINE libnope_engine.so"),
    ("    POTENTIAL LJ\n    CELL SOMETIMES", "CELL is PERIODIC or NONE"),
    ("    POTENTIAL LJ\n    EMBEDDING ELECTROSTATIC", "EMBEDDING ELECTROSTATIC is not implemented"),
    ("    POTENTIAL LJ\n    POTENTIAL Morse", "POTENTIAL is given twice"),
    ("    POTENTIAL LJ\n    SPEED fast", "unknown &RGPOT keyword SPEED"),
    ("    POTENTIAL LJ\n    &KIND", "takes no subsections"),
])
def test_bad_rgpot_section_fails_loudly(tmp_path, rgpot, why):
    assert why in refused(tmp_path, rgpot)


def test_input_without_rgpot_section_is_refused(tmp_path):
    raw = dc.qmi("", pdb="PDB_PATH").replace("  &RGPOT\n\n  &END RGPOT\n", "")
    assert "has no &RGPOT section" in refused(tmp_path, None, raw=raw)


def test_link_section_is_refused(tmp_path):
    raw = dc.qmi("    POTENTIAL LJ", pdb="PDB_PATH") + "&LINK\n&END LINK\n"
    assert "has a &LINK section" in refused(tmp_path, None, raw=raw)


def test_charged_mm_atoms_need_embedding_none(tmp_path):
    why = refused(tmp_path, "    POTENTIAL LJ", charges={30: 0.5, 31: -0.5})
    assert "2 MM atoms carry charges" in why and "EMBEDDING NONE" in why


def test_charge_on_a_classical_potential_is_refused(tmp_path):
    assert "CHARGE 0 and MULTIPLICITY 1" in refused(tmp_path, "    POTENTIAL LJ", charge=-1)


def test_non_triangular_cell_is_refused(tmp_path):
    cell = np.array([[1.0, 0.1, 0.0], [0.1, 1.0, 0.0], [0.0, 0.0, 1.0]])
    assert "neither lower nor upper triangular" in refused(tmp_path, "    POTENTIAL LJ",
                                                           layout_cell=cell)


def test_charged_mm_atoms_run_with_embedding_none(tmp_path):
    x, box = dc.fcc(0.3924, 3, jiggle=0.005)
    symbols = ["Pt"] * len(x)
    n = len(x)
    qm = list(range(n - 2))
    inp = write_input(tmp_path, "    POTENTIAL SuttonChen\n    OPTIONS preset platinum cutoff 5.0\n"
                      "    EMBEDDING NONE", symbols, x, qm=qm, charges={n - 2: 0.5, n - 1: -0.5})
    shim = Shim(inp, tmp_path / "out.log")
    e, f = shim.step(x, box)
    shim.close()
    e_ref, f_ref = reference(PT, symbols[:n - 2], x[qm], box)
    assert_close(e, f[qm], e_ref, f_ref)


if __name__ == "__main__" and sys.argv[1:2] == ["step"]:
    inp, work = Path(sys.argv[2]), Path(sys.argv[3])
    shim = Shim(inp, work / "out.log")
    shim.step(np.load(work / "x.npy"), np.load(work / "box.npy"))
    shim.close()
    sys.exit(0)
