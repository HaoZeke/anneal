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
STUB_ENGINE = os.environ.get("RGPOT_ENGINE_STUB", "")

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


def stub_config(path: Path, *, reentrancy: int = 0, no_hooks: int = 0, model: int = 0,
                with_hvp: int = 1, spin: int | None = None) -> Path:
    """The 8-byte config tests/engine_stub.cc reads.

    ``spin`` is an explicit engine spin. 0 is a real value. Omit it when the
    config does not set one; that is not the same as spin 0.
    """
    if spin is not None and not 0 <= int(spin) <= 255:
        raise ValueError(spin)
    flag = 0 if spin is None else 1
    value = 0 if spin is None else int(spin)
    path.write_bytes(bytes([reentrancy, no_hooks, model, with_hvp, flag, value, 0, 0]))
    return path


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


def _bare_engine(path: Path, *, refuses_charge_spin: bool = False) -> Path:
    """An engine library with no setter, or one that rejects charge/spin."""
    cc = shutil.which("cc") or shutil.which("gcc")
    if cc is None:
        pytest.skip("needs a C compiler")
    src = path.parent / "bare_engine.c"
    refusal = (
        "#include <string.h>\n"
        "int rgpot_engine_set_charge_spin(RgpotEnginePot *pot, int charge, int spin) {\n"
        "  (void)pot; (void)charge; (void)spin; return 3;\n"
        "}\n"
        "size_t rgpot_engine_last_error(const RgpotEnginePot *pot, char *buf, size_t len) {\n"
        "  const char *s = \"spin multiplicity must be >= 1\";\n"
        "  size_t n = strlen(s), m = n;\n"
        "  (void)pot;\n"
        "  if (buf && len) { if (m >= len) m = len - 1; memcpy(buf, s, m); buf[m] = 0; }\n"
        "  return n;\n"
        "}\n"
        if refuses_charge_spin else ""
    )
    src.write_text(
        "#include <stddef.h>\n"
        "typedef struct RgpotEnginePot RgpotEnginePot;\n"
        "int rgpot_engine_abi_version(void) { return 1; }\n"
        "RgpotEnginePot *rgpot_engine_create(const void *config, size_t config_len,\n"
        "                                     char *errbuf, size_t errlen) {\n"
        "  (void)config; (void)config_len; (void)errbuf; (void)errlen;\n"
        "  return (RgpotEnginePot *)1;\n"
        "}\n"
        "void rgpot_engine_destroy(RgpotEnginePot *pot) { (void)pot; }\n"
        "int rgpot_engine_force(RgpotEnginePot *pot, long n_atoms, const double *positions,\n"
        "                       const int *atomic_nrs, double *forces, double *energy,\n"
        "                       double *variance, const double *box, void *transform,\n"
        "                       void *transform_user) {\n"
        "  (void)pot; (void)n_atoms; (void)positions; (void)atomic_nrs; (void)forces;\n"
        "  (void)energy; (void)variance; (void)box; (void)transform; (void)transform_user;\n"
        "  return 0;\n"
        "}\n"
        + refusal
    )
    done = subprocess.run(
        [cc, "-shared", "-fPIC", "-o", str(path), str(src)],
        capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr
    return path


def test_engine_without_charge_spin_refuses_a_nondefault_charge(tmp_path):
    so = _bare_engine(tmp_path / "libbare_engine.so")
    why = refused(tmp_path, f"    ENGINE {so}", charge=-1)
    assert "CHARGE 0 and MULTIPLICITY 1" in why
    assert "takes no charge or spin" in why


def test_engine_spin_validation_failure_reports_the_backend_reason(tmp_path):
    so = _bare_engine(tmp_path / "librefusing_engine.so", refuses_charge_spin=True)
    why = refused(tmp_path, f"    ENGINE {so}", multiplicity=0)
    assert "refused CHARGE 0 and spin 0 from input MULTIPLICITY 0 (status 3)" in why
    assert "spin multiplicity must be >= 1" in why


def _stub_charge_spin(work: Path, *, spin: int | None, charge: int, multiplicity: int):
    """Run one engine step and return the charge and spin the stub recorded."""
    work.mkdir(parents=True, exist_ok=True)
    symbols = ["Ar", "Ar"]
    x = np.array([[0.1, 0.2, 0.3], [0.6, 0.5, 0.4]], dtype=float)
    box = np.eye(3) * 1.5
    config = stub_config(work / "stub.bin", spin=spin)
    stub = ctypes.CDLL(STUB_ENGINE)
    stub.stub_reset()
    inp = write_input(work, f"    ENGINE {STUB_ENGINE}\n    CONFIG {config}",
                      symbols, x, charge=charge, multiplicity=multiplicity)
    shim = Shim(inp, work / "out.log")
    shim.step(x, box)
    shim.close()
    return stub.stub_charge(), stub.stub_spin()


@pytest.mark.skipif(not STUB_ENGINE, reason="needs RGPOT_ENGINE_STUB")
def test_engine_route_keeps_explicit_engine_spin_when_qmi_uses_defaults(tmp_path):
    # CHARGE 0 and MULTIPLICITY 1 are left on the CONFIG. The call is skipped
    # and the explicit spin stays.
    assert _stub_charge_spin(tmp_path, spin=4, charge=0, multiplicity=1) == (0, 4)


@pytest.mark.skipif(not STUB_ENGINE, reason="needs RGPOT_ENGINE_STUB")
def test_engine_route_keeps_explicit_spin_0_when_qmi_sets_only_the_charge(tmp_path):
    # MULTIPLICITY 1 is the grompp default even beside CHARGE -1. The charge
    # is applied and the explicit spin 0 stays.
    assert _stub_charge_spin(tmp_path, spin=0, charge=-1, multiplicity=1) == (-1, 0)


@pytest.mark.skipif(not STUB_ENGINE, reason="needs RGPOT_ENGINE_STUB")
def test_engine_route_nondefault_multiplicity_overrides_configured_spin(tmp_path):
    assert _stub_charge_spin(tmp_path, spin=0, charge=-1, multiplicity=3) == (-1, 3)


@pytest.mark.skipif(not STUB_ENGINE, reason="needs RGPOT_ENGINE_STUB")
def test_engine_route_without_explicit_spin_uses_the_qmi_values(tmp_path):
    # No byte-4 flag: the zero the stub starts from is not a configured spin.
    # The qmi charge and multiplicity are both applied, including a
    # multiplicity of 1.
    assert _stub_charge_spin(tmp_path / "one", spin=None, charge=-1, multiplicity=1) == (-1, 1)
    assert _stub_charge_spin(tmp_path / "both", spin=None, charge=2, multiplicity=4) == (2, 4)


def _named_engine(path: Path) -> Path:
    """An engine named by RGPOT_FAKE_ENGINE_NAME that records set_charge_spin.

    Like the real engines, it refuses a spin below 1 unless it is named uma.
    """
    cc = shutil.which("cc") or shutil.which("gcc")
    if cc is None:
        pytest.skip("needs a C compiler")
    src = path.parent / "named_engine.c"
    src.write_text(
        "#include <stddef.h>\n"
        "#include <stdlib.h>\n"
        "#include <string.h>\n"
        "typedef struct RgpotEnginePot RgpotEnginePot;\n"
        "static int g_charge = -7, g_spin = -7, g_calls = 0;\n"
        "static const char *g_error = \"\";\n"
        "int rgpot_engine_abi_version(void) { return 1; }\n"
        "const char *rgpot_engine_name(void) {\n"
        "  const char *name = getenv(\"RGPOT_FAKE_ENGINE_NAME\");\n"
        "  return name != 0 ? name : \"uma\";\n"
        "}\n"
        "RgpotEnginePot *rgpot_engine_create(const void *config, size_t config_len,\n"
        "                                     char *errbuf, size_t errlen) {\n"
        "  (void)config; (void)config_len; (void)errbuf; (void)errlen;\n"
        "  return (RgpotEnginePot *)1;\n"
        "}\n"
        "void rgpot_engine_destroy(RgpotEnginePot *pot) { (void)pot; }\n"
        "int rgpot_engine_force(RgpotEnginePot *pot, long n_atoms, const double *positions,\n"
        "                       const int *atomic_nrs, double *forces, double *energy,\n"
        "                       double *variance, const double *box, void *transform,\n"
        "                       void *transform_user) {\n"
        "  (void)pot; (void)positions; (void)atomic_nrs; (void)box;\n"
        "  (void)transform; (void)transform_user;\n"
        "  if (forces) { for (long i = 0; i < 3 * n_atoms; ++i) forces[i] = 0.0; }\n"
        "  if (energy) *energy = 0.0;\n"
        "  if (variance) *variance = 0.0;\n"
        "  return 0;\n"
        "}\n"
        "int rgpot_engine_set_charge_spin(RgpotEnginePot *pot, int charge, int spin) {\n"
        "  const char *name = rgpot_engine_name();\n"
        "  int min_spin = strcmp(name, \"uma\") == 0 ? 0 : 1;\n"
        "  (void)pot; g_calls += 1;\n"
        "  if (spin < min_spin) {\n"
        "    g_error = strcmp(name, \"skala\") == 0\n"
        "      ? \"skala spin multiplicity must be >= 1\"\n"
        "      : \"metatomic spin multiplicity must be >= 1\";\n"
        "    return 3;\n"
        "  }\n"
        "  g_charge = charge; g_spin = spin; g_error = \"\"; return 0;\n"
        "}\n"
        "size_t rgpot_engine_last_error(const RgpotEnginePot *pot, char *buf, size_t len) {\n"
        "  size_t n = strlen(g_error), m = n;\n"
        "  (void)pot;\n"
        "  if (buf && len) { if (m >= len) m = len - 1; memcpy(buf, g_error, m); buf[m] = 0; }\n"
        "  return n;\n"
        "}\n"
        "void recorded_reset(void) {\n"
        "  g_charge = -7; g_spin = -7; g_calls = 0; g_error = \"\";\n"
        "}\n"
        "int recorded_charge(void) { return g_charge; }\n"
        "int recorded_spin(void) { return g_spin; }\n"
        "int recorded_calls(void) { return g_calls; }\n"
    )
    done = subprocess.run(
        [cc, "-shared", "-fPIC", "-o", str(path), str(src)],
        capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr
    return path


SCHEMA_PATH = Path(SCHEMA) if SCHEMA else None


def _encode_config(tmp_path: Path, schema: str, text: str) -> Path:
    blob = subprocess.run(["capnp", "encode", str(SCHEMA_PATH), schema],
                          input=text.encode(), capture_output=True, check=True).stdout
    config = tmp_path / "params.bin"
    config.write_bytes(blob)
    return config


@pytest.mark.skipif(not shutil.which("capnp") or not SCHEMA_PATH or not SCHEMA_PATH.is_file(),
                    reason="needs capnp and the built effective Potentials.capnp")
@pytest.mark.parametrize("engine,schema,text,charge,multiplicity,expect", [
    pytest.param("uma", "UmaParams", '(spin = 0, cacheDir = "unused")',
                 -1, 1, (-1, 0), id="uma-effective-schema-spin-zero"),
    ("uma", "UmaParams", "(spin = 0)", -1, 3, (-1, 3)),
    ("uma", "UmaParams", "()", -1, 1, (-1, 1)),
    ("uma", "UmaParams", "(spin = 2)", 0, 1, None),
    ("metatomic", "MetatomicParams", "()", -1, 1, (-1, 1)),
    ("metatomic", "MetatomicParams", "(spin = 3)", 2, 1, (2, 3)),
    ("skala", "NWChemParams", "()", -1, 1, (-1, 1)),
    ("skala", "NWChemParams", "(multiplicity = 3)", -1, 1, (-1, 3)),
    ("skala", "NWChemParams", "(multiplicity = 5)", 0, 1, None),
    ("nwchem", "NWChemParams", "()", -1, 4, (-1, 4)),
])
def test_engine_route_reads_an_explicit_spin_from_the_config_message(
        tmp_path, monkeypatch, engine, schema, text, charge, multiplicity, expect):
    """The same rule as the stub, read from the engine's own params message."""
    monkeypatch.setenv("RGPOT_FAKE_ENGINE_NAME", engine)
    config = _encode_config(tmp_path, schema, text)
    so = _named_engine(tmp_path / "libnamed_engine.so")
    lib = ctypes.CDLL(so)
    lib.recorded_reset()
    symbols = ["Ar", "Ar"]
    x = np.array([[0.1, 0.2, 0.3], [0.6, 0.5, 0.4]], dtype=float)
    box = np.eye(3) * 1.5
    inp = write_input(tmp_path, f"    ENGINE {so}\n    CONFIG {config}", symbols, x,
                      charge=charge, multiplicity=multiplicity)
    shim = Shim(inp, tmp_path / "out.log")
    shim.step(x, box)
    shim.close()
    if expect is None:
        assert lib.recorded_calls() == 0
    else:
        assert lib.recorded_calls() == 1
        assert (lib.recorded_charge(), lib.recorded_spin()) == expect


@pytest.mark.skipif(not shutil.which("capnp") or not SCHEMA_PATH or not SCHEMA_PATH.is_file(),
                    reason="needs capnp and the built effective Potentials.capnp")
@pytest.mark.parametrize("engine,schema,text,spin,minimum,multiplicity", [
    ("metatomic", "MetatomicParams", "(spin = 0)", 0, 1, 3),
    ("metatomic", "MetatomicParams", "(spin = -2)", -2, 1, 1),
    ("skala", "NWChemParams", "(multiplicity = 0)", 0, 1, 3),
    ("nwchem", "NWChemParams", "(multiplicity = 0)", 0, 1, 1),
    ("uma", "UmaParams", "(spin = -1)", -1, 0, 1),
])
def test_engine_route_refuses_a_config_spin_below_the_layout_minimum(
        tmp_path, monkeypatch, engine, schema, text, spin, minimum, multiplicity):
    monkeypatch.setenv("RGPOT_FAKE_ENGINE_NAME", engine)
    config = _encode_config(tmp_path, schema, text)
    so = _named_engine(tmp_path / "libnamed_engine.so")
    why = refused(tmp_path, f"    ENGINE {so}\n    CONFIG {config}", charge=-1,
                  multiplicity=multiplicity)
    assert f"the CONFIG message sets spin {spin}" in why
    assert f"the {engine} engine needs a spin of at least {minimum}" in why
    assert "refused CHARGE" not in why


@pytest.mark.skipif(not shutil.which("capnp") or not SCHEMA_PATH or not SCHEMA_PATH.is_file(),
                    reason="needs capnp and the built effective Potentials.capnp")
@pytest.mark.parametrize("engine,schema", [
    ("metatomic", "MetatomicParams"),
    ("skala", "NWChemParams"),
])
def test_engine_route_reports_family_rejection_of_qmi_spin_zero(
        tmp_path, monkeypatch, engine, schema):
    monkeypatch.setenv("RGPOT_FAKE_ENGINE_NAME", engine)
    config = _encode_config(tmp_path, schema, "()")
    so = _named_engine(tmp_path / "libnamed_engine.so")
    why = refused(tmp_path, f"    ENGINE {so}\n    CONFIG {config}",
                  charge=-1, multiplicity=0)
    assert "refused CHARGE -1 and spin 0 from input MULTIPLICITY 0 (status 3)" in why
    assert f"{engine} spin multiplicity must be >= 1" in why


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
