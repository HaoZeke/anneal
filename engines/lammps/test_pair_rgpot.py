"""pair_style rgpot against native LAMMPS styles and rgpot itself.

Meson runs this file with pytest when the plugin is built
(-Dlammps_src and -Dlammps_exe). The environment names the inputs:

    RGPOT_PAIR_PLUGIN  libpair_rgpot.so
    RGPOT_LMP          the lmp built from lammps_src; its liblammps is
                       preloaded so the Python module matches the plugin
    RGPOT_EVAL         rgpot-eval
    RGPOT_LJ_ENGINE    liblj_engine.so
    RGPOT_EAM_ALLOY    CuNi.eam.alloy
    RGPOT_MPIRUN       mpirun; the MPI cases are skipped without it

Every LAMMPS case runs in a subprocess through lmp_case.py.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
CASE = HERE / "lmp_case.py"

PLUGIN = os.environ.get("RGPOT_PAIR_PLUGIN", "")
LMP = os.environ.get("RGPOT_LMP", "")
EVAL = os.environ.get("RGPOT_EVAL", "")
LJ_ENGINE = os.environ.get("RGPOT_LJ_ENGINE", "")
EAM_ALLOY = os.environ.get("RGPOT_EAM_ALLOY", "")
MPIRUN = os.environ.get("RGPOT_MPIRUN", "")

pytestmark = pytest.mark.skipif(
    not (PLUGIN and LMP), reason="needs RGPOT_PAIR_PLUGIN and RGPOT_LMP"
)

# Argon-like Lennard-Jones in metal units.
U0 = 0.0104
PSI = 3.40
CUT = 8.5
A_LJ = 5.26
A_CUNI = 3.6

THERMO = [
    "compute rgpress all pressure NULL virial",
    "thermo_style custom step pe press "
    + " ".join(f"c_rgpress[{k}]" for k in range(1, 7)),
]


def liblammps_env() -> dict:
    env = os.environ.copy()
    env.pop("LD_PRELOAD", None)
    exe = Path(LMP).resolve()
    for libdir in (exe.parent, exe.parent.parent / "lib"):
        lib = libdir / "liblammps.so"
        if lib.exists():
            env["LD_PRELOAD"] = str(lib)
            env["LD_LIBRARY_PATH"] = ":".join(
                p for p in (str(libdir), env.get("LD_LIBRARY_PATH", "")) if p
            )
            break
    return env


def mpirun_prefix(nprocs: int) -> list[str]:
    prefix = [MPIRUN, "-np", str(nprocs)]
    version = subprocess.run(
        [MPIRUN, "--version"], capture_output=True, text=True, check=False
    )
    if "Open MPI" in version.stdout + version.stderr:
        prefix += ["--oversubscribe"]
    return prefix


def run_case(tmp_path: Path, commands: list[str], strains=(), nprocs: int = 1,
             peratom: dict | None = None) -> dict:
    tag = f"case{len(list(tmp_path.glob('case*.json')))}"
    case = tmp_path / f"{tag}.json"
    out = tmp_path / f"{tag}.out.json"
    case.write_text(json.dumps(
        {"commands": commands, "strains": list(strains), "peratom": peratom or {}}
    ))
    cmd = [sys.executable, str(CASE), str(case), str(out)]
    if nprocs > 1:
        cmd = mpirun_prefix(nprocs) + cmd
    proc = subprocess.run(
        cmd, env=liblammps_env(), cwd=tmp_path, capture_output=True, text=True,
        timeout=600, check=False,
    )
    if not out.exists():
        raise AssertionError(f"no result from {cmd}\n{proc.stdout}\n{proc.stderr}")
    result = json.loads(out.read_text())
    counts = [tmp_path / f"{out.name}.nlocal{r}" for r in range(nprocs)]
    if all(c.exists() for c in counts):
        result["nlocal"] = [int(c.read_text()) for c in counts]
    return result


def ok(result: dict) -> dict:
    assert "error" not in result, result.get("error")
    return result


def fcc(a: float, triclinic: bool, extra: list[str] = ()) -> list[str]:
    region = (
        "region box prism 0 4 0 4 0 4 1 1 1"
        if triclinic
        else "region box block 0 4 0 4 0 4"
    )
    lines = [
        "units metal",
        "atom_style atomic",
        "atom_modify map array",
        "boundary p p p",
        f"lattice fcc {a}",
        region,
        "create_box 2 box",
        "create_atoms 1 box",
        "mass * 39.948",
    ]
    if triclinic:
        lines.append("change_box all xy delta 0.4 xz delta -0.3 yz delta 0.2 remap units box")
    lines += jiggle(0.08) + list(extra)
    return lines


def jiggle(amplitude: float) -> list[str]:
    """Displacements that depend on the atom ID only, not on the ranks."""
    return [
        f"variable jx atom {amplitude}*sin(1.37*id)",
        f"variable jy atom {amplitude}*sin(2.11*id+0.5)",
        f"variable jz atom {amplitude}*cos(0.73*id)",
        "displace_atoms all move v_jx v_jy v_jz units box",
    ]


def lj_rgpot(form: str) -> list[str]:
    style = f"pair_style rgpot {CUT}"
    if form == "u0psi":
        coeff = f"pair_coeff * * {U0} {PSI}"
    else:
        coeff = f"pair_coeff * * LJ u0 {U0} psi {PSI} cutoff {CUT} Ar Ar"
    if form == "named-fd":
        style += " virial fd"
    return [f"plugin load {PLUGIN}", style, coeff]


LJ_CUT = [f"pair_style lj/cut {CUT}", "pair_modify shift yes", f"pair_coeff * * {U0} {PSI}"]


def max_abs_diff(a, b) -> float:
    return max(abs(x - y) for x, y in zip(a, b, strict=True))


def lower_strains(h: float) -> list[list[list[float]]]:
    """+h and -h for xx, yy, zz and the lower entries yx, zx, zy."""
    out = []
    for i, j in ((0, 0), (1, 1), (2, 2), (1, 0), (2, 0), (2, 1)):
        for sign in (1.0, -1.0):
            e = [[0.0] * 3 for _ in range(3)]
            e[i][j] = sign * h
            out.append(e)
    return out


def fd_pressure(result: dict, h: float) -> list[float]:
    """-(1/V) dE/de nktv2p, ordered xx yy zz xy xz yz like compute pressure."""
    energies = result["strained"]
    scale = -result["nktv2p"] / result["volume"]
    return [scale * (energies[2 * k] - energies[2 * k + 1]) / (2 * h) for k in range(6)]


@pytest.mark.parametrize("triclinic", [False, True], ids=["orthogonal", "triclinic"])
@pytest.mark.parametrize("form", ["u0psi", "named", "named-fd"])
def test_lj_matches_lj_cut(tmp_path, triclinic, form):
    ref = ok(run_case(tmp_path, fcc(A_LJ, triclinic) + LJ_CUT + THERMO + ["run 0"]))
    got = ok(run_case(tmp_path, fcc(A_LJ, triclinic) + lj_rgpot(form) + THERMO + ["run 0"]))
    de = abs(got["pe"] - ref["pe"])
    df = max_abs_diff(got["forces"], ref["forces"])
    dp = max_abs_diff(got["press"], ref["press"])
    print(f"lj/cut {form} triclinic={triclinic}: |dE| {de:.3e} eV, "
          f"max|dF| {df:.3e} eV/A, max|dP| {dp:.3e} bar, P {ref['press']}")
    assert de < 1e-9
    assert df < 1e-9
    # The strain difference has a truncation and a rounding error.
    rtol = 1e-5 if form == "named-fd" else 1e-6
    assert dp < rtol * max(1.0, max(abs(p) for p in ref["press"]))
    if triclinic:
        assert max(abs(p) for p in ref["press"][3:]) > 10.0


@pytest.mark.parametrize("triclinic", [False, True], ids=["orthogonal", "triclinic"])
def test_lj_pressure_is_the_strain_derivative(tmp_path, triclinic):
    h = 1e-6
    got = ok(run_case(
        tmp_path, fcc(A_LJ, triclinic) + lj_rgpot("named") + THERMO + ["run 0"],
        strains=lower_strains(h),
    ))
    fd = fd_pressure(got, h)
    dp = max_abs_diff(got["press"], fd)
    print(f"strain derivative triclinic={triclinic}: max|dP| {dp:.3e} bar, "
          f"P {got['press']}, FD {fd}")
    assert dp < 1e-3 * max(1.0, max(abs(p) for p in fd))


def cuni(triclinic: bool) -> list[str]:
    return fcc(A_CUNI, triclinic, ["set type 1 type/fraction 2 0.5 9137", "mass * 60.0"])


EAM_RGPOT = [
    f"plugin load {PLUGIN}",
    "pair_style rgpot 6.5",
    f"pair_coeff * * EAMTable file {EAM_ALLOY} Cu Ni",
]
EAM_NATIVE = ["pair_style eam/alloy", f"pair_coeff * * {EAM_ALLOY} Cu Ni"]


def rgpot_eval(result: dict, z_by_type: dict[int, int], spec: list[str], e=None) -> dict:
    """rgpot-eval on the cell and atoms of a LAMMPS result, strained by e."""
    m = [[(1.0 if i == j else 0.0) + (e[i][j] if e else 0.0) for j in range(3)]
         for i in range(3)]

    def apply(r):
        return [sum(r[k] * m[k][j] for k in range(3)) for j in range(3)]

    lines = [str(len(result["types"]))]
    lines.append(" ".join(f"{v:.17g}" for row in result["cell"] for v in apply(row)))
    lo = result["boxlo"]
    for t, x in zip(result["types"], result["x"], strict=True):
        r = apply([x[0] - lo[0], x[1] - lo[1], x[2] - lo[2]])
        lines.append(f"{z_by_type[t]} {r[0]:.17g} {r[1]:.17g} {r[2]:.17g}")
    text = subprocess.run(
        [EVAL, *spec], input="\n".join(lines) + "\n",
        capture_output=True, text=True, check=True,
    ).stdout
    out = {"forces": [], "stress": None}
    for line in text.splitlines():
        parts = line.split()
        if parts[0] == "energy":
            out["pe"] = float(parts[1])
        elif parts[0] == "force":
            out["forces"] += [float(v) for v in parts[2:5]]
        elif parts[0] == "stress":
            out["stress"] = [float(v) for v in parts[1:]]
    return out


@pytest.mark.skipif(not EAM_ALLOY, reason="needs RGPOT_EAM_ALLOY")
@pytest.mark.parametrize("triclinic", [False, True], ids=["orthogonal", "triclinic"])
def test_eam_alloy_element_map(tmp_path, triclinic):
    dump = ["run 0"]
    ref = ok(run_case(tmp_path, cuni(triclinic) + EAM_NATIVE + THERMO + dump))
    got = ok(run_case(tmp_path, cuni(triclinic) + EAM_RGPOT + THERMO + dump))
    de = abs(got["pe"] - ref["pe"])
    df = max_abs_diff(got["forces"], ref["forces"])
    dp = max_abs_diff(got["press"], ref["press"])
    print(f"eam/alloy triclinic={triclinic}: |dE| {de:.3e} eV, max|dF| {df:.3e} eV/A, "
          f"max|dP| {dp:.3e} bar, P {ref['press']}")
    assert de < 1e-8
    assert df < 1e-8
    assert dp < 1e-6 * max(1.0, max(abs(p) for p in ref["press"]))

    if EVAL:
        own = rgpot_eval(got, {1: 29, 2: 28}, ["EAMTable", "file", EAM_ALLOY])
        s = own["stress"]
        virial_p = [-got["nktv2p"] * s[k] for k in (0, 4, 8, 1, 2, 5)]
        de = abs(own["pe"] - got["pe"])
        df = max_abs_diff(own["forces"], got["forces"])
        dp = max_abs_diff(virial_p, got["press"])
        print(f"rgpot-eval triclinic={triclinic}: |dE| {de:.3e} eV, max|dF| {df:.3e} eV/A, "
              f"max|dP| {dp:.3e} bar")
        assert de < 1e-9
        assert df < 1e-9
        assert dp < 1e-6 * max(1.0, max(abs(p) for p in virial_p))


NAMED = {
    # name: (lattice, pair_style cutoff, pair_coeff words, rgpot-eval words)
    "SuttonChen": (3.92, 7.0, "SuttonChen preset platinum cutoff 7.0 Pt Pt",
                   ["SuttonChen", "preset", "platinum", "cutoff", "7.0"]),
    "Morse": (3.92, 7.0, "Morse cutoff 7.0 Pt Pt", ["Morse", "cutoff", "7.0"]),
}


@pytest.mark.skipif(not EVAL, reason="needs RGPOT_EVAL")
@pytest.mark.parametrize("name", sorted(NAMED))
def test_named_potential_matches_rgpot_eval(tmp_path, name):
    lattice, cutoff, coeff, spec = NAMED[name]
    pair = [f"plugin load {PLUGIN}", f"pair_style rgpot {cutoff}", f"pair_coeff * * {coeff}"]
    got = ok(run_case(tmp_path, fcc(lattice, True) + pair + THERMO + ["run 0"]))
    own = rgpot_eval(got, {1: 78}, spec)
    s = own["stress"]
    virial_p = [-got["nktv2p"] * s[k] for k in (0, 4, 8, 1, 2, 5)]
    de = abs(own["pe"] - got["pe"])
    df = max_abs_diff(own["forces"], got["forces"])
    dp = max_abs_diff(virial_p, got["press"])
    print(f"{name} vs rgpot-eval: |dE| {de:.3e} eV, max|dF| {df:.3e} eV/A, "
          f"max|dP| {dp:.3e} bar, P {got['press']}")
    assert de < 1e-9
    assert df < 1e-9
    assert dp < 1e-6 * max(abs(p) for p in virial_p)


@pytest.mark.skipif(not MPIRUN, reason="needs RGPOT_MPIRUN")
@pytest.mark.parametrize("system", ["lj", "eam"])
def test_mpi_ranks_agree(tmp_path, system):
    # create_atoms numbers atoms by rank, so every run reads one data file.
    if system == "lj":
        start, pair = fcc(A_LJ, True), lj_rgpot("named")
    else:
        if not EAM_ALLOY:
            pytest.skip("needs RGPOT_EAM_ALLOY")
        start, pair = cuni(True), EAM_RGPOT
    ok(run_case(tmp_path, start + ["pair_style zero 1.0", "pair_coeff * *", "run 0",
                                   "write_data start.data nocoeff"]))
    commands = [
        "units metal",
        "atom_style atomic",
        "atom_modify map array",
        "boundary p p p",
        "read_data start.data",
        *pair,
        *THERMO,
        "velocity all create 300 7741 dist gaussian",
        "fix 1 all nve",
        "run 20",
    ]
    results = {n: ok(run_case(tmp_path, commands, nprocs=n)) for n in (1, 2, 4)}
    assert [results[n]["nprocs"] for n in (1, 2, 4)] == [1, 2, 4]
    base = results[1]
    for n in (2, 4):
        de = abs(results[n]["pe"] - base["pe"])
        df = max_abs_diff(results[n]["forces"], base["forces"])
        dp = max_abs_diff(results[n]["press"], base["press"])
        print(f"{system} np {n} vs 1 after 20 steps: |dE| {de:.3e} eV, max|dF| {df:.3e} eV/A, "
              f"max|dP| {dp:.3e} bar")
        assert de <= 1e-10 * max(1.0, abs(base["pe"]))
        assert df <= 1e-10
        assert dp <= 1e-8 * max(1.0, max(abs(p) for p in base["press"]))


PER_ATOM = [
    "compute eatom all pe/atom",
    "compute vatom all stress/atom NULL virial",
    "compute cvatom all centroid/stress/atom NULL virial",
]
# centroid/stress/atom adds yx zx zy after the six of stress/atom.
PER_ATOM_COLUMNS = {"eatom": 1, "vatom": 6, "cvatom": 9}


def per_atom_case(tmp_path, start: list[str], pair: list[str], nprocs: int = 1) -> dict:
    return ok(run_case(tmp_path, start + pair + PER_ATOM + THERMO + ["run 0"],
                       nprocs=nprocs, peratom=PER_ATOM_COLUMNS))


def flat(rows) -> list[float]:
    return [v for row in rows for v in row]


def check_per_atom_sums(label: str, got: dict) -> None:
    """pe/atom sums to pe, stress/atom to -P V of the global virial, and
    centroid/stress/atom equals stress/atom."""
    de = abs(sum(flat(got["eatom"])) - got["pe"])
    total = [sum(row[k] for row in got["vatom"]) for k in range(6)]
    want = [-p * got["volume"] for p in got["press"]]
    scale = max(abs(w) for w in want)
    dw = max_abs_diff(total, want)
    dc = max_abs_diff(
        flat(row[:6] + row[3:6] for row in got["vatom"]), flat(got["cvatom"])
    )
    print(f"{label} sums: |sum pe/atom - pe| {de:.3e} eV, "
          f"max|sum stress/atom + PV| {dw:.3e} of {scale:.3e} bar A^3, "
          f"max|centroid - stress/atom| {dc:.3e}")
    assert de < 1e-12 * max(1.0, abs(got["pe"])) * len(got["eatom"])
    assert dw < 1e-10 * scale
    assert dc == 0.0


def per_atom_systems(system: str, triclinic: bool):
    if system == "lj":
        return fcc(A_LJ, triclinic), lj_rgpot("named"), LJ_CUT
    if not EAM_ALLOY:
        pytest.skip("needs RGPOT_EAM_ALLOY")
    return cuni(triclinic), EAM_RGPOT, EAM_NATIVE


@pytest.mark.parametrize("nprocs", [1, 2, 4], ids=["np1", "np2", "np4"])
@pytest.mark.parametrize("triclinic", [False, True], ids=["orthogonal", "triclinic"])
@pytest.mark.parametrize("system", ["lj", "eam"])
def test_per_atom_matches_native(tmp_path, system, triclinic, nprocs):
    """compute pe/atom and stress/atom against lj/cut and eam/alloy atom by
    atom in tag order, with the same ranks for both."""
    if nprocs > 1 and not MPIRUN:
        pytest.skip("needs RGPOT_MPIRUN")
    start, pair, native = per_atom_systems(system, triclinic)
    ref = per_atom_case(tmp_path, start, native, nprocs)
    got = per_atom_case(tmp_path, start, pair, nprocs)
    assert got["nprocs"] == nprocs
    de = max_abs_diff(flat(got["eatom"]), flat(ref["eatom"]))
    dv = max_abs_diff(flat(got["vatom"]), flat(ref["vatom"]))
    scale = max(abs(v) for v in flat(ref["vatom"]))
    print(f"{system} np {nprocs} triclinic={triclinic}: max|d pe/atom| {de:.3e} eV, "
          f"max|d stress/atom| {dv:.3e} of {scale:.3e} bar A^3")
    assert de < 1e-11
    assert dv < 1e-10 * scale
    check_per_atom_sums(f"{system} np {nprocs} triclinic={triclinic}", got)


def test_per_atom_morse_matches_native(tmp_path):
    """The default (platinum) Morse against pair_style morse, shifted."""
    lattice, cutoff, coeff, _ = NAMED["Morse"]
    start = fcc(lattice, True)
    pair = [f"plugin load {PLUGIN}", f"pair_style rgpot {cutoff}", f"pair_coeff * * {coeff}"]
    native = [f"pair_style morse {cutoff}", "pair_modify shift yes",
              "pair_coeff * * 0.7102 1.6047 2.8970"]
    ref = per_atom_case(tmp_path, start, native)
    got = per_atom_case(tmp_path, start, pair)
    de = max_abs_diff(flat(got["eatom"]), flat(ref["eatom"]))
    dv = max_abs_diff(flat(got["vatom"]), flat(ref["vatom"]))
    scale = max(abs(v) for v in flat(ref["vatom"]))
    print(f"morse: max|d pe/atom| {de:.3e} eV, max|d stress/atom| {dv:.3e} of {scale:.3e}")
    assert de < 1e-11
    assert dv < 1e-10 * scale


@pytest.mark.parametrize("name", ["u0psi", "SuttonChen", "Morse"])
def test_per_atom_sums_of_other_forms(tmp_path, name):
    if name == "u0psi":
        start, pair = fcc(A_LJ, True), lj_rgpot("u0psi")
    else:
        lattice, cutoff, coeff, _ = NAMED[name]
        start = fcc(lattice, True)
        pair = [f"plugin load {PLUGIN}", f"pair_style rgpot {cutoff}",
                f"pair_coeff * * {coeff}"]
    check_per_atom_sums(name, per_atom_case(tmp_path, start, pair))


@pytest.mark.parametrize("newton", ["on", "off"])
@pytest.mark.parametrize("mode", ["gather", "domain"])
def test_null_type_under_hybrid_overlay(tmp_path, mode, newton):
    """NULL leaves type 2 out; lj/cut adds the pairs that involve it. On
    the domain path rgpot walks the skip list LAMMPS builds for it."""
    nprocs = 2 if MPIRUN else 1
    start = [f"newton {newton}"] + fcc(A_LJ, True, ["set type 1 type/fraction 2 0.3 77"])
    hybrid = [
        f"plugin load {PLUGIN}",
        f"pair_style hybrid/overlay rgpot {CUT} mode {mode} lj/cut {CUT}",
        "pair_modify shift yes",
        f"pair_coeff * * rgpot LJ u0 {U0} psi {PSI} cutoff {CUT} Ar NULL",
        f"pair_coeff 1 2 lj/cut {U0} {PSI}",
        f"pair_coeff 2 2 lj/cut {U0} {PSI}",
    ]
    got = ok(run_case(tmp_path, start + hybrid + THERMO + ["run 0"], nprocs=nprocs))
    ref = ok(run_case(tmp_path, start + LJ_CUT + THERMO + ["run 0"], nprocs=nprocs))
    de = abs(got["pe"] - ref["pe"])
    df = max_abs_diff(got["forces"], ref["forces"])
    dp = max_abs_diff(got["press"], ref["press"])
    print(f"hybrid/overlay NULL {mode} newton {newton} np {nprocs}: |dE| {de:.3e} eV, "
          f"max|dF| {df:.3e} eV/A, max|dP| {dp:.3e} bar")
    assert de < 1e-9
    assert df < 1e-9
    assert dp < 1e-6 * max(abs(p) for p in ref["press"])


def lj_fcc_zero_pressure_lattice() -> float:
    """a where the static fcc lattice sum of the truncated pair has dE/da = 0."""
    nmax = int(math.ceil(2 * CUT / A_LJ)) + 1
    sites = []
    for i in range(-2 * nmax, 2 * nmax + 1):
        for j in range(-2 * nmax, 2 * nmax + 1):
            for k in range(-2 * nmax, 2 * nmax + 1):
                if (i + j + k) % 2 == 0 and (i, j, k) != (0, 0, 0):
                    sites.append(math.sqrt(i * i + j * j + k * k) / 2.0)

    def virial(a: float) -> float:
        total = 0.0
        for s in sites:
            r = s * a
            if r < CUT:
                sr6 = (PSI / r) ** 6
                total += r * 24.0 * U0 * (2.0 * sr6 * sr6 - sr6) / r
        return total

    lo, hi = 5.0, 5.6
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if virial(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def relax(triclinic: bool, pair: list[str]) -> list[str]:
    lines = [
        "units metal",
        "atom_style atomic",
        "atom_modify map array",
        "boundary p p p",
        "lattice fcc 5.45",
        "region box prism 0 4 0 4 0 4 1 1 1" if triclinic else "region box block 0 4 0 4 0 4",
        "create_box 2 box",
        "create_atoms 1 box",
        "mass * 39.948",
    ]
    if triclinic:
        lines.append("change_box all xy delta 0.5 xz delta -0.4 yz delta 0.3 remap units box")
    lines += pair + THERMO
    lines += [
        f"fix relax all box/relax {'tri' if triclinic else 'iso'} 0.0 vmax 0.001",
        "min_modify line quadratic",
        "minimize 0.0 1.0e-12 20000 200000",
        "unfix relax",
        "run 0",
    ]
    return lines


@pytest.mark.parametrize("triclinic", [False, True], ids=["iso", "tri"])
def test_box_relax_finds_the_lattice(tmp_path, triclinic):
    a0 = lj_fcc_zero_pressure_lattice()
    got = ok(run_case(tmp_path, relax(triclinic, lj_rgpot("named"))))
    ref = ok(run_case(tmp_path, relax(triclinic, LJ_CUT)))
    h = got["cell"]
    a_got = h[0][0] / 4.0
    tilt = [h[1][0] / h[1][1], h[2][0] / h[2][2], h[2][1] / h[2][2]]
    print(f"box/relax triclinic={triclinic}: a {a_got:.8f} A, lattice sum {a0:.8f} A, "
          f"lj/cut {ref['cell'][0][0] / 4.0:.8f} A, tilt/length {tilt}, "
          f"P {got['press']}")
    assert abs(a_got - a0) < 1e-4
    assert max_abs_diff(sum(h, []), sum(ref["cell"], [])) < 1e-6
    assert max(abs(p) for p in got["press"]) < 1.0
    if triclinic:
        assert max(abs(t - 0.25) for t in tilt) < 1e-5


def error_of(tmp_path, commands: list[str]) -> str:
    result = run_case(tmp_path, commands)
    assert "error" in result, "the case ran without an error"
    return result["error"]


def test_refuses_other_units(tmp_path):
    commands = fcc(A_LJ, False) + lj_rgpot("named") + ["run 0"]
    commands[0] = "units real"
    assert "needs units metal" in error_of(tmp_path, commands)


@pytest.mark.parametrize("compute", ["pe/atom", "stress/atom NULL"])
def test_refuses_per_atom_requests_without_them(tmp_path, compute):
    """EMT has no per-atom split; the error names it."""
    commands = fcc(3.61, False) + [
        f"plugin load {PLUGIN}", "pair_style rgpot 6.0", "pair_coeff * * EMT Cu Cu",
        f"compute peratom all {compute}",
        "compute total all reduce sum c_peratom" + ("[1]" if "stress" in compute else ""),
        "thermo_style custom step pe c_total",
        "run 0",
    ]
    message = error_of(tmp_path, commands)
    assert "the rgpot potential EMT has no per-atom energy or virial" in message


@pytest.mark.skipif(not LJ_ENGINE, reason="needs RGPOT_LJ_ENGINE")
def test_refuses_per_atom_requests_from_an_engine(tmp_path):
    commands = dilute_lj([f"plugin load {PLUGIN}", "pair_style rgpot 15.0",
                          f"pair_coeff * * engine {LJ_ENGINE} H"])
    commands[-1:-1] = [
        "compute peratom all pe/atom",
        "compute total all reduce sum c_peratom",
        "thermo_style custom step pe c_total",
    ]
    message = error_of(tmp_path, commands)
    assert f"the engine {LJ_ENGINE} has no per-atom energy or virial" in message


def dilute_lj(pair: list[str]) -> list[str]:
    """A dilute triclinic gas for the engine's default LJ, cutoff 15 A."""
    return [
        "units metal",
        "atom_style atomic",
        "atom_modify map array",
        "boundary p p p",
        "lattice sc 4.0",
        "region box prism 0 8 0 8 0 8 1 0 1",
        "create_box 1 box",
        "create_atoms 1 box",
        "mass 1 1.0",
        "change_box all xy delta 0.6 xz delta -0.4 yz delta 0.3 remap units box",
        *jiggle(0.4),
        *pair,
        *THERMO,
        "run 0",
    ]


@pytest.mark.skipif(not LJ_ENGINE, reason="needs RGPOT_LJ_ENGINE")
def test_engine_virial_from_strain_difference(tmp_path):
    """The engine has no stress; its pressure is the strain difference."""
    engine = [f"plugin load {PLUGIN}", "pair_style rgpot 15.0",
              f"pair_coeff * * engine {LJ_ENGINE} H"]
    named = [f"plugin load {PLUGIN}", "pair_style rgpot 15.0",
             "pair_coeff * * LJ u0 1.0 psi 1.0 cutoff 15.0 H"]
    got = ok(run_case(tmp_path, dilute_lj(engine)))
    ref = ok(run_case(tmp_path, dilute_lj(named)))
    de = abs(got["pe"] - ref["pe"])
    df = max_abs_diff(got["forces"], ref["forces"])
    dp = max_abs_diff(got["press"], ref["press"])
    print(f"engine vs LJ stress: |dE| {de:.3e} eV, max|dF| {df:.3e} eV/A, "
          f"max|dP| {dp:.3e} bar, P {ref['press']}")
    assert de < 1e-10
    assert df < 1e-10
    assert dp < 1e-5 * max(abs(p) for p in ref["press"])


@pytest.mark.skipif(not EVAL, reason="needs RGPOT_EVAL")
def test_potential_without_stress_uses_strain_difference(tmp_path):
    """EMT reports no stress. Its pressure is the same strain difference
    taken outside LAMMPS with rgpot-eval, at the plugin's step 1e-5.
    EMT is not smooth enough for a step-independent comparison."""
    h = 1e-5
    emt = [f"plugin load {PLUGIN}", "pair_style rgpot 6.0", "pair_coeff * * EMT Cu Cu"]
    got = ok(run_case(tmp_path, fcc(3.61, True) + emt + THERMO + ["run 0"]))
    fd = []
    for k in range(6):
        energies = [rgpot_eval(got, {1: 29}, ["EMT"], e)["pe"]
                    for e in lower_strains(h)[2 * k: 2 * k + 2]]
        fd.append(-got["nktv2p"] / got["volume"] * (energies[0] - energies[1]) / (2 * h))
    dp = max_abs_diff(got["press"], fd)
    print(f"EMT strain difference: max|dP| {dp:.3e} bar, P {got['press']}")
    assert dp < 1e-6 * max(abs(p) for p in fd)


def test_refuses_unknown_names_and_symbols(tmp_path):
    base = fcc(A_LJ, False) + [f"plugin load {PLUGIN}", f"pair_style rgpot {CUT}"]
    assert "unknown rgpot potential" in error_of(tmp_path, base + ["pair_coeff * * Nope Ar Ar"])
    assert "not an element symbol" in error_of(
        tmp_path, base + [f"pair_coeff * * LJ u0 {U0} Ar Xx"]
    )
    assert "takes no option" in error_of(
        tmp_path, base + [f"pair_coeff * * LJ depth {U0} Ar Ar"]
    )


def test_refuses_free_boundaries(tmp_path):
    commands = fcc(A_LJ, False) + lj_rgpot("named") + ["run 0"]
    commands[3] = "boundary p p f"
    assert "needs boundary p p p" in error_of(tmp_path, commands)


# The domain path. Every case reads one data file, so atom IDs do not
# depend on the ranks that created the atoms.

def read_data_start(tmp_path: Path, start: list[str], name: str,
                    newton: str = "on") -> list[str]:
    ok(run_case(tmp_path, start + ["pair_style zero 1.0", "pair_coeff * *", "run 0",
                                   f"write_data {name} nocoeff"]))
    return [
        f"newton {newton}",
        "units metal",
        "atom_style atomic",
        "atom_modify map array",
        "boundary p p p",
        "processors * 1 1",
        f"read_data {name}",
    ]


def with_mode(pair: list[str], mode: str) -> list[str]:
    return [f"{line} mode {mode}" if line.startswith("pair_style rgpot") else line
            for line in pair]


MORSE_NATIVE = ["pair_style morse 7.0", "pair_modify shift yes",
                "pair_coeff * * 0.7102 1.6047 2.8970"]


def named_pair(name: str) -> list[str]:
    _, cutoff, coeff, _ = NAMED[name]
    return [f"plugin load {PLUGIN}", f"pair_style rgpot {cutoff}", f"pair_coeff * * {coeff}"]


def domain_systems(system: str, triclinic: bool, fill: str):
    """Start, rgpot pair lines, native pair lines (None for Sutton-Chen,
    which LAMMPS lacks). fill "part" keeps the atoms with x under 0.43 of
    the box, so under processors * 1 1 the upper half of the ranks own
    none."""
    if system == "lj":
        start, pair, native, side = fcc(A_LJ, triclinic), lj_rgpot("named"), LJ_CUT, 4 * A_LJ
    elif system in ("sc", "morse"):
        name = "SuttonChen" if system == "sc" else "Morse"
        lattice = NAMED[name][0]
        start, pair, side = fcc(lattice, triclinic), named_pair(name), 4 * lattice
        native = MORSE_NATIVE if system == "morse" else None
    else:
        if not EAM_ALLOY:
            pytest.skip("needs RGPOT_EAM_ALLOY")
        start, pair, native, side = cuni(triclinic), EAM_RGPOT, EAM_NATIVE, 4 * A_CUNI
    if fill == "part":
        start = start + [
            f"variable far atom x>{0.43 * side!r}",
            "group far variable far",
            "delete_atoms group far",
        ]
    return start, pair, native


def compare(label: str, got: dict, ref: dict, tol: dict) -> None:
    de = abs(got["pe"] - ref["pe"])
    df = max_abs_diff(got["forces"], ref["forces"])
    dp = max_abs_diff(got["press"], ref["press"])
    pscale = max(1.0, max(abs(p) for p in ref["press"]))
    dea = max_abs_diff(flat(got["eatom"]), flat(ref["eatom"]))
    dva = max_abs_diff(flat(got["vatom"]), flat(ref["vatom"]))
    vscale = max(abs(v) for v in flat(ref["vatom"]))
    print(f"{label}: |dE| {de:.3e} eV, max|dF| {df:.3e} eV/A, max|dP| {dp:.3e} of "
          f"{pscale:.3e} bar, max|d pe/atom| {dea:.3e} eV, max|d stress/atom| {dva:.3e} "
          f"of {vscale:.3e} bar A^3")
    assert de < tol["e"] * max(1.0, abs(ref["pe"]))
    assert df < tol["f"]
    assert dp < tol["p"] * pscale
    assert dea < tol["ea"]
    assert dva < tol["va"] * vscale


DOMAIN_CASES = [
    pytest.param(system, triclinic, fill,
                 id=f"{system}-{'triclinic' if triclinic else 'orthogonal'}-{fill}")
    for system in ("lj", "eam") for triclinic in (False, True) for fill in ("full", "part")
] + [pytest.param(system, True, "part", id=f"{system}-triclinic-part")
     for system in ("sc", "morse")]


@pytest.mark.parametrize("newton", ["on", "off"])
@pytest.mark.parametrize("system,triclinic,fill", DOMAIN_CASES)
def test_domain_matches_gather_and_native(tmp_path, system, triclinic, fill, newton):
    """Energy, forces, global virial, pe/atom and stress/atom of the
    domain path against the gather path and the native style, on 1, 2
    and 4 ranks, all against the native style on one rank (the gather
    path for Sutton-Chen), with newton_pair on and off."""
    ranks = (1, 2, 4) if MPIRUN else (1,)
    start, pair, native = domain_systems(system, triclinic, fill)
    head = read_data_start(tmp_path, start, "start.data", newton)
    tail = PER_ATOM + THERMO + ["run 0"]

    def case(lines, n):
        return ok(run_case(tmp_path, head + lines + tail, nprocs=n,
                           peratom=PER_ATOM_COLUMNS))

    ref_name = "native" if native else "gather"
    ref = case(native or with_mode(pair, "gather"), 1)
    natoms = len(ref["forces"]) // 3
    if fill == "part":
        assert 0 < natoms < 128
    # Within rounding of the summation order; the native styles sum the
    # same terms in another order.
    tol = {"e": 1e-12, "f": 1e-11, "p": 1e-10, "ea": 1e-12, "va": 1e-10}
    label = f"{system} {fill} triclinic={triclinic} newton {newton}"
    for n in ranks:
        runs = {"gather": case(with_mode(pair, "gather"), n),
                "domain": case(with_mode(pair, "domain"), n)}
        if native:
            runs["native"] = case(native, n)
        for name, got in runs.items():
            assert got["nprocs"] == n
            assert sum(got["nlocal"]) == natoms
            if fill == "part" and n > 1:
                assert 0 in got["nlocal"], got["nlocal"]
            compare(f"{label} np {n} {name} vs {ref_name} np 1", got, ref, tol)
        compare(f"{label} np {n} domain vs gather", runs["domain"], runs["gather"], tol)
        check_per_atom_sums(f"{label} np {n} domain", runs["domain"])


@pytest.mark.skipif(not MPIRUN, reason="needs RGPOT_MPIRUN")
@pytest.mark.parametrize("newton", ["on", "off"])
@pytest.mark.parametrize("system", ["lj", "eam"])
def test_domain_dynamics_match_gather(tmp_path, system, newton):
    """200 hot NVE steps on 4 ranks, through reneighbourings and atoms
    moving between ranks: the domain path follows the gather path. The
    0.5 A skin makes LAMMPS rebuild its lists about ten times, the solid
    CuNi too."""
    start, pair, _ = domain_systems(system, True, "full")
    head = read_data_start(tmp_path, start, "start.data", newton)
    md = [
        *THERMO,
        "neighbor 0.5 bin",
        "velocity all create 1200 4127 dist gaussian",
        "fix 1 all nve",
        "run 200",
    ]
    gather = ok(run_case(tmp_path, head + with_mode(pair, "gather") + md, nprocs=4))
    domain = ok(run_case(tmp_path, head + with_mode(pair, "domain") + md, nprocs=4))
    de = abs(domain["pe"] - gather["pe"])
    df = max_abs_diff(domain["forces"], gather["forces"])
    dx = max_abs_diff(flat(domain["x"]), flat(gather["x"]))
    print(f"{system} 200 steps np 4 newton {newton} domain vs gather: |dE| {de:.3e} eV, "
          f"max|dF| {df:.3e} eV/A, max|dx| {dx:.3e} A")
    assert de < 1e-8 * max(1.0, abs(gather["pe"]))
    assert df < 1e-7
    assert dx < 1e-8


REUSE_MD = [
    *THERMO,
    "velocity all create 1200 4127 dist gaussian",
    "fix 1 all nve",
    "run 60",
    "change_box all x scale 1.004 y scale 0.997 remap",
    "run 40",
    "fix 2 all deform 1 x scale 1.002",
    "run 20",
]


@pytest.mark.parametrize("nprocs", [1, 2])
@pytest.mark.parametrize("system", ["lj", "eam"])
def test_gather_list_reuse(tmp_path, system, nprocs):
    """120 hot NVE steps on the gather path, with atoms wrapped through
    the faces, moves past the skin of the potential's own pair list, a
    change_box between runs and a fix deform. EAMTable with its 1 A skin
    follows skin 0, which rebuilds the list whenever an atom moves, to
    the bit. LJ, whose skin is fixed, follows the domain path, which runs
    on the LAMMPS list."""
    if nprocs > 1 and not MPIRUN:
        pytest.skip("needs RGPOT_MPIRUN")
    start, pair, _ = domain_systems(system, True, "full")
    head = read_data_start(tmp_path, start, "start.data")
    got = ok(run_case(tmp_path, head + with_mode(pair, "gather") + REUSE_MD, nprocs=nprocs))
    if system == "eam":
        rebuilt = [line.replace(f"file {EAM_ALLOY}", f"file {EAM_ALLOY} skin 0")
                   for line in with_mode(pair, "gather")]
        ref = ok(run_case(tmp_path, head + rebuilt + REUSE_MD, nprocs=nprocs))
        label = "skin 1 vs skin 0"
    else:
        ref = ok(run_case(tmp_path, head + with_mode(pair, "domain") + REUSE_MD, nprocs=nprocs))
        label = "gather vs domain"
    de = abs(got["pe"] - ref["pe"])
    df = max_abs_diff(got["forces"], ref["forces"])
    dx = max_abs_diff(flat(got["x"]), flat(ref["x"]))
    dp = max_abs_diff(got["press"], ref["press"])
    print(f"{system} np {nprocs} {label} after 120 steps: |dE| {de:.3e} eV, "
          f"max|dF| {df:.3e} eV/A, max|dx| {dx:.3e} A, max|dP| {dp:.3e} bar")
    if system == "eam":
        assert got["pe"] == ref["pe"]
        assert got["forces"] == ref["forces"]
        assert got["x"] == ref["x"]
        assert got["press"] == ref["press"]
    else:
        assert de < 1e-8 * max(1.0, abs(ref["pe"]))
        assert df < 1e-7
        assert dx < 1e-8


def mode_log(tmp_path: Path, commands: list[str], nprocs: int = 1) -> str:
    log = tmp_path / f"mode{len(list(tmp_path.glob('mode*.log')))}.log"
    result = run_case(tmp_path, [f"log {log}"] + commands, nprocs=nprocs)
    text = log.read_text() if log.exists() else ""
    if "error" in result:
        return text + result["error"]
    return text


@pytest.mark.skipif(not MPIRUN, reason="needs RGPOT_MPIRUN")
def test_mode_follows_the_capabilities(tmp_path):
    """On two ranks auto takes the domain path for a finite-range
    potential with a staged evaluation and the gather path otherwise, and
    says which. The ghosts reach one cutoff plus the skin, many-body
    potentials included."""
    lj = mode_log(tmp_path, fcc(A_LJ, False) + lj_rgpot("named") + ["run 0"], 2)
    assert "pair_style rgpot: domain decomposition, cutoff 8.5, densities 0" in lj
    u0psi = mode_log(tmp_path, fcc(A_LJ, False) + lj_rgpot("u0psi") + ["run 0"], 2)
    assert "domain decomposition, cutoff 8.5, densities 0" in u0psi
    sc = mode_log(tmp_path, fcc(3.92, False) + named_pair("SuttonChen") + ["run 0"], 2)
    assert "domain decomposition, cutoff 7, densities 1" in sc
    assert "ghost atom cutoff = 9" in sc
    fd = mode_log(tmp_path, fcc(A_LJ, False) + lj_rgpot("named-fd") + ["run 0"], 2)
    assert "gather on rank 0; virial fd differentiates the whole box" in fd
    emt_pair = [f"plugin load {PLUGIN}", "pair_style rgpot 6.0", "pair_coeff * * EMT Cu Cu"]
    emt = mode_log(tmp_path, fcc(3.61, False) + emt_pair + ["run 0"], 2)
    assert "gather on rank 0; the rgpot potential EMT declares no finite range" in emt
    forced = mode_log(tmp_path, fcc(A_LJ, False) + with_mode(lj_rgpot("named"), "gather")
                      + ["run 0"], 2)
    assert "pair_style rgpot: gather on rank 0\n" in forced


def test_mode_on_one_rank(tmp_path):
    """One rank gathers unless told otherwise; mode domain refuses a
    potential that cannot use it."""
    lj = mode_log(tmp_path, fcc(A_LJ, False) + lj_rgpot("named") + ["run 0"])
    assert "pair_style rgpot: gather on rank 0; one rank" in lj
    forced = mode_log(tmp_path, fcc(A_LJ, False) + with_mode(lj_rgpot("named"), "domain")
                      + ["run 0"])
    assert "domain decomposition, cutoff 8.5, densities 0" in forced
    emt_pair = [f"plugin load {PLUGIN}", "pair_style rgpot 6.0", "pair_coeff * * EMT Cu Cu"]
    refused = mode_log(tmp_path, fcc(3.61, False) + with_mode(emt_pair, "domain") + ["run 0"])
    assert ("pair_style rgpot mode domain: the rgpot potential EMT declares no finite range"
            in refused)


@pytest.mark.skipif(not LJ_ENGINE, reason="needs RGPOT_LJ_ENGINE")
def test_engine_takes_the_gather_path(tmp_path):
    text = mode_log(tmp_path, dilute_lj([f"plugin load {PLUGIN}", "pair_style rgpot 15.0",
                                         f"pair_coeff * * engine {LJ_ENGINE} H"]))
    assert f"gather on rank 0; the engine {LJ_ENGINE} is an engine" in text


def test_refuses_unknown_style_keywords(tmp_path):
    base = fcc(A_LJ, False) + [f"plugin load {PLUGIN}"]
    assert "mode is auto, gather or domain" in error_of(
        tmp_path, base + [f"pair_style rgpot {CUT} mode fast"])
    assert "has no keyword" in error_of(tmp_path, base + [f"pair_style rgpot {CUT} speed 2"])
