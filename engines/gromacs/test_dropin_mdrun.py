"""GROMACS mdrun on librgpot_gromacs, against direct rgpot evaluation.

Every run is a double-precision, thread-MPI GROMACS built with
-DGMX_CP2K=ON against librgpot_gromacs (build_gromacs.sh). The topologies
give every atom zero charge and zero Lennard-Jones, so GROMACS adds nothing
of its own: "Quantum En." is the rgpot energy and the forces in the .trr
are the rgpot forces. Each is compared with rgpot-eval (or the engine
library) on the positions and box of the same frame.

Environment: RGPOT_GMX (or gmx_d on PATH), RGPOT_EVAL, and optionally
RGPOT_EAM_ALLOY, RGPOT_CLASSICAL_ENGINE and RGPOT_CAPNP_SCHEMA.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dropin_case as dc  # noqa: E402
import test_dropin_api as api  # noqa: E402

GMX, WHY = dc.gmx_problem()
EVAL = os.environ.get("RGPOT_EVAL", "")
EAM = os.environ.get("RGPOT_EAM_ALLOY", "")
ENGINE = os.environ.get("RGPOT_CLASSICAL_ENGINE", "")
SCHEMA = os.environ.get("RGPOT_CAPNP_SCHEMA", "")

pytestmark = [pytest.mark.skipif(GMX is None, reason=WHY or "no GROMACS"),
              pytest.mark.skipif(not EVAL, reason="needs RGPOT_EVAL")]

LJ_AR = ["LJ", "u0", "0.0104", "psi", "3.4", "cutoff", "7.5"]
PT = ["SuttonChen", "preset", "platinum", "cutoff", "5.0"]


def rgpot_section(spec: list[str], extra: str = "") -> str:
    lines = [f"    POTENTIAL {spec[0]}"]
    if len(spec) > 1:
        lines.append("    OPTIONS " + " ".join(spec[1:]))
    if extra:
        lines.append(extra)
    return "\n".join(lines)


def argon(n=3, jiggle=0.01):
    x, box = dc.fcc(0.53, n, jiggle=jiggle, seed=1)
    return ["Ar"] * len(x), x, box


def platinum(n=4):
    x, box = dc.fcc(0.3924, n, jiggle=0.006, tilt=(0.10, 0.06, -0.08))
    return ["Pt"] * len(x), x, box


def silicon():
    x, box = dc.diamond(0.5431, 2, jiggle=0.004)
    return ["Si"] * len(x), x, box


def cuni():
    x, box = dc.fcc(0.3560, 4, jiggle=0.004, seed=9)
    rng = np.random.default_rng(4)
    return [("Ni" if rng.random() < 0.3 else "Cu") for _ in range(len(x))], x, box


def engine_config(work: Path) -> Path:
    config = work.parent / f"{work.name}-tersoff.bin"
    config.write_bytes(subprocess.run(["capnp", "encode", SCHEMA, "BuiltinParams"],
                                      input=b'(name = "tersoff")', capture_output=True,
                                      check=True).stdout)
    return config


def single_point(work: Path, symbols, x, box, rgpot: str, qm=None, mdp=None):
    dc.write_system(work, symbols=symbols, positions=x, box=box, qmi_text=dc.qmi(rgpot),
                    mdp=mdp or {}, qm=qm)
    dc.run(GMX, work)
    terms = dc.energies(GMX, work, "Quantum-En.", "Potential")
    frame = dc.read_trr(work / "md.trr")[0]
    assert frame["double"], "the .trr is single precision"
    return terms["Quantum En."][0], terms["Potential"][0], frame


def check(e_gmx, f_gmx, e_ref_ev, f_ref_ev, e_rel=1e-10, f_rel=1e-9):
    e_ref = e_ref_ev * dc.KJ_PER_EV
    f_ref = f_ref_ev * dc.KJ_PER_EV * 10
    assert abs(e_gmx - e_ref) <= e_rel * abs(e_ref), (e_gmx, e_ref)
    scale = np.max(np.abs(f_ref))
    worst = np.max(np.abs(f_gmx - f_ref))
    assert worst <= f_rel * scale, (worst, scale)
    return abs(e_gmx - e_ref) / abs(e_ref), worst / scale


CASES = {
    "lj-argon": (argon, LJ_AR),
    "suttonchen-platinum-triclinic": (platinum, PT),
    "tersoff-silicon": (silicon, ["TersoffSi"]),
    "eamtable-cuni": (cuni, ["EAMTable", "file", EAM]),
}


@pytest.mark.parametrize("name", list(CASES))
def test_single_point_matches_rgpot_eval(tmp_path, name):
    system, spec = CASES[name]
    if name == "eamtable-cuni" and not EAM:
        pytest.skip("needs RGPOT_EAM_ALLOY (rgpot built with with_eam_table)")
    symbols, x, box = system()
    e, pot, frame = single_point(tmp_path, symbols, x, box, rgpot_section(spec))
    assert pot == e, "GROMACS added a term of its own"
    e_ref, f_ref = dc.rgpot_eval(EVAL, spec, symbols, dc.wrap(frame["x"], frame["box"]) * 10,
                                 frame["box"] * 10)
    de, df = check(e, frame["f"], e_ref, f_ref)
    print(f"{name} on GROMACS {dc.gmx_version(GMX)}: energy {de:.1e} relative, "
          f"forces {df:.1e} of the largest")


@pytest.mark.skipif(not (ENGINE and SCHEMA and shutil.which("capnp")),
                    reason="needs RGPOT_CLASSICAL_ENGINE, RGPOT_CAPNP_SCHEMA and capnp")
def test_engine_route_matches_the_engine(tmp_path):
    symbols, x, box = silicon()
    config = engine_config(tmp_path)
    e, pot, frame = single_point(tmp_path, symbols, x, box,
                                 f"    ENGINE {ENGINE}\n    CONFIG {config}")
    assert pot == e
    e_ref, f_ref = api.engine_force(config, symbols, dc.wrap(frame["x"], frame["box"]) * 10,
                                    frame["box"] * 10)
    check(e, frame["f"], e_ref, f_ref)
    assert "engine classical" in (tmp_path / "md_cp2k.out").read_text()


def test_qm_subset_only_moves_qm_atoms(tmp_path):
    symbols, x, box = argon()
    qm = [i for i in range(len(x)) if i % 3 != 1]
    e, pot, frame = single_point(tmp_path, symbols, x, box, rgpot_section(LJ_AR), qm=qm)
    assert pot == e
    e_ref, f_ref = dc.rgpot_eval(EVAL, LJ_AR, [symbols[i] for i in qm],
                                 dc.wrap(frame["x"][qm], frame["box"]) * 10, frame["box"] * 10)
    check(e, frame["f"][qm], e_ref, f_ref)
    mm = [i for i in range(len(x)) if i % 3 == 1]
    assert np.all(frame["f"][mm] == 0.0)


def test_openmp_threads(tmp_path):
    symbols, x, box = platinum()
    dc.write_system(tmp_path, symbols=symbols, positions=x, box=box,
                    qmi_text=dc.qmi(rgpot_section(PT)), mdp={"nsteps": 5})
    dc.run(GMX, tmp_path, ntomp=4)
    assert "each using 4 OpenMP threads" in (tmp_path / "md.log").read_text()
    energy = dc.energies(GMX, tmp_path, "Quantum-En.")["Quantum En."]
    frames = dc.read_trr(tmp_path / "md.trr")
    assert len(frames) == len(energy) == 6
    for e, frame in zip(energy, frames):
        e_ref, f_ref = dc.rgpot_eval(EVAL, PT, symbols, dc.wrap(frame["x"], frame["box"]) * 10,
                                     frame["box"] * 10)
        check(e, frame["f"], e_ref, f_ref)


def unwrap_about_first(x: np.ndarray, box: np.ndarray) -> np.ndarray:
    inv = np.linalg.inv(box)
    d = x - x[0]
    return x[0] + d - np.rint(d @ inv) @ box


def test_cell_none_follows_a_cluster_across_the_boundary(tmp_path):
    rng = np.random.default_rng(2)
    cluster = [np.zeros(3)]
    while len(cluster) < 13:
        trial = rng.uniform(-0.5, 0.5, 3)
        if all(np.linalg.norm(trial - c) > 0.36 for c in cluster):
            cluster.append(trial)
    box = np.eye(3) * 3.0
    x = dc.wrap(np.array(cluster), box)
    symbols = ["Ar"] * len(x)
    mdp = {"nsteps": 200, "dt": 0.004, "nstxout": 20, "nstvout": 20, "nstfout": 20, "nstenergy": 20,
           "nstcalcenergy": 20, "comm-mode": "None"}
    dc.write_system(tmp_path, symbols=symbols, positions=x, box=box, mdp=mdp,
                    qmi_text=dc.qmi(rgpot_section(LJ_AR, "    CELL NONE")))
    dc.run(GMX, tmp_path)
    frames = dc.read_trr(tmp_path / "md.trr")
    energy = dc.energies(GMX, tmp_path, "Quantum-En.")["Quantum En."]
    assert len(frames) == len(energy) == 11
    moved = 0.0
    for frame, e in zip(frames, energy):
        whole = unwrap_about_first(frame["x"], frame["box"])
        moved = max(moved, np.max(np.abs(whole - unwrap_about_first(x, box))))
        e_ref, f_ref = dc.rgpot_eval(EVAL, LJ_AR, symbols, (whole - whole.mean(0) + 5.0) * 10,
                                     np.eye(3) * 100.0)
        check(e, frame["f"], e_ref, f_ref, e_rel=1e-9, f_rel=1e-8)
    assert moved > 0.005, "the cluster did not move"


def nve(work: Path, symbols, x, box, spec, dt, nsteps, temperature):
    mdp = {"integrator": "md-vv", "nsteps": nsteps, "dt": dt, "continuation": "no",
           "gen-vel": "yes", "gen-temp": temperature, "gen-seed": 7, "nstxout": 0,
           "nstvout": 0, "nstfout": 0, "nstenergy": 10, "nstcalcenergy": 10}
    dc.write_system(work, symbols=symbols, positions=x, box=box, mdp=mdp,
                    qmi_text=dc.qmi(rgpot_section(spec)))
    dc.run(GMX, work)
    # No thermostat or barostat: the total energy is the conserved one.
    terms = dc.energies(GMX, work, "Potential", "Total-Energy", "Temperature")
    return terms["Potential"], terms["Total Energy"], terms["Temperature"]


# Only smoothly cut-off potentials: SuttonChen is hard-truncated, so its
# energy jumps whenever a pair crosses the cutoff.
@pytest.mark.parametrize("name", ["lj-argon-switched", "tersoff-silicon", "eamtable-cuni"])
def test_nve_conserves_energy(tmp_path, name):
    if name == "lj-argon-switched":
        symbols, x, box = argon(jiggle=0.0)
        spec = LJ_AR + ["switch_width", "1.0"]
        dt, nsteps, temperature = 0.004, 2500, 60
    elif name == "tersoff-silicon":
        symbols, x, box = silicon()
        spec = ["TersoffSi"]
        dt, nsteps, temperature = 0.001, 2500, 300
    else:
        if not EAM:
            pytest.skip("needs RGPOT_EAM_ALLOY (rgpot built with with_eam_table)")
        symbols, x, box = cuni()
        spec = ["EAMTable", "file", EAM]
        dt, nsteps, temperature = 0.002, 2500, 300
    pot, cons, temp = nve(tmp_path, symbols, x, box, spec, dt, nsteps, temperature)
    swing = np.max(pot) - np.min(pot)
    slope = np.polyfit(np.arange(len(cons)) * 10 * dt, cons, 1)[0]
    drift = abs(slope) * nsteps * dt
    spread = np.max(cons) - np.min(cons)
    print(f"{name} on GROMACS {dc.gmx_version(GMX)}: {nsteps} steps of {dt} ps, mean T "
          f"{np.mean(temp):.1f} K, potential swing {swing:.4f} kJ/mol, conserved-energy "
          f"drift {drift:.2e} kJ/mol, spread {spread:.2e} kJ/mol")
    assert swing > 1.0, "the run did not move"
    assert drift < 1e-3 * swing
    assert spread < 1e-2 * swing


# --- refused inputs through mdrun ----------------------------------------

def refused(work: Path, qmi_text: str | None, *, mdp=None, mm_charges=None) -> str:
    # grompp refuses a triclinic box for generated CP2K input.
    symbols, x, box = argon()
    qm = list(range(len(x) - 2)) if mm_charges else None
    dc.write_system(work, symbols=symbols, positions=x, box=box, qmi_text=qmi_text,
                    mdp=mdp or {}, qm=qm, mm_charges=mm_charges)
    done = dc.run(GMX, work, check=False)
    assert done.returncode != 0, "mdrun accepted a bad configuration"
    assert "rgpot-gromacs: " in done.stderr, done.stderr[-2000:]
    assert not (work / "md.gro").exists(), "mdrun finished a run it should have refused"
    return done.stderr


@pytest.mark.parametrize("rgpot, why", [
    ("    POTENTIAL Nope", "unknown rgpot potential Nope"),
    ("    POTENTIAL LJ\n    OPTIONS u0 0.0104 colour blue", "LJ takes no option colour"),
    ("    ENGINE libnope_engine.so", "cannot open ENGINE libnope_engine.so"),
    ("    POTENTIAL LJ\n    EMBEDDING ELECTROSTATIC",
     "EMBEDDING ELECTROSTATIC is not implemented"),
])
def test_bad_config_stops_mdrun(tmp_path, rgpot, why):
    assert why in refused(tmp_path, dc.qmi(rgpot))


def test_cp2k_input_without_rgpot_section_stops_mdrun(tmp_path):
    raw = dc.qmi("").replace("  &RGPOT\n\n  &END RGPOT\n", "")
    assert "has no &RGPOT section" in refused(tmp_path, raw)


def test_generated_cp2k_input_stops_mdrun(tmp_path):
    why = refused(tmp_path, None, mdp={"qmmm-cp2k-qmmethod": "PBE"})
    assert "has no &RGPOT section" in why


def test_charged_mm_atoms_stop_mdrun(tmp_path):
    why = refused(tmp_path, dc.qmi(rgpot_section(LJ_AR)), mm_charges={106: 1.0, 107: -1.0})
    assert "2 MM atoms carry charges" in why
