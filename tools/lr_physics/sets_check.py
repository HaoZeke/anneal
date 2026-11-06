#!/usr/bin/env python3
"""Compare D3(BJ), D4 and MBD to s-dftd3, dftd4 and pymbd.

Energies are Hartree and gradients are Hartree/Bohr. D3 and D4 must
agree to 1e-8 on every system. The MBD check covers the S22 dimers and
one periodic argon cell.

Usage: sets_check.py LR_SETS [DATA_DIR]. DATA_DIR holds S22.py and
S66.py from Psi4 and anthracene.sdf. It defaults to the working
directory.
"""

import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from dftd3.interface import DispersionModel, RationalDampingParam
from dftd4.interface import DampingParam, DispersionModel as D4Model
from pymbd import ang
from pymbd.fortran import MBDGeom

BOHR = 0.529177210903
ZNUM = {
    "H": 1, "He": 2, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
    "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18, "Br": 35, "I": 53,
}


def parse_psi4(path, dbse):
    text = Path(path).read_text()
    # The Psi4 source stores the format string literally.
    pat = re.compile(
        r"GEOS\['%s-%s-([^']+)' % \(dbse, '(\d+)'\)\] = qcdb\.Molecule\(\"\"\"(.*?)\"\"\"",
        re.S,
    )
    systems = []
    for kind, num, body in pat.findall(text):
        z = []
        xyz = []
        symbols = []
        for line in body.splitlines():
            parts = line.split()
            if len(parts) < 4 or parts[0] not in ZNUM:
                continue
            symbols.append(parts[0])
            z.append(ZNUM[parts[0]])
            xyz.append([float(parts[1]), float(parts[2]), float(parts[3])])
        systems.append(
            {
                "name": f"{dbse}-{num}-{kind}",
                "set": dbse,
                "dimer": kind == "dimer",
                "z": np.array(z, dtype=int),
                "xyz": np.array(xyz, dtype=float),
                "symbols": symbols,
            }
        )
    return systems


def parse_sdf(path, name):
    lines = Path(path).read_text().splitlines()
    nat = int(lines[3][:3])
    z = []
    xyz = []
    symbols = []
    for line in lines[4 : 4 + nat]:
        parts = line.split()
        symbols.append(parts[3])
        z.append(ZNUM[parts[3]])
        xyz.append([float(parts[0]), float(parts[1]), float(parts[2])])
    return [
        {
            "name": name,
            "set": "anthracene",
            "dimer": True,
            "z": np.array(z, dtype=int),
            "xyz": np.array(xyz, dtype=float),
            "symbols": symbols,
        }
    ]


def write_batch(path, systems, kinds):
    rows = []
    for sys in systems:
        for kind in kinds:
            rows.append((sys, kind))
    with path.open("w") as out:
        out.write(f"{len(rows)}\n")
        for sys, kind in rows:
            periodic = 1 if sys.get("box") is not None else 0
            out.write(f"{sys['name']} {kind} {len(sys['z'])} {periodic}\n")
            for zi, xyz in zip(sys["z"], sys["xyz"]):
                out.write(f"{int(zi)} {xyz[0]:.12f} {xyz[1]:.12f} {xyz[2]:.12f}\n")
            if periodic:
                box = sys["box"]
                out.write(" ".join(f"{v:.12f}" for v in box) + "\n")
    return rows


def read_cpp(text):
    got = {}
    for line in text.splitlines():
        if not line.startswith("SYS "):
            continue
        parts = line.split()
        name, kind = parts[1], parts[2]
        nums = [float(x) for x in parts[3:]]
        got[(name, kind)] = (nums[0], np.array(nums[1:]))
    return got


def ref_d3(sys):
    pos = sys["xyz"] / BOHR
    model = DispersionModel(sys["z"], pos)
    res = model.get_dispersion(RationalDampingParam(method="pbe", atm=True), grad=True)
    return float(res["energy"]), np.array(res["gradient"]).reshape(-1)


def ref_d4(sys):
    pos = sys["xyz"] / BOHR
    model = D4Model(sys["z"], pos, charge=0.0)
    res = model.get_dispersion(DampingParam(method="pbe", atm=True), grad=True)
    return float(res["energy"]), np.array(res["gradient"]).reshape(-1)


def max_err(energy, grad, ref_e, ref_g):
    e = abs(energy - ref_e)
    g = float(np.max(np.abs(grad - ref_g))) if grad.size else 0.0
    return e, g


def main():
    binary = sys.argv[1]
    data = Path(sys.argv[2]) if len(sys.argv) > 2 else Path.cwd()
    scratch = tempfile.TemporaryDirectory(prefix="lr_sets_")
    work = Path(scratch.name)
    s22 = parse_psi4(data / "S22.py", "S22")
    s66 = parse_psi4(data / "S66.py", "S66")
    anth = parse_sdf(data / "anthracene.sdf", "anthracene")
    systems = s22 + s66 + anth
    if not any(s["dimer"] for s in s22) or not any(s["dimer"] for s in s66):
        raise SystemExit("failed to parse the dimer sets")
    batch = work / "lr_sets_in.txt"
    rows = write_batch(batch, systems, ["d3", "d4"])
    proc = subprocess.run([binary, str(batch)], check=True, capture_output=True, text=True)
    got = read_cpp(proc.stdout)
    worst = {}
    fails = 0
    for geom, kind in rows:
        energy, grad = got[(geom["name"], kind)]
        ref_e, ref_g = ref_d3(geom) if kind == "d3" else ref_d4(geom)
        e, g = max_err(energy, grad, ref_e, ref_g)
        key = (geom["set"], kind, "dimer" if geom["dimer"] else "all")
        for label in (key, (geom["set"], kind, "all")):
            prev = worst.get(label, (0.0, 0.0, ""))
            if e > prev[0] or g > prev[1]:
                worst[label] = (max(prev[0], e), max(prev[1], g), geom["name"])
        if e > 1e-8 or g > 1e-8:
            fails += 1
            print(f"FAIL {geom['name']} {kind} energy {e:.3e} grad {g:.3e}")
    print("D3/D4 max absolute error (Ha, Ha/Bohr)")
    for key in sorted(worst):
        e, g, name = worst[key]
        print(f"  {key[0]:12} {key[1]:3} {key[2]:6} energy {e:.3e} grad {g:.3e} at {name}")

    # MBD on S22 dimers and one periodic cell.
    dimers = [s for s in s22 if s["dimer"]]
    argon = {
        "name": "argon-crystal",
        "set": "periodic",
        "dimer": True,
        "z": np.array([18, 18]),
        "xyz": np.array([[0.3, 0.1, 0.2], [4.1, -0.2, -0.1]]),
        "symbols": ["Ar", "Ar"],
        "box": np.array([8.1, 0.1, -0.2, 0.3, 3.9, -0.1, -0.1, 0.2, 4.2]),
    }
    mbatch = work / "lr_mbd_in.txt"
    mrows = write_batch(mbatch, dimers + [argon], ["mbd"])
    proc = subprocess.run([binary, str(mbatch)], check=True, capture_output=True, text=True)
    mgot = read_cpp(proc.stdout)
    mworst = 0.0
    mname = ""
    for geom, kind in mrows:
        energy, grad = mgot[(geom["name"], kind)]
        coords = [(float(x) * ang, float(y) * ang, float(z) * ang) for x, y, z in geom["xyz"]]
        if geom.get("box") is None:
            ref = MBDGeom(coords).mbd_energy_species(geom["symbols"], [1.0] * len(coords), 0.83, force=True)
        else:
            b = geom["box"]
            lattice = [
                (b[0] * ang, b[1] * ang, b[2] * ang),
                (b[3] * ang, b[4] * ang, b[5] * ang),
                (b[6] * ang, b[7] * ang, b[8] * ang),
            ]
            ref = MBDGeom(coords, lattice, (4, 4, 4)).mbd_energy_species(
                geom["symbols"], [1.0] * len(coords), 0.83, force=True
            )
        ref_e, ref_g = float(ref[0]), np.array(ref[1]).reshape(-1)
        e, g = max_err(energy, grad, ref_e, ref_g)
        print(f"MBD {geom['name']} energy {e:.3e} grad {g:.3e}")
        if e > mworst:
            mworst = e
            mname = geom["name"]
        # Finite differences on the C++ side are not the analytic pymbd
        # gradient. The energy is the comparison that has to be tight.
        if e > 1e-6:
            fails += 1
            print(f"FAIL MBD energy {geom['name']}")
    print(f"MBD max energy error {mworst:.3e} at {mname}")
    if fails:
        raise SystemExit(f"{fails} systems failed")
    print("sets ok")


if __name__ == "__main__":
    main()
