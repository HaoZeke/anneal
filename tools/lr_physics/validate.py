#!/usr/bin/env python3
"""Independent checks for the long-range kernels.

Reads CASE lines from lr_phys_test. Compares DFT-D3/D4 to the Python
reference bindings, SPME/Ewald to OpenMM, the NaCl energy to an mpmath
Madelung sum, and the damped Coulomb derivative to SymPy.
"""

import math
import os
import subprocess
import sys

import mpmath as mp
import numpy as np
from sympy import diff, erf, erfc, exp, pi, sqrt, symbols

HARTREE_TO_EV = 27.211386245988
BOHR = 0.529177210903
ANGSTROM_TO_BOHR = 1.0 / BOHR
KE = HARTREE_TO_EV * BOHR


def cases_from(text):
    out = {}
    for line in text.splitlines():
        if not line.startswith("CASE "):
            continue
        parts = line.split()
        out[parts[1]] = [float(x) for x in parts[2:]]
    return out


def mp_nacl_ewald():
    """Independent Ewald sum for the 8-ion NaCl cell used by the C++ test."""
    mp.mp.dps = 25
    a = mp.mpf(2)
    alpha = mp.mpf("2.5")
    rc = mp.mpf(8)
    kmax = 12
    ke = mp.mpf("27.211386245988") * mp.mpf("0.529177210903")
    sites = []
    for p, q in (
        ((0, 0, 0), 1),
        ((0.5, 0.5, 0), 1),
        ((0.5, 0, 0.5), 1),
        ((0, 0.5, 0.5), 1),
        ((0.5, 0, 0), -1),
        ((0, 0.5, 0), -1),
        ((0, 0, 0.5), -1),
        ((0.5, 0.5, 0.5), -1),
    ):
        sites.append((mp.mpf(p[0]) * a, mp.mpf(p[1]) * a, mp.mpf(p[2]) * a, mp.mpf(q)))
    energy = mp.mpf(0)
    nimg = 6
    for i, (xi, yi, zi, qi) in enumerate(sites):
        for j, (xj, yj, zj, qj) in enumerate(sites):
            if j < i:
                continue
            for nx in range(-nimg, nimg + 1):
                for ny in range(-nimg, nimg + 1):
                    for nz in range(-nimg, nimg + 1):
                        if i == j and nx == ny == nz == 0:
                            continue
                        dx = xj - xi + nx * a
                        dy = yj - yi + ny * a
                        dz = zj - zi + nz * a
                        r = mp.sqrt(dx * dx + dy * dy + dz * dz)
                        if r >= rc:
                            continue
                        term = qi * qj * mp.erfc(alpha * r) / r
                        energy += term if i != j else mp.mpf("0.5") * term
    energy *= ke
    for s in sites:
        energy -= ke * alpha / mp.sqrt(mp.pi) * s[3] ** 2
    pref = ke * 2 * mp.pi / (a ** 3)
    for mx in range(-kmax, kmax + 1):
        for my in range(-kmax, kmax + 1):
            for mz in range(-kmax, kmax + 1):
                if mx == my == mz == 0:
                    continue
                kx = 2 * mp.pi * mx / a
                ky = 2 * mp.pi * my / a
                kz = 2 * mp.pi * mz / a
                k2 = kx * kx + ky * ky + kz * kz
                f = mp.e ** (-k2 / (4 * alpha * alpha)) / k2
                c = mp.mpf(0)
                d = mp.mpf(0)
                for x, y, z, q in sites:
                    th = kx * x + ky * y + kz * z
                    c += q * mp.cos(th)
                    d += q * mp.sin(th)
                energy += pref * f * (c * c + d * d)
    return float(energy)


def sympy_dsf():
    r, alpha, rc = symbols("r alpha rc", positive=True)
    phi = erfc(alpha * r) / r
    bare_rc = erfc(alpha * rc) / rc
    dphi = diff(phi, r)
    dphi_rc = diff(phi, r).subs(r, rc)
    shifted = phi - bare_rc - dphi_rc * (r - rc)
    d_shifted = diff(shifted, r)
    f = d_shifted.subs({alpha: 0.25, rc: 8.0, r: 1.7})
    return float(f.evalf(30))


def dftd_reference():
    from dftd3.interface import DispersionModel, RationalDampingParam
    from dftd4.interface import DispersionModel as D4Model
    from dftd4.interface import DampingParam

    numbers = np.array([8, 1, 1])
    pos_a = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])
    pos = pos_a * ANGSTROM_TO_BOHR
    d3 = DispersionModel(numbers, pos)
    param = RationalDampingParam(method="pbe", atm=True)
    res = d3.get_dispersion(param, grad=True)
    e3 = float(res["energy"]) * HARTREE_TO_EV
    g3 = np.array(res["gradient"]) * (-HARTREE_TO_EV / BOHR)
    d4 = D4Model(numbers, pos, charge=0.0)
    p4 = DampingParam(method="pbe", atm=True)
    res4 = d4.get_dispersion(p4, grad=True)
    e4 = float(res4["energy"]) * HARTREE_TO_EV
    g4 = np.array(res4["gradient"]) * (-HARTREE_TO_EV / BOHR)
    return e3, g3.reshape(-1), e4, g4.reshape(-1)


def openmm_pme():
    # Orthorhombic NaCl, lattice constant 10 A. The C++ Madelung cell is
    # 2 A, so the energy scales by 2/10.
    from openmm import Context, NonbondedForce, System, Vec3, VerletIntegrator
    from openmm.unit import angstrom, nanometer, picosecond, kilojoule_per_mole

    a = 10.0
    pos = []
    q = []
    for p, charge in (
        ((0, 0, 0), 1),
        ((0.5, 0.5, 0), 1),
        ((0.5, 0, 0.5), 1),
        ((0, 0.5, 0.5), 1),
        ((0.5, 0, 0), -1),
        ((0, 0.5, 0), -1),
        ((0, 0, 0.5), -1),
        ((0.5, 0.5, 0.5), -1),
    ):
        pos.append([p[0] * a, p[1] * a, p[2] * a])
        q.append(charge)
    system = System()
    nb = NonbondedForce()
    nb.setNonbondedMethod(NonbondedForce.PME)
    nb.setCutoffDistance(0.45 * nanometer)
    nb.setEwaldErrorTolerance(1.0e-7)
    for charge in q:
        system.addParticle(1.0)
        nb.addParticle(charge, 1.0, 0.0)
    system.addForce(nb)
    system.setDefaultPeriodicBoxVectors(
        Vec3(a, 0, 0) * angstrom,
        Vec3(0, a, 0) * angstrom,
        Vec3(0, 0, a) * angstrom,
    )
    ctx = Context(system, VerletIntegrator(0.001 * picosecond))
    ctx.setPositions((np.array(pos)) * angstrom)
    state = ctx.getState(getEnergy=True)
    kj = state.getPotentialEnergy().value_in_unit(kilojoule_per_mole)
    return kj / 96.48533212331002


def so3lr_qdo_pair():
    """SO3LR vdw_QDO_disp_damp for one unordered C-H pair, ratios 1."""
    # Free-atom current table, Z=6 and Z=1, from the same source arrays.
    alpha = {1: 4.6013, 6: 11.205754}
    c6 = {1: 6.5, 6: 46.6}  # placeholder overwritten below by the inc file values
    return alpha, c6


def numpy_qdo(pos_a, z, h):
    text = open(
        os.path.join(
            os.path.dirname(__file__),
            "../../CppCore/rgpot/LongRange/QdoTable.inc",
        )
    ).read()

    def grab(name):
        start = text.index(name)
        body = text[text.index("{", start) + 1 : text.index("};", start)]
        return np.array([float(x) for x in body.replace("\n", " ").split(",") if x.strip()])

    alpha_tab = grab("kQdoAlpha")
    c6_tab = grab("kQdoC6")
    fine = 7.2973525693e-3
    energy = 0.0
    for i in range(len(z)):
        for j in range(i + 1, len(z)):
            r_a = np.linalg.norm(pos_a[j] - pos_a[i])
            R = r_a / BOHR
            ai = alpha_tab[z[i] - 1] * h[i]
            aj = alpha_tab[z[j] - 1] * h[j]
            ci = c6_tab[z[i] - 1] * h[i] ** 2
            cj = c6_tab[z[j] - 1] * h[j] ** 2
            aij = 0.5 * (ai + aj)
            c6ij = 2 * ci * cj * ai * aj / (ai**2 * cj + aj**2 * ci)
            radius = fine ** (-4.0 / 21.0) * aij ** (1.0 / 7.0)
            sigma = (
                -0.00078893 * radius**3
                + 0.04125273 * radius**2
                + 0.24428889 * radius
                - 0.00433008
            )
            gamma = 0.5 / sigma**2
            c8 = 5 / gamma * c6ij
            c10 = 245 / 8 / gamma**2 * c6ij
            p = 2 * 2.54 * aij ** (1.0 / 7.0)
            vh = (
                -c6ij / (R**6 + p**6)
                - c8 / (R**8 + p**8)
                - c10 / (R**10 + p**10)
            )
            energy += vh * HARTREE_TO_EV
    return energy


def main():
    binary = sys.argv[1]
    proc = subprocess.run([binary], check=True, capture_output=True, text=True)
    if "lr_phys_test ok" not in proc.stdout:
        print(proc.stdout)
        print(proc.stderr)
        raise SystemExit("C++ suite failed")
    got = cases_from(proc.stdout + "\n" + proc.stderr)
    fails = 0

    def close(name, a, b, tol):
        nonlocal fails
        err = abs(a - b)
        scale = max(1.0, abs(b))
        if err > tol * scale:
            print(f"FAIL {name} got {a} ref {b} err {err}")
            fails += 1
        else:
            print(f"ok {name} {a} ref {b}")

    mp_nacl = mp_nacl_ewald()
    close("mpmath Ewald vs C++ NaCl", got["nacl"][0], mp_nacl, 1e-8)
    # OpenMM on the 10 A cell. C++ reference is the 2 A cell.
    close("OpenMM PME vs scaled NaCl", openmm_pme(), got["nacl"][0] * 0.2, 2e-4)

    dphi = sympy_dsf()
    r = 1.7
    alpha = 0.25
    rc = 8.0
    # Numeric derivative of the same shifted potential, independent of the C++ binary.
    def phi(rr):
        return math.erfc(alpha * rr) / rr

    bare_rc = phi(rc)
    d_rc = (phi(rc + 1e-6) - phi(rc - 1e-6)) / 2e-6
    shifted = phi(r) - bare_rc - d_rc * (r - rc)
    shifted_p = phi(r + 1e-6) - bare_rc - d_rc * (r + 1e-6 - rc)
    shifted_m = phi(r - 1e-6) - bare_rc - d_rc * (r - 1e-6 - rc)
    numeric = (shifted_p - shifted_m) / 2e-6
    close("sympy DSF derivative", dphi, numeric, 1e-6)

    e3, g3, e4, g4 = dftd_reference()
    close("dftd3 water energy", got["d3_water"][0], e3, 1e-8)
    for i, (a, b) in enumerate(zip(got["d3_water"][1:], g3)):
        close(f"dftd3 force {i}", a, b, 1e-6)
    close("dftd4 water energy", got["d4_water"][0], e4, 1e-8)
    for i, (a, b) in enumerate(zip(got["d4_water"][1:], g4)):
        close(f"dftd4 force {i}", a, b, 1e-6)

    pos = np.array([[0.0, 0.0, 0.0], [1.4, 0.1, 0.0], [0.2, 1.6, 0.3]])
    qdo = numpy_qdo(pos, [6, 1, 8], [1.0, 0.8, 1.1])
    close("SO3LR QDO formula", got["qdo_cho"][0], qdo, 1e-8)

    if fails:
        print(fails, "validation checks failed")
        return 1
    print("validate ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
