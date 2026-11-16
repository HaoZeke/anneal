"""rgpot-ipi-driver against ASE's SocketIOCalculator and i-PI.

RGPOT_IPI_DRIVER names the program. RGPOT_PYTHON_BUILD names the build
directory that holds the rgpot extension; the package is staged from
python/rgpot beside it so RgpotCalculator gives the reference values.
Without it an installed rgpot is used. ASE carries CODATA 2014 units and rgpot CODATA 2018, so
the socket results agree to about 1e-8 relative, not to round-off.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import uuid
from pathlib import Path

import numpy as np
import pytest

ase = pytest.importorskip("ase")
from ase.build import bulk  # noqa: E402
from ase.calculators.socketio import SocketIOCalculator  # noqa: E402

DRIVER = os.environ.get("RGPOT_IPI_DRIVER", "")
SOURCE_PACKAGE = Path(__file__).resolve().parents[2] / "python" / "rgpot"

AR = {"u0": 0.0104, "psi": 3.40, "cutoff": 8.5}
LJ_SPEC = "LJ u0 0.0104 psi 3.40 cutoff 8.5"
EAM_AL_CUTOFF = 7.1

pytestmark = pytest.mark.skipif(
    not DRIVER or not Path(DRIVER).is_file(),
    reason="RGPOT_IPI_DRIVER does not name the built rgpot-ipi-driver",
)


@pytest.fixture(scope="session")
def rgpot_calc(tmp_path_factory):
    """RgpotCalculator from the staged package, or skip."""
    build = os.environ.get("RGPOT_PYTHON_BUILD", "")
    cores = sorted(Path(build).glob("_core*.so")) if build else []
    if cores and SOURCE_PACKAGE.is_dir():
        core = cores[0]
        stage = tmp_path_factory.mktemp("python")
        pkg = stage / "rgpot"
        pkg.mkdir()
        for item in SOURCE_PACKAGE.iterdir():
            if item.name == "__pycache__" or item.name.startswith("_core."):
                continue
            (pkg / item.name).symlink_to(item)
        (pkg / core.name).symlink_to(core.resolve())
        sys.path.insert(0, str(stage))
    try:
        from rgpot.ase import RgpotCalculator
    except ImportError as err:
        pytest.skip(f"rgpot Python package unavailable: {err}")
    return RgpotCalculator


def driver_names() -> list[str]:
    out = subprocess.run([DRIVER, "--list"], capture_output=True, text=True, check=True)
    return out.stdout.split()


def socket_name(tag: str) -> str:
    return f"rgpot-{tag}-{uuid.uuid4().hex[:12]}"


def sheared(atoms):
    """A periodic, non-orthogonal copy with displaced atoms."""
    atoms = atoms.copy()
    cell = np.array(atoms.cell)
    cell[1] += 0.18 * cell[0]
    cell[2] += 0.11 * cell[0] - 0.07 * cell[1]
    atoms.set_cell(cell, scale_atoms=True)
    atoms.rattle(0.04, seed=7)
    return atoms


def via_socket(atoms, args: list[str], stress: bool):
    """Energy, forces and (stress or None) through the driver."""
    name = socket_name("ase")
    cmd = [DRIVER, *args, "-u", "-a", name, "--timeout", "60"]
    procs = []

    def launch(*_args, **_kwargs):
        proc = subprocess.Popen(cmd)
        procs.append(proc)
        return proc

    atoms = atoms.copy()
    with SocketIOCalculator(unixsocket=name, launch_client=launch) as calc:
        atoms.calc = calc
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        sigma = atoms.get_stress() if stress else None
    assert procs and procs[0].wait(timeout=60) == 0
    return energy, forces, sigma


def direct(atoms, calc, stress: bool):
    atoms = atoms.copy()
    atoms.calc = calc
    return (
        atoms.get_potential_energy(),
        atoms.get_forces(),
        atoms.get_stress() if stress else None,
    )


def assert_close(got, want, scale_forces: float, scale_stress: float, rtol: float):
    e, f, s = got
    e0, f0, s0 = want
    assert e == pytest.approx(e0, rel=rtol, abs=1e-12)
    np.testing.assert_allclose(f, f0, rtol=rtol, atol=rtol * scale_forces)
    if s0 is not None:
        np.testing.assert_allclose(s, s0, rtol=rtol, atol=rtol * scale_stress)


def lj_bulk():
    return sheared(bulk("Ar", "fcc", a=5.26, cubic=True).repeat((2, 2, 2)))


def test_lj_periodic_non_orthogonal(rgpot_calc):
    atoms = lj_bulk()
    want = direct(atoms, rgpot_calc("lj", **AR), stress=True)
    got = via_socket(atoms, ["--pot", LJ_SPEC, "--species", f"Ar*{len(atoms)}"], stress=True)
    scale_f = float(np.abs(want[1]).max())
    scale_s = float(np.abs(want[2]).max())
    assert_close(got, want, scale_f, scale_s, rtol=1e-7)


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def test_lj_tcp(rgpot_calc):
    atoms = lj_bulk()
    want = direct(atoms, rgpot_calc("lj", **AR), stress=True)
    port = free_port()
    cmd = [DRIVER, "--pot", LJ_SPEC, "--species", f"Ar*{len(atoms)}", "-a", "127.0.0.1", "-p", str(port)]
    procs = []

    def launch(*_args, **_kwargs):
        proc = subprocess.Popen(cmd)
        procs.append(proc)
        return proc

    with SocketIOCalculator(port=port, launch_client=launch) as calc:
        atoms.calc = calc
        got = (atoms.get_potential_energy(), atoms.get_forces(), atoms.get_stress())
    assert procs and procs[0].wait(timeout=60) == 0
    assert_close(got, want, float(np.abs(want[1]).max()), float(np.abs(want[2]).max()), rtol=1e-7)


def test_lj_cluster_open_box(rgpot_calc):
    atoms = bulk("Ar", "fcc", a=5.26, cubic=True).repeat((2, 2, 2))
    atoms.set_cell(np.zeros((3, 3)))
    atoms.pbc = False
    atoms.rattle(0.05, seed=3)
    want = direct(atoms, rgpot_calc("lj", **AR), stress=False)
    got = via_socket(atoms, ["--pot", LJ_SPEC, "--species", f"Ar*{len(atoms)}"], stress=False)
    assert_close(got, want, float(np.abs(want[1]).max()), 0.0, rtol=1e-7)


def test_eam_al_periodic_non_orthogonal(rgpot_calc):
    if "EAMAl" not in driver_names():
        pytest.skip("this build has no Fortran EAMAl")
    atoms = sheared(bulk("Al", "fcc", a=4.05).repeat((3, 3, 3)))
    want = direct(atoms, rgpot_calc("eam_al", numerical_stress=True), stress=True)
    got = via_socket(
        atoms,
        [
            "-m",
            "EAMAl",
            "--species",
            f"Al*{len(atoms)}",
            "--cutoff",
            str(EAM_AL_CUTOFF),
            "--stress-step",
            "1e-5",
        ],
        stress=True,
    )
    scale_f = float(np.abs(want[1]).max())
    scale_s = float(np.abs(want[2]).max())
    assert_close(got[:2] + (None,), want[:2] + (None,), scale_f, 0.0, rtol=1e-7)
    np.testing.assert_allclose(got[2], want[2], rtol=1e-5, atol=1e-5 * scale_s)


def test_species_from_xyz(rgpot_calc, tmp_path):
    atoms = lj_bulk()
    path = tmp_path / "start.xyz"
    atoms.write(path, format="extxyz")
    want = direct(atoms, rgpot_calc("lj", **AR), stress=True)
    got = via_socket(atoms, ["--pot", LJ_SPEC, "--species-from", str(path)], stress=True)
    assert_close(got, want, float(np.abs(want[1]).max()), float(np.abs(want[2]).max()), rtol=1e-7)


def test_wrong_species_count_fails():
    atoms = lj_bulk()
    name = socket_name("count")
    cmd = [DRIVER, "--pot", LJ_SPEC, "--species", f"Ar*{len(atoms) - 1}", "-u", "-a", name]
    procs = []

    def launch(*_args, **_kwargs):
        proc = subprocess.Popen(cmd, stderr=subprocess.PIPE, text=True)
        procs.append(proc)
        return proc

    with pytest.raises(Exception):
        with SocketIOCalculator(unixsocket=name, launch_client=launch) as calc:
            atoms.calc = calc
            atoms.get_potential_energy()
    proc = procs[0]
    assert proc.wait(timeout=60) == 1
    assert "does not match" in proc.stderr.read()


@pytest.mark.parametrize(
    "args",
    [
        ["--species", "Ar"],
        ["--pot", LJ_SPEC],
        ["--pot", "NoSuchPotential", "--species", "Ar"],
        ["--pot", LJ_SPEC, "--species", "Xx"],
        ["--pot", LJ_SPEC, "--species", "Ar", "--bogus"],
        ["--pot", LJ_SPEC, "-m", "LJ", "--species", "Ar"],
    ],
)
def test_usage_errors_exit_2(args):
    out = subprocess.run([DRIVER, *args], capture_output=True, text=True, timeout=30)
    assert out.returncode == 2
    assert out.stderr.startswith("rgpot-ipi-driver: ")


def test_list_and_help():
    names = driver_names()
    assert {"LJ", "Morse", "SuttonChen"} <= set(names)
    out = subprocess.run([DRIVER, "--help"], capture_output=True, text=True, check=True)
    assert out.stdout.startswith("usage: rgpot-ipi-driver")


IPI_INPUT = """<simulation verbosity='low' safe_stride='1000'>
  <output prefix='sim'>
    <properties stride='1' filename='out'> [ step, potential{{electronvolt}} ] </properties>
  </output>
  <total_steps>4</total_steps>
  <prng><seed>31415</seed></prng>
  <ffsocket name='rgpot' mode='unix' pbc='false'>
    <address>{name}</address>
    <latency>1e-4</latency>
  </ffsocket>
  <system>
    <initialize nbeads='2'>
      <file mode='xyz' units='angstrom'>init.xyz</file>
      <velocities mode='thermal' units='kelvin'>20</velocities>
    </initialize>
    <forces><force forcefield='rgpot'/></forces>
    <motion mode='dynamics'>
      <dynamics mode='nve'><timestep units='femtosecond'>2.0</timestep></dynamics>
    </motion>
    <ensemble><temperature units='kelvin'>20</temperature></ensemble>
  </system>
</simulation>
"""


def test_ipi_pimd_smoke(rgpot_calc, tmp_path):
    ipi = shutil.which("i-pi")
    if ipi is None:
        pytest.skip("i-pi is not on PATH")
    atoms = lj_bulk()
    a, b, c = atoms.cell.lengths()
    alpha, beta, gamma = atoms.cell.angles()
    lines = [
        str(len(atoms)),
        f"# CELL(abcABC): {a:.10f} {b:.10f} {c:.10f} {alpha:.10f} {beta:.10f} {gamma:.10f}"
        " positions{angstrom} cell{angstrom}",
    ]
    # i-PI builds its cell upper triangular from abcABC; ASE's cellpar
    # standard form matches, so rotate the positions into it.
    standard = atoms.copy()
    standard.set_cell(atoms.cell.cellpar(), scale_atoms=True)
    for sym, pos in zip(standard.get_chemical_symbols(), standard.positions):
        lines.append(f"{sym} {pos[0]:.12f} {pos[1]:.12f} {pos[2]:.12f}")
    (tmp_path / "init.xyz").write_text("\n".join(lines) + "\n")
    name = socket_name("ipi")
    (tmp_path / "input.xml").write_text(IPI_INPUT.format(name=name))
    server = subprocess.Popen(
        [ipi, "input.xml"], cwd=tmp_path, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    try:
        driver = subprocess.run(
            [DRIVER, "--pot", LJ_SPEC, "--species", f"Ar*{len(atoms)}", "-u", "-a", name, "--connect-timeout", "60"],
            capture_output=True,
            text=True,
            timeout=300,
        )
        log, _ = server.communicate(timeout=120)
    finally:
        if server.poll() is None:
            server.kill()
            server.wait()
    assert driver.returncode == 0, driver.stderr
    assert server.returncode == 0, log
    rows = [
        line.split()
        for line in (tmp_path / "sim.out").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert len(rows) == 5
    want = direct(standard, rgpot_calc("lj", **AR), stress=False)[0]
    assert float(rows[0][1]) == pytest.approx(want, rel=1e-6)
