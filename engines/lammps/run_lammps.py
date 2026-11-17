#!/usr/bin/env python3
"""Run a two-atom LAMMPS job and compare it with rgpot-lj.

The first comparison is pair_style lj/cut with pair_modify shift, the
LAMMPS form of the shifted 12-6 potential. When a pair_rgpot plugin is
passed as the second argument, the same geometry is repeated with
pair_style rgpot.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GEOM = ROOT / "compare_geom.txt"


def rgpot_energy(tool: str) -> tuple[float, list[tuple[float, float, float]]]:
    text = subprocess.check_output([tool], input=GEOM.read_text(), text=True)
    energy = None
    forces = []
    for line in text.splitlines():
        parts = line.split()
        if parts[0] == "energy":
            energy = float(parts[1])
        elif parts[0] == "force":
            forces.append((float(parts[2]), float(parts[3]), float(parts[4])))
    if energy is None or len(forces) != 2:
        raise SystemExit(text)
    return energy, forces


def lammps_energy(commands: list[str]) -> tuple[float, list[tuple[float, float, float]]]:
    from lammps import lammps

    lmp = lammps(cmdargs=["-log", "none", "-screen", "none", "-nocite"])
    try:
        for command in commands:
            lmp.command(command)
        energy = float(lmp.get_thermo("pe"))
        forces_np = lmp.numpy.extract_atom("f")
        forces = [
            (float(forces_np[0][0]), float(forces_np[0][1]), float(forces_np[0][2])),
            (float(forces_np[1][0]), float(forces_np[1][1]), float(forces_np[1][2])),
        ]
    finally:
        lmp.close()
    return energy, forces


def close(label: str, got, ref) -> None:
    energy, forces = got
    ref_energy, ref_forces = ref
    if abs(energy - ref_energy) > 1e-8:
        raise SystemExit(f"{label} energy {energy} ref {ref_energy}")
    for index, (force, ref_force) in enumerate(zip(forces, ref_forces)):
        for component, ref_component in zip(force, ref_force):
            if abs(component - ref_component) > 1e-6:
                raise SystemExit(f"{label} force {index} {force} ref {ref_force}")
    print(f"{label} energy {energy:.10f}")


def lmp_energy(binary: str, plugin: Path) -> float:
    deck = Path("in.rgpot.run")
    deck.write_text(
        "\n".join(
            [
                "units metal",
                "atom_style atomic",
                "boundary p p p",
                "region box block 0 20 0 20 0 20",
                "create_box 1 box",
                "create_atoms 1 single 0 0 0",
                "create_atoms 1 single 1.5 0 0",
                "mass 1 1.0",
                f"plugin load {plugin}",
                "pair_style rgpot 8.0",
                "pair_coeff * * 1.0 1.0",
                "run 0",
                "",
            ]
        )
    )
    env = os.environ.copy()
    env.pop("LD_PRELOAD", None)
    lib_dir = str(Path(binary).resolve().parent)
    rgpot_lib = str(plugin.resolve().parent.parent / "CppCore")
    env["LD_LIBRARY_PATH"] = ":".join(
        part for part in (lib_dir, rgpot_lib, env.get("LD_LIBRARY_PATH", "")) if part
    )
    # A conda liblammps earlier on the linker's RPATH must not hide the
    # liblammps this binary was built with.
    env["LD_PRELOAD"] = str(Path(lib_dir) / "liblammps.so")
    log = Path("lmp.log")
    subprocess.run(
        [binary, "-in", str(deck), "-log", str(log), "-screen", "none", "-nocite"],
        check=True, env=env,
    )
    lines = log.read_text().splitlines()
    for index, line in enumerate(lines):
        if line.split()[:1] == ["Step"] and index + 1 < len(lines):
            parts = lines[index + 1].split()
            return float(parts[2])
    raise SystemExit(log.read_text())


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: run_lammps.py rgpot-lj [pair_rgpot.so lmp]")
    ref = rgpot_energy(sys.argv[1])
    setup = [
        "units metal",
        "atom_style atomic",
        "boundary p p p",
        "region box block 0 20 0 20 0 20",
        "create_box 1 box",
        "create_atoms 1 single 0 0 0",
        "create_atoms 1 single 1.5 0 0",
        "mass 1 1.0",
    ]
    cut = setup + [
        "pair_style lj/cut 8.0",
        "pair_modify shift yes",
        "pair_coeff * * 1.0 1.0",
        "run 0",
    ]
    close("lj/cut", lammps_energy(cut), ref)
    binary = sys.argv[3] if len(sys.argv) > 3 else os.environ.get("RGPOT_LMP", "")
    if len(sys.argv) > 2 and binary:
        plugin = Path(sys.argv[2]).resolve()
        energy = lmp_energy(binary, plugin)
        if abs(energy - ref[0]) > 1e-8:
            raise SystemExit(f"pair_style rgpot energy {energy} ref {ref[0]}")
        print(f"pair_style rgpot energy {energy:.10f}")


if __name__ == "__main__":
    main()
