#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Relax the archived GFN2-xTB water hexamers with xtb 6.7.1 and keep the lowest of each class.

Every archived minimum of the GFN2-xTB archives that data/system_visuals/provenance.txt
records (data/hq_sci_h2o6/ and data/w6gofee4k/) is written as an xyz file and relaxed by

    xtb input.xyz --gfn 2 --opt vtight

on one thread in an empty folder. A relaxed geometry of six whole molecules takes the
contact class of its own oxygens, the convention of system_visuals.py, which need not be
the class of its source; one that is not whole molecules, or whose optimization xtb does
not report converged, is counted and not drawn.
The lowest relaxed geometry of each class, ties broken by source file and row, is xtb's
xtbopt.xyz copied byte for byte to data/system_visuals/hexamer_gfn2/<class>.xyz. RECORD in
that folder gives the xtb build, the release tarball, the command, BASE (a develop commit
the checkout must descend from), the git blob of this script, the number of minima relaxed
into each class, and the source and energies of each drawn hexamer.

The script stops when xtb is not the 6.7.1 release build, when the tarball is not the
release, when the tree has changes or does not descend from BASE, when xtb exits with an
error, or when no relaxed geometry is of a class. --check relaxes the archives again and
stops unless every hexamer file is the one written.

Usage: hexamer_gfn2_relax.py --xtb PATH --tarball PATH [--check]
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import system_visuals

OUT = system_visuals.DATA / "hexamer_gfn2"
ENGINE = "GFN2-xTB"
XTB_VERSION = "xtb version 6.7.1 (edcfbbe) compiled by 'albert@albert-system' on 2024-07-22"
RELEASE = "https://github.com/grimme-lab/xtb/releases/download/v6.7.1/xtb-6.7.1-linux-x86_64.tar.xz"
RELEASE_SHA256 = "62a8d18778286e815292ee53d76ce447daf460a4dea3782c0f25cbac7019b5df"
COMMAND = ("--gfn", "2", "--opt", "vtight")
BASE = "dd6ff9273be30ab317bfd5508596f4beb19cb3a4"
THREADS = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "OMP_STACKSIZE": "1G"}


def die(message: str) -> None:
    raise SystemExit(f"hexamer_gfn2_relax: {message}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=system_visuals.ROOT, capture_output=True, text=True,
                          check=True).stdout.strip()


def candidates() -> list[tuple[float, tuple[str, int], list[str], list[tuple[float, float, float]]]]:
    """Every archived GFN2-xTB minimum, with the energy its archive gives and its file and row."""
    _, digests = system_visuals.provenance()
    return system_visuals._hexamer_candidates(system_visuals.DATA, digests, ENGINE)


def whole(symbols, coords) -> bool:
    """Whether the atoms are six whole water molecules by the O--H rule of system_visuals.py."""
    try:
        system_visuals._waters(symbols, coords)
    except SystemExit:
        return False
    return True


def xtb_version(xtb: Path) -> str:
    out = subprocess.run([str(xtb), "--version"], capture_output=True, text=True, check=False).stdout
    line = next((line.strip(" *") for line in out.splitlines() if "xtb version" in line), "")
    if line != XTB_VERSION:
        die(f"{xtb} reports {line!r}, not {XTB_VERSION!r}")
    return line


def relax(xtb: Path, symbols, coords) -> tuple[bytes, float, float, bool]:
    """xtbopt.xyz of one relaxation, its energy in Eh, its gradient norm in Eh/bohr, and whether xtb
    reported convergence."""
    with tempfile.TemporaryDirectory() as work:
        folder = Path(work)
        lines = [str(len(symbols)), ""] + [f"{s} {x:.10f} {y:.10f} {z:.10f}" for s, (x, y, z) in zip(symbols, coords)]
        (folder / "input.xyz").write_text("\n".join(lines) + "\n")
        env = {key: value for key, value in os.environ.items() if not key.startswith(("OMP_", "MKL_", "XTB"))}
        env.update(THREADS)
        env["XTBPATH"] = str(xtb.resolve().parents[1] / "share" / "xtb")
        done = subprocess.run([str(xtb), "input.xyz", *COMMAND], cwd=folder, env=env, capture_output=True, text=True,
                              check=False)
        if done.returncode != 0 or not (folder / "xtbopt.xyz").is_file():
            die(f"xtb failed:\n{done.stdout[-2000:]}\n{done.stderr[-2000:]}")
        converged = "GEOMETRY OPTIMIZATION CONVERGED" in done.stdout
        relaxed = (folder / "xtbopt.xyz").read_bytes()
    header = relaxed.decode().splitlines()[1].split()
    return (relaxed, float(header[header.index("energy:") + 1]), float(header[header.index("gnorm:") + 1]),
            converged)


def written(xtb: Path, tarball: Path) -> dict[str, bytes]:
    if sha256(tarball) != RELEASE_SHA256:
        die(f"{tarball} is not the release {RELEASE}")
    version = xtb_version(xtb)
    script = Path(__file__).resolve()
    if git("status", "--porcelain", "--untracked-files=no"):
        die("the tree has changes; run at a commit")
    if subprocess.run(["git", "merge-base", "--is-ancestor", BASE, "HEAD"], cwd=system_visuals.ROOT,
                      check=False).returncode != 0:
        die(f"the checkout does not descend from {BASE}")
    blob = git("hash-object", str(script))
    files: dict[str, bytes] = {}
    record = [
        f"xtb_version {version}",
        f"xtb_release {RELEASE}",
        f"xtb_release_sha256 {RELEASE_SHA256}",
        f"xtb_binary_sha256 {sha256(xtb)}",
        f"command xtb input.xyz {' '.join(COMMAND)}",
        "environment " + " ".join(f"{key}={value}" for key, value in THREADS.items()),
        f"base_commit {BASE}",
        f"script paper/jctc/scripts/hexamer_gfn2_relax.py blob {blob}",
    ]
    found = candidates()
    lowest: dict[str, tuple] = {}
    counts: dict[str, int] = {}
    for energy, (name, row), symbols, coords in found:
        relaxed, hartree, gnorm, converged = relax(xtb, symbols, coords)
        _, out_symbols, out_coords = system_visuals._xyz(relaxed.decode())
        if [s.upper() for s in out_symbols] != [s.upper() for s in symbols]:
            die(f"the relaxation of {name} row {row} does not keep its atoms")
        if not converged:
            cls = "not_converged"
        elif not whole(out_symbols, out_coords):
            cls = "not_whole"
        else:
            cls = system_visuals.contact_class(out_symbols, out_coords)
        counts[cls] = counts.get(cls, 0) + 1
        source = system_visuals.contact_class(symbols, coords) if whole(symbols, coords) else "not_whole"
        entry = (hartree, name, row, relaxed, gnorm, energy, source)
        if cls in system_visuals.HEXAMER_CLASSES and (cls not in lowest or entry[:3] < lowest[cls][:3]):
            lowest[cls] = entry
    record.append(f"relaxed {len(found)} archived minima; by relaxed class "
                  + " ".join(f"{cls}={counts[cls]}" for cls in sorted(counts)))
    for cls in system_visuals.HEXAMER_CLASSES:
        if cls not in lowest:
            die(f"no relaxed hexamer is a {cls}")
        hartree, name, row, relaxed, gnorm, energy, source = lowest[cls]
        files[f"{cls}.xyz"] = relaxed
        record.append(f"{cls} source {name} row {row} archive_class {source} archive_energy_eV {energy!r} "
                      f"relaxed_energy_Eh {hartree!r} gnorm_Eh_per_bohr {gnorm!r}")
    files["RECORD"] = ("\n".join(record) + "\n").encode()
    return files


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--xtb", type=Path, required=True)
    parser.add_argument("--tarball", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    if shutil.which(str(args.xtb)) is None and not args.xtb.is_file():
        die(f"{args.xtb} is not an executable")
    files = written(args.xtb, args.tarball)
    if args.check:
        stale = [name for name, body in files.items()
                 if name != "RECORD" and (not (OUT / name).is_file() or (OUT / name).read_bytes() != body)]
        if stale:
            die(f"{', '.join(stale)} differ from a fresh relaxation")
        return 0
    OUT.mkdir(exist_ok=True)
    for name, body in files.items():
        (OUT / name).write_bytes(body)
    return 0


if __name__ == "__main__":
    sys.exit(main())
