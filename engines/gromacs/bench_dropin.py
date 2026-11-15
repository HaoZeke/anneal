#!/usr/bin/env python3
"""Per-step cost of the GROMACS drop-in against bare rgpot evaluation.

Each system first runs --short + --steps MD steps with the drop-in and
keeps every frame. Every number below is then a median over --repeats
fresh processes, in microseconds per step or call, on one thread
(OMP_NUM_THREADS=1, mdrun -ntomp 1):

fixed  pot_spec's eval of the first frame, 200 calls of the same
       geometry. A kernel with a Verlet skin reuses its list throughout.
bare   pot_spec's eval of each recorded frame in order, wrapped into the
       cell as the shim wraps it: the mean, so the neighbour-list
       rebuilds the trajectory forces are counted.
shim   the same frames, unwrapped as GROMACS holds them, through
       librgpot_gromacs with the calls GROMACS makes each step.
off    gmx mdrun of the same topology with QM/MM off. Every atom has zero
       charge and zero Lennard-Jones, so GROMACS integrates free particles.
on     gmx mdrun with the drop-in.

The mdrun figures are (t(short + steps) - t(short)) / steps from the wall
time on the "Time:" line of md.log, so start-up and the force-environment
set-up cancel. The drop-in overhead per step is on - off - bare.

    bench_dropin.py --gmx gmx_d --build BUILD_DIR [--eam CuNi.eam.alloy]

BUILD_DIR is an rgpot build with -Dwith_gromacs_dropin=enabled; its
librgpot_gromacs is put first on LD_LIBRARY_PATH, so a GROMACS linked
against another rgpot build directory uses this one.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dropin_case as dc  # noqa: E402
import test_dropin_api as api  # noqa: E402


def systems(eam: str | None):
    x, box = dc.fcc(0.53, 6, jiggle=0.01, seed=1)
    yield "lj-argon", ["Ar"] * len(x), x, box, ["LJ", "u0", "0.0104", "psi", "3.4",
                                                 "cutoff", "7.5"]
    x, box = dc.diamond(0.5431, 4, jiggle=0.004)
    yield "tersoff-silicon", ["Si"] * len(x), x, box, ["TersoffSi"]
    x, box = dc.fcc(0.3924, 6, jiggle=0.006)
    yield "suttonchen-platinum", ["Pt"] * len(x), x, box, ["SuttonChen", "preset",
                                                            "platinum", "cutoff", "5.0"]
    if eam:
        rng = np.random.default_rng(4)
        x, box = dc.fcc(0.3560, 8, jiggle=0.004, seed=9)
        symbols = [("Ni" if rng.random() < 0.3 else "Cu") for _ in range(len(x))]
        yield "eamtable-cuni", symbols, x, box, ["EAMTable", "file", eam]


def rgpot_section(spec) -> str:
    text = f"    POTENTIAL {spec[0]}"
    if len(spec) > 1:
        text += "\n    OPTIONS " + " ".join(spec[1:])
    return text


def frames_text(symbols, box_nm, frames_nm) -> str:
    """rgpot-gromacs-bench input: the first frame as rgpot-eval reads it, then the rest."""
    lines = [str(len(symbols)), " ".join(f"{v:.17g}" for v in (box_nm * 10).ravel())]
    first = frames_nm[0] * 10
    lines += [f"{dc.SYMBOLS.index(s)} {a:.17g} {b:.17g} {c:.17g}"
              for s, (a, b, c) in zip(symbols, first)]
    for frame in frames_nm[1:]:
        lines.append(" ".join(f"{v:.17g}" for v in (frame * 10).ravel()))
    return "\n".join(lines) + "\n"


def timed(cmd, text: str, cwd: Path) -> float:
    out = subprocess.run(cmd, input=text, text=True, capture_output=True, cwd=cwd,
                         check=True).stdout.split()
    return float(out[out.index("mean") + 1])


def wall(work: Path) -> float:
    for line in (work / "md.log").read_text().splitlines():
        m = re.match(r"\s+Time:\s+(\S+)\s+(\S+)", line)
        if m:
            return float(m.group(2))
    raise RuntimeError(f"no Time: line in {work / 'md.log'}")


def mdrun(gmx, work, symbols, x, box, spec, on: bool, nsteps: int, keep: bool = False) -> float:
    mdp = {"nsteps": nsteps, "nstxout": 1 if keep else 0, "nstvout": 0, "nstfout": 0,
           "nstenergy": 0, "nstcalcenergy": 100, "nstlog": 0}
    if not on:
        mdp["qmmm-cp2k-active"] = "false"
    dc.write_system(work, symbols=symbols, positions=x, box=box, mdp=mdp,
                    qmi_text=dc.qmi(rgpot_section(spec)) if on else None)
    dc.run(gmx, work, ntomp=1)
    return wall(work)


def hardware() -> dict[str, str]:
    info = {"machine": platform.machine(), "cpus": str(os.cpu_count())}
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            info["cpu"] = line.split(":", 1)[1].strip()
            break
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemTotal"):
            info["memory"] = line.split(":", 1)[1].strip()
    return info


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--gmx", required=True)
    p.add_argument("--build", required=True, type=Path)
    p.add_argument("--eam", default=None)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--short", type=int, default=20)
    p.add_argument("--json", type=Path, default=None)
    args = p.parse_args()

    bench = str((args.build / "engines" / "gromacs" / "rgpot-gromacs-bench").resolve())
    shim_dir = str((args.build / "engines" / "gromacs").resolve())
    old = [d for d in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep) if d]
    os.environ.update(OMP_NUM_THREADS="1", LD_LIBRARY_PATH=os.pathsep.join([shim_dir] + old))

    report = {"hardware": hardware(), "gromacs": dc.gmx_version(args.gmx),
              "repeats": args.repeats, "steps": args.steps, "short": args.short,
              "systems": []}
    print(f"# {report['hardware']}, GROMACS {report['gromacs']}, median of {args.repeats} "
          f"repeats, {args.steps} timed MD steps, microseconds per step")
    print(f"{'system':20s} {'atoms':>5s} {'fixed':>9s} {'bare':>9s} {'shim':>9s} "
          f"{'off':>9s} {'on':>9s} {'overhead':>9s}")
    with tempfile.TemporaryDirectory(prefix="rgpot-gmx-bench-") as tmp:
        root = Path(tmp)
        for name, symbols, x, box, spec in systems(args.eam):
            work = root / name
            mdrun(args.gmx, work / "record", symbols, x, box, spec, True,
                  args.short + args.steps, keep=True)
            frames = [f["x"] for f in dc.read_trr(work / "record" / "md.trr")]
            frames = frames[args.short:args.short + args.steps]
            wrapped = [dc.wrap(f, box) for f in frames]
            inp = api.write_input(work / "shim", rgpot_section(spec), symbols, wrapped[0])
            fixed_in = frames_text(symbols, box, wrapped[:1])
            bare_in = frames_text(symbols, box, wrapped)
            shim_in = frames_text(symbols, box, frames)
            cols = {k: [] for k in ("fixed", "bare", "shim", "off", "on")}
            for _ in range(args.repeats):
                cols["fixed"].append(timed([bench, "bare", *spec], fixed_in, work))
                cols["bare"].append(timed([bench, "bare", *spec], bare_in, work))
                cols["shim"].append(timed([bench, "shim", str(inp)], shim_in, work / "shim"))
                for key, on in (("off", False), ("on", True)):
                    t = [mdrun(args.gmx, work / "md", symbols, x, box, spec, on, n)
                         for n in (args.short, args.short + args.steps)]
                    cols[key].append(1e6 * (t[1] - t[0]) / args.steps)
            row = {"system": name, "atoms": len(x), "potential": spec[0],
                   **{k: statistics.median(v) for k, v in cols.items()}, "raw": cols}
            row["overhead"] = row["on"] - row["off"] - row["bare"]
            report["systems"].append(row)
            print(f"{name:20s} {len(x):5d} {row['fixed']:9.1f} {row['bare']:9.1f} "
                  f"{row['shim']:9.1f} {row['off']:9.1f} {row['on']:9.1f} "
                  f"{row['overhead']:9.1f}", flush=True)
    if args.json:
        args.json.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
