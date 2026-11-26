#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Central-difference sign check for the (H2O)6 GFN2 arms.

The anneal side calls the same callback the water binary searches with.
The CREST side is ``negate_gradient_script`` from the crest-equal runner:
that script is what xtb reads back through ``--driver``.
"""

from __future__ import annotations

import importlib.util
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "bench" / "crest-equal" / "reference-prism.xyz"
PANEL = ROOT / "scripts" / "crest_equal" / "run_panel.py"
REPORT = ROOT / "bench" / "crest-equal" / "gfn2-fd-gate.txt"


def cargo_target() -> Path:
    text = (ROOT / ".cargo" / "config.toml").read_text()
    match = re.search(r'target-dir\s*=\s*"([^"]+)"', text)
    if match is None:
        return ROOT / "target"
    return (ROOT / match.group(1)).resolve()


TARGET = cargo_target()
PROBE = TARGET / "release" / "gfn2-fd-gate"
ANNEAL = TARGET / "release" / "examples" / "gfn2_water_h2h"
DRIVER = TARGET / "release" / "examples" / "gfn2_water_crest_driver"

MOLECULES = 6
STEP_A = 1.0e-3
SHIFT_A = 0.05
ABS_TOL_EV_PER_A = 2.0e-3
COSINE_MIN = 0.999
WALL_ONSET_A = 1.15
CORE = os.environ.get("CREST_EQUAL_CORE", "0")

THREAD_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "RAYON_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
    "GOTO_NUM_THREADS",
    "OMP_THREAD_LIMIT",
)


def threads() -> None:
    for name in THREAD_VARS:
        os.environ[name] = "1"
    os.environ.setdefault("RGPOT_XTB_ETEMP", "300")


def load_panel():
    spec = importlib.util.spec_from_file_location("crest_equal_panel", PANEL)
    if spec is None or spec.loader is None:
        raise SystemExit("could not load the crest-equal runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_xyz(path: Path) -> tuple[list[str], list[float]]:
    lines = path.read_text().splitlines()
    count = int(lines[0].split()[0])
    symbols: list[str] = []
    coords: list[float] = []
    for line in lines[2 : 2 + count]:
        parts = line.split()
        symbols.append(parts[0])
        coords.extend(float(part) for part in parts[1:4])
    if len(symbols) != count:
        raise SystemExit("short reference xyz")
    return symbols, coords


def write_xyz(path: Path, symbols: list[str], coords: list[float], comment: str) -> None:
    rows = [str(len(symbols)), comment]
    for index, symbol in enumerate(symbols):
        x, y, z = coords[3 * index : 3 * index + 3]
        rows.append(f"{symbol} {x:.14f} {y:.14f} {z:.14f}")
    path.write_text("\n".join(rows) + "\n")


def max_covalent_oh(coords: list[float]) -> float:
    worst = 0.0
    for molecule in range(MOLECULES):
        oxygen = 9 * molecule
        for hydrogen in (oxygen + 3, oxygen + 6):
            square = 0.0
            for axis in range(3):
                delta = coords[hydrogen + axis] - coords[oxygen + axis]
                square += delta * delta
            worst = max(worst, math.sqrt(square))
    return worst


def pin(cmd: list[str]) -> list[str]:
    if shutil.which("taskset"):
        return ["taskset", "-c", CORE, *cmd]
    return cmd


def run(cmd: list[str], cwd: Path | None = None, env: dict | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        pin(cmd),
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_probe(path: Path) -> dict:
    values: dict[str, object] = {}
    lines = path.read_text().splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        if line in {"g", "fd"}:
            count = int(str(values["n"]))
            block = [float(item) for item in lines[index + 1 : index + 1 + count]]
            values[line] = block
            index += 1 + count
            continue
        parts = line.split()
        if len(parts) == 2:
            key, raw = parts
            if key == "restart":
                values[key] = raw
            elif key == "n":
                values[key] = int(raw)
            else:
                values[key] = float(raw)
        index += 1
    return values


def vector_stats(gradient: list[float], finite: list[float]) -> dict[str, float]:
    if len(gradient) != len(finite) or not gradient:
        raise SystemExit("gradient and finite difference differ in length")
    abs_err = [abs(left - right) for left, right in zip(gradient, finite)]
    peak = max(max(abs(value) for value in gradient), max(abs(value) for value in finite))
    rel_all = []
    rel_large = []
    for left, right, err in zip(gradient, finite, abs_err):
        denom = max(abs(left), abs(right), 1e-12)
        rel_all.append(err / denom)
        if max(abs(left), abs(right)) >= 0.01 * peak:
            rel_large.append(err / max(abs(left), abs(right)))
    dot = sum(left * right for left, right in zip(gradient, finite))
    norm_g = math.sqrt(sum(value * value for value in gradient))
    norm_fd = math.sqrt(sum(value * value for value in finite))
    cosine = dot / (norm_g * norm_fd) if norm_g * norm_fd else math.nan
    flipped = max(abs(left + right) for left, right in zip(gradient, finite))
    return {
        "max_abs": max(abs_err),
        "max_rel": max(rel_all),
        "max_rel_large": max(rel_large) if rel_large else math.nan,
        "cosine": cosine,
        "norm_g": norm_g,
        "norm_fd": norm_fd,
        "flipped_max_abs": flipped,
    }


def judge(stats: dict[str, float], downhill: float) -> bool:
    return (
        math.isfinite(stats["cosine"])
        and stats["cosine"] >= COSINE_MIN
        and stats["max_abs"] <= ABS_TOL_EV_PER_A
        and stats["max_abs"] < stats["flipped_max_abs"]
        and math.isfinite(downhill)
        and downhill < 0.0
        and stats["norm_g"] > 1.0e-2
    )


def fmt(value: float) -> str:
    if not math.isfinite(value):
        return "nan"
    return f"{value:.6e}"


def version_line(binary: Path) -> str:
    proc = run([str(binary), "--version"])
    text = proc.stdout
    xtb_match = re.search(r"xtb version (\d+\.\d+\.\d+ \([^)]+\))", text)
    if xtb_match:
        return xtb_match.group(1)
    crest_version = re.search(r"Version (\d+\.\d+\.\d+)", text)
    crest_commit = re.search(r"commit \(([^)]+)\)", text)
    if crest_version and crest_commit:
        return f"{crest_version.group(1)} ({crest_commit.group(1)})"
    if crest_version:
        return crest_version.group(1)
    return "unavailable"


def audit_energy(xyz: Path, work: Path) -> float:
    env = dict(os.environ)
    env["H2H_AUDIT_XYZ"] = str(xyz)
    proc = run([str(ANNEAL), "6", "1", "2000", str(work)], env=env)
    if proc.returncode != 0:
        raise SystemExit(f"anneal audit exited {proc.returncode}")
    found = re.search(r"objective_eV:\s*([-+0-9.eE]+)", (work / "force-audit.txt").read_text())
    if found is None:
        raise SystemExit("anneal audit did not write objective_eV")
    return float(found.group(1))


def parse_gradient_file(text: str) -> tuple[float, list[float]]:
    energy = None
    gradient: list[float] = []
    for line in text.splitlines():
        if "SCF energy" in line:
            match = re.search(r"SCF energy =\s*([-+0-9.eE]+)", line)
            if match:
                energy = float(match.group(1))
        parts = line.split()
        if len(parts) == 3 and ("E" in parts[0] or "e" in parts[0]):
            gradient.extend(float(part) for part in parts)
    if energy is None or not gradient:
        raise SystemExit("gradient file had no energy or gradient")
    return energy, gradient


class CrestPath:
    def __init__(self, driver: Path, panel) -> None:
        self.short = Path(tempfile.mkdtemp(prefix="ce"))
        self.socket = self.short / "s"
        self.call = self.short / "c"
        self.work = self.short / "w"
        self.served = self.short / "d"
        self.work.mkdir()
        self.served.mkdir()
        self.call.write_text(panel.negate_gradient_script(driver, self.socket))
        self.call.chmod(0o755)
        self.log = open(self.served / "server.out", "w")
        self.server = subprocess.Popen(
            pin(
                [
                    str(driver),
                    "serve",
                    str(self.socket),
                    str(self.served),
                    str(MOLECULES),
                    "4000",
                    "-1e9",
                ]
            ),
            stdout=self.log,
            stderr=subprocess.STDOUT,
        )
        for _ in range(400):
            if (self.served / "ready").exists():
                break
            time.sleep(0.05)
        else:
            self.server.kill()
            raise SystemExit("crest driver did not become ready")

    def evaluate(self, symbols: list[str], coords: list[float]) -> tuple[float, list[float]]:
        write_xyz(self.work / "xtbdriver.xyz", symbols, coords, "gate")
        gradient = self.work / "gradient"
        if gradient.exists():
            gradient.unlink()
        proc = run([str(self.call)], cwd=self.work)
        if proc.returncode != 0 or not gradient.is_file():
            raise SystemExit(f"crest wrapper exited {proc.returncode}")
        return parse_gradient_file(gradient.read_text())

    def close(self) -> None:
        subprocess.run(
            [str(DRIVER), "stop", str(self.socket)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        try:
            self.server.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self.server.kill()
        self.log.close()


def crest_check(
    symbols: list[str], coords: list[float], hartree_ev: float, bohr_a: float
) -> tuple[dict[str, float], float, float]:
    panel = load_panel()
    path = CrestPath(DRIVER, panel)
    try:
        energy_ha, gradient_ha = path.evaluate(symbols, coords)
        finite_ha = []
        for index in range(len(coords)):
            plus = coords.copy()
            minus = coords.copy()
            plus[index] += STEP_A
            minus[index] -= STEP_A
            above = path.evaluate(symbols, plus)[0]
            below = path.evaluate(symbols, minus)[0]
            finite_ha.append((above - below) / (2.0 * STEP_A) * bohr_a)
        scale = hartree_ev / bohr_a
        gradient = [value * scale for value in gradient_ha]
        finite = [value * scale for value in finite_ha]
        stats = vector_stats(gradient, finite)
        norm = stats["norm_g"]
        step = 0.01 / norm
        again = path.evaluate(symbols, coords)[0]
        moved = [value - step * component for value, component in zip(coords, gradient)]
        lowered = path.evaluate(symbols, moved)[0]
        delta_ev = (lowered - again) * hartree_ev
        return stats, delta_ev, energy_ha * hartree_ev
    finally:
        path.close()


def require_tool(name: str) -> Path:
    found = shutil.which(name)
    if found is None:
        raise SystemExit(f"{name} is not on PATH")
    return Path(found)


def main() -> int:
    threads()
    if "RGPOT_XTB_ENGINE" not in os.environ:
        raise SystemExit("RGPOT_XTB_ENGINE is unset")
    engine = Path(os.environ["RGPOT_XTB_ENGINE"])
    if not engine.is_file():
        raise SystemExit("RGPOT_XTB_ENGINE does not name a file")
    for binary in (PROBE, ANNEAL, DRIVER):
        if not binary.is_file():
            raise SystemExit(f"missing binary {binary.name}")
    symbols, coords = read_xyz(REFERENCE)
    coords[0] += SHIFT_A
    covalent = max_covalent_oh(coords)
    if covalent > WALL_ONSET_A:
        raise SystemExit(f"shifted geometry enters the O-H wall: {covalent:.4f} A")
    xtb = require_tool("xtb")
    crest = require_tool("crest")
    with tempfile.TemporaryDirectory(prefix="fdgate-") as scratch:
        scratch_path = Path(scratch)
        xyz = scratch_path / "geom.xyz"
        write_xyz(xyz, symbols, coords, "shifted prism")
        once_path = scratch_path / "once.txt"
        fd_path = scratch_path / "fd.txt"
        audit_dir = scratch_path / "audit"
        audit_dir.mkdir()
        once = run([str(PROBE), "once", str(xyz), "0", str(once_path)])
        if once.returncode != 0:
            raise SystemExit(f"probe once exited {once.returncode}")
        finite = run([str(PROBE), "fd", str(xyz), f"{STEP_A}", "1", str(fd_path)])
        if finite.returncode != 0:
            raise SystemExit(f"probe fd exited {finite.returncode}")
        cold = parse_probe(once_path)
        warm = parse_probe(fd_path)
        audited = audit_energy(xyz, audit_dir)
        anneal_stats = vector_stats(warm["g"], warm["fd"])  # type: ignore[arg-type]
        anneal_delta = float(warm["downhill_delta_eV"])  # type: ignore[arg-type]
        anneal_pass = judge(anneal_stats, anneal_delta)
        correspond = abs(float(cold["energy_eV"]) - audited)  # type: ignore[arg-type]
        if correspond > 1.0e-5:
            anneal_pass = False
        hartree_ev = float(warm["hartree_eV"])  # type: ignore[arg-type]
        bohr_a = float(warm["bohr_A"])  # type: ignore[arg-type]
        crest_stats, crest_delta, crest_energy = crest_check(symbols, coords, hartree_ev, bohr_a)
        crest_pass = judge(crest_stats, crest_delta)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        check=True,
    ).stdout.strip()
    lines = [
        "system h2o6",
        "geometry reference prism, atom 1 x shifted by +0.05 angstrom",
        f"max_covalent_OH_A {covalent:.6f}",
        f"wall_onset_A {WALL_ONSET_A}",
        f"step_A {STEP_A:.1e}",
        "step_source tests/rgpot_xtb_gradient.rs STEP_A",
        f"abs_tol_eV_per_A {ABS_TOL_EV_PER_A:.1e}",
        f"cosine_min {COSINE_MIN}",
        "anneal_path ForceObjective value_and_gradient, restart once, then central differences",
        "crest_path negate_gradient_script gradient file read by xtb --driver",
        f"anneal_energy_eV {fmt(float(warm['energy_eV']))}",
        f"anneal_audit_energy_eV {fmt(audited)}",
        f"anneal_cold_energy_eV {fmt(float(cold['energy_eV']))}",
        f"anneal_audit_delta_eV {fmt(correspond)}",
        f"anneal_norm_eV_per_A {fmt(anneal_stats['norm_g'])}",
        f"anneal_fd_norm_eV_per_A {fmt(anneal_stats['norm_fd'])}",
        f"anneal_max_abs_eV_per_A {fmt(anneal_stats['max_abs'])}",
        f"anneal_max_rel {fmt(anneal_stats['max_rel'])}",
        f"anneal_max_rel_large {fmt(anneal_stats['max_rel_large'])}",
        f"anneal_cosine {fmt(anneal_stats['cosine'])}",
        f"anneal_flipped_max_abs_eV_per_A {fmt(anneal_stats['flipped_max_abs'])}",
        f"anneal_downhill_delta_eV {fmt(anneal_delta)}",
        f"anneal_pass {'true' if anneal_pass else 'false'}",
        f"crest_energy_eV {fmt(crest_energy)}",
        f"crest_norm_eV_per_A {fmt(crest_stats['norm_g'])}",
        f"crest_fd_norm_eV_per_A {fmt(crest_stats['norm_fd'])}",
        f"crest_max_abs_eV_per_A {fmt(crest_stats['max_abs'])}",
        f"crest_max_rel {fmt(crest_stats['max_rel'])}",
        f"crest_max_rel_large {fmt(crest_stats['max_rel_large'])}",
        f"crest_cosine {fmt(crest_stats['cosine'])}",
        f"crest_flipped_max_abs_eV_per_A {fmt(crest_stats['flipped_max_abs'])}",
        f"crest_downhill_delta_eV {fmt(crest_delta)}",
        f"crest_pass {'true' if crest_pass else 'false'}",
        f"hartree_eV {hartree_ev:.16e}",
        f"bohr_A {bohr_a:.16e}",
        f"xtb_version {version_line(xtb)}",
        f"crest_version {version_line(crest)}",
        f"engine_sha256 {sha256(engine)}",
        f"anneal_sha256 {sha256(ANNEAL)}",
        f"driver_sha256 {sha256(DRIVER)}",
        f"probe_sha256 {sha256(PROBE)}",
        f"revision {revision}",
        f"gate {'pass' if anneal_pass and crest_pass else 'fail'}",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print(f"gate {'pass' if anneal_pass and crest_pass else 'fail'}")
    print(f"anneal cosine {fmt(anneal_stats['cosine'])} max_abs {fmt(anneal_stats['max_abs'])}")
    print(f"crest cosine {fmt(crest_stats['cosine'])} max_abs {fmt(crest_stats['max_abs'])}")
    return 0 if anneal_pass and crest_pass else 1


if __name__ == "__main__":
    sys.exit(main())
