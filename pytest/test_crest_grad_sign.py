# Copyright (c) anneal developers
"""Crest-equal call script gradient against a central difference.

The script is what xtb reads back through ``--driver``. The driver's
gradient file is already dE/dx. On the shifted water prism a reversed
file has cosine near -1 against the slope of its own energy.
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

import pytest

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "bench" / "crest-equal" / "reference-prism.xyz"
PANEL = ROOT / "scripts" / "crest_equal" / "run_panel.py"
STEP_A = 1.0e-3
SHIFT_A = 0.05
MOLECULES = 6

sys.path.insert(0, str(ROOT / "scripts"))
from codata2018 import BOHR_ANGSTROM, HARTREE_EV  # noqa: E402


def _cargo_target() -> Path:
    text = (ROOT / ".cargo" / "config.toml").read_text()
    match = re.search(r'target-dir\s*=\s*"([^"]+)"', text)
    if match is None:
        return ROOT / "target"
    return (ROOT / match.group(1)).resolve()


def _driver() -> Path | None:
    named = os.environ.get("GFN2_WATER_CREST_DRIVER", "")
    if named:
        path = Path(named)
        return path if path.is_file() else None
    path = _cargo_target() / "release" / "examples" / "gfn2_water_crest_driver"
    return path if path.is_file() else None


def _load_panel():
    spec = importlib.util.spec_from_file_location("crest_equal_panel_grad", PANEL)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the crest-equal runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_xyz(path: Path) -> tuple[list[str], list[float]]:
    lines = path.read_text().splitlines()
    count = int(lines[0].split()[0])
    symbols: list[str] = []
    coords: list[float] = []
    for line in lines[2 : 2 + count]:
        parts = line.split()
        symbols.append(parts[0])
        coords.extend(float(part) for part in parts[1:4])
    if len(symbols) != count or count != 3 * MOLECULES:
        raise RuntimeError("reference xyz is not (H2O)6")
    return symbols, coords


def _write_xyz(path: Path, symbols: list[str], coords: list[float]) -> None:
    rows = [str(len(symbols)), "gate"]
    for index, symbol in enumerate(symbols):
        x, y, z = coords[3 * index : 3 * index + 3]
        rows.append(f"{symbol} {x:.14f} {y:.14f} {z:.14f}")
    path.write_text("\n".join(rows) + "\n")


def _parse_gradient(text: str) -> tuple[float, list[float]]:
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
        raise RuntimeError("gradient file had no energy or gradient")
    return energy, gradient


class _CrestCall:
    def __init__(self, driver: Path, panel) -> None:
        self.driver = driver
        self.short = Path(tempfile.mkdtemp(prefix="ce"))
        self.socket = self.short / "s"
        self.call = self.short / "c"
        self.work = self.short / "w"
        self.served = self.short / "d"
        self.work.mkdir()
        self.served.mkdir()
        self.call.write_text(panel.negate_gradient_script(driver, self.socket))
        self.call.chmod(0o755)
        env = dict(os.environ)
        for name in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "RAYON_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            env[name] = "1"
        env.setdefault("RGPOT_XTB_ETEMP", "300")
        self.log = open(self.served / "server.out", "w")
        self.server = subprocess.Popen(
            [
                str(driver),
                "serve",
                str(self.socket),
                str(self.served),
                str(MOLECULES),
                "4000",
                "-1e9",
            ],
            stdout=self.log,
            stderr=subprocess.STDOUT,
            env=env,
        )
        ready = False
        for _ in range(400):
            if (self.served / "ready").exists():
                ready = True
                break
            if self.server.poll() is not None:
                break
            time.sleep(0.05)
        if not ready:
            self.server.kill()
            raise RuntimeError(f"crest driver did not become ready\n{self._server_tail()}")

    def _server_tail(self) -> str:
        self.log.flush()
        return (self.served / "server.out").read_text(errors="replace")[-2000:]

    def evaluate(self, symbols: list[str], coords: list[float]) -> tuple[float, list[float]]:
        _write_xyz(self.work / "xtbdriver.xyz", symbols, coords)
        gradient = self.work / "gradient"
        if gradient.exists():
            gradient.unlink()
        proc = subprocess.run(
            [str(self.call)],
            cwd=self.work,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if proc.returncode != 0 or not gradient.is_file():
            raise RuntimeError(
                f"crest wrapper exited {proc.returncode}: {proc.stdout}\n{self._server_tail()}"
            )
        return _parse_gradient(gradient.read_text())

    def close(self) -> None:
        subprocess.run(
            [str(self.driver), "stop", str(self.socket)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        try:
            self.server.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self.server.kill()
        self.log.close()


def _cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    norm_l = math.sqrt(sum(a * a for a in left))
    norm_r = math.sqrt(sum(b * b for b in right))
    if norm_l == 0.0 or norm_r == 0.0:
        return math.nan
    return dot / (norm_l * norm_r)


@pytest.mark.skipif(shutil.which("xtb") is None, reason="xtb is absent")
def test_crest_wrapper_gradient_matches_central_differences():
    driver = _driver()
    if driver is None:
        pytest.skip("gfn2_water_crest_driver is absent")
    if not os.environ.get("RGPOT_XTB_ENGINE"):
        pytest.skip("RGPOT_XTB_ENGINE is unset")
    symbols, coords = _read_xyz(REFERENCE)
    coords[0] += SHIFT_A
    panel = _load_panel()
    path = _CrestCall(driver, panel)
    try:
        _energy, gradient_ha = path.evaluate(symbols, coords)
        finite_ha = []
        for index in range(len(coords)):
            plus = coords.copy()
            minus = coords.copy()
            plus[index] += STEP_A
            minus[index] -= STEP_A
            above = path.evaluate(symbols, plus)[0]
            below = path.evaluate(symbols, minus)[0]
            finite_ha.append((above - below) / (2.0 * STEP_A) * BOHR_ANGSTROM)
        scale = HARTREE_EV / BOHR_ANGSTROM
        gradient = [value * scale for value in gradient_ha]
        finite = [value * scale for value in finite_ha]
        cosine = _cosine(gradient, finite)
        max_abs = max(abs(a - b) for a, b in zip(gradient, finite))
        flipped = max(abs(a + b) for a, b in zip(gradient, finite))
        norm_g = math.sqrt(sum(value * value for value in gradient))
        norm_fd = math.sqrt(sum(value * value for value in finite))
        summary = (
            f"cosine {cosine:.6f} max_abs {max_abs:.6e} eV/A "
            f"flipped_max_abs {flipped:.6e} norm_g {norm_g:.6e} norm_fd {norm_fd:.6e}"
        )
        print(summary)
        assert cosine > 0.999, summary
    finally:
        path.close()
