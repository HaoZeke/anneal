# Copyright (c) anneal developers
"""The G1 Morse panel data files follow one reproducible rule."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "scripts" / "morse_g1_panel" / "datafile.py"
BASE = ROOT / "examples" / "h2h_gmin_morse.data"
OUTPUT = ROOT / "bench" / "morse-g1-panel" / "data"


def load_module():
    spec = importlib.util.spec_from_file_location("morse_g1_datafile", MODULE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def keywords(text: str) -> list[str]:
    return [
        line
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("!")
    ]


def test_every_data_file_is_regenerated_from_n_and_rho(tmp_path: Path) -> None:
    module = load_module()
    module.generate(tmp_path, BASE)
    base = keywords(BASE.read_text())
    for name, n, rho in module.CELLS:
        generated = (tmp_path / f"{name}.data").read_text()
        recorded = (OUTPUT / f"{name}.data").read_text()
        assert generated == recorded
        lines = keywords(generated)
        assert lines[:-2] == base
        assert lines[-2] == f"MORSE {rho:.1f}"
        assert lines[-1] == f"RADIUS {module.radius(n):.10f}"
        assert not any(line.startswith("RANSEED ") for line in lines)


def test_morse_radius_has_no_default_potential_scale() -> None:
    module = load_module()
    for _name, n, _rho in module.CELLS:
        unscaled = 2.0 + (3.0 * n / module.DENSITY) ** (1.0 / 3.0)
        assert module.radius(n) == unscaled
        assert module.radius(n) != unscaled * 2.0 ** (1.0 / 6.0)
