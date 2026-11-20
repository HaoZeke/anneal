# Copyright (c) anneal developers
"""Each Lennard-Jones size gets its own GMIN data file and container."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "scripts" / "gmin_panel" / "run_panel.py"
BASE = ROOT / "examples" / "h2h_gmin_lj.data"
# source/io1.f90 literal, then the default-potential factor 2**(1/6).
DENSITY = 17.77153175


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def radius_text(n: int) -> str:
    value = (2.0 + (3.0 * n / DENSITY) ** (1.0 / 3.0)) * 2.0 ** (1.0 / 6.0)
    return f"{value:.10f}"


def keyword_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.strip() and not line.startswith("!")]


@pytest.fixture(scope="module")
def panel():
    return _load("gmin_run_panel", PANEL)


def test_each_lj_size_gets_its_own_data_file_and_radius(panel) -> None:
    shared = BASE.read_text()
    if not shared.endswith("\n"):
        shared += "\n"
    base_keywords = keyword_lines(shared)
    seen: dict[str, str] = {}
    for name, kind, n, *_rest in panel.SYSTEMS:
        if kind != "lj":
            continue
        text = panel.gmin_data(name, 2000, shared, "")
        lines = text.splitlines()
        radii = [line for line in lines if line.startswith("RADIUS ")]
        assert radii == [f"RADIUS {radius_text(n)}"], name
        kept = [
            line
            for line in keyword_lines(text)
            if line != "RANSEED 2000" and not line.startswith("RADIUS ")
        ]
        assert kept == base_keywords, name
        assert lines[-1] == "RANSEED 2000"
        disk = (ROOT / "bench" / "gmin-panel" / "lj-data" / f"{name}.data").read_text()
        if not disk.endswith("\n"):
            disk += "\n"
        assert text == disk + "RANSEED 2000\n"
        seen[name] = text
    assert len(seen) == 8
    assert len(set(seen.values())) == len(seen)
    for name, text in seen.items():
        n = panel.spec(name)[2]
        own = radius_text(n)
        for other, other_text in seen.items():
            if other == name:
                continue
            foreign = radius_text(panel.spec(other)[2])
            assert foreign != own
            assert f"RADIUS {foreign}" not in text
            assert other_text != text
