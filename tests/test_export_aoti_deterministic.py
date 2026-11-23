# MIT License
# Copyright 2023--present rgpot developers
"""An exported package gives one output for one input at any export thread count."""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("ase")

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import export_uma_aoti as exp  # noqa: E402


class EdgeScatter(torch.nn.Module):
    """Per-edge messages summed onto their centre atoms, as UMA does."""

    def __init__(self, width: int):
        super().__init__()
        self.lin = torch.nn.Linear(3, width)

    def forward(self, pos, edge_index):
        vec = pos[edge_index[0]] - pos[edge_index[1]]
        msg = torch.tanh(self.lin(vec))
        out = torch.zeros(pos.shape[0], msg.shape[1], dtype=pos.dtype)
        return out.index_add(0, edge_index[1], msg).sum(1)


def _sources(pkg: Path) -> str:
    with zipfile.ZipFile(pkg) as z:
        return "".join(
            z.read(n).decode() for n in z.namelist() if n.endswith(".cpp")
        )


def test_multithreaded_export_has_no_atomic_scatter(tmp_path, monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path / "inductor"))
    threads = torch.get_num_threads()
    torch.manual_seed(0)
    n, width = 15, 1152
    pos = torch.randn(n, 3)
    src, dst = zip(*[(a, b) for b in range(n) for a in range(n) if a != b])
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    model = EdgeScatter(width).eval()
    with torch.no_grad():
        exported = torch.export.export(model, (pos, edge_index))
        pkg = exp.aoti_package(exported, tmp_path / "scatter.pt2")
    try:
        assert "atomic_add" not in _sources(pkg)
        loaded = exp.load_aoti(pkg)
        with torch.inference_mode():
            first = loaded(pos, edge_index)
            for _ in range(20):
                assert torch.equal(loaded(pos, edge_index), first)
    finally:
        torch.set_num_threads(threads)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("warn_only", [False, True])
@pytest.mark.parametrize("fails", [False, True])
def test_export_restores_deterministic_policy(tmp_path, monkeypatch, enabled, warn_only, fails):
    original = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    observed = []

    def compile_package(*args, **kwargs):
        observed.append((
            torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled(),
        ))
        if fails:
            raise RuntimeError("compile failed")

    monkeypatch.setattr(exp, "_first_import", lambda *args: compile_package)
    monkeypatch.setattr(exp, "_honor_omp_threads", lambda: None)
    monkeypatch.setattr(exp, "canonicalize_strided_slices", lambda value: value)
    monkeypatch.setattr(exp, "clear_pt2_execstack", lambda path: None)
    try:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
        if fails:
            with pytest.raises(RuntimeError, match="compile failed"):
                exp.aoti_package(object(), tmp_path / "policy.pt2")
        else:
            exp.aoti_package(object(), tmp_path / "policy.pt2")
        assert observed == [(True, False)]
        assert torch.are_deterministic_algorithms_enabled() == enabled
        assert torch.is_deterministic_algorithms_warn_only_enabled() == warn_only
    finally:
        torch.use_deterministic_algorithms(original[0], warn_only=original[1])
