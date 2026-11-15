"""Metadata the UMA AOTI exporters embed for rgpot's UmaPot contract."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import numpy as np

torch = pytest.importorskip("torch")
ase = pytest.importorskip("ase")

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import export_uma_aoti as exp  # noqa: E402


def _hcn():
    return exp.HCN.copy()


def test_counts_of_is_exact_and_sorted():
    atoms = ase.Atoms("C2H4", positions=[[0, 0, i] for i in range(6)])
    assert exp.counts_of(atoms) == {1: 4, 6: 2}
    assert list(exp.counts_of(atoms)) == [1, 6]
    assert exp.counts_of(_hcn()) == {1: 1, 6: 1, 7: 1}


def test_runtime_metadata_carries_the_contract():
    atoms = _hcn()
    n = len(atoms)
    example = [torch.zeros(n, 3), torch.zeros(n, dtype=torch.long)]
    meta = exp.runtime_metadata(
        Path("hcn.pt2"),
        10.0,
        300,
        example,
        torch.float32,
        task_name="omol",
        charge=-1,
        spin=2,
        z_set=exp.z_set_of(atoms),
        counts=exp.counts_of(atoms),
        label="hcn",
        model="uma-s-1p1",
    )
    assert meta["charge"] == -1
    assert meta["spin"] == 2
    assert meta["task_name"] == "omol"
    assert meta["z_set"] == [1, 6, 7]
    assert meta["natoms"] == 3
    # Embedded as a string (aot_inductor.metadata is str -> str); the
    # C++ side reads it back as {Z: n} pairs.
    assert json.loads(meta["counts"]) == {"1": 1, "6": 1, "7": 1}
    assert meta["model"] == "uma-s-1p1"
    assert meta["pos_dtype"] == "float32"
    assert meta["model_dtype"] == "float32"
    assert meta["torch_version"] == torch.__version__
    assert isinstance(meta["fairchem_version"], str)
    assert meta["fairchem_version"]


def test_float64_metadata_names_both_dtypes():
    atoms = _hcn()
    n = len(atoms)
    example = [torch.zeros(n, 3), torch.zeros(n, dtype=torch.long)]
    meta = exp.runtime_metadata(
        Path("hcn.pt2"),
        10.0,
        300,
        example,
        torch.float64,
        model_dtype=torch.float64,
        task_name="omol",
        z_set=exp.z_set_of(atoms),
        counts=exp.counts_of(atoms),
        label="hcn",
        model="uma-s-1p1",
    )
    assert meta["pos_dtype"] == "float64"
    assert meta["model_dtype"] == "float64"


def test_floating_dtype_rejects_a_mixed_parameter_and_buffer():
    class Mixed(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(2, 2)
            self.register_buffer("wide", torch.zeros(2, dtype=torch.float64))

        def forward(self, values):
            return self.linear(values)

    with pytest.raises(RuntimeError, match="floating dtype"):
        exp.floating_dtype(Mixed())


def test_apply_export_dtype_uses_double_and_float():
    module = torch.nn.Linear(3, 4)
    exp.apply_export_dtype(module, "float64")
    assert next(module.parameters()).dtype == torch.float64
    exp.apply_export_dtype(module, "float32")
    assert next(module.parameters()).dtype == torch.float32


def test_export_gate_compares_each_dtype_to_its_own_reference():
    """A 1.2e-4 eV/A force gap fails the 1e-4 gate across dtypes.

    The same eager forces pass when the reference is the float64 one.
    That is the HCN case: float64 versus a float32 ASE result sits just
    outside 1e-4, and versus a float64 ASE result it does not.
    """
    float32_forces = np.zeros((2, 3))
    float64_forces = np.array([[1.2e-4, 0.0, 0.0], [0.0, 0.0, 0.0]])
    eager = float64_forces.copy()
    assert not exp.compare("cross", 0.0, eager, 0.0, float32_forces)
    energy, forces = exp.reference_for_dtype(
        "float64",
        (0.0, float32_forces),
        (-1.5, float64_forces),
    )
    assert energy == -1.5
    assert exp.compare("float64", -1.5, eager, energy, forces)
    energy32, forces32 = exp.reference_for_dtype(
        "float32",
        (0.25, float32_forces),
        (-1.5, float64_forces),
    )
    assert energy32 == 0.25
    assert exp.compare("float32", 0.25, float32_forces, energy32, forces32)


def test_dynamic_package_omits_natoms_when_counts_are_absent():
    example = [torch.zeros(4, 3), torch.zeros(4, dtype=torch.long)]
    meta = exp.runtime_metadata(
        Path("dyn.pt2"),
        6.0,
        300,
        example,
        torch.float32,
        model_dtype=torch.float32,
        z_set=[6],
        counts=None,
        label="dyn",
        model="tiny",
    )
    assert "natoms" not in meta
    assert "counts" not in meta
    assert meta["z_set"] == [6]


def test_dynamic_bounds_default_to_the_neighbour_cap():
    assert exp.dynamic_bounds(1024, 0, 300) == (2, 1024, 1, 307200)
    assert exp.dynamic_bounds(64, 5000, 300) == (2, 64, 1, 5000)
    with pytest.raises(SystemExit):
        exp.dynamic_bounds(1, 0, 300)


def test_trace_safe_refuses_sizes_that_merge_with_static_axes():
    assert not exp.trace_safe(3, 6)  # HCN: 3 is the coordinate axis
    assert not exp.trace_safe(2, 2)
    assert not exp.trace_safe(8, 8)
    assert not exp.trace_safe(9, 3)
    assert exp.trace_safe(8, 56)
    assert exp.trace_safe(9, 72)


class _NeighbourSum(torch.nn.Module):
    def forward(self, pos, atomic_numbers, cell, pbc, edge_index,
                cell_offsets, charge, spin, batch, natoms):
        shift = cell_offsets @ cell.squeeze(0)
        vec = pos.index_select(0, edge_index[0]) - pos.index_select(
            0, edge_index[1]) + shift
        energy = vec.norm(dim=-1).sum() + atomic_numbers.sum() * 0
        return energy, torch.zeros_like(pos)


def _toy_example(n, e):
    pos = torch.arange(3 * n, dtype=torch.float64).reshape(n, 3)
    edge_index = torch.stack(
        [torch.arange(e) % n, (torch.arange(e) + 1) % n]
    ).long()
    return (
        pos,
        torch.full((n,), 6, dtype=torch.long),
        torch.eye(3, dtype=torch.float64).reshape(1, 3, 3) * 25,
        torch.ones((1, 3), dtype=torch.bool),
        edge_index,
        torch.zeros((e, 3), dtype=torch.float64),
        torch.zeros(1, dtype=torch.long),
        torch.ones(1, dtype=torch.long),
        torch.zeros(n, dtype=torch.long),
        torch.tensor([n]),
    )


def test_export_dynamic_records_the_ranges_it_traced():
    module = _NeighbourSum()
    program, record = exp.export_dynamic(
        module, _toy_example(9, 72), (2, 64, 1, 4096)
    )
    assert record["natoms_min"] == 2
    assert record["natoms_max"] == 64
    assert record["nedges_min"] == 1
    assert record["nedges_max"] == 4096
    assert record["pos_shape"] == "natoms,3"
    assert record["edge_index_shape"] == "2,nedges"
    for n, e in ((3, 6), (12, 7), (40, 900)):
        example = _toy_example(n, e)
        got, _ = program.module()(*example)
        want, _ = module(*example)
        torch.testing.assert_close(got, want, rtol=0, atol=0)


def test_baker_dedup_keeps_compositions_that_share_an_element_set(
    tmp_path, monkeypatch
):
    import export_baker_uma_aoti as baker

    structures = {
        "01_c2h2": "C2H2",
        "02_c2h4": "C2H4",
        "03_h2cc": "H2C2",
        "04_ch3o": "CH3O",
        "05_ch3o_closed": "CH3O",
    }
    for label in structures:
        (tmp_path / label).mkdir()
        (tmp_path / label / "reactant.con").write_text("")

    def fake_load(path):
        formula = structures[Path(path).parent.name]
        n = len(ase.Atoms(formula))
        return ase.Atoms(formula, positions=[[0, 0, i] for i in range(n)])

    monkeypatch.setattr(baker, "load_atoms", fake_load)
    jobs = baker.collect_jobs(tmp_path)
    labels = [j["label"] for j in jobs]
    # C2H2 and C2H4 share z_set [1, 6] and are different packages; H2C2
    # is C2H2 again. CH3O doublet and singlet differ in spin.
    assert labels == ["01_c2h2", "02_c2h4", "04_ch3o", "05_ch3o_closed"]
    assert jobs[0]["counts"] == {1: 2, 6: 2}
    assert jobs[1]["counts"] == {1: 4, 6: 2}
    assert jobs[1]["z_set"] == [1, 6]


class _StridedWrite(torch.nn.Module):
    def __init__(self, size, stride, offset):
        super().__init__()
        self.size = size
        self.stride = stride
        self.offset = offset

    def forward(self, base, values):
        return torch.as_strided_scatter(
            base, values, self.size, self.stride, self.offset
        )


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("start", [0, 1])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_strided_slices_match_independent_tensor_assignment(axis, start, dtype):
    base = torch.arange(120, dtype=dtype).reshape(4, 5, 6)
    shape = list(base.shape)
    shape[axis] = 2
    values = -torch.arange(int(np.prod(shape)), dtype=dtype).reshape(shape) - 1
    module = _StridedWrite(shape, base.stride(), start * base.stride()[axis])
    program = torch.export.export(module, (base, values), strict=False)
    canonical = exp.canonicalize_strided_slices(program)
    expected = base.clone()
    selected = [slice(None)] * 3
    selected[axis] = slice(start, start + 2)
    expected[tuple(selected)] = values
    torch.testing.assert_close(
        canonical.module()(base, values), expected, rtol=0, atol=0
    )
    assert any(node.target == torch.ops.aten.slice_scatter.default
               for node in canonical.graph_module.graph.nodes)


@pytest.mark.parametrize("case", ["noncontiguous", "multiple_axes", "unaligned"])
def test_general_strided_writes_retain_their_exact_address_map(case):
    base = torch.arange(120, dtype=torch.float64).reshape(4, 5, 6)
    offset = 0
    if case == "noncontiguous":
        base = base.transpose(0, 1)
        shape = (2, 4, 6)
    elif case == "multiple_axes":
        shape = (2, 2, 6)
    else:
        shape = (2, 5, 6)
        offset = 1
    values = -torch.arange(int(np.prod(shape)), dtype=base.dtype).reshape(shape) - 1
    module = _StridedWrite(shape, base.stride(), offset)
    program = torch.export.export(module, (base, values), strict=False)
    canonical = exp.canonicalize_strided_slices(program)
    # Enumerate storage addresses independently of the tensor scatter operators.
    expected_storage = np.arange(120, dtype=np.float64)
    for index in np.ndindex(shape):
        address = offset + sum(i * stride for i, stride in
                               zip(index, base.stride()))
        expected_storage[address] = float(values[index])
    expected = torch.empty_like(base)
    for index in np.ndindex(tuple(base.shape)):
        address = sum(i * stride for i, stride in zip(index, base.stride()))
        expected[index] = expected_storage[address]
    torch.testing.assert_close(
        canonical.module()(base, values), expected, rtol=0, atol=0
    )
    assert any(node.target == torch.ops.aten.as_strided_scatter.default
               for node in canonical.graph_module.graph.nodes)


class _ReverseViewGradient(torch.nn.Module):
    def forward(self, gradient):
        source = torch.ops.aten.slice_backward.default(
            gradient, [12, 9, 128], 1, 0, 1, 1
        )
        empty = torch.ops.aten.new_empty_strided.default(
            source, [12, 9, 128], [1152, 128, 1]
        )
        buffer = torch.ops.aten.copy_.default(empty, source)
        view = torch.ops.aten.as_strided.default(
            buffer, [12, 1, 128], [1152, 128, 1], 0
        )
        snapshot = view.clone(memory_format=torch.contiguous_format)
        view.copy_(torch.zeros_like(snapshot))
        restored = torch.ops.aten.slice_backward.default(
            snapshot, [12, 9, 128], 1, 0, 1, 1
        )
        return gradient.sum(), buffer + restored


def test_aoti_package_preserves_reverse_view_gradient(tmp_path):
    gradient = torch.arange(12 * 128, dtype=torch.float32).reshape(12, 1, 128) / 128
    expected = torch.zeros((12, 9, 128), dtype=gradient.dtype)
    expected[:, :1, :] = gradient
    program = torch.export.export(_ReverseViewGradient(), (gradient,), strict=False)
    package = exp.aoti_package(program, tmp_path / "reverse-view.pt2")
    energy, restored = exp.run_aoti(exp.load_aoti(package), (gradient,))
    torch.testing.assert_close(energy, gradient.sum(), rtol=0, atol=0)
    torch.testing.assert_close(restored, expected, rtol=0, atol=0)
