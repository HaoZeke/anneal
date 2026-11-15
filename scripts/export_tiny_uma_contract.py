#!/usr/bin/env python3
# Copyright (c) rgpot developers
"""AOTI packages with UmaPot's input contract and no UMA weights.

The graphs are small on purpose. ``precision`` is the sum of coordinates,
so a 1e-7 Angstrom shift changes the energy by 1e-7 when the inputs stay
float64. ``cost`` is a pair of linear layers, used to time float32 against
float64. ``batch`` is the same sum, traced for two systems.

``--meta both`` writes pos_dtype and model_dtype. ``model`` writes only
model_dtype, ``pos`` only pos_dtype, ``none`` writes neither.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import torch
import torch.nn as nn

from export_uma_aoti import (  # noqa: E402
    aoti_package,
    apply_export_dtype,
    runtime_metadata,
)


class SumEnergy(nn.Module):
    def forward(
        self,
        pos,
        atomic_numbers,
        cell,
        pbc,
        edge_index,
        cell_offsets,
        charge,
        spin,
        batch,
        natoms,
    ):
        return pos.reshape(-1).sum(), -torch.ones_like(pos)


class SumEnergyBatch(nn.Module):
    def forward(
        self,
        pos,
        atomic_numbers,
        cell,
        pbc,
        edge_index,
        cell_offsets,
        charge,
        spin,
        batch,
        natoms,
        nedges,
    ):
        energy = torch.zeros(natoms.shape[0], dtype=pos.dtype, device=pos.device)
        energy = energy.index_add(0, batch.long(), pos.sum(-1))
        return energy, -torch.ones_like(pos)


class GatherEnergy(nn.Module):
    """Energy is the sum of (pos[neighbor] - pos[center]) over edges.

    The value changes when the neighbour list changes, so a static graph
    and a dynamic graph disagree on a stretched geometry.
    """

    def forward(
        self,
        pos,
        atomic_numbers,
        cell,
        pbc,
        edge_index,
        cell_offsets,
        charge,
        spin,
        batch,
        natoms,
    ):
        # Neighbour positions only. A complete graph of n atoms then
        # has energy (n - 1) times the sum of coordinates, so dropping
        # edges changes the energy. The reversed difference cancels.
        src = pos.index_select(0, edge_index[0])
        energy = src.sum()
        forces = torch.zeros_like(pos)
        # The host force is minus the derivative of the energy.
        forces.index_add_(0, edge_index[0], -torch.ones_like(src))
        return energy, forces


class CostEnergy(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 64)

    def forward(
        self,
        pos,
        atomic_numbers,
        cell,
        pbc,
        edge_index,
        cell_offsets,
        charge,
        spin,
        batch,
        natoms,
    ):
        hidden = torch.relu(self.fc2(torch.relu(self.fc1(pos))))
        return hidden.sum(), -torch.ones_like(pos)


def _line_geometry(dtype):
    """Three atoms on a line. Every coordinate is exact in float32."""
    pos = torch.tensor(
        [
            [12.5, 12.5, 12.5],
            [12.5, 12.5, 13.5],
            [12.5, 12.5, 11.5],
        ],
        dtype=dtype,
    )
    numbers = torch.tensor([6, 7, 1], dtype=torch.long)
    return pos, numbers


def _single_example(dtype, nedges: int):
    pos, numbers = _line_geometry(dtype)
    n = pos.shape[0]
    cell = torch.eye(3, dtype=dtype).reshape(1, 3, 3) * 25
    pbc = torch.tensor([[True, True, True]])
    # Directed pairs, self excluded. UmaPot's vesin list for this geometry
    # at a 6 Å cutoff has this length (no periodic image is inside 6 Å).
    neighbor = [1, 2, 0, 2, 0, 1]
    center = [0, 0, 1, 1, 2, 2]
    edge_index = torch.tensor([neighbor[:nedges], center[:nedges]], dtype=torch.long)
    offsets = torch.zeros((nedges, 3), dtype=dtype)
    charge = torch.tensor([0], dtype=torch.long)
    spin = torch.tensor([1], dtype=torch.long)
    batch = torch.zeros(n, dtype=torch.long)
    natoms = torch.tensor([n], dtype=torch.long)
    return (
        pos,
        numbers,
        cell,
        pbc,
        edge_index,
        offsets,
        charge,
        spin,
        batch,
        natoms,
    )


def _batch_example(dtype, nedges_each: int):
    one = _single_example(dtype, nedges_each)
    pos, numbers, cell, pbc, edge_index, offsets, charge, spin, batch, natoms = one
    n = pos.shape[0]
    pos = torch.cat((pos, pos), dim=0)
    numbers = torch.cat((numbers, numbers), dim=0)
    cell = torch.cat((cell, cell), dim=0)
    pbc = torch.cat((pbc, pbc), dim=0)
    second = edge_index + n
    edge_index = torch.cat((edge_index, second), dim=1)
    offsets = torch.cat((offsets, offsets), dim=0)
    charge = torch.cat((charge, charge), dim=0)
    spin = torch.cat((spin, spin), dim=0)
    batch = torch.cat((batch, torch.ones(n, dtype=torch.long)), dim=0)
    natoms = torch.cat((natoms, natoms), dim=0)
    nedges = torch.tensor([nedges_each, nedges_each], dtype=torch.long)
    return (
        pos,
        numbers,
        cell,
        pbc,
        edge_index,
        offsets,
        charge,
        spin,
        batch,
        natoms,
        nedges,
    )


def _cost_example(dtype):
    n, nedges = 4, 8
    pos = torch.zeros((n, 3), dtype=dtype)
    pos[:, 0] = torch.arange(n, dtype=dtype) * 2 + 15
    pos[:, 1] = 15
    pos[:, 2] = 15
    numbers = torch.full((n,), 6, dtype=torch.long)
    cell = torch.eye(3, dtype=dtype).reshape(1, 3, 3) * 50
    pbc = torch.tensor([[True, True, True]])
    edge_index = torch.zeros((2, nedges), dtype=torch.long)
    offsets = torch.zeros((nedges, 3), dtype=dtype)
    charge = torch.tensor([0], dtype=torch.long)
    spin = torch.tensor([1], dtype=torch.long)
    batch = torch.zeros(n, dtype=torch.long)
    natoms = torch.tensor([n], dtype=torch.long)
    return (
        pos,
        numbers,
        cell,
        pbc,
        edge_index,
        offsets,
        charge,
        spin,
        batch,
        natoms,
    )


def _apply_meta(meta: dict, mode: str) -> dict:
    if mode == "both":
        return meta
    if mode == "model":
        meta.pop("pos_dtype", None)
    elif mode == "pos":
        meta.pop("model_dtype", None)
    elif mode == "none":
        meta.pop("pos_dtype", None)
        meta.pop("model_dtype", None)
    elif mode == "mismatch":
        # Float32 positions recorded against a float64 graph. UmaPot
        # must refuse this instead of running it.
        meta["pos_dtype"] = "float32"
        meta["model_dtype"] = "float64"
    else:
        raise SystemExit(
            "--meta must be both, model, pos, none or mismatch, not " + mode
        )
    return meta


def _enforce_metadata(path: Path, meta: dict) -> None:
    """Rewrite dtype keys after packaging.

    The packager fills a missing pos_dtype with float32. The fixture
    needs the keys it asked for, including a package that records only
    model_dtype.
    """
    import json
    import zipfile
    from io import BytesIO

    blob = path.read_bytes()
    source = zipfile.ZipFile(BytesIO(blob))
    out = BytesIO()
    with zipfile.ZipFile(out, "w") as dest:
        for info in source.infolist():
            data = source.read(info.filename)
            if info.filename.endswith(".json"):
                # A zero-length member is not valid JSON and the metadata
                # parse throws. An empty object is the same "no keys"
                # contract.
                if not data.strip():
                    data = b"{}"
                try:
                    payload = json.loads(data)
                except json.JSONDecodeError:
                    payload = None
                if isinstance(payload, dict) and (
                    "pos_dtype" in payload or "model_dtype" in payload
                ):
                    payload.pop("pos_dtype", None)
                    payload.pop("model_dtype", None)
                    if "pos_dtype" in meta:
                        payload["pos_dtype"] = meta["pos_dtype"]
                    if "model_dtype" in meta:
                        payload["model_dtype"] = meta["model_dtype"]
                    data = json.dumps(payload).encode()
            dest.writestr(info, data)
    path.write_bytes(out.getvalue())


def export_precision(path: Path, dtype_name: str, meta_mode: str) -> None:
    dtype = torch.float64 if dtype_name == "float64" else torch.float32
    example = _single_example(dtype, 6)
    module = apply_export_dtype(SumEnergy().eval(), dtype_name)
    exported = torch.export.export(module, example, strict=False)
    meta = runtime_metadata(
        path,
        6.0,
        300,
        example,
        dtype,
        model_dtype=dtype,
        task_name="omol",
        charge=0,
        spin=1,
        z_set=[1, 6, 7],
        counts={1: 1, 6: 1, 7: 1},
        label="line",
        model="tiny-sum",
    )
    meta["nedges_min"] = 6
    meta["nedges_max"] = 6
    meta["export_path"] = "tiny-sum"
    wanted = _apply_meta(meta, meta_mode)
    aoti_package(exported, path, metadata=wanted)
    _enforce_metadata(path, wanted)


def _drop_metadata_keys(path: Path, keys: tuple[str, ...]) -> None:
    import json
    import zipfile
    from io import BytesIO

    blob = path.read_bytes()
    source = zipfile.ZipFile(BytesIO(blob))
    out = BytesIO()
    with zipfile.ZipFile(out, "w") as dest:
        for info in source.infolist():
            data = source.read(info.filename)
            if info.filename.endswith(".json"):
                try:
                    payload = json.loads(data)
                except json.JSONDecodeError:
                    payload = None
                if isinstance(payload, dict):
                    for key in keys:
                        payload.pop(key, None)
                    data = json.dumps(payload).encode()
            dest.writestr(info, data)
    path.write_bytes(out.getvalue())


def export_packed(path: Path, dtype_name: str) -> None:
    """Dynamic disjoint graph: one, two or three systems, no padding.

    ``batch_packed`` tells UmaPot to send the real system count. The
    traced example is two copies of the three-atom line.
    """
    dtype = torch.float64 if dtype_name == "float64" else torch.float32
    example = _batch_example(dtype, 6)
    module = apply_export_dtype(SumEnergyBatch().eval(), dtype_name)
    nsys = torch.export.Dim("nsys", min=1, max=4)
    atoms = torch.export.Dim("atoms", min=3, max=12)
    edges = torch.export.Dim("edges", min=1, max=48)
    exported = torch.export.export(
        module,
        example,
        dynamic_shapes={
            "pos": {0: atoms},
            "atomic_numbers": {0: atoms},
            "cell": {0: nsys},
            "pbc": {0: nsys},
            "edge_index": {1: edges},
            "cell_offsets": {0: edges},
            "charge": {0: nsys},
            "spin": {0: nsys},
            "batch": {0: atoms},
            "natoms": {0: nsys},
            "nedges": {0: nsys},
        },
        strict=False,
    )
    meta = runtime_metadata(
        path,
        6.0,
        300,
        example,
        dtype,
        model_dtype=dtype,
        task_name="omol",
        charge=0,
        spin=1,
        z_set=[1, 6, 7],
        counts={1: 1, 6: 1, 7: 1},
        label="line",
        model="tiny-sum-packed",
        batch_max=4,
        batch_packed=1,
    )
    meta["nedges_min"] = 1
    meta["nedges_max"] = 48
    meta["edge_index_shape"] = "2,nedges"
    meta["export_path"] = "packed-dynamic"
    meta["pos_shape"] = "3,3"
    aoti_package(exported, path, metadata=meta)


def export_batch(path: Path, dtype_name: str, static: bool = False) -> None:
    dtype = torch.float64 if dtype_name == "float64" else torch.float32
    example = _batch_example(dtype, 6)
    module = apply_export_dtype(SumEnergyBatch().eval(), dtype_name)
    nedges_dim = torch.export.Dim("nedges", min=1, max=100000)
    exported = torch.export.export(
        module,
        example,
        dynamic_shapes={
            "pos": None,
            "atomic_numbers": None,
            "cell": None,
            "pbc": None,
            "edge_index": {1: nedges_dim},
            "cell_offsets": {0: nedges_dim},
            "charge": None,
            "spin": None,
            "batch": None,
            "natoms": None,
            "nedges": None,
        },
        strict=False,
    )
    meta = runtime_metadata(
        path,
        6.0,
        300,
        example,
        dtype,
        model_dtype=dtype,
        task_name="omol",
        charge=0,
        spin=1,
        z_set=[1, 6, 7],
        counts={1: 1, 6: 1, 7: 1},
        label="line",
        model="tiny-sum-batch",
        batch_max=2,
    )
    if static:
        exported = torch.export.export(module, example, strict=False)
        meta["nedges_min"] = int(example[4].shape[1])
        meta["nedges_max"] = int(example[4].shape[1])
        meta["edge_index_shape"] = f"2,{int(example[4].shape[1])}"
        meta["export_path"] = "tiny-sum-batch-static"
    else:
        meta["nedges_min"] = 1
        meta["nedges_max"] = 100000
        meta["export_path"] = "tiny-sum-batch"
    aoti_package(exported, path, metadata=meta)


def export_guard(path: Path, dtype_name: str, mode: str) -> None:
    """Packages used to time the edge guard. The kernel is a sum.

    ``single`` and ``batch`` carry a molecular box and a wide edge
    range so several atom counts stay inside the traced window while
    the runtime still requires the complete pair list. ``periodic``
    has no box. The real UMA exporter does not emit the wide range.
    """
    dtype = torch.float64 if dtype_name == "float64" else torch.float32
    molecular = mode != "periodic"
    batch = mode == "batch"
    example = _batch_example(dtype, 6) if batch else _cluster_example(dtype, 4, 12)
    module = apply_export_dtype(
        (SumEnergyBatch() if batch else SumEnergy()).eval(), dtype_name
    )
    natoms_dim = torch.export.Dim("natoms", min=1, max=8192)
    nedges_dim = torch.export.Dim("nedges", min=1, max=40_000_000)
    shapes = {
        "pos": {0: natoms_dim},
        "atomic_numbers": {0: natoms_dim},
        "cell": None,
        "pbc": None,
        "edge_index": {1: nedges_dim},
        "cell_offsets": {0: nedges_dim},
        "charge": None,
        "spin": None,
        "batch": {0: natoms_dim},
        "natoms": None,
    }
    if batch:
        shapes["nedges"] = None
        shapes["cell"] = {0: torch.export.Dim("nsys", min=1, max=8)}
        shapes["pbc"] = {0: torch.export.Dim("nsys2", min=1, max=8)}
        shapes["charge"] = {0: torch.export.Dim("nsys3", min=1, max=8)}
        shapes["spin"] = {0: torch.export.Dim("nsys4", min=1, max=8)}
    exported = torch.export.export(module, example, dynamic_shapes=shapes, strict=False)
    meta = runtime_metadata(
        path,
        80.0 if molecular else 6.0,
        300,
        example,
        dtype,
        model_dtype=dtype,
        task_name="omol",
        charge=0,
        spin=1,
        z_set=None,
        counts=None,
        label="guard",
        model="tiny-guard-" + mode,
        molecular_box=200.0 if molecular else 0.0,
        batch_max=2 if batch else 0,
    )
    meta["max_neighbors"] = 8192 if molecular else 300
    meta["nedges_min"] = 1
    meta["nedges_max"] = 40_000_000
    meta["natoms_min"] = 1
    meta["natoms_max"] = 8192
    meta["edge_index_shape"] = "2,nedges"
    meta["pos_shape"] = "natoms,3"
    meta["molecular"] = molecular
    meta["export_path"] = "tiny-guard-" + mode
    aoti_package(exported, path, metadata=meta)


def export_batch_molecular(path: Path, dtype_name: str) -> None:
    """Two-system sum with a molecular box and a wide edge range.

    The range is what a package looked like before export paths pinned
    the traced count. The runtime must still refuse a system that is
    not the complete pair list.
    """
    dtype = torch.float64 if dtype_name == "float64" else torch.float32
    example = _batch_example(dtype, 6)
    module = apply_export_dtype(SumEnergyBatch().eval(), dtype_name)
    nedges_dim = torch.export.Dim("nedges", min=1, max=100000)
    exported = torch.export.export(
        module,
        example,
        dynamic_shapes={
            "pos": None,
            "atomic_numbers": None,
            "cell": None,
            "pbc": None,
            "edge_index": {1: nedges_dim},
            "cell_offsets": {0: nedges_dim},
            "charge": None,
            "spin": None,
            "batch": None,
            "natoms": None,
            "nedges": None,
        },
        strict=False,
    )
    meta = runtime_metadata(
        path,
        6.0,
        300,
        example,
        dtype,
        model_dtype=dtype,
        task_name="omol",
        charge=0,
        spin=1,
        z_set=[1, 6, 7],
        counts={1: 1, 6: 1, 7: 1},
        label="line",
        model="tiny-sum-batch-molecular",
        molecular_box=80.0,
        batch_max=2,
    )
    meta["nedges_min"] = 1
    meta["nedges_max"] = 100000
    meta["edge_index_shape"] = "2,nedges"
    meta["molecular"] = True
    meta["export_path"] = "tiny-sum-batch-molecular"
    aoti_package(exported, path, metadata=meta)


def _cluster(dtype, n: int):
    pos = torch.zeros((n, 3), dtype=dtype)
    pos[:, 0] = 12.5
    pos[:, 1] = 12.5
    pos[:, 2] = 12.5 + torch.arange(n, dtype=dtype)
    numbers = torch.full((n,), 6, dtype=torch.long)
    return pos, numbers


def _cluster_example(dtype, n: int, nedges: int):
    pos, numbers = _cluster(dtype, n)
    cell = torch.eye(3, dtype=dtype).reshape(1, 3, 3) * 40
    pbc = torch.tensor([[True, True, True]])
    # A complete directed graph, self excluded, truncated to nedges.
    neighbor = []
    center = []
    for c in range(n):
        for nbr in range(n):
            if nbr == c:
                continue
            neighbor.append(nbr)
            center.append(c)
    neighbor = neighbor[:nedges]
    center = center[:nedges]
    edge_index = torch.tensor([neighbor, center], dtype=torch.long)
    offsets = torch.zeros((nedges, 3), dtype=dtype)
    charge = torch.tensor([0], dtype=torch.long)
    spin = torch.tensor([1], dtype=torch.long)
    batch = torch.zeros(n, dtype=torch.long)
    natoms = torch.tensor([n], dtype=torch.long)
    return (
        pos,
        numbers,
        cell,
        pbc,
        edge_index,
        offsets,
        charge,
        spin,
        batch,
        natoms,
    )


def export_gather(path: Path, dtype_name: str, dynamic: bool) -> None:
    """Gather graph. ``dynamic`` traces a symbolic edge count and atom count.

    The static package is eight atoms and 56 edges, the fixed graph that
    a tetrazine export bakes in. The dynamic package accepts other sizes.
    """
    dtype = torch.float64 if dtype_name == "float64" else torch.float32
    if dynamic:
        example = _cluster_example(dtype, 4, 12)
    else:
        example = _cluster_example(dtype, 8, 56)
    module = apply_export_dtype(GatherEnergy().eval(), dtype_name)
    nedges_dim = torch.export.Dim("nedges", min=1, max=4096)
    natoms_dim = torch.export.Dim("natoms", min=1, max=64)
    shapes = None
    if dynamic:
        shapes = {
            "pos": {0: natoms_dim},
            "atomic_numbers": {0: natoms_dim},
            "cell": None,
            "pbc": None,
            "edge_index": {1: nedges_dim},
            "cell_offsets": {0: nedges_dim},
            "charge": None,
            "spin": None,
            "batch": {0: natoms_dim},
            "natoms": None,
        }
    exported = torch.export.export(
        module, example, dynamic_shapes=shapes, strict=False
    )
    meta = runtime_metadata(
        path,
        6.0,
        300,
        example,
        dtype,
        model_dtype=dtype,
        task_name="omol",
        charge=0,
        spin=1,
        z_set=None,
        counts=None,
        label="cluster",
        model="tiny-gather",
    )
    if dynamic:
        meta["nedges_min"] = 1
        meta["nedges_max"] = 4096
        meta["natoms_min"] = 1
        meta["natoms_max"] = 64
        meta["pos_shape"] = "natoms,3"
        meta["edge_index_shape"] = "2,nedges"
        meta["z_max"] = 99
        meta["export_path"] = "tiny-gather-dynamic"
    else:
        meta["natoms"] = 8
        meta["nedges_min"] = 56
        meta["nedges_max"] = 56
        meta["pos_shape"] = "8,3"
        meta["edge_index_shape"] = "2,56"
        meta["export_path"] = "tiny-gather-static"
    aoti_package(exported, path, metadata=meta)
    _enforce_metadata(path, meta)


def export_cost(path: Path, dtype_name: str) -> None:
    dtype = torch.float64 if dtype_name == "float64" else torch.float32
    example = _cost_example(dtype)
    module = apply_export_dtype(CostEnergy().eval(), dtype_name)
    natoms_dim = torch.export.Dim("natoms", min=1, max=4096)
    nedges_dim = torch.export.Dim("nedges", min=1, max=2_000_000)
    exported = torch.export.export(
        module,
        example,
        dynamic_shapes={
            "pos": {0: natoms_dim},
            "atomic_numbers": {0: natoms_dim},
            "cell": None,
            "pbc": None,
            "edge_index": {1: nedges_dim},
            "cell_offsets": {0: nedges_dim},
            "charge": None,
            "spin": None,
            "batch": {0: natoms_dim},
            "natoms": None,
        },
        strict=False,
    )
    meta = runtime_metadata(
        path,
        6.0,
        300,
        example,
        dtype,
        model_dtype=dtype,
        task_name="omol",
        charge=0,
        spin=1,
        z_set=[6],
        counts=None,
        label="lattice",
        model="tiny-cost",
    )
    meta["nedges_min"] = 1
    meta["nedges_max"] = 2_000_000
    meta["export_path"] = "tiny-cost"
    aoti_package(exported, path, metadata=meta)


def export_tetrazine_keyless(path: Path, dtype_name: str) -> None:
    """Eight-atom complete graph with the edge keys removed.

    Cutoff 10 angstrom and a 25 angstrom box match the control packages.
    The compiled graph is still 56 edges. A package from before the edge
    keys were written looks like this.
    """
    dtype = torch.float64 if dtype_name == "float64" else torch.float32
    example = _cluster_example(dtype, 8, 56)
    module = apply_export_dtype(GatherEnergy().eval(), dtype_name)
    exported = torch.export.export(module, example, strict=False)
    meta = runtime_metadata(
        path,
        10.0,
        300,
        example,
        dtype,
        model_dtype=dtype,
        task_name="omol",
        charge=0,
        spin=1,
        z_set=None,
        counts=None,
        label="tetrazine",
        model="tiny-gather-keyless",
        molecular_box=25.0,
    )
    meta["natoms"] = 8
    meta["export_path"] = "make_fx-nonstrict"
    # The control packages omit both keys. Drop them after packaging so
    # the fixture is keyless even if a later packager inserts defaults.
    meta.pop("nedges_min", None)
    meta.pop("nedges_max", None)
    aoti_package(exported, path, metadata=meta)
    _drop_metadata_keys(
        path,
        ("nedges_min", "nedges_max", "edge_index_shape", "pos_shape"),
    )


def export_no_edges(path: Path, dtype_name: str) -> None:
    export_precision(path, dtype_name, "both")
    # ``shapes`` still carries the example edge list. Drop that too, or
    # the package is a fixed-shape export rather than one with no range.
    _drop_metadata_keys(
        path,
        (
            "nedges_min",
            "nedges_max",
            "edge_index_shape",
            "pos_shape",
            "shapes",
        ),
    )


def export_box(path: Path, dtype_name: str) -> None:
    """Line package recorded with a 15 angstrom molecular box."""
    dtype = torch.float64 if dtype_name == "float64" else torch.float32
    example = _single_example(dtype, 6)
    module = apply_export_dtype(SumEnergy().eval(), dtype_name)
    exported = torch.export.export(module, example, strict=False)
    meta = runtime_metadata(
        path,
        6.0,
        300,
        example,
        dtype,
        model_dtype=dtype,
        task_name="omol",
        charge=0,
        spin=1,
        z_set=[1, 6, 7],
        counts={1: 1, 6: 1, 7: 1},
        label="line",
        model="tiny-sum-box",
        molecular_box=15.0,
    )
    meta["nedges_min"] = 6
    meta["nedges_max"] = 6
    meta["export_path"] = "tiny-sum-box"
    aoti_package(exported, path, metadata=meta)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kind",
        choices=(
            "precision",
            "batch",
            "packed",
            "cost",
            "gather",
            "no-edges",
            "box",
            "tetrazine-keyless",
            "batch-molecular",
            "guard-single",
            "guard-batch",
            "guard-periodic",
        ),
        required=True,
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="gather only: symbolic atom count and edge count",
    )
    parser.add_argument(
        "--static",
        action="store_true",
        help="batch only: freeze the traced edge count",
    )
    parser.add_argument("--dtype", choices=("float32", "float64"), required=True)
    parser.add_argument(
        "--meta",
        choices=("both", "model", "pos", "none", "mismatch"),
        default="both",
        help="Which dtype keys to embed. precision only.",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.kind == "precision":
        export_precision(args.out, args.dtype, args.meta)
    elif args.kind == "gather":
        export_gather(args.out, args.dtype, args.dynamic)
    elif args.kind == "packed":
        if args.meta != "both":
            raise SystemExit("packed packages always record both dtype keys")
        export_packed(args.out, args.dtype)
    elif args.kind == "batch":
        if args.meta != "both":
            raise SystemExit("batch packages always record both dtype keys")
        export_batch(args.out, args.dtype, static=args.static)
    elif args.kind == "no-edges":
        export_no_edges(args.out, args.dtype)
    elif args.kind == "box":
        export_box(args.out, args.dtype)
    elif args.kind == "tetrazine-keyless":
        export_tetrazine_keyless(args.out, args.dtype)
    elif args.kind == "batch-molecular":
        export_batch_molecular(args.out, args.dtype)
    elif args.kind == "guard-single":
        export_guard(args.out, args.dtype, "single")
    elif args.kind == "guard-batch":
        export_guard(args.out, args.dtype, "batch")
    elif args.kind == "guard-periodic":
        export_guard(args.out, args.dtype, "periodic")
    else:
        if args.meta != "both":
            raise SystemExit("cost packages always record both dtype keys")
        export_cost(args.out, args.dtype)
    print(f"WROTE {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
