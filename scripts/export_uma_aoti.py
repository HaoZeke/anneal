#!/usr/bin/env python3
"""Export UMA as a tensor-only module and AOTI-package it.

Neighbor lists stay outside the graph (vesin). The exported module consumes
precomputed edge_index / cell_offsets, matching eSCNMDBackbone with
otf_graph=False.

Stages:
  1. ASE FAIRChemCalculator reference on the chosen atoms
  2. Eager tensor wrapper + vesin vs that reference
  3. torch.export
  4. AOTInductor .pt2
  5. Load .pt2 in-process and compare again

Dynamic export (the default: no --molecular-box, no --batch-max):
  One graph with torch.export.Dim for the atom count and the edge count:
    natoms in [2, --max-atoms]           (default [2, 1024])
    nedges in [1, --max-edges]           (default [1, max_atoms * 300])
  Every input whose size follows them (pos, atomic_numbers, batch;
  edge_index, cell_offsets) shares the symbol. The bounds the program
  accepts are read back from it and embedded as natoms_min/natoms_max
  and nedges_min/nedges_max, with pos_shape natoms,3 and
  edge_index_shape 2,nedges. rgpot's UmaPot refuses anything outside.
  The MoLE experts stay in the graph and their mix is computed from the
  input atoms, charge and spin, so one package covers every composition
  of the --task head; it records no z_set, natoms or counts, and z_max
  bounds the atomic numbers. --merge-mole folds the mix of --atoms into
  the weights instead: a smaller package for that composition only.
  Before the package is kept it must match FAIRChemCalculator on --atoms,
  on --atoms with its last atom moved past the cutoff (fewer edges), on
  a second molecule (unless --merge-mole) and on each --check-atoms.
  Limits: one system per call; at least two atoms and one edge (a lone
  atom has no edge and fairchem answers it from isolated-atom
  references); charge, spin and task are recorded and checked as the
  exported values; the package is compiled for the CPU ISA of the
  export host. A float64 package keeps all experts, about 1.2 GB for
  uma-s-1p1 (half that in float32; about 55 MB with --merge-mole).

--molecular-box and --batch-max keep the earlier static or band graphs
with merge_mole on: one composition, n(n-1) edges per system.

``--dtype float64`` converts the loaded model with ``hydra.double()`` before
the trace and records ``pos_dtype`` and ``model_dtype`` as ``float64``.
``--dtype float32`` calls ``hydra.float()`` and records ``float32``.

The checkpoint is --checkpoint, else $RGPOT_UMA_CHECKPOINT, else --model
downloaded from Hugging Face. The path taken is embedded as export_path.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from aoti_execstack import clear_pt2_execstack  # noqa: E402
from aoti_graph import canonicalize_strided_slices  # noqa: E402

import numpy as np
import torch
import torch.nn as nn
from ase import Atoms


HCN = Atoms(
    numbers=[6, 7, 1],
    positions=[
        [12.49734736216627162, 12.49892801474515913, 12.54059929828148512],
        [12.50115413363106498, 12.50036504272228832, 11.38209979880783251],
        [12.50149850420264563, 12.50069809648255514, 13.61514544631068446],
    ],
    cell=[[25.0, 0.0, 0.0], [0.0, 25.0, 0.0], [0.0, 0.0, 25.0]],
    pbc=True,
)
HCN.info.update({"charge": 0, "spin": 1})


def z_set_of(atoms: Atoms) -> list[int]:
    return sorted({int(z) for z in atoms.get_atomic_numbers()})


def counts_of(atoms: Atoms) -> dict[int, int]:
    """Atom count per atomic number, sorted by atomic number."""
    counts: dict[int, int] = {}
    for z in atoms.get_atomic_numbers():
        counts[int(z)] = counts.get(int(z), 0) + 1
    return dict(sorted(counts.items()))


def _dist_version(name: str) -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def default_label(atoms_path: str | None) -> str:
    if not atoms_path:
        return "hcn"
    path = Path(atoms_path)
    if path.name.lower() in {
        "reactant.con",
        "product.con",
        "reactant.xyz",
        "product.xyz",
    }:
        return path.parent.name
    return path.stem


def _ensure_box(atoms: Atoms) -> Atoms:
    cell = atoms.get_cell()
    if cell is None or float(cell.volume) < 1e-8:
        atoms.set_cell([[25.0, 0.0, 0.0], [0.0, 25.0, 0.0], [0.0, 0.0, 25.0]])
        atoms.set_pbc(True)
    elif not bool(atoms.pbc.any()):
        atoms.set_pbc(True)
    return atoms


def load_atoms(path: str | os.PathLike | None) -> Atoms:
    """Load a structure. ``.con`` goes through ``atoms_from_con``. Default is Baker HCN."""
    if path is None:
        return HCN.copy()
    path = Path(path)
    if not path.is_file():
        raise SystemExit(f"--atoms not found: {path}")
    if path.suffix.lower() == ".con":
        rpc = Path(__file__).resolve().parents[1] / "CppCore" / "rgpot" / "rpc"
        if str(rpc) not in sys.path:
            sys.path.insert(0, str(rpc))
        from ase_helpers import atoms_from_con

        return _ensure_box(atoms_from_con(path))
    from ase.io import read

    return _ensure_box(read(str(path)))


def _vesin_edges(
    pos: np.ndarray,
    cell: np.ndarray,
    cutoff: float,
    pbc=(True, True, True),
    max_neighbors: int | None = None,
):
    """Fairchem convention: edge_index = [neighbor, center], drop self-loops."""
    try:
        from vesin import NeighborList
    except ImportError as exc:
        raise SystemExit("vesin is required for export_uma_aoti") from exc

    nl = NeighborList(cutoff=float(cutoff), full_list=True)
    i, j, S = nl.compute(points=pos, box=cell, periodic=pbc, quantities="ijS")
    # vesin (i, j) is a pair; treat i as center, j as neighbor (fairchem n,c)
    n_index = np.asarray(j, dtype=np.int64)
    c_index = np.asarray(i, dtype=np.int64)
    shifts = np.asarray(S, dtype=np.float64)
    dvec = pos[n_index] - pos[c_index] + shifts @ cell
    dist = np.linalg.norm(dvec, axis=1)
    keep = dist >= 1e-8
    n_index, c_index, shifts, dist = n_index[keep], c_index[keep], shifts[keep], dist[keep]
    if max_neighbors is not None and max_neighbors > 0:
        keep_idx = []
        for center in np.unique(c_index):
            idx = np.where(c_index == center)[0]
            order = np.argsort(dist[idx])[: int(max_neighbors)]
            keep_idx.append(idx[order])
        if keep_idx:
            sel = np.concatenate(keep_idx)
            n_index, c_index, shifts = n_index[sel], c_index[sel], shifts[sel]
    edge_index = np.stack([n_index, c_index], axis=0)
    return edge_index, shifts


def vesin_atomic_data(
    atoms: Atoms,
    cutoff: float,
    dtype,
    device,
    max_neighbors: int | None,
    *,
    task_name: str = "omol",
    sid: str = "hcn",
):
    """Official AtomicData carrying vesin edges (otf_graph=False path)."""
    from fairchem.core.datasets.atomic_data import AtomicData

    pos_np = np.asarray(atoms.get_positions(), dtype=np.float64)
    cell_np = np.asarray(atoms.get_cell(), dtype=np.float64)
    edge_index, shifts = _vesin_edges(pos_np, cell_np, cutoff, max_neighbors=max_neighbors)
    pos = torch.tensor(pos_np, dtype=dtype)
    data = AtomicData(
        pos=pos,
        atomic_numbers=torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long),
        cell=torch.tensor(cell_np, dtype=dtype).unsqueeze(0),
        pbc=torch.tensor([[True, True, True]]),
        natoms=torch.tensor([pos.shape[0]], dtype=torch.long),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        cell_offsets=torch.tensor(shifts, dtype=dtype),
        nedges=torch.tensor([edge_index.shape[1]], dtype=torch.long),
        charge=torch.tensor([int(atoms.info.get("charge", 0))], dtype=torch.long),
        spin=torch.tensor([int(atoms.info.get("spin", 1))], dtype=torch.long),
        fixed=torch.zeros(pos.shape[0], dtype=torch.long),
        tags=torch.zeros(pos.shape[0], dtype=torch.long),
        dataset=[task_name],
        sid=[sid],
    )
    return data.to(device)


class BakedDatasetEmbedding(nn.Module):
    """Single-system omol embedding. No Python string list."""

    def __init__(self, emb: torch.Tensor):
        super().__init__()
        self.register_buffer("emb", emb.detach().contiguous())

    def forward(self, dataset_list=None):
        return self.emb


class TensorData(dict):
    """AtomicData-shaped mapping. fairchem uses both data[k] and data.k."""

    def get(self, key, default=None, **kwargs):
        if "default" in kwargs:
            default = kwargs["default"]
        return super().get(key, default)

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            self[name] = value

    def __len__(self):
        natoms = self.get("natoms")
        if natoms is None:
            return super().__len__()
        return int(natoms.numel())


def _lookup_task(out, task_name: str, prop: str):
    """Hydra output is either {task: {prop: t}} or a flattened pass-through."""
    if task_name in out:
        payload = out[task_name]
        if isinstance(payload, dict) and prop in payload:
            return payload[prop]
        if torch.is_tensor(payload):
            return payload
    if isinstance(out, dict):
        for key, payload in out.items():
            if not isinstance(payload, dict):
                continue
            if key == task_name and prop in payload:
                return payload[prop]
            if prop in payload and (task_name in key or key == prop):
                return payload[prop]
            nested = _lookup_task(payload, task_name, prop)
            if nested is not None:
                return nested
    return None


class TensorUma(nn.Module):
    """Tensor-only UMA forward. Graph is an input, not computed here.

    Calls the merged EFS head directly (no MoE dataset routing) and then
    matches MLIPPredictUnit._process_outputs: denorm, undo elemental refs.
    """

    def __init__(self, hydra_model: nn.Module, pred_unit=None, task_name: str = "omol"):
        super().__init__()
        self.task_name = task_name
        self.energy_task = f"{task_name}_energy"
        self.force_task = f"{task_name}_forces"
        e_mean = torch.zeros((), dtype=torch.float64)
        e_rmsd = torch.ones((), dtype=torch.float64)
        f_mean = torch.zeros((), dtype=torch.float64)
        f_rmsd = torch.ones((), dtype=torch.float64)
        elem_refs = torch.zeros(128, dtype=torch.float64)
        if pred_unit is not None:
            tasks = pred_unit.dataset_to_tasks[task_name]
            for task in tasks:
                if task.property == "energy":
                    self.energy_task = task.name
                    e_mean = task.normalizer.mean.detach().reshape(()).to(torch.float64)
                    e_rmsd = task.normalizer.rmsd.detach().reshape(()).to(torch.float64)
                    if task.element_references is not None:
                        elem_refs = (
                            task.element_references.element_references.detach().to(
                                torch.float64
                            )
                        )
                elif task.property == "forces":
                    self.force_task = task.name
                    f_mean = task.normalizer.mean.detach().reshape(()).to(torch.float64)
                    f_rmsd = task.normalizer.rmsd.detach().reshape(()).to(torch.float64)
        self.register_buffer("e_mean", e_mean)
        self.register_buffer("e_rmsd", e_rmsd)
        self.register_buffer("f_mean", f_mean)
        self.register_buffer("f_rmsd", f_rmsd)
        self.register_buffer("elem_refs", elem_refs)
        hydra = hydra_model.module if hasattr(hydra_model, "module") else hydra_model
        backbone = hydra.backbone
        if hasattr(backbone, "dataset_embedding"):
            de = backbone.dataset_embedding
            idx = torch.tensor(0, dtype=torch.long)
            with torch.no_grad():
                baked = de.dataset_emb_dict[task_name](
                    idx.to(next(de.parameters()).device)
                )
            if baked.dim() == 1:
                baked = baked.unsqueeze(0)
            backbone.dataset_embedding = BakedDatasetEmbedding(baked)
            # The baked embedding is one row; the charge embedding
            # carries the (dynamic) system count, so the dataset row
            # expands to it inside the graph. Overriding csd_embedding
            # keeps fairchem untouched and the expansion symbolic.
            import types

            def _csd_batched(bself, charge, spin, dataset):
                chg_emb = bself.charge_embedding(charge)
                spin_emb = bself.spin_embedding(spin)
                d_emb = bself.dataset_embedding.emb.to(chg_emb.dtype).expand(
                    chg_emb.shape[0], -1
                )
                return torch.nn.SiLU()(
                    bself.mix_csd(
                        torch.cat((chg_emb, spin_emb, d_emb), dim=1)
                    )
                )

            backbone.csd_embedding = types.MethodType(_csd_batched, backbone)
        efs = None
        for _name, head in hydra.output_heads.items():
            inner = getattr(head, "head", head)
            if hasattr(inner, "energy_block"):
                efs = inner
                break
        if efs is None:
            raise RuntimeError(
                f"no EFS head in {list(hydra.output_heads.keys())}"
            )
        self.backbone = backbone
        self.efs = efs

    def forward(
        self,
        pos: torch.Tensor,
        atomic_numbers: torch.Tensor,
        cell: torch.Tensor,
        pbc: torch.Tensor,
        edge_index: torch.Tensor,
        cell_offsets: torch.Tensor,
        charge: torch.Tensor,
        spin: torch.Tensor,
        batch: torch.Tensor,
        natoms: torch.Tensor,
    ):
        # Gradients stay inside this function. The returned energy and
        # forces are detached so the exported graph does not hand a
        # requires-grad tensor back to the caller.
        pos = pos.clone().requires_grad_(True)
        cell = cell.clone().requires_grad_(True)
        # The patched graph always applies the batched cell shift, which
        # reads nedges. int(edge_index.shape[1]) would bake the traced
        # count into the program as a constant; a row count of ones
        # stays a function of the input. A zero placeholder makes
        # repeat_interleave allocate the wrong number of rows.
        nedges = torch.ones_like(edge_index[1]).sum().reshape(1)
        z = atomic_numbers.long()
        b = batch.long()
        data = TensorData(
            pos=pos,
            atomic_numbers=z,
            atomic_numbers_full=z,
            cell=cell,
            pbc=pbc,
            edge_index=edge_index.long(),
            cell_offsets=cell_offsets,
            nedges=nedges,
            charge=charge.long(),
            spin=spin.long(),
            batch=b,
            batch_full=b,
            natoms=natoms.long(),
            fixed=torch.zeros_like(z),
            tags=torch.zeros_like(z),
            dataset=[self.task_name],
            sid=[""],
        )
        emb = self.backbone(data)
        # Bypass MLP_EFS_Head: compute_energy is @torch.compiler.disable
        # (index_add + float64), which export lifts as a fake constant.
        # Single-system sum is the same reduction and is traceable.
        node_emb = emb["node_embedding"]
        scalar = node_emb.narrow(1, 0, 1).squeeze(1)
        node_e = self.efs.energy_block(scalar).reshape(-1)
        energy_raw = node_e.to(torch.float64).sum()
        energy = (
            energy_raw * self.e_rmsd
            + self.e_mean
            + self.elem_refs.to(device=z.device)[z].sum()
        )
        (g,) = torch.autograd.grad(energy_raw, pos, create_graph=False)
        forces = (-g).to(torch.float64) * self.f_rmsd + self.f_mean
        return energy.detach(), forces.detach()


class TensorUmaBatched(TensorUma):
    """Band-batched forward: per-system nedges input and segment
    reductions, so one call evaluates B same-composition systems."""

    def forward(  # type: ignore[override]
        self,
        pos: torch.Tensor,
        atomic_numbers: torch.Tensor,
        cell: torch.Tensor,
        pbc: torch.Tensor,
        edge_index: torch.Tensor,
        cell_offsets: torch.Tensor,
        charge: torch.Tensor,
        spin: torch.Tensor,
        batch: torch.Tensor,
        natoms: torch.Tensor,
        nedges: torch.Tensor,
    ):
        pos = pos.clone().requires_grad_(True)
        z = atomic_numbers.long()
        b = batch.long()
        data = TensorData(
            pos=pos,
            atomic_numbers=z,
            atomic_numbers_full=z,
            cell=cell,
            pbc=pbc,
            edge_index=edge_index.long(),
            cell_offsets=cell_offsets,
            nedges=nedges.long(),
            charge=charge.long(),
            spin=spin.long(),
            batch=b,
            batch_full=b,
            natoms=natoms.long(),
            fixed=torch.zeros_like(z),
            tags=torch.zeros_like(z),
            dataset=[self.task_name],
            sid=[""],
        )
        emb = self.backbone(data)
        node_emb = emb["node_embedding"]
        scalar = node_emb.narrow(1, 0, 1).squeeze(1)
        node_e = self.efs.energy_block(scalar).reshape(-1).to(torch.float64)
        B = natoms.shape[0]
        energy_raw = torch.zeros(
            B, dtype=torch.float64, device=node_e.device
        ).index_add(0, b, node_e)
        ref_sum = torch.zeros(
            B, dtype=torch.float64, device=node_e.device
        ).index_add(0, b, self.elem_refs.to(device=z.device)[z])
        energy = energy_raw * self.e_rmsd + self.e_mean + ref_sum
        (g,) = torch.autograd.grad(
            energy_raw.sum(), pos, create_graph=False
        )
        forces = (-g).to(torch.float64) * self.f_rmsd + self.f_mean
        return energy, forces


def _settings(merge_mole: bool = True, precision=None):
    from fairchem.core.units.mlip_unit.api.inference import InferenceSettings

    extra = {} if precision is None else {"base_precision_dtype": precision}
    return InferenceSettings(
        tf32=False,
        activation_checkpointing=False,
        merge_mole=merge_mole,
        compile=False,
        external_graph_gen=True,
        internal_graph_gen_version=2,
        execution_mode="general",
        auto_add_default_untrained_tasks=True,
        **extra,
    )


def checkpoint_from_env(explicit: str | None = None) -> str | None:
    """``explicit``, else ``RGPOT_UMA_CHECKPOINT``, else None (download)."""
    raw = explicit if explicit else os.environ.get("RGPOT_UMA_CHECKPOINT", "")
    raw = raw.strip()
    if not raw:
        return None
    if not Path(raw).is_file():
        raise SystemExit(f"UMA checkpoint not found: {raw}")
    return raw


def predict_unit(model: str, device: str, settings, checkpoint: str | None = None):
    """A fairchem predict unit from a local checkpoint or by model name.

    A local checkpoint carries no isolated-atom references; fairchem only
    reads those for a one-atom system, which no export accepts.
    """
    if checkpoint:
        from fairchem.core.units.mlip_unit import load_predict_unit

        return load_predict_unit(
            checkpoint, inference_settings=settings, device=device
        )
    from fairchem.core import pretrained_mlip

    return pretrained_mlip.get_predict_unit(
        model, device=device, inference_settings=settings
    )


def load_predictor(name: str, device: str, checkpoint: str | None = None,
                   merge_mole: bool = True):
    return predict_unit(name, device, _settings(merge_mole), checkpoint)


def ase_reference(
    atoms: Atoms,
    model: str,
    device: str,
    task_name: str = "omol",
    dtype_name: str = "float32",
    checkpoint: str | None = None,
):
    """ASE FAIRChemCalculator in ``dtype_name``, internal graph gen.

    float64 sets ``base_precision_dtype`` so the checkpoint is built in
    float64 and the calculator feeds float64 positions, then calls
    ``hydra.double()``. float32 does the same with ``hydra.float()``.
    The 1e-4 gate compares an export only to this same-dtype result.
    """
    from fairchem.core import FAIRChemCalculator
    from fairchem.core.units.mlip_unit.api.inference import InferenceSettings

    if dtype_name == "float64":
        precision = torch.float64
    elif dtype_name == "float32":
        precision = torch.float32
    else:
        raise ValueError(
            f"dtype_name must be float32 or float64, not {dtype_name}"
        )
    settings = InferenceSettings(
        tf32=False,
        activation_checkpointing=False,
        merge_mole=True,
        compile=False,
        external_graph_gen=False,
        execution_mode="general",
        base_precision_dtype=precision,
    )
    pred = predict_unit(model, device, settings, checkpoint)
    apply_export_dtype(pred.model, dtype_name)
    a = atoms.copy()
    a.calc = FAIRChemCalculator(pred, task_name=task_name)
    e = float(a.get_potential_energy())
    f = np.asarray(a.get_forces(), dtype=np.float64)
    return e, f, pred


def pack_example(atoms: Atoms, cutoff: float, device, dtype, max_neighbors: int | None = None,
                 n_systems: int = 1):
    """Pack one geometry as a fairchem-convention input tuple.

    n_systems > 1 replicates the geometry into a batch of independent
    systems (shared composition, per-system cell/charge/spin/natoms and
    offset edge lists): the tracing example for a band-batched export.
    """
    pos_np = np.asarray(atoms.get_positions(), dtype=np.float64)
    cell_np = np.asarray(atoms.get_cell(), dtype=np.float64)
    edge_index, shifts = _vesin_edges(pos_np, cell_np, cutoff, max_neighbors=max_neighbors)
    n = pos_np.shape[0]
    B = max(1, int(n_systems))
    pos = torch.tensor(np.tile(pos_np, (B, 1)), dtype=dtype, device=device)
    atomic_numbers = torch.tensor(
        np.tile(np.asarray(atoms.get_atomic_numbers()), B),
        dtype=torch.long, device=device)
    cell = torch.tensor(np.tile(cell_np[None, :, :], (B, 1, 1)),
                        dtype=dtype, device=device)
    pbc = torch.tensor([[True, True, True]] * B, device=device)
    ei_np = np.concatenate(
        [np.asarray(edge_index) + b * n for b in range(B)], axis=1)
    ei = torch.tensor(ei_np, dtype=torch.long, device=device)
    off = torch.tensor(np.tile(np.asarray(shifts), (B, 1)), dtype=dtype,
                       device=device)
    charge = torch.tensor([int(atoms.info.get("charge", 0))] * B,
                          dtype=torch.long, device=device)
    spin = torch.tensor([int(atoms.info.get("spin", 1))] * B,
                        dtype=torch.long, device=device)
    batch = torch.arange(B, dtype=torch.long,
                         device=device).repeat_interleave(n)
    natoms = torch.tensor([n] * B, dtype=torch.long, device=device)
    if B > 1:
        nedges = torch.tensor([np.asarray(edge_index).shape[1]] * B,
                              dtype=torch.long, device=device)
        return (pos, atomic_numbers, cell, pbc, ei, off, charge, spin,
                batch, natoms, nedges)
    return (pos, atomic_numbers, cell, pbc, ei, off, charge, spin, batch, natoms)


def compare_batched(tag: str, e, f, e_ref, f_ref, n: int,
                    e_tol=1e-4, f_tol=1e-4) -> bool:
    """Compare a batched output (B replicas of one geometry) against
    the single-system reference, per subsystem."""
    e_arr = np.atleast_1d(np.asarray(e, dtype=np.float64))
    f_arr = np.asarray(f, dtype=np.float64).reshape(-1, 3)
    ok = True
    for b in range(e_arr.shape[0]):
        ok &= compare(f"{tag}[{b}]", e_arr[b], f_arr[b * n:(b + 1) * n],
                      e_ref, f_ref, e_tol, f_tol)
    return ok


def compare(tag: str, e, f, e_ref, f_ref, e_tol=1e-4, f_tol=1e-4) -> bool:
    de = float(e) - float(e_ref)
    df = np.max(np.abs(np.asarray(f, dtype=np.float64) - f_ref))
    print(f"{tag}: E={float(e):.17g}  dE={de:.3e}  max|dF|={df:.3e}", flush=True)
    return abs(de) <= e_tol and df <= f_tol


def reference_for_dtype(export_dtype, float32_ref, float64_ref):
    """(energy, forces) of the reference that matches ``export_dtype``.

    The 1e-4 gate must not score a float64 evaluation against a float32
    reference. ``float32_ref`` and ``float64_ref`` are those pairs.
    """
    if export_dtype == "float64":
        return float64_ref
    if export_dtype == "float32":
        return float32_ref
    raise ValueError(
        f"export dtype must be float32 or float64, not {export_dtype}"
    )


def patch_export_ops():
    """Replace compiler-disabled custom autograd with plain torch ops."""
    import fairchem.core.models.uma.common.rotation as rot
    import fairchem.core.models.uma.outputs as outputs

    def _acos(x):
        return torch.acos(x.clamp(-1 + rot.EPS, 1 - rot.EPS))

    def _atan2(y, x):
        return torch.atan2(y, x)

    rot.Safeacos.apply = staticmethod(_acos)
    rot.Safeatan2.apply = staticmethod(_atan2)
    if getattr(outputs.compute_energy, "__wrapped__", None) is not None:
        outputs.compute_energy = outputs.compute_energy.__wrapped__
    elif hasattr(outputs.compute_energy, "_torchdynamo_inline"):
        pass
    # Drop the compiler.disable wrapper if present.
    fn = outputs.compute_energy
    inner = getattr(fn, "__wrapped__", None) or getattr(fn, "fn", None)
    if inner is not None:
        outputs.compute_energy = inner
    print("patched Safeacos/Safeatan2/compute_energy for export", flush=True)
    # Deterministic edge frames: the model's rotation frame per edge carries
    # a random roll gamma (torch.rand_like in init_edge_rot_euler_angles).
    # The network is equivariant in that roll, so any fixed roll is an
    # equally valid gauge; a random one leaves a random op in the exported
    # graph and the float32 output differs from run to run at rounding
    # level. Fix the roll to zero (fixed roll gauge) in every module that
    # bound the function by name.
    def _euler_angles_fixed_roll(edge_distance_vec):
        xyz = torch.nn.functional.normalize(edge_distance_vec).clamp(-1.0, 1.0)
        x, y, z = torch.split(xyz, 1, dim=1)
        beta = _acos(y.squeeze(-1))
        alpha = _atan2(x.squeeze(-1), z.squeeze(-1))
        gamma = torch.zeros_like(alpha)
        return -gamma, -beta, -alpha
    rot.init_edge_rot_euler_angles = _euler_angles_fixed_roll
    import fairchem.core.models.uma.escn_md as escn_md
    escn_md.init_edge_rot_euler_angles = _euler_angles_fixed_roll
    print("patched init_edge_rot_euler_angles: fixed roll gauge", flush=True)
    # The quaternion Wigner path (use_quaternion_wigner, the default) draws
    # the same random roll with torch.rand inside axis_angle_wigner_hybrid.
    # Route that module's torch.rand to zeros; every other torch attribute
    # is forwarded unchanged.
    import fairchem.core.models.uma.common.quaternion.wigner_d_hybrid as wdh
    class _FixedRollTorch:
        def __getattr__(self, name):
            return getattr(torch, name)
        def rand(self, *shape, **kwargs):
            return torch.zeros(*shape, **kwargs)
    wdh.torch = _FixedRollTorch()
    print("patched axis_angle_wigner_hybrid: fixed roll gauge", flush=True)
    # Integer-array indexing specialises the edge count to the traced
    # example. index_select keeps that dimension symbolic and is the
    # same gather.
    import inspect
    import textwrap

    import fairchem.core.models.uma.escn_md as escn_index
    import fairchem.core.models.uma.nn.execution_backends as backends

    src = textwrap.dedent(inspect.getsource(escn_index.eSCNMDBackbone._generate_graph))
    src = src.replace(
        'data_dict["pos"][data_dict["edge_index"][0]]',
        'data_dict["pos"].index_select(0, data_dict["edge_index"][0])',
    )
    src = src.replace(
        'data_dict["pos"][data_dict["edge_index"][1]]',
        'data_dict["pos"].index_select(0, data_dict["edge_index"][1])',
    )
    # len(natoms) == 1 is a Python branch. Tracing it freezes the system
    # count, so a packed graph could not vary the batch. The batched
    # shift is also correct for one system.
    single_shift = (
        '        if len(data_dict["natoms"]) == 1:\n'
        "            # Single system: use matmul (compile-friendly, no data-dependent ops)\n"
        '            shifts = data_dict["cell_offsets"].to(\n'
        '                data_dict["cell"].dtype\n'
        "            ) @ data_dict[\"cell\"].squeeze(0)\n"
        "        else:\n"
        "            # Batched: need repeat_interleave for variable edges per system\n"
        '            cell_per_edge = data_dict["cell"].repeat_interleave(\n'
        '                data_dict["nedges"],\n'
        "                dim=0,\n"
        '                output_size=data_dict["cell_offsets"].shape[0],\n'
        "            )\n"
        "            shifts = torch.einsum(\n"
        '                "ij,ijk->ik",\n'
        '                data_dict["cell_offsets"].to(cell_per_edge.dtype),\n'
        "                cell_per_edge,\n"
        "            )\n"
    )
    batched_shift = (
        '        cell_per_edge = data_dict["cell"].repeat_interleave(\n'
        '            data_dict["nedges"],\n'
        "            dim=0,\n"
        '            output_size=data_dict["cell_offsets"].shape[0],\n'
        "        )\n"
        "        shifts = torch.einsum(\n"
        '            "ij,ijk->ik",\n'
        '            data_dict["cell_offsets"].to(cell_per_edge.dtype),\n'
        "            cell_per_edge,\n"
        "        )\n"
    )
    if single_shift not in src:
        raise RuntimeError("UMA graph shift branch was not where the exporter expects it")
    src = src.replace(single_shift, batched_shift)
    namespace: dict = {}
    exec(src, escn_index.__dict__, namespace)  # noqa: S102
    escn_index.eSCNMDBackbone._generate_graph = namespace["_generate_graph"]

    # len(natoms) is a Python int, so the export freezes the system count.
    balance_src = textwrap.dedent(
        inspect.getsource(escn_index.balance_channels_batched)
    )
    balance_src = balance_src.replace(
        "num_systems = len(natoms)", "num_systems = natoms.shape[0]"
    )
    balance_ns: dict = {}
    exec(balance_src, escn_index.__dict__, balance_ns)  # noqa: S102
    escn_index.balance_channels_batched = balance_ns["balance_channels_batched"]

    def _node_to_edge(x_full, edge_index, wigner):
        gathered = torch.cat(
            (
                x_full.index_select(0, edge_index[0]),
                x_full.index_select(0, edge_index[1]),
            ),
            dim=2,
        )
        return torch.bmm(wigner, gathered)

    backends.UMASFastPytorchBackend.node_to_edge_wigner_permute = staticmethod(
        _node_to_edge
    )
    print("patched edge gathers: index_select", flush=True)
    # MoLE walks per-system segments with cumsum().tolist(). That Python
    # loop is a data-dependent guard, so a dynamic edge count cannot be
    # exported, and a static band specialises the segment sizes. One
    # composition shares one expert mix (checked on two Cope geometries:
    # the coefficient rows differ by 0), so the segments are one GEMM.
    import fairchem.core.models.uma.nn.mole as mole_mod
    from fairchem.core.models.uma.nn.matmul import linear_with_folded_batch

    def _mole_one_gemm(self, x):
        coeffs = self.global_mole_tensors.expert_mixing_coefficients
        with torch.autocast(device_type=self.weights.device.type, enabled=False):
            weights = torch.einsum(
                "eoi, be->boi",
                self.weights,
                coeffs,
            )
        # weights is [systems, out, in]. A same-composition pack shares
        # one expert mix, so the first row is every system's matrix.
        return linear_with_folded_batch(x, weights[0], self.bias)

    mole_mod.MOLE.forward = _mole_one_gemm
    print("patched MOLE: one GEMM for a same-composition pack", flush=True)


def dtype_name(dtype) -> str:
    """``float32`` or ``float64``. Accepts a torch dtype or that name."""
    if isinstance(dtype, str):
        name = dtype.replace("torch.", "")
    else:
        name = str(dtype).replace("torch.", "")
    if name not in ("float32", "float64"):
        raise ValueError(f"unsupported export dtype {dtype}")
    return name


def floating_dtype(module: nn.Module):
    """The single floating dtype of every parameter and buffer.

    ``None`` when the module has no floating tensor. Raises when two
    floating tensors disagree, instead of trusting the first parameter.
    """
    found: dict[torch.dtype, str] = {}
    groups = (
        ("parameter", module.named_parameters()),
        ("buffer", module.named_buffers()),
    )
    for kind, tensors in groups:
        for name, tensor in tensors:
            if tensor.is_floating_point():
                found.setdefault(tensor.dtype, f"{kind} {name}")
    if not found:
        return None
    if len(found) != 1:
        detail = "; ".join(
            f"{dtype_name(dt)} ({where})" for dt, where in found.items()
        )
        raise RuntimeError(
            "parameters and buffers do not share one floating dtype: " + detail
        )
    return next(iter(found))


def apply_export_dtype(module: nn.Module, wanted: str) -> nn.Module:
    """Convert ``module`` before the AOTI trace.

    float64 calls ``Module.double()`` so the weights the tracer sees are
    float64. float32 calls ``Module.float()`` and is the explicit
    checkpoint dtype, not an implicit default. Every floating parameter
    and buffer must agree afterwards.
    """
    if wanted == "float64":
        converted = module.double()
    elif wanted == "float32":
        converted = module.float()
    else:
        raise SystemExit(f"--dtype must be float32 or float64, not {wanted}")
    actual = floating_dtype(converted)
    if actual is not None and dtype_name(actual) != wanted:
        raise RuntimeError(
            f"after {wanted} conversion the module is {dtype_name(actual)}"
        )
    return converted


def runtime_metadata(
    path: Path,
    cutoff: float,
    max_neighbors: int,
    example,
    dtype,
    *,
    task_name: str = "omol",
    charge: int = 0,
    spin: int = 1,
    z_set: list[int] | None = None,
    counts: dict[int, int] | None = None,
    label: str = "hcn",
    model: str = "",
    molecular_box: float = 0.0,
    batch_max: int = 0,
    batch_packed: int = 0,
    model_dtype=None,
):
    input_names = [
        "pos",
        "atomic_numbers",
        "cell",
        "pbc",
        "edge_index",
        "cell_offsets",
        "charge",
        "spin",
        "batch",
        "natoms",
    ]
    if len(example) > 10:
        input_names.append("nedges")
    meta = {
        "cutoff": float(cutoff),
        "molecular_box": float(molecular_box),
        "max_neighbors": int(max_neighbors),
        "task_name": str(task_name),
        "charge": int(charge),
        "spin": int(spin),
        "z_set": [int(z) for z in (z_set or [])],
        # Per system, not per band: rgpot's UmaPot checks every input
        # against natoms and counts, so a package for C2H2 refuses C2H4
        # although both have z_set [1, 6].
        "natoms": int(sum((counts or {}).values())),
        "counts": json.dumps(
            {str(z): int(n) for z, n in sorted((counts or {}).items())}
        ),
        "label": str(label),
        "model": str(model),
        "torch_version": str(torch.__version__),
        "fairchem_version": _dist_version("fairchem-core"),
        "batch_max": int(batch_max),
        "batch_packed": int(batch_packed),
        "edge_convention": "fairchem_neighbor_center",
        "inputs": input_names,
        "outputs": ["energy", "forces"],
        "pos_dtype": dtype_name(dtype),
        "model_dtype": dtype_name(model_dtype if model_dtype is not None else dtype),
        "shapes": {
            name: list(t.shape) for name, t in zip(input_names, example)
        },
    }
    # natoms and counts are optional. A dynamic package (several atom
    # counts, one trace) leaves them out so UmaPot does not pin the
    # example size. z_set still limits the elements when it is passed.
    if counts is None:
        meta.pop("natoms", None)
        meta.pop("counts", None)
    if z_set is None:
        meta.pop("z_set", None)
    return meta


def _first_import(candidates):
    last = None
    for spec in candidates:
        mod, name = spec.rsplit(".", 1)
        try:
            module = __import__(mod, fromlist=[name])
            return getattr(module, name)
        except Exception as exc:
            last = exc
    raise RuntimeError(f"none of {candidates} imported") from last


def _honor_omp_threads() -> None:
    """The CPU GEMM template bakes ``torch.get_num_threads()`` into the kernel.

    A package compiled at one thread count and run at another is a different
    kernel than the one that was timed. When ``OMP_NUM_THREADS`` is set, the
    export uses that count.
    """
    raw = os.environ.get("OMP_NUM_THREADS", "").strip()
    if not raw.isdigit():
        return
    threads = int(raw)
    if threads < 1:
        return
    torch.set_num_threads(threads)
    print(f"torch.set_num_threads({threads})", flush=True)


def aoti_package(exported, path: Path, metadata=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fn = _first_import(
        [
            "torch.export.aoti_compile_and_package",
            "torch._inductor.aoti_compile_and_package",
            "torch._inductor.package.aoti_compile_and_package",
        ]
    )
    kwargs = {"package_path": str(path)}
    if metadata:
        # Bake the runtime contract into the package itself: the C++
        # loader reads it via AOTIModelPackageLoader::get_metadata(),
        # so the .pt2 is self-describing and needs no sidecar.
        kwargs["inductor_configs"] = {
            "aot_inductor.metadata": {
                str(k): str(v) for k, v in metadata.items()
            }
        }
    _honor_omp_threads()
    fn(canonicalize_strided_slices(exported), **kwargs)
    clear_pt2_execstack(path)
    return path


def load_aoti(path: Path):
    fn = _first_import(
        [
            "torch._inductor.aoti_load_package",
            "torch._inductor.package.load_package",
            "torch.export.load",
        ]
    )
    return fn(str(path))


def run_aoti(pkg, args):
    if hasattr(pkg, "run"):
        outs = pkg.run(list(args))
        if isinstance(outs, (list, tuple)):
            return outs[0], outs[1]
        return outs
    if callable(pkg):
        return pkg(*args)
    raise TypeError(type(pkg))


def _separated(atoms: Atoms, cutoff: float) -> Atoms:
    """Move the last atom just past the cutoff from the others.

    The neighbour list of a molecule that started fully connected then
    loses every edge that touched that atom.
    """
    out = atoms.copy()
    pos = np.array(out.get_positions(), dtype=np.float64, copy=True)
    if len(pos) < 2:
        return out
    origin = pos[:-1].mean(axis=0)
    pos[-1] = origin + np.array([0.0, 0.0, float(cutoff) + 2.0])
    out.set_positions(pos)
    return out


def try_symbolic_edges(wrap: nn.Module, example):
    """Export one graph whose edge count is symbolic.

    Returns the program, or None when the trace or the export cannot
    keep that dimension. The caller still has to score a second geometry
    against the same-dtype reference before trusting it.
    """
    from torch.fx.experimental.proxy_tensor import make_fx

    traced = tuple(tensor.clone() for tensor in example)
    torch._dynamo.mark_dynamic(traced[4], 1)
    torch._dynamo.mark_dynamic(traced[5], 0)
    print("try make_fx(tracing_mode=symbolic)", flush=True)
    try:
        graph = make_fx(
            wrap,
            tracing_mode="symbolic",
            _allow_non_fake_inputs=True,
            _allow_fake_constant=True,
        )(*traced)
    except Exception as exc:
        print(f"symbolic make_fx failed: {type(exc).__name__}: {exc}", flush=True)
        return None
    print("symbolic nodes", len(graph.graph.nodes), flush=True)
    placeholders = [node.name for node in graph.graph.nodes if node.op == "placeholder"]
    if len(placeholders) < 6:
        print("symbolic graph has no edge inputs", flush=True)
        return None
    nedges_dim = torch.export.Dim("nedges", min=1, max=65536)
    dynamic = {name: None for name in placeholders}
    dynamic[placeholders[4]] = {1: nedges_dim}
    dynamic[placeholders[5]] = {0: nedges_dim}
    try:
        return torch.export.export(
            graph, example, dynamic_shapes=dynamic, strict=False
        )
    except Exception as exc:
        print(f"symbolic export failed: {type(exc).__name__}: {exc}", flush=True)
        return None


class ExportRefusal(RuntimeError):
    """This export path would ship an edge count it did not trace."""


# Paths whose old metadata was the symbolic window 1..65536. That window
# accepts a neighbour list the trace never saw.
_WIDE_EXPORT_PATHS = ("symbolic-nedges", "dynamic-nonstrict")


def edge_export_record(export_path, traced_nedges, molecular_box, alt_nedges=None):
    """Metadata every export path must write, or a refusal.

    A molecular box pins the traced edge count: ``nedges_min`` and
    ``nedges_max`` are that count, and ``molecular`` is true. A static
    path with no box records the same exact count and ``molecular``
    false, so a later change is refused at call time. ``symbolic-nedges``,
    ``dynamic-nonstrict`` and ``make_fx-nonstrict`` with no box are
    refused, the same way ``make_fx-nonstrict`` already was. None of
    these paths records 1..65536.
    """
    known = _WIDE_EXPORT_PATHS + (
        "static-nonstrict",
        "static-strict",
        "make_fx-nonstrict",
    )
    if export_path not in known:
        raise ExportRefusal(f"unknown export path {export_path}")
    traced = int(traced_nedges)
    if traced < 1:
        raise ExportRefusal("traced edge count must be positive")
    box = float(molecular_box or 0.0)
    if box > 0.0:
        if alt_nedges is not None and int(alt_nedges) != traced:
            raise ExportRefusal(
                f"{export_path} with a molecular box changed the edge "
                f"count from {traced} to {int(alt_nedges)}"
            )
        return {
            "molecular_box": box,
            "molecular": True,
            "nedges_min": traced,
            "nedges_max": traced,
            "edge_index_shape": f"2,{traced}",
        }
    if export_path in _WIDE_EXPORT_PATHS or export_path == "make_fx-nonstrict":
        raise ExportRefusal(
            f"{export_path} without a molecular box: the traced graph "
            f"has {traced} edges, and a later geometry can have another "
            f"count"
        )
    return {
        "molecular_box": 0.0,
        "molecular": False,
        "nedges_min": traced,
        "nedges_max": traced,
        "edge_index_shape": f"2,{traced}",
    }


def _export_packed(
    wrap,
    example,
    atoms,
    cutoff,
    device,
    dtype,
    max_n,
    batch_max,
    molecular_box,
    e_ref,
    f_ref,
):
    """Disjoint graph with a dynamic atom count and a dynamic edge count.

    The trace is four systems, so the system count is not unified with
    the static 2 of ``edge_index`` or the static 3 of the coordinates.
    The exported program is then run at one system and at three. UmaPot
    sends the real count and does not pad.
    """
    n = len(atoms)
    each = n * (n - 1)
    atom_hi = n * int(batch_max)
    edge_hi = each * int(batch_max)
    nsys = torch.export.Dim("nsys", min=1, max=int(batch_max))
    atoms_d = torch.export.Dim("atoms", min=n, max=atom_hi)
    edges_d = torch.export.Dim("edges", min=each, max=edge_hi)
    print(
        f"packed dynamic atoms=[{n},{atom_hi}] edges=[{each},{edge_hi}] "
        f"systems=[1,{batch_max}]",
        flush=True,
    )
    # Positional, so an fx graph whose placeholders are not the original
    # argument names still receives the same symbols.
    dyn_seq = (
        {0: atoms_d},
        {0: atoms_d},
        {0: nsys},
        {0: nsys},
        {1: edges_d},
        {0: edges_d},
        {0: nsys},
        {0: nsys},
        {0: atoms_d},
        {0: nsys},
        {0: nsys},
    )
    exported = None
    try:
        exported = torch.export.export(
            wrap, example, dynamic_shapes=dyn_seq, strict=False
        )
        print("packed export ok torch.export", flush=True)
    except Exception as exc:
        print(
            f"packed torch.export failed: {type(exc).__name__}: {exc}",
            flush=True,
        )
    if exported is None:
        # Real autograd leaves a fake tensor in the strict exporter's
        # constants. Symbolic make_fx records the same dynamic sizes
        # without that constant.
        print("try packed make_fx symbolic", flush=True)
        from torch.fx.experimental.proxy_tensor import make_fx

        traced = tuple(tensor.clone() for tensor in example)
        try:
            graph = make_fx(
                wrap,
                tracing_mode="symbolic",
                _allow_non_fake_inputs=True,
                _allow_fake_constant=True,
            )(*traced)
            print("packed symbolic nodes", len(graph.graph.nodes), flush=True)
            exported = torch.export.export(
                graph, example, dynamic_shapes=dyn_seq, strict=False
            )
            print("packed export ok make_fx-symbolic", flush=True)
        except Exception as exc:
            print(
                f"packed export failed: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            return None

    def as_batch(n_systems):
        packed_example = pack_example(
            atoms, cutoff, device, dtype, max_n, n_systems=n_systems
        )
        if len(packed_example) == 10:
            nedges = torch.tensor(
                [int(packed_example[4].shape[1])],
                dtype=torch.long,
                device=device,
            )
            packed_example = packed_example + (nedges,)
        return packed_example

    sizes = [1, 3, 4]
    for systems in sizes:
        trial = as_batch(systems)
        with torch.enable_grad():
            energy, forces = exported.module()(*trial)
        if not compare_batched(
            f"packed[{systems}]",
            energy.detach().cpu(),
            forces.detach().cpu().numpy(),
            e_ref,
            f_ref,
            n,
        ):
            print(
                f"FAIL: packed graph at {systems} systems does not match ASE",
                file=sys.stderr,
            )
            return None
    return (
        exported,
        "packed-dynamic",
        {
            "molecular_box": float(molecular_box),
            "molecular": True,
            "nedges_min": int(each),
            "nedges_max": int(edge_hi),
            "edge_index_shape": "2,nedges",
            "batch_packed": 1,
        },
    )


# Bounds of the dynamic single-system graph. One atom has no edge and
# fairchem answers it from isolated-atom references, not the network.
DYNAMIC_NATOMS_MIN = 2
DYNAMIC_NATOMS_MAX = 1024
DYNAMIC_NEDGES_MIN = 1
# Traced instead of --atoms when --atoms cannot carry free symbols.
TRACE_MOLECULE = "CH3CH2OH"
SECOND_MOLECULE = "CH3CH2OH"
SECOND_MOLECULE_ALT = "H2O"


def dynamic_bounds(max_atoms: int, max_edges: int, max_neighbors: int):
    """(natoms_min, natoms_max, nedges_min, nedges_max) of a dynamic export.

    ``max_edges`` of 0 means every atom at the per-atom neighbour cap.
    """
    hi_atoms = int(max_atoms)
    if hi_atoms < DYNAMIC_NATOMS_MIN:
        raise SystemExit(
            f"--max-atoms must be at least {DYNAMIC_NATOMS_MIN}, not {hi_atoms}"
        )
    hi_edges = int(max_edges) if int(max_edges) > 0 else hi_atoms * int(max_neighbors)
    if hi_edges < DYNAMIC_NEDGES_MIN:
        raise SystemExit(f"--max-edges must be positive, not {hi_edges}")
    return DYNAMIC_NATOMS_MIN, hi_atoms, DYNAMIC_NEDGES_MIN, hi_edges


def trace_safe(natoms: int, nedges: int) -> bool:
    """Whether a geometry can be the example of a dynamic trace.

    Symbolic make_fx gives every input axis a symbol and merges axes of
    equal size. An atom or edge count of 1, 2 or 3 merges with the
    batch, edge_index or coordinate axis, and equal atom and edge counts
    merge with each other; export then refuses the declared Dim.
    """
    static = {0, 1, 2, 3}
    return natoms not in static and nedges not in static and natoms != nedges


def boxed_molecule(name: str, charge: int = 0, spin: int = 1,
                   side: float = 25.0) -> Atoms:
    from ase.build import molecule

    atoms = molecule(name)
    atoms.set_cell([side, side, side])
    atoms.set_pbc(True)
    atoms.center()
    atoms.info.update({"charge": int(charge), "spin": int(spin)})
    return atoms


def _input_axis(program, index: int, axis: int):
    """Symbol or int of one axis of user input ``index`` in ``program``."""
    names = list(program.graph_signature.user_inputs)
    nodes = {n.name: n for n in program.graph.nodes if n.op == "placeholder"}
    val = nodes[names[index]].meta.get("val")
    return val.shape[axis]


def dynamic_export_record(program) -> dict:
    """Atom and edge ranges the exported program accepts, read back from it.

    A specialised axis is refused: it would freeze the traced size.
    """
    out = {}
    for key, index, axis in (("natoms", 0, 0), ("nedges", 4, 1)):
        size = _input_axis(program, index, axis)
        if isinstance(size, int):
            raise ExportRefusal(f"{key} was specialised to {size}")
        expr = size.node.expr
        bound = program.range_constraints.get(expr)
        if bound is None:
            raise ExportRefusal(f"{key} has no range in the exported program")
        out[f"{key}_min"] = int(bound.lower)
        out[f"{key}_max"] = int(bound.upper)
    return {
        "molecular_box": 0.0,
        "molecular": False,
        **out,
        "pos_shape": "natoms,3",
        "edge_index_shape": "2,nedges",
    }


def export_dynamic(wrap: nn.Module, example, bounds):
    """One program with a symbolic atom count and a symbolic edge count.

    torch.export of the module itself leaves a fake tensor from the
    in-graph autograd.grad in the program constants. Symbolic make_fx
    records the same graph without it, and the export then pins the
    symbols to ``bounds``.
    """
    from torch.fx.experimental.proxy_tensor import make_fx

    lo_a, hi_a, lo_e, hi_e = bounds
    atoms_d = torch.export.Dim("natoms", min=lo_a, max=hi_a)
    edges_d = torch.export.Dim("nedges", min=lo_e, max=hi_e)
    # Positional: the make_fx graph's placeholder names are not the
    # wrapper's argument names.
    dyn = (
        {0: atoms_d},  # pos
        {0: atoms_d},  # atomic_numbers
        None,  # cell
        None,  # pbc
        {1: edges_d},  # edge_index
        {0: edges_d},  # cell_offsets
        None,  # charge
        None,  # spin
        {0: atoms_d},  # batch
        None,  # natoms
    )
    traced = tuple(tensor.clone() for tensor in example)
    graph = make_fx(
        wrap,
        tracing_mode="symbolic",
        _allow_non_fake_inputs=True,
        _allow_fake_constant=True,
    )(*traced)
    print("dynamic make_fx nodes", len(graph.graph.nodes), flush=True)
    program = torch.export.export(graph, example, dynamic_shapes=dyn, strict=False)
    return program, dynamic_export_record(program)


def merge_task_heads(hydra: nn.Module, data) -> None:
    """Fold each output head onto the task of ``data``.

    The head's expert is chosen by the dataset name alone, so this does
    not depend on the composition. The backbone keeps its MoLE layers.
    """
    module = hydra.module if hasattr(hydra, "module") else hydra
    for name, head in module.output_heads.items():
        if getattr(head, "merged_on_dataset", None) is not None:
            continue
        if hasattr(head, "merge_MOLE_model"):
            head.merge_MOLE_model(data)
            print(f"merged head {name} on {data.dataset[0]}", flush=True)


def _dynamic_checks(args, atoms, cutoff, max_n, bounds, task_name, charge,
                    spin, e_ref, f_ref, merge_mole):
    """Geometries a dynamic package must reproduce before it is kept.

    The input, the input with its last atom moved past the cutoff (fewer
    edges), a second molecule when the composition is free, and every
    --check-atoms structure. References come from FAIRChemCalculator and
    must be computed before patch_export_ops() changes fairchem.
    """
    lo_a, hi_a, lo_e, hi_e = bounds
    checks = [("input", atoms, e_ref, f_ref)]
    candidates = []
    crossed = _separated(atoms, cutoff)
    candidates.append(("cutoff", crossed))
    if not merge_mole:
        name = SECOND_MOLECULE
        if counts_of(boxed_molecule(name)) == counts_of(atoms):
            name = SECOND_MOLECULE_ALT
        candidates.append(("second", boxed_molecule(name, charge, spin)))
    for path in args.check_atoms or []:
        extra = load_atoms(path)
        extra.info["charge"] = int(charge)
        extra.info["spin"] = int(spin)
        if merge_mole and counts_of(extra) != counts_of(atoms):
            raise SystemExit(
                f"--check-atoms {path}: --merge-mole fixes the composition "
                f"{counts_of(atoms)}, this structure is {counts_of(extra)}"
            )
        candidates.append((Path(path).stem, extra))
    base_edges = int(_vesin_edges(
        np.asarray(atoms.get_positions(), dtype=np.float64),
        np.asarray(atoms.get_cell(), dtype=np.float64), cutoff,
        max_neighbors=max_n)[0].shape[1])
    for tag, cand in candidates:
        n = len(cand)
        edges = int(_vesin_edges(
            np.asarray(cand.get_positions(), dtype=np.float64),
            np.asarray(cand.get_cell(), dtype=np.float64), cutoff,
            max_neighbors=max_n)[0].shape[1])
        if tag == "cutoff" and (edges == base_edges or edges < lo_e):
            print(f"skip cutoff check: {edges} edges", flush=True)
            continue
        if not (lo_a <= n <= hi_a and lo_e <= edges <= hi_e):
            raise SystemExit(
                f"check {tag}: {n} atoms and {edges} edges are outside "
                f"natoms [{lo_a}, {hi_a}] and nedges [{lo_e}, {hi_e}]"
            )
        print(f"reference {tag}: natoms={n} nedges={edges}", flush=True)
        e_c, f_c, _ = ase_reference(
            cand, args.model, args.device, task_name,
            dtype_name=args.dtype, checkpoint=args.checkpoint,
        )
        checks.append((tag, cand, e_c, f_c))
    return checks


def run_dynamic(args, wrap, hydra, cutoff, max_n, dtype, device,
                meta, checks, bounds, merge_mole) -> int:
    """Export, package and check the dynamic single-system graph."""
    inputs = [
        (tag, a, pack_example(a, cutoff, device, dtype, max_n), e_c, f_c)
        for tag, a, e_c, f_c in checks
    ]
    for tag, _a, x, e_c, f_c in inputs:
        with torch.enable_grad():
            e_e, f_e = wrap(*x)
        if not compare(f"eager-{tag}", e_e.detach().cpu(),
                       f_e.detach().cpu().numpy(), e_c, f_c):
            print(f"FAIL: eager wrapper does not match ASE on {tag}",
                  file=sys.stderr)
            return 2
    trace = next(((tag, x) for tag, a, x, _e, _f in inputs
                  if trace_safe(len(a), int(x[4].shape[1]))), None)
    if trace is None:
        trace_atoms = boxed_molecule(
            TRACE_MOLECULE, int(meta.get("charge", 0)), int(meta.get("spin", 1))
        )
        trace = (TRACE_MOLECULE, pack_example(trace_atoms, cutoff, device,
                                              dtype, max_n))
    trace_tag, example = trace
    print(
        f"dynamic export: trace {trace_tag} natoms={example[0].shape[0]} "
        f"nedges={example[4].shape[1]} bounds natoms=[{bounds[0]},{bounds[1]}] "
        f"nedges=[{bounds[2]},{bounds[3]}]",
        flush=True,
    )
    patch_export_ops()
    dynamo_config = getattr(getattr(torch, "_dynamo", None), "config", None)
    if dynamo_config is not None and hasattr(dynamo_config, "trace_autograd_ops"):
        dynamo_config.trace_autograd_ops = True
    try:
        exported, record = export_dynamic(wrap, example, bounds)
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: dynamic export: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 5
    print(f"dynamic export ok {record}", flush=True)
    for tag, _a, x, e_c, f_c in inputs:
        with torch.enable_grad():
            e_x, f_x = exported.module()(*x)
        if not compare(f"exported-{tag}", e_x.detach().cpu(),
                       f_x.detach().cpu().numpy(), e_c, f_c):
            print(f"FAIL: exported program does not match ASE on {tag}",
                  file=sys.stderr)
            return 3
    module = hydra.module if hasattr(hydra, "module") else hydra
    meta.update(record)
    meta["export_path"] = "dynamic"
    meta["composition"] = "fixed" if merge_mole else "any"
    meta["z_max"] = int(getattr(module.backbone, "max_num_elements", 100)) - 1
    meta["validated"] = ",".join(
        f"{tag}:{len(a)}/{int(x[4].shape[1])}" for tag, a, x, _e, _f in inputs
    )
    meta["shapes"] = {
        "pos": ["natoms", 3],
        "atomic_numbers": ["natoms"],
        "cell": [1, 3, 3],
        "pbc": [1, 3],
        "edge_index": [2, "nedges"],
        "cell_offsets": ["nedges", 3],
        "charge": [1],
        "spin": [1],
        "batch": ["natoms"],
        "natoms": [1],
    }
    if args.skip_aoti:
        return 0
    out = Path(args.out)
    print("AOTI package", out, flush=True)
    aoti_package(exported, out, metadata=meta)
    pkg = load_aoti(out)
    for tag, _a, x, e_c, f_c in inputs:
        e_a, f_a = run_aoti(pkg, x)
        if not compare(f"aoti-{tag}", torch.as_tensor(e_a).detach().cpu(),
                       torch.as_tensor(f_a).detach().cpu().numpy(), e_c, f_c):
            print(f"FAIL: AOTI package does not match ASE on {tag}; removed",
                  file=sys.stderr)
            out.unlink(missing_ok=True)
            return 4
    print("WROTE", out.resolve())
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--model",
        default="uma-s-1p1",
        help="fairchem model name; downloaded unless --checkpoint is given",
    )
    p.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Local checkpoint file for --model (default: $RGPOT_UMA_CHECKPOINT; "
            "unset downloads --model from Hugging Face)"
        ),
    )
    p.add_argument(
        "--merge-mole",
        action="store_true",
        help=(
            "Dynamic export only: fold the expert mix of --atoms into the "
            "weights. Smaller package, but it is valid for that composition "
            "only and records z_set and counts. Default keeps the MoLE "
            "layers, so one package covers every composition of the task."
        ),
    )
    p.add_argument(
        "--max-atoms",
        type=int,
        default=DYNAMIC_NATOMS_MAX,
        help=f"Dynamic export: largest atom count (default {DYNAMIC_NATOMS_MAX})",
    )
    p.add_argument(
        "--max-edges",
        type=int,
        default=0,
        help=(
            "Dynamic export: largest edge count (default 0: --max-atoms "
            "times the model's per-atom neighbour cap, 300 for UMA)"
        ),
    )
    p.add_argument(
        "--check-atoms",
        action="append",
        default=[],
        metavar="PATH",
        help=(
            "Dynamic export: another structure the package must match "
            "FAIRChemCalculator on before it is kept (repeatable)"
        ),
    )
    p.add_argument("--device", default="cpu")
    p.add_argument("--atoms", default=None, help="Structure (.xyz/.con). Default: Baker HCN.")
    p.add_argument("--charge", type=int, default=None, help="Total charge (default: atoms.info or 0)")
    p.add_argument("--spin", type=int, default=None, help="Spin multiplicity (default: atoms.info or 1)")
    p.add_argument("--label", default=None, help="Metadata / filename label (default: hcn or atoms path)")
    p.add_argument("--task", default="omol", help="FAIRChem task_name (default: omol)")
    p.add_argument(
        "--dtype",
        choices=("float32", "float64"),
        default="float32",
        help=(
            "Floating dtype of the exported graph. float64 converts the "
            "model with hydra.double() before the AOTI trace. float32 "
            "converts with hydra.float() and is an explicit option."
        ),
    )
    p.add_argument("--out", default=None, help="Output .pt2 path")
    p.add_argument(
        "--molecular-box",
        type=float,
        default=0.0,
        help=(
            "Trace with the molecule re-centered in this synthetic cube and "
            "a cutoff of 0.4*L: every intramolecular pair is an edge, no "
            "periodic image ever is, so the vesin edge count is n(n-1) at "
            "every geometry (what a static-shape AOTI graph requires). "
            "The runtime reads molecular_box back from the package metadata."
        ),
    )
    p.add_argument(
        "--batch-max",
        type=int,
        default=0,
        help=(
            "Trace a fixed band of this many same-composition systems and "
            "record it as batch_max. UmaPot runs a stencil in chunks of "
            "that size and pads a short chunk by repeating its last "
            "system. 0 or 1 keeps the single-system graph. Pass "
            "--molecular-box so every geometry has the same edge count "
            "n(n-1); the make_fx fallback refuses a band with no box."
        ),
    )
    p.add_argument(
        "--batch-style",
        choices=("static", "packed"),
        default="static",
        help=(
            "static traces exactly --batch-max systems and UmaPot pads "
            "every chunk to that size. packed is one disjoint graph "
            "(concatenated atoms, a batch index, block-diagonal edges) "
            "with a dynamic atom count and a dynamic edge count; UmaPot "
            "sends the real system count, up to --batch-max, and does "
            "not repeat a system. Requires --molecular-box."
        ),
    )
    p.add_argument("--skip-aoti", action="store_true")
    p.add_argument("--eager-only", action="store_true")
    p.add_argument(
        "--compare-only",
        action="store_true",
        help="Load an existing .pt2 and compare to ASE; do not export.",
    )
    args = p.parse_args()

    atoms = load_atoms(args.atoms)
    label = args.label or default_label(args.atoms)
    charge = args.charge if args.charge is not None else int(atoms.info.get("charge", 0))
    spin = args.spin if args.spin is not None else int(atoms.info.get("spin", 1))
    atoms.info["charge"] = int(charge)
    atoms.info["spin"] = int(spin)
    if args.molecular_box and args.molecular_box > 0.0:
        L = float(args.molecular_box)
        pos = atoms.get_positions()
        atoms.set_positions(pos - pos.mean(axis=0) + [L / 2.0] * 3)
        atoms.set_cell([[L, 0.0, 0.0], [0.0, L, 0.0], [0.0, 0.0, L]])
        atoms.set_pbc(True)
    task_name = str(args.task)
    z_set = z_set_of(atoms)
    counts = counts_of(atoms)
    if args.out is None:
        args.out = f"bench_data/uma/{args.model}-{task_name}-{label}.pt2"
    out = Path(args.out)

    print(
        f"label={label} task={task_name} charge={charge} spin={spin} "
        f"z_set={z_set} counts={counts} natoms={len(atoms)}",
        flush=True,
    )
    args.checkpoint = checkpoint_from_env(args.checkpoint)
    molecular = bool(args.molecular_box and args.molecular_box > 0.0)
    dynamic = (
        not molecular
        and int(args.batch_max) <= 1
        and args.batch_style == "static"
    )
    merge_mole = bool(args.merge_mole) or not dynamic
    print(
        f"ASE FAIRChemCalculator reference {args.model} dtype={args.dtype}",
        flush=True,
    )
    e_ref, f_ref, pred = ase_reference(
        atoms, args.model, args.device, task_name, dtype_name=args.dtype,
        checkpoint=args.checkpoint,
    )
    print(f"ASE FAIRChemCalculator dtype={args.dtype} E={e_ref:.17g}", flush=True)
    for i, row in enumerate(f_ref):
        print(f"  F[{i}] {row[0]:.17g} {row[1]:.17g} {row[2]:.17g}")
    if not merge_mole:
        # The reference unit merged MoLE on this composition. The dynamic
        # graph keeps the experts and computes their mix from the input.
        precision = torch.float64 if args.dtype == "float64" else torch.float32
        pred = predict_unit(
            args.model, args.device, _settings(False, precision), args.checkpoint
        )
        apply_export_dtype(pred.model, args.dtype)

    hydra = pred.model
    backbone = hydra.module.backbone
    # ASE used internal graph gen. The exported module takes vesin edges.
    backbone.otf_graph = False
    cutoff = float(backbone.cutoff)
    max_n = int(getattr(backbone, "max_neighbors", 300) or 300)
    if args.molecular_box and args.molecular_box > 0.0:
        # Complete-graph tracing: the model envelope zeroes pairs beyond
        # its own cutoff, so widening the graph cutoff only fixes the
        # edge count (validated on claisen reactant + 1.35x stretch:
        # dE <= 3.0e-4 eV, dF <= 8.7e-4 eV/A vs the internal graph).
        cutoff = 0.4 * float(args.molecular_box)
        max_n = 100000
    print(f"cutoff={cutoff} max_neighbors={max_n}", flush=True)
    tasks = pred.dataset_to_tasks.get(task_name, [])
    for task in tasks:
        mean = float(task.normalizer.mean)
        rmsd = float(task.normalizer.rmsd)
        refs = task.element_references is not None
        print(
            f"task {task.name} prop={task.property} mean={mean:.6g} rmsd={rmsd:.6g} refs={refs}",
            flush=True,
        )

    device = torch.device(args.device)
    # ASE already ran in args.dtype (hydra.double()/float() and matching
    # inputs). AtomicData and the eager wrapper stay on that dtype.
    native = floating_dtype(hydra)
    native_dtype = native if native is not None else torch.float32

    if args.compare_only:
        if not out.is_file():
            print(f"FAIL: --compare-only needs an existing .pt2: {out}", file=sys.stderr)
            return 4
        apply_export_dtype(hydra, args.dtype)
        found = floating_dtype(hydra)
        dtype = found if found is not None else (
            torch.float64 if args.dtype == "float64" else torch.float32
        )
        example = pack_example(atoms, cutoff, device, dtype, max_n)
        meta = runtime_metadata(
            out,
            cutoff,
            max_n,
            example,
            dtype,
            model_dtype=dtype,
            task_name=task_name,
            charge=charge,
            spin=spin,
            z_set=z_set,
            counts=counts,
            label=label,
            model=args.model,
            molecular_box=float(args.molecular_box or 0.0),
            batch_max=max(0, int(args.batch_max)),
        )
        pkg = load_aoti(out)
        e_a, f_a = run_aoti(pkg, example)
        e_a = torch.as_tensor(e_a).detach().cpu()
        f_a = torch.as_tensor(f_a).detach().cpu().numpy()
        if not compare("aoti", e_a, f_a, e_ref, f_ref):
            print("FAIL: AOTI package does not match ASE", file=sys.stderr)
            return 4
        print("OK existing", out.resolve())
        return 0

    data_v = vesin_atomic_data(
        atoms, cutoff, native_dtype, device, max_n, task_name=task_name, sid=label
    )
    print(f"vesin AtomicData edges={int(data_v.nedges.item())}", flush=True)
    pred_v = pred.predict(data_v)
    e_ad = pred_v["energy"].detach().cpu()
    f_ad = pred_v["forces"].detach().cpu().numpy()
    ok_ad = compare("atomicdata+vesin", e_ad, f_ad, e_ref, f_ref)
    if not ok_ad:
        print("FAIL: vesin AtomicData via MLIPPredictUnit does not match ASE", file=sys.stderr)
        return 2
    if not merge_mole:
        merge_task_heads(hydra, data_v)

    # float64: hydra.double() before the trace. float32: hydra.float().
    apply_export_dtype(hydra, args.dtype)
    found = floating_dtype(hydra)
    dtype = found if found is not None else (
        torch.float64 if args.dtype == "float64" else torch.float32
    )
    print(
        f"export pos_dtype={dtype_name(dtype)} model_dtype={dtype_name(dtype)}",
        flush=True,
    )

    batch_max = max(0, int(args.batch_max))
    packed = args.batch_style == "packed"
    if packed and not (args.molecular_box and args.molecular_box > 0.0):
        print("FAIL: --batch-style packed needs --molecular-box", file=sys.stderr)
        return 2
    # A static band is traced at exactly batch_max systems and UmaPot
    # pads every chunk. A packed graph is traced at four systems: two
    # would unify with edge_index's leading 2, and three with the
    # coordinate axis. The atom and edge counts stay dynamic up to
    # batch_max.
    if packed and batch_max < 4:
        print(
            "FAIL: --batch-style packed needs --batch-max >= 4 "
            "(2 and 3 unify with static input axes)",
            file=sys.stderr,
        )
        return 2
    trace_systems = 4 if packed else (batch_max if batch_max > 1 else 1)
    example = pack_example(atoms, cutoff, device, dtype, max_n,
                           n_systems=trace_systems)
    print(f"vesin edges={example[4].shape[1]} systems={trace_systems}",
          flush=True)

    wrap_cls = TensorUmaBatched if packed or batch_max > 1 else TensorUma
    wrap = wrap_cls(hydra, pred_unit=pred, task_name=task_name).to(device).eval()
    with torch.enable_grad():
        e_e, f_e = wrap(*example)
    ok_eager = compare_batched("eager+vesin", e_e.detach().cpu(), f_e.detach().cpu().numpy(), e_ref, f_ref, len(atoms))
    if not ok_eager:
        print("FAIL: eager tensor wrapper does not match ASE", file=sys.stderr)
        return 2
    meta = runtime_metadata(
        out,
        cutoff,
        max_n,
        example,
        dtype,
        model_dtype=dtype,
        task_name=task_name,
        charge=charge,
        spin=spin,
        # An unmerged graph computes the expert mix from its input, so
        # it records no composition.
        z_set=z_set if merge_mole else None,
        counts=counts if merge_mole else None,
        label=label,
        model=args.model,
        molecular_box=float(args.molecular_box or 0.0),
        batch_max=max(0, int(args.batch_max)),
        batch_packed=1 if packed else 0,
    )
    if args.eager_only:
        return 0

    if dynamic:
        bounds = dynamic_bounds(args.max_atoms, args.max_edges, max_n)
        checks = _dynamic_checks(
            args, atoms, cutoff, max_n, bounds, task_name, charge, spin,
            e_ref, f_ref, merge_mole,
        )
        return run_dynamic(
            args, wrap, hydra, cutoff, max_n, dtype, device, meta,
            checks, bounds, merge_mole,
        )

    patch_export_ops()
    # torch 2.14 refuses torch.autograd.grad inside a strict export while
    # this is false. The grad is the force, so the export has to trace it.
    # Older torch builds have no such switch and are left alone.
    dynamo = getattr(torch, "_dynamo", None)
    dynamo_config = getattr(dynamo, "config", None)
    if dynamo_config is not None and hasattr(dynamo_config, "trace_autograd_ops"):
        dynamo_config.trace_autograd_ops = True
        print("torch._dynamo.config.trace_autograd_ops = True", flush=True)
    print("torch.export", flush=True)
    exported = None
    export_path = ""
    edge_record = None
    example_alt = None
    e_alt_ref = None
    f_alt_ref = None
    if packed:
        packed_out = _export_packed(
            wrap,
            example,
            atoms,
            cutoff,
            device,
            dtype,
            max_n,
            batch_max,
            float(args.molecular_box),
            e_ref,
            f_ref,
        )
        if packed_out is None:
            return 5
        exported, export_path, edge_record = packed_out
    natoms_dim = torch.export.Dim("natoms", min=1, max=256)
    nedges_min, nedges_max = 1, 65536
    nedges_dim = torch.export.Dim("nedges", min=nedges_min, max=nedges_max)
    dyn = {
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
    if packed or batch_max > 1:
        # A packed graph was exported above. A static band is exact-B.
        dyn = None
    # Atom count stays fixed: a three-atom trace unifies that count with
    # the trailing 3 of the coordinates, so natoms is not a free symbol.
    # The edge count is free. A second geometry, scored against a
    # same-dtype reference, is what makes the package eligible.
    if dyn is not None and int(example[4].shape[1]) > 1:
        alt_atoms = _separated(atoms, cutoff)
        example_alt = pack_example(alt_atoms, cutoff, device, dtype, max_n)
        if int(example_alt[4].shape[1]) != int(example[4].shape[1]):
            print(
                f"alt edges={int(example_alt[4].shape[1])} "
                f"(traced {int(example[4].shape[1])})",
                flush=True,
            )
            e_alt_ref, f_alt_ref, _alt_pred = ase_reference(
                alt_atoms, args.model, args.device, task_name, dtype_name=args.dtype
            )
            with torch.enable_grad():
                e_eager_alt, f_eager_alt = wrap(*example_alt)
            if not compare(
                "eager-alt",
                e_eager_alt.detach().cpu(),
                f_eager_alt.detach().cpu().numpy(),
                e_alt_ref,
                f_alt_ref,
            ):
                print(
                    "FAIL: eager tensor wrapper does not match the "
                    "same-dtype reference on the second graph",
                    file=sys.stderr,
                )
                return 2
            candidate = try_symbolic_edges(wrap, example)
            if candidate is not None:
                with torch.no_grad():
                    e_sym, f_sym = candidate.module()(*example)
                    e_sym_alt, f_sym_alt = candidate.module()(*example_alt)
                ok_sym = compare_batched(
                    "symbolic", e_sym.detach().cpu(),
                    f_sym.detach().cpu().numpy(), e_ref, f_ref, len(atoms),
                )
                ok_alt = compare(
                    "symbolic-alt",
                    e_sym_alt.detach().cpu(),
                    f_sym_alt.detach().cpu().numpy(),
                    e_alt_ref,
                    f_alt_ref,
                )
                if ok_sym and ok_alt:
                    try:
                        edge_record = edge_export_record(
                            "symbolic-nedges",
                            int(example[4].shape[1]),
                            float(args.molecular_box or 0.0),
                            int(example_alt[4].shape[1]),
                        )
                    except ExportRefusal as exc:
                        print(f"symbolic-nedges refused: {exc}", flush=True)
                        example_alt = None
                    else:
                        exported = candidate
                        export_path = "symbolic-nedges"
                        print("export ok symbolic-nedges", flush=True)
                else:
                    print(
                        "symbolic graph does not match the same-dtype "
                        "reference; keeping a static package",
                        flush=True,
                    )
                    example_alt = None
    attempts = [
        ("static-nonstrict", dict(strict=False)),
        ("static-strict", dict(strict=True)),
    ]
    if dyn is not None:
        attempts.insert(0, ("dynamic-nonstrict",
                            dict(dynamic_shapes=dyn, strict=False)))
    last = None
    if exported is None:
      for attempt, kwargs in attempts:
        try:
            print("try export", attempt, flush=True)
            candidate = torch.export.export(wrap, example, **kwargs)
            # Export acceptance requires executable tensors and the same
            # reference comparison as the packaged model.
            with torch.enable_grad():
                e_trial, f_trial = candidate.module()(*example)
            if not compare_batched(
                f"exported-{attempt}", e_trial.detach().cpu(),
                f_trial.detach().cpu().numpy(), e_ref, f_ref, len(atoms)
            ):
                raise RuntimeError("exported program does not match ASE")
            try:
                edge_record = edge_export_record(
                    attempt,
                    int(example[4].shape[1]),
                    float(args.molecular_box or 0.0),
                )
            except ExportRefusal as exc:
                print(f"{attempt} refused: {exc}", flush=True)
                continue
            exported = candidate
            print("export ok", attempt, type(exported), flush=True)
            export_path = attempt
            break
        except Exception as exc:
            last = exc
            print(f"{attempt} failed: {type(exc).__name__}: {exc}", flush=True)
    if exported is None:
        # torch 2.13 with fairchem-core 2.23 lands here for every UMA
        # export: the nonstrict exports fail on a fake tensor in the
        # exported program's constants and the strict export on
        # torch.autograd.grad. Real-mode make_fx records the concrete
        # example shapes, so the package is static: atom count and edge
        # count are those of the traced geometry.
        print("try make_fx(tracing_mode=real)", flush=True)
        from torch.fx.experimental.proxy_tensor import make_fx

        gm = make_fx(wrap, tracing_mode="real")(*example)
        print("make_fx nodes", len(gm.graph.nodes), flush=True)
        try:
            exported = torch.export.export(gm, example, strict=False)
            print("export ok make_fx-nonstrict", type(exported), flush=True)
            export_path = "make_fx-nonstrict"
            edge_record = edge_export_record(
                export_path,
                int(example[4].shape[1]),
                float(args.molecular_box or 0.0),
            )
        except ExportRefusal as exc:
            print(f"FAIL: {exc}", file=sys.stderr)
            return 5
        except Exception as exc:
            last = exc
            print(f"make_fx export failed: {type(exc).__name__}: {exc}", flush=True)
            raise last from exc

    print("export ok", type(exported), flush=True)
    if edge_record is None:
        print(
            f"FAIL: {export_path} produced no edge record",
            file=sys.stderr,
        )
        return 5
    meta["export_path"] = export_path
    meta["pos_shape"] = f"{len(atoms)},3"
    # Exact traced bounds. A count outside them is refused at call time.
    meta.update(edge_record)
    with torch.enable_grad():
        e_x, f_x = exported.module()(*example)
    if not compare_batched("exported", e_x.detach().cpu(), f_x.detach().cpu().numpy(), e_ref, f_ref, len(atoms)):
        print("FAIL: exported module does not match ASE", file=sys.stderr)
        return 3

    if args.skip_aoti:
        return 0

    out = Path(args.out)
    print("AOTI package", out, flush=True)
    aoti_package(exported, out, metadata=meta)
    pkg = load_aoti(out)
    e_a, f_a = run_aoti(pkg, example)
    e_a = torch.as_tensor(e_a).detach().cpu()
    f_a = torch.as_tensor(f_a).detach().cpu().numpy()
    if not compare_batched("aoti", e_a, f_a, e_ref, f_ref, len(atoms)):
        print("FAIL: AOTI package does not match ASE", file=sys.stderr)
        return 4
    if export_path == "symbolic-nedges" and example_alt is not None:
        e_b, f_b = run_aoti(pkg, example_alt)
        e_b = torch.as_tensor(e_b).detach().cpu()
        f_b = torch.as_tensor(f_b).detach().cpu().numpy()
        if not compare("aoti-alt", e_b, f_b, e_alt_ref, f_alt_ref):
            print(
                "FAIL: AOTI package does not match the same-dtype "
                "reference on the second graph",
                file=sys.stderr,
            )
            return 4
    # The band contract is exact-B: every call carries batch_max
    # systems (padded by the caller); the per-subsystem comparison
    # above already exercised the packaged graph at that size.
    print("WROTE", out.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
