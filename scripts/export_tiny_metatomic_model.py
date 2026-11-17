#!/usr/bin/env python3
# Copyright (c) rgpot developers
"""Metatomic models with no weights that read the total charge and spin.

The energy in eV is ``0.5 * sum(positions**2) + 10 * charge + multiplicity``,
so the forces are ``-positions`` and a change of charge or multiplicity moves
the energy by a known amount. ``--inputs`` names the system data the model
declares in ``requested_inputs`` and reads: ``charge`` is ``rgpot::charge``,
``spin_multiplicity`` and ``spin`` are ``rgpot::spin_multiplicity`` and
``rgpot::spin``. An empty list declares and reads nothing. ``--undeclared``
reads the inputs without declaring them. Data the system lacks is read as
charge 0 and multiplicity 1.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import (
    AtomisticModel,
    ModelCapabilities,
    ModelMetadata,
    ModelOutput,
    System,
)

# name: (energy per unit, value of a system without the data)
INPUTS = {
    "charge": (10.0, 0.0),
    "spin_multiplicity": (1.0, 1.0),
    "spin": (1.0, 1.0),
}


class ChargeSpinEnergy(torch.nn.Module):
    names: List[str]
    weights: List[float]
    defaults: List[float]
    declared: bool

    def __init__(self, inputs: List[str], declared: bool):
        super().__init__()
        self.names = ["rgpot::" + name for name in inputs]
        self.weights = [INPUTS[name][0] for name in inputs]
        self.defaults = [INPUTS[name][1] for name in inputs]
        self.declared = declared

    def requested_inputs(self) -> Dict[str, ModelOutput]:
        requested: Dict[str, ModelOutput] = {}
        if not self.declared:
            return requested
        for name in self.names:
            if name == "rgpot::charge":
                requested[name] = ModelOutput(
                    quantity="charge", unit="e", per_atom=False
                )
            else:
                requested[name] = ModelOutput(per_atom=False)
        return requested

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels],
    ) -> Dict[str, TensorMap]:
        energies: List[torch.Tensor] = []
        for system in systems:
            energy = 0.5 * torch.sum(system.positions**2)
            known = system.known_data()
            for i in range(len(self.names)):
                if self.names[i] in known:
                    value = system.get_data(self.names[i]).block().values.sum()
                    energy = energy + self.weights[i] * value
                else:
                    energy = energy + self.weights[i] * self.defaults[i]
            energies.append(energy.reshape(1, 1))
        device = systems[0].positions.device
        block = TensorBlock(
            values=torch.cat(energies),
            samples=Labels(
                "system",
                torch.arange(len(systems), dtype=torch.int32, device=device).reshape(
                    -1, 1
                ),
            ),
            components=[],
            properties=Labels(
                "energy", torch.zeros((1, 1), dtype=torch.int32, device=device)
            ),
        )
        keys = Labels("_", torch.zeros((1, 1), dtype=torch.int32, device=device))
        return {"energy": TensorMap(keys, [block])}


def export(inputs: List[str], declared: bool, path: Path) -> None:
    capabilities = ModelCapabilities(
        outputs={
            "energy": ModelOutput(quantity="energy", unit="eV", per_atom=False)
        },
        atomic_types=[1, 6, 7, 8],
        interaction_range=0.0,
        length_unit="angstrom",
        supported_devices=["cpu"],
        dtype="float64",
    )
    name = "rgpot-charge-spin-" + ("-".join(inputs) or "none")
    if not declared:
        name += "-undeclared"
    model = AtomisticModel(
        ChargeSpinEnergy(inputs, declared).eval(),
        ModelMetadata(name=name),
        capabilities,
    )
    model.save(str(path))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        default="",
        help="comma-separated subset of: " + ", ".join(INPUTS),
    )
    parser.add_argument(
        "--undeclared",
        action="store_true",
        help="read the inputs without declaring them in requested_inputs",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    inputs = [name for name in args.inputs.split(",") if name]
    unknown = [name for name in inputs if name not in INPUTS]
    if unknown:
        raise SystemExit(f"unknown inputs: {', '.join(unknown)}")
    export(inputs, not args.undeclared, args.out)
    print(f"WROTE {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
