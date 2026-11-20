#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Label every accepted minimum of a trace with CNA shares and a funnel.

Usage: classify.py SYSTEM TRACE.xyz[.zst] > minima.csv

A trace is a run of extended XYZ frames, one per accepted minimum, whose
comment line carries ``call=<force calls> hop=<hop> move=<move>
energy=<energy>``. The rules, cutoffs and thresholds are the ones declared in
``bench/g3-funnel/classify-rules.md``.
"""

from __future__ import annotations

import csv
import io
import sys
from collections import Counter
from pathlib import Path

import numpy as np

LJ_CUTOFF = 1.39
MORSE_CUTOFF = LJ_CUTOFF / 2.0 ** (1.0 / 6.0)
ORDERED_MIN = 0.25
ICOSAHEDRAL_555 = 0.035
FIELDS = ["index", "call", "hop", "move", "energy", "f421", "f422", "f555", "fother", "funnel"]


def cutoff(system: str) -> float:
    if system.startswith("lj"):
        return LJ_CUTOFF
    if system.startswith("m"):
        return MORSE_CUTOFF
    raise ValueError(f"no CNA cutoff for {system}")


def funnel(f421: float, f422: float, f555: float) -> str:
    if f421 + f422 + f555 < ORDERED_MIN:
        return "other"
    if f555 >= ICOSAHEDRAL_555:
        return "icosahedral"
    if f555 > 0.0:
        return "decahedral"
    return "fcc"


def open_text(path: Path) -> io.TextIOBase:
    if path.suffix == ".zst":
        import zstandard

        return io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(path.open("rb")))
    return path.open()


def frames(path: Path):
    with open_text(path) as handle:
        while True:
            head = handle.readline()
            if not head.strip():
                return
            n = int(head)
            meta = dict(field.split("=", 1) for field in handle.readline().split())
            xyz = np.array([[float(v) for v in handle.readline().split()[1:4]] for _ in range(n)])
            yield meta, xyz


class Cna:
    def __init__(self, rc: float):
        from ovito.data import DataCollection
        from ovito.modifiers import CommonNeighborAnalysisModifier, CreateBondsModifier
        from ovito.pipeline import Pipeline, StaticSource

        self.data_collection = DataCollection
        self.source = StaticSource()
        self.pipeline = Pipeline(source=self.source)
        self.pipeline.modifiers.append(CreateBondsModifier(cutoff=rc))
        self.pipeline.modifiers.append(
            CommonNeighborAnalysisModifier(mode=CommonNeighborAnalysisModifier.Mode.BondBased)
        )

    def shares(self, xyz: np.ndarray) -> tuple[float, float, float, float]:
        data = self.data_collection()
        span = xyz.max(axis=0) - xyz.min(axis=0) + 20.0
        cell = data.create_cell(np.diag(span), pbc=(False, False, False))
        cell[:, 3] = xyz.min(axis=0) - 10.0
        data.create_particles(count=len(xyz)).create_property("Position", data=xyz)
        self.source.data = data
        out = self.pipeline.compute()
        if out.particles.bonds is None or out.particles.bonds.count == 0:
            return 0.0, 0.0, 0.0, 1.0
        counts = Counter(map(tuple, np.asarray(out.particles.bonds["CNA Indices"]).tolist()))
        total = sum(counts.values())
        f421 = counts[(4, 2, 1)] / total
        f422 = counts[(4, 2, 2)] / total
        f555 = counts[(5, 5, 5)] / total
        return f421, f422, f555, 1.0 - f421 - f422 - f555


def classify(system: str, path: Path, out) -> None:
    cna = Cna(cutoff(system))
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(FIELDS)
    for index, (meta, xyz) in enumerate(frames(path)):
        f421, f422, f555, fother = cna.shares(xyz)
        writer.writerow(
            [
                index,
                meta["call"],
                meta["hop"],
                meta["move"],
                meta["energy"],
                f"{f421:.4f}",
                f"{f422:.4f}",
                f"{f555:.4f}",
                f"{fother:.4f}",
                funnel(f421, f422, f555),
            ]
        )


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    classify(sys.argv[1], Path(sys.argv[2]), sys.stdout)


if __name__ == "__main__":
    main()
