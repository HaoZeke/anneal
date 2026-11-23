#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Cold-rescore recorded GFN2 geometries with standalone xtb."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from gfn2_crest import fisher_exact, two_sided_mw

EH_TO_EV = 27.21138505
ENERGY = re.compile(r":: total energy\s+([-+0-9.Ee]+) Eh")
FIELDS = [
    "system",
    "code",
    "seed",
    "budget",
    "recorded_eV",
    "geometry_kept",
    "default_scc_converged",
    "default_xtb_eV",
    "default_recorded_minus_xtb_eV",
    "tight_scc_converged",
    "tight_xtb_eV",
    "tight_recorded_minus_xtb_eV",
    "min_distance_A",
    "coordination_min",
    "coordination_max",
    "false_hit",
    "rescored_hit",
]


def parse_xtb(text: str, returncode: int) -> tuple[bool, float | None]:
    matches = ENERGY.findall(text)
    converged = (
        returncode == 0
        and bool(matches)
        and "normal termination of xtb" in text
        and "SCC iterator did not converge" not in text
        and "Self consistent charge iterator did not converge" not in text
    )
    return converged, float(matches[-1]) * EH_TO_EV if converged else None


def read_xyz(path: Path) -> list[tuple[str, tuple[float, float, float]]]:
    lines = path.read_text().splitlines()
    n = int(lines[0])
    atoms = []
    for line in lines[2 : 2 + n]:
        symbol, x, y, z = line.split()[:4]
        atoms.append((symbol, (float(x), float(y), float(z))))
    if len(atoms) != n:
        raise ValueError(f"{path} has {len(atoms)} atoms, expected {n}")
    return atoms


def geometry_metrics(
    atoms: list[tuple[str, tuple[float, float, float]]], cutoff: float
) -> tuple[float, int, int]:
    distances = []
    coordination = [0] * len(atoms)
    for i, (_, left) in enumerate(atoms):
        for j, (_, right) in enumerate(atoms[i + 1 :], i + 1):
            distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))
            distances.append(distance)
            if distance <= cutoff:
                coordination[i] += 1
                coordination[j] += 1
    return min(distances), min(coordination), max(coordination)


def geometry_path(root: Path, row: dict[str, str]) -> Path:
    name = "crest_best.xyz" if row["code"] == "crest" else "best.xyz"
    return root / row["system"] / row["code"] / row["seed"] / name


def run_xtb(
    xtb: Path, geometry: Path, accuracy: float | None, where: Path
) -> tuple[bool, float | None, str]:
    where.mkdir(parents=True)
    local = where / "geometry.xyz"
    shutil.copyfile(geometry, local)
    command = [str(xtb), local.name, "--gfn", "2", "--sp"]
    if accuracy is not None:
        command.extend(["--acc", str(accuracy)])
    environment = dict(os.environ)
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "RAYON_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "GOTO_NUM_THREADS",
    ):
        environment[name] = "1"
    result = subprocess.run(
        command,
        cwd=where,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    text = result.stdout
    return (*parse_xtb(text, result.returncode), text)


def number(value: float | None, digits: int = 8) -> str:
    return "" if value is None else f"{value:.{digits}f}"


def append_rows(path: Path, rows: list[dict[str, object]]) -> None:
    new = not path.exists() or path.stat().st_size == 0
    if not new:
        with path.open(newline="") as handle:
            if next(csv.reader(handle)) != FIELDS:
                raise SystemExit(f"{path} has another header")
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        if new:
            writer.writeheader()
        writer.writerows(rows)


def rank(row: dict[str, object]) -> float:
    return (
        float(row["original_evals_to_hit"])
        if row["rescored_hit"] == "true"
        else float(row["budget"]) + 1.0
    )


def summary(rows: list[dict[str, object]]) -> dict[str, object]:
    groups = {
        code: [row for row in rows if row["code"] == code]
        for code in ("anneal-general", "crest")
    }
    anneal, crest = groups["anneal-general"], groups["crest"]
    anneal_hits = sum(row["rescored_hit"] == "true" for row in anneal)
    crest_hits = sum(row["rescored_hit"] == "true" for row in crest)
    return {
        "anneal_hits": anneal_hits,
        "anneal_runs": len(anneal),
        "crest_hits": crest_hits,
        "crest_runs": len(crest),
        "fisher_p": fisher_exact(
            [
                [crest_hits, len(crest) - crest_hits],
                [anneal_hits, len(anneal) - anneal_hits],
            ]
        ).pvalue,
        "mannwhitney_p": two_sided_mw(
            [rank(row) for row in crest], [rank(row) for row in anneal]
        ),
    }


def rescore(args: argparse.Namespace) -> None:
    with args.rows.open(newline="") as handle:
        source = list(csv.DictReader(handle))
    output = []
    args.logs.mkdir(parents=True, exist_ok=True)
    for row in source:
        geometry = geometry_path(args.runs, row)
        record: dict[str, object] = {
            "system": row["system"],
            "code": row["code"],
            "seed": row["seed"],
            "budget": row["budget"],
            "recorded_eV": row["best"],
            "geometry_kept": "true" if geometry.is_file() else "false",
            "original_evals_to_hit": row["evals_to_hit"],
        }
        if not geometry.is_file():
            record.update(
                {
                    "default_scc_converged": "false",
                    "default_xtb_eV": "",
                    "default_recorded_minus_xtb_eV": "",
                    "tight_scc_converged": "false",
                    "tight_xtb_eV": "",
                    "tight_recorded_minus_xtb_eV": "",
                    "min_distance_A": "",
                    "coordination_min": "",
                    "coordination_max": "",
                    "false_hit": "true",
                    "rescored_hit": "false",
                }
            )
            output.append(record)
            continue
        minimum, coord_min, coord_max = geometry_metrics(
            read_xyz(geometry), args.cutoff
        )
        results = {}
        for label, accuracy in (("default", None), ("tight", args.tight_accuracy)):
            where = args.work / row["system"] / row["code"] / row["seed"] / label
            converged, energy, text = run_xtb(args.xtb, geometry, accuracy, where)
            (args.logs / f"{row['system']}-{row['code']}-{row['seed']}-{label}.log").write_text(
                text
            )
            results[label] = (converged, energy)
        default_ok, default_energy = results["default"]
        tight_ok, tight_energy = results["tight"]
        recorded = float(row["best"])
        default_difference = (
            recorded - default_energy if default_energy is not None else None
        )
        tight_difference = recorded - tight_energy if tight_energy is not None else None
        false_hit = (
            not default_ok
            or not tight_ok
            or abs(default_difference) > args.energy_tolerance
            or abs(tight_difference) > args.energy_tolerance
        )
        rescored_hit = (
            tight_ok
            and tight_energy is not None
            and tight_energy <= args.target + args.hit_tolerance
        )
        record.update(
            {
                "default_scc_converged": str(default_ok).lower(),
                "default_xtb_eV": number(default_energy),
                "default_recorded_minus_xtb_eV": number(default_difference),
                "tight_scc_converged": str(tight_ok).lower(),
                "tight_xtb_eV": number(tight_energy),
                "tight_recorded_minus_xtb_eV": number(tight_difference),
                "min_distance_A": number(minimum, 6),
                "coordination_min": coord_min,
                "coordination_max": coord_max,
                "false_hit": str(false_hit).lower(),
                "rescored_hit": str(rescored_hit).lower(),
            }
        )
        output.append(record)
    append_rows(args.out, [{key: row[key] for key in FIELDS} for row in output])
    print(json.dumps(summary(output), sort_keys=True))


def self_test() -> None:
    text = ":: total energy -2.000000000000 Eh ::\nnormal termination of xtb"
    converged, energy = parse_xtb(text, 0)
    assert converged and abs(energy + 2 * EH_TO_EV) < 1e-12
    assert parse_xtb(text, 1) == (False, None)
    atoms = [
        ("Si", (0.0, 0.0, 0.0)),
        ("Si", (2.0, 0.0, 0.0)),
        ("Si", (5.0, 0.0, 0.0)),
    ]
    assert geometry_metrics(atoms, 2.8) == (2.0, 0, 1)
    rows = [
        {
            "code": "anneal-general",
            "rescored_hit": "true",
            "original_evals_to_hit": "2",
            "budget": "10",
        },
        {
            "code": "crest",
            "rescored_hit": "false",
            "original_evals_to_hit": "7",
            "budget": "7",
        },
    ]
    result = summary(rows)
    assert result["anneal_hits"] == 1 and result["crest_hits"] == 0
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "rows.csv"
        row = {field: "" for field in FIELDS}
        append_rows(path, [row])
        append_rows(path, [row])
        assert len(path.read_text().splitlines()) == 3
    print("ok")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path)
    parser.add_argument("--runs", type=Path)
    parser.add_argument("--xtb", type=Path)
    parser.add_argument("--target", type=float)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--logs", type=Path)
    parser.add_argument("--work", type=Path)
    parser.add_argument("--tight-accuracy", type=float, default=0.01)
    parser.add_argument("--energy-tolerance", type=float, default=0.01)
    parser.add_argument("--hit-tolerance", type=float, default=0.001)
    parser.add_argument("--cutoff", type=float, default=2.8)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    required = ("rows", "runs", "xtb", "target", "out", "logs", "work")
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        parser.error("missing " + ", ".join("--" + name for name in missing))
    rescore(args)


if __name__ == "__main__":
    main()
