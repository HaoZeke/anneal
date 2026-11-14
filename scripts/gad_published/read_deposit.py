#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Energy evaluations in the deposited Guided Adaptive Diffusion runs.

The archive is Zenodo 10.5281/zenodo.19001618. Each Lennard-Jones file
holds one evolutionary run. A generation stores one fitness per sample.
Fitness is the negated potential when the run maximises. The relaxer
string records a FIRE step cap. No dataset records the number of steps
a sample actually took. One generation therefore costs

    samples * recorded relaxation steps

potential evaluations, and the run costs that quantity times the number
of generations up to and including the first generation whose best
energy is within 0.001 of the Cambridge value. ``model_loss`` is the
diffusion training loss and is not included. These figures are read
from the deposit. They are not a rerun.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path

import h5py
import numpy as np

ZENODO_DOI = "10.5281/zenodo.19001618"
ZENODO_MD5 = "e140fde0c62c71f08b326b12ac50d77e"
HIT_TOL = 0.001
REFERENCES = {31: -133.586422, 38: -173.928427}
MEMBERS = {
    31: "lj-31/lj-31_evolution.h5",
    38: "lj-38/lj-38_evolution.h5",
}
STEP_DATASETS = ("relax_steps", "nsteps", "n_steps", "relaxation_steps")


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def attr_text(value) -> str:
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.ndarray) and value.shape == ():
        return attr_text(value[()])
    return str(value)


def recorded_step_cap(es_config: str) -> int:
    found = re.search(r"steps=(\d+)", es_config)
    if not found:
        raise ValueError("the relaxer string records no step cap")
    steps = int(found.group(1))
    if steps < 1:
        raise ValueError(f"recorded step cap {steps}")
    return steps


def maximises(objective: str) -> bool:
    payload = json.loads(objective)
    return bool(payload["maximize"])


def generation_index(name: str) -> int:
    if not name.startswith("gen_"):
        raise ValueError(f"unexpected group {name}")
    return int(name.split("_", 1)[1])


def calls_in_generation(group, n_samples: int, cap: int) -> tuple[int, str]:
    """Potential evaluations charged to one generation, and where the step count came from."""
    for key in STEP_DATASETS:
        if key not in group:
            continue
        recorded = np.asarray(group[key])
        if recorded.shape == ():
            return n_samples * int(recorded), key
        return int(np.sum(recorded)), key
    return n_samples * cap, "relaxer cap"


def read_run(handle, n_atoms: int) -> dict:
    runs = sorted(key for key in handle.keys() if key.startswith("run_"))
    if not runs:
        raise ValueError(f"LJ{n_atoms} file has no run")
    reference = REFERENCES[n_atoms]
    level = reference + HIT_TOL
    parsed = []
    for run_name in runs:
        run = handle[run_name]
        es_config = attr_text(run.attrs["es_config"])
        cap = recorded_step_cap(es_config)
        negate = maximises(attr_text(run.attrs["objective"]))
        names = [key for key in run.keys() if key.startswith("gen_")]
        order = sorted(names, key=generation_index)
        indexes = [generation_index(name) for name in order]
        if indexes != list(range(len(indexes))):
            raise ValueError(f"{run_name} generations are not 0..{len(indexes) - 1}")
        best = None
        hit_index = None
        evals_to_hit = None
        spent = 0
        step_source = None
        n_samples = None
        for index, name in zip(indexes, order):
            group = run[name]
            fitness = np.asarray(group["fitness"], dtype=np.float64)
            if fitness.ndim != 1 or fitness.size < 1:
                raise ValueError(f"{run_name}/{name} fitness is not a sample vector")
            if n_samples is None:
                n_samples = int(fitness.size)
            elif int(fitness.size) != n_samples:
                raise ValueError(f"{run_name}/{name} changed the sample count")
            calls, source = calls_in_generation(group, n_samples, cap)
            if step_source is None:
                step_source = source
            elif source != step_source:
                raise ValueError(f"{run_name} mixes step records")
            spent += calls
            energy = -fitness if negate else fitness
            generation_best = float(np.min(energy))
            if best is None or generation_best < best:
                best = generation_best
            if hit_index is None and generation_best <= level:
                hit_index = index
                evals_to_hit = spent
        if best is None or n_samples is None or step_source is None:
            raise ValueError(f"{run_name} has no generations")
        reached = hit_index is not None
        parsed.append(
            {
                "system": f"lj{n_atoms}",
                "n_atoms": n_atoms,
                "run": run_name,
                "generations": len(indexes),
                "samples": n_samples,
                "relax_steps": cap,
                "step_source": step_source,
                "reached": reached,
                "hit_generation": "" if hit_index is None else str(hit_index),
                "evals_to_hit": spent if evals_to_hit is None else evals_to_hit,
                "charged": spent,
                "best": best,
                "reference": reference,
            }
        )
    return {
        "n_atoms": n_atoms,
        "reference": reference,
        "runs": parsed,
    }


def read_zip(path: Path) -> list[dict]:
    digest = md5_file(path)
    if digest != ZENODO_MD5:
        raise SystemExit(f"{path} md5 {digest} is not {ZENODO_MD5}")
    found = []
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        for n_atoms, member in MEMBERS.items():
            if member not in names:
                raise SystemExit(f"{path} has no {member}")
            archive.extract(member, path=path.parent / "zenodo")
            with h5py.File(path.parent / "zenodo" / member, "r") as handle:
                found.append(read_run(handle, n_atoms))
    return found


def _synthetic(path: Path) -> None:
    with h5py.File(path, "w") as handle:
        run = handle.create_group("run_0")
        run.attrs["es_config"] = '{"relaxer": "AtomsRelaxer(steps=4, fmax=0.001)"}'
        run.attrs["objective"] = json.dumps({"maximize": True, "dim": 6})
        run.create_group("gen_0").create_dataset("fitness", data=np.array([1.0, 3.0]))
        run.create_group("gen_1").create_dataset("fitness", data=np.array([9.0, 12.0]))
        run.create_group("gen_2").create_dataset("fitness", data=np.array([12.0, 4.0]))
        varied = handle.create_group("run_1")
        varied.attrs["es_config"] = '{"relaxer": "AtomsRelaxer(steps=9)"}'
        varied.attrs["objective"] = json.dumps({"maximize": False})
        early = varied.create_group("gen_0")
        early.create_dataset("fitness", data=np.array([-1.0, -2.0]))
        early.create_dataset("relax_steps", data=np.array([2, 5]))


def self_test() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "synthetic.h5"
        _synthetic(path)
        with h5py.File(path, "r") as handle:
            parsed = read_run(handle, 31)
    runs = {row["run"]: row for row in parsed["runs"]}
    cap_run = runs["run_0"]
    # Generation 1 is the first whose negated fitness reaches -133.586422 + 0.001.
    # Fitness 12 negates to -12, which is not the Cambridge energy. The synthetic
    # reference is the real LJ31 value, so this run does not hit. Charged is
    # 3 generations * 2 samples * cap 4.
    if cap_run["reached"] or cap_run["charged"] != 24 or cap_run["step_source"] != "relaxer cap":
        raise SystemExit(f"cap run {cap_run}")
    if cap_run["best"] != -12.0 or cap_run["evals_to_hit"] != 24:
        raise SystemExit(f"cap energy {cap_run}")
    varied = runs["run_1"]
    # Energies are already the potential. The per-sample record sums to 7,
    # and -2 is within 0.001 of nothing near -133, so this run also misses.
    if varied["step_source"] != "relax_steps" or varied["charged"] != 7:
        raise SystemExit(f"varied run {varied}")
    # A file whose reference is the stored energy does hit on generation 0.
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "hit.h5"
        with h5py.File(path, "w") as handle:
            run = handle.create_group("run_0")
            run.attrs["es_config"] = "AtomsRelaxer(steps=10)"
            run.attrs["objective"] = json.dumps({"maximize": True})
            run.create_group("gen_0").create_dataset(
                "fitness", data=np.array([100.0, 133.5860])
            )
            run.create_group("gen_1").create_dataset(
                "fitness", data=np.array([133.586422, 1.0])
            )
        saved = REFERENCES[31]
        try:
            REFERENCES[31] = -133.586422
            with h5py.File(path, "r") as handle:
                hit = read_run(handle, 31)["runs"][0]
        finally:
            REFERENCES[31] = saved
    # 133.5860 is 0.000422 below 133.586422, inside the 0.001 band.
    if hit["hit_generation"] != "0" or hit["evals_to_hit"] != 20 or not hit["reached"]:
        raise SystemExit(f"hit run {hit}")
    print("ok")


def main(argv: list[str]) -> int:
    if len(argv) == 2 and argv[1] == "--self-test":
        self_test()
        return 0
    parser = argparse.ArgumentParser()
    parser.add_argument("zip", type=Path)
    args = parser.parse_args(argv[1:])
    for size in read_zip(args.zip):
        for run in size["runs"]:
            print(
                "{system} {run} generations={generations} samples={samples} "
                "relax_steps={relax_steps} source={step_source} "
                "reached={reached} hit_generation={hit_generation} "
                "evals_to_hit={evals_to_hit} best={best:.10f}".format(**run)
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv))
