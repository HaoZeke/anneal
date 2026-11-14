#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Energy evaluations in the deposited Guided Adaptive Diffusion runs.

The archive is Zenodo 10.5281/zenodo.19001618. Each Lennard-Jones file
holds one evolutionary run. A generation stores one fitness per sample.
Fitness is the negated potential when the run maximises. ``model_loss``
is the diffusion training loss and is not an energy evaluation. The
Lennard-Jones XYZ files are one frame each and do not record a step
count.

A generation dataset named ``relax_steps``, ``nsteps``, ``n_steps``, or
``relaxation_steps`` is the number of potential calls in that
relaxation. The deposited files have no such dataset. The relaxer
string records only the FIRE cap, ``steps=1000``. That cap is an upper
bound. The lower bound charges one evaluation per sample and treats
the relaxation as free. Through the first generation whose best energy
is within 0.001 of the Cambridge value,

    lower = generations * samples
    upper = generations * samples * cap

These figures are read from the deposit. They are not a rerun.
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
        spent = 0
        used = 0
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
            energy = -fitness if negate else fitness
            generation_best = float(np.min(energy))
            if best is None or generation_best < best:
                best = generation_best
            if hit_index is None:
                spent += calls
                used += 1
                if generation_best <= level:
                    hit_index = index
        if best is None or n_samples is None or step_source is None or used < 1:
            raise ValueError(f"{run_name} has no generations")
        reached = hit_index is not None
        if step_source == "relaxer cap":
            lower = used * n_samples
            upper = lower * cap
        else:
            lower = spent
            upper = spent
        parsed.append(
            {
                "system": f"lj{n_atoms}",
                "n_atoms": n_atoms,
                "run": run_name,
                "generations": len(indexes),
                "generations_charged": used,
                "samples": n_samples,
                "relax_steps": cap,
                "step_source": step_source,
                "reached": reached,
                "hit_generation": "" if hit_index is None else str(hit_index),
                "evals_lower": lower,
                "evals_upper": upper,
                "best": best,
                "reference": reference,
            }
        )
    return {
        "n_atoms": n_atoms,
        "reference": reference,
        "runs": parsed,
    }


def lj_xyz_has_step_counts(archive: zipfile.ZipFile) -> bool:
    """True when a Lennard-Jones XYZ records a relaxation length."""
    found = False
    for name in archive.namelist():
        if not name.startswith("lj-") or not name.endswith(".xyz"):
            continue
        found = True
        lines = archive.read(name).decode().splitlines()
        if len(lines) < 2:
            raise ValueError(f"{name} has no comment line")
        count = int(lines[0].split()[0])
        if len(lines) != count + 2:
            raise ValueError(f"{name} is not one frame")
        comment = lines[1]
        if any(token in comment.lower() for token in ("step", "force", "call", "iter")):
            return True
    if not found:
        raise ValueError("the archive has no Lennard-Jones XYZ")
    return False


def read_zip(path: Path) -> list[dict]:
    digest = md5_file(path)
    if digest != ZENODO_MD5:
        raise SystemExit(f"{path} md5 {digest} is not {ZENODO_MD5}")
    found = []
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        if lj_xyz_has_step_counts(archive):
            raise SystemExit(f"{path} XYZ files record relaxation steps")
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
    # Fitness 12 negates to -12, short of the Cambridge energy, so the run
    # misses. Three generations and two samples give a lower bound of 6.
    # The cap of 4 makes the upper bound 24.
    if cap_run["reached"] or cap_run["step_source"] != "relaxer cap":
        raise SystemExit(f"cap run {cap_run}")
    if cap_run["evals_lower"] != 6 or cap_run["evals_upper"] != 24 or cap_run["best"] != -12.0:
        raise SystemExit(f"cap energy {cap_run}")
    varied = runs["run_1"]
    # The per-sample record sums to 7, so the two bounds agree.
    if varied["step_source"] != "relax_steps" or varied["evals_lower"] != 7 or varied["evals_upper"] != 7:
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
    if hit["hit_generation"] != "0" or hit["evals_lower"] != 2 or hit["evals_upper"] != 20 or not hit["reached"]:
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
                "{system} {run} generations_charged={generations_charged} samples={samples} "
                "relax_steps={relax_steps} source={step_source} "
                "reached={reached} hit_generation={hit_generation} "
                "lower={evals_lower} upper={evals_upper} best={best:.10f}".format(**run)
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv))
