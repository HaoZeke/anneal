#!/usr/bin/env python3
"""Validate a surface-evidence pair directory and report ensemble outcomes.

Usage: python3 scripts/surface_evidence_report.py RESULT_DIRECTORY
The directory contains manifest.txt and one private/shared log per worker.
Incomplete or confounded runs raise an error instead of supplying a score.
"""

import argparse
import json
import math
import re
from pathlib import Path


SUMMARY = re.compile(
    r"^  seed (\d+): best (\S+).*?\bhops (\d+).*?\bcharged (\d+).*$", re.M
)
SURFACE = re.compile(
    r"^SURFACE_EVIDENCE seed (\d+) local_blocks (\d+) peer_blocks (\d+) "
    r"local_draws (\[[^\]]*\]) local_means (\[[^\]]*\])$", re.M
)
POLICY = re.compile(
    r"^  policy: leaves (\d+) other (\d+) walk (\d+) hole (\d+) refused (\d+)$", re.M
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def worker(path, seed, budget, n, mode):
    text = path.read_text()
    rows = SUMMARY.findall(text)
    surfaces = SURFACE.findall(text)
    policies = POLICY.findall(text)
    require(len(rows) == len(surfaces) == len(policies) == 1, f"{path}: incomplete worker")
    require("catalog channels: surface evidence only; geometry policy disabled" in text,
            f"{path}: evidence-only execution is absent")
    require(all(int(value) == 0 for value in policies[0]), f"{path}: geometry policy executed")
    found_seed, best, hops, charged = rows[0]
    best = float(best)
    require(int(found_seed) == seed, f"{path}: seed mismatch")
    require(int(charged) == budget, f"{path}: unmatched objective budget")
    require(math.isfinite(best) and best >= -n * (n - 1) / 2 - 1e-5,
            f"{path}: invalid Lennard-Jones objective value")
    require(f"LJ{n}, budget {budget} charged evaluations, 1 seeds," in text,
            f"{path}: system or input budget mismatch")
    surface_seed, blocks, peers, draws, means = surfaces[0]
    draws, means = json.loads(draws), json.loads(means)
    require(int(surface_seed) == seed, f"{path}: surface seed mismatch")
    require(len(draws) == len(means) > 0, f"{path}: incompatible reward arrays")
    require(all(type(value) is int and value >= 0 for value in draws), f"{path}: invalid draws")
    require(sum(draws) == int(blocks), f"{path}: local block count mismatch")
    require(all(type(value) in (int, float) and math.isfinite(value) for value in means),
            f"{path}: nonfinite surface rewards")
    peers = int(peers)
    require(mode != "private" or peers == 0, f"{path}: private arm imported peer rewards")
    charges = []
    for line in text.splitlines():
        if line.startswith("{"):
            event = json.loads(line)
            charge = event.get("aggregate_charged")
            if charge is not None:
                require(type(charge) is int and 0 <= charge <= budget,
                        f"{path}: invalid ledger charge")
                charges.append(charge)
    require(charges and charges == sorted(charges) and charges[-1] == budget,
            f"{path}: incomplete or nonmonotonic cooperative ledger")
    solved = " SOLVED" in SUMMARY.search(text).group(0)
    return {"seed": seed, "best": best, "hops": int(hops), "solved": solved,
            "local_blocks": int(blocks), "peer_blocks": peers}


def summarize(root):
    root = Path(root)
    manifest = {}
    for line in (root / "manifest.txt").read_text().splitlines():
        key, separator, value = line.partition("=")
        if separator:
            manifest[key] = value
    require(manifest.get("channels") == "surface-evidence-only", "wrong communication channels")
    parameters = {key: int(manifest[key]) for key in ("n", "budget", "replicas", "ensembles", "block")}
    require(all(value > 0 for value in parameters.values()), "positive parameters required")
    budget, replicas, ensembles = (parameters[key] for key in ("budget", "replicas", "ensembles"))
    best, hits, workers = {}, {}, {}
    for mode in ("private", "shared"):
        best[mode], hits[mode], workers[mode] = [], [], []
        for ensemble in range(ensembles):
            records = [worker(root / f"{mode}-{ensemble}-{replica}.log",
                              ensemble * replicas + replica, budget, parameters["n"], mode)
                       for replica in range(replicas)]
            if mode == "shared":
                require(sum(row["peer_blocks"] for row in records) > 0,
                        f"shared ensemble {ensemble}: no peer evidence received")
            best[mode].append(min(row["best"] for row in records))
            hits[mode].append(any(row["solved"] for row in records))
            workers[mode].extend(records)
    return {"source": manifest["source"], "features": manifest["features"], **parameters,
            "ensemble_budget": budget * replicas, "ensemble_best": best,
            "ensemble_hits": {mode: sum(values) for mode, values in hits.items()},
            "shared_peer_blocks": sum(row["peer_blocks"] for row in workers["shared"]),
            "workers": workers}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_directory", type=Path)
    args = parser.parse_args()
    try:
        result = summarize(args.result_directory)
    except (ValueError, KeyError, OSError) as error:
        parser.exit(1, f"invalid comparison: {error}\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
