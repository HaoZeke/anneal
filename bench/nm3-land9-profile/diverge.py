#!/usr/bin/env python3
# Copyright (c) anneal developers
"""First step where two traces disagree, and why."""

from __future__ import annotations

import argparse
import csv


def load(path: str) -> list[dict[str, str]]:
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def classify(prev: dict[str, str] | None, row_a: dict[str, str], row_b: dict[str, str]) -> str:
    move_diff = row_a["move"] != row_b["move"]
    draws_iter_diff = row_a["draws_iter"] != row_b["draws_iter"]
    begin_delta_a = int(row_a["draws_begin"]) - int(row_a["draws_iter"])
    begin_delta_b = int(row_b["draws_begin"]) - int(row_b["draws_iter"])
    body_a = int(row_a["draws_end"]) - int(row_a["draws_begin"])
    body_b = int(row_b["draws_end"]) - int(row_b["draws_begin"])
    proposal_diff = row_a["proposal_bits"] != row_b["proposal_bits"]
    accept_diff = row_a["accepted"] != row_b["accepted"]
    charged_diff = row_a["charged"] != row_b["charged"]
    if draws_iter_diff:
        if prev is None:
            return "rng before the first step"
        return "rng between steps"
    if move_diff and begin_delta_a == begin_delta_b:
        return "move choice"
    if move_diff:
        return "move choice after a different draw count"
    if proposal_diff and body_a != body_b:
        return "rng inside the move"
    if proposal_diff and charged_diff:
        return "potential calls inside the move"
    if proposal_diff:
        return "floating point"
    if accept_diff:
        return "acceptance"
    if charged_diff:
        return "potential calls"
    if row_a["iter_energy_bits"] != row_b["iter_energy_bits"]:
        return "chain energy"
    return "other"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    left = load(args.a)
    right = load(args.b)
    limit = min(len(left), len(right))
    found = None
    prev = None
    for index in range(limit):
        row_a = left[index]
        row_b = right[index]
        keys = (
            "move",
            "iter_energy_bits",
            "proposal_bits",
            "draws_iter",
            "draws_begin",
            "draws_end",
            "accepted",
            "charged",
        )
        if any(row_a[key] != row_b[key] for key in keys):
            found = (index, row_a, row_b, classify(prev, row_a, row_b))
            break
        prev = row_a
    new_file = True
    try:
        with open(args.out, newline="") as handle:
            new_file = handle.read(1) == ""
    except FileNotFoundError:
        new_file = True
    with open(args.out, "a", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "seed",
                "step",
                "cause",
                "move_a",
                "move_b",
                "accepted_a",
                "accepted_b",
                "proposal_a",
                "proposal_b",
                "draws_iter_a",
                "draws_iter_b",
                "draws_begin_a",
                "draws_begin_b",
                "draws_end_a",
                "draws_end_b",
                "charged_a",
                "charged_b",
                "steps_a",
                "steps_b",
            ],
            lineterminator="\n",
        )
        if new_file:
            writer.writeheader()
        if found is None:
            cause = "none"
            if len(left) != len(right):
                cause = "length"
            writer.writerow(
                {
                    "seed": args.seed,
                    "step": "",
                    "cause": cause,
                    "move_a": "",
                    "move_b": "",
                    "accepted_a": "",
                    "accepted_b": "",
                    "proposal_a": "",
                    "proposal_b": "",
                    "draws_iter_a": "",
                    "draws_iter_b": "",
                    "draws_begin_a": "",
                    "draws_begin_b": "",
                    "draws_end_a": "",
                    "draws_end_b": "",
                    "charged_a": "",
                    "charged_b": "",
                    "steps_a": len(left),
                    "steps_b": len(right),
                }
            )
            return
        index, row_a, row_b, cause = found
        writer.writerow(
            {
                "seed": args.seed,
                "step": index,
                "cause": cause,
                "move_a": row_a["move"],
                "move_b": row_b["move"],
                "accepted_a": row_a["accepted"],
                "accepted_b": row_b["accepted"],
                "proposal_a": row_a["proposal_bits"],
                "proposal_b": row_b["proposal_bits"],
                "draws_iter_a": row_a["draws_iter"],
                "draws_iter_b": row_b["draws_iter"],
                "draws_begin_a": row_a["draws_begin"],
                "draws_begin_b": row_b["draws_begin"],
                "draws_end_a": row_a["draws_end"],
                "draws_end_b": row_b["draws_end"],
                "charged_a": row_a["charged"],
                "charged_b": row_b["charged"],
                "steps_a": len(left),
                "steps_b": len(right),
            }
        )


if __name__ == "__main__":
    main()
