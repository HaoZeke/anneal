#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Add gate-only observed-basis counters to a source checkout."""

from __future__ import annotations

import argparse
from pathlib import Path


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise SystemExit(f"instrumentation anchor count is {text.count(old)}")
    return text.replace(old, new, 1)


def instrument(root: Path, candidate: bool) -> None:
    path = root / "src/methods/cluster_hopping.rs"
    text = path.read_text()
    if "ANNEAL_GATE_OBSERVED_BASIS_COUNTER" in text:
        raise SystemExit("source is already instrumented")

    if candidate:
        text = replace_once(
            text,
            "    let mut observed_basis_spent = 0usize;\n",
            "    let mut observed_basis_spent = 0usize;\n"
            "    let mut gate_observed_basis_offered = 0usize;\n"
            "    let mut gate_observed_basis_hits = 0usize;\n",
        )
        anchor = (
            "            {\n"
            "                // The slice is the room left on the walk, not its original\n"
        )
        text = replace_once(
            text,
            anchor,
            "            {\n"
            "                gate_observed_basis_offered += 1;\n"
            "                // The slice is the room left on the walk, not its original\n",
        )
        anchor = (
            "                } else if ledger.below_target(basis_energy) {\n"
            "                    // Recording settles the quench that already ran. It is\n"
        )
        text = replace_once(
            text,
            anchor,
            "                } else if ledger.below_target(basis_energy) {\n"
            "                    gate_observed_basis_hits += 1;\n"
            "                    // Recording settles the quench that already ran. It is\n",
        )
        values = (
            "gate_observed_basis_offered, gate_observed_basis_hits, "
            "observed_basis_spent"
        )
    else:
        values = "0usize, 0usize, 0usize"

    marker = "    Ok(Outcome {\n"
    index = text.rfind(marker)
    if index < 0:
        raise SystemExit("Outcome return anchor is missing")
    output = (
        "    if let Ok(path) = "
        "std::env::var(\"ANNEAL_GATE_OBSERVED_BASIS_COUNTER\") {\n"
        f"        let (offered, hits, calls) = ({values});\n"
        "        let text = format!(\n"
        "            \"offered={offered}\\nhit={hits}\\ncalls={calls}\\n\"\n"
        "        );\n"
        "        let _ = std::fs::write(path, text);\n"
        "    }\n\n"
    )
    text = text[:index] + output + text[index:]
    path.write_text(text, newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--candidate", action="store_true")
    args = parser.parse_args()
    instrument(args.root, args.candidate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
