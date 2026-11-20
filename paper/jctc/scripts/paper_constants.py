#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Generate TeX constants from the library and its locked dependencies."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MACROS = ROOT / "paper/jctc/macros.tex"
START = "% generated paper constants: start"
END = "% generated paper constants: end"


def values() -> dict[str, str]:
    tolerance = (ROOT / "scripts/target_tol.py").read_text()
    tol_match = re.search(r"^DEFAULT_HIT_TOL: float = ([0-9.e+-]+)$", tolerance, re.M)
    if not tol_match:
        raise SystemExit("DEFAULT_HIT_TOL is missing")
    hit_tolerance = float(tol_match.group(1))

    molecular = (ROOT / "src/catalog/molecular.rs").read_text()
    revision_match = re.search(
        r'pub const RGPOT_XTB_ENGINE_REV: &str = "([0-9a-f]{40})";',
        molecular,
    )
    if not revision_match:
        raise SystemExit("RGPOT_XTB_ENGINE_REV is missing")
    revision = revision_match.group(1)
    lock = (ROOT / "Cargo.lock").read_text()
    locked = re.search(
        r'name = "rgpot-pair".*?source = "git\+[^"]+#([0-9a-f]{40})"',
        lock,
        re.S,
    )
    if not locked or locked.group(1) != revision:
        raise SystemExit("RGPOT_XTB_ENGINE_REV does not match the Cargo.lock pin")

    return {
        "TIPFourPHitToleranceKJ": f"{hit_tolerance:g}",
        "RgpotRevision": revision,
    }


def block() -> str:
    generated = values()
    return "\n".join(
        [
            START,
            rf"\newcommand{{\TIPFourPHitToleranceKJ}}{{{generated['TIPFourPHitToleranceKJ']}}}",
            rf"\newcommand{{\RgpotRevision}}{{{generated['RgpotRevision']}}}",
            END,
        ]
    )


def rendered(text: str) -> str:
    replacement = block()
    if START not in text or END not in text:
        return text.rstrip() + "\n\n" + replacement + "\n"
    return re.sub(
        re.escape(START) + r".*?" + re.escape(END),
        lambda _: replacement,
        text,
        flags=re.S,
    )


def stale() -> bool:
    return MACROS.read_text() != rendered(MACROS.read_text())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    current = MACROS.read_text()
    updated = rendered(current)
    if args.check:
        return int(current != updated)
    MACROS.write_text(updated)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
