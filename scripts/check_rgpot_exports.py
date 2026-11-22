#!/usr/bin/env python3
"""Pin how many rgpot_* symbols librgpot exports.

A normal link of librgpot_core.a keeps only members the umbrella
references. It references none of the archive's C ABI, so the shared
library exports 49 rgpot_* symbols, all from the C++ side.

With the fast pair list on, the archive is librgpot_bundle.a and the
umbrella links it with --whole-archive (or -force_load on Darwin). That
archive defines 46 rgpot_* symbols: the public C ABI in
scripts/abi_core_baseline.txt plus the four fast-list entry points in
scripts/abi_baseline_fast.txt. None of those names are internal, so the
shared library exports them too, 95 in all.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Measured with `nm -D --defined-only` on the default umbrella
# (thread_limit, and the same pots with the fast pair list off).
CORE_ONLY_EXPORTS = 49
BUNDLE_EXPORTS = 95


def rgpot_names(nm_text: str) -> set[str]:
    """Defined rgpot_* names from an `nm --defined-only` listing."""
    names: set[str] = set()
    for line in nm_text.splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        kind, name = parts[-2], parts[-1]
        if kind == "U" or not name.startswith("rgpot_"):
            continue
        names.add(name)
    return names


def read_baseline(path: Path) -> set[str]:
    names = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            names.add(line)
    return names


def defined_names(lib: Path) -> set[str]:
    if lib.name.endswith((".a", ".lib")):
        cmd = ["nm", "-g", "--defined-only", str(lib)]
    else:
        cmd = ["nm", "-D", "--defined-only", str(lib)]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"nm failed on {lib}")
    return rgpot_names(proc.stdout)


def evaluate(
    exported: set[str],
    archived: set[str],
    wanted: set[str],
    whole: bool,
) -> None:
    if archived != wanted:
        missing = ", ".join(sorted(wanted - archived)) or "-"
        extra = ", ".join(sorted(archived - wanted)) or "-"
        raise SystemExit(
            "rust archive rgpot_* symbols do not match the baseline "
            f"(missing: {missing}; extra: {extra})"
        )
    if whole:
        dropped = sorted(wanted - exported)
        if dropped:
            raise SystemExit(
                "whole-archive link dropped public rgpot_* exports: "
                + ", ".join(dropped)
            )
    else:
        leaked = sorted(wanted & exported)
        if leaked:
            raise SystemExit(
                "core-only link exported rust archive symbols: "
                + ", ".join(leaked)
            )
    expected = BUNDLE_EXPORTS if whole else CORE_ONLY_EXPORTS
    if len(exported) != expected:
        raise SystemExit(
            f"librgpot exports {len(exported)} rgpot_* symbols, "
            f"expected {expected}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("shared", type=Path)
    parser.add_argument("archive", type=Path)
    parser.add_argument("baselines", type=Path, nargs="+")
    parser.add_argument("--whole-archive", action="store_true")
    args = parser.parse_args()
    wanted: set[str] = set()
    for path in args.baselines:
        wanted.update(read_baseline(path))
    evaluate(
        defined_names(args.shared),
        defined_names(args.archive),
        wanted,
        args.whole_archive,
    )


if __name__ == "__main__":
    main()
