#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Exit non-zero when the repository contains raw target-energy hit checks."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from target_tol_lint_rules import scan_repository  # noqa: E402


def main() -> int:
    violations = scan_repository(ROOT)
    if violations:
        print("raw target-energy hit comparisons must use energy_hits:", file=sys.stderr)
        print("\n".join(violations), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())