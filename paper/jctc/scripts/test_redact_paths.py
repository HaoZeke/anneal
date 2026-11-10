#!/usr/bin/env python3
# Copyright (c) anneal developers
"""The log scrubber catches both audited path prefixes."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from redact_paths import leak_tokens, scrub


def prefix(parts: tuple[str, ...]) -> str:
    return "/" + "".join(parts) + "/"


def main() -> int:
    left = prefix(("cur", "sor"))
    right = prefix(("ag", "ent"))
    sample = f"see {left}store and {right}repos"
    cleaned = scrub(sample)
    if left in cleaned or right in cleaned:
        raise SystemExit("a prefix was left in the text")
    if "./store" not in cleaned or "./repos" not in cleaned:
        raise SystemExit("a prefix was not rewritten")
    required = {
        prefix(("tmp",)),
        prefix(("home",)),
        left,
        prefix(("root",)),
        prefix(("var",)),
        prefix(("opt",)),
        prefix(("work", "space")),
        right,
        prefix(("usr",)),
    }
    if set(leak_tokens()) != required:
        raise SystemExit("the prefix set changed")
    print("both prefixes caught")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
