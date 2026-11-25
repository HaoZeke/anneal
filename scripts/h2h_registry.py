# Copyright (c) anneal developers
"""Load the immutable head-to-head registry and append-only additions."""

from __future__ import annotations

import json
from pathlib import Path


def load_registry(root: Path) -> dict:
    registry = json.loads((root / "bench/h2h.json").read_text())
    additions = root / "bench/h2h-additions.jsonl"
    if not additions.is_file():
        return registry

    h2h = {entry["rule"]: entry for entry in registry["h2h"]}
    not_h2h = dict(registry["not_h2h"])
    for number, line in enumerate(additions.read_text().splitlines(), start=1):
        operation = json.loads(line)
        kind, key, value = operation["kind"], operation["key"], operation["value"]
        if kind == "missing_outputs":
            if key not in h2h or not isinstance(value, dict):
                raise ValueError(f"{additions}:{number}: invalid missing_outputs update for {key!r}")
            h2h[key] = {**h2h[key], "missing_outputs": value}
            continue
        if key in h2h or key in not_h2h:
            raise ValueError(f"{additions}:{number}: {key} is already classified")
        if kind == "h2h":
            h2h[key] = value
        elif kind == "not_h2h":
            not_h2h[key] = value
        else:
            raise ValueError(f"{additions}:{number}: unknown operation {kind!r}")
    return {"h2h": list(h2h.values()), "not_h2h": not_h2h}
