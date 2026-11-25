#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Write the pre-registered plan for one G2 budget-stop cell.

The cap comes from the protocol module. This command does not read an
environment override and it does not start a search.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


def _protocol():
    name = "g2_budget_protocol"
    path = Path(__file__).resolve().with_name("protocol.py")
    loaded = sys.modules.get(name)
    if loaded is not None and Path(loaded.__file__).resolve() == path:
        return loaded
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


protocol = _protocol()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Emit one pre-registered G2 budget-stop cell.")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--system")
    parser.add_argument("--arm")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--stamp")
    args = parser.parse_args(argv)
    if args.list:
        if args.system or args.arm or args.seed is not None or args.stamp:
            parser.error("--list does not take a cell")
        for cell in protocol.cells():
            print(json.dumps(protocol.cell_record(cell), sort_keys=True))
        return 0
    if not args.system or not args.arm or args.seed is None or not args.stamp:
        parser.error("a cell needs --system, --arm, --seed, and --stamp")
    cell = protocol.find_cell(args.system, args.arm, args.seed)
    path = Path(args.stamp)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(protocol.cell_record(cell), indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
