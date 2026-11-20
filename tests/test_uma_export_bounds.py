#!/usr/bin/env python3
# Copyright (c) rgpot developers
"""Every UMA export path records the traced edge count, or refuses."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    import torch  # noqa: F401
except ImportError:
    if "pytest" in sys.modules:
        import pytest

        pytest.skip("torch is absent", allow_module_level=True)
    raise SystemExit("torch is required")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import export_uma_aoti as exp  # noqa: E402


def expect_exact(path, traced, box, molecular):
    record = exp.edge_export_record(path, traced, box)
    if record["nedges_min"] != traced or record["nedges_max"] != traced:
        raise SystemExit(f"{path}: bounds are not the traced count {record}")
    if record["molecular"] is not molecular:
        raise SystemExit(f"{path}: molecular flag is {record['molecular']}")
    if float(record["molecular_box"]) != float(box or 0.0):
        raise SystemExit(f"{path}: box is {record['molecular_box']}")
    if record["nedges_min"] == 1 and record["nedges_max"] == 65536:
        raise SystemExit(f"{path}: still records 1..65536")
    print(f"ok {path} box={box} nedges={traced}")


def expect_refusal(path, traced, box, alt=None):
    try:
        record = exp.edge_export_record(path, traced, box, alt)
    except exp.ExportRefusal as exc:
        print(f"ok {path} refused: {exc}")
        return
    raise SystemExit(f"{path}: exported {record}")


def main() -> int:
    expect_refusal("symbolic-nedges", 56, 0.0)
    expect_refusal("dynamic-nonstrict", 56, 0.0)
    expect_refusal("make_fx-nonstrict", 56, 0.0)
    expect_refusal("symbolic-nedges", 56, 40.0, alt=42)
    expect_exact("symbolic-nedges", 56, 40.0, True)
    expect_exact("dynamic-nonstrict", 56, 40.0, True)
    expect_exact("static-nonstrict", 56, 0.0, False)
    expect_exact("static-strict", 56, 0.0, False)
    expect_exact("static-nonstrict", 12, 25.0, True)
    expect_exact("static-strict", 12, 25.0, True)
    expect_exact("make_fx-nonstrict", 56, 40.0, True)
    print("export paths: 11")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
