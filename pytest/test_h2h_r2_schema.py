# Copyright (c) anneal developers
"""The fresh output of the TIP4P rerun has the columns of the committed artifacts."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_r2():
    spec = importlib.util.spec_from_file_location("run_h2h_r2_schema", ROOT / "bench/tip4p-h2h/run_h2h_r2.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def csv_header(name: str) -> list[str]:
    with (ROOT / "bench/tip4p-h2h" / name).open(newline="") as handle:
        return next(csv.reader(handle))


def sample_pair(r2):
    common = dict(
        molecules=8, seed=2000, start_id="085c05455b80ec67", hit=1, calls=15599, wall_s=0.058794,
        best_energy=-305.5173712255, core=0, affinity_source="taskset -c 0", affinity_seen="0",
        run_order_index=1, arm_order="anneal,gmin", start_ts="2026-11-18T22:00:00Z",
        end_ts="2026-11-18T22:00:01Z",
    )
    return [r2.ArmRun(arm="anneal", **common), r2.ArmRun(arm="gmin", **{**common, "run_order_index": 2})]


MACHINE = {"machine_id": "a" * 64, "cpu_model": "Example CPU"}


def test_the_fresh_output_has_the_columns_of_the_committed_artifacts():
    r2 = load_r2()
    pair = sample_pair(r2)
    row = r2.row_record(pair[0], MACHINE, "x86_64", 8, "a" * 40, "b" * 40)
    assert list(row) == r2.ROW_FIELDS == csv_header("rows-h2h-r2.csv")
    host = r2.host_record(pair[0], MACHINE)
    assert list(host) == r2.HOST_FIELDS == csv_header("hosts-r2.csv")
    payload = r2.session_payload(pair, MACHINE, "x86_64", 8, "taskset -c 0", 0.001, "a" * 40, "b" * 40)
    committed = json.loads((ROOT / "bench/tip4p-h2h/sessions-r2.jsonl").read_text().splitlines()[0])
    assert list(payload) == list(committed)
    assert list(payload["arms"][0]) == list(committed["arms"][0])
    for record in (row, host, payload):
        assert "hostname" not in record and "host" not in record


def test_the_tip4p_rerun_does_not_read_or_write_a_hostname():
    source = (ROOT / "bench/tip4p-h2h/run_h2h_r2.py").read_text()
    assert "hostname" not in source and "socket" not in source
