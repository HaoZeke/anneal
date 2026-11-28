#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Recompute ABBA call columns from the logs that are in this tree.

Anneal ``first_hit_call`` is ``calls_to_first_hit`` in ``rows-A.csv`` or
``rows-B.csv`` for that seed. It is not the campaign ``SEED`` line.
CREST ``first_hit_call`` is the first counted xtb evaluation at or below
``-40.681336013150 + 1e-6`` Eh. If the log has no energy for every counted
evaluation, the cell is ``NA``. A scored miss is 151739.
"""

from __future__ import annotations

import csv
import importlib.util
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOGS = HERE / "logs"
_TARGET_TOL = HERE.parents[1] / "scripts" / "target_tol.py"
_spec = importlib.util.spec_from_file_location("target_tol", _TARGET_TOL)
_target_tol = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_target_tol)
energy_hits = _target_tol.energy_hits
REFERENCE_EH = -40.681336013150
TOL_EH = 1e-6
THRESHOLD_EH = REFERENCE_EH + TOL_EH
BUDGET = 151738
MISS = BUDGET + 1
TAG_2000 = "da5ae9b9-9259-4a28-95e3-53ff0ab39756"
TAG_2006 = "vm0"
CREST_LINE = re.compile(
    r"^seed (\d+) first_hit_call=\d+ total_calls=(\d+) .* wall_s=(\S+) .* exit=(\d+)$"
)
ANNEAL_LINE = re.compile(
    r"^seed (\d+) repeat (\d+) energy_calls (\d+) exit (\d+)$"
)
ORDER_TO_REPEAT = {"1": ("anneal", "1"), "2": ("crest", "1"), "3": ("crest", "2"), "4": ("anneal", "2")}
USER_HOST = re.compile(r"[A-Za-z][\w.+-]*@[A-Za-z0-9][\w.-]*")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def shard_first_hit() -> dict[str, str]:
    found: dict[str, str] = {}
    for path in (HERE / "rows-A.csv", HERE / "rows-B.csv"):
        for row in read_csv(path):
            found[row["seed"]] = row["calls_to_first_hit"]
    return found


def anneal_calls(text: str) -> dict[tuple[str, str], tuple[str, str]]:
    found: dict[tuple[str, str], tuple[str, str]] = {}
    for line in text.splitlines():
        match = ANNEAL_LINE.match(line)
        if match:
            found[(match.group(1), match.group(2))] = (match.group(3), match.group(4))
    return found


def crest_calls(text: str) -> dict[tuple[str, str], tuple[str, str]]:
    found: dict[tuple[str, str], tuple[str, str]] = {}
    for line in text.splitlines():
        match = CREST_LINE.match(line)
        if match:
            found[(match.group(1), match.group(3))] = (match.group(2), match.group(4))
    return found


def first_hit_from_events(events: list[dict]) -> str:
    """Score a complete per-evaluation energy series. Gaps return NA."""
    offset = 0
    for event in events:
        kind = event.get("kind")
        count = int(event.get("n") or 0)
        if kind not in {"opt", "md"}:
            continue
        energies = event.get("energies")
        if not isinstance(energies, list) or len(energies) < count:
            return "NA"
        for index, energy in enumerate(energies[:count], start=1):
            if isinstance(energy, (int, float)) and energy_hits(float(energy), REFERENCE_EH, TOL_EH):
                return str(offset + index)
        offset += count
    if offset == 0:
        return "NA"
    return str(MISS)


def first_hit_from_crest_out(text: str) -> str:
    """CREST stdout does not list one energy per counted xtb evaluation."""
    del text
    return "NA"


def expected_rows() -> list[dict[str, str]]:
    recorded = read_csv(HERE / "rows-abba.csv")
    anneal_hit = shard_first_hit()
    anneal_log = anneal_calls((LOGS / "anneal-abba.txt").read_text())
    crest_log = crest_calls((LOGS / "campaign.txt").read_text())
    expect: list[dict[str, str]] = []
    for row in recorded:
        seed = row["seed"]
        repeat = row["repeat"]
        if row["arm"] == "anneal":
            calls = anneal_log.get((seed, repeat))
            total, code = calls if calls else ("NA", "NA")
            first = anneal_hit[seed]
        else:
            calls = crest_log.get((seed, row["wall_s"]))
            total, code = calls if calls else ("NA", "NA")
            out = LOGS / "crest" / f"{seed}-r{repeat}.txt"
            if out.is_file():
                first = first_hit_from_crest_out(out.read_text(errors="replace"))
            elif int(seed) <= 2005:
                raise SystemExit(f"missing CREST log {out.name}")
            else:
                first = "NA"
        expect.append(
            {
                "arm": row["arm"],
                "seed": seed,
                "repeat": repeat,
                "total_calls": total,
                "first_hit_call": first,
                "exit_code": code,
                "host_tag": TAG_2000 if int(seed) <= 2005 else TAG_2006,
            }
        )
    return expect


def same_measurement(left: dict[str, str], right: dict[str, str]) -> bool:
    return (
        left["arm"] == right["arm"]
        and left["seed"] == right["seed"]
        and left["wall_s"] == right["wall_s"]
        and left["cpu_s"] == right["cpu_s"]
        and left["best_Eh"] == right["best_Eh"]
    )


def assert_sources(rows: list[dict[str, str]]) -> None:
    early = {(row["seed"], row["arm"], row["repeat"]): row for row in rows if int(row["seed"]) <= 2005}
    for source in read_csv(LOGS / "source-2000-2005.csv"):
        arm, repeat = ORDER_TO_REPEAT[source["order"]]
        if source["arm"] != arm:
            raise SystemExit(f"source order {source['order']} arm {source['arm']}")
        got = early[(source["seed"], arm, repeat)]
        if not same_measurement(got, source):
            raise SystemExit(f"2000-2005 measurement changed {source['seed']} {arm} {repeat}")
    later = {(row["seed"], row["arm"], row["repeat"]): row for row in rows if int(row["seed"]) >= 2006}
    for source in read_csv(LOGS / "source-2006-2011.csv"):
        got = later[(source["seed"], source["arm"], source["repeat"])]
        if not same_measurement(got, source):
            raise SystemExit(f"2006-2011 measurement changed {source['seed']} {source['arm']} {source['repeat']}")


def assert_rows() -> None:
    self_check()
    rows = read_csv(HERE / "rows-abba.csv")
    expect = expected_rows()
    if len(rows) != len(expect):
        raise SystemExit(f"rows {len(rows)} expect {len(expect)}")
    fields = ("arm", "seed", "repeat", "total_calls", "first_hit_call", "exit_code", "host_tag")
    for index, (row, want) in enumerate(zip(rows, expect), start=1):
        for field in fields:
            if row[field] != want[field]:
                raise SystemExit(f"row {index} {field} csv {row[field]} log {want[field]}")
    assert_sources(rows)
    assert_logs_have_no_user_host()
    print(f"log check ok {len(rows)} rows")


def assert_logs_have_no_user_host() -> None:
    for path in sorted(LOGS.rglob("*")):
        if not path.is_file():
            continue
        if USER_HOST.search(path.read_text(errors="replace")):
            raise SystemExit(f"user@host token in {path.relative_to(HERE)}")


def self_check() -> None:
    local = "usr"
    probe = "compiled by '" + local + "@" + "host'"
    if not USER_HOST.search(probe):
        raise SystemExit("user-host detector")
    if USER_HOST.search("compiled by 'HOST'"):
        raise SystemExit("HOST is not a user-host token")
    hit = first_hit_from_events([{"kind": "opt", "n": 2, "energies": [-1.0, THRESHOLD_EH]}])
    if hit != "2":
        raise SystemExit(f"complete series {hit}")
    gap = first_hit_from_events(
        [{"kind": "md", "n": 3}, {"kind": "opt", "n": 1, "energies": [THRESHOLD_EH]}]
    )
    if gap != "NA":
        raise SystemExit(f"missing md energies {gap}")
    miss = first_hit_from_events([{"kind": "opt", "n": 1, "energies": [-1.0]}])
    if miss != str(MISS):
        raise SystemExit(f"miss {miss}")
    if first_hit_from_crest_out("no per-evaluation energies\n") != "NA":
        raise SystemExit("crest stdout")


def main() -> None:
    assert_rows()


if __name__ == "__main__":
    sys.exit(main() or 0)
