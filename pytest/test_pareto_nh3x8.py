# Copyright (c) anneal developers
"""Merged NH3 octamer Pareto rows and the summary script."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "bench" / "pareto-nh3x8"
SHARDS = HERE / "shards"
MERGED = HERE / "rows.csv"
STATS = HERE / "stats.json"
HOST_COLUMNS = ("host_id", "host_id_sha256", "host_id_hash")


def _shard_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for letter in "ABCD":
        with (SHARDS / f"rows-{letter}.csv").open(newline="") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def _host_digests(rows: list[dict[str, str]]) -> list[str]:
    found: list[str] = []
    for row in rows:
        for column in HOST_COLUMNS:
            value = row.get(column, "").strip()
            if value:
                found.append(value)
    return found


def test_merged_row_count_equals_shard_rows():
    shards = _shard_rows()
    with MERGED.open(newline="") as handle:
        merged = list(csv.DictReader(handle))
    assert len(merged) == len(shards)
    assert len(merged) == sum(
        sum(1 for _ in (SHARDS / f"rows-{letter}.csv").open(newline="")) - 1
        for letter in "ABCD"
    )


def test_merged_rows_have_no_machine_digest():
    digests = _host_digests(_shard_rows())
    assert len(digests) == 24
    text = MERGED.read_text()
    with MERGED.open(newline="") as handle:
        merged = list(csv.DictReader(handle))
    for row in merged:
        assert row["machine_id"] == "unrecorded"
        for value in row.values():
            assert not (len(value) in (16, 64) and all(c in "0123456789abcdef" for c in value))
    for digest in digests:
        assert digest not in text
    assert text.count("unrecorded") == len(merged)


def test_stats_script_is_deterministic(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    for dest in (first, second):
        subprocess.run(
            [sys.executable, str(HERE / "stats.py"), "--out", str(dest)],
            check=True,
            cwd=ROOT,
        )
    assert first.read_bytes() == second.read_bytes()
    assert first.read_bytes() == STATS.read_bytes()


def test_crest_build_split_keeps_every_row():
    payload = json.loads(STATS.read_text())
    builds = payload["crest_builds"]["by_build"]
    assert list(builds) == ["6914a25", "af7eb99"]
    seeds: list[int] = []
    hit_sum = {"anneal": 0, "crest": 0}
    row_sum = 0
    pooled = payload["crest_builds"]["pooled"]
    for block in builds.values():
        seeds.extend(block["seeds"])
        row_sum += block["n"]["rows"]
        for arm in ("anneal", "crest"):
            hit_sum[arm] += block["hits"][arm]
        same = (
            block["crest_hits_exceed_anneal_hits"] == pooled["crest_hits_exceed_anneal_hits"]
            and block["call_ratio_below_one"] == pooled["call_ratio_below_one"]
        )
        assert block["pooled_result_holds"] is same
    assert seeds == list(range(2000, 2012))
    assert row_sum == payload["n"]["rows"] == 24
    assert hit_sum == payload["hits"]
    assert builds["6914a25"]["pooled_result_holds"] is False
    assert builds["af7eb99"]["pooled_result_holds"] is True
