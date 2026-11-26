"""Recompute the comparison from inputs.csv and check results.json."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_stats():
    spec = importlib.util.spec_from_file_location("poi_vs_crest_stats", HERE / "stats.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_results_match_inputs():
    stats = load_stats()
    rows = stats.read_inputs(HERE / "inputs.csv")
    fresh = stats.summarize(rows)
    stored = json.loads((HERE / "results.json").read_text())
    assert fresh == stored

    anneal = [row for row in rows if row["source"] == "anneal"]
    crest = [row for row in rows if row["source"] == "crest"]
    assert [int(row["seed"]) for row in anneal] == list(range(2000, 2012))
    assert fresh["anneal"]["calls"] == [int(row["calls_to_hit"]) for row in anneal]
    assert fresh["crest"]["calls"] == [int(row["first_hit_call"]) for row in crest]
    seed_2000 = next(row for row in anneal if row["seed"] == "2000")
    assert int(seed_2000["calls_to_hit"]) == 5874
    assert int(seed_2000["charged_calls"]) == 7738
    assert 5874 != 7738
    assert [int(row["seed"]) for row in crest] == list(range(17000, 17012))
    assert fresh["anneal"]["n"] == 12
    assert fresh["anneal"]["hits"] == 12
    assert fresh["crest"]["n"] == 12
    assert fresh["crest"]["hits"] == 7
    assert fresh["crest"]["seeds_absent"] == []
    assert fresh["crest"]["source_commit"] == "7ad8eab67d61f1d5835a9ce7b67e7b2cb2b4ede9"
    assert fresh["crest"]["commit_requested"] == fresh["crest"]["source_commit"]
    for row in crest:
        if row["hit"] == "0":
            assert int(row["first_hit_call"]) == 151739
