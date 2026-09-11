"""Verify direct peer-height diagnostics from a matched box comparison file."""

import argparse
import json
import math
from pathlib import Path


COUNTS = (
    "comparisons",
    "unresolved",
    "accepted",
    "peer_overlap",
    "peer_delta_changes",
    "probability_changes",
    "drawn_comparisons",
    "drawn_disagreements",
)
MEASURES = (
    "probability_change_sum",
    "max_probability_change",
    "max_abs_peer_delta",
    "max_abs_peer_delta_over_temperature",
)


def records(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line]
    assert rows[0]["record"] == "configuration"
    config, results = rows[0], rows[1:]
    assert config["comparison"] == "coverage-only"
    assert len(results) == 12 * config["seeds"]
    assert all(row["record"] == "result" for row in results)
    return config, results


def key(row):
    return row["landscape"], row["seed"], row["noise"], row["coverage"]


def verify(control, path, reference):
    config, rows = records(path)
    for row in rows:
        assert row["history"] == "none"
        assert row["history_observations"] == row["history_minima"] == 0
        assert 0 < row["n_evals"] + row["n_grads"] <= row["budget"]
        d = row["coverage_decisions"]
        assert set(d) == set(COUNTS + MEASURES)
        assert all(isinstance(d[k], int) and d[k] >= 0 for k in COUNTS)
        assert all(math.isfinite(d[k]) and d[k] >= 0 for k in MEASURES)
        assert d["comparisons"] == row["hops"]
        assert d["unresolved"] == 0
        assert d["accepted"] <= d["comparisons"]
        assert (
            d["probability_changes"]
            <= d["peer_delta_changes"]
            <= d["peer_overlap"]
            <= d["comparisons"]
        )
        assert d["drawn_comparisons"] <= d["comparisons"]
        assert d["drawn_disagreements"] <= min(d["drawn_comparisons"], d["probability_changes"])
        assert d["max_probability_change"] <= 1.0
        assert d["probability_change_sum"] <= d["probability_changes"]
        if row["coverage"] == "private":
            assert d["peer_overlap"] == d["peer_delta_changes"] == d["probability_changes"] == 0
            assert all(d[k] == 0.0 for k in MEASURES)
        if control in ("zero", "cover-all"):
            assert d["peer_delta_changes"] == d["probability_changes"] == 0
            assert d["drawn_disagreements"] == 0
            assert all(d[k] == 0.0 for k in MEASURES)
        if control == "cover-all":
            assert row["coverage_regions_per_chain"] == [1] * config["replicas"]
    by_key = {key(row): row for row in rows}
    assert len(by_key) == len(rows)
    pairs = []
    for row in rows:
        if row["coverage"] != "shared":
            continue
        peer = by_key[(*key(row)[:3], "private")]
        assert row["initial_position"] == peer["initial_position"]
        pairs.append((peer, row))
        if control == "zero":
            for field in ("best_value", "n_evals", "n_grads", "hops"):
                assert row[field] == peer[field], field
    if control == "zero":
        assert config["coverage_height"] == 0.0
    if control == "cover-all":
        assert config["coverage_radius"] >= 1.0
        assert sum(row["coverage_decisions"]["peer_overlap"] for _, row in pairs) > 0
    if control == "overlap":
        assert sum(row["coverage_decisions"]["probability_changes"] for _, row in pairs) > 0
    if reference:
        reference_config, reference_rows = records(reference)
        for field in ("dimension", "budget", "seeds", "replicas", "coverage_radius", "coverage_height"):
            assert config[field] == reference_config[field], field
        expected = {key(row): row for row in reference_rows}
        assert by_key.keys() == expected.keys()
        for identity, row in by_key.items():
            for field in ("initial_position", "best_value", "n_evals", "n_grads", "hops"):
                assert row[field] == expected[identity][field], (identity, field)
    summary = {
        "control": control,
        "dimension": config["dimension"],
        "radius": config["coverage_radius"],
        "height": config["coverage_height"],
        "outcomes": len(rows),
        "pairs": len(pairs),
        "exact_reference_match": bool(reference),
        "groups": [],
    }
    for landscape in sorted({row["landscape"] for row in rows}):
        for noise in sorted({row["noise"] for row in rows}):
            selected = [(p, s) for p, s in pairs if s["landscape"] == landscape and s["noise"] == noise]
            ds = [s["coverage_decisions"] for _, s in selected]
            summary["groups"].append({
                "landscape": landscape,
                "noise": noise,
                **{name: sum(d[name] for d in ds) for name in COUNTS},
                "probability_change_sum": sum(d["probability_change_sum"] for d in ds),
                **{name: max(d[name] for d in ds) for name in MEASURES if name != "probability_change_sum"},
                "shared_better": sum(s["best_value"] < p["best_value"] for p, s in selected),
                "equal": sum(s["best_value"] == p["best_value"] for p, s in selected),
                "private_better": sum(s["best_value"] > p["best_value"] for p, s in selected),
            })
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("control", choices=("zero", "cover-all", "overlap", "measured"))
    parser.add_argument("path", type=Path)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.control, args.path, args.reference), sort_keys=True))


if __name__ == "__main__":
    main()
