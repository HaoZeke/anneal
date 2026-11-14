#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Write the (H2O)16 panel tables.

`seeds.csv` keeps every recorded row. Rows already in the file are copied
through unchanged. `--clean measured.csv` appends the six-seed rerun: each
new row is named onehost, names code revision ca6adf99, and names host
bench. Without that flag the summary is rebuilt from the rows on disk and
the seed file is left untouched.
"""
import argparse
import math
import random
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SEEDS = ROOT / "bench" / "h2o16-crest" / "seeds.csv"
SUMMARY = ROOT / "bench" / "h2o16-crest" / "summary.csv"
REFERENCE = Decimal("-81.396292669509")
TOLERANCE = Decimal("1e-6")
BOOT_SEED = 20261110
BOOT_DRAWS = 10000
CLEAN = "ca6adf99d6d8c8e3d432c3ecc7e788242fc13f36"
HOST = "bench"
ROW = "onehost"
BASE = (
    "seed,row,anneal_Eh,crest_Eh,anneal_wall_s,crest_wall_s,hostname"
)
HEADER = (
    BASE + ",code_revision,anneal_potential_s,anneal_outside_s,"
    "crest_potential_s,crest_outside_s"
)
SUMMARY_HEADER = (
    "panel,n,W_plus,W_minus,p,rank_biserial,median_gap_Eh,"
    "bootstrap_low_Eh,bootstrap_high_Eh,anneal_hits,crest_hits,"
    "fisher_p,mannwhitney_p,anneal_potential_median_s,anneal_outside_median_s,"
    "crest_potential_median_s,crest_outside_median_s"
)


def ranks(values):
    order = sorted(range(len(values)), key=lambda i: abs(values[i]))
    out = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and abs(values[order[end]]) == abs(values[order[start]]):
            end += 1
        rank = (start + 1 + end) / 2
        for pos in range(start, end):
            out[order[pos]] = rank
        start = end
    return out


def wilcoxon(gaps):
    kept = [gap for gap in gaps if gap != 0.0]
    n = len(kept)
    if n == 0:
        return n, 1.0, 0.0, 0.0, 0.0
    weight = ranks(kept)
    observed = sum(rank for gap, rank in zip(kept, weight) if gap > 0.0)
    total = n * (n + 1) / 2
    patterns = 1 << n
    low = high = 0
    for mask in range(patterns):
        score = sum(rank for bit, rank in enumerate(weight) if mask & (1 << bit))
        if score <= observed:
            low += 1
        if score >= observed:
            high += 1
    p_value = min(1.0, 2.0 * min(low, high) / patterns)
    negative = total - observed
    effect = 0.0 if observed + negative == 0 else (observed - negative) / (observed + negative)
    return n, p_value, observed, negative, effect


def median(values):
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def bootstrap(gaps):
    rng = random.Random(BOOT_SEED)
    n = len(gaps)
    draws = []
    for _ in range(BOOT_DRAWS):
        sample = [gaps[rng.randrange(n)] for _ in range(n)]
        draws.append(median(sample))
    draws.sort()
    return draws[int(0.025 * BOOT_DRAWS)], draws[min(BOOT_DRAWS - 1, int(0.975 * BOOT_DRAWS))]


def fisher(hits_a, n_a, hits_b, n_b):
    """Two-sided Fisher exact p, summing tables no more likely than the observed."""
    hits = hits_a + hits_b
    total = n_a + n_b
    ways = math.comb(total, hits)

    def prob(k):
        return math.comb(n_a, k) * math.comb(n_b, hits - k) / ways

    observed = prob(hits_a)
    lo, hi = max(0, hits - n_b), min(n_a, hits)
    return min(1.0, sum(prob(k) for k in range(lo, hi + 1) if prob(k) <= observed * (1 + 1e-7)))


def mann_whitney(left, right):
    """Two-sided Mann-Whitney, tie-corrected normal approximation, continuity correction."""
    n1, n2 = len(left), len(right)
    pooled = sorted(list(left) + list(right))
    n = n1 + n2
    rank = {}
    ties = 0.0
    i = 0
    while i < n:
        j = i
        while j + 1 < n and pooled[j + 1] == pooled[i]:
            j += 1
        rank[pooled[i]] = (i + j + 2) / 2.0
        width = j - i + 1
        ties += width**3 - width
        i = j + 1
    u = sum(rank[v] for v in left) - n1 * (n1 + 1) / 2.0
    var = n1 * n2 / 12.0 * ((n + 1) - ties / (n * (n - 1)))
    if var <= 0.0:
        return 1.0
    dev = abs(u - n1 * n2 / 2.0) - 0.5
    return min(1.0, math.erfc(max(dev, 0.0) / math.sqrt(var) / math.sqrt(2.0)))


def plain(value):
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def median_text(values):
    present = [value for value in values if value != ""]
    if len(present) != len(values) or not present:
        return ""
    ordered = sorted(Decimal(value) for value in present)
    n = len(ordered)
    mid = n // 2
    chosen = ordered[mid] if n % 2 else (ordered[mid - 1] + ordered[mid]) / 2
    return plain(chosen)


def read_lines(path):
    text = path.read_text()
    if text and not text.endswith("\n"):
        text += "\n"
    return text.splitlines()


def cells(header, line):
    parts = line.split(",")
    return {name: parts[i] if i < len(parts) else "" for i, name in enumerate(header)}


def load(path):
    lines = read_lines(path)
    header = lines[0].split(",")
    rows = []
    body = []
    for line in lines[1:]:
        if not line.strip():
            continue
        body.append(line)
        item = cells(header, line)
        rows.append(
            {
                "seed": int(item["seed"]),
                "row": item["row"],
                "anneal": Decimal(item["anneal_Eh"]),
                "crest": Decimal(item["crest_Eh"]),
                "awall": float(item["anneal_wall_s"]),
                "cwall": float(item["crest_wall_s"]),
                "host": item["hostname"],
                "code": item.get("code_revision", ""),
                "apot": item.get("anneal_potential_s", ""),
                "aout": item.get("anneal_outside_s", ""),
                "cpot": item.get("crest_potential_s", ""),
                "cout": item.get("crest_outside_s", ""),
            }
        )
    return header, body, rows


def load_clean(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.startswith("seed,"):
            continue
        parts = line.split(",")
        if len(parts) < 5:
            raise SystemExit(f"measured row needs five fields: {line}")
        seed, anneal, crest, awall, cwall = parts[:5]
        apot = parts[5] if len(parts) > 5 else ""
        aout = parts[6] if len(parts) > 6 else ""
        cpot = parts[7] if len(parts) > 7 else ""
        cout = parts[8] if len(parts) > 8 else ""
        rows.append(
            {
                "seed": int(seed),
                "line": (
                    f"{seed},{ROW},{anneal},{crest},{awall},{cwall},{HOST},{CLEAN},"
                    f"{apot},{aout},{cpot},{cout}"
                ),
            }
        )
    return rows


def append_clean(path, extra):
    lines = read_lines(path)
    if lines[0] not in (BASE, HEADER):
        raise SystemExit(f"unexpected seeds header: {lines[0]}")
    body = [line for line in lines[1:] if line.strip()]
    header = HEADER.split(",")
    have = set()
    for line in body:
        item = cells(header if lines[0] == HEADER else BASE.split(","), line)
        if item["row"] == ROW:
            have.add(int(item["seed"]))
    added = []
    for row in extra:
        if row["seed"] in have:
            continue
        added.append(row["line"])
        have.add(row["seed"])
    if lines[0] == HEADER and not added:
        return False
    path.write_text("\n".join([HEADER, *body, *added]) + "\n")
    return True


def panel(rows, which):
    chosen = []
    for seed in range(15000, 15006):
        if which == ROW:
            row = next(
                item
                for item in rows
                if item["seed"] == seed and item["row"] == ROW and item["code"] == CLEAN and item["host"] == HOST
            )
        elif which == "hygienic" and seed == 15000:
            row = next(item for item in rows if item["seed"] == seed and item["row"] == "rerun")
        else:
            row = next(item for item in rows if item["seed"] == seed and item["row"] == "original")
        chosen.append(row)
    return chosen


def line(name, rows):
    gaps = [float(row["anneal"]) - float(row["crest"]) for row in rows]
    n, p_value, positive, negative, effect = wilcoxon(gaps)
    low, high = bootstrap(gaps)
    anneal_hits = sum(row["anneal"] <= REFERENCE + TOLERANCE for row in rows)
    crest_hits = sum(row["crest"] <= REFERENCE + TOLERANCE for row in rows)
    fisher_p = fisher(anneal_hits, n, crest_hits, n)
    mann_p = mann_whitney([float(row["anneal"]) for row in rows], [float(row["crest"]) for row in rows])
    return (
        f"{name},{n},{positive:.1f},{negative:.1f},{p_value:.10g},{effect:.10g},"
        f"{median(gaps):.10g},{low:.10g},{high:.10g},{anneal_hits},{crest_hits},"
        f"{fisher_p:.10g},{mann_p:.10g},"
        f"{median_text([row['apot'] for row in rows])},"
        f"{median_text([row['aout'] for row in rows])},"
        f"{median_text([row['cpot'] for row in rows])},"
        f"{median_text([row['cout'] for row in rows])}\n"
    )


def complete(rows):
    return all(
        any(row["seed"] == seed and row["row"] == ROW and row["code"] == CLEAN and row["host"] == HOST for row in rows)
        for seed in range(15000, 15006)
    )


def write_summary(rows):
    names = ["hygienic", "original"]
    if complete(rows):
        names.insert(0, ROW)
    text = SUMMARY_HEADER + "\n" + "".join(line(name, panel(rows, name)) for name in names)
    SUMMARY.write_text(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", type=Path)
    args = parser.parse_args()
    if args.clean is not None:
        append_clean(SEEDS, load_clean(args.clean))
    _header, _body, rows = load(SEEDS)
    write_summary(rows)


if __name__ == "__main__":
    main()
