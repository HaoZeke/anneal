# Copyright (c) anneal developers
"""Paired analysis of the n=24 anneal-GMIN panel.

Reads the pre-run manifest and the stored rows. For every cell it
reports hits, the discordant counts, an exact two-sided McNemar test,
the Newcombe paired 95 percent interval (method 10), a Wilcoxon
signed-rank test on wall time, and the median calls with misses ranked
after every hit. The report is this program's stdout written to a file.
"""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_panel  # noqa: E402
from target_tol import energy_hits  # noqa: E402

# 0.975 quantile of the standard normal, used by the Wilson score interval.
Z95 = 1.959963984540054

ROW_FIELDS = [
    "cell",
    "seed",
    "arm",
    "slot",
    "core",
    "hit",
    "best_energy",
    "calls",
    "wall_s",
    "code_revision",
    "gmin_svnrev",
    "harness_blob_ids",
    "machine_id",
    "error",
]


def wilson(successes: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for a single proportion."""
    if n <= 0:
        raise ValueError("wilson needs a positive sample")
    p = successes / n
    z2 = z * z
    den = 1.0 + z2 / n
    centre = (p + z2 / (2.0 * n)) / den
    half = z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / den
    return centre - half, centre + half


def newcombe_method10(a: int, b: int, c: int, d: int, z: float = Z95) -> tuple[float, float, float]:
    """Newcombe paired method 10 for the difference of hit proportions.

    ``a`` both hit, ``b`` anneal only, ``c`` GMIN only, ``d`` neither.
    The point is the anneal proportion minus the GMIN proportion.
    Method 10 (Newcombe, Statistics in Medicine 17, 2635-2650, 1998)
    combines the two Wilson intervals and the continuity-corrected phi
    coefficient: when ``a*d > b*c`` the numerator of phi is
    ``max(a*d - b*c - n/2, 0)``.
    """
    n = a + b + c + d
    if n <= 0:
        raise ValueError("empty table")
    p1 = (a + b) / n
    p2 = (a + c) / n
    l1, u1 = wilson(a + b, n, z)
    l2, u2 = wilson(a + c, n, z)
    marginal = (a + b) * (c + d) * (a + c) * (b + d)
    if marginal == 0:
        phi = 0.0
    else:
        raw = a * d - b * c
        num = max(raw - n / 2.0, 0.0) if raw > 0 else float(raw)
        phi = num / math.sqrt(marginal)
    dl1, du1 = p1 - l1, u1 - p1
    dl2, du2 = p2 - l2, u2 - p2
    lower_in = dl1 * dl1 - 2.0 * phi * dl1 * du2 + du2 * du2
    upper_in = du1 * du1 - 2.0 * phi * du1 * dl2 + dl2 * dl2
    if -1e-12 < lower_in < 0.0:
        lower_in = 0.0
    if -1e-12 < upper_in < 0.0:
        upper_in = 0.0
    if lower_in < 0.0 or upper_in < 0.0:
        raise ValueError("newcombe method 10 lost its positive width")
    diff = p1 - p2
    return diff, diff - math.sqrt(lower_in), diff + math.sqrt(upper_in)


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar p, the two-sided binomial test at one half.

    Outcomes whose null probability is no larger than the observed one
    are summed. With a fair coin this is the two tails.
    """
    n = b + c
    if n == 0:
        return 1.0
    ways = [math.comb(n, k) for k in range(n + 1)]
    observed = ways[b]
    return sum(w for w in ways if w <= observed) / float(1 << n)


def _average_ranks(abs_diffs: list[float]) -> tuple[list[float], float]:
    n = len(abs_diffs)
    order = sorted(range(n), key=lambda i: abs_diffs[i])
    ranks = [0.0] * n
    tie = 0.0
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs_diffs[order[j + 1]] == abs_diffs[order[i]]:
            j += 1
        avg = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        t = j - i + 1
        tie += t**3 - t
        i = j + 1
    return ranks, tie


def wilcoxon_wall(anneal: list[float], gmin: list[float]) -> dict[str, str]:
    """Two-sided signed-rank test of anneal wall minus GMIN wall.

    Zero differences are dropped. The remaining test is the exact
    permutation of the ranks when every absolute difference is unique,
    and otherwise the normal approximation with the tie correction and
    a continuity correction.
    """
    if len(anneal) != len(gmin) or not anneal:
        raise ValueError("wilcoxon needs paired samples")
    raw = [a - g for a, g in zip(anneal, gmin)]
    zeros = sum(d == 0.0 for d in raw)
    diffs = [d for d in raw if d != 0.0]
    if not diffs:
        return {
            "p": "",
            "method": "all differences zero",
            "zeros": str(zeros),
            "median_diff": f"{run_panel.type7(raw, 0.5):.6f}",
        }
    abs_diffs = [abs(d) for d in diffs]
    unique = len(set(abs_diffs)) == len(abs_diffs)
    if unique:
        ranks = list(range(1, len(diffs) + 1))
        order = sorted(range(len(diffs)), key=lambda i: abs_diffs[i])
        ranked = [0] * len(diffs)
        for rank, index in zip(ranks, order):
            ranked[index] = rank
        positive = sum(rank for rank, diff in zip(ranked, diffs) if diff > 0.0)
        total = len(diffs) * (len(diffs) + 1) // 2
        dp = [0] * (total + 1)
        dp[0] = 1
        for rank in range(1, len(diffs) + 1):
            for score in range(total, rank - 1, -1):
                dp[score] += dp[score - rank]
        dev = abs(2 * positive - total)
        extreme = sum(count for score, count in enumerate(dp) if abs(2 * score - total) >= dev)
        p = extreme / float(1 << len(diffs))
        method = "exact"
    else:
        ranks, tie = _average_ranks(abs_diffs)
        positive = sum(rank for rank, diff in zip(ranks, diffs) if diff > 0.0)
        n = len(diffs)
        mu = n * (n + 1) / 4.0
        var = n * (n + 1) * (2 * n + 1) / 24.0 - tie / 48.0
        if var <= 0.0:
            p = 1.0
        else:
            z = max(abs(positive - mu) - 0.5, 0.0) / math.sqrt(var)
            p = min(1.0, math.erfc(z / math.sqrt(2.0)))
        method = "normal"
    return {
        "p": format(p, ".12g"),
        "method": method,
        "zeros": str(zeros),
        "median_diff": f"{run_panel.type7(raw, 0.5):.6f}",
    }


def median_calls(rows: list[dict[str, str]]) -> float:
    """Type 7 median of calls, with a miss ordered after every hit."""
    ordered = sorted(rows, key=lambda row: (row["hit"] == "0", int(row["calls"]), int(row["seed"])))
    return run_panel.type7([float(row["calls"]) for row in ordered], 0.5)


def show_calls(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.2f}".rstrip("0").rstrip(".")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ROW_FIELDS:
            raise SystemExit(f"unexpected columns: {reader.fieldnames}")
        return list(reader)


def require_git_object(repo: Path, oid: str) -> None:
    proc = subprocess.run(
        ["git", "cat-file", "-e", oid],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise SystemExit(f"provenance object missing: {oid}")


def committed_seeds(root: Path) -> dict[str, list[int]]:
    path = root / "bench" / "gmin-panel" / "rows.csv"
    found: dict[str, set[int]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            found.setdefault(row["system"], set()).add(int(row["seed"]))
    return {name: sorted(seeds) for name, seeds in sorted(found.items())}


def lj103_in_base(repo: Path, base: str) -> bool:
    text = subprocess.run(
        ["git", "show", f"{base}:examples/h2h_gmin_proof.rs"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return '"lj103" => (103, -575.766131)' in text


def seed_paragraph(found: dict[str, list[int]]) -> str:
    original = list(range(2000, 2012))
    uniform = all(seeds == original for seeds in found.values()) and len(found) == 13
    if uniform:
        return (
            "The committed n=12 rows in bench/gmin-panel use seeds "
            "2000-2011 on every cell. The declared seeds are 2000-2023, "
            "that original twelve plus 2012-2023."
        )
    lines = ["The committed n=12 rows do not all use seeds 2000-2011."]
    for name, seeds in found.items():
        lines.append(f"{name}: {' '.join(str(seed) for seed in seeds)}")
    lines.append(
        "The run keeps each cell's original twelve seeds and adds twelve declared seeds."
    )
    return "\n".join(lines)


def wrap(text: str, width: int = 72) -> str:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        trial = word if not current else current + " " + word
        if len(trial) <= width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return "\n".join(lines)


def analyse(rows: list[dict[str, str]], cells: list[str], budgets: dict[str, int], references: dict[str, float]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in rows:
        if row["error"]:
            continue
        key = (row["cell"], row["seed"], row["arm"])
        if key in grouped:
            raise SystemExit(f"duplicate row {key}")
        grouped[key] = row
        reference = references[row["cell"]]
        best = float(row["best_energy"])
        hit = row["hit"] == "1"
        if hit != energy_hits(best, reference, run_panel.HIT_TOL):
            raise SystemExit(f"stored hit disagrees with the hit rule at {key}")
        budget = budgets[row["cell"]]
        calls = int(row["calls"])
        if hit and not (0 < calls <= budget):
            raise SystemExit(f"hit calls out of range at {key}")
        if not hit and calls != budget:
            raise SystemExit(f"miss is not at the budget at {key}")
    out = []
    for cell in cells:
        seeds = sorted({seed for (name, seed, _arm) in grouped if name == cell})
        paired = []
        for seed in seeds:
            anneal = grouped.get((cell, seed, "anneal"))
            gmin = grouped.get((cell, seed, "gmin"))
            if anneal and gmin:
                paired.append((anneal, gmin))
        anneal_rows = [grouped[key] for key in grouped if key[0] == cell and key[2] == "anneal"]
        gmin_rows = [grouped[key] for key in grouped if key[0] == cell and key[2] == "gmin"]
        record: dict[str, str] = {
            "cell": cell,
            "budget": str(budgets[cell]),
            "anneal_hits": str(sum(row["hit"] == "1" for row in anneal_rows)),
            "anneal_n": str(len(anneal_rows)),
            "gmin_hits": str(sum(row["hit"] == "1" for row in gmin_rows)),
            "gmin_n": str(len(gmin_rows)),
            "pairs": str(len(paired)),
            "b": "",
            "c": "",
            "mcnemar_p": "",
            "diff": "",
            "newcombe_low": "",
            "newcombe_high": "",
            "wilcoxon_p": "",
            "wilcoxon_method": "",
            "wilcoxon_zeros": "",
            "median_wall_diff": "",
            "anneal_median_calls": show_calls(median_calls(anneal_rows)) if anneal_rows else "",
            "gmin_median_calls": show_calls(median_calls(gmin_rows)) if gmin_rows else "",
        }
        if paired:
            both = anneal_only = gmin_only = neither = 0
            for anneal, gmin in paired:
                a_hit = anneal["hit"] == "1"
                g_hit = gmin["hit"] == "1"
                if a_hit and g_hit:
                    both += 1
                elif a_hit:
                    anneal_only += 1
                elif g_hit:
                    gmin_only += 1
                else:
                    neither += 1
            diff, low, high = newcombe_method10(both, anneal_only, gmin_only, neither)
            walls = wilcoxon_wall(
                [float(anneal["wall_s"]) for anneal, _gmin in paired],
                [float(gmin["wall_s"]) for _anneal, gmin in paired],
            )
            record.update(
                {
                    "b": str(anneal_only),
                    "c": str(gmin_only),
                    "mcnemar_p": format(mcnemar_exact(anneal_only, gmin_only), ".12g"),
                    "diff": f"{diff:.6f}",
                    "newcombe_low": f"{low:.6f}",
                    "newcombe_high": f"{high:.6f}",
                    "wilcoxon_p": walls["p"],
                    "wilcoxon_method": walls["method"],
                    "wilcoxon_zeros": walls["zeros"],
                    "median_wall_diff": walls["median_diff"],
                }
            )
        out.append(record)
    return out


def balance_text(rows: list[dict[str, str]]) -> str:
    counts = Counter((row["slot"], row["arm"]) for row in rows if not row["error"])
    parts = [f"slot {slot} {arm} {counts[(slot, arm)]}" for slot in ("0", "1") for arm in ("anneal", "gmin")]
    return ", ".join(parts)


def provenance(repo: Path, rows: list[dict[str, str]], base: str, svnrev: str) -> None:
    if not rows:
        raise SystemExit("no rows")
    revisions = {row["code_revision"] for row in rows}
    if revisions != {base}:
        raise SystemExit(f"code_revision set is {revisions}")
    svn = {row["gmin_svnrev"] for row in rows}
    if svn != {svnrev}:
        raise SystemExit(f"gmin_svnrev set is {svn}")
    require_git_object(repo, base)
    blobs: set[str] = set()
    for row in rows:
        ids = row["harness_blob_ids"].split()
        if len(ids) < 2:
            raise SystemExit("row is missing harness blob ids")
        blobs.update(ids)
    for oid in sorted(blobs):
        require_git_object(repo, oid)


def render(
    root: Path,
    manifest: dict,
    rows: list[dict[str, str]],
    table: list[dict[str, str]],
) -> str:
    base = manifest["base"]
    present = lj103_in_base(root, base)
    lj = [row for row in rows if row["cell"] == "lj103" and not row["error"]]
    lj_arms = sorted({row["arm"] for row in lj})
    if present:
        lj_text = (
            "h2h_gmin_proof at BASE contains the lj103 entry "
            "(103 atoms, reference -575.766131, budget 4000000). "
            "The lj103 table patch was not applied."
        )
    else:
        lj_text = (
            "h2h_gmin_proof at BASE has no lj103 entry. "
            "The lj103 table patch was not applied, and the GMIN arm was left unrun."
        )
    if lj_arms == ["anneal", "gmin"]:
        lj_text += " Both arms have stored rows."
    elif lj_arms == ["anneal"]:
        lj_text += " Only the anneal arm has stored rows. The GMIN arm was left unrun."
    elif lj_arms == ["gmin"]:
        lj_text += " Only the GMIN arm has stored rows."
    elif not lj_arms:
        lj_text += " Neither arm has a stored row."
    errors = [row for row in rows if row["error"]]
    seeds = committed_seeds(root)
    header = [
        "Cell",
        "Budget",
        "Anneal hits",
        "GMIN hits",
        "b",
        "c",
        "McNemar p",
        "Diff",
        "Newcombe low",
        "Newcombe high",
        "Wilcoxon p",
        "Wilcoxon",
        "Zeros",
        "Median wall diff",
        "Anneal median calls",
        "GMIN median calls",
    ]
    lines = [
        "# Anneal against GMIN, n=24",
        "",
        wrap(seed_paragraph(seeds)),
        "",
        wrap(lj_text),
        "",
        wrap(
            "Cells are the GMIN main panel. A hit is a quenched energy at or "
            "below the reference plus 0.001. A miss is stored at the budget. "
            "b counts seeds where anneal hits and GMIN misses. c counts the "
            "opposite. The difference is the anneal hit proportion minus the "
            "GMIN hit proportion. McNemar is the exact two-sided binomial "
            "test on the discordant pairs. The interval is Newcombe's paired "
            "method 10 at 95 percent. Wilcoxon is the two-sided signed-rank "
            "test of anneal wall minus GMIN wall: exact when every nonzero "
            "absolute difference is unique, and the normal approximation with "
            "the tie and continuity corrections otherwise. Zero wall "
            "differences are dropped before ranking. Median calls use "
            "Hyndman-Fan type 7 with a miss ranked after every hit. Median "
            "wall diff is the type 7 median of the paired wall differences."
        ),
        "",
        wrap("Declared cells and paper budgets:"),
        "",
    ]
    for cell in manifest["cells"]:
        lines.append(f"- {cell}: {manifest['budgets'][cell]}")
    lines.append("")
    lines.append(wrap(f"Declared seeds: {manifest['seeds'][0]}-{manifest['seeds'][-1]} ({len(manifest['seeds'])})."))
    lines.append("")
    lines.append(wrap(f"Stored successful (slot, arm) counts: {balance_text(rows)}."))
    lines.append("")
    if errors:
        lines.append(wrap(f"Rows with an error: {len(errors)}."))
        for row in errors:
            lines.append(f"- {row['cell']} seed {row['seed']} {row['arm']} slot {row['slot']}: {row['error']}")
        lines.append("")
    else:
        lines.append("Rows with an error: 0.")
        lines.append("")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join("---" for _ in header) + " |")
    for row in table:
        cells = [
            row["cell"],
            row["budget"],
            f"{row['anneal_hits']}/{row['anneal_n']}" if row["anneal_n"] else "0/0",
            f"{row['gmin_hits']}/{row['gmin_n']}" if row["gmin_n"] else "0/0",
            row["b"],
            row["c"],
            row["mcnemar_p"],
            row["diff"],
            row["newcombe_low"],
            row["newcombe_high"],
            row["wilcoxon_p"],
            row["wilcoxon_method"],
            row["wilcoxon_zeros"],
            row["median_wall_diff"],
            row["anneal_median_calls"],
            row["gmin_median_calls"],
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(
        wrap(
            f"Code revision {base}. GMIN svn revision {manifest['gmin_svnrev']}. "
            "Every code revision and harness blob id resolves as a git object."
        )
    )
    lines.append("")
    return "\n".join(lines)


def self_test() -> None:
    diff, low, high = newcombe_method10(20, 12, 2, 16)
    if abs(diff - 0.2) > 1e-15 or abs(low - 0.056156424841748265) > 1e-12 or abs(high - 0.3292069954001405) > 1e-12:
        raise SystemExit(f"sputum interval {diff} {low} {high}")
    diff, low, high = newcombe_method10(10, 5, 2, 20)
    if abs(low + 0.06439260507116965) > 1e-12 or abs(high - 0.22111601789582339) > 1e-12:
        raise SystemExit(f"paired example {diff} {low} {high}")
    if abs(mcnemar_exact(12, 2) - 212 / 16384) > 1e-15:
        raise SystemExit("mcnemar")
    if mcnemar_exact(6, 6) != 1.0 or mcnemar_exact(0, 0) != 1.0:
        raise SystemExit("mcnemar ties")
    walls = wilcoxon_wall([1.0, 2.0, 3.0], [0.0, 0.0, 0.0])
    if walls["method"] != "exact" or abs(float(walls["p"]) - 0.25) > 1e-15:
        raise SystemExit(f"wilcoxon exact {walls}")
    walls = wilcoxon_wall([1.0, 1.0, -2.0, 3.0], [0.0, 0.0, 0.0, 0.0])
    if walls["method"] != "normal":
        raise SystemExit(f"wilcoxon ties should use the normal approximation {walls}")
    sample = [
        {"hit": "1", "calls": "10", "seed": "1"},
        {"hit": "0", "calls": "10", "seed": "2"},
        {"hit": "1", "calls": "4", "seed": "3"},
        {"hit": "1", "calls": "6", "seed": "4"},
    ]
    # Misses sort after a hit with the same call count. The type 7 median
    # of four call counts that are all in {4, 6, 10, 10} is 8.
    if abs(median_calls(sample) - 8.0) > 1e-12:
        raise SystemExit("median calls")
    try:
        from scipy.stats import binomtest, wilcoxon
    except ImportError:
        pass
    else:
        if abs(mcnemar_exact(8, 3) - binomtest(8, 11, 0.5, alternative="two-sided").pvalue) > 1e-12:
            raise SystemExit("mcnemar against scipy")
        paired = [0.2, -0.1, 0.5, -0.4, 1.2, -0.3]
        got = wilcoxon_wall(paired, [0.0] * len(paired))
        want = wilcoxon(paired, alternative="two-sided", method="exact", zero_method="wilcox")
        if got["method"] != "exact" or abs(float(got["p"]) - float(want.pvalue)) > 1e-12:
            raise SystemExit(f"wilcoxon against scipy {got} {want.pvalue}")
        tied = [1.0, 1.0, -2.0, 3.0, -3.0, 4.0, 0.5, 0.5]
        got = wilcoxon_wall(tied, [0.0] * len(tied))
        want = wilcoxon(tied, alternative="two-sided", method="approx", correction=True, zero_method="wilcox")
        if got["method"] != "normal" or abs(float(got["p"]) - float(want.pvalue)) > 1e-12:
            raise SystemExit(f"wilcoxon normal against scipy {got} {want.pvalue}")
    print("ok")


def main(argv: list[str]) -> int:
    if len(argv) == 2 and argv[1] == "--self-test":
        self_test()
        return 0
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv[1:])
    import json

    manifest = json.loads(args.manifest.read_text())
    rows = read_rows(args.rows)
    provenance(args.root, rows, manifest["base"], manifest["gmin_svnrev"])
    references = {name: float(value) for name, value in manifest["references"].items()}
    budgets = {name: int(value) for name, value in manifest["budgets"].items()}
    table = analyse(rows, list(manifest["cells"]), budgets, references)
    text = render(args.root, manifest, rows, table)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
