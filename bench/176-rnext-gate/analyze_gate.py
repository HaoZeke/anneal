"""Round-2 decision rule for the 176 r-next gate.

The functions in this file are the pre-registered bar. They are committed
before any round-2 row is written.
"""

from __future__ import annotations

import math
from pathlib import Path

from gate_io import durable_write_text, load_rows
from gate_panel import (
    BASE_COMMIT,
    BASE_TREE,
    CELLS,
    EXPECTED_B_TREE,
    MW_ALPHA,
    OUTSIDE_AP_HI,
    OUTSIDE_AP_LO,
    OUTSIDE_FAIL_RATIO,
    PLACEMENT_PADS,
    PLACEMENT_SEED,
    REPLACEMENT_CAP,
    RSS_ABS_ALLOW_KB,
    SEEDS,
    SESSION_R2,
    SUMMARY_COLUMNS,
)
from gate_stats import (
    fisher_loss_one_sided_p,
    fisher_two_sided_p,
    format_num,
    format_p,
    mann_whitney_p,
    median_type7,
)

MARKER_BEGIN = "<!-- superseded-r1-begin -->"
MARKER_END = "<!-- superseded-r1-end -->"


def bar_lines() -> list[str]:
    return [
        "## Pre-Registered Round 2 Bar",
        "",
        "This bar is fixed before any round-2 row. Later commits do not edit it.",
        "",
        "Arms. A and A' are two release builds of "
        f"`{BASE_COMMIT}` (tree `{BASE_TREE}`). "
        f"B is tree `{EXPECTED_B_TREE}`. "
        "P1..P5 are A rebuilt five times with padding 4096, 8192, 16384, "
        "32768, and 65536 bytes and no other source change. "
        f"Placement uses seed {PLACEMENT_SEED} only.",
        "",
        "Protocol. Seeds 2000..2011, paper budgets, release binaries, "
        "one pinned core, one thread, ABBA slot order. "
        "No budget is taken from the environment. "
        "A failed attempt is kept. An exit before RssAnon is reported is an "
        "aborted launch: it is kept and is not a replacement. "
        f"At most {REPLACEMENT_CAP} measured replacements are started per cell; "
        "beyond that the cell is INCONCLUSIVE.",
        "",
        "A kept B row must report basis_walks >= 1 and basis_evidence >= 1. "
        "Those counters are printed by the timed B process. "
        "RssAnon is read by that same process from its own status file "
        "after exec. ru_maxrss and VmHWM are reported and never gate.",
        "",
        "RssAnon bar, in kB. Let d = |median(B) - median(A)|. "
        "The rebuild term is |median(A') - median(A)|. "
        "The placement term is the max of |P_j - median(A)| over the five pads. "
        "The band is the max of those two terms. "
        f"The cell passes the RssAnon clause when d is within the band "
        f"or d <= {RSS_ABS_ALLOW_KB} (two pages).",
        "",
        "## Protocol amendment after round-2 data",
        "",
        "This amendment was written after round-2 rows were already in hand. "
        "It supersedes only the direction of the RssAnon clause above. "
        "lj38 and m38r14 failed that two-sided test because B was lower "
        "(1896 against 3908 kB, and 1546 against 7224 kB). B stops early "
        "on those cells, so it holds fewer basins. Holding less anonymous "
        "memory is not a regression. A regression is median(B) - median(A) "
        f"greater than the larger of the placement band and {RSS_ABS_ALLOW_KB} kB. "
        "The two-sided comparison is still computed and disclosed. "
        "It does not decide the cell.",
        "",
        "Other clauses, unchanged from round 1:",
        "",
        "1. Hit loss: hits(B) < hits(A) is a FAIL. Fisher exact p is reported.",
        "2. Calls: misses rank at +inf. FAIL when Mann-Whitney p < "
        f"{MW_ALPHA} and the median ratio B/A is above the A'/A band.",
        "3. Outside time: FAIL when median(outside_s/charged) has B/A > "
        f"{OUTSIDE_FAIL_RATIO} while A'/A is inside "
        f"[{OUTSIDE_AP_LO}, {OUTSIDE_AP_HI}]. "
        "If A'/A is outside that window the cell is UNRESOLVED.",
        "",
        "Overall PASS requires all 35 cells to be PASS.",
        "",
    ]


def _r2(rows: list[dict[str, str]], cell: str) -> list[dict[str, str]]:
    return [
        row for row in rows
        if row.get("session_id") == SESSION_R2 and row.get("cell") == cell
    ]


def ok_arm(rows: list[dict[str, str]], cell: str, arm: str) -> list[dict[str, str]]:
    found = [
        row for row in _r2(rows, cell)
        if row.get("arm") == arm
        and row.get("status") == "ok"
        and not row.get("superseded")
    ]
    return sorted(found, key=lambda row: int(row["seed"]))


def replacement_count(rows: list[dict[str, str]], cell: str) -> int:
    """Extra attempts that reported telemetry and were not kept."""
    return sum(
        1
        for row in _r2(rows, cell)
        if int(row.get("attempt") or "0") >= 2
        and row.get("rss_anon_kb")
        and row.get("status") != "ok"
    )


def _empty(spec, replacements: int, verdict: str, reason: str) -> dict[str, str]:
    row = {key: "" for key in SUMMARY_COLUMNS}
    row["cell"] = spec.cell
    row["family"] = spec.family
    row["budget"] = str(spec.budget)
    row["replacements"] = str(replacements)
    row["verdict"] = verdict
    row["reason"] = reason
    return row


def _median_field(rows: list[dict[str, str]], key: str) -> float:
    return median_type7([float(row[key]) for row in rows])


def summarize_cell(spec, rows: list[dict[str, str]]) -> dict[str, str]:
    cell = spec.cell
    replacements = replacement_count(rows, cell)
    rows_a = ok_arm(rows, cell, "A")
    rows_ap = ok_arm(rows, cell, "A'")
    rows_b = ok_arm(rows, cell, "B")
    complete = len(rows_a) == len(rows_ap) == len(rows_b) == len(SEEDS)
    if (not complete) or replacements > REPLACEMENT_CAP:
        why = []
        if not complete:
            why.append(
                f"incomplete ok rows A={len(rows_a)} A'={len(rows_ap)} B={len(rows_b)}"
            )
        if replacements > REPLACEMENT_CAP:
            why.append(f"replacements {replacements} exceed {REPLACEMENT_CAP}")
        return _empty(spec, replacements, "INCONCLUSIVE", "; ".join(why))

    hits_a = sum(int(row["hit"]) for row in rows_a)
    hits_ap = sum(int(row["hit"]) for row in rows_ap)
    hits_b = sum(int(row["hit"]) for row in rows_b)
    n_seeds = len(SEEDS)
    fisher_p = fisher_two_sided_p(hits_a, n_seeds, hits_b, n_seeds)
    fisher_loss_p = fisher_loss_one_sided_p(hits_a, n_seeds, hits_b, n_seeds)
    hit_diff = hits_b - hits_a
    if hit_diff > 0:
        hit_change = f"+{hit_diff} ({hits_a}->{hits_b})"
    elif hit_diff < 0:
        hit_change = f"{hit_diff} ({hits_a}->{hits_b})"
    else:
        hit_change = f"0 ({hits_a}->{hits_b})"

    ranks_a = [float(r["charged"]) if int(r["hit"]) == 1 else math.inf for r in rows_a]
    ranks_b = [float(r["charged"]) if int(r["hit"]) == 1 else math.inf for r in rows_b]
    med_calls_a = _median_field(rows_a, "charged")
    med_calls_ap = _median_field(rows_ap, "charged")
    med_calls_b = _median_field(rows_b, "charged")
    calls_b_a = med_calls_b / med_calls_a if med_calls_a > 0 else 1.0
    calls_ap_a = med_calls_ap / med_calls_a if med_calls_a > 0 else 1.0
    ap_calls_hi = max(1.0, calls_ap_a, (1.0 / calls_ap_a) if calls_ap_a > 0 else 1.0)
    mw_p, mw_worse_p = mann_whitney_p(ranks_b, ranks_a)

    out_a = [float(r["outside_s"]) / max(1, int(r["charged"])) for r in rows_a]
    out_ap = [float(r["outside_s"]) / max(1, int(r["charged"])) for r in rows_ap]
    out_b = [float(r["outside_s"]) / max(1, int(r["charged"])) for r in rows_b]
    med_out_a = median_type7(out_a)
    med_out_ap = median_type7(out_ap)
    med_out_b = median_type7(out_b)
    outside_b_a = med_out_b / med_out_a if med_out_a > 0 else 1.0
    outside_ap_a = med_out_ap / med_out_a if med_out_a > 0 else 1.0
    med_out_s_a = _median_field(rows_a, "outside_s")
    med_out_s_ap = _median_field(rows_ap, "outside_s")
    med_out_s_b = _median_field(rows_b, "outside_s")

    rss_a = _median_field(rows_a, "rss_anon_kb")
    rss_ap = _median_field(rows_ap, "rss_anon_kb")
    rss_b = _median_field(rows_b, "rss_anon_kb")
    delta = abs(rss_b - rss_a)
    aa = abs(rss_ap - rss_a)
    pad_vals: list[float] = []
    pad_missing = False
    for arm, _size in PLACEMENT_PADS:
        placed = ok_arm(rows, cell, arm)
        if len(placed) != 1 or int(placed[0]["seed"]) != PLACEMENT_SEED:
            pad_missing = True
            continue
        pad_vals.append(abs(float(placed[0]["rss_anon_kb"]) - rss_a))
    place = max(pad_vals) if pad_vals and not pad_missing else None
    band = max(aa, place) if place is not None else None

    walks = [int(row["basis_walks"]) for row in rows_b]
    evidence = [int(row["basis_evidence"]) for row in rows_b]
    claims = [float(row["basis_claims"]) for row in rows_b]
    probes = [float(row["basis_probes"]) for row in rows_b]
    candidates = [float(row["basis_candidates"]) for row in rows_b]
    spent = [float(row["basis_spent"]) for row in rows_b]

    reasons: list[str] = []
    failed = False
    unresolved = False
    if hits_b < hits_a:
        failed = True
        reasons.append(f"hit loss B={hits_b} < A={hits_a} (Fisher p={format_p(fisher_p)})")
    if (mw_p < MW_ALPHA or mw_worse_p < MW_ALPHA) and calls_b_a > ap_calls_hi:
        failed = True
        reasons.append(
            f"calls worse MW p={format_p(mw_p)} B/A={calls_b_a:.4g} > A'/A band {ap_calls_hi:.4g}"
        )
    if OUTSIDE_AP_LO <= outside_ap_a <= OUTSIDE_AP_HI:
        if outside_b_a > OUTSIDE_FAIL_RATIO:
            failed = True
            reasons.append(
                f"outside B/A={outside_b_a:.4f} > {OUTSIDE_FAIL_RATIO} with A'/A={outside_ap_a:.4f}"
            )
    else:
        unresolved = True
        reasons.append(f"A'/A={outside_ap_a:.4f} outside [{OUTSIDE_AP_LO}, {OUTSIDE_AP_HI}]")
    if min(walks) < 1 or min(evidence) < 1:
        failed = True
        reasons.append("B path counter is zero")
    allowance = float(RSS_ABS_ALLOW_KB) if band is None else max(band, float(RSS_ABS_ALLOW_KB))
    excess = rss_b - rss_a
    two_sided = "FAIL" if delta > allowance else "PASS"
    if excess > allowance:
        if band is None:
            unresolved = True
            reasons.append(
                f"placement band incomplete and rss excess B-A={excess:.4g} > {RSS_ABS_ALLOW_KB}"
            )
        else:
            failed = True
            reasons.append(
                f"rss excess B-A={excess:.4g} above max(band, {RSS_ABS_ALLOW_KB})"
            )
    reasons.append(
        f"two-sided disclosure: {two_sided} (|B-A|={delta:.4g}, allowance {allowance:.4g})"
    )
    if failed:
        verdict = "FAIL"
    elif unresolved:
        verdict = "UNRESOLVED"
    else:
        verdict = "PASS"
    if hit_diff > 0 and verdict == "PASS":
        reasons.append(f"hit gain +{hit_diff} ({hits_a}->{hits_b}, Fisher p={format_p(fisher_p)})")

    return {
        "cell": cell,
        "family": spec.family,
        "budget": str(spec.budget),
        "hits_a": str(hits_a),
        "hits_ap": str(hits_ap),
        "hits_b": str(hits_b),
        "fisher_p": format_p(fisher_p),
        "fisher_loss_p": format_p(fisher_loss_p),
        "hit_change": hit_change,
        "median_calls_a": format_num(med_calls_a, 8),
        "median_calls_ap": format_num(med_calls_ap, 8),
        "median_calls_b": format_num(med_calls_b, 8),
        "mw_p": format_p(mw_p),
        "mw_worse_p": format_p(mw_worse_p),
        "calls_b_a": f"{calls_b_a:.6g}",
        "calls_ap_a": f"{calls_ap_a:.6g}",
        "outside_med_a_s": f"{med_out_s_a:.6f}",
        "outside_med_ap_s": f"{med_out_s_ap:.6f}",
        "outside_med_b_s": f"{med_out_s_b:.6f}",
        "outside_b_a": f"{outside_b_a:.6f}",
        "outside_ap_a": f"{outside_ap_a:.6f}",
        "rss_med_a_kb": format_num(rss_a, 8),
        "rss_med_ap_kb": format_num(rss_ap, 8),
        "rss_med_b_kb": format_num(rss_b, 8),
        "rss_delta_kb": format_num(delta, 8),
        "rss_aa_kb": format_num(aa, 8),
        "rss_place_kb": "" if place is None else format_num(place, 8),
        "rss_band_kb": "" if band is None else format_num(band, 8),
        "rss_excess_kb": format_num(excess, 8),
        "rss_two_sided": two_sided,
        "vm_hwm_med_a_kb": format_num(_median_field(rows_a, "vm_hwm_kb"), 8),
        "vm_hwm_med_b_kb": format_num(_median_field(rows_b, "vm_hwm_kb"), 8),
        "ru_maxrss_med_a_kb": format_num(_median_field(rows_a, "ru_maxrss_kb"), 8),
        "ru_maxrss_med_b_kb": format_num(_median_field(rows_b, "ru_maxrss_kb"), 8),
        "basis_walks_min_b": str(min(walks)),
        "basis_evidence_min_b": str(min(evidence)),
        "basis_claims_med_b": format_num(median_type7(claims), 8),
        "basis_probes_med_b": format_num(median_type7(probes), 8),
        "basis_candidates_med_b": format_num(median_type7(candidates), 8),
        "basis_spent_med_b": format_num(median_type7(spent), 8),
        "replacements": str(replacements),
        "verdict": verdict,
        "reason": "; ".join(reasons),
    }


def superseded_block(verdict_path: Path) -> str:
    text = verdict_path.read_text(encoding="utf-8")
    begin = text.find(MARKER_BEGIN)
    end = text.find(MARKER_END)
    if begin < 0 or end < 0 or end < begin:
        raise RuntimeError("superseded round 1 block is missing")
    return text[begin:end + len(MARKER_END)] + "\n"


def overall_label(summary_rows: list[dict[str, str]]) -> str:
    if len(summary_rows) == len(CELLS) and all(row["verdict"] == "PASS" for row in summary_rows):
        return "PASS"
    if any(row["verdict"] == "FAIL" for row in summary_rows):
        return "FAIL"
    if any(row["verdict"] == "INCONCLUSIVE" for row in summary_rows):
        return "INCONCLUSIVE"
    if any(row["verdict"] == "UNRESOLVED" for row in summary_rows):
        return "UNRESOLVED"
    return "INCONCLUSIVE"


def render_verdict(summary_rows: list[dict[str, str]], kept: str, pending: bool) -> str:
    lines = [
        "# Default-On Gate: PR 176 r-next",
        "",
        "Round 1 is superseded. The round 1 verdict is kept at the bottom.",
        "",
    ]
    lines.extend(bar_lines())
    if pending:
        lines.extend([
            "## Round 2 Result",
            "",
            "Pending. No round 2 row has been committed.",
            "",
        ])
    else:
        label = overall_label(summary_rows)
        counts = {name: sum(1 for row in summary_rows if row["verdict"] == name)
                  for name in ("PASS", "FAIL", "UNRESOLVED", "INCONCLUSIVE")}
        lines.extend([
            "## Round 2 Result",
            "",
            f"Overall verdict: **{label}** "
            f"({counts['PASS']} PASS, {counts['FAIL']} FAIL, "
            f"{counts['UNRESOLVED']} UNRESOLVED, {counts['INCONCLUSIVE']} INCONCLUSIVE, "
            f"{len(summary_rows)} cells with rows).",
            "",
            "| cell | hits A/A'/B | Fisher | calls A/A'/B | MW | outside B/A | rss A/B/delta/band | B walks/evidence/claims/probes | verdict |",
            "| --- | :---: | ---: | :---: | ---: | ---: | :---: | :---: | :---: |",
        ])
        changes: list[str] = []
        for row in summary_rows:
            hits = f"{row['hits_a']}/{row['hits_ap']}/{row['hits_b']}"
            calls = f"{row['median_calls_a']} / {row['median_calls_ap']} / {row['median_calls_b']}"
            rss = f"{row['rss_med_a_kb']} / {row['rss_med_b_kb']} / {row['rss_delta_kb']} / {row['rss_band_kb']}"
            counters = (
                f"{row['basis_walks_min_b']} / {row['basis_evidence_min_b']} / "
                f"{row['basis_claims_med_b']} / {row['basis_probes_med_b']}"
            )
            lines.append(
                f"| `{row['cell']}` | {hits} | {row['fisher_p']} | {calls} | {row['mw_p']} | "
                f"{row['outside_b_a']} | {rss} | {counters} | **{row['verdict']}** |"
            )
            if row["hits_b"] and row["hits_a"] and row["hits_b"] != row["hits_a"]:
                changes.append(
                    f"- `{row['cell']}`: A={row['hits_a']}/12, A'={row['hits_ap']}/12, "
                    f"B={row['hits_b']}/12 ({row['hit_change']}, Fisher p={row['fisher_p']})"
                )
        lines.extend(["", "## Hit Changes", ""])
        lines.extend(changes or ["- None.", ""])
        lines.append("")
        lines.extend(["## Notes", ""])
        for row in summary_rows:
            if row["reason"]:
                lines.append(f"- `{row['cell']}`: {row['reason']}")
        lines.append("")
        disclosed = [
            row for row in summary_rows if row.get("rss_two_sided") == "FAIL"
        ]
        lines.extend(["## Disclosed two-sided RssAnon result", ""])
        if disclosed:
            lines.append(
                "The original two-sided bar would fail these cells. "
                "The amended gate does not use that result."
            )
            lines.append("")
            for row in disclosed:
                lines.append(
                    f"- `{row['cell']}`: A={row['rss_med_a_kb']} kB, "
                    f"B={row['rss_med_b_kb']} kB, |B-A|={row['rss_delta_kb']} kB, "
                    f"excess B-A={row['rss_excess_kb']} kB, band={row['rss_band_kb']} kB."
                )
        else:
            lines.append("No complete cell fails the original two-sided RssAnon bar.")
        lines.append("")
    lines.extend(["## Superseded Round 1 Verdict", "", kept])
    text = "\n".join(lines)
    if not text.endswith("\n"):
        text += "\n"
    return text


def write_summary(path: Path, summary_rows: list[dict[str, str]]) -> None:
    import csv
    import os
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SUMMARY_COLUMNS), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary_rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def summarize(rows_csv: Path, summary_csv: Path, verdict_md: Path) -> str:
    rows = load_rows(rows_csv)
    summary_rows = [
        summarize_cell(spec, rows)
        for spec in CELLS
        if _r2(rows, spec.cell)
    ]
    write_summary(summary_csv, summary_rows)
    text = render_verdict(summary_rows, superseded_block(verdict_md), pending=False)
    durable_write_text(verdict_md, text)
    return overall_label(summary_rows)


def stamp_pending(verdict_md: Path, original_r1: str) -> None:
    kept = MARKER_BEGIN + "\n" + original_r1.rstrip() + "\n" + MARKER_END + "\n"
    text = render_verdict([], kept, pending=True)
    durable_write_text(verdict_md, text)
