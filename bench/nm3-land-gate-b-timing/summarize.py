#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Write the timing-recheck verdict from recorded rows.

The pass rule is stats.verdict. This script only formats that result.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
GATE_DIR = HERE.parent / "nm3-land-gate-b"
for _path in (str(HERE), str(GATE_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import run_gate  # noqa: E402
import stats  # noqa: E402


def fmt_ratio(value: float) -> str:
    return f"{value:.8f}"


def fmt_seconds(value: float) -> str:
    return f"{value:.6f}"


def render_readme(
    reports: list[dict[str, object]],
    sessions: list[dict[str, str]],
    bin_rows: list[dict[str, str]],
    slot_counts: dict[tuple[int, str], int],
) -> str:
    if len(sessions) != 1:
        raise ValueError("one session is required")
    session = sessions[0]
    verdict = stats.overall_verdict(reports)
    lines = [
        "# nm3-land gate B timing recheck",
        "",
        f"**Verdict: {verdict}**",
        "",
        "## Rule",
        "",
        "For each cell, over the 24 seeds,",
        "",
        "R(X/A) = (sum of outside_s for arm X / sum of charged for arm X) / (sum of outside_s for arm A / sum of charged for arm A).",
        "",
        (
            f"The 95% interval is a paired percentile bootstrap with {stats.N_RESAMPLES} resamples, "
            f"random.Random({stats.RNG_SEED}) and randrange, endpoints at the type 7 quantiles "
            f"{stats.BOOTSTRAP_ALPHA / 2.0:.3f} and {1.0 - stats.BOOTSTRAP_ALPHA / 2.0:.3f}."
        ),
        (
            f"A cell passes if the lower endpoint of the B/A interval is <= {stats.THRESHOLD:.2f}, "
            "or if the B/A point ratio lies inside the A2/A interval, endpoints included. Otherwise it fails."
        ),
        "The per-seed median is the type 7 median of the 24 per-seed outside-per-call ratios.",
        "A and A2 outcomes match when both finished, the hit flag matches, evals_to_hit matches, charged matches, and best_bits_hex matches.",
        "",
        "## Provenance",
        "",
        f"- Session `{session['session_id']}` at `{session['utc']}`",
        f"- Machine id sha256 `{session['machine_id_sha256']}`",
        f"- CPU `{session['cpu_model']}` ({session['arch']}, kernel `{session['kernel']}`, affinity `{session['affinity']}`)",
        f"- Compilers `{session['rustc_version']}`; `{session['cargo_version']}`; Python `{session['python_version']}`",
        f"- Arm A `{session['arm_a_library']}` tree `{session['arm_a_tree']}`",
        f"- Arm B `{session['arm_b_library']}` tree `{session['arm_b_tree']}`",
        f"- Arm A2 `{session['arm_a2_library']}` tree `{session['arm_a2_tree']}`",
        f"- Seeds `{session['seeds']}`; cells `{session['cells']}`",
        f"- Order `{session['arm_order_protocol']}`; timeout `{session['job_timeout_s']}` s; seat `{session['ljos_seat']}`",
        (
            f"- Bootstrap resamples `{session['n_resamples']}`; generator seed `{session['bootstrap_rng_seed']}`; "
            f"threshold `{session['pass_threshold']}`"
        ),
        f"- Sutton-Chen library sha256 `{session['rgpot_sha256']}`",
        "- Slot counts: "
        + ", ".join(
            f"slot {slot} {arm}={slot_counts.get((slot, arm), 0)}"
            for slot in (1, 2, 3)
            for arm in stats.ARMS
        ),
        "",
    ]
    if bin_rows:
        lines.extend(
            [
                "### Release binary digests",
                "",
                "| arm | binary | sha256 |",
                "| --- | --- | --- |",
            ]
        )
        for row in bin_rows:
            lines.append(f"| {row['arm']} | `{row['binary']}` | `{row['sha256']}` |")
        lines.append("")
    lines.extend(
        [
            "## Cells",
            "",
            "| cell | R B/A | B/A 95% CI | R A2/A | A2/A 95% CI | median B/A | median A2/A | verdict |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for report in reports:
        lines.append(
            f"| {report['cell']} | {fmt_ratio(float(report['r_ba']))} | "
            f"[{fmt_ratio(float(report['ba_lo']))}, {fmt_ratio(float(report['ba_hi']))}] | "
            f"{fmt_ratio(float(report['r_aa']))} | "
            f"[{fmt_ratio(float(report['aa_lo']))}, {fmt_ratio(float(report['aa_hi']))}] | "
            f"{fmt_ratio(float(report['median_ba']))} | {fmt_ratio(float(report['median_aa']))} | "
            f"{report['verdict']} |"
        )
    lines.extend(
        [
            "",
            "## Hits and calls",
            "",
            "| cell | hits A | hits B | hits A2 | charged A | charged B | charged A2 | outside A (s) | outside B (s) | outside A2 (s) | evals median A | evals median B | evals median A2 | failed A | failed B | failed A2 |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for report in reports:
        hits = report["hits"]
        charged = report["charged"]
        outside = report["outside"]
        evals = report["evals_median"]
        failed = report["failed"]
        n = int(report["n_seeds"])
        assert isinstance(hits, dict) and isinstance(charged, dict)
        assert isinstance(outside, dict) and isinstance(evals, dict) and isinstance(failed, dict)
        lines.append(
            f"| {report['cell']} | {hits['A']}/{n} | {hits['B']}/{n} | {hits['A2']}/{n} | "
            f"{charged['A']} | {charged['B']} | {charged['A2']} | "
            f"{fmt_seconds(float(outside['A']))} | {fmt_seconds(float(outside['B']))} | "
            f"{fmt_seconds(float(outside['A2']))} | {evals['A']:.6g} | {evals['B']:.6g} | "
            f"{evals['A2']:.6g} | {failed['A']} | {failed['B']} | {failed['A2']} |"
        )
    lines.extend(
        [
            "",
            "## A versus A2 outcome identity",
            "",
            "| cell | matching seeds | seeds that differ |",
            "| --- | --- | --- |",
        ]
    )
    for report in reports:
        mismatches = report["mismatch_seeds"]
        assert isinstance(mismatches, list)
        shown = " ".join(str(seed) for seed in mismatches) if mismatches else ""
        lines.append(f"| {report['cell']} | {report['identical']}/{report['n_seeds']} | {shown} |")
    lines.extend(
        [
            "",
            "Outside seconds are the sums of recorded outside_s. Charged values are the sums of recorded charged calls. R divides those sums; it is not the mean of the per-seed ratios.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--sessions", type=Path, required=True)
    parser.add_argument("--bin-sha256", type=Path, required=True)
    parser.add_argument("--readme", type=Path, required=True)
    args = parser.parse_args(argv)

    rows = run_gate.read_csv(args.rows)
    if not rows:
        raise SystemExit("no rows")
    sessions = run_gate.read_csv(args.sessions)
    bin_rows = run_gate.read_csv(args.bin_sha256)
    slot_counts = stats.assert_schedule(rows)
    session_ids = {row["session_id"] for row in rows}
    if len(sessions) != 1 or session_ids != {sessions[0]["session_id"]}:
        raise SystemExit("rows and the session record disagree")
    reports = [stats.evaluate_cell(rows, cell) for cell in stats.CELLS]
    text = render_readme(reports, sessions, bin_rows, slot_counts)
    args.readme.parent.mkdir(parents=True, exist_ok=True)
    args.readme.write_text(text, encoding="utf-8", newline="\n")
    print(stats.overall_verdict(reports))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
