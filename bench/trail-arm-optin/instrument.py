#!/usr/bin/env python3
"""Add semantically inert counters to a benchmark worktree."""

from __future__ import annotations

import argparse
from pathlib import Path


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise RuntimeError(f"expected one instrumentation anchor: {old!r}")
    return text.replace(old, new, 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    text = args.source.read_text()
    text = replace_once(
        text,
        "    let mut trail_escapes = 0usize;\n",
        (
            "    let mut trail_escapes = 0usize;\n"
            "    let mut bench_stall_triggers = 0usize;\n"
            "    let mut bench_stall_redraws = 0usize;\n"
        ),
    )
    text = replace_once(
        text,
        "        let stuck = stalled(cfg, quiet, longest_quiet);\n",
        (
            "        let stuck = stalled(cfg, quiet, longest_quiet);\n"
            "        bench_stall_triggers += usize::from(stuck);\n"
        ),
    )
    redraw_anchor = (
        "        let stalled_from = e;\n"
        "        let mut stall_outcome: Option<f64> = None;\n"
    )
    redraw_count = (
        "        bench_stall_redraws += usize::from(\n"
        "            drawn.is_some() && drawn != stall_response,\n"
        "        );\n"
    )
    if "        let drawn = if !hop_room(hops) {" in text:
        text = replace_once(
            text,
            redraw_anchor,
            redraw_count + redraw_anchor,
        )
    text = replace_once(
        text,
        "    Ok(Outcome {\n",
        (
            "    eprintln!(\n"
            "        \"BENCH_STALL redraws={} triggers={}\",\n"
            "        bench_stall_redraws,\n"
            "        bench_stall_triggers,\n"
            "    );\n"
            "    Ok(Outcome {\n"
        ),
    )
    args.source.write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
