#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Write PROSE-RECEIPT.txt for the raw panel files and the prose run."""

from __future__ import annotations

import sys
from pathlib import Path

from prose_rate import balance


def main() -> None:
    here = Path(__file__).resolve().parent
    lines = [
        "Copyright (c) anneal developers",
        "",
        "The subject listing (SCOPE) is one generated note.",
        "The push plan (PUSH) is the publish note.",
        "We list each raw record and the writer that produced it.",
        "non-prose: raw run record.",
        "proselint.Typography on the copyright header is a false positive.",
        "The reason is owner-mandated copyright form.",
        "The header stays Copyright (c) anneal developers.",
        "lj104_sota_runner.py writes the panel logs.",
        "lj104_sota_runner.py writes the start receipt and the end receipt.",
        "lj104_sota_runner.py writes panel/session.txt.",
        "lj104_sota_runner.py writes T.txt at the panel start.",
        "write_floors.py writes floors.md.",
        "requench_panel.py writes requench.md.",
        "scope_paths.py writes the subject listing.",
        "write_tree_diff_stat.py writes the diff stat.",
        "write_bundle_verify.py writes the bundle note.",
        "write_replay_log.py writes the replay log.",
        "write_seal.py writes the seal stamps from stat.",
        "run_coverage.py writes the coverage table.",
        "The superseded panel copies are not a toolchain citation.",
        "discarded/README.txt labels those copies SUPERSEDED.",
    ]
    for folder, writer in (
        (here / "panel", "lj104_sota_runner.py"),
        (here / "discarded" / "panel", "lj104_sota_runner.py"),
    ):
        names = sorted(
            p.name
            for p in folder.iterdir()
            if p.is_file() and (p.suffix in {".csv", ".xyz"} or p.name.endswith((".span", ".stderr")) or p.name == "rows.tsv")
        )
        # rows.tsv and the four suffixes. .csv matches trace.csv via suffix .csv
        if not names:
            raise SystemExit(f"no raw files in {folder}")
        for name in names:
            rel = f"{folder.relative_to(here).as_posix()}/{name}"
            line = f"{writer} writes {rel}."
            if len(line) > 72:
                raise SystemExit(f"receipt line over 72: {len(line)} {line}")
            lines.append(line)
    lines = balance(lines)
    lines.append("")
    (here / "PROSE-RECEIPT.txt").write_text("\n".join(lines), encoding="utf-8")
    print("lines", len(lines))


if __name__ == "__main__":
    sys.exit(main() or 0)
