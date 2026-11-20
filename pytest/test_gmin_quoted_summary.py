# Copyright (c) anneal developers
"""quoted_summary keeps a GMIN row only when this repository holds its commits."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROWS = ROOT / "bench" / "gmin-panel" / "rows.csv"


def panel():
    spec = importlib.util.spec_from_file_location("gmin_run_panel", ROOT / "scripts" / "gmin_panel" / "run_panel.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_quoted_summary_drops_a_system_whose_commit_is_absent(tmp_path):
    module = panel()
    before = hashlib.sha256(ROWS.read_bytes()).hexdigest()
    rows = module.read_rows(ROWS)
    summary = module.summarise(rows, "latest")
    quoted = module.quoted_summary(summary)
    kept = {row["system"] for row in quoted}
    dropped = {row["system"] for row in summary} - kept
    assert {"lj38", "lj55", "lj75", "lj98", "lj102", "lj103", "lj104", "lj150"} <= kept
    assert {"au55", "m55r6", "m55r14", "m80r6", "m80r14"} <= dropped
    manuscript = tmp_path / "manuscript.tex"
    manuscript.write_text(
        "% gmin-panel table: start\n"
        "old\n"
        "% gmin-panel table: end\n"
    )
    readme = tmp_path / "README.md"
    readme.write_text(module.README_MARKS[0] + "\nold\n" + module.README_MARKS[1] + "\n")
    summary_path = tmp_path / "summary.csv"
    module.write_tables(ROOT, rows, "latest", True, summary_path, readme, manuscript)
    spliced = manuscript.read_text()
    table = spliced.split("\\begin{tabular}", 1)[1]
    assert "LJ\\(_{38}\\)" in table
    assert "Au\\(_{55}\\)" not in spliced
    assert "M\\(_{80}\\)" not in table
    assert hashlib.sha256(ROWS.read_bytes()).hexdigest() == before
