# Copyright (c) anneal developers
"""The head-to-head drivers refuse a knockout name they do not know."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_drivers_refuse_an_unknown_knockout(tmp_path: Path):
    subprocess.run(
        [
            "cargo",
            "build",
            "--features",
            "rgpot-ex",
            "--example",
            "h2h_gmin_proof",
            "--example",
            "h2h_morse",
            "--example",
            "h2h_metal",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    meta = json.loads(
        subprocess.check_output(["cargo", "metadata", "--format-version", "1", "--no-deps"], cwd=ROOT, text=True)
    )
    examples_dir = Path(meta["target_directory"]) / "debug" / "examples"
    for binary, args in (
        (examples_dir / "h2h_gmin_proof", ["run", "lj38", "2000"]),
        (examples_dir / "h2h_morse", ["run", "m38r6", "2000"]),
        (examples_dir / "h2h_metal", ["coords", "zn38", "2000", str(tmp_path / "coords.txt")]),
    ):
        env = {key: value for key, value in os.environ.items() if not key.startswith("H2H_")}
        env["H2H_KNOCKOUT"] = "templates"
        proc = subprocess.run([str(binary), *args], cwd=ROOT, env=env, capture_output=True, text=True)
        assert proc.returncode != 0, binary.name
        assert "unknown knockout `templates`" in proc.stderr, (binary.name, proc.stderr)
