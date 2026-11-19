# Copyright (c) anneal developers
"""The paired runner forwards a quench gradient only to the candidate arm."""

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "openconf_h2h" / "run_bench.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("openconf_run_bench", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_quench_gradient_is_forwarded_only_to_the_candidate_arm(tmp_path, monkeypatch):
    module = load_runner()
    seen = {}

    def fake_run(cmd, **_kwargs):
        seen["cmd"] = list(cmd)
        out = Path(cmd[cmd.index("--out") + 1])
        out.mkdir(parents=True, exist_ok=True)
        (out / "job.json").write_text(json.dumps({"hostname": "bench"}))
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    args = argparse.Namespace(
        counter="counter",
        crest="crest",
        core=0,
        local_step=None,
        quench_gradient=1e-3,
        baseline=tmp_path / "baseline",
        any_host=False,
    )
    module.job("anneal", "rq3", 2048, tmp_path / "anneal", args, budget=10)
    assert seen["cmd"][seen["cmd"].index("--quench-gradient") + 1] == "0.001"
    module.job("develop", "rq3", 2048, tmp_path / "develop", args, budget=10)
    assert "--quench-gradient" not in seen["cmd"]
    assert "--anneal-path" in seen["cmd"]


if __name__ == "__main__":
    sys.exit("pytest runs this module")
