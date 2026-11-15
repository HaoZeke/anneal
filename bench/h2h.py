# Copyright (c) anneal developers
"""Smoke runs shared by the head-to-head asv suites.

A smoke run builds one release example and runs it on one seed at a small
budget, outside the bench namespace, so a suite can be checked to import
and run on any machine. It times nothing the paper reports: the paper rows
come from the Snakemake rules at ``PAPER_BUDGETS``.

A suite's ``Smoke.setup`` calls the ``need_*`` checks; one whose engine is
not on this machine raises ``NotImplementedError`` there, which asv reports
as skipped and pytest turns into a skip with the reason.
"""

from __future__ import annotations

import functools
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SMOKE_BUDGET = 2000


@functools.cache
def target_dir() -> Path:
    out = subprocess.run(
        ["cargo", "metadata", "--format-version", "1", "--no-deps"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return Path(json.loads(out)["target_directory"])


def build(example: str, features: tuple[str, ...] = ()) -> Path:
    command = ["cargo", "build", "--release", "--locked", "--example", example]
    if features:
        command += ["--features", ",".join(features)]
    subprocess.run(command, cwd=ROOT, check=True, env=dict(os.environ, RAYON_NUM_THREADS="1"))
    return target_dir() / "release" / "examples" / example


def need_env(*names: str) -> dict[str, str]:
    """The named variables, each naming an existing path."""
    found = {}
    for name in names:
        value = os.environ.get(name, "")
        if not value or not Path(value).exists():
            raise NotImplementedError(f"{name} does not name a file on this machine")
        found[name] = value
    return found


def need_library(name: str, default: Path) -> None:
    """``$name``, or ``default`` when it is unset, is a file on this machine."""
    path = Path(os.environ.get(name, default))
    if not path.exists():
        raise NotImplementedError(f"{name} ({path}) is not on this machine")


def need_bench_host() -> None:
    if subprocess.run(["sudo", "-n", "true"], capture_output=True, check=False).returncode:
        raise NotImplementedError("the bench host needs passwordless sudo for unshare --uts")


def need_modules(*names: str) -> None:
    missing = [name for name in names if importlib.util.find_spec(name) is None]
    if missing:
        raise NotImplementedError(f"python modules not installed: {', '.join(missing)}")


def smoke(
    example: str,
    args: list[str],
    *,
    budget: int | None = SMOKE_BUDGET,
    features: tuple[str, ...] = (),
    env: dict[str, str] | None = None,
    bench_host: bool = False,
) -> str:
    """Build ``example``, run it once on one thread and return its stdout.

    ``budget`` goes in ``H2H_BUDGET``; ``None`` leaves the example's own
    (for examples that take the budget as an argument). ``{work}`` in an
    argument is a fresh scratch directory. ``bench_host`` runs the example
    in a UTS namespace named bench, for examples that refuse any other host.
    """
    extra = dict(env or {}, RAYON_NUM_THREADS="1", OMP_NUM_THREADS="1")
    if budget is not None:
        extra["H2H_BUDGET"] = str(budget)
    if bench_host:
        need_bench_host()
    binary = build(example, features)
    with tempfile.TemporaryDirectory(prefix=f"{example}-") as work:
        command = [str(binary), *(arg.replace("{work}", work) for arg in args)]
        if bench_host:
            user = os.environ.get("USER") or subprocess.run(
                ["id", "-un"], capture_output=True, text=True, check=True
            ).stdout.strip()
            pairs = [f"{key}={value}" for key, value in extra.items()]
            command = [
                "sudo", "-n", "unshare", "--uts", "sh", "-c",
                'hostname bench; user=$1; shift; exec sudo -u "$user" env "$@"',
                "_", user, *pairs, *command,
            ]
        proc = subprocess.run(
            command,
            cwd=work,
            env=dict(os.environ, **extra),
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"{example} {' '.join(args)} exited {proc.returncode}:\n{proc.stderr[-2000:]}")
    return proc.stdout


def check_budget(example: str, system: str, expected: int, *, features: tuple[str, ...] = (),
                 env: dict[str, str] | None = None) -> None:
    """The example's own paper budget for ``system`` is ``expected``."""
    binary = build(example, features)
    proc = subprocess.run([str(binary), "budget", system], capture_output=True, text=True, check=True,
                          env=dict(os.environ, **(env or {})))
    budget = int(proc.stdout.split()[0])
    if budget != expected:
        raise AssertionError(f"{example} budget {system} is {budget}, PAPER_BUDGETS says {expected}")


def smoke_script(script: Path, *runs: list[str], modules: tuple[str, ...] = (),
                 libraries: dict[str, Path] | None = None) -> None:
    """Run a Python bench script once per argument list, in one scratch directory.

    ``modules`` must import and every path in ``libraries`` (keyed by the
    variable that overrides it) must exist, or the smoke is skipped.
    """
    need_modules(*modules)
    for name, default in (libraries or {}).items():
        need_library(name, default)
    with tempfile.TemporaryDirectory(prefix=f"{script.stem}-") as work:
        for args in runs:
            proc = subprocess.run(
                [sys.executable, "-u", str(script), *(arg.replace("{work}", work) for arg in args)],
                cwd=work,
                env=dict(os.environ, RAYON_NUM_THREADS="1", OMP_NUM_THREADS="1"),
                capture_output=True,
                text=True,
                timeout=1200,
                check=False,
            )
            if proc.returncode != 0:
                raise RuntimeError(f"{script.name} {' '.join(args)} exited {proc.returncode}:\n{proc.stderr[-2000:]}")
