# Copyright (c) anneal developers
"""Every head-to-head is a Snakemake rule and an asv suite at the paper budgets.

``bench/h2h.json`` lists each head-to-head rule with its runner, the source
of its paper budgets and seeds, its asv suite, the committed rows it writes
and every known shortfall. These tests hold that file to the Snakefile, the
runner sources, the suites and the rows, so a new rule, a drifted budget, a
short seed count or a runner that stops stamping provenance fails here.

The dry run needs ``snakemake`` on PATH (or ``SNAKEMAKE``); the only
outputs it may find missing are the ones the registry records. The smoke runs
build each suite's example and run one seed at a small budget; they are
opt-in with ``ANNEAL_H2H_SMOKE=1``.
"""

from __future__ import annotations

import ast
import builtins
import contextlib
import csv
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from h2h_registry import load_registry  # noqa: E402

REGISTRY = load_registry(ROOT)
ENTRIES = REGISTRY["h2h"]
RULES = [entry["rule"] for entry in ENTRIES]
MIN_SEEDS = 6
SAFE_BUILTINS = {
    name: getattr(builtins, name)
    for name in ("dict", "enumerate", "float", "frozenset", "int", "len", "list", "max", "min",
                 "range", "set", "sorted", "str", "sum", "tuple", "zip")
}
PINNED = re.compile(r"""taskset["',\s]+-c""")


def by_rule(function):
    return pytest.mark.parametrize("entry", ENTRIES, ids=RULES)(function)


def snakefile_rules() -> list[str]:
    return re.findall(r"^rule (\w+):", (ROOT / "workflow" / "Snakefile").read_text(), re.MULTILINE)


def constants(path: Path) -> dict:
    """Top-level constants of a runner, without importing it.

    Python: every top-level assignment to an upper-case name that evaluates
    from literals and earlier such names. Shell: ``NAME=value`` and
    ``NAME=${VAR:-value}``, with comma lists read as integers.
    """
    text = path.read_text()
    if path.suffix == ".sh":
        found = {}
        for name, value in re.findall(r"^([A-Z0-9_]+)=(?:\$\{[A-Z0-9_]+:-([^}]*)\}|)", text, re.MULTILINE):
            parts = value.split(",")
            if value and all(p.strip().isdigit() for p in parts):
                found[name] = tuple(int(p) for p in parts) if len(parts) > 1 else int(value)
        return found
    namespace = {"__builtins__": SAFE_BUILTINS}
    for node in ast.parse(text).body:
        if isinstance(node, ast.AnnAssign) and node.value is not None:
            targets = [node.target]
        elif isinstance(node, ast.Assign):
            targets = node.targets
        else:
            continue
        if all(isinstance(t, ast.Name) and t.id.isupper() for t in targets):
            code = compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec")
            with contextlib.suppress(AttributeError, KeyError, NameError, TypeError, ValueError):
                exec(code, namespace)  # noqa: S102
    namespace.pop("__builtins__")
    return namespace


def norm(entry: dict, key) -> str:
    text = str(key)
    prefix = entry.get("prefix", "")
    if prefix and text.startswith(prefix) and text[len(prefix):].isdigit():
        return text[len(prefix):]
    return text


def budget_set(value) -> frozenset[int]:
    values = value if isinstance(value, (list, tuple, set, frozenset)) else [value]
    return frozenset(int(v) for v in values)


def runner_budgets(entry: dict) -> dict[str, frozenset[int]] | None:
    spec = entry["budgets"]
    if "csv" in spec:
        with (ROOT / spec["csv"]).open(newline="") as handle:
            return {norm(entry, r[spec["key"]]): budget_set(r[spec["value"]]) for r in csv.DictReader(handle)}
    if "source" not in spec:
        return None
    found = constants(ROOT / spec["source"])
    assert spec["name"] in found, f"{spec['source']} defines no {spec['name']}"
    value = found[spec["name"]]
    field = spec.get("field")
    keys = spec.get("keys")
    if isinstance(keys, str):
        keys = constants(ROOT / spec.get("keys_source", spec["source"]))[keys]
    if isinstance(value, dict):
        table = {norm(entry, k): budget_set(v[field] if field is not None else v) for k, v in value.items()}
        return {k: v for k, v in table.items() if keys is None or k in {norm(entry, n) for n in keys}}
    if field is not None:
        return {norm(entry, row[0]): budget_set(row[field]) for row in value}
    return {norm(entry, k): budget_set(value) for k in keys}


def suite_of(entry: dict):
    path, _, cls = entry["asv"].partition("::")
    return ROOT / path, cls or None


def load_suite(path: Path):
    name = "asv_suite_" + re.sub(r"\W", "_", str(path.relative_to(ROOT)))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def suite_budgets(entry: dict) -> dict[str, frozenset[int]]:
    path, cls = suite_of(entry)
    module = load_suite(path)
    owner = getattr(module, cls) if cls else module
    table = getattr(owner, "PAPER_BUDGETS", None)
    if table is None:
        table = getattr(module, "PAPER_BUDGETS", None)
    assert isinstance(table, dict) and table, f"{entry['asv']} has no PAPER_BUDGETS"
    return {norm(entry, k): budget_set(v) for k, v in table.items()}


def read_rows(path: Path) -> list[dict[str, str]]:
    lines = [line for line in path.read_text().splitlines() if not line.startswith("#")]
    return list(csv.DictReader(lines, delimiter="\t" if path.suffix == ".tsv" else ","))


def runner_text(entry: dict) -> str:
    return "\n".join((ROOT / path).read_text() for path in entry["runners"])


def skip_gap(entry: dict) -> None:
    if "gap" in entry:
        pytest.skip(entry["gap"])


def test_every_rule_is_classified():
    rules = snakefile_rules()
    assert len(rules) == len(set(rules))
    assert len(RULES) == len(set(RULES))
    not_h2h = set(REGISTRY["not_h2h"])
    assert not set(RULES) & not_h2h
    assert set(rules) - set(RULES) - not_h2h == set(), "classify the new rules in bench/h2h.json"
    assert (set(RULES) | not_h2h) - set(rules) == set(), "bench/h2h.json names rules the Snakefile lacks"
    assert all(REGISTRY["not_h2h"].values())


@by_rule
def test_files_exist(entry):
    path, _ = suite_of(entry)
    for name in [*entry["runners"], *(r["path"] for r in entry["rows"])]:
        assert (ROOT / name).is_file(), name
    assert path.is_file(), path
    assert path.parent.name == "benchmarks"
    assert (path.parent / "__init__.py").is_file(), "asv refuses a benchmarks directory without __init__.py"
    config = json.loads((path.parent.parent / "asv.conf.json").read_text())
    assert config["benchmarks"] == "benchmarks"
    assert entry["competitors"]


@pytest.mark.parametrize("path", sorted({suite_of(e)[0] for e in ENTRIES}), ids=lambda p: str(p.relative_to(ROOT)))
def test_suite_imports_and_has_a_smoke(path):
    module = load_suite(path)
    smokes = [v for v in vars(module).values() if isinstance(v, type) and hasattr(v, "time_smoke")]
    assert smokes, f"{path} has no class with time_smoke"
    assert all(not getattr(cls, "params", None) for cls in smokes)


@by_rule
def test_paper_budgets(entry):
    spec = entry["budgets"]
    if "paired" in spec:
        assert spec["paired"]
        return
    suite = suite_budgets(entry)
    runner = runner_budgets(entry)
    if runner is not None:
        assert suite == runner, f"{entry['asv']} PAPER_BUDGETS differ from {spec}"
    uncapped = set(entry.get("uncapped_arms", {}))
    checked = 0
    for table in entry["rows"]:
        if "budget" not in table:
            continue
        for row in read_rows(ROOT / table["path"]):
            if table.get("arm") and row[table["arm"]] in uncapped:
                continue
            key = norm(entry, row[table["key"]])
            assert key in suite, f"{table['path']}: {key} has no paper budget"
            assert int(float(row[table["budget"]])) in suite[key], f"{table['path']}: {row}"
            checked += 1
    assert checked or runner is not None, "nothing ties the suite budgets to a runner or to rows"


@by_rule
def test_seeds(entry):
    if "seeds" in entry:
        seeds = constants(ROOT / entry["seeds"]["source"])[entry["seeds"]["name"]]
        tables = seeds.values() if isinstance(seeds, dict) else [seeds]
        fewest = min(len(set(table)) for table in tables)
        assert fewest >= MIN_SEEDS, f"{entry['seeds']} has {fewest} seeds"
    counts: dict[tuple[str, str], int] = {}
    for table in entry["rows"]:
        groups: dict[tuple[str, str, str], set | int] = {}
        for row in read_rows(ROOT / table["path"]):
            key = norm(entry, row[table["key"]]) if "key" in table else "-"
            arm = row[table["arm"]] if "arm" in table else "-"
            if table.get("keys") and key not in {norm(entry, k) for k in table["keys"]}:
                continue
            if table.get("arms") and arm not in table["arms"]:
                continue
            group = (key, arm, row[table["group"]] if "group" in table else "-")
            if "seed" in table:
                if row[table["seed"]]:
                    groups.setdefault(group, set()).add(row[table["seed"]])
                else:
                    groups.setdefault(group, set())
            elif "n" in table:
                groups[group] = int(row[table["n"]])
        for (key, arm, _), value in groups.items():
            n = len(value) if isinstance(value, set) else value
            counts[(key, arm)] = min(n, counts.get((key, arm), n))
        for key in table.get("keys", []):
            for arm in table.get("arms", []):
                counts.setdefault((norm(entry, key), arm), 0)
    if not counts:
        assert "seeds" in entry, "no seed count in the rows and no SEEDS in the runner"
        return
    short = {key: n for key, n in counts.items() if n < MIN_SEEDS}
    known = {(norm(entry, s["key"]), s["arm"]): s["n"] for s in entry.get("short", [])}
    assert all(s["reason"] for s in entry.get("short", []))
    assert short == known, "seed shortfalls differ from the ones bench/h2h.json records"


@by_rule
def test_serial_on_one_core(entry):
    skip_gap(entry)
    assert PINNED.search(runner_text(entry)), f"{entry['runners']} never pins a job with taskset -c"


@by_rule
def test_rows_carry_provenance(entry):
    skip_gap(entry)
    text = runner_text(entry)
    assert "code_revision" in text and "host" in text, f"{entry['runners']} do not write code_revision and host"


def test_snakemake_resolves_every_head_to_head():
    snakemake = os.environ.get("SNAKEMAKE") or shutil.which("snakemake")
    if not snakemake:
        pytest.skip("snakemake is not on PATH; set SNAKEMAKE")
    proc = subprocess.run(
        [snakemake, "-n", "-s", "workflow/Snakefile", *RULES],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    missing = {
        path
        for listed in re.findall(r"reason: Missing output files: (.*)", proc.stdout + proc.stderr)
        for path in listed.split(", ")
    }
    known = {path for entry in ENTRIES for path in entry.get("missing_outputs", {})}
    assert missing == known, "outputs missing from the tree differ from bench/h2h.json missing_outputs"


@pytest.mark.parametrize("path", sorted({suite_of(e)[0] for e in ENTRIES}), ids=lambda p: str(p.relative_to(ROOT)))
def test_smoke(path):
    if os.environ.get("ANNEAL_H2H_SMOKE") != "1":
        pytest.skip("set ANNEAL_H2H_SMOKE=1 to build and run the smoke seeds")
    module = load_suite(path)
    for cls in [v for v in vars(module).values() if isinstance(v, type) and hasattr(v, "time_smoke")]:
        bench = cls()
        try:
            if hasattr(bench, "setup"):
                bench.setup()
            bench.time_smoke()
        except NotImplementedError as missing:
            pytest.skip(str(missing))
