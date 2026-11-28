# Copyright (c) anneal developers
"""Check consolidated G2 budget rows against their cell manifest.

Each shapes line keeps every measured field. The host identity field is an
opaque label, host-1 onward, assigned by first appearance. The manifest
stores, per cell, the sha256 of the original csv line and the shapes line
with that field removed, taken before the label was written.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from pathlib import Path

SYSTEMS = (
    ("lj38", 36, ("anneal", "ase_bh", "ase_mh")),
    ("lj55", 36, ("anneal", "ase_bh", "ase_mh")),
    ("lj98", 36, ("anneal", "ase_bh", "ase_mh")),
    ("lj104", 36, ("anneal", "ase_bh", "ase_mh")),
    ("au55", 36, ("anneal", "ase_bh", "ase_mh")),
    ("lj75", 12, ("ase_mh",)),
    ("lj150", 12, ("ase_mh",)),
)
SEEDS = tuple(range(2000, 2012))
HEADER = (
    "system,arm,seed,budget,stop,calls_reached,fraction_of_budget,"
    "best_e,cpu_s,wall_s,code_revision"
)
CODE_REVISION = "1308d8678a8912de95ded423291a25430bb073fc"
FORBIDDEN = "machine_id_sha256"
NOTE = (
    "Copyright (c) anneal developers\n"
    "\n"
    "The shapes lines beside this note are the single-seed records for this\n"
    "system. The host identity field was replaced by opaque labels (host-1,\n"
    "host-2, and so on) assigned by first appearance, so rows from the same\n"
    "host stay grouped. The CPU model string is unchanged.\n"
)
_STRING_FIELD = re.compile(r'"([^"\\]+)": "(?:[^"\\]|\\.)*"')


def strip_field(line, key):
    """Drop one string field and its adjacent comma, leaving the other bytes."""
    match = None
    for found in _STRING_FIELD.finditer(line):
        if found.group(1) == key:
            match = found
            break
    if match is None:
        raise ValueError(f"missing {key}")
    start, end = match.span()
    if start >= 2 and line[start - 2 : start] == ", ":
        start -= 2
    elif line[end : end + 2] == ", ":
        end += 2
    else:
        raise ValueError(f"separator around {key}")
    return line[:start] + line[end:]


def cell_sha256(csv_line, shapes_line, host_key):
    """Sha256 of the csv line plus the shapes line without its host field."""
    body = csv_line + "\n" + strip_field(shapes_line, host_key) + "\n"
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def resolve_root(explicit=None):
    if explicit is not None:
        return Path(explicit)
    env = os.environ.get("G2_BUDGET_ROOT")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[1]


def present_systems(root):
    found = []
    for system, _count, _arms in SYSTEMS:
        path = root / "bench" / f"ase-{system}" / "rows-ljfix-budget.csv"
        if path.is_file():
            found.append(system)
    return found


def _read_lines(path):
    data = path.read_bytes()
    if b"\r" in data:
        raise ValueError(f"{path.name} has CR")
    if not data.endswith(b"\n") or data.endswith(b"\n\n"):
        raise ValueError(f"{path.name} newline")
    return data.decode("utf-8").splitlines()


def _check_system(root, system, count, arms):
    errors = []
    directory = root / "bench" / f"ase-{system}"
    files = {
        "csv": directory / "rows-ljfix-budget.csv",
        "shapes": directory / "shapes-ljfix-budget.jsonl",
        "manifest": directory / "rows-ljfix-budget.sha256",
        "note": directory / "rows-ljfix-budget.README",
    }
    for kind, path in files.items():
        if not path.is_file():
            errors.append(f"{system}: missing {kind}")
            return errors
        blob = path.read_bytes()
        if FORBIDDEN.encode("utf-8") in blob:
            errors.append(f"{system}: {kind} contains {FORBIDDEN}")
    if files["note"].read_text(encoding="utf-8") != NOTE:
        errors.append(f"{system}: host-label note mismatch")
    try:
        csv_lines = _read_lines(files["csv"])
        shape_lines = _read_lines(files["shapes"])
        manifest_lines = _read_lines(files["manifest"])
    except ValueError as exc:
        errors.append(f"{system}: {exc}")
        return errors
    if not csv_lines or csv_lines[0] != HEADER or csv_lines.count(HEADER) != 1:
        errors.append(f"{system}: csv header")
    data = csv_lines[1:]
    expected = [(arm, seed) for arm in arms for seed in SEEDS]
    if len(data) != count or len(data) != len(expected):
        errors.append(f"{system}: row count {len(data)} != {count}")
    if len(shape_lines) != len(data) or len(manifest_lines) != len(data) + 1:
        errors.append(f"{system}: shapes or manifest length")
    if manifest_lines and manifest_lines[0] != "arm,seed,sha256":
        errors.append(f"{system}: manifest header")
    parsed = []
    seen = {}
    for line in data:
        try:
            row = next(csv.reader([line]))
        except csv.Error as exc:
            errors.append(f"{system}: csv {exc}")
            continue
        if len(row) != 11 or row[0] != system:
            errors.append(f"{system}: csv columns")
            continue
        arm, seed_text = row[1], row[2]
        try:
            seed = int(seed_text)
        except ValueError:
            errors.append(f"{system}: seed {seed_text}")
            continue
        key = (arm, seed)
        if key in seen:
            errors.append(f"{system}: duplicate {arm} {seed}")
        seen[key] = line
        if row[10] != CODE_REVISION:
            errors.append(f"{system}: code_revision {arm} {seed}")
        parsed.append(key)
    if parsed != expected:
        errors.append(f"{system}: row order")
    if set(seen) != set(expected):
        errors.append(f"{system}: cell set")
    manifest = {}
    for line in manifest_lines[1:]:
        arm, seed_text, digest = line.split(",")
        manifest[(arm, int(seed_text))] = digest
    if list(manifest) != expected:
        errors.append(f"{system}: manifest order")
    hosts = []
    for index, key in enumerate(expected):
        if key not in seen:
            continue
        shape = shape_lines[index] if index < len(shape_lines) else ""
        if FORBIDDEN in shape:
            errors.append(f"{system}: forbidden key {key[0]} {key[1]}")
            continue
        try:
            obj = json.loads(shape)
        except json.JSONDecodeError:
            errors.append(f"{system}: shapes json {key[0]} {key[1]}")
            continue
        if obj.get("system") != system or obj.get("arm") != key[0]:
            errors.append(f"{system}: shapes identity {key[0]} {key[1]}")
        if str(obj.get("seed")) != str(key[1]):
            errors.append(f"{system}: shapes seed {key[0]} {key[1]}")
        if obj.get("code_revision") != CODE_REVISION:
            errors.append(f"{system}: shapes revision {key[0]} {key[1]}")
        label = obj.get("host")
        if not isinstance(label, str) or re.fullmatch(r"host-[1-9][0-9]*", label) is None:
            errors.append(f"{system}: host label {key[0]} {key[1]}")
        else:
            hosts.append(int(label.split("-", 1)[1]))
        if "model_name" not in obj or not obj["model_name"]:
            errors.append(f"{system}: cpu model {key[0]} {key[1]}")
        try:
            digest = cell_sha256(seen[key], shape, "host")
        except ValueError as exc:
            errors.append(f"{system}: {exc} {key[0]} {key[1]}")
            continue
        if manifest.get(key) != digest:
            errors.append(f"{system}: sha256 {key[0]} {key[1]}")
    return errors, hosts


def check(root, systems=None):
    root = Path(root)
    present = present_systems(root)
    if systems is None:
        systems = present
    else:
        missing = [system for system in systems if system not in present]
        if missing:
            return [f"absent: {','.join(missing)}"]
    if not systems:
        return ["no consolidated rows"]
    known = {system: (count, arms) for system, count, arms in SYSTEMS}
    errors = []
    hosts = []
    for system in systems:
        if system not in known:
            errors.append(f"unknown system {system}")
            continue
        count, arms = known[system]
        result = _check_system(root, system, count, arms)
        if result and isinstance(result[0], str):
            errors.extend(result)
            continue
        system_errors, system_hosts = result
        errors.extend(system_errors)
        hosts.extend(system_hosts)
    if hosts:
        found = set(hosts)
        span = set(range(1, max(hosts) + 1))
        if found != span:
            errors.append(f"host labels {sorted(found)} not 1..{max(hosts)}")
    return errors


def main(argv=None):
    parser = argparse.ArgumentParser(description="Check consolidated G2 budget rows.")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--systems", default=None, help="comma-separated subset")
    args = parser.parse_args(argv)
    systems = None
    if args.systems:
        systems = [item for item in args.systems.split(",") if item]
    root = resolve_root(args.root)
    errors = check(root, systems)
    checked = systems if systems is not None else present_systems(root)
    print("checked", ",".join(checked) if checked else "-")
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
