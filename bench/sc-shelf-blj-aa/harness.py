#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Three-arm outside-time gate for the binary Lennard-Jones cells.

The decision rule is stated in README.md. This file runs that rule and
does not choose another one after the rows exist.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import threading
import time
from collections import Counter

TREE_A = "cff796c507a8fa1d0bfb28888c3570b61988055e"
TREE_B = "809f59b4670937fb8e048739cba6c74b28ff5800"
HIT_TOL = 0.001
STALL_S = 30 * 60
SEED_LO = 2000
SEED_HI = 2023

CELLS = (
    ("blj115n38", 400_000, -187.3676867453),
    ("blj115n55", 1_000_000, -289.2931853058),
    ("blj105n55", 1_000_000, -283.1545313810),
)

COLUMNS = [
    "cell",
    "arm",
    "seed",
    "slot",
    "core",
    "affinity",
    "machine_id_sha256",
    "hit",
    "calls",
    "charged",
    "wall_s",
    "pot_s",
    "outside_s",
    "best",
    "best_bits",
    "library_tree",
    "binary_sha256",
    "session",
    "sequence",
]

HEADER = (
    "system,code,seed,budget,hit,evals_to_hit,charged,"
    "wall_s,best,pot_s,outside_s"
)

THREAD_ONES = (
    "OMP_NUM_THREADS",
    "OMP_THREAD_LIMIT",
    "MKL_NUM_THREADS",
    "MKL_DOMAIN_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "GOTO_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "RAYON_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMBA_NUM_THREADS",
    "TBB_NUM_THREADS",
    "JULIA_NUM_THREADS",
    "POLARS_MAX_THREADS",
)

ARMS = ("A", "A'", "B")
FORWARD = (("A", "A'", "B"), ("A'", "B", "A"), ("B", "A", "A'"))


def order_for(seed: int) -> tuple[str, str, str]:
    k = seed - SEED_LO
    forward = FORWARD[(k // 2) % 3]
    if k % 2:
        return tuple(reversed(forward))
    return forward


def plan() -> list[dict]:
    rows = []
    sequence = 0
    for cell, budget, reference in CELLS:
        for seed in range(SEED_LO, SEED_HI + 1):
            for slot, arm in enumerate(order_for(seed), start=1):
                sequence += 1
                rows.append(
                    {
                        "cell": cell,
                        "budget": budget,
                        "reference": reference,
                        "seed": seed,
                        "slot": slot,
                        "arm": arm,
                        "sequence": sequence,
                    }
                )
    return rows


def type7_median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        raise ValueError("empty sample")
    # Hyndman-Fan type 7 at p = 1/2. h = 1 + (n - 1) / 2.
    h = 1.0 + (n - 1) * 0.5
    floor = math.floor(h)
    gamma = h - floor
    left = ordered[floor - 1]
    if gamma == 0.0:
        return left
    return (1.0 - gamma) * left + gamma * ordered[floor]


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    n = len(values)
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = 0.5 * ((i + 1) + (j + 1))
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def tie_term(values: list[float]) -> float:
    total = 0.0
    for size in Counter(values).values():
        total += size**3 - size
    return total


def mannwhitney_p(x: list[float], y: list[float]) -> float | None:
    """Two-sided asymptotic p-value of the first sample against the second.

    Average ranks, tie correction, continuity correction of one half.
    Returns None when the tie-corrected variance is zero.
    """
    n1 = len(x)
    n2 = len(y)
    if n1 == 0 or n2 == 0:
        return None
    pooled = list(x) + list(y)
    ranks = average_ranks(pooled)
    r1 = math.fsum(ranks[:n1])
    u1 = r1 - n1 * (n1 + 1) / 2.0
    mu = n1 * n2 / 2.0
    n = n1 + n2
    variance = n1 * n2 / 12.0 * ((n + 1) - tie_term(pooled) / (n * (n - 1)))
    if not variance > 0.0:
        return None
    z0 = u1 - mu
    if z0 > 0.0:
        sign = 1.0
    elif z0 < 0.0:
        sign = -1.0
    else:
        sign = 0.0
    z = (z0 - 0.5 * sign) / math.sqrt(variance)
    p = math.erfc(abs(z) / math.sqrt(2.0))
    if p < 0.0:
        return 0.0
    if p > 1.0:
        return 1.0
    return p


def fisher_p(hits_a: int, n_a: int, hits_b: int, n_b: int) -> float:
    """Two-sided exact test on hits and misses with fixed margins."""
    a_obs = hits_a
    row1 = n_a
    row2 = n_b
    col1 = hits_a + hits_b
    n = n_a + n_b
    lo = max(0, col1 - row2)
    hi = min(row1, col1)

    def log_p(a: int) -> float:
        b = row1 - a
        c = col1 - a
        d = n - a - b - c
        return (
            math.lgamma(row1 + 1)
            + math.lgamma(row2 + 1)
            + math.lgamma(col1 + 1)
            + math.lgamma(n - col1 + 1)
            - math.lgamma(a + 1)
            - math.lgamma(b + 1)
            - math.lgamma(c + 1)
            - math.lgamma(d + 1)
            - math.lgamma(n + 1)
        )

    observed = log_p(a_obs)
    logs = [log_p(a) for a in range(lo, hi + 1)]
    # Probability scale, relative to the largest table, so a table counts
    # as at most the observed one when its probability does.
    top = max(logs)
    total = math.fsum(math.exp(item - top) for item in logs)
    extreme = math.fsum(
        math.exp(item - top) for item in logs if item <= observed + 1e-12
    )
    return min(1.0, extreme / total)


def best_bits(token: str) -> str:
    value = float(token)
    import struct

    packed = struct.pack(">d", value)
    return struct.unpack(">Q", packed)[0].to_bytes(8, "big").hex()


def machine_id_sha256() -> str:
    with open("/etc/machine-id", "rb") as handle:
        raw = handle.read()
    if not raw:
        raise SystemExit("machine-id file is empty")
    return hashlib.sha256(raw).hexdigest()


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def child_env(budget: int | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env.pop("H2H_BUDGET", None)
    if budget is not None:
        env["H2H_BUDGET"] = str(budget)
    for name in THREAD_ONES:
        env[name] = "1"
    env["OMP_DYNAMIC"] = "FALSE"
    env["MKL_DYNAMIC"] = "FALSE"
    return env


def cpus_allowed(pid: int) -> str:
    try:
        text = open(f"/proc/{pid}/status", encoding="ascii", errors="replace").read()
    except OSError:
        return ""
    for line in text.splitlines():
        if line.startswith("Cpus_allowed_list:"):
            return line.split(":", 1)[1].strip()
    return ""


def cpu_ticks(pid: int) -> int | None:
    try:
        text = open(f"/proc/{pid}/stat", encoding="ascii", errors="replace").read()
    except OSError:
        return None
    rest = text.rsplit(")", 1)[-1].split()
    # utime and stime are the 14th and 15th fields; field 3 is rest[0].
    return int(rest[11]) + int(rest[12])


def _drain(pipe, sink: list[bytes]) -> None:
    sink.append(pipe.read())
    pipe.close()


def run_task(
    binary: str,
    args: list[str],
    log_path: str | None,
    budget: int | None = None,
) -> tuple[int, str, str, str]:
    """Run one pinned command. Returns exit, stdout, stderr, affinity."""
    command = ["taskset", "-c", "0", binary, *args]
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=child_env(budget),
    )
    out_box: list[bytes] = []
    err_box: list[bytes] = []
    threads = [
        threading.Thread(target=_drain, args=(proc.stdout, out_box), daemon=True),
        threading.Thread(target=_drain, args=(proc.stderr, err_box), daemon=True),
    ]
    for thread in threads:
        thread.start()
    # taskset applies the pin after it starts, so the first sample can
    # still show every core. Keep the first sample that is core 0, and
    # otherwise the last sample seen before the deadline.
    affinity = ""
    sample_until = time.monotonic() + 5.0
    while proc.poll() is None and time.monotonic() < sample_until:
        current = cpus_allowed(proc.pid)
        if current:
            affinity = current
            if current == "0":
                break
        time.sleep(0.005)
    last_cpu = None
    last_change = time.monotonic()
    stalled = False
    while proc.poll() is None:
        if not affinity:
            affinity = cpus_allowed(proc.pid)
        ticks = cpu_ticks(proc.pid)
        now = time.monotonic()
        if ticks is None or ticks != last_cpu:
            last_cpu = ticks
            last_change = now
        elif now - last_change >= STALL_S:
            stalled = True
            proc.kill()
            break
        time.sleep(0.2)
    proc.wait()
    for thread in threads:
        thread.join(timeout=5)
    stdout = b"".join(out_box).decode("utf-8", errors="replace")
    stderr = b"".join(err_box).decode("utf-8", errors="replace")
    code = proc.returncode if proc.returncode is not None else 1
    if stalled:
        code = 124
    if log_path:
        with open(log_path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(f"exit {code}\n")
            if stalled:
                handle.write("stall 1\n")
            handle.write("-- stdout --\n")
            handle.write(stdout)
            if stdout and not stdout.endswith("\n"):
                handle.write("\n")
            handle.write("-- stderr --\n")
            handle.write(stderr)
    return code, stdout, stderr, affinity


def parse_result(stdout: str) -> dict[str, str] | None:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if HEADER not in lines:
        return None
    index = lines.index(HEADER)
    if index + 1 >= len(lines):
        return None
    fields = next(csv.reader([lines[index + 1]]))
    names = HEADER.split(",")
    if len(fields) != len(names):
        return None
    return dict(zip(names, fields))


def blank_row(item: dict, common: dict, affinity: str) -> dict[str, str]:
    row = {key: "" for key in COLUMNS}
    row.update(common)
    row["cell"] = item["cell"]
    row["arm"] = item["arm"]
    row["seed"] = str(item["seed"])
    row["slot"] = str(item["slot"])
    row["core"] = "0"
    row["affinity"] = affinity
    row["hit"] = "0"
    row["sequence"] = str(item["sequence"])
    return row


def measured_row(item: dict, parsed: dict[str, str], common: dict, affinity: str) -> dict[str, str]:
    row = blank_row(item, common, affinity)
    best = parsed["best"]
    value = float(best)
    hit = math.isfinite(value) and value <= item["reference"] + HIT_TOL
    row["hit"] = "1" if hit else "0"
    row["calls"] = parsed["evals_to_hit"]
    row["charged"] = parsed["charged"]
    row["wall_s"] = parsed["wall_s"]
    row["pot_s"] = parsed["pot_s"]
    row["outside_s"] = parsed["outside_s"]
    row["best"] = best
    row["best_bits"] = best_bits(best)
    return row


def load_rows(path: str) -> list[dict[str, str]]:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return []
    with open(path, encoding="utf-8", newline="") as handle:
        text = handle.read()
    if text and not text.endswith("\n"):
        text = text.rsplit("\n", 1)[0] + "\n"
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
    with open(path, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != COLUMNS:
            raise SystemExit("rows header does not match the declared columns")
        return list(reader)


def append_row(path: str, row: dict[str, str]) -> None:
    new_file = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\n")
        if new_file:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def append_json(path: str, payload: dict) -> None:
    line = json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n"
    with open(path, "a", encoding="utf-8", newline="\n") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def session_id(path: str) -> str:
    if os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                if payload.get("event") == "start":
                    return str(payload["session"])
    return hashlib.sha256(os.urandom(32)).hexdigest()[:32]


def check_budget(binary: str, cell: str, budget: int, reference: float) -> None:
    code, stdout, _stderr, _affinity = run_task(binary, ["budget", cell], None)
    if code != 0:
        raise SystemExit(f"budget command failed for {cell} with exit {code}")
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if len(lines) < 2:
        raise SystemExit(f"budget command returned too little text for {cell}")
    if int(lines[0]) != budget:
        raise SystemExit(f"budget mismatch for {cell}")
    if lines[1] != f"{reference:.10f}":
        raise SystemExit(f"reference mismatch for {cell}")


def arm_table(args: argparse.Namespace) -> dict[str, dict[str, str]]:
    found = {
        "A": {"binary": args.arm_a, "library_tree": args.tree_a},
        "A'": {"binary": args.arm_ap, "library_tree": args.tree_ap},
        "B": {"binary": args.arm_b, "library_tree": args.tree_b},
    }
    if found["A"]["library_tree"] != TREE_A or found["A'"]["library_tree"] != TREE_A:
        raise SystemExit("A and A' must both carry the develop library tree")
    if found["B"]["library_tree"] != TREE_B:
        raise SystemExit("B must carry the shelf library tree")
    for arm, meta in found.items():
        if not os.path.isfile(meta["binary"]):
            raise SystemExit(f"missing binary for arm {arm}")
        meta["binary_sha256"] = file_sha256(meta["binary"])
    return found


def command_run(args: argparse.Namespace) -> None:
    if not args.rustc.startswith("rustc 1.88.0"):
        raise SystemExit("rustc must be 1.88.0")
    arms = arm_table(args)
    os.makedirs(args.log_dir, exist_ok=True)
    for meta in arms.values():
        for cell, budget, reference in CELLS:
            check_budget(meta["binary"], cell, budget, reference)
    rows_path = os.path.join(args.out, "rows.csv")
    sessions_path = os.path.join(args.out, "sessions.jsonl")
    os.makedirs(args.out, exist_ok=True)
    done = {(row["cell"], row["arm"], row["seed"]) for row in load_rows(rows_path)}
    sid = session_id(sessions_path)
    machine = machine_id_sha256()
    if not os.path.exists(sessions_path) or os.path.getsize(sessions_path) == 0:
        append_json(
            sessions_path,
            {
                "event": "start",
                "session": sid,
                "machine_id_sha256": machine,
                "rustc": args.rustc,
                "library_tree_a": TREE_A,
                "library_tree_b": TREE_B,
                "binary_sha256": {arm: meta["binary_sha256"] for arm, meta in arms.items()},
                "stall_s": STALL_S,
                "seeds": [SEED_LO, SEED_HI],
                "cells": [cell for cell, _, _ in CELLS],
            },
        )
    for item in plan():
        key = (item["cell"], item["arm"], str(item["seed"]))
        if key in done:
            continue
        meta = arms[item["arm"]]
        common = {
            "machine_id_sha256": machine,
            "library_tree": meta["library_tree"],
            "binary_sha256": meta["binary_sha256"],
            "session": sid,
        }
        log_path = os.path.join(
            args.log_dir,
            f"{item['sequence']:03d}-{item['cell']}-{item['arm'].replace(chr(39), 'p')}-{item['seed']}.log",
        )
        code, stdout, _stderr, affinity = run_task(
            meta["binary"],
            ["run", item["cell"], str(item["seed"])],
            log_path,
        )
        status = "ok"
        if code != 0:
            status = "stall" if code == 124 else "exit"
            row = blank_row(item, common, affinity)
        else:
            parsed = parse_result(stdout)
            if parsed is None:
                status = "parse"
                row = blank_row(item, common, affinity)
            else:
                try:
                    if int(parsed["budget"]) != item["budget"]:
                        status = "budget"
                        row = blank_row(item, common, affinity)
                    elif int(parsed["seed"]) != item["seed"] or parsed["system"] != item["cell"]:
                        status = "identity"
                        row = blank_row(item, common, affinity)
                    else:
                        row = measured_row(item, parsed, common, affinity)
                except (ValueError, KeyError):
                    status = "parse"
                    row = blank_row(item, common, affinity)
        if affinity and affinity != "0":
            status = "affinity"
            row = blank_row(item, common, affinity)
        append_row(rows_path, row)
        print(
            f"{item['sequence']} {item['cell']} {item['arm']} {item['seed']} {status} {row['hit']}",
            flush=True,
        )
        append_json(
            sessions_path,
            {
                "event": "row",
                "session": sid,
                "sequence": item["sequence"],
                "cell": item["cell"],
                "arm": item["arm"],
                "seed": item["seed"],
                "slot": item["slot"],
                "exit": code,
                "status": status,
                "hit": int(row["hit"]),
                "affinity": row["affinity"],
            },
        )
        done.add(key)
    append_json(sessions_path, {"event": "end", "session": sid, "rows": len(done)})


def ratio_of(row: dict[str, str]) -> float | None:
    if row["outside_s"] == "" or row["charged"] == "":
        return None
    try:
        outside = float(row["outside_s"])
        charged = float(row["charged"])
    except ValueError:
        return None
    if not math.isfinite(outside) or not math.isfinite(charged) or charged <= 0.0:
        return None
    return outside / charged


def calls_rank(row: dict[str, str]) -> float:
    if row["hit"] != "1":
        return math.inf
    return float(row["calls"])


def fmt(value: float | None) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return format(value, ".16g")


def command_summarize(args: argparse.Namespace) -> None:
    rows = load_rows(args.rows)
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault((row["cell"], row["arm"]), []).append(row)
    summary_rows = []
    for cell, _budget, _reference in CELLS:
        by_arm = {arm: grouped.get((cell, arm), []) for arm in ARMS}
        medians = {}
        n_ratio = {}
        for arm in ARMS:
            ratios = [value for value in (ratio_of(row) for row in by_arm[arm]) if value is not None]
            n_ratio[arm] = len(ratios)
            medians[arm] = type7_median(ratios) if ratios else None
        notes = []
        unresolved = False
        if any(len(by_arm[arm]) != 24 for arm in ARMS):
            unresolved = True
            notes.append("row-count")
        if any(n_ratio[arm] != 24 for arm in ARMS):
            unresolved = True
            notes.append("ratio-count")
        if not (medians["A"] is not None and medians["A"] > 0.0):
            unresolved = True
            notes.append("base-median")
        ba = apa = None
        if medians["A"] not in (None, 0.0) and medians["B"] is not None and medians["A'"] is not None:
            ba = medians["B"] / medians["A"]
            apa = medians["A'"] / medians["A"]
        samples = {}
        call_samples = {}
        for arm in ARMS:
            samples[arm] = [value for value in (ratio_of(row) for row in by_arm[arm]) if value is not None]
            call_samples[arm] = [calls_rank(row) for row in by_arm[arm]] if len(by_arm[arm]) == 24 else []
        mw = {}
        fisher = {}
        calls_mw = {}
        for arm in ("A'", "B"):
            if len(samples["A"]) == 24 and len(samples[arm]) == 24:
                mw[arm] = mannwhitney_p(samples[arm], samples["A"])
            else:
                mw[arm] = None
            if len(by_arm["A"]) == 24 and len(by_arm[arm]) == 24:
                fisher[arm] = fisher_p(
                    sum(row["hit"] == "1" for row in by_arm["A"]),
                    24,
                    sum(row["hit"] == "1" for row in by_arm[arm]),
                    24,
                )
                calls_mw[arm] = mannwhitney_p(call_samples[arm], call_samples["A"])
            else:
                fisher[arm] = None
                calls_mw[arm] = None
        if mw["B"] is None or not math.isfinite(mw["B"] if mw["B"] is not None else math.nan):
            if not unresolved and ba is not None:
                notes.append("p-undefined")
        if unresolved or ba is None or apa is None or mw["B"] is None:
            verdict = "UNRESOLVED"
        elif ba > 1.10 and 0.95 <= apa <= 1.05 and mw["B"] < 0.05:
            verdict = "REAL"
        else:
            verdict = "NOT-REAL"
        a_by_seed = {row["seed"]: row for row in by_arm["A"]}
        match_bits = {arm: 0 for arm in ARMS}
        match_calls = {arm: 0 for arm in ARMS}
        for arm in ("A'", "B"):
            for row in by_arm[arm]:
                other = a_by_seed.get(row["seed"])
                if other is None:
                    continue
                if row["best_bits"] != "" and row["best_bits"] == other["best_bits"]:
                    match_bits[arm] += 1
                if row["calls"] != "" and row["calls"] == other["calls"]:
                    match_calls[arm] += 1
        if (
            calls_mw.get("B") is None
            and len(by_arm["A"]) == 24
            and len(by_arm["B"]) == 24
        ):
            notes.append("calls-p-undefined")
        note = ",".join(notes)

        def against_a(arm: str) -> float | None:
            if arm == "A" and medians["A"] not in (None, 0.0):
                return 1.0
            if arm == "B":
                return ba
            if arm == "A'":
                return apa
            return None

        for arm in ARMS:
            summary_rows.append(
                {
                    "cell": cell,
                    "arm": arm,
                    "n": str(len(by_arm[arm])),
                    "hits": str(sum(row["hit"] == "1" for row in by_arm[arm])),
                    "n_ratio": str(n_ratio[arm]),
                    "median_outside_per_call": fmt(medians[arm]),
                    "ratio_to_a": fmt(against_a(arm)),
                    "mw_p_vs_a": fmt(mw.get(arm)),
                    "fisher_p_vs_a": fmt(fisher.get(arm)),
                    "calls_mw_p_vs_a": fmt(calls_mw.get(arm)),
                    "best_bits_match": "" if arm == "A" else str(match_bits[arm]),
                    "calls_match": "" if arm == "A" else str(match_calls[arm]),
                    "verdict": verdict,
                    "note": note,
                }
            )
    fields = list(summary_rows[0].keys())
    with open(args.summary, "w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary_rows)
    for row in summary_rows:
        if row["arm"] == "B":
            print(
                "{cell} {verdict} B/A={ratio_to_a} mw={mw_p_vs_a} "
                "fisher={fisher_p_vs_a} calls_mw={calls_mw_p_vs_a} "
                "bits={best_bits_match} calls={calls_match} {note}".format(**row)
            )


def self_test() -> None:
    assert type7_median([float(v) for v in range(1, 25)]) == 12.5
    assert type7_median([1.0, 2.0, 3.0]) == 2.0
    slots = {arm: [0, 0, 0] for arm in ARMS}
    for seed in range(SEED_LO, SEED_HI + 1):
        for slot, arm in enumerate(order_for(seed)):
            slots[arm][slot] += 1
    for arm in ARMS:
        assert slots[arm] == [8, 8, 8], (arm, slots[arm])
    whole = plan()
    assert len(whole) == 216
    assert [item["sequence"] for item in whole] == list(range(1, 217))
    assert mannwhitney_p([1.0] * 8, [1.0] * 8) is None
    separated = mannwhitney_p([1.0, 2, 3, 4], [10.0, 11, 12, 13])
    assert separated is not None and separated < 0.05
    assert abs(fisher_p(12, 24, 12, 24) - 1.0) < 1e-12
    extreme = fisher_p(24, 24, 0, 24)
    assert extreme < 1e-12
    tied = mannwhitney_p([1.0, 1, 2, 3], [1.0, 2, 3, 4])
    assert tied is not None and 0.0 < tied <= 1.0
    ranked = mannwhitney_p([1.0, 2.0, math.inf], [1.0, 2.0, 3.0])
    assert ranked is not None
    print("self-test ok")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("self-test")
    pre = sub.add_parser("preflight")
    run = sub.add_parser("run")
    summary = sub.add_parser("summarize")
    for target in (pre, run):
        target.add_argument("--arm-a", required=True)
        target.add_argument("--arm-ap", required=True)
        target.add_argument("--arm-b", required=True)
        target.add_argument("--tree-a", required=True)
        target.add_argument("--tree-ap", required=True)
        target.add_argument("--tree-b", required=True)
    run.add_argument("--out", required=True)
    run.add_argument("--log-dir", required=True)
    run.add_argument("--rustc", required=True)
    pre.add_argument("--seed", type=int, default=1)
    pre.add_argument("--budget", type=int, default=200)
    summary.add_argument("--rows", required=True)
    summary.add_argument("--summary", required=True)
    return parser


def command_preflight(args: argparse.Namespace) -> None:
    if args.seed >= SEED_LO and args.seed <= SEED_HI:
        raise SystemExit("preflight seed must sit outside the measured range")
    arms = arm_table(args)
    for meta in arms.values():
        for cell, budget, reference in CELLS:
            check_budget(meta["binary"], cell, budget, reference)
        code, stdout, _stderr, affinity = run_task(
            meta["binary"],
            ["run", "blj115n38", str(args.seed)],
            None,
            budget=args.budget,
        )
        if code != 0:
            raise SystemExit("preflight run failed")
        parsed = parse_result(stdout)
        if parsed is None:
            raise SystemExit("preflight output did not parse")
        if int(parsed["budget"]) != args.budget:
            raise SystemExit("preflight did not use the short budget")
        if affinity != "0":
            raise SystemExit("preflight was not pinned")
        _ = best_bits(parsed["best"])
    print("preflight ok")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.cmd == "self-test":
        self_test()
    elif args.cmd == "preflight":
        command_preflight(args)
    elif args.cmd == "run":
        command_run(args)
    elif args.cmd == "summarize":
        command_summarize(args)
    else:
        raise SystemExit("unknown command")


if __name__ == "__main__":
    main()
