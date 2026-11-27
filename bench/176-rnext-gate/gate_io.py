# Copyright (c) anneal developers
"""Durable row and log writes, and in-process telemetry capture."""

from __future__ import annotations

import csv
import hashlib
import os
import struct
import subprocess
import threading
from pathlib import Path

from gate_panel import ROW_COLUMNS, THREAD_ENV

TELEMETRY_KEYS: tuple[str, ...] = (
    "rss_anon_kb",
    "vm_hwm_kb",
    "basis_walks",
    "basis_evidence",
    "basis_claims",
    "basis_probes",
    "basis_candidates",
    "basis_spent",
    "pad_bytes",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def durable_write_bytes(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    read_back = path.read_bytes()
    digest = sha256_bytes(read_back)
    if digest != sha256_bytes(data):
        raise RuntimeError(f"read-back mismatch for {path.name}")
    return digest


def durable_write_text(path: Path, text: str) -> str:
    if "\r" in text:
        raise RuntimeError("refusing a CR character in a text write")
    return durable_write_bytes(path, text.encode("utf-8"))


def machine_id_sha256() -> str:
    return "unrecorded"


def double_bits_hex(value: float) -> str:
    return "0x" + struct.pack(">d", value).hex()


def child_env(extra_ld: str) -> dict[str, str]:
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "RUST_BACKTRACE": "0",
    }
    env.update(THREAD_ENV)
    if extra_ld:
        env["LD_LIBRARY_PATH"] = extra_ld
    return env


def load_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, str]]) -> str:
    tmp = path.with_name(path.name + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(ROW_COLUMNS), lineterminator="\n", extrasaction="raise"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in ROW_COLUMNS})
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    return sha256_file(path)


def scored_stdout(stdout: str) -> str:
    """Keep only scored driver lines. Drop paths and other chatter."""
    lines = stdout.splitlines()
    kept: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.startswith("system,code,seed,budget,") and index + 1 < len(lines):
            kept.append(line)
            kept.append(lines[index + 1])
            index += 2
            continue
        if line.startswith("h2h_row ") or line.startswith("gate_telemetry "):
            kept.append(line)
        index += 1
    body = "\n".join(kept)
    if body:
        body += "\n"
    return body


def parse_fields(line: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for part in line.split()[1:]:
        key, val = part.split("=", 1)
        fields[key] = val
    return fields


def parse_driver(stdout: str, seeded: bool) -> dict[str, str]:
    if seeded:
        for line in stdout.splitlines():
            if line.startswith("h2h_row "):
                return parse_fields(line)
        raise RuntimeError("missing h2h_row")
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    for index, line in enumerate(lines):
        if line.startswith("system,code,seed,budget,") and index + 1 < len(lines):
            keys = line.split(",")
            vals = lines[index + 1].split(",")
            if len(keys) == len(vals):
                return dict(zip(keys, vals))
    raise RuntimeError("missing csv record")


def parse_telemetry(stdout: str) -> dict[str, str]:
    for line in stdout.splitlines():
        if line.startswith("gate_telemetry "):
            fields = parse_fields(line)
            missing = [key for key in TELEMETRY_KEYS if key not in fields]
            if missing:
                raise RuntimeError("telemetry missing keys")
            return fields
    raise RuntimeError("missing gate_telemetry")


def launch_timed(cmd: list[str], env: dict[str, str]) -> tuple[int, str, int]:
    """Run cmd. Reap it with wait4 so ru_maxrss comes from this process."""
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    out_box: list[bytes] = []
    err_box: list[bytes] = []

    def drain(pipe, box: list[bytes]) -> None:
        box.append(pipe.read())
        pipe.close()

    threads = (
        threading.Thread(target=drain, args=(proc.stdout, out_box)),
        threading.Thread(target=drain, args=(proc.stderr, err_box)),
    )
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    _pid, status, usage = os.wait4(proc.pid, 0)
    exit_code = os.waitstatus_to_exitcode(status)
    stdout = out_box[0].decode("utf-8", errors="replace") if out_box else ""
    return exit_code, stdout, int(usage.ru_maxrss)
