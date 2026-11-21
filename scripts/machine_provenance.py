#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Stable physical-machine provenance for benchmark row writers."""

from __future__ import annotations

import csv
import hashlib
import io
import os
from collections.abc import Iterable, Mapping
from pathlib import Path

MACHINE_ID_SALT = b"anneal-bench-machine-id-v1\0"
MACHINE_ID_PATH = Path("/etc/machine-id")
PRODUCT_UUID_PATH = Path("/sys/class/dmi/id/product_uuid")
CPUINFO_PATH = Path("/proc/cpuinfo")
MACHINE_FIELDS = ("machine_id", "cpu_model")


def machine_provenance(
    machine_id_path: Path = MACHINE_ID_PATH,
    product_uuid_path: Path = PRODUCT_UUID_PATH,
    cpuinfo_path: Path = CPUINFO_PATH,
) -> tuple[str, str]:
    """Return a salted machine digest and the normalized CPU model."""
    identifier = ""
    for path in (machine_id_path, product_uuid_path):
        if path.is_file():
            identifier = path.read_text().strip()
            if identifier:
                break
    if not identifier:
        raise RuntimeError(
            "neither /etc/machine-id nor the DMI product_uuid is available"
        )

    cpu_model = ""
    if cpuinfo_path.is_file():
        fields: dict[str, str] = {}
        for line in cpuinfo_path.read_text(errors="replace").splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                fields.setdefault(key.strip(), " ".join(value.split()))
        for key in ("model name", "Model", "Processor", "Hardware"):
            if fields.get(key):
                cpu_model = fields[key]
                break
    if not cpu_model:
        raise RuntimeError("/proc/cpuinfo has no CPU model string")

    digest = hashlib.sha256(MACHINE_ID_SALT + identifier.encode()).hexdigest()
    return digest, cpu_model


def machine_columns() -> dict[str, str]:
    """Return the machine columns for one row."""
    machine_id, cpu_model = machine_provenance()
    return {"machine_id": machine_id, "cpu_model": cpu_model}


def with_machine_provenance(row: Mapping[str, object]) -> dict[str, object]:
    """Copy one row and add the required machine columns."""
    return {**row, **machine_columns()}


def machine_fieldnames(fieldnames: Iterable[str]) -> list[str]:
    """Append the machine columns once, preserving existing field order."""
    fields = list(fieldnames)
    fields.extend(field for field in MACHINE_FIELDS if field not in fields)
    return fields


def aggregate_machine_columns(rows: Iterable[Mapping[str, object]]) -> dict[str, str]:
    """Aggregate machine provenance, preserving wholly legacy row groups."""
    materialized = list(rows)
    present = [
        bool(str(row.get("machine_id", ""))) and bool(str(row.get("cpu_model", "")))
        for row in materialized
    ]
    if not any(present):
        return {}
    if not all(present):
        raise RuntimeError("cannot mix legacy and machine-provenance rows")
    return {
        "machine_id": ";".join(
            sorted({str(row["machine_id"]) for row in materialized})
        ),
        "cpu_model": ";".join(
            sorted({str(row["cpu_model"]) for row in materialized})
        ),
    }


def csv_fields(values: Iterable[object]) -> str:
    """Encode fields for safe insertion after an existing CSV prefix."""
    stream = io.StringIO()
    csv.writer(stream, lineterminator="").writerow(values)
    return stream.getvalue()


def environment() -> dict[str, str]:
    """Read an inherited identity, falling back to this machine."""
    machine_id = os.environ.get("ANNEAL_MACHINE_ID", "")
    cpu_model = os.environ.get("ANNEAL_CPU_MODEL", "")
    if bool(machine_id) != bool(cpu_model):
        raise RuntimeError("incomplete ANNEAL_MACHINE_ID/ANNEAL_CPU_MODEL hand-off")
    if not machine_id:
        machine_id, cpu_model = machine_provenance()
    return {"machine_id": machine_id, "cpu_model": cpu_model}


def require_csv_schema(path: Path, fieldnames: Iterable[str]) -> None:
    """Refuse to append a new-schema row beneath a legacy CSV header."""
    if not path.is_file() or path.stat().st_size == 0:
        return
    with path.open(newline="") as handle:
        actual = next(csv.reader(handle), [])
    expected = list(fieldnames)
    if actual != expected:
        raise RuntimeError(
            f"{path} has legacy columns; write new benchmark rows to a new table"
        )
