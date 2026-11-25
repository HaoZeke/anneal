#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Stable physical-machine provenance for benchmark row writers."""

from __future__ import annotations

import csv
import hashlib
import io
import os
import re
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

MACHINE_ID_SALT = b"anneal-bench-machine-id-v1\0"
MACHINE_ID_PATH = Path("/etc/machine-id")
PRODUCT_UUID_PATH = Path("/sys/class/dmi/id/product_uuid")
CPUINFO_PATH = Path("/proc/cpuinfo")
MACHINE_FIELDS = ("machine_id", "cpu_model")
GMIN_BUILD_KEY_RE = re.compile(r"[0-9a-f]{64}")


class SavedRowProvenanceMismatch(ValueError):
    """Raised when a saved row disagrees with expected producer provenance."""


def _darwin_machine_id() -> str:
    try:
        result = subprocess.run(
            ["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return ""
    if result.returncode != 0:
        return ""
    match = re.search(r'"IOPlatformUUID"\s*=\s*"([^"]+)"', result.stdout)
    return match.group(1).strip() if match else ""


def _darwin_cpu_model() -> str:
    for key in ("machdep.cpu.brand_string", "hw.model"):
        try:
            result = subprocess.run(
                ["sysctl", "-n", key],
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError:
            return ""
        if result.returncode == 0:
            value = " ".join(result.stdout.split())
            if value:
                return value
    return ""


def machine_digest(
    machine_id_path: Path = MACHINE_ID_PATH,
    product_uuid_path: Path = PRODUCT_UUID_PATH,
) -> str:
    """Return the salted SHA-256 of this machine's identifier."""
    identifier = ""
    for path in (machine_id_path, product_uuid_path):
        if path.is_file():
            identifier = path.read_text().strip()
            if identifier:
                break
    if (
        not identifier
        and sys.platform == "darwin"
        and machine_id_path == MACHINE_ID_PATH
        and product_uuid_path == PRODUCT_UUID_PATH
    ):
        identifier = _darwin_machine_id()
    if not identifier:
        raise RuntimeError(
            "neither /etc/machine-id nor the DMI product_uuid is available"
        )
    return hashlib.sha256(MACHINE_ID_SALT + identifier.encode()).hexdigest()


def machine_provenance(
    machine_id_path: Path = MACHINE_ID_PATH,
    product_uuid_path: Path = PRODUCT_UUID_PATH,
    cpuinfo_path: Path = CPUINFO_PATH,
) -> tuple[str, str]:
    """Return a salted machine digest and the normalized CPU model."""
    digest = machine_digest(machine_id_path, product_uuid_path)
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
    if not cpu_model and sys.platform == "darwin" and cpuinfo_path == CPUINFO_PATH:
        cpu_model = _darwin_cpu_model()
    if not cpu_model:
        raise RuntimeError("/proc/cpuinfo has no CPU model string")
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


def require_lf(path: Path) -> None:
    """Refuse a row file that holds a carriage return."""
    if b"\r" in Path(path).read_bytes():
        raise SystemExit(f"{path} contains CR")


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


def validate_saved_row_identity(
    row: Mapping[str, object],
    *,
    path: Path | None = None,
    system: str,
    arm: str,
    seed: int,
    budget: int,
) -> None:
    """Reject a saved per-seed row whose system, arm/code, seed, or budget mismatches."""
    where = f"{path}: " if path is not None else ""
    raw_system = row.get("system")
    actual_system = "" if raw_system is None else str(raw_system).strip()
    if actual_system != system:
        raise ValueError(
            f"{where}saved row system mismatch: got {actual_system!r}, expected {system!r}"
        )
    arm_fields = tuple(field for field in ("code", "arm") if field in row)
    if not arm_fields:
        raise ValueError(
            f"{where}saved row arm mismatch: missing 'code'/'arm', expected {arm!r}"
        )
    arm_values = {
        field: ("" if row[field] is None else str(row[field]).strip())
        for field in arm_fields
    }
    for field, actual_arm in arm_values.items():
        if actual_arm != arm:
            detail = ", ".join(f"{key}={val!r}" for key, val in arm_values.items())
            raise ValueError(
                f"{where}saved row arm mismatch ({field}): got {detail}, expected {arm!r}"
            )
    raw_seed = row.get("seed")
    seed_text = "" if raw_seed is None else str(raw_seed).strip()
    try:
        actual_seed = int(seed_text)
    except ValueError as error:
        raise ValueError(
            f"{where}saved row seed mismatch: got {seed_text!r}, expected {seed}"
        ) from error
    if actual_seed != seed:
        raise ValueError(
            f"{where}saved row seed mismatch: got {actual_seed}, expected {seed}"
        )
    raw_budget = row.get("budget")
    budget_text = "" if raw_budget is None else str(raw_budget).strip()
    try:
        actual_budget = int(budget_text)
    except ValueError as error:
        raise ValueError(
            f"{where}saved row budget mismatch: got {budget_text!r}, expected {budget}"
        ) from error
    if actual_budget != budget:
        raise ValueError(
            f"{where}saved row budget mismatch: got {actual_budget}, expected {budget}"
        )


def require_gmin_build_key(value: object, label: str = "gmin_build_key") -> str:
    """Validate and return a 64-hex GMIN build key, rejecting absent or malformed values."""
    cleaned = "" if value is None else str(value).strip()
    if not GMIN_BUILD_KEY_RE.fullmatch(cleaned):
        raise ValueError(f"missing or invalid {label}: {value!r}")
    return cleaned


def _gmin_pin_module():
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    import gmin_pin

    return gmin_pin


def default_gmin_source_metadata(svnrev: Path) -> Path:
    """Derive `<work>/gmin-source.txt` from `<work>/<src>/wales/GMIN/source/SVNREV`."""
    return _gmin_pin_module().default_source_metadata(svnrev)


def read_built_gmin_provenance(svnrev: Path, metadata: Path) -> tuple[str, str]:
    """Return the verified `(revision, build_key)` pair for a built GMIN binary."""
    return _gmin_pin_module().read_built_provenance(svnrev, metadata)


def read_built_gmin_source(metadata: Path, svnrev: Path) -> tuple[str, str, str]:
    """Return the verified `(url, revision, build_key)` triple for a built GMIN binary."""
    return _gmin_pin_module().read_built_source(metadata, svnrev)


def validate_saved_gmin_provenance(
    row: Mapping[str, object],
    *,
    path: Path | str | None = None,
    revision: str,
    build_key: str,
    url: str = "",
    revision_fields: Sequence[str] = ("code_revision",),
    build_key_field: str = "gmin_build_key",
) -> tuple[str, str]:
    """Ensure a saved GMIN row matches both expected SVN revision and 64-hex build key.

    Caller-configuration errors (missing/invalid expected revision or build key)
    raise plain ``ValueError``; stored-row disagreements raise
    ``SavedRowProvenanceMismatch`` (a ``ValueError`` subclass) naming the field.
    """
    where = f"{path}: " if path is not None else ""
    expected_rev = str(revision).strip()
    if not expected_rev:
        raise ValueError(f"{where}missing expected GMIN code_revision")
    expected_key = require_gmin_build_key(
        build_key, f"{where}expected {build_key_field}"
    )
    for field in revision_fields:
        raw_rev = row.get(field)
        actual_rev = "" if raw_rev is None else str(raw_rev).strip()
        if actual_rev != expected_rev:
            raise SavedRowProvenanceMismatch(
                f"{where}saved row {field} mismatch: "
                f"got {actual_rev!r}, expected {expected_rev!r}"
            )
    if "gmin_revision" not in revision_fields and "gmin_revision" in row:
        raw_extra = row.get("gmin_revision")
        extra_rev = "" if raw_extra is None else str(raw_extra).strip()
        if extra_rev != expected_rev:
            raise SavedRowProvenanceMismatch(
                f"{where}saved row gmin_revision mismatch: "
                f"got {extra_rev!r}, expected {expected_rev!r}"
            )
    if url:
        raw_url = row.get("gmin_url")
        actual_url = "" if raw_url is None else str(raw_url).strip()
        if actual_url != url:
            raise SavedRowProvenanceMismatch(
                f"{where}saved row gmin_url mismatch: "
                f"got {actual_url!r}, expected {url!r}"
            )
    raw_key = row.get(build_key_field)
    actual_key = "" if raw_key is None else str(raw_key).strip()
    if not GMIN_BUILD_KEY_RE.fullmatch(actual_key) or actual_key != expected_key:
        raise SavedRowProvenanceMismatch(
            f"{where}saved row {build_key_field} mismatch: "
            f"got {actual_key!r}, expected {expected_key!r}"
        )
    return expected_rev, expected_key


def gmin_row_key(
    row: Mapping[str, object],
    *,
    arm_field: str = "code",
    gmin_arms: Iterable[str] = ("gmin",),
    revision_field: str = "code_revision",
    build_key_field: str = "gmin_build_key",
) -> tuple[str, str]:
    """Return `(code_revision, gmin_build_key)` for keying stored rows."""
    arm = str(row.get(arm_field) or "").strip()
    rev = str(row.get(revision_field) or "").strip()
    if not rev:
        raise ValueError(f"saved row is missing {revision_field}")
    raw_key = row.get(build_key_field)
    key = "" if raw_key is None else str(raw_key).strip()
    if arm in set(gmin_arms):
        return rev, require_gmin_build_key(key, build_key_field)
    if key:
        raise ValueError(
            f"non-GMIN arm {arm!r} must not carry {build_key_field}: {key!r}"
        )
    return rev, ""


def validate_gmin_group_provenance(
    rows: Iterable[Mapping[str, object]],
    *,
    label: str = "",
    expected_revision: str = "",
    expected_build_key: str = "",
    revision_field: str = "code_revision",
    build_key_field: str = "gmin_build_key",
) -> None:
    """Validate revision and build-key consistency across a GMIN row group.

    Wholly legacy pre-build-key groups (where ``build_key_field`` is absent on
    every row and ``expected_build_key`` is not provided) are permitted for
    offline summarisation of historical committed tables.
    """
    materialized = list(rows)
    if not materialized:
        return
    prefix = f"{label}: " if label else ""
    revs = {
        str(row.get(revision_field) or "").strip()
        for row in materialized
    }
    if "" in revs:
        raise ValueError(f"{prefix}missing {revision_field} on GMIN row")
    if len(revs) > 1:
        raise ValueError(
            f"{prefix}mixed {revision_field} values {sorted(revs)}"
        )
    if expected_revision and next(iter(revs)) != expected_revision:
        raise ValueError(
            f"{prefix}{revision_field} mismatch: "
            f"got {next(iter(revs))!r}, expected {expected_revision!r}"
        )
    has_col = [build_key_field in row for row in materialized]
    if not any(has_col) and not expected_build_key:
        return
    if not all(has_col):
        raise ValueError(
            f"{prefix}cannot mix legacy and {build_key_field} GMIN rows"
        )
    keys = {
        require_gmin_build_key(row.get(build_key_field), f"{prefix}{build_key_field}")
        for row in materialized
    }
    if len(keys) > 1:
        raise ValueError(
            f"{prefix}mixed {build_key_field} values {sorted(keys)}"
        )
    if expected_build_key:
        wanted = require_gmin_build_key(
            expected_build_key, f"{prefix}expected {build_key_field}"
        )
        actual = next(iter(keys))
        if actual != wanted:
            raise ValueError(
                f"{prefix}{build_key_field} mismatch: got {actual!r}, expected {wanted!r}"
            )
