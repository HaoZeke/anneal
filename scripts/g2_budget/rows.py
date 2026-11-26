# Copyright (c) anneal developers
"""Append one pre-registered G2 budget-stop row and its run shape."""

from __future__ import annotations

import csv
import importlib.util
import io
import json
import math
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

ROW_FIELDS = (
    "system",
    "arm",
    "seed",
    "budget",
    "stop",
    "calls_reached",
    "fraction_of_budget",
    "best_e",
    "cpu_s",
    "wall_s",
    "code_revision",
)
TERMINAL_STOPS = frozenset({"hit", "budget", "cap"})
CPU_WALL_TOL_S = 1e-6


def _protocol():
    name = "g2_budget_protocol"
    path = Path(__file__).resolve().with_name("protocol.py")
    loaded = sys.modules.get(name)
    if loaded is not None and Path(loaded.__file__).resolve() == path:
        return loaded
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


protocol = _protocol()


@dataclass(frozen=True)
class Segment:
    calls_reached: int
    best_e: float
    cpu_s: float
    wall_s: float
    stop: str


def calls_to_hit(row: dict) -> int:
    """Calls used by a calls-to-hit statistic.

    A hit contributes the call that found it. A cap, a budget stop, and
    any other non-hit contribute one more than the budget.
    """
    budget = int(row["budget"])
    if budget <= 0:
        raise ValueError("budget must be positive")
    if row["stop"] == "hit":
        return int(row["calls_reached"])
    return budget + 1


def fraction_text(calls: int, budget: int) -> str:
    text = format(Decimal(calls) / Decimal(budget), "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def _integer(value, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _seconds(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number of seconds")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{label} must be a non-negative finite number")
    return number


def build_row(
    cell,
    segments: list[Segment],
    revision: str,
    *,
    production: bool = True,
) -> dict[str, str]:
    if not segments:
        raise ValueError("a seed needs a segment")
    if len(segments) > 1 and cell.stop_rule != "cap":
        raise ValueError("checkpoint/resume is only allowed for capped seeds")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("revision must be the full git hash")
    budget = _integer(cell.budget, "budget")
    if budget <= 0:
        raise ValueError("budget must be positive")
    previous = -1
    for index, segment in enumerate(segments):
        calls = _integer(segment.calls_reached, "calls_reached")
        if calls < previous:
            raise ValueError("calls moved backwards")
        previous = calls
        if not math.isfinite(segment.best_e):
            raise ValueError("best energy is not finite")
        _seconds(segment.cpu_s, "cpu_s")
        _seconds(segment.wall_s, "wall_s")
        terminal = index == len(segments) - 1
        if terminal and segment.stop not in TERMINAL_STOPS:
            raise ValueError("the last segment needs a terminal stop")
        if not terminal and segment.stop != "checkpoint":
            raise ValueError("an intermediate segment must be a checkpoint")
    last = segments[-1]
    calls = last.calls_reached
    stop = last.stop
    cpu = math.fsum(segment.cpu_s for segment in segments)
    wall = math.fsum(segment.wall_s for segment in segments)
    if stop == "cap":
        if cell.stop_rule != "cap":
            raise ValueError("a budget-stopped seed cannot stop at the cap")
        if not 0 <= calls < budget:
            raise ValueError("a cap stop is short of the budget")
        if production and cpu < protocol.CPU_CAP_S:
            raise ValueError("a production cap stop records at least the pre-registered CPU cap")
        if production and cpu > wall + CPU_WALL_TOL_S:
            raise ValueError("cpu seconds exceed wall seconds")
    elif stop == "budget":
        if cell.arm == "anneal":
            allowed = budget - protocol.ANNEAL_STEP_CALLS <= calls <= budget
        else:
            allowed = calls == budget
        if not allowed:
            raise ValueError("a budget stop spends the call budget")
    elif stop == "hit":
        if not 0 < calls <= budget:
            raise ValueError("a hit records the call that found it")
    return {
        "system": cell.system,
        "arm": cell.arm,
        "seed": str(cell.seed),
        "budget": str(budget),
        "stop": stop,
        "calls_reached": str(calls),
        "fraction_of_budget": fraction_text(calls, budget),
        "best_e": format(min(segment.best_e for segment in segments), ".8f"),
        "cpu_s": format(cpu, ".6f"),
        "wall_s": format(wall, ".6f"),
        "code_revision": revision,
    }


def _complete_bytes(path: Path) -> bytes:
    content = path.read_bytes() if path.exists() else b""
    if content and not content.endswith(b"\n"):
        newline = content.rfind(b"\n")
        content = content[:newline + 1] if newline >= 0 else b""
    return content


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _encode_row(row: dict[str, str]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=ROW_FIELDS, lineterminator="\n")
    writer.writerow(row)
    return buffer.getvalue()


def _read_rows(path: Path) -> tuple[str, list[dict[str, str]]]:
    text = _complete_bytes(path).decode()
    if not text:
        return "", []
    if text.splitlines()[0] != ",".join(ROW_FIELDS):
        raise ValueError("header does not match the pre-registered columns")
    return text, list(csv.DictReader(io.StringIO(text)))


def _read_shapes(path: Path) -> tuple[str, list[dict]]:
    text = _complete_bytes(path).decode()
    return text, [json.loads(line) for line in text.splitlines() if line]


def _same_row(stored: dict, row: dict[str, str]) -> bool:
    return all(str(stored[field]) == row[field] for field in ROW_FIELDS)


def _shape_matches(stored: dict, row: dict[str, str], segments: int) -> bool:
    try:
        return _same_row(stored, row) and int(stored["segments"]) == segments
    except (KeyError, TypeError, ValueError):
        return False


def _key(record: dict) -> tuple[str, str, str]:
    return (str(record["system"]), str(record["arm"]), str(record["seed"]))


def _shape_body(row: dict[str, str], segments: int, shape_extra) -> dict:
    shape = dict(row)
    shape["segments"] = segments
    if shape_extra:
        overlap = set(shape_extra).intersection(ROW_FIELDS)
        if overlap or "segments" in shape_extra:
            raise ValueError("shape metadata collides with the row")
        shape.update(shape_extra)
    return shape


def _with_amendment(cell, shape_extra):
    if cell.system not in protocol.AMENDMENT_SYSTEMS:
        return shape_extra
    extra = dict(shape_extra or {})
    extra["amendment"] = protocol.AMENDMENT_ID
    return extra


def _commit(root: Path, cell, row: dict[str, str], segments: int, shape_extra=None) -> None:
    """Write the shape first and the row second.

    A retry that finds the shape and not the row writes the row when the
    shape is the same record, and rejects the seed when it is not.
    """
    shape_extra = _with_amendment(cell, shape_extra)
    rows_path = protocol.rows_path(root, cell.system)
    shapes_path = protocol.shapes_path(root, cell.system)
    if rows_path.name != protocol.ROWS_NAME or protocol.is_scope1_rows(rows_path):
        raise ValueError(rows_path.name)
    if shapes_path.name != protocol.SHAPES_NAME:
        raise ValueError(shapes_path.name)
    rows_text, stored_rows = _read_rows(rows_path)
    shapes_text, stored_shapes = _read_shapes(shapes_path)
    key = _key(row)
    old_rows = [item for item in stored_rows if _key(item) == key]
    old_shapes = [item for item in stored_shapes if _key(item) == key]
    if len(old_rows) > 1 or len(old_shapes) > 1:
        raise ValueError("seed has more than one record")
    old_row = old_rows[0] if old_rows else None
    old_shape = old_shapes[0] if old_shapes else None
    row_matches = old_row is not None and _same_row(old_row, row)
    shape_matches = old_shape is not None and _shape_matches(old_shape, row, segments)
    if old_row is not None and old_shape is not None:
        if row_matches and shape_matches:
            raise ValueError("seed already has a row")
        raise ValueError("stored row and shape disagree")
    if old_shape is not None and old_row is None:
        if not shape_matches:
            raise ValueError("shape does not match the row")
        if not rows_text:
            rows_text = ",".join(ROW_FIELDS) + "\n"
        _atomic_write(rows_path, (rows_text + _encode_row(row)).encode())
        return
    if old_row is not None and old_shape is None:
        if not row_matches:
            raise ValueError("row does not match the shape")
        shape = _shape_body(row, segments, shape_extra)
        _atomic_write(shapes_path, (shapes_text + json.dumps(shape, sort_keys=True) + "\n").encode())
        return
    if not rows_text:
        rows_text = ",".join(ROW_FIELDS) + "\n"
    shape = _shape_body(row, segments, shape_extra)
    _atomic_write(shapes_path, (shapes_text + json.dumps(shape, sort_keys=True) + "\n").encode())
    _atomic_write(rows_path, (rows_text + _encode_row(row)).encode())


def append_result(
    root: Path,
    cell,
    segments: list[Segment],
    revision: str,
    shape_extra=None,
) -> dict[str, str]:
    row = build_row(cell, segments, revision, production=True)
    _commit(root, cell, row, len(segments), shape_extra)
    return row


def record_dry_run(root: Path, cell, segments: list[Segment], revision: str) -> dict[str, str]:
    row = build_row(cell, segments, revision, production=False)
    _commit(root, cell, row, len(segments))
    return row
