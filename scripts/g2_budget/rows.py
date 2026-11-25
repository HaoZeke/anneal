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


def build_row(cell, segments: list[Segment], revision: str) -> dict[str, str]:
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
    if stop == "cap":
        if cell.stop_rule != "cap":
            raise ValueError("a budget-stopped seed cannot stop at the cap")
        if not 0 <= calls < budget:
            raise ValueError("a cap stop is short of the budget")
    elif stop == "budget":
        if calls != budget:
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
        "best_e": format(last.best_e, ".8f"),
        "cpu_s": format(math.fsum(segment.cpu_s for segment in segments), ".6f"),
        "wall_s": format(math.fsum(segment.wall_s for segment in segments), ".6f"),
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


def _csv_text(path: Path, row: dict[str, str]) -> str:
    if path.name != protocol.ROWS_NAME or protocol.is_scope1_rows(path):
        raise ValueError(path.name)
    header = ",".join(ROW_FIELDS)
    text = _complete_bytes(path).decode()
    if not text:
        text = header + "\n"
    elif text.splitlines()[0] != header:
        raise ValueError("header does not match the pre-registered columns")
    else:
        for old in csv.DictReader(io.StringIO(text)):
            if (old["system"], old["arm"], old["seed"]) == (row["system"], row["arm"], row["seed"]):
                raise ValueError("seed already has a row")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=ROW_FIELDS, lineterminator="\n")
    writer.writerow(row)
    return text + buffer.getvalue()


def _shape_text(path: Path, row: dict[str, str], segments: int) -> str:
    if path.name != protocol.SHAPES_NAME:
        raise ValueError(path.name)
    shape = dict(row)
    shape["segments"] = segments
    text = _complete_bytes(path).decode()
    if text:
        for line in text.splitlines():
            if not line:
                continue
            old = json.loads(line)
            if (old["system"], old["arm"], old["seed"]) == (row["system"], row["arm"], row["seed"]):
                raise ValueError("seed already has a shape")
        if not text.endswith("\n"):
            text += "\n"
    return text + json.dumps(shape, sort_keys=True) + "\n"


def append_result(root: Path, cell, segments: list[Segment], revision: str) -> dict[str, str]:
    row = build_row(cell, segments, revision)
    rows = protocol.rows_path(root, cell.system)
    shapes = protocol.shapes_path(root, cell.system)
    rows_text = _csv_text(rows, row)
    shape_text = _shape_text(shapes, row, len(segments))
    _atomic_write(rows, rows_text.encode())
    _atomic_write(shapes, shape_text.encode())
    return row
