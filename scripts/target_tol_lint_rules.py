# Copyright (c) anneal developers
"""Shared rules for the target-energy hit comparison linter."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ALLOWED_FILES = frozenset(
    {
        "src/target_tol.rs",
        "scripts/target_tol.py",
        "scripts/target_tol_lint_rules.py",
        "tests/target_tol_lint.rs",
        "pytest/test_target_tol.py",
        "tests/target_tol_prefix_fixtures.txt",
    }
)

RUNTIME_LITERAL_ALLOWLIST = frozenset(
    {
        "scripts/occat_h2h/stub.S",
        "bench/lj104-lattice/v2/h2h_cap.f90",
    }
)

SKIP_PARTS = frozenset(
    {"vendor", "target", ".git", ".ci", ".venv", "crates/anneal-python/target"}
)

OPS = ("<=", ">=", "==", "!=", "<", ">")

ENERGY_MARKERS = (
    "energy",
    "best",
    "bf0",
    "bf1",
    "bf2",
    "lowest",
    "total",
    "fv",
    "shelf_energy",
    "value",
    "out.best",
    "self.energy",
    "self.best",
    "seen.lowest",
    "generation_best",
    "reduced_energy",
    "native_energy",
    "best_val",
    "record.energy",
    "run.best",
    "board.best",
    "outcome.best",
    "point.energy",
    "h.energy",
    "occupied",
    "reoccupied",
    "be ",
    "(e ",
    ", e ",
    "mf_hits",
    "hits +=",
    "hit =",
    "<= stop",
    "<= stop)",
    "+ hit_tol",
    "<= stop,",
    "hit==",
    "solved =",
    "reached =",
    "anneal_hits",
    "crest_hits",
)

THRESHOLD_MARKERS = (
    "target",
    "reference",
    "level",
    "stop",
    "limit",
    "thresh",
    "tolerance",
    " self.tol",
    "self.target",
    "self.level",
    "self.meter.level",
    "self.reference",
    "published",
    "global_best",
    "spec.reference",
    "plateau",
    "plateau_target",
    "hit_tol",
    "default_hit_tol",
    "target_tolerance",
    "hit_window",
    " + hit",
    "+ hit",
    "+ tol",
    "+ target",
    "+ reference",
    "+ r ",
    "+ t ",
    "ref +",
    "r +",
)

NON_ENERGY_BEFORE = (
    "distance",
    "connectivity",
    "charged_work",
    "gradient",
    "largest_component",
    "trial -",
    "gmax",
    "worst",
    "bond",
    "covalent",
    "reached -",
    "spread",
    ".dim",
    "abs(stated",
    "abs(lj_energy",
    "abs(row",
    "sum(g <",
    "audit.record",
    "audit.cell",
    "rhat",
    "nsteps",
    "self.nsteps",
    "moves <",
    "moves <",
    "parent +",
    "population",
    "chord[",
    "used <",
    "engine.value",
    "neighbor.",
    "conservation",
    "curvature",
    "needed <=",
    "needed >",
    "coordinates[",
    "candidate_energy",
    "reported_energy",
    "stored -",
    "got -",
    "float(row",
    "marks[0].energy",
    "total /",
    "lower + upper",
    "self.checked",
    "plateau)",
    "ordered[j]",
    "successful[j]",
    "threshold >=",
    "threshold <=",
    "prob.dim",
    "packing_reference",
    "overlap",
    "converged()",
    "family.energy >",
    "operation.set_scale",
    "feedback =",
    "value < self.checked",
    "seen_energy",
    "minimum.energy",
    "step - published",
    "deviation <=",
    "found[",
    "reference[",
    "live_target",
    "target_n",
    "target_curvature",
    "not).abs()",
    "rate <",
    "structure_is_sane",
    "energy_conservation",
    "filter(|(e, _)|",
    "dos.rs",
    "cur == target",
    "rng.random",
)

HIT_LITERAL_RE = re.compile(
    r"""
    (?:
        H2H_TOL\s*=\s*["']?0\.001|
        H2H_TOL\s*=\s*["']?1[eE]-3|
        HIT_TOL\s*=\s*0\.001|
        HIT:\s*f64\s*=\s*0\.001|
        HIT:\s*f64\s*=\s*1[eE]-3|
        HIT_TOL:\s*f64\s*=\s*0\.001|
        HIT_TOL:\s*f64\s*=\s*1[eE]-3|
        TARGET_TOLERANCE:\s*f64\s*=\s*1[eE]-3|
        DEFAULT_HIT_TOL\s*=\s*1[eE]-3|
        DEFAULT_HIT_TOL\s*=\s*0\.001|
        \+\s*0\.001\b|
        ,\s*0\.001\s*\)|
        or\s+0\.001\b|
        target\s*\+\s*0\.001|
        reference\s*\+\s*0\.001
    )
    """,
    re.VERBOSE,
)

HIT_TOL_NAME_RE = re.compile(
    r"(?:H2H_TOL|TARGET_TOL|TARGET_TOLERANCE)\s*=\s*[\"']?(?:0\.001|1(?:\.0)?[eE]-3)",
    re.IGNORECASE,
)


def strip_comments(line: str, ext: str) -> str:
    if ext == "py":
        return _strip_py(line)
    if ext == "rs":
        trimmed = line.lstrip()
        if trimmed.startswith(("///", "//!", "*")):
            return ""
        return line.split("//", 1)[0].strip()
    if ext in {"sh", "f90"}:
        return line.split("#", 1)[0].strip()
    return line.strip()


def _strip_py(line: str) -> str:
    out: list[str] = []
    in_string = False
    quote = ""
    escape = False
    for ch in line:
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == quote:
                in_string = False
            continue
        if ch in "\"'":
            in_string = True
            quote = ch
            continue
        if ch == "#":
            break
        out.append(ch)
    return "".join(out).strip()


def line_has_exemption(code: str) -> bool:
    return any(
        token in code
        for token in (
            "energy_hits",
            "stop_below",
            "below_target",
            "below_reference",
        )
    )


def _energy_like(before: str) -> bool:
    tail = before.lower()
    if tail.endswith(" e") or tail.endswith(".e") or ".energy" in tail:
        return True
    return any(m in tail for m in ENERGY_MARKERS)


def _threshold_like(after: str) -> bool:
    rhs = after.strip().lower()
    if rhs in {"tol", "hit"}:
        return True
    if rhs.startswith(("thresh", "target", "tolerance", "level", "reference", "limit")):
        return True
    return any(m in rhs for m in THRESHOLD_MARKERS if m not in {"+ t ", "+ r "})


def _abs_energy_window(stripped: str) -> bool:
    lower = stripped.lower()
    if "abs(" not in lower:
        return False
    return any(
        needle in lower
        for needle in (
            "abs(",
            "- ref",
            "- reference",
            "- published",
            "- global_best",
            "- target",
        )
    ) and any(op in lower for op in ("<", "<=", ">=", ">", "=="))


SKIP_LINE_FRAGMENTS = (
    "is_hit(",
    "neighbor.",
    "connectivity <",
    "largest_component",
    "charged_work <",
    "engine.value()",
    "moves < limit",
    "chord[j]",
    "self.checked",
    "current_e - tolerance",
    "energy > floor",
    "dimer_tolerance",
    "cambridge_tolerance",
    "gmin_tolerance",
    "calibration_energy",
    "energy_tolerance",
    "default_hit_tol=1e-3",
    "reference * epsilon",
    "criterion e <",
    "plateau -",
    "mdtimestepfeedback",
    "rng.random",
    "cur == target",
    "let occupied",
    "value.is_finite() && value < current",
    "(\", \"anneal\",",
    "threshold)",
)


def has_raw_energy_comparison(stripped: str) -> bool:
    lower = stripped.lower()
    if not stripped or line_has_exemption(stripped):
        return False
    if any(frag in lower for frag in SKIP_LINE_FRAGMENTS):
        return False
    if any(
        tok in lower
        for tok in (
            "is_hit(",
            "check_hit(",
            "abs(hit_tol",
            "abs(level -",
            "abs(energy - published) >",
            "neighbor.distance",
            "d2 <=",
            "d2 <",
            "gradient_tolerance",
            "energy_threshold",
            "negative_tolerance",
            "negative_curvature",
            "rigid_accept_target",
            "target_trips",
            "target_n",
            "trial - current",
            "ordered[j][1] - plateau",
            "successful[j][1] - plateau",
            "*energy == target",
            "solved=1 iff",
            "best_val <=",
            "f_star",
            "limit_value - x",
            "prob.dim <=",
            "feedback::new(",
            "mdtimestepfeedback",
            "charged_work <",
            "largest_component(gradient)",
            "engine.value() >=",
            "moves < limit",
            "chord[j] < target",
            "self.checked - threshold",
            "value < current_e",
            "energy > floor",
            "default_hit_tol=1e-3",
            "connectivity < threshold",
            "references",
            "dimer_tolerance",
            "cambridge_tolerance",
            "calibration_energy_tolerance",
            "energy_tolerance",
            "gmin_tolerance",
        )
    ):
        return False
    if _abs_energy_window(stripped):
        if any(tok in lower for tok in ("published", "reference", "global_best")):
            if any(
                pat in lower
                for pat in (
                    "abs(energy - published",
                    "abs(energy - reference",
                    "abs(best - global_best",
                )
            ):
                return True
    heavy_ops = ("<=", "==", "<") if True else OPS
    for op in heavy_ops:
        rest = lower
        while op in rest:
            pos = rest.find(op)
            before = rest[:pos].strip()
            after = rest[pos + len(op) :].strip()
            if any(skip in before for skip in NON_ENERGY_BEFORE):
                rest = rest[pos + len(op) :]
                continue
            if before.endswith("threshold") and after[:1].isdigit():
                rest = rest[pos + len(op) :]
                continue
            if op in (">", ">="):
                rest = rest[pos + len(op) :]
                continue
            if _energy_like(before) and _threshold_like(after):
                return True
            if before.endswith("best") and _threshold_like(after):
                return True
            if "hit" in before and "int(" in before and _threshold_like(after):
                return True
            rest = rest[pos + len(op) :]
    if "<= thresh" in lower or "< thresh" in lower or "== thresh" in lower:
        return True
    if "_hits" in lower and ("<=" in lower or " < " in lower) and (
        "thresh" in lower or "target" in lower
    ):
        return True
    return False


def has_hit_literal(raw_line: str, stripped: str, *, rel: str) -> bool:
    if rel in RUNTIME_LITERAL_ALLOWLIST:
        return False
    if any(
        token in raw_line
        for token in (
            "is_hit(",
            'r#"DEFAULT_HIT_TOL',
            "MdTimeStepFeedback::new",
            '("sw10"',
        )
    ):
        return False
    if HIT_TOL_NAME_RE.search(raw_line) and "DEFAULT_HIT_TOL" not in raw_line:
        return True
    if HIT_LITERAL_RE.search(stripped):
        return True
    if "python3 -c" in stripped and "e<=r+w" in stripped.replace(" ", ""):
        return True
    return False


def classify_line(rel: str, line: str, ext: str) -> str | None:
    if rel in ALLOWED_FILES or rel in RUNTIME_LITERAL_ALLOWLIST:
        return None
    stripped = strip_comments(line, ext)
    if not stripped:
        return None
    raw = line.rstrip("\n")
    if has_hit_literal(raw, stripped, rel=rel):
        return stripped
    if has_raw_energy_comparison(stripped):
        return stripped
    return None


def git_tracked(root: Path, ext: str) -> list[Path]:
    out = subprocess.check_output(
        ["git", "ls-files", f"*{ext}"],
        cwd=root,
        text=True,
    )
    return [root / part for part in out.splitlines() if part]


def scan_repository(root: Path) -> list[str]:
    violations: list[str] = []
    for ext in ("py", "rs", "sh", "f90"):
        for path in git_tracked(root, ext):
            if any(part in SKIP_PARTS for part in path.parts):
                continue
            rel = path.relative_to(root).as_posix()
            if rel in ALLOWED_FILES:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            for index, line in enumerate(text.splitlines(), 1):
                hit = classify_line(rel, line, ext)
                if hit:
                    violations.append(f"{rel}:{index}: {line.strip()}")
    return violations


def fixture_must_violate(fixture_line: str) -> bool:
    """A single pre-fix code snippet from the fixture file must trip the linter."""
    line = fixture_line.strip()
    if not line or line.startswith("#"):
        return True
    lang, _, body = line.partition("|")
    lang = lang.strip().lower()
    ext = {"py": "py", "rs": "rs", "sh": "sh", "f90": "f90"}.get(lang, "py")
    stripped = body.split("#", 1)[0].strip() if ext in {"sh", "f90"} else body.strip()
    if line_has_exemption(stripped):
        return False
    if has_hit_literal(body, stripped, rel=f"fixture/{lang}_snippet"):
        return True
    compact = stripped.replace(" ", "")
    if "e<=r+w" in compact:
        return True
    return has_raw_energy_comparison(stripped)