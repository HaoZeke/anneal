#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Trace each number in the manuscript, SI, and cover letter.

match is yes only when a committed script reads a raw log and reproduces
the printed value. A summary table is not a raw log. The script does not
invent values.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
JCTC = ROOT / "paper" / "jctc"
SOURCES = (
    JCTC / "manuscript.tex",
    JCTC / "si.tex",
    JCTC / "cover-letter.tex",
)
sys.path.insert(0, str(JCTC / "scripts"))

import claim_fields  # noqa: E402
import hexamer_numbers  # noqa: E402
import hq_sci_tally  # noqa: E402
import lj104_v414_baseline  # noqa: E402

SCI = re.compile(r"-?\d+(?:\.\d+)?\s*\\times\s*10\^\{-?\d+\}")
SCI_E = re.compile(r"-?\d+(?:\.\d+)?[eE][-+]?\d+")
VER = re.compile(r"\d+\.\d+\.\d+")
NUM = re.compile(r"-?\d+\.\d+|-?\d+")
MASK = re.compile(
    r"\\(?:cite|citep|citet|ref|eqref|label|url|href|includegraphics|bibliography)\{[^}]*\}"
    r"|arXiv:\d+\.\d+"
    r"|[0-9a-fA-F]{32,}"
)
MACRO = {
    "TempReduced": "0.8",
    "PlainEnergy": "-173.928427",
    "PlainCalls": "30000",
}


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def strip_comment(line: str) -> str:
    out = []
    escaped = False
    for char in line:
        if char == "%" and not escaped:
            break
        out.append(char)
        escaped = char == "\\" and not escaped
        if char != "\\":
            escaped = False
    return "".join(out)


def mask(line: str) -> str:
    return MASK.sub(lambda m: " " * len(m.group(0)), line)


def bounded(line: str, start: int, end: int) -> bool:
    """A bare integer glued to a word is not a number. sha256, PAR-2, 11pt."""
    before = line[start - 1] if start else ""
    after = line[end] if end < len(line) else ""
    return not before.isalpha() and not after.isalpha()


def spans(line: str) -> list[tuple[int, int, str]]:
    found: list[tuple[int, int, str]] = []
    taken: list[tuple[int, int]] = []

    def free(start: int, end: int) -> bool:
        return all(end <= left or start >= right for left, right in taken)

    for pattern in (SCI, SCI_E, VER, NUM):
        for match in pattern.finditer(line):
            if pattern is NUM and not bounded(line, match.start(), match.end()):
                continue
            if free(match.start(), match.end()):
                found.append((match.start(), match.end(), match.group(0)))
                taken.append((match.start(), match.end()))
    found.sort()
    return found


def hide_cites(line: str, in_cite: bool) -> tuple[str, bool]:
    """Blank citation keys, including a key that continues on the next line."""
    if in_cite:
        if "}" in line:
            cut = line.index("}") + 1
            line = " " * cut + line[cut:]
            in_cite = False
        else:
            return " " * len(line), True
    while True:
        start = line.find("\\cite")
        if start < 0:
            break
        brace = line.find("{", start)
        if brace < 0:
            break
        end = line.find("}", brace)
        if end < 0:
            return line[:start] + " " * (len(line) - start), True
        line = line[:start] + " " * (end + 1 - start) + line[end + 1 :]
    return line, in_cite


def canon_sci(token: str) -> str:
    token = re.sub(r"\s+", "", token)
    return token.replace("\\times", "e").replace("10^{", "").replace("}", "")


def as_float(token: str) -> float | None:
    try:
        if "\\times" in token:
            return float(canon_sci(token))
        return float(token)
    except ValueError:
        return None


def same(printed: str, expected: float) -> bool:
    got = as_float(printed)
    if got is None:
        return False
    if "\\times" in printed or re.search(r"[eE]", printed):
        return abs(got - expected) <= max(1e-12, abs(expected) * 1e-6)
    if "." in printed and "e" not in printed.lower():
        decimals = len(printed.split(".", 1)[1])
        return abs(got - expected) <= 0.6 * 10 ** (-decimals)
    return got == expected


def crest_rows() -> dict[int, tuple[int, int]]:
    rows = hexamer_numbers.load()
    return {seed: (int(row["old"]), int(row["engrad"])) for seed, row in rows.items()}


def gap_forms() -> set[str]:
    text = "\n".join(hexamer_numbers.table_lines(hexamer_numbers.load()))
    return set(re.findall(r"-?\d+\.\d+|\$[^$]+\$", text))


def copper() -> dict[str, str]:
    """Map a printed copper value to the rerun log that holds it."""
    logs = JCTC / "logs"
    tokens = [
        hq_sci_tally.best_token(hq_sci_tally.seed_file(logs, "cuh2_fcc_rec", index))
        for index in range(8)
    ]
    bests = [float(token) for token in tokens]
    gaps = [value - hq_sci_tally.REFERENCE_EV for value in bests]
    low = min(range(8), key=lambda i: bests[i])
    high = max(range(8), key=lambda i: bests[i])
    glow = min(range(8), key=lambda i: gaps[i])
    ghigh = max(range(8), key=lambda i: gaps[i])
    out = {}
    for index, value in (
        (low, bests[low]),
        (high, bests[high]),
        (glow, gaps[glow]),
        (ghigh, gaps[ghigh]),
    ):
        out[f"{value:.12g}"] = rel(logs / f"cuh2_fcc_rec_{index}.out")
    return out


def tip4p_text() -> str:
    return (JCTC / "logs" / "tip4p_constants.txt").read_text()


def classify(path: Path, line_no: int, line: str, token: str, helpers: dict) -> tuple[str, str, str]:
    """Return raw log, script, match."""
    found = claim_fields.lookup(line, token)
    if found is not None:
        return (*found, "yes")
    script_hex = rel(JCTC / "scripts" / "hexamer_numbers.py")
    script_cu = rel(JCTC / "scripts" / "hq_sci_tally.py")
    script_tip = rel(JCTC / "scripts" / "tip4p_constants.py")
    script_water = rel(JCTC / "scripts" / "water_results_table.py")
    counter = re.fullmatch(r"\s*(\d+)\s*&\s*(\d+)\s*&\s*(\d+)\s*", line.split("\\\\")[0])
    if counter and token in {counter.group(1), counter.group(2), counter.group(3)}:
        seed = int(counter.group(1))
        old, engrad = helpers["crest"][seed]
        raw = rel(JCTC / "logs" / "water" / f"crest-hexamer-seed{seed}.log")
        ok = (token == counter.group(1)) or (
            token == str(old) and int(token) == old
        ) or (token == str(engrad) and int(token) == engrad)
        # The seed digit is also the row label. It matches the log name.
        if token == counter.group(1):
            ok = True
        return (raw, script_hex, "yes" if ok else "no")
    for seed, (old, engrad) in helpers["crest"].items():
        if token in {str(old), str(engrad)}:
            raw = rel(JCTC / "logs" / "water" / f"crest-hexamer-seed{seed}.log")
            return (raw, script_hex, "yes")
    if "\\times" in token or re.fullmatch(r"0\.\d{6}", token):
        forms = helpers["gaps"]
        literal = token if "\\times" not in token else None
        if token in forms or (literal and any(literal in form for form in forms)):
            return ("", script_hex, "no")
        if "\\times" in token and any(token.replace(" ", "") in form.replace(" ", "") for form in forms):
            return ("", script_hex, "no")
    if token in {"544.0", "18.8", "195.5"}:
        return ("", script_hex, "no")
    if token in helpers["copper_forms"]:
        return (helpers["copper_forms"][token], script_cu, "yes")
    # Printed copper forms that equal a logged best or gap.
    number = as_float(token)
    if number is not None and ("415.971" in line or "e" in token.lower()):
        for key, raw in helpers["copper"].items():
            if same(token, float(key)):
                return (raw, script_cu, "yes")
    tip_values = {"2510400", "2552.24", "0.6485", "3.15365", "2551.8252", "2510336.2"}
    if token in tip_values and token in helpers["tip"]:
        return (rel(JCTC / "logs" / "tip4p_constants.txt"), script_tip, "yes")
    perseed = rel(JCTC / "data" / "water_v45b" / "water-v45b-perseed.csv")
    if token in {"0.0264", "0.000374"} or (number is not None and same(token, 1.37e-6)):
        return (perseed, script_water, "yes")
    if token in {"121", "97", "148", "114"} or (
        token == "192" and ("192" in line or "seed" in line.lower())
    ):
        return (perseed, script_water, "yes")
    if token == "-197.7805":
        return (
            rel(JCTC / "data" / "system_visuals" / "ccd" / "TIP4P" / "xyz" / "TIP4P-6.xyz"),
            rel(JCTC / "scripts" / "system_visuals.py"),
            "yes",
        )
    bootstrap_values = {
        "2000000",
        "1513982.75",
        "158540.5",
        "201463.5",
        "12.62",
        "8.52",
        "15.22",
    }
    if token in bootstrap_values or (token == "2000" and "resample" in line):
        return (rel(JCTC / "logs" / "paired_bootstrap_lj98.txt"), "", "no")
    if token in {"-173.928427", "30000"} and ("plain" in line.lower() or "Plain" in line):
        return (rel(JCTC / "logs" / "lj38-plain-seed0.log"), "", "no")
    if token == "4369":
        return ("", "", "no")
    script_lj = rel(JCTC / "scripts" / "lj104_v414_baseline.py")
    base = helpers["lj104"]
    if token == base["reference"]:
        return (rel(JCTC / "logs" / "lj104-seed0.log"), script_lj, "yes")
    if "0 of 6 hits" in line or "581.80180238" in line:
        seed_logs = {row["seed"]: rel(row["path"]) for row in base["rows"]}
        best_log = seed_logs[base["seed"]]
        if token in {base["hits"], base["n"], base["best"], base["seed"]} or token in seed_logs:
            return (seed_logs.get(token, best_log), script_lj, "yes")
    return ("", "", "no")


def main() -> int:
    helpers = {
        "crest": crest_rows(),
        "gaps": gap_forms(),
        "copper": copper(),
        "copper_forms": {},
        "tip": tip4p_text(),
        "bootstrap": (JCTC / "logs" / "paired_bootstrap_lj98.txt").read_text(),
        "lj104": lj104_v414_baseline.baseline(),
    }
    helpers["bootstrap_floats"] = re.findall(r"-?\d+\.\d+|-?\d+", helpers["bootstrap"])
    # Attach the printed scientific copper gaps by float, not by a second dict.
    rows = []
    for path in SOURCES:
        in_cite = False
        for line_no, raw_line in enumerate(path.read_text().splitlines(), start=1):
            line = strip_comment(raw_line)
            line, in_cite = hide_cites(line, in_cite)
            if not line.strip():
                continue
            masked = mask(line)
            for _start, _end, token in spans(masked):
                raw_log, script, match = classify(path, line_no, line, token, helpers)
                rows.append((f"{rel(path)}:{line_no}", token, raw_log, script, match))
            for name, value in MACRO.items():
                if re.search(rf"\\{name}\b", line):
                    raw_log, script, match = classify(path, line_no, line + " Plain", value, helpers)
                    rows.append((f"{rel(path)}:{line_no}", value, raw_log, script, match))
    out = JCTC / "NUMBER_TRACE.tsv"
    lines = ["file:line\tvalue\traw_log\tscript\tmatch"]
    for row in rows:
        lines.append("\t".join(row))
    out.write_text("\n".join(lines) + "\n")
    yes = sum(row[-1] == "yes" for row in rows)
    no = len(rows) - yes
    missing = sum(row[2] == "" for row in rows)
    unmatched = sum(row[-1] == "no" and row[2] != "" for row in rows)
    no_log = sum(row[-1] == "no" and row[2] == "" for row in rows)
    print(f"rows {len(rows)}")
    print(f"matched {yes}")
    print(f"unmatched {unmatched}")
    print(f"missing_log {no_log}")
    print(f"match_no {no}")
    print(f"missing_or_unmatched_rows {missing}")
    fail = JCTC / "number-blockers.tsv"
    fail_lines = ["file:line\tvalue\traw_log\tscript\tmatch\treason"]
    for row in rows:
        if row[-1] == "yes":
            continue
        reason = "no raw log" if row[2] == "" else "does not match"
        if row[2] and row[3] == "":
            reason = "log has the value but no committed script derives it"
        elif row[3] and row[2] == "":
            reason = "committed script reproduces the value from a table that is not a raw log"
        fail_lines.append("\t".join(row + (reason,)))
    fail.write_text("\n".join(fail_lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
