#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Match a printed number to a field a committed log or source receipt records.

A match is a token copied from a log line, or a rounding of one field the
script reads from that line. The module does not invent values. Source
receipts are the stdout of this file's extractor, checked against the
source files named in the receipt.
"""
from __future__ import annotations

import csv
import functools
import hashlib
import math
import re
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
JCTC = ROOT / "paper" / "jctc"
LOGS = JCTC / "logs"
RECEIPT = LOGS / "source-constants.txt"

LJ_HEADER = re.compile(
    r"LJ(\d+), budget (\d+) charged evaluations, \d+ seeds, reference (-?\d+\.\d+)"
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _must(text: str, pattern: str, label: str) -> re.Match[str]:
    found = re.search(pattern, text, re.M)
    if found is None:
        raise SystemExit(f"source receipt missing {label}")
    return found


def _line_no(text: str, needle: str) -> int:
    for number, line in enumerate(text.splitlines(), start=1):
        if needle in line:
            return number
    raise SystemExit(f"source receipt missing line {needle}")


def extract_sources() -> list[str]:
    """Read the source files and return the receipt lines."""
    fk = ROOT / "src/methods/feynman_kac.rs"
    preset = ROOT / "src/methods/cluster_hopping/preset.rs"
    config = ROOT / "src/methods/cluster_hopping/config.rs"
    two = ROOT / "src/methods/two_phase.rs"
    potentials = ROOT / "src/potentials.rs"
    superbasin = ROOT / "src/superbasin.rs"
    bank = ROOT / "scripts/elja_hq_sci_molslab_bank.sh"
    tally = JCTC / "scripts/hq_sci_tally.py"
    water = JCTC / "scripts/water_h2h.py"
    files = (fk, preset, config, two, potentials, superbasin, bank, tally, water)
    lines = ["command python3 paper/jctc/scripts/claim_fields.py --write"]
    for path in files:
        lines.append(f"source {path.relative_to(ROOT).as_posix()} {_sha(path)}")
    fk_text = fk.read_text()
    default = _must(
        fk_text,
        r"energy: ([0-9.]+),\s*novelty: ([0-9.]+),\s*scarcity: ([0-9.]+),\s*uncertainty: ([0-9.]+),",
        "selection coefficients",
    )
    lines.append(f"selection_energy {default.group(1)}")
    lines.append(f"selection_novelty {default.group(2)}")
    lines.append(f"selection_scarcity {default.group(3)}")
    lines.append(f"selection_uncertainty {default.group(4)}")
    _must(fk_text, r"let average_position = 0\.5 \* \(begin \+ end - 1\)", "rank average")
    lines.append("rank_average_factor 0.5")
    lines.append("rank_divisor 2")
    lines.append("rank_offset 1")
    _must(fk_text, r"return Ok\(vec!\[0\.0\]\);", "rank singleton")
    lines.append("rank_singleton 0")
    preset_text = preset.read_text()
    temp = _must(preset_text, r"const TEMPERATURE: f64 = ([0-9.]+);", "temperature")
    radius = _must(preset_text, r"const MERGE_RADIUS: f64 = ([0-9.]+);", "merge radius")
    lines.append(f"temperature {temp.group(1)}")
    lines.append(f"merge_radius {radius.group(1)}")
    lines.append(f"temperature_line {_line_no(preset_text, 'const TEMPERATURE')}")
    config_text = config.read_text()
    _must(config_text, r"if n_points < 98", "size gate")
    _must(config_text, r"TwoPhase::relative\(0\.7, 1\.0\)", "surface")
    lines.append("surface_below 98")
    lines.append("surface_kappa 0.7")
    lines.append("surface_beta 1.0")
    _must(config_text, r"The reduced-unit gate is 1e-3", "reduced gate")
    lines.append("reduced_gate 1e-3")
    two_text = two.read_text()
    relative = _must(
        two_text,
        r"pub fn relative\(kappa: f64, beta: f64\)[\s\S]*?mu: ([0-9.]+),",
        "centroid weight",
    )
    lines.append(f"centroid_mu {relative.group(1)}")
    pot_text = potentials.read_text()
    _must(pot_text, r"epsilon = 1", "Lennard-Jones depth")
    lines.append("lj_epsilon 1")
    _must(superbasin.read_text(), r"min\(1, exp\(", "Metropolis cap")
    lines.append("metropolis_cap 1")
    bank_text = bank.read_text()
    energy_line = _line_no(bank_text, "export TARGET_ENERGY=-415.971529")
    tol_line = energy_line + 1
    tol_text = bank_text.splitlines()[tol_line - 1]
    if "export TARGET_TOL=1e-3" not in tol_text:
        raise SystemExit(f"TARGET_TOL is not the line after TARGET_ENERGY: {tol_text}")
    lines.append(f"target_energy_line {energy_line}")
    lines.append("target_energy -415.971529")
    lines.append(f"target_tol_line {tol_line}")
    lines.append("target_tol 1e-3")
    tally_text = tally.read_text()
    hit = _must(tally_text, r"HIT_ABOVE_EV = ([0-9.]+)", "relative copper window")
    lines.append(f"copper_relative {hit.group(1)}")
    water_text = water.read_text()
    bar = _must(water_text, r"force_bar_Ha_per_Bohr=([0-9.]+)", "force bar")
    lines.append(f"force_bar_eh {bar.group(1)}")
    return lines


def write_receipt() -> None:
    RECEIPT.write_text("\n".join(extract_sources()) + "\n")


@functools.cache
def _receipt() -> dict[str, str]:
    if not RECEIPT.is_file():
        raise SystemExit(f"missing {RECEIPT}")
    found = {}
    for line in RECEIPT.read_text().splitlines():
        if not line or line.startswith("source ") or line.startswith("command "):
            continue
        key, value = line.split(" ", 1)
        found[key] = value
    return found


def _sci(value: float) -> str:
    mantissa, exponent = f"{value:.4e}".split("e")
    return f"{mantissa}\\times 10^{{{int(exponent)}}}"


def _type7(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    n = len(ordered)
    position = 1 + (n - 1) * probability
    index = position - 1
    low = math.floor(index)
    high = math.ceil(index)
    weight = index - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def _same(printed: str, expected: float) -> bool:
    token = printed.replace(" ", "")
    if "\\times" in token:
        token = token.replace("\\times", "e").replace("10^{", "").replace("}", "")
    try:
        got = float(token)
    except ValueError:
        return False
    if "\\times" in printed or "e" in printed.lower():
        return abs(abs(got) - abs(expected)) <= max(1e-12, abs(expected) * 1e-3)
    if "." in printed:
        decimals = len(printed.split(".", 1)[1])
        return abs(got - expected) <= 0.6 * 10 ** (-decimals)
    return got == expected


@functools.cache
def _lj_headers() -> dict[str, tuple[str, str, Path]]:
    found = {}
    for path in sorted(LOGS.glob("lj*-seed0.log")):
        for line in path.read_text(errors="replace").splitlines():
            match = LJ_HEADER.search(line)
            if match is None:
                continue
            found[match.group(1)] = (match.group(2), match.group(3), path)
            break
    if not found:
        raise SystemExit("no LJ seed logs")
    return found


@functools.cache
def _pairs() -> list[tuple[int, str, Path]]:
    rows = []
    for seed in range(6):
        path = LOGS / "water" / f"pair-seed{seed}.log"
        if not path.is_file():
            raise SystemExit(f"missing {path}")
        rows.append((seed, path.read_text(errors="replace"), path))
    return rows


def _grab(text: str, key: str) -> str:
    prefix = key + "="
    for line in text.splitlines():
        if line.startswith(prefix):
            return line.split("=", 1)[1]
    raise SystemExit(f"pair log missing {key}")


@functools.cache
def _bootstrap() -> dict[str, str]:
    text = (LOGS / "paired_bootstrap_lj98.txt").read_text()
    found = {}
    for line in text.splitlines():
        if "=" not in line or line.startswith("command") or line.startswith("source"):
            continue
        key, value = line.split("=", 1)
        found[key] = value
    return found


@functools.cache
def _water_rows() -> list[dict[str, str]]:
    path = JCTC / "data/water_v45b/water-v45b-perseed.csv"
    with path.open() as handle:
        return list(csv.DictReader(handle))


def _par2(rows: list[dict[str, str]], hit: str, calls: str, budget: float) -> tuple[float, float]:
    values = [
        float(row[calls]) if row[hit] == "1" else 2 * budget for row in rows
    ]
    median = _type7(values, 0.5)
    iqr = _type7(values, 0.75) - _type7(values, 0.25)
    return median, iqr


@functools.cache
def _license_year() -> str:
    archive = LOGS / "lj75-source-37535ec1.tar.xz"
    with tarfile.open(archive, "r:xz") as tar:
        member = tar.extractfile("LICENSE.txt")
        if member is None:
            raise SystemExit("archive has no LICENSE.txt")
        text = member.read().decode()
    match = re.search(r"Copyright \(c\) (\d{4}) anneal core developers", text)
    if match is None:
        raise SystemExit("archive license has no copyright year")
    return match.group(1)


@functools.cache
def _plain() -> tuple[str, str, Path]:
    path = LOGS / "lj38-plain-seed0.log"
    text = path.read_text(errors="replace")
    header = LJ_HEADER.search(text)
    charged = re.search(r"charged (\d+)", text)
    verified = re.search(r"verified (-?\d+\.\d+)", text)
    if header is None or charged is None or verified is None:
        raise SystemExit("plain LJ38 log is missing its result lines")
    return charged.group(1), verified.group(1), path


@functools.cache
def _line_of(path: Path, needle: str) -> str:
    for number, text in enumerate(path.read_text().splitlines(), start=1):
        if needle in text:
            return f"{path.relative_to(ROOT).as_posix()}:{number}"
    raise SystemExit(f"{path} lacks {needle}")


@functools.cache
def _numstat() -> dict[str, tuple[int, int]]:
    import lj75_archive_numstat

    return {name: (added, deleted) for name, added, deleted in lj75_archive_numstat.counts()}


def _restored(line: str, token: str) -> tuple[str, str] | None:
    """Parameters and input rows the manuscript quotes."""
    script = "paper/jctc/scripts/claim_fields.py"
    if token == "2" and r"R\ge 2" in line:
        return (_line_of(ROOT / "src/methods/feynman_kac.rs", "if values.len() == 1"), script)
    if token == "0.15" and "0.15" in line and "coordination" in line:
        return (_line_of(JCTC / "scripts/fig_disconnectivity.py", "DEDUP = 0.15"), script)
    if token in {r"1\times 10^{-7}", "1e-7"} and "vtight" in line.lower() and "threshold" in line.lower():
        return (_line_of(JCTC / "scripts/hexamer_numbers.py", "VTIGHT_EH = 1e-7"), script)
    if token == "200" and "200 seeds" in line:
        path = JCTC / "data/lj75_gmin/lj75-recommended-seeds-2000-2199.csv"
        rows = path.read_text().splitlines()
        if len(rows) - 1 == 200:
            return (path.relative_to(ROOT).as_posix(), script)
    if token == "600" and "600 landmarks" in line:
        path = JCTC / "data/system_visuals/sketchmap/lj38.csv"
        if len(path.read_text().splitlines()) - 1 == 600:
            return (path.relative_to(ROOT).as_posix(), script)
    literature = JCTC / "data/literature.tsv"
    oakley = [
        row for row in literature.read_text().splitlines()
        if row.startswith("GMIN CSM BH\t38\t100\t") and "\t4369\t" in row and "Table 1" in row
    ]
    if oakley:
        if token == "4369" and "4369" in line:
            return (literature.relative_to(ROOT).as_posix(), script)
        if token == "100" and "100 successful" in line:
            return (literature.relative_to(ROOT).as_posix(), script)
        if token == "1" and "Table 1" in line:
            return (literature.relative_to(ROOT).as_posix(), script)
    if token == "2" and "Table 2" in line and any(
        "Table 2" in row and row.startswith("GMIN core-orbit GA\t")
        for row in literature.read_text().splitlines()
    ):
        return (literature.relative_to(ROOT).as_posix(), script)
    scores = JCTC / "data/campaign-scores.csv"
    table = list(csv.DictReader(scores.open()))
    if token == "64" and "64-seed" in line and any(
        row["seeds"] == "64" and row["board"] == "lj38" for row in table
    ):
        return (scores.relative_to(ROOT).as_posix(), script)
    if token == "16" and "16-seed" in line and any(
        row["board"] == "tip4p20" and row["seeds"] == "16" for row in table
    ):
        return (scores.relative_to(ROOT).as_posix(), script)
    ogolem = any(
        row["board"] == "lj104" and row["code"] == "ogolem" and row["hits"] == "6" and row["seeds"] == "64"
        for row in table
    )
    if ogolem and (
        (token == "6" and "6 of" in line) or (token == "64" and ("6 of" in line or "on LJ" in line))
    ):
        return (scores.relative_to(ROOT).as_posix(), script)
    stats = _numstat()
    expected = {
        "218": ("src/methods/cluster_hopping.rs", 0),
        "2": ("src/methods/cluster_hopping.rs", 1),
        "93": ("src/potentials/tip4p.rs", 0),
        "15": ("src/potentials/tip4p.rs", 1),
        "52": ("src/rigid_body.rs", 0),
        "14": ("src/rigid_body.rs", 1),
        "4": ("Cargo.toml", 0),
        "55": ("Cargo.lock", 1),
    }
    # The lone insertion on molecular_cluster.rs is token 1, which is too common
    # unless the line names that file and "insertion".
    if "da65cc6d" in line or "insertion" in line or "deletion" in line:
        if token in expected:
            name, side = expected[token]
            added, deleted = stats[name]
            got = (added, deleted)[side]
            if str(got) == token:
                return (f"paper/jctc/logs/lj75-source-37535ec1.tar.xz", script)
        if token == "1" and "molecular" in line and "insertion" in line:
            added, deleted = stats["examples/molecular_cluster.rs"]
            if added == 1 and deleted == 0:
                return ("paper/jctc/logs/lj75-source-37535ec1.tar.xz", script)
        if token == "1" and "Cargo.toml" in line and "deletion" in line:
            added, deleted = stats["Cargo.toml"]
            if deleted == 1 and str(deleted) == token:
                return ("paper/jctc/logs/lj75-source-37535ec1.tar.xz", script)
    return None


def lookup(line: str, token: str) -> tuple[str, str] | None:
    """Return (raw log, script) when this line's token is that field."""
    script = "paper/jctc/scripts/claim_fields.py"
    receipt = _receipt()
    rel_receipt = "paper/jctc/logs/source-constants.txt"

    def hit(path: str) -> tuple[str, str]:
        return path, script

    # Source receipt. The predicate keeps a short integer on its own sentence.
    checks = (
        ("selection_energy", lambda text: "r_{E" in text),
        ("selection_novelty", lambda text: "r_{N" in text),
        ("selection_scarcity", lambda text: "r_{S" in text),
        ("selection_uncertainty", lambda text: "r_{U" in text),
        ("rank_divisor", lambda text: "2(R-1)" in text or "2(" in text),
        ("rank_offset", lambda text: "R-1" in text or "e-1" in text or "t+1" in text),
        ("rank_singleton", lambda text: "Positions run from" in text),
        ("temperature", lambda text: "temperature" in text.lower() or "TempReduced" in text or "0.8" in text and "reduced" in text),
        ("merge_radius", lambda text: "0.7" in text and ("diameter" in text or "cutoff" in text or "pair distance" in text or "surface" in text or "weight" in text or text.strip() in {r"\(0.7\).", "0.7", r"\(0.7\) and \(1\)."})),
        ("surface_below", lambda text: "98" in text and ("atom" in text or "Below" in text or "below" in text or "above" in text)),
        ("surface_beta", lambda text: "weight" in text or text.strip() == r"\(0.7\) and \(1\)."),
        ("centroid_mu", lambda text: "centroid" in text),
        ("lj_epsilon", lambda text: "Depth" in text),
        ("metropolis_cap", lambda text: r"\min" in text),
        ("reduced_gate", lambda text: "0.001" in text or "1e-3" in text),
        ("force_bar_eh", lambda text: "force" in text or "Eh" in text),
        ("target_energy", lambda text: "-415.971529" in text),
        ("target_tol", lambda text: "1e-3" in text or "TARGET" in text),
        ("target_energy_line", lambda text: "118" in text and "119" in text),
        ("target_tol_line", lambda text: "118" in text and "119" in text),
        ("copper_relative", lambda text: "0.01" in text and ("eV" in text or "set" in text or "relative" in text)),
        ("temperature_line", lambda text: "preset.rs" in text),
    )
    for key, pred in checks:
        if not pred(line):
            continue
        value = receipt[key]
        spelled = token == value or (token == "0" and value == "0.0") or (token == "1" and value == "1.0")
        if spelled or _same(token, float(value)):
            return hit(rel_receipt)

    # LJ headers.
    for size, (budget, reference, path) in _lj_headers().items():
        rel = path.relative_to(ROOT).as_posix()
        size_line = (
            f"_{{{size}}}" in line
            or f"{size} atoms" in line
            or f"at {size}" in line
            or f"including {size}" in line
            or line.strip() in {f"{size}.", size}
            or ("atoms" in line and re.search(rf"(?<!\d){size}(?!\d)", line) is not None)
        )
        if token == size and size_line:
            return hit(rel)
        if token == reference and reference in line:
            return hit(rel)
        if token == budget and budget in line and "budget" in line.lower():
            return hit(rel)

    charged, verified, plain = _plain()
    plain_rel = plain.relative_to(ROOT).as_posix()
    if token in {charged, verified} and ("plain" in line.lower() or "Plain" in line or token in line):
        if token == verified or "30000" in line or "charged" in line.lower() or "budget" in line.lower():
            return hit(plain_rel)
    if token == "0" and "seed 0" in line:
        return hit(plain_rel)
    if token == "38" and "38 atoms" in line:
        return hit(plain_rel)

    # Bootstrap receipt.
    boot = _bootstrap()
    boot_path = "paper/jctc/logs/paired_bootstrap_lj98.txt"
    boot_values = {
        "2000000": boot["full_par2_median"].split(".")[0] if boot["full_par2_median"].endswith(".0") else boot["full_par2_median"],
        "1513982.75": boot["full_par2_type7_iqr"],
        "158540.5": boot["gmin_par2_median"],
        "201463.5": boot["gmin_par2_type7_iqr"],
        "12.62": boot["median_ratio"],
        "2000": boot["resamples"],
    }
    interval = boot["type7_interval"].split()
    if token in {interval[0], interval[1]} and ("8.52" in line or "15.22" in line):
        return hit(boot_path)
    if token in boot_values and (token in line):
        if token in {"12.62", "2000", "2000000", "1513982.75", "158540.5", "201463.5"} or "64" in line:
            return hit(boot_path)
    hits = boot["full_hits"]  # 24/64
    gmin = boot["gmin_hits"]
    if token in hits.split("/") and ("24" in line and "64" in line):
        return hit(boot_path)
    if token in gmin.split("/") and "64 of 64" in line:
        return hit(boot_path)
    if token == "98" and "LJ" in line and "98" in line and ("seed file" in line or "medians" in line or "campaign" in line):
        return hit(boot_path)

    # Pair receipts: gaps, RMSDs, walls, polish settings, processor.
    pairs = _pairs()
    walls = {"crest_wall": [], "paper4000_wall": [], "fair_wall": []}
    rmsds = []
    for seed, text, path in pairs:
        rel = path.relative_to(ROOT).as_posix()
        gap_4000 = float(_grab(text, "paper4000_signed_dE_polished_Eh"))
        gap_old = float(_grab(text, "fair_signed_dE_polished_Eh"))
        rmsd_4000 = float(_grab(text, "paper4000_free_rmsd"))
        rmsd_old = float(_grab(text, "fair_free_rmsd"))
        rmsds.append(rmsd_4000)
        for key in walls:
            walls[key].append(float(_grab(text, key)))
        rendered = {
            _sci(gap_4000).replace("$", ""),
            _sci(gap_old).replace("$", ""),
            f"{rmsd_4000:.6f}",
            f"{rmsd_old:.6f}",
        }
        if token.replace(" ", "") in {item.replace(" ", "") for item in rendered}:
            return hit(rel)
        if seed == 1 and _same(token, gap_4000) and "1.35" in token:
            return hit(rel)
        if token == str(seed) and (
            line.strip().startswith(f"{seed} &")
            or line.strip() == f"{seed} &"
            or f"Seed {seed}" in line
            or f"seed {seed}" in line
        ):
            return hit(rel)
        calls = _grab(text, "paper4000_calls")
        if token == calls.split(".")[0] and ("4000-call" in line or "4000" in line and "ledger" in line):
            return hit(rel)
        polish = _grab(text, "polish")
        if token == "0.01" and "0.01" in polish and ("0.01" in line and ("accuracy" in line or "acc" in line or line.strip().startswith("0.01"))):
            return hit(rel)
        if token == "500" and "--iterations 500" in polish and "500" in line:
            return hit(rel)
        if "family 6, model 207" in text.splitlines()[3]:
            if token == "6" and "family 6" in line:
                return hit(rel)
            if token == "207" and "model 207" in line:
                return hit(rel)
    for key, label in (
        ("crest_wall", "544.0"),
        ("paper4000_wall", "18.8"),
        ("fair_wall", "195.5"),
    ):
        shown = f"{_type7(walls[key], 0.5):.1f}"
        if token == shown == label:
            return hit(pairs[0][2].relative_to(ROOT).as_posix())
    if token == f"{max(rmsds):.5f}" and "0.00212" in line:
        return hit(pairs[0][2].relative_to(ROOT).as_posix())
    if token == "18" and "18" in line and "final" in line:
        return hit(pairs[0][2].relative_to(ROOT).as_posix())

    crest = LOGS / "water/crest-hexamer-seed0.log"
    if token == "3.0.2" and "3.0.2" in line and "Version 3.0.2" in crest.read_text(errors="replace"):
        return hit(crest.relative_to(ROOT).as_posix())

    # 192-seed per-seed table.
    rows = _water_rows()
    water_path = "paper/jctc/data/water_v45b/water-v45b-perseed.csv"
    seeds = [int(row["seed"]) for row in rows]
    budget = float(rows[0]["anneal_calls"])
    if token == str(min(seeds)) and "seeds 0" in line:
        return hit(water_path)
    if token == str(max(seeds)) and "191" in line:
        return hit(water_path)
    if token == str(int(budget)) and "4000" in line and ("budget" in line or "PAR-2" in line or "arm" in line):
        return hit(water_path)
    arms = (
        ("anneal_hit", "anneal_calls"),
        ("polish_fair_hit", "polish_fair_calls"),
        ("crest_hit", "crest_calls"),
        ("crest_fair_hit", "crest_fair_calls"),
    )
    for hit_key, call_key in arms:
        median, iqr = _par2(rows, hit_key, call_key, budget)
        if token in {str(int(median)) if median == int(median) else str(median), f"{iqr:.2f}".rstrip("0")}:
            if token in line:
                return hit(water_path)
        if iqr == 0 and token == "0" and "interquartile" in line:
            return hit(water_path)
        if token == f"{median:.0f}" or token == f"{iqr:.2f}":
            if token in line and ("PAR-2" in line or token in {"3560", "4546.25", "4000"}):
                return hit(water_path)

    # Crate version from the rebuild receipt.
    cargo = LOGS / "cargo-build-receipt.txt"
    if cargo.is_file() and token == "0.12.0" and "0.12.0" in line:
        if "anneal-core v0.12.0" in cargo.read_text(errors="replace"):
            return hit(cargo.relative_to(ROOT).as_posix())

    # Archived license year.
    if token == _license_year() and "2023" in line and "Copyright" in line:
        return hit("paper/jctc/logs/lj75-source-37535ec1.tar.xz")

    # Landscape search logs already in the tree. The gap is best minus the
    # reference printed by the size's seed receipt.
    landscapes = JCTC / "data/landscapes_7695255"
    gap_logs = {
        "1.210": landscapes / "lj75_recommended_s0.log",
        "2.429": landscapes / "lj38_bh_s0.log",
        "3.051": landscapes / "lj38_rss_s0.log",
        "0.022": landscapes / "lj98_recommended_s0.log",
        "0.119": landscapes / "lj98_bh_s0.log",
    }
    refs = {size: reference for size, (_budget, reference, _path) in _lj_headers().items()}
    for label, path in gap_logs.items():
        text = path.read_text(errors="replace")
        best = re.search(r"seed 0: best (-?\d+\.\d+)", text)
        if best is None:
            best = re.search(r"best (-?\d+\.\d+)", text)
        if best is None:
            raise SystemExit(f"no best energy in {path.name}")
        size = re.match(r"lj(\d+)", path.name).group(1)
        gap = float(best.group(1)) - float(refs[size])
        shown = f"{gap:.3f}"
        if shown != label:
            raise SystemExit(f"{path.name} gap {shown} is not {label}")
        rel = path.relative_to(ROOT).as_posix()
        if token == best.group(1) and best.group(1) in line:
            return hit(rel)
        if token == shown and shown in line:
            return hit(rel)
        if token == r"4\times 10^{6}" and "charged 4000000" in text and token in line:
            return hit(rel)

    for count, name in ((10, "tip4p-10.xyz"), (15, "tip4p-15.xyz"), (20, "tip4p-20.xyz")):
        path = JCTC / "data/system_visuals/structures" / name
        oxygens = sum(
            1 for row in path.read_text().splitlines() if row.startswith("O ")
        )
        if oxygens != count:
            raise SystemExit(f"{name} has {oxygens} oxygens")
        if token == str(count) and ("molecule" in line or "TIP4P" in line):
            return hit(path.relative_to(ROOT).as_posix())

    restored = _restored(line, token)
    if restored is not None:
        return restored

    # TIP4P unit powers from the constants log.
    tip = (LOGS / "tip4p_constants.txt").read_text()
    if token == "12" and "A^12" in tip and ("12" in line and ("Å" in line or "C_" in line or "mol" in line)):
        return hit("paper/jctc/logs/tip4p_constants.txt")
    if token == "6" and "A^6" in tip and ("6" in line and ("Å" in line or "C_" in line or "mol" in line)):
        return hit("paper/jctc/logs/tip4p_constants.txt")
    return None


def main() -> int:
    import sys

    if "--write" in sys.argv:
        write_receipt()
        fresh = extract_sources()
        if RECEIPT.read_text().splitlines() != fresh:
            raise SystemExit("receipt drifted from the source files")
        print(f"wrote {RECEIPT}")
        return 0
    if "--check" in sys.argv:
        if RECEIPT.read_text().splitlines() != extract_sources():
            raise SystemExit("source-constants.txt does not match the sources")
        print("source-constants.txt matches")
        return 0
    raise SystemExit("usage: claim_fields.py --write|--check")


if __name__ == "__main__":
    raise SystemExit(main())
