#!/usr/bin/env python3
"""Code-only A/B of the water hexamer paper panel.

Two builds of the search code, A and B, run the four arms of the panel on the
same GFN2 engine, one thread on one pinned core, strictly serially. For every
seed and every arm the two codes run back to back, and which code goes first
alternates with the seed parity.

Arms
  anneal       the example driver at the full budget; the answer is its best
               structure.
  polish_fair  the example driver at the search budget, then the L-BFGS polish
               of its best structure within the reserve; the answer is the
               polished structure.
  crest        CREST in its default flow on the template, stopped after the
               full budget of engine calls; the answer is the set of frames of
               its metadynamics trajectories.
  crest_fair   CREST stopped after the search budget, then the polish of its
               lowest-energy frame within the reserve; the answer is the frames
               and the polished structure.

Hit criterion: a structure is a hit when its monomers are intact and the
species-aware IRA match against the reference isomers is inside the gate, the
gate being half the smallest IRA RMSD between two references, computed by the
example drivers before the first seed. A set of frames is a hit when any frame
is a hit. Each code scores with its own build; the other build also scores the
same structures so that code effects on the search and on the scoring can be
told apart.

Time inside the potential is the time spent in the engine force call, counted
by a shim loaded in place of the engine. Wall time is the elapsed time of the
whole procedure; for CREST it ends when the last engine call of the budget has
been answered. Time outside the potential is the difference.
"""
import argparse
import csv
import hashlib
import json
import os
import re
import resource
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

ARMS = ("anneal", "polish_fair", "crest", "crest_fair")
CODES = ("A", "B")
REFERENCES = ("prism", "cage", "book")

SCRUBBED_PREFIXES = ("WATER_", "PAPER_", "RGPOT_", "XTB_SHIM_", "CREST_", "OMP_", "OPENBLAS_", "MKL_")
SINGLE_THREAD = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "GOTO_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}

RUN_FIELDS = [
    "seed", "code", "arm", "job_index", "position_in_pair", "status", "hit",
    "calls", "search_calls", "polish_calls",
    "wall_s", "in_s", "out_s",
    "search_wall_s", "search_in_s", "polish_wall_s", "polish_in_s",
    "final_energy_eV", "search_energy_eV",
    "isomer", "isomer_rmsd_A", "isomer_reason", "gate_A",
    "cross_isomer", "cross_rmsd_A",
    "hit_search_or_frames", "hit_polish", "polished_isomer", "polished_rmsd_A",
    "polish_outcome", "polish_start_eV", "polish_max_gradient_eV_A",
    "frames", "frames_intact", "frames_ira_failed", "frames_inside_gate",
    "frames_inside_gate_cross", "min_rmsd_A", "min_rmsd_cross_A",
    "best_frame", "best_frame_epot_eV",
    "reported_hit", "geometry_check", "process_wall_s", "notes",
]
FRAME_FIELDS = [
    "seed", "code", "arm", "frame", "epot_Eh", "intact", "rmsd_A", "isomer",
    "cross_intact", "cross_rmsd_A", "cross_isomer",
    "rmsd_prism_A", "rmsd_cage_A", "rmsd_book_A",
]
GOFEE_LINE = re.compile(
    r"^gofee_geometry: (\S+) intact: (\S+) isomer: (\S+) "
    r"(?:isomer_rmsd_A: (\S+)|isomer_reason: (.*))$"
)
EV_PER_HARTREE = 27.21138505


class Context:
    def __init__(self, args):
        self.args = args
        self.binaries = {"A": Path(args.binary_a).resolve(), "B": Path(args.binary_b).resolve()}
        self.engine = Path(args.engine).resolve()
        self.shim = Path(args.shim).resolve()
        self.driver = Path(args.driver).resolve()
        self.crest = Path(args.crest).resolve()
        self.water = Path(args.water_xyz).resolve()
        self.references = Path(args.references_dir).resolve()
        self.out = Path(args.out_dir).resolve()
        self.scratch = Path(args.scratch_dir).resolve()
        self.polish_script = Path(__file__).resolve().with_name("polish.py")
        self.single_references = {}
        self.job_index = 0


def base_environment():
    env = {k: v for k, v in os.environ.items() if not k.startswith(SCRUBBED_PREFIXES)}
    env.update(SINGLE_THREAD)
    return env


def shim_environment(ctx, stats, buffer_mode):
    env = base_environment()
    env["XTB_SHIM_REAL"] = str(ctx.engine)
    env["XTB_SHIM_STATS"] = str(stats)
    env["XTB_SHIM_BUFFER"] = buffer_mode
    return env


def code_environment(ctx, code, stats):
    env = shim_environment(ctx, stats, "force" if code == "A" else "gradient")
    env["RGPOT_XTB_ENGINE"] = str(ctx.shim)
    return env


def limit_file_size():
    limit = 4 << 30
    resource.setrlimit(resource.RLIMIT_FSIZE, (limit, limit))


def run_process(argv, env, cwd, stdout_path, stderr_path, timeout):
    with open(stdout_path, "wb") as out, open(stderr_path, "wb") as err:
        begin = time.monotonic_ns()
        proc = subprocess.Popen(
            [str(a) for a in argv], env=env, cwd=cwd, stdout=out, stderr=err,
            start_new_session=True, preexec_fn=limit_file_size,
        )
        timed_out = False
        try:
            code = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            code = None
        end = time.monotonic_ns()
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    return code, (end - begin) / 1e9, timed_out, begin


def read_stats(path):
    try:
        data = Path(path).read_bytes()[:24]
        calls, failed, inside_ns = struct.unpack("<QQQ", data)
    except (OSError, struct.error):
        return 0, 0, 0.0
    return calls, failed, inside_ns / 1e9


def parse_report(text):
    fields = {}
    inside_seed = False
    for line in text.splitlines():
        if line.startswith("seed: "):
            inside_seed = True
        if inside_seed and ": " in line:
            key, value = line.split(": ", 1)
            fields.setdefault(key, value.strip())
    return fields


def as_float(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def read_xyz_energy(path):
    try:
        comment = Path(path).read_text().splitlines()[1]
    except (OSError, IndexError):
        return None
    match = re.search(r"energy\s+(-?\d+\.\d+)", comment)
    return float(match.group(1)) if match else None


def classify(ctx, code, directory, references, tag, work):
    stats = work / f"classify-{tag}.stats"
    env = code_environment(ctx, code, stats)
    env["WATER_ISOMER_DIR"] = str(references)
    env["WATER_GOFEE_DIR"] = str(directory)
    env["PAPER_SMOKE"] = "1"
    stdout = work / f"classify-{tag}.out"
    status, _, timed_out, _ = run_process(
        [ctx.binaries[code], "gfn2", "recommended"], env, work, stdout,
        work / f"classify-{tag}.err", ctx.args.classify_timeout,
    )
    if status != 0 or timed_out:
        raise RuntimeError(f"classification {tag} by {code} failed")
    rows, gate = {}, None
    for line in stdout.read_text().splitlines():
        if gate is None and line.startswith("isomer_gate_A: "):
            gate = as_float(line.split(": ", 1)[1])
        found = GOFEE_LINE.match(line.strip())
        if found:
            path, intact, name, rmsd, reason = found.groups()
            rows[os.path.basename(path)] = {
                "intact": intact,
                "isomer": name,
                "rmsd": as_float(rmsd),
                "reason": reason or "",
            }
    return rows, gate


def hit_of(rows):
    return any(row["isomer"] != "unassigned" for row in rows.values())


def single_answer(rows):
    row = next(iter(rows.values()))
    return row["isomer"], row["rmsd"]


def anneal_search(ctx, code, seed, budget, work):
    stats = work / "search.stats"
    geometry_dir = work / "geometry"
    env = code_environment(ctx, code, stats)
    env["WATER_ISOMER_DIR"] = str(ctx.references)
    env["WATER_FORCE_BUDGET"] = str(budget)
    env["WATER_SEED_FIRST"] = str(seed)
    env["WATER_SEED_COUNT"] = "1"
    env["WATER_GEOM_DIR"] = str(geometry_dir)
    if code == "B":
        env["WATER_START_XYZ"] = str(ctx.water)
    stdout = work / "search.out"
    status, wall, timed_out, _ = run_process(
        [ctx.binaries[code], "gfn2", "recommended"], env, work, stdout,
        work / "search.err", ctx.args.search_timeout,
    )
    calls_shim, failed, inside = read_stats(stats)
    report = parse_report(stdout.read_text()) if stdout.exists() else {}
    return {
        "status": status, "timed_out": timed_out, "wall": wall, "inside": inside,
        "calls_shim": calls_shim, "failed": failed, "report": report,
        "geometry": geometry_dir / f"seed-{seed}.xyz",
    }


def polish_geometry(ctx, geometry, work, tag):
    stats = work / f"{tag}.stats"
    env = shim_environment(ctx, stats, "force")
    out = work / f"{tag}.xyz"
    report_path = work / f"{tag}.report"
    status, wall, timed_out, _ = run_process(
        [
            sys.executable, ctx.polish_script, "--engine", ctx.shim,
            "--geometry", geometry, "--out", out, "--report", report_path,
            "--budget", ctx.args.polish_budget, "--gtol", ctx.args.polish_gtol,
            "--history", ctx.args.polish_history,
        ],
        env, work, work / f"{tag}.out", work / f"{tag}.err", ctx.args.polish_timeout,
    )
    calls, failed, inside = read_stats(stats)
    fields = {}
    if report_path.exists():
        for line in report_path.read_text().splitlines():
            key, _, value = line.partition(": ")
            fields[key] = value
    ok = status == 0 and not timed_out and out.exists()
    return {
        "ok": ok, "geometry": out, "wall": as_float(fields.get("wall_s")) or wall,
        "inside": inside, "calls": calls, "fields": fields, "process_wall": wall,
    }


def read_frames(trj_path):
    frames = []
    try:
        lines = Path(trj_path).read_text().splitlines()
    except OSError:
        return frames
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        count = int(lines[i].split()[0])
        atoms = lines[i + 2 : i + 2 + count]
        if len(atoms) != count:
            break
        comment = lines[i + 1]
        found = re.search(r"Epot =\s*(-?\d+\.\d+)", comment)
        frames.append({"comment": comment, "atoms": atoms, "epot": float(found.group(1)) if found else None})
        i += 2 + count
    return frames


def write_frames(frames, directory):
    directory.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(frames):
        text = "\n".join([str(len(frame["atoms"])), frame["comment"], *frame["atoms"]]) + "\n"
        (directory / f"frame-{index:04d}.xyz").write_text(text)


def crest_run(ctx, cap, work):
    run_dir = work / "run"
    crest_dir = work / "crest"
    run_dir.mkdir()
    crest_dir.mkdir()
    shutil.copy(ctx.water, crest_dir / "start.xyz")
    (crest_dir / "calc.toml").write_text(
        "[calculation]\n[[calculation.level]]\n"
        f'method = "generic"\nbinary = "{ctx.driver}"\n'
        'gradfile = "genericinp.engrad"\ngradtype = "engrad"\n'
    )
    stats = work / "crest.stats"
    server_env = shim_environment(ctx, stats, "force")
    server_out = open(work / "server.out", "wb")
    server = subprocess.Popen(
        [str(ctx.driver), "serve", str(run_dir), str(cap), str(ctx.shim)],
        env=server_env, stdout=server_out, stderr=subprocess.STDOUT, start_new_session=True,
    )
    deadline = time.monotonic() + 60
    while not (run_dir / "ready").exists():
        if server.poll() is not None or time.monotonic() > deadline:
            server_out.close()
            raise RuntimeError("the call server did not start")
        time.sleep(0.002)
    env = base_environment()
    env["CREST_DRIVER_RUNDIR"] = str(run_dir)
    status, process_wall, timed_out, begin = run_process(
        [ctx.crest, "start.xyz", "-T", "1", "--input", "calc.toml"], env, crest_dir,
        work / "crest.out", work / "crest.err", ctx.args.crest_timeout,
    )
    subprocess.run([str(ctx.driver), "stop", str(run_dir)], check=False)
    try:
        server.wait(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(server.pid, signal.SIGKILL)
        server.wait()
    server_out.close()
    summary = {}
    summary_path = run_dir / "driver.txt"
    if summary_path.exists():
        for line in summary_path.read_text().splitlines():
            key, _, value = line.partition(": ")
            summary[key] = value
    calls = int(summary.get("calls", 0))
    cap_ns = int(summary.get("cap_monotonic_ns", 0))
    reached = cap_ns > 0
    wall = (cap_ns - begin) / 1e9 if reached else process_wall
    shim_calls, failed, inside = read_stats(stats)
    return {
        "status": status, "timed_out": timed_out, "wall": wall, "process_wall": process_wall,
        "calls": calls, "failed": failed, "shim_calls": shim_calls, "inside": inside,
        "cap_reached": reached, "trj": crest_dir / "crest_dynamics.trj",
        "mdfiles": crest_dir / "MDFILES",
    }


def collect_frames(crest_result):
    frames = read_frames(crest_result["trj"])
    if frames:
        return frames
    gathered = []
    mdfiles = crest_result["mdfiles"]
    if mdfiles.exists():
        for path in sorted(mdfiles.glob("crest_*.trj"), key=lambda p: int(re.findall(r"\d+", p.stem)[0])):
            gathered.extend(read_frames(path))
    return gathered


def empty_row(seed, code, arm, job_index, position):
    row = {field: "" for field in RUN_FIELDS}
    row.update(seed=seed, code=code, arm=arm, job_index=job_index, position_in_pair=position,
               status="error", hit=0)
    return row


def other(code):
    return "B" if code == "A" else "A"


def job_anneal(ctx, row, work, fair):
    seed, code = row["seed"], row["code"]
    budget = ctx.args.search_budget if fair else ctx.args.budget
    search = anneal_search(ctx, code, seed, budget, work)
    report = search["report"]
    row["process_wall_s"] = round(search["wall"], 6)
    row["search_wall_s"] = round(search["wall"], 6)
    row["search_in_s"] = round(search["inside"], 6)
    row["search_calls"] = report.get("force_calls", "")
    row["search_energy_eV"] = report.get("best_energy", "")
    row["gate_A"] = report.get("isomer_gate_A", "")
    row["reported_hit"] = report.get("hit", "")
    if search["status"] != 0 or search["timed_out"] or "force_calls" not in report:
        row["status"] = "search_failed" if not search["timed_out"] else "search_timeout"
        return
    row["calls"] = int(report["force_calls"])
    row["wall_s"] = search["wall"]
    row["in_s"] = search["inside"]
    final_geometry = search["geometry"]
    row["isomer"] = report.get("isomer", "")
    row["isomer_rmsd_A"] = report.get("isomer_rmsd_A", "")
    row["isomer_reason"] = report.get("isomer_reason", "")
    row["final_energy_eV"] = report.get("best_energy", "")
    hit_search = int(report.get("isomer_reason") == "inside gate")
    row["hit_search_or_frames"] = hit_search
    hit = hit_search
    geometry_check = ""
    if final_geometry.exists():
        directory = work / "answer-search"
        directory.mkdir()
        shutil.copy(final_geometry, directory / final_geometry.name)
        own, _ = classify(ctx, code, directory, ctx.references, "search-own", work)
        name, rmsd = single_answer(own)
        energy = read_xyz_energy(final_geometry)
        reported = as_float(report.get("best_energy"))
        energy_ok = energy is not None and reported is not None and abs(energy - reported) < 1e-6
        geometry_check = "ok" if (energy_ok and (name != "unassigned") == bool(hit_search)) else "mismatch"
        cross, _ = classify(ctx, other(code), directory, ctx.references, "search-cross", work)
        row["cross_isomer"], row["cross_rmsd_A"] = single_answer(cross)
    else:
        geometry_check = "no_geometry"
    row["geometry_check"] = geometry_check
    if fair:
        if not final_geometry.exists():
            row["status"] = "no_geometry_to_polish"
            return
        polished = polish_geometry(ctx, final_geometry, work, "polish")
        row["polish_wall_s"] = round(polished["wall"], 6)
        row["polish_in_s"] = round(polished["inside"], 6)
        row["polish_calls"] = polished["calls"]
        row["polish_outcome"] = polished["fields"].get("outcome", "")
        row["polish_start_eV"] = polished["fields"].get("energy_start_eV", "")
        row["polish_max_gradient_eV_A"] = polished["fields"].get("max_gradient_final_eV_A", "")
        if not polished["ok"]:
            row["status"] = "polish_failed"
            return
        row["calls"] = int(report["force_calls"]) + polished["calls"]
        row["wall_s"] = search["wall"] + polished["wall"]
        row["in_s"] = search["inside"] + polished["inside"]
        row["final_energy_eV"] = polished["fields"].get("energy_final_eV", "")
        directory = work / "answer-polished"
        directory.mkdir()
        shutil.copy(polished["geometry"], directory / "polished.xyz")
        own, _ = classify(ctx, code, directory, ctx.references, "polish-own", work)
        row["isomer"], rmsd = single_answer(own)
        row["isomer_rmsd_A"] = "" if rmsd is None else f"{rmsd:.6f}"
        row["isomer_reason"] = ""
        row["polished_isomer"], row["polished_rmsd_A"] = row["isomer"], row["isomer_rmsd_A"]
        hit = int(row["isomer"] != "unassigned")
        row["hit_polish"] = hit
        cross, _ = classify(ctx, other(code), directory, ctx.references, "polish-cross", work)
        row["cross_isomer"], row["cross_rmsd_A"] = single_answer(cross)
    row["hit"] = hit
    row["status"] = "ok"


def job_crest(ctx, row, work, fair, frame_writer):
    seed, code, arm = row["seed"], row["code"], row["arm"]
    cap = ctx.args.search_budget if fair else ctx.args.budget
    result = crest_run(ctx, cap, work)
    row["process_wall_s"] = round(result["process_wall"], 6)
    row["search_wall_s"] = round(result["wall"], 6)
    row["search_in_s"] = round(result["inside"], 6)
    row["search_calls"] = result["calls"]
    if result["timed_out"]:
        row["status"] = "crest_timeout"
        return
    if not result["cap_reached"]:
        row["status"] = "cap_not_reached"
    frames = collect_frames(result)
    row["frames"] = len(frames)
    row["calls"] = result["calls"]
    row["wall_s"] = result["wall"]
    row["in_s"] = result["inside"]
    if not frames:
        row["status"] = row["status"] if row["status"] != "error" else "no_frames"
        return
    frames_dir = work / "frames"
    write_frames(frames, frames_dir)
    own, gate = classify(ctx, code, frames_dir, ctx.references, "frames-own", work)
    cross, _ = classify(ctx, other(code), frames_dir, ctx.references, "frames-cross", work)
    per_reference = {name: classify_single(ctx, code, frames_dir, name, work) for name in REFERENCES}
    row["gate_A"] = "" if gate is None else gate
    inside = sum(1 for r in own.values() if r["isomer"] != "unassigned")
    inside_cross = sum(1 for r in cross.values() if r["isomer"] != "unassigned")
    row["frames_intact"] = sum(1 for r in own.values() if r["intact"] == "true")
    row["frames_ira_failed"] = sum(1 for r in own.values() if r["intact"] == "unknown")
    row["frames_inside_gate"] = inside
    row["frames_inside_gate_cross"] = inside_cross
    rmsds = [r["rmsd"] for r in own.values() if r["rmsd"] is not None]
    rmsds_cross = [r["rmsd"] for r in cross.values() if r["rmsd"] is not None]
    row["min_rmsd_A"] = "" if not rmsds else f"{min(rmsds):.6f}"
    row["min_rmsd_cross_A"] = "" if not rmsds_cross else f"{min(rmsds_cross):.6f}"
    energies = [(f["epot"], i) for i, f in enumerate(frames) if f["epot"] is not None]
    for index, frame in enumerate(frames):
        name = f"frame-{index:04d}.xyz"
        mine, theirs = own.get(name), cross.get(name)
        frame_writer.writerow({
            "seed": seed, "code": code, "arm": arm, "frame": index,
            "epot_Eh": "" if frame["epot"] is None else f"{frame['epot']:.9f}",
            "intact": mine["intact"] if mine else "",
            "rmsd_A": "" if not mine or mine["rmsd"] is None else f"{mine['rmsd']:.5f}",
            "isomer": "" if not mine else ("-" if mine["isomer"] == "unassigned" else mine["isomer"]),
            "cross_intact": theirs["intact"] if theirs else "",
            "cross_rmsd_A": "" if not theirs or theirs["rmsd"] is None else f"{theirs['rmsd']:.5f}",
            "cross_isomer": "" if not theirs else ("-" if theirs["isomer"] == "unassigned" else theirs["isomer"]),
            **{
                f"rmsd_{ref}_A": "" if name not in per_reference[ref] or per_reference[ref][name]["rmsd"] is None
                else f"{per_reference[ref][name]['rmsd']:.5f}"
                for ref in REFERENCES
            },
        })
    hit_frames = int(inside > 0)
    row["hit_search_or_frames"] = hit_frames
    hit = hit_frames
    if energies:
        best_epot, best_index = min(energies)
        row["best_frame"] = best_index
        row["best_frame_epot_eV"] = f"{best_epot * EV_PER_HARTREE:.8f}"
        row["final_energy_eV"] = row["best_frame_epot_eV"]
    names = sorted({r["isomer"] for r in own.values() if r["isomer"] != "unassigned"})
    row["isomer"] = "+".join(names) if names else "-"
    if fair and energies:
        directory = work / "best-frame"
        directory.mkdir()
        shutil.copy(frames_dir / f"frame-{best_index:04d}.xyz", directory / "best.xyz")
        polished = polish_geometry(ctx, directory / "best.xyz", work, "polish")
        row["polish_wall_s"] = round(polished["wall"], 6)
        row["polish_in_s"] = round(polished["inside"], 6)
        row["polish_calls"] = polished["calls"]
        row["polish_outcome"] = polished["fields"].get("outcome", "")
        row["polish_start_eV"] = polished["fields"].get("energy_start_eV", "")
        row["polish_max_gradient_eV_A"] = polished["fields"].get("max_gradient_final_eV_A", "")
        if not polished["ok"]:
            row["status"] = "polish_failed"
            row["hit"] = hit
            return
        row["calls"] = result["calls"] + polished["calls"]
        row["wall_s"] = result["wall"] + polished["wall"]
        row["in_s"] = result["inside"] + polished["inside"]
        energy_final = as_float(polished["fields"].get("energy_final_eV"))
        if energy_final is not None:
            row["final_energy_eV"] = f"{min(energy_final, best_epot * EV_PER_HARTREE):.8f}"
        pdir = work / "answer-polished"
        pdir.mkdir()
        shutil.copy(polished["geometry"], pdir / "polished.xyz")
        polished_own, _ = classify(ctx, code, pdir, ctx.references, "polish-own", work)
        name, rmsd = single_answer(polished_own)
        hit_polish = int(name != "unassigned")
        polished_cross, _ = classify(ctx, other(code), pdir, ctx.references, "polish-cross", work)
        row["polished_isomer"] = name
        row["polished_rmsd_A"] = "" if rmsd is None else f"{rmsd:.6f}"
        row["hit_polish"] = hit_polish
        row["cross_isomer"], row["cross_rmsd_A"] = single_answer(polished_cross)
        hit = int(hit_frames or hit_polish)
    row["hit"] = hit
    if row["status"] == "error":
        row["status"] = "ok"


def classify_single(ctx, code, frames_dir, name, work):
    rows, _ = classify(ctx, code, frames_dir, ctx.single_references[name], "frames-" + name, work)
    return rows


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def record_inputs(ctx):
    rows = []
    entries = [
        ("start_geometry", ctx.water),
        ("binary_A", ctx.binaries["A"]),
        ("binary_B", ctx.binaries["B"]),
        ("engine", ctx.engine),
        ("engine_shim", ctx.shim),
        ("crest_driver", ctx.driver),
        ("crest", ctx.crest),
    ]
    for name in REFERENCES:
        entries.append((f"reference_{name}", ctx.references / f"{name}.xyz"))
    for label, path in entries:
        rows.append({"input": label, "file": path.name, "bytes": path.stat().st_size, "sha256": sha256_of(path)})
    return rows


def open_writer(path, fields):
    fresh = not path.exists() or path.stat().st_size == 0
    handle = open(path, "a", newline="")
    writer = csv.DictWriter(handle, fieldnames=fields)
    if fresh:
        writer.writeheader()
    return handle, writer


def parse_seed_list(text):
    seeds = []
    for part in text.split(","):
        if "-" in part:
            low, high = part.split("-")
            seeds.extend(range(int(low), int(high) + 1))
        else:
            seeds.append(int(part))
    return seeds


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary-a", required=True, help="water_hexamer example built from code A")
    parser.add_argument("--binary-b", required=True, help="water_hexamer example built from code B")
    parser.add_argument("--engine", required=True, help="GFN2 engine library")
    parser.add_argument("--shim", required=True, help="engine shim library")
    parser.add_argument("--driver", required=True, help="CREST call driver")
    parser.add_argument("--crest", required=True, help="CREST binary")
    parser.add_argument("--water-xyz", required=True, help="start geometry of every seed")
    parser.add_argument("--references-dir", required=True, help="directory with prism.xyz, cage.xyz, book.xyz")
    parser.add_argument("--out-dir", required=True, help="where the CSV files are written")
    parser.add_argument("--scratch-dir", required=True, help="working space for the jobs")
    parser.add_argument("--core", type=int, required=True, help="CPU the whole run is pinned to")
    parser.add_argument("--seeds", required=True, help="seed list, for example 0-191")
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--budget", type=int, default=4000, help="engine calls of the full arms")
    parser.add_argument("--search-budget", type=int, default=3382, help="engine calls before the polish of the fair arms")
    parser.add_argument("--polish-budget", type=int, default=618, help="engine calls of the polish")
    parser.add_argument("--polish-gtol", type=float, default=5e-4, help="eV per angstrom")
    parser.add_argument("--polish-history", type=int, default=10)
    parser.add_argument("--search-timeout", type=float, default=900)
    parser.add_argument("--crest-timeout", type=float, default=600)
    parser.add_argument("--polish-timeout", type=float, default=300)
    parser.add_argument("--classify-timeout", type=float, default=120)
    args = parser.parse_args()

    ctx = Context(args)
    arms = [arm for arm in args.arms.split(",") if arm]
    for arm in arms:
        if arm not in ARMS:
            parser.error(f"unknown arm {arm}")
    if args.search_budget + args.polish_budget != args.budget:
        parser.error("search budget plus polish budget must equal the budget")
    seeds = parse_seed_list(args.seeds)

    os.sched_setaffinity(0, {args.core})
    ctx.out.mkdir(parents=True, exist_ok=True)
    ctx.scratch.mkdir(parents=True, exist_ok=True)
    for name in REFERENCES:
        directory = ctx.scratch / f"reference-{name}"
        directory.mkdir(exist_ok=True)
        shutil.copy(ctx.references / f"{name}.xyz", directory / f"{name}.xyz")
        ctx.single_references[name] = directory

    inputs = record_inputs(ctx)
    inputs_path = ctx.out / "inputs.csv"
    if not inputs_path.exists():
        with open(inputs_path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["input", "file", "bytes", "sha256"])
            writer.writeheader()
            writer.writerows(inputs)

    runs_handle, runs = open_writer(ctx.out / "runs.csv", RUN_FIELDS)
    frames_handle, frame_writer = open_writer(ctx.out / "crest_frames.csv", FRAME_FIELDS)
    errors_path = ctx.out / "errors.log"
    started = time.time()
    for seed in seeds:
        order = CODES if seed % 2 == 0 else CODES[::-1]
        for arm in arms:
            for position, code in enumerate(order):
                ctx.job_index += 1
                row = empty_row(seed, code, arm, ctx.job_index, position)
                work = Path(tempfile.mkdtemp(dir=ctx.scratch, prefix="job-"))
                try:
                    if arm in ("anneal", "polish_fair"):
                        job_anneal(ctx, row, work, arm == "polish_fair")
                    else:
                        job_crest(ctx, row, work, arm == "crest_fair", frame_writer)
                except Exception as error:
                    row["status"] = f"error: {type(error).__name__}"
                    row["notes"] = str(error)[:200]
                    with open(errors_path, "a") as handle:
                        handle.write(f"seed {seed} {code} {arm}\n{traceback.format_exc()}\n")
                finally:
                    shutil.rmtree(work, ignore_errors=True)
                if row["wall_s"] != "" and row["in_s"] != "":
                    row["out_s"] = round(float(row["wall_s"]) - float(row["in_s"]), 6)
                    row["wall_s"] = round(float(row["wall_s"]), 6)
                    row["in_s"] = round(float(row["in_s"]), 6)
                runs.writerow(row)
                runs_handle.flush()
                frames_handle.flush()
                print(
                    f"seed {seed} {code} {arm} status={row['status']} hit={row['hit']} "
                    f"calls={row['calls']} wall={row['wall_s']} elapsed={time.time() - started:.0f}s",
                    flush=True,
                )
    meta = {
        "arms": arms,
        "seeds": args.seeds,
        "budget": args.budget,
        "search_budget": args.search_budget,
        "polish_budget": args.polish_budget,
        "polish_gtol_eV_A": args.polish_gtol,
        "polish_history": args.polish_history,
        "core": args.core,
        "affinity": sorted(os.sched_getaffinity(0)),
        "wall_s": round(time.time() - started, 1),
    }
    (ctx.out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    runs_handle.close()
    frames_handle.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
