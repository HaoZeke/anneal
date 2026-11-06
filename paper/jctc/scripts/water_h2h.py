#!/usr/bin/env python3
"""Serial water hexamer comparison of anneal and CREST.

CREST runs at its documented default, Geometries, Frequencies, and
Noncovalent interactions version 2, on one thread.
The fair anneal budget for that seed is the number of xtb energy
evaluations CREST produced.
The start geometry is the anneal rigid repack for that seed.
A second anneal uses the named paper budget of 4000 evaluations.
"""

import math
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

from water_census import label_free_rmsd

ROOT = Path("paper/jctc")
LOGS = ROOT / "logs" / "water"
SCRATCH = ROOT / "logs" / "scratch" / "water"
SEEDS = range(6)
MOLECULES = 6
PAPER_BUDGET = 4000
FORCE_BAR = 0.0514221
GFN2_ACCURACY = 0.01


def git_stamp():
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    tree = subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], text=True).strip()
    return head, tree


def type7_quantile(values, probability):
    ordered = sorted(values)
    count = len(ordered)
    if count == 0:
        return float("nan")
    if count == 1:
        return float(ordered[0])
    position = (count - 1) * probability + 1.0
    lower = int(math.floor(position))
    fraction = position - lower
    left = ordered[lower - 1]
    if fraction == 0.0 or lower >= count:
        return float(left)
    right = ordered[lower]
    return float(left + fraction * (right - left))


def type7_iqr(values):
    return type7_quantile(values, 0.75) - type7_quantile(values, 0.25)


def check_type7():
    sample = [1, 2, 3, 4, 5, 6]
    if abs(type7_quantile(sample, 0.25) - 2.25) > 1e-12:
        raise SystemExit("type 7 lower quartile is wrong")
    if abs(type7_quantile(sample, 0.75) - 4.75) > 1e-12:
        raise SystemExit("type 7 upper quartile is wrong")
    if abs(type7_iqr(sample) - 2.5) > 1e-12:
        raise SystemExit("type 7 interquartile range is wrong")


def assert_paper_budget():
    text = Path("src/catalog/molecular.rs").read_text()
    water = re.search(r"WATER_HEXAMER_BUDGET: usize = (\d+)", text)
    gfn2 = re.search(r"GFN2_HEXAMER_BUDGET: usize = WATER_HEXAMER_BUDGET", text)
    if not water or int(water.group(1)) != PAPER_BUDGET or not gfn2:
        raise SystemExit("named paper budget is not 4000")


def cpu_description():
    model = "unknown"
    family = ""
    model_number = ""
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            model = line.split(":", 1)[1].strip()
        elif line.startswith("cpu family"):
            family = line.split(":", 1)[1].strip()
        elif line.startswith("model"):
            model_number = line.split(":", 1)[1].strip()
    model = model.replace("(R)", "").replace("(TM)", "").replace("®", "")
    model = re.sub(r"\s+", " ", model).strip()
    return f"{model}, family {family}, model {model_number}"


SOURCE_HEAD = "unset"
SOURCE_TREE = "unset"


def write_header(handle, command, head=None, tree=None):
    head = SOURCE_HEAD if head is None else head
    tree = SOURCE_TREE if tree is None else tree
    host = subprocess.check_output(["nproc", "--all"], text=True).strip()
    machine = subprocess.check_output(["uname", "-m"], text=True).strip()
    date = subprocess.check_output(["date"], text=True).rstrip("\n")
    handle.write(
        "\n".join(
            [
                date,
                machine,
                host,
                cpu_description(),
                "clone=https://github.com/HaoZeke/anneal.git",
                f"source_commit={head}",
                f"source_tree={tree}",
                "cwd=.",
                "TMPDIR=paper/jctc/logs/scratch",
                f"command={command}",
                "",
            ]
        )
    )


def scrub(text):
    text = re.sub(
        r"(?<![\w])/(?:home|tmp|opt|agent|workspace|usr|proc|lib|lib64|var|etc)(?:/[\w.+@=-]+)+",
        "path",
        text,
    )
    for key in (
        "XTB_BIN",
        "CREST_BIN",
        "ANNEAL_WATER_BIN",
        "RGPOT_XTB_ENGINE",
        "LD_LIBRARY_PATH",
        "PATH",
        "HOME",
        "TMPDIR",
    ):
        value = os.environ.get(key)
        if value:
            text = text.replace(value, key.lower())
    return text


def child_env():
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["GOTO_NUM_THREADS"] = "1"
    env["TMPDIR"] = "paper/jctc/logs/scratch"
    env["ANNEAL_SOURCE_COMMIT"] = SOURCE_HEAD
    env["ANNEAL_SOURCE_TREE"] = SOURCE_TREE
    return env


def md_length_lines(text):
    kept = []
    for line in text.splitlines():
        lowered = line.lower()
        if "md simulation time" in lowered or "timestep dt" in lowered or "md length" in lowered:
            kept.append(line.strip())
    return kept


def xyz_coordinates(path):
    rows = []
    for line in Path(path).read_text(errors="ignore").splitlines()[2:]:
        parts = line.split()
        if len(parts) < 4:
            continue
        rows.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return rows


def alignment_rmsd(reference, trial):
    """Root-mean-square deviation after translating and rotating `trial` onto `reference`."""
    import numpy as np

    left = np.asarray(xyz_coordinates(reference), dtype=float)
    right = np.asarray(xyz_coordinates(trial), dtype=float)
    if left.shape != right.shape or left.size == 0:
        return None
    left = left - left.mean(axis=0)
    right = right - right.mean(axis=0)
    covariance = right.T @ left
    unitary, _, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(unitary @ vt) < 0.0:
        correction[2, 2] = -1.0
    rotation = unitary @ correction @ vt
    aligned = right @ rotation
    gap = aligned - left
    return float(np.sqrt((gap * gap).sum() / left.shape[0]))


def dynamics_steps(text):
    steps = 0
    time_ps = None
    dt_fs = None
    for line in text.splitlines():
        if "MD simulation time" in line:
            time_ps = float(line.split(":")[1].split()[0])
        elif "timestep dt" in line:
            dt_fs = float(line.split(":")[1].split()[0])
        elif "completed successfully" in line and time_ps and dt_fs:
            steps += int(round(time_ps * 1000.0 / dt_fs))
            time_ps = None
            dt_fs = None
    return steps


def read_opt_steps(seed):
    path = LOGS / f"crest-seed{seed}-optsteps.txt"
    if not path.exists():
        return 0
    for line in path.read_text().splitlines():
        if line.startswith("opt_steps="):
            return int(line.split("=", 1)[1])
    return 0


def run_crest(work, log_path, seed):
    xtb = os.environ["XTB_BIN"]
    crest = os.environ["CREST_BIN"]
    wrapper_dir = work / "path"
    wrapper_dir.mkdir(parents=True, exist_ok=True)
    counter = work / "xtb-calls"
    wrapper = wrapper_dir / "xtb"
    wrapper.write_text(
        "#!/bin/sh\n"
        "out=./xtb-one.out\n"
        f"'{xtb}' \"$@\" >\"$out\" 2>&1\n"
        "rc=$?\n"
        "cat \"$out\"\n"
        "python3 - \"$out\" << 'PY'\n"
        "import re, sys\n"
        "text = open(sys.argv[1], errors='ignore').read()\n"
        "cycles = len(re.findall(r'(?m)^\\s*\\* total energy\\b', text))\n"
        "if not cycles:\n"
        "    cycles = len(re.findall(r'(?m):: total energy\\b', text))\n"
        "open('xtb-calls', 'a').write(str(cycles) + '\\n')\n"
        "PY\n"
        "rm -f \"$out\"\n"
        "exit $rc\n"
    )
    wrapper.chmod(0o755)
    env = child_env()
    env["PATH"] = f"{wrapper_dir}:{env.get('PATH', '')}"
    env["TMPDIR"] = "."
    if counter.exists():
        counter.unlink()
    command = "crest start.xyz -T 1"
    started = time.perf_counter()
    proc = subprocess.run(
        [crest, "start.xyz", "-T", "1"],
        cwd=work,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    wall = time.perf_counter() - started
    dynamics = dynamics_steps(proc.stdout)
    time.sleep(0.5)
    opt_steps = read_opt_steps(seed)
    calls = dynamics + opt_steps
    best = work / "crest_best.xyz"
    with log_path.open("w") as handle:
        write_header(handle, command)
        handle.write(f"code=crest molecules=6 seed={seed}\n")
        handle.write("defaults=crest iMTD-GC -gfn2 -T 1\n")
        handle.write(f"calls={calls}\n")
        handle.write(f"dynamics_steps={dynamics}\n")
        handle.write(f"opt_steps={opt_steps}\n")
        for line in md_length_lines(proc.stdout):
            handle.write(f"md_length_line={line}\n")
        handle.write("call_counter=MD steps from the log plus Etot lines in crestopt.log\n")
        handle.write(f"wall_s={wall:.6f}\n")
        handle.write("best=" + ("crest_best.xyz" if best.exists() else "missing") + "\n")
        handle.write(scrub(proc.stdout))
        if proc.stdout and not proc.stdout.endswith("\n"):
            handle.write("\n")
        handle.write(f"exit_code={proc.returncode}\n")
    return proc.returncode, calls, wall, best if best.exists() else None


def anneal_start(binary, seed, outdir):
    env = child_env()
    env["H2H_START_ONLY"] = "1"
    env["RGPOT_XTB_ENGINE"] = os.environ["RGPOT_XTB_ENGINE"]
    subprocess.run(
        [binary, str(MOLECULES), "1", str(seed), str(outdir)],
        cwd=".",
        env=env,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def anneal_run(binary, seed, budget, outdir, log_path, label):
    env = child_env()
    env.pop("H2H_START_ONLY", None)
    env.pop("H2H_AUDIT_XYZ", None)
    env["RGPOT_XTB_ENGINE"] = os.environ["RGPOT_XTB_ENGINE"]
    command = f"gfn2_water_h2h {MOLECULES} {budget} {seed} {outdir.name}"
    started = time.perf_counter()
    proc = subprocess.run(
        [binary, str(MOLECULES), str(budget), str(seed), str(outdir)],
        cwd=".",
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    wall = time.perf_counter() - started
    meta = (outdir / "run.txt").read_text() if (outdir / "run.txt").exists() else ""
    with log_path.open("w") as handle:
        write_header(handle, command)
        handle.write(f"code=anneal molecules={MOLECULES} seed={seed} budget={budget}\n")
        handle.write(f"arm={label}\n")
        handle.write("defaults=recommended_molecular\n")
        handle.write("water_arm=recommended\n")
        handle.write(f"wall_s={wall:.6f}\n")
        handle.write(meta)
        handle.write(scrub(proc.stdout))
        if not meta.endswith("\n") and meta:
            handle.write("\n")
        handle.write(f"exit_code={proc.returncode}\n")
    return proc.returncode, wall


def audit_geometry(binary, seed, xyz, outdir, log_path):
    env = child_env()
    env["H2H_AUDIT_XYZ"] = str(xyz)
    env["RGPOT_XTB_ENGINE"] = os.environ["RGPOT_XTB_ENGINE"]
    command = f"gfn2_water_h2h {MOLECULES} 1 {seed} {outdir.name}"
    started = time.perf_counter()
    proc = subprocess.run(
        [binary, str(MOLECULES), "1", str(seed), str(outdir)],
        cwd=".",
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    wall = time.perf_counter() - started
    meta = (outdir / "force-audit.txt").read_text() if (outdir / "force-audit.txt").exists() else ""
    text = scrub(proc.stdout)
    with log_path.open("w") as handle:
        write_header(handle, command)
        handle.write(f"code=force-audit seed={seed}\n")
        handle.write(f"wall_s={wall:.6f}\n")
        handle.write(meta)
        handle.write(text)
        if text and not text.endswith("\n"):
            handle.write("\n")
        handle.write(f"exit_code={proc.returncode}\n")
    return proc.returncode, text


def single_point(xyz):
    xtb = os.environ["XTB_BIN"]
    proc = subprocess.run(
        [xtb, str(xyz), "--gfn", "2", "--sp", "--acc", "0.01", "--iterations", "500"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env={**child_env(), "OMP_NUM_THREADS": "1"},
    )
    energy = re.search(r"TOTAL ENERGY\s+([+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)\s+Eh", proc.stdout)
    window = re.search(
        r"-> SCF convergence\s+([+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)",
        proc.stdout,
    )
    if proc.returncode != 0 or not energy or not window:
        return None, None, scrub(proc.stdout)
    return float(energy.group(1)), float(window.group(1)), scrub(proc.stdout)


def finite_number(text):
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def parse_force(text):
    match = re.search(
        r"FORCE_AUDIT seed \d+ quench_gradient (\S+) bar (\S+) (PASS|FAIL)",
        text,
    )
    if not match:
        return None, False
    value = finite_number(match.group(1))
    bar = finite_number(match.group(2))
    passed = match.group(3) == "PASS" and value is not None and bar == FORCE_BAR and value <= FORCE_BAR
    return value, passed


def log_field(text, name):
    match = re.search(rf"(?m)^{re.escape(name)}=([^\n]*)$", text)
    if match:
        return match.group(1).strip()
    match = re.search(rf"(?m)^{re.escape(name)}: ([^\n]*)$", text)
    if match:
        return match.group(1).strip()
    return ""


def write_host_topology():
    host = subprocess.check_output(["nproc", "--all"], text=True).strip()
    machine = subprocess.check_output(["uname", "-m"], text=True).strip()
    text = "\n".join(
        [
            f"machine={machine}",
            f"cores={host}",
            f"cpu={cpu_description()}",
            "threads_per_run=1",
            "omp=1",
            "",
        ]
    )
    if "(R)" in text or "®" in text:
        raise SystemExit("cpu line still carries a trademark mark")
    (LOGS / "HOST-TOPOLOGY").write_text(text)


def command_output(argv):
    proc = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return scrub(proc.stdout)


def file_sha256(path):
    proc = subprocess.run(["sha256sum", path], text=True, stdout=subprocess.PIPE, check=True)
    return proc.stdout.split()[0]


def write_versions():
    crest = os.environ["CREST_BIN"]
    xtb = os.environ["XTB_BIN"]
    engine = os.environ["RGPOT_XTB_ENGINE"]
    lines = [
        f"crest_sha256={file_sha256(crest)}",
        f"xtb_sha256={file_sha256(xtb)}",
        f"engine_sha256={file_sha256(engine)}",
        "crest_reference_sha256=57857d81dd5bc363081c285d756b18aad02679abce52ae797a38069ab774324b",
        "gfn2_accuracy=0.01",
        "paper_budget=4000",
        "force_bar_eV_per_A=0.0514221",
        "force_bar_Ha_per_Bohr=0.001",
    ]
    banner = command_output([crest, "--version"])
    version_line = ""
    for line in banner.splitlines():
        if "Version 3.0.2" in line:
            version_line = line.strip()
    lines.append(f"crest_banner={version_line}")
    xtb_banner = command_output([xtb, "--version"])
    for line in xtb_banner.splitlines():
        if "xtb version" in line:
            lines.append("xtb_banner=" + line.strip().lstrip("* ").strip())
    (LOGS / "VERSIONS").write_text("\n".join(lines) + "\n")


def arm_row(seed, arm, calls, wall, success, energy, force, force_pass, window, delta, energy_pass, objective, exit_code, rmsd, signed=None, free_rmsd=None):
    def cell(value):
        if value is None:
            return "nan"
        if isinstance(value, float):
            return f"{value:.10g}"
        return str(value)

    return [
        str(seed),
        arm,
        cell(calls),
        cell(wall),
        str(int(success)),
        cell(energy),
        cell(force),
        "1" if force_pass else "0",
        cell(window),
        cell(delta),
        cell(rmsd),
        "1" if energy_pass else "0",
        cell(objective),
        cell(exit_code),
        cell(signed),
        cell(free_rmsd),
    ]


def write_numbers(rows):
    header = [
        "seed",
        "arm",
        "pes_calls",
        "wall_s",
        "success",
        "energy_Eh",
        "max_force_eV_per_A",
        "force_pass",
        "window_Eh",
        "abs_dE_Eh",
        "rmsd_A",
        "energy_pass",
        "objective_best_eV",
        "exit_code",
        "signed_dE_Eh",
        "rmsd_label_free_A",
    ]
    lines = [
        "# n=6 serial same host no best-of",
        "# quartile=Hyndman-Fan type 7",
        "# window=absolute energy gap at or below the SCF convergence printed by the scoring single point",
        "# signed_dE_Eh=anneal minus crest; rmsd_label_free_A minimises over molecule and hydrogen labels and reflection",
        "# " + "\t".join(header),
    ]
    for row in rows:
        lines.append("\t".join(row))
    by_arm = {}
    for row in rows:
        by_arm.setdefault(row[1], []).append(row)
    lines.append("# summary arm n calls_median calls_iqr wall_median wall_iqr success_median success_iqr success_rate")
    for arm in ("crest", "fair", "paper4000"):
        group = by_arm.get(arm, [])
        calls = [float(item[2]) for item in group if item[2] != "nan"]
        walls = [float(item[3]) for item in group if item[3] != "nan"]
        success = [float(item[4]) for item in group]
        rate = sum(success) / len(success) if success else float("nan")
        lines.append(
            "summary\t{arm}\t{n}\t{cm:.10g}\t{ci:.10g}\t{wm:.10g}\t{wi:.10g}\t{sm:.10g}\t{si:.10g}\t{rate:.10g}".format(
                arm=arm,
                n=len(group),
                cm=type7_quantile(calls, 0.5) if calls else float("nan"),
                ci=type7_iqr(calls) if calls else float("nan"),
                wm=type7_quantile(walls, 0.5) if walls else float("nan"),
                wi=type7_iqr(walls) if walls else float("nan"),
                sm=type7_quantile(success, 0.5) if success else float("nan"),
                si=type7_iqr(success) if success else float("nan"),
                rate=rate,
            )
        )
    (LOGS / "NUMBERS.tsv").write_text("\n".join(lines) + "\n")


def finished_log(path, needle):
    if not path.exists():
        return False
    text = path.read_text(errors="ignore")
    return "exit_code=0" in text and needle in text


def main():
    global SOURCE_HEAD, SOURCE_TREE
    SOURCE_HEAD, SOURCE_TREE = git_stamp()
    check_type7()
    assert_paper_budget()
    binary = os.environ["ANNEAL_WATER_BIN"]
    for key in ("CREST_BIN", "XTB_BIN", "RGPOT_XTB_ENGINE", "ANNEAL_WATER_BIN"):
        if key not in os.environ:
            raise SystemExit(f"missing {key}")
    LOGS.mkdir(parents=True, exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    write_host_topology()
    write_versions()
    watcher = subprocess.Popen(
        ["python3", "paper/jctc/scripts/watch_crest_steps.py"],
        cwd=".",
    )
    rows = []
    try:
        for seed in SEEDS:
            fair_dir = LOGS / f"anneal-seed{seed}"
            paper_dir = LOGS / f"anneal-paper4000-seed{seed}"
            crest_audit_dir = LOGS / f"crest-audit-seed{seed}"
            fair_dir.mkdir(parents=True, exist_ok=True)
            paper_dir.mkdir(parents=True, exist_ok=True)
            crest_audit_dir.mkdir(parents=True, exist_ok=True)
            if not (fair_dir / "start.xyz").exists():
                anneal_start(binary, seed, fair_dir)
            crest_log = LOGS / f"crest-hexamer-seed{seed}.log"
            fair_log = LOGS / f"anneal-hexamer-seed{seed}.log"
            paper_log = LOGS / f"anneal-paper4000-seed{seed}.log"
            crest_force_log = LOGS / f"crest-force-seed{seed}.log"
            work = SCRATCH / f"crest-seed{seed}"
            if not finished_log(crest_log, "best=crest_best.xyz") or not (fair_dir / "crest_best.xyz").exists():
                if work.exists():
                    shutil.rmtree(work)
                work.mkdir(parents=True)
                shutil.copy(fair_dir / "start.xyz", work / "start.xyz")
                code, calls, wall, best = run_crest(work, crest_log, seed)
                print(f"crest seed {seed} exit {code} calls {calls} wall {wall:.1f}", flush=True)
                if best is not None:
                    shutil.copy(best, fair_dir / "crest_best.xyz")
            crest_text = crest_log.read_text(errors="ignore") if crest_log.exists() else ""
            calls = finite_number(log_field(crest_text, "calls"))
            crest_wall = finite_number(log_field(crest_text, "wall_s"))
            crest_exit = log_field(crest_text, "exit_code")
            best_xyz = fair_dir / "crest_best.xyz"
            if best_xyz.exists() and not finished_log(crest_force_log, "FORCE_AUDIT"):
                audit_geometry(binary, seed, best_xyz, crest_audit_dir, crest_force_log)
            crest_force_text = crest_force_log.read_text(errors="ignore") if crest_force_log.exists() else ""
            crest_force, crest_force_pass = parse_force(crest_force_text)
            crest_energy = None
            window = None
            if best_xyz.exists():
                crest_energy, window, crest_sp = single_point(best_xyz)
                (LOGS / f"crest-sp-seed{seed}.txt").write_text(crest_sp)
            crest_success = int(
                crest_exit == "0"
                and best_xyz.exists()
                and crest_energy is not None
                and crest_force_pass
            )
            rows.append(
                arm_row(
                    seed,
                    "crest",
                    calls,
                    crest_wall,
                    crest_success,
                    crest_energy,
                    crest_force,
                    crest_force_pass,
                    window,
                    0.0 if crest_energy is not None else None,
                    crest_energy is not None,
                    None,
                    crest_exit or "nan",
                    0.0 if best_xyz.exists() else None,
                    0.0 if crest_energy is not None else None,
                    0.0 if best_xyz.exists() else None,
                )
            )
            if calls is None or calls < 1:
                print(f"fair seed {seed} skipped, crest calls missing", flush=True)
                rows.append(arm_row(seed, "fair", None, None, 0, None, None, False, window, None, False, None, "nan", None))
            elif not finished_log(fair_log, "FORCE_AUDIT"):
                status, awall = anneal_run(binary, seed, int(calls), fair_dir, fair_log, "fair")
                print(f"anneal fair seed {seed} exit {status} budget {int(calls)} wall {awall:.1f}", flush=True)
            fair_text = fair_log.read_text(errors="ignore") if fair_log.exists() else ""
            fair_calls = finite_number(log_field(fair_text, "energy_calls"))
            fair_wall = finite_number(log_field(fair_text, "wall_s"))
            fair_exit = log_field(fair_text, "exit_code")
            fair_objective = finite_number(log_field(fair_text, "objective_best_eV"))
            fair_force, fair_force_pass = parse_force(fair_text)
            fair_xyz = fair_dir / "best.xyz"
            fair_energy = None
            fair_delta = None
            fair_rmsd = None
            fair_free_rmsd = None
            fair_signed = None
            fair_energy_pass = False
            if fair_xyz.exists() and best_xyz.exists():
                fair_rmsd = alignment_rmsd(best_xyz, fair_xyz)
                fair_free_rmsd = label_free_rmsd(best_xyz, fair_xyz)
            if fair_xyz.exists() and crest_energy is not None:
                fair_energy, fair_window, fair_sp = single_point(fair_xyz)
                (LOGS / f"anneal-sp-seed{seed}.txt").write_text(fair_sp)
                if window is None:
                    window = fair_window
                if fair_energy is not None and window is not None:
                    fair_signed = fair_energy - crest_energy
                    fair_delta = abs(fair_signed)
                    fair_energy_pass = fair_delta <= window
            fair_success = int(
                fair_exit == "0"
                and fair_xyz.exists()
                and fair_objective is not None
                and fair_energy_pass
                and fair_force_pass
            )
            if calls is not None and calls >= 1:
                rows.append(
                    arm_row(
                        seed,
                        "fair",
                        fair_calls,
                        fair_wall,
                        fair_success,
                        fair_energy,
                        fair_force,
                        fair_force_pass,
                        window,
                        fair_delta,
                        fair_energy_pass,
                        fair_objective,
                        fair_exit or "nan",
                        fair_rmsd,
                        fair_signed,
                        fair_free_rmsd,
                    )
                )
            if not finished_log(paper_log, "FORCE_AUDIT"):
                status, pwall = anneal_run(binary, seed, PAPER_BUDGET, paper_dir, paper_log, "paper4000")
                print(f"anneal paper seed {seed} exit {status} budget {PAPER_BUDGET} wall {pwall:.1f}", flush=True)
            paper_text = paper_log.read_text(errors="ignore") if paper_log.exists() else ""
            paper_calls = finite_number(log_field(paper_text, "energy_calls"))
            paper_wall = finite_number(log_field(paper_text, "wall_s"))
            paper_exit = log_field(paper_text, "exit_code")
            paper_objective = finite_number(log_field(paper_text, "objective_best_eV"))
            paper_force, paper_force_pass = parse_force(paper_text)
            paper_xyz = paper_dir / "best.xyz"
            paper_energy = None
            paper_delta = None
            paper_rmsd = None
            paper_free_rmsd = None
            paper_signed = None
            paper_energy_pass = False
            if paper_xyz.exists() and best_xyz.exists():
                paper_rmsd = alignment_rmsd(best_xyz, paper_xyz)
                paper_free_rmsd = label_free_rmsd(best_xyz, paper_xyz)
            if paper_xyz.exists() and crest_energy is not None:
                paper_energy, paper_window, paper_sp = single_point(paper_xyz)
                (LOGS / f"anneal-paper-sp-seed{seed}.txt").write_text(paper_sp)
                if window is None:
                    window = paper_window
                if paper_energy is not None and window is not None:
                    paper_signed = paper_energy - crest_energy
                    paper_delta = abs(paper_signed)
                    paper_energy_pass = paper_delta <= window
            paper_success = int(
                paper_exit == "0"
                and paper_xyz.exists()
                and paper_objective is not None
                and paper_energy_pass
                and paper_force_pass
            )
            rows.append(
                arm_row(
                    seed,
                    "paper4000",
                    paper_calls,
                    paper_wall,
                    paper_success,
                    paper_energy,
                    paper_force,
                    paper_force_pass,
                    window,
                    paper_delta,
                    paper_energy_pass,
                    paper_objective,
                    paper_exit or "nan",
                    paper_rmsd,
                    paper_signed,
                    paper_free_rmsd,
                )
            )
            with (LOGS / f"pair-seed{seed}.log").open("w") as handle:
                write_header(handle, f"pair-seed {seed}")
                handle.write(f"seed={seed}\n")
                handle.write(f"crest_calls={calls if calls is not None else 'nan'}\n")
                handle.write(f"crest_wall_s={crest_wall if crest_wall is not None else 'nan'}\n")
                handle.write(f"crest_Eh={crest_energy if crest_energy is not None else 'nan'}\n")
                handle.write(f"crest_max_force={crest_force if crest_force is not None else 'nan'}\n")
                handle.write(f"crest_success={crest_success}\n")
                handle.write(f"fair_calls={fair_calls if fair_calls is not None else 'nan'}\n")
                handle.write(f"fair_wall_s={fair_wall if fair_wall is not None else 'nan'}\n")
                handle.write(f"fair_Eh={fair_energy if fair_energy is not None else 'nan'}\n")
                handle.write(f"fair_objective_eV={fair_objective if fair_objective is not None else 'nan'}\n")
                handle.write(f"fair_max_force={fair_force if fair_force is not None else 'nan'}\n")
                handle.write(f"fair_success={fair_success}\n")
                handle.write(f"paper_calls={paper_calls if paper_calls is not None else 'nan'}\n")
                handle.write(f"paper_wall_s={paper_wall if paper_wall is not None else 'nan'}\n")
                handle.write(f"paper_Eh={paper_energy if paper_energy is not None else 'nan'}\n")
                handle.write(
                    f"paper_objective_eV={paper_objective if paper_objective is not None else 'nan'}\n"
                )
                handle.write(f"paper_max_force={paper_force if paper_force is not None else 'nan'}\n")
                handle.write(f"paper_success={paper_success}\n")
                handle.write(f"window_Eh={window if window is not None else 'nan'}\n")
                handle.write("window_definition=abs(anneal_Eh-crest_Eh)<=scf_convergence\n")
                handle.write("window_source=SCF convergence printed by xtb --gfn 2 --sp --acc 0.01\n")
                handle.write(f"gfn2_accuracy={GFN2_ACCURACY}\n")
                handle.write(f"force_bar={FORCE_BAR}\n")
                handle.write("score=xtb --gfn 2 --sp --acc 0.01 --iterations 500\n")
                fair_start = fair_dir / "start.xyz"
                paper_start = paper_dir / "start.xyz"
                if fair_start.exists() and paper_start.exists():
                    same_start = fair_start.read_bytes() == paper_start.read_bytes()
                    handle_note = "1" if same_start else "0"
                else:
                    handle_note = "missing"
                handle.write(f"fair_rmsd_A={fair_rmsd if fair_rmsd is not None else 'nan'}\n")
                handle.write(f"paper_rmsd_A={paper_rmsd if paper_rmsd is not None else 'nan'}\n")
                handle.write(f"fair_rmsd_label_free_A={fair_free_rmsd if fair_free_rmsd is not None else 'nan'}\n")
                handle.write(f"paper_rmsd_label_free_A={paper_free_rmsd if paper_free_rmsd is not None else 'nan'}\n")
                handle.write(f"fair_signed_dE_Eh={fair_signed if fair_signed is not None else 'nan'}\n")
                handle.write(f"paper_signed_dE_Eh={paper_signed if paper_signed is not None else 'nan'}\n")
                handle.write("rmsd_definition=Kabsch alignment of all atoms against crest_best.xyz\n")
                handle.write(f"shared_start={handle_note}\n")
                handle.write("exit_code=0\n")
            print(
                f"pair seed {seed} crest {crest_success} fair {fair_success} paper {paper_success}",
                flush=True,
            )
            write_numbers(rows)
    finally:
        watcher.terminate()
        try:
            watcher.wait(timeout=5)
        except subprocess.TimeoutExpired:
            watcher.kill()
    write_numbers(rows)
    print("numbers", LOGS / "NUMBERS.tsv", flush=True)


if __name__ == "__main__":
    main()
