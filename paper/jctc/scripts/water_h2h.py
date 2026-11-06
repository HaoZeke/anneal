#!/usr/bin/env python3
"""Serial water hexamer comparison of anneal and CREST.

CREST runs at its documented default, Geometries, Frequencies, and
Noncovalent interactions version 2, on one thread.
The fair anneal budget for that seed is the number of xtb energy
evaluations CREST produced.
The start geometry is the anneal rigid repack for that seed.
A second anneal uses the named paper budget of 4000 evaluations.

Scoring follows the Papers B' bar. Every final, CREST and anneal alike,
is relaxed with the same xtb --opt vtight before the energies are
compared, and the comparison uses a symmetric 1e-8 Eh window on the
polished pair. An anneal minimum more than the window below CREST is a
lower minimum found by anneal and passes. A CREST answer that is not one
intact cluster is a CREST failure and is not the reference; the
reference is then the lowest intact polished minimum from either code
that is a confirmed minimum (force at or below the bar and no imaginary
vibrational mode). Success stays exit + energy + force. Both RMSD
columns are reported and neither is a gate.
"""

import math
import os
import re
import shutil
import socket
import subprocess
import time
from pathlib import Path

from water_census import HBOND_OO_A, label_free_rmsd, molecules, read_xyz

ROOT = Path("paper/jctc")
LOGS = ROOT / "logs" / "water"
SCRATCH = ROOT / "logs" / "scratch" / "water"
SEEDS = range(6)
MOLECULES = 6
PAPER_BUDGET = 4000
FORCE_BAR = 0.0514221
GFN2_ACCURACY = 0.01
WINDOW_EH = 1e-8
OH_BOND_A = 1.15
PAPER_CREST_SHA256 = "16c5de4578000dfd8ea5758da9f5c497778be0dd7a6a1c1e8933030479d0e56e"
PAPER_CREST_CONDA = "crest 3.0.2 h8876d29_1"
POLISH_ARGS = ["--gfn", "2", "--opt", "vtight", "--acc", "0.01", "--iterations", "500"]
HESSIAN_ARGS = ["--gfn", "2", "--hess", "--acc", "0.01", "--iterations", "500"]
DISSOCIATED = "CREST 3.0.2 returned a dissociated structure"


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
            f"host={socket.gethostname()}",
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
    crest_sha256 = file_sha256(crest)
    if crest_sha256 != PAPER_CREST_SHA256:
        raise SystemExit(f"CREST binary {crest_sha256} is not the paper pin {PAPER_CREST_SHA256}")
    lines = [
        f"crest_sha256={crest_sha256}",
        f"paper_pin_crest_sha256={PAPER_CREST_SHA256}",
        f"paper_pin_crest_conda={PAPER_CREST_CONDA}",
        f"xtb_sha256={file_sha256(xtb)}",
        f"engine_sha256={file_sha256(engine)}",
        "gfn2_accuracy=0.01",
        "paper_budget=4000",
        "force_bar_eV_per_A=0.0514221",
        "force_bar_Ha_per_Bohr=0.001",
        f"window_Eh={WINDOW_EH}",
        "polish=xtb " + " ".join(POLISH_ARGS),
        "hessian=xtb " + " ".join(HESSIAN_ARGS),
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


def cluster_intact(path):
    """(intact, farthest nearest-neighbour O...O in angstrom) for an xyz of whole waters.

    A cluster is intact when every oxygen keeps two hydrogens inside the
    O-H wall onset and the oxygens form one network under the Luzar and
    Chandler hydrogen-bond O...O distance.
    """
    import numpy as np

    mol = molecules(*read_xyz(path))
    if mol is None:
        return False, None
    oxygens = mol[:, 0, :]
    bonds = np.linalg.norm(mol[:, 1:, :] - mol[:, :1, :], axis=2)
    distances = np.linalg.norm(oxygens[:, None, :] - oxygens[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    nearest = float(distances.min(axis=1).max())
    seen = {0}
    frontier = [0]
    while frontier:
        here = frontier.pop()
        for there in np.nonzero(distances[here] < HBOND_OO_A)[0]:
            if int(there) not in seen:
                seen.add(int(there))
                frontier.append(int(there))
    return bool(bonds.max() <= OH_BOND_A and len(seen) == len(oxygens)), nearest


def xtb_in(work, xyz, args):
    xtb = os.environ["XTB_BIN"]
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    shutil.copy(xyz, work / "in.xyz")
    env = child_env()
    env["TMPDIR"] = "."
    return subprocess.run(
        [xtb, "in.xyz", *args],
        cwd=work,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def hessian_modes(xyz, work):
    """(imaginary vibrational modes, lowest vibrational frequency in cm-1, xtb's own count) from xtb --hess.

    The six modes nearest zero are the projected translations and rotations.
    Every other negative frequency counts, with no cutoff.
    """
    proc = xtb_in(work, xyz, HESSIAN_ARGS)
    (work / "hess.out").write_text(scrub(proc.stdout))
    blocks = re.findall(r"projected vibrational frequencies[^\n]*\n((?:\s*eigval :[^\n]*\n)+)", proc.stdout)
    xtb_count = re.search(r"# imaginary freq\.\s+(\d+)", proc.stdout)
    if proc.returncode != 0 or not blocks:
        return None, None, None
    values = [float(value) for line in blocks[-1].splitlines() for value in line.split(":", 1)[1].split()]
    vibrations = sorted(values, key=abs)[6:]
    imaginary = sum(1 for value in vibrations if value < 0.0)
    return imaginary, min(vibrations), int(xtb_count.group(1)) if xtb_count else None


def polish_candidate(binary, seed, arm, xyz):
    """Relax one final with the shared vtight polish and check that it is a true minimum."""
    work = LOGS / f"polish-seed{seed}-{arm}"
    proc = xtb_in(work, xyz, POLISH_ARGS)
    (work / "opt.out").write_text(scrub(proc.stdout))
    polished = work / "xtbopt.xyz"
    converged = proc.returncode == 0 and "GEOMETRY OPTIMIZATION CONVERGED" in proc.stdout and polished.exists()
    raw_intact, raw_nearest = cluster_intact(xyz)
    found = {
        "arm": arm,
        "xyz": polished if polished.exists() else None,
        "converged": converged,
        "raw_intact": raw_intact,
        "raw_nearest_oo_A": raw_nearest,
        "intact": False,
        "nearest_oo_A": None,
        "energy": None,
        "scf": None,
        "force": None,
        "force_pass": False,
        "imaginary": None,
        "lowest_cm1": None,
        "xtb_imaginary": None,
        "confirmed": False,
    }
    if polished.exists():
        polished_intact, found["nearest_oo_A"] = cluster_intact(polished)
        found["intact"] = raw_intact and polished_intact
        found["energy"], found["scf"], sp = single_point(polished)
        (work / "sp.out").write_text(sp)
        _, audit = audit_geometry(binary, seed, polished, work, work / "force.log")
        found["force"], found["force_pass"] = parse_force(audit)
        found["imaginary"], found["lowest_cm1"], found["xtb_imaginary"] = hessian_modes(polished, work / "hessian")
    found["confirmed"] = bool(
        found["converged"]
        and found["intact"]
        and found["energy"] is not None
        and found["force_pass"]
        and found["imaginary"] == 0
    )
    with (work / "polish.txt").open("w") as handle:
        for key, value in found.items():
            handle.write(f"{key}={value if key != 'xyz' else (value.name if value else 'missing')}\n")
    return found


def choose_reference(candidates, crest_intact):
    """The polished minimum every arm is scored against, and the rule that picked it."""
    crest = candidates.get("crest")
    if crest_intact and crest is not None and crest["confirmed"]:
        return crest, "crest_polished"
    pool = [
        found
        for found in candidates.values()
        if found["confirmed"] and (found["arm"] != "crest" or crest_intact)
    ]
    if not pool:
        return None, "no_confirmed_intact_minimum"
    return min(pool, key=lambda found: found["energy"]), "lowest_intact_confirmed_polished"


def energy_verdict(found, reference):
    """(energy pass, polished signed gap to the reference, plain-text reason)."""
    if found is None or found["energy"] is None:
        return False, None, "no polished energy"
    if reference is None:
        return False, None, "no confirmed intact reference"
    signed = found["energy"] - reference["energy"]
    if abs(signed) <= WINDOW_EH:
        if reference["arm"] == found["arm"]:
            return True, signed, "is the reference"
        return True, signed, f"matches the {reference['arm']} polished reference within {WINDOW_EH:g} Eh"
    if signed < 0.0:
        if found["confirmed"] and reference["arm"] == "crest" and found["arm"] != "crest":
            return True, signed, f"lower minimum found by anneal, {-signed:.3e} Eh below CREST polished"
        return False, signed, f"{-signed:.3e} Eh below the reference but not a confirmed minimum"
    return False, signed, f"{signed:.3e} Eh above the {reference['arm']} polished reference"


def anneal_raw(log_path, xyz, crest_xyz, crest_energy, sp_path):
    text = log_path.read_text(errors="ignore") if log_path.exists() else ""
    force, force_pass = parse_force(text)
    found = {
        "calls": finite_number(log_field(text, "energy_calls")),
        "wall": finite_number(log_field(text, "wall_s")),
        "exit": log_field(text, "exit_code"),
        "objective": finite_number(log_field(text, "objective_best_eV")),
        "force": force,
        "force_pass": force_pass,
        "xyz": xyz if xyz.exists() else None,
        "energy": None,
        "signed": None,
        "rmsd": None,
        "free_rmsd": None,
    }
    if xyz.exists() and crest_xyz.exists():
        found["rmsd"] = alignment_rmsd(crest_xyz, xyz)
        found["free_rmsd"] = label_free_rmsd(crest_xyz, xyz)
    if xyz.exists():
        found["energy"], _, sp = single_point(xyz)
        sp_path.write_text(sp)
        if found["energy"] is not None and crest_energy is not None:
            found["signed"] = found["energy"] - crest_energy
    return found


HEADER = [
    "seed",
    "arm",
    "pes_calls",
    "wall_s",
    "success",
    "exit_code",
    "force_pass",
    "max_force_eV_per_A",
    "energy_pass",
    "window_Eh",
    "raw_energy_Eh",
    "polished_energy_Eh",
    "signed_dE_raw_Eh",
    "signed_dE_polished_Eh",
    "reference_arm",
    "reference_rule",
    "reference_Eh",
    "intact",
    "nearest_oo_A",
    "polish_converged",
    "polished_max_force_eV_per_A",
    "imaginary_modes",
    "lowest_mode_cm1",
    "confirmed_minimum",
    "rmsd_A",
    "rmsd_label_free_A",
    "rmsd_raw_A",
    "rmsd_label_free_raw_A",
    "objective_best_eV",
    "verdict",
]


def cell(value):
    if value is None:
        return "nan"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        return f"{value:.10g}"
    return str(value)


def arm_row(seed, arm, raw, polished, reference, rule, success, energy_pass, signed, verdict):
    reference_xyz = reference["xyz"] if reference is not None else None
    rmsd = None
    free_rmsd = None
    if polished is not None and polished["xyz"] is not None and reference_xyz is not None:
        rmsd = alignment_rmsd(reference_xyz, polished["xyz"])
        free_rmsd = label_free_rmsd(reference_xyz, polished["xyz"])
    polished = polished or {}
    values = {
        "seed": seed,
        "arm": arm,
        "pes_calls": raw.get("calls"),
        "wall_s": raw.get("wall"),
        "success": bool(success),
        "exit_code": raw.get("exit") or "nan",
        "force_pass": bool(raw.get("force_pass")),
        "max_force_eV_per_A": raw.get("force"),
        "energy_pass": bool(energy_pass),
        "window_Eh": WINDOW_EH,
        "raw_energy_Eh": raw.get("energy"),
        "polished_energy_Eh": polished.get("energy"),
        "signed_dE_raw_Eh": raw.get("signed"),
        "signed_dE_polished_Eh": signed,
        "reference_arm": reference["arm"] if reference is not None else "none",
        "reference_rule": rule,
        "reference_Eh": reference["energy"] if reference is not None else None,
        "intact": bool(polished.get("intact")),
        "nearest_oo_A": polished.get("raw_nearest_oo_A"),
        "polish_converged": bool(polished.get("converged")),
        "polished_max_force_eV_per_A": polished.get("force"),
        "imaginary_modes": polished.get("imaginary"),
        "lowest_mode_cm1": polished.get("lowest_cm1"),
        "confirmed_minimum": bool(polished.get("confirmed")),
        "rmsd_A": rmsd,
        "rmsd_label_free_A": free_rmsd,
        "rmsd_raw_A": raw.get("rmsd"),
        "rmsd_label_free_raw_A": raw.get("free_rmsd"),
        "objective_best_eV": raw.get("objective"),
        "verdict": verdict,
    }
    return [cell(values[name]) for name in HEADER]


def write_numbers(rows):
    lines = [
        "# n=6 serial same host no best-of",
        "# quartile=Hyndman-Fan type 7",
        "# bar=Papers B': success = exit + energy + force; force is the reported answer's maximum force at 0.001 Eh/bohr",
        "# polish=xtb " + " ".join(POLISH_ARGS) + " on every final, CREST and anneal alike",
        "# score=xtb --gfn 2 --sp --acc 0.01 --iterations 500 on the raw and on the polished final",
        f"# window={WINDOW_EH:g} Eh symmetric on the polished pair; anneal more than the window below CREST polished is a lower minimum found by anneal and passes",
        "# reference=CREST polished when CREST returned an intact cluster that is a confirmed minimum, else the lowest intact confirmed polished minimum from either code",
        f"# intact=every oxygen keeps two hydrogens within {OH_BOND_A} A and the oxygens form one network under {HBOND_OO_A} A; nearest_oo_A is the farthest nearest-neighbour O...O of the raw final",
        "# confirmed_minimum=intact, polish converged, polished maximum force at or below the bar, no imaginary vibrational mode in xtb --hess",
        "# signed_dE_raw_Eh=anneal minus crest on the unpolished finals; signed_dE_polished_Eh=arm minus reference after the polish",
        "# rmsd_A=labelled Kabsch of the polished final on the polished reference; rmsd_label_free_A minimises over molecule and hydrogen labels and reflection; the _raw columns are the unpolished finals on crest_best.xyz; no RMSD column is a gate",
        "# " + "\t".join(HEADER),
    ]
    for row in rows:
        lines.append("\t".join(row))
    by_arm = {}
    for row in rows:
        by_arm.setdefault(row[1], []).append(row)
    lines.append("# summary arm n calls_median calls_iqr wall_median wall_iqr success_median success_iqr success_rate passes")
    for arm in ("crest", "fair", "paper4000"):
        group = by_arm.get(arm, [])
        calls = [float(item[2]) for item in group if item[2] != "nan"]
        walls = [float(item[3]) for item in group if item[3] != "nan"]
        success = [float(item[4]) for item in group]
        rate = sum(success) / len(success) if success else float("nan")
        lines.append(
            "summary\t{arm}\t{n}\t{cm:.10g}\t{ci:.10g}\t{wm:.10g}\t{wi:.10g}\t{sm:.10g}\t{si:.10g}\t{rate:.10g}\t{passes}".format(
                arm=arm,
                n=len(group),
                cm=type7_quantile(calls, 0.5) if calls else float("nan"),
                ci=type7_iqr(calls) if calls else float("nan"),
                wm=type7_quantile(walls, 0.5) if walls else float("nan"),
                wi=type7_iqr(walls) if walls else float("nan"),
                sm=type7_quantile(success, 0.5) if success else float("nan"),
                si=type7_iqr(success) if success else float("nan"),
                rate=rate,
                passes=int(sum(success)),
            )
        )
    (LOGS / "NUMBERS.tsv").write_text("\n".join(lines) + "\n")


def finished_log(path, needle):
    if not path.exists():
        return False
    text = path.read_text(errors="ignore")
    return "exit_code=0" in text and needle in text


def anneal_verdict(raw, energy_pass, reason):
    if raw.get("exit") != "0":
        return f"anneal exited with code {raw.get('exit') or 'nan'}"
    if raw.get("xyz") is None or raw.get("objective") is None:
        return "anneal reported no answer"
    if not raw.get("force_pass"):
        return "reported answer is above the force bar"
    return ("pass: " if energy_pass else "") + reason


def crest_verdict(crest, polished, energy_pass, reason):
    if crest["exit"] != "0" or crest["xyz"] is None:
        return f"CREST 3.0.2 exited with code {crest['exit'] or 'nan'} and no structure"
    if not polished["raw_intact"]:
        return DISSOCIATED
    if not polished["intact"]:
        return "CREST 3.0.2 structure dissociated under the shared polish"
    if crest["energy"] is None:
        return "CREST 3.0.2 structure has no single-point energy"
    if not crest["force_pass"]:
        return "CREST 3.0.2 answer is above the force bar"
    return ("pass: " if energy_pass else "") + reason


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
            best_xyz = fair_dir / "crest_best.xyz"
            if best_xyz.exists() and not finished_log(crest_force_log, "FORCE_AUDIT"):
                audit_geometry(binary, seed, best_xyz, crest_audit_dir, crest_force_log)
            crest_force_text = crest_force_log.read_text(errors="ignore") if crest_force_log.exists() else ""
            crest_force, crest_force_pass = parse_force(crest_force_text)
            crest = {
                "calls": calls,
                "wall": finite_number(log_field(crest_text, "wall_s")),
                "exit": log_field(crest_text, "exit_code"),
                "objective": None,
                "force": crest_force,
                "force_pass": crest_force_pass,
                "xyz": best_xyz if best_xyz.exists() else None,
                "energy": None,
                "signed": None,
                "rmsd": None,
                "free_rmsd": None,
            }
            scf = None
            if best_xyz.exists():
                crest["energy"], scf, crest_sp = single_point(best_xyz)
                (LOGS / f"crest-sp-seed{seed}.txt").write_text(crest_sp)
                crest["signed"] = 0.0 if crest["energy"] is not None else None
                crest["rmsd"] = 0.0
                crest["free_rmsd"] = 0.0
            fair_ran = calls is not None and calls >= 1
            if not fair_ran:
                print(f"fair seed {seed} skipped, crest calls missing", flush=True)
            elif not finished_log(fair_log, "FORCE_AUDIT"):
                status, awall = anneal_run(binary, seed, int(calls), fair_dir, fair_log, "fair")
                print(f"anneal fair seed {seed} exit {status} budget {int(calls)} wall {awall:.1f}", flush=True)
            fair = anneal_raw(fair_log, fair_dir / "best.xyz", best_xyz, crest["energy"], LOGS / f"anneal-sp-seed{seed}.txt")
            if not finished_log(paper_log, "FORCE_AUDIT"):
                status, pwall = anneal_run(binary, seed, PAPER_BUDGET, paper_dir, paper_log, "paper4000")
                print(f"anneal paper seed {seed} exit {status} budget {PAPER_BUDGET} wall {pwall:.1f}", flush=True)
            paper = anneal_raw(
                paper_log, paper_dir / "best.xyz", best_xyz, crest["energy"], LOGS / f"anneal-paper-sp-seed{seed}.txt"
            )

            raws = {"crest": crest, "fair": fair, "paper4000": paper}
            candidates = {}
            for arm, raw in raws.items():
                if raw["xyz"] is not None and (arm != "fair" or fair_ran):
                    candidates[arm] = polish_candidate(binary, seed, arm, raw["xyz"])
            crest_intact = crest["exit"] == "0" and "crest" in candidates and candidates["crest"]["intact"]
            reference, rule = choose_reference(candidates, crest_intact)
            outcome = {}
            for arm, raw in raws.items():
                polished = candidates.get(arm)
                energy_pass, signed, reason = energy_verdict(polished, reference)
                if arm == "crest":
                    if polished is None:
                        verdict = crest_verdict(crest, {"raw_intact": False, "intact": False}, False, reason)
                    else:
                        verdict = crest_verdict(crest, polished, energy_pass, reason)
                    success = verdict.startswith("pass: ")
                elif arm == "fair" and not fair_ran:
                    verdict = "skipped, crest calls missing"
                    success = False
                else:
                    verdict = anneal_verdict(raw, energy_pass, reason)
                    success = bool(
                        raw["exit"] == "0"
                        and raw["xyz"] is not None
                        and raw["objective"] is not None
                        and energy_pass
                        and raw["force_pass"]
                    )
                outcome[arm] = (success, energy_pass, signed, verdict)
                rows.append(arm_row(seed, arm, raw, polished, reference, rule, success, energy_pass, signed, verdict))
            with (LOGS / f"pair-seed{seed}.log").open("w") as handle:
                write_header(handle, f"pair-seed {seed}")
                handle.write(f"seed={seed}\n")
                for arm, raw in raws.items():
                    polished = candidates.get(arm) or {}
                    success, energy_pass, signed, verdict = outcome[arm]
                    for key in ("calls", "wall", "exit", "objective", "force", "energy", "signed", "rmsd", "free_rmsd"):
                        value = raw.get(key)
                        handle.write(f"{arm}_{key}={value if value is not None else 'nan'}\n")
                    for key in ("converged", "raw_intact", "raw_nearest_oo_A", "intact", "energy", "force", "imaginary", "lowest_cm1", "xtb_imaginary", "confirmed"):
                        value = polished.get(key)
                        handle.write(f"{arm}_polished_{key}={value if value is not None else 'nan'}\n")
                    handle.write(f"{arm}_signed_dE_polished_Eh={signed if signed is not None else 'nan'}\n")
                    handle.write(f"{arm}_energy_pass={int(energy_pass)}\n")
                    handle.write(f"{arm}_success={int(success)}\n")
                    handle.write(f"{arm}_verdict={verdict}\n")
                handle.write(f"reference_arm={reference['arm'] if reference is not None else 'none'}\n")
                handle.write(f"reference_rule={rule}\n")
                handle.write(f"reference_Eh={reference['energy'] if reference is not None else 'nan'}\n")
                handle.write(f"window_Eh={WINDOW_EH}\n")
                handle.write(f"scf_convergence_Eh={scf if scf is not None else 'nan'}\n")
                handle.write("window_definition=abs(arm_polished_Eh-reference_polished_Eh)<=1e-8, or anneal below CREST polished\n")
                handle.write("polish=xtb " + " ".join(POLISH_ARGS) + "\n")
                handle.write("score=xtb --gfn 2 --sp --acc 0.01 --iterations 500\n")
                handle.write(f"gfn2_accuracy={GFN2_ACCURACY}\n")
                handle.write(f"force_bar={FORCE_BAR}\n")
                fair_start = fair_dir / "start.xyz"
                paper_start = paper_dir / "start.xyz"
                if fair_start.exists() and paper_start.exists():
                    handle_note = "1" if fair_start.read_bytes() == paper_start.read_bytes() else "0"
                else:
                    handle_note = "missing"
                handle.write("rmsd_definition=Kabsch alignment of all atoms; polished against the polished reference, raw against crest_best.xyz\n")
                handle.write(f"shared_start={handle_note}\n")
                handle.write("exit_code=0\n")
            print(
                "pair seed {} crest {} fair {} paper {} reference {}".format(
                    seed, int(outcome["crest"][0]), int(outcome["fair"][0]), int(outcome["paper4000"][0]),
                    reference["arm"] if reference is not None else "none",
                ),
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
