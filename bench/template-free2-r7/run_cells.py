# Copyright (c) anneal developers
"""Serial runs on one core at the drivers' paper budgets.

    run_cells.py ROWS CELLS SEEDS ARM [ARM ...]

CELLS is a comma list such as lj38,m55r14. SEEDS is FIRST:LAST, inclusive.
Each ARM is LABEL=EXAMPLES_DIR=REVISION[=ENV], where ENV is an optional
KEY:VALUE pair set for that arm only. Within a seed the arm order rotates
with the seed. The runner skips rows ROWS already has, so it resumes.
"""
import csv, hashlib, os, struct, subprocess, sys, tempfile, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from target_tol import energy_hits  # noqa: E402

REFERENCES = {
    "lj38": ("h2h_gmin_proof", -173.928427),
    "lj75": ("h2h_gmin_proof", -397.492331),
    "lj98": ("h2h_gmin_proof", -543.665361),
    "lj102": ("h2h_gmin_proof", -569.363652),
    "lj103": ("h2h_gmin_proof", -575.766131),
    "lj104": ("h2h_gmin_proof", -582.086642),
    "m38r14": ("h2h_morse", -144.321054),
    "m55r14": ("h2h_morse", -220.646208),
}
THREADS = ["OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "MKL_NUM_THREADS", "MKL_DOMAIN_NUM_THREADS",
           "OPENBLAS_NUM_THREADS", "RAYON_NUM_THREADS", "NUMBA_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "GOTO_NUM_THREADS", "TOKIO_WORKER_THREADS",
           "POLARS_MAX_THREADS", "TBB_NUM_THREADS"]
FIELDS = ["cell", "arm", "seed", "slot", "hit", "calls_to_hit", "charged", "best", "best_bits",
          "wall_s", "pot_s", "outside_s", "code_revision", "arm_env", "binary_sha256", "host",
          "core", "started_unix", "elapsed_s"]


def env(extra):
    e = {"HOME": os.environ["HOME"], "PATH": os.environ["PATH"], "MKL_DYNAMIC": "FALSE",
         "OMP_PROC_BIND": "TRUE", "OMP_PLACES": "cores", "PYTHONHASHSEED": "0"}
    e.update({k: "1" for k in THREADS})
    e.update(extra)
    return e


def main():
    rows_path = Path(sys.argv[1])
    cells = sys.argv[2].split(",")
    first, last = (int(v) for v in sys.argv[3].split(":"))
    arms = {}
    for spec in sys.argv[4:]:
        parts = spec.split("=")
        extra = dict([parts[3].split(":", 1)]) if len(parts) > 3 else {}
        arms[parts[0]] = (parts[1], parts[2], extra)
    labels = list(arms)
    logs = rows_path.parent / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    done = set()
    if rows_path.exists():
        done = {(r["cell"], r["arm"], r["seed"]) for r in csv.DictReader(rows_path.open())}
    else:
        with rows_path.open("w", newline="") as f:
            csv.writer(f, lineterminator="\n").writerow(FIELDS)
    host = "unrecorded"
    for cell in cells:
        driver, ref = REFERENCES[cell]
        for seed in range(first, last + 1):
            k = seed % len(labels)
            for slot, arm in enumerate(labels[k:] + labels[:k], 1):
                if (cell, arm, str(seed)) in done:
                    continue
                directory, revision, extra = arms[arm]
                exe = Path(directory, driver)
                sha = hashlib.sha256(exe.read_bytes()).hexdigest()
                with tempfile.TemporaryDirectory() as wd:
                    started = time.time()
                    p = subprocess.run(["taskset", "-c", "0", str(exe), "run", cell, str(seed)],
                                       cwd=wd, env=env(extra), capture_output=True, text=True)
                    elapsed = time.time() - started
                (logs / f"{cell}-{seed}-{arm}.log").write_text(
                    f"rc={p.returncode}\n--- stdout ---\n{p.stdout}\n--- stderr ---\n{p.stderr}")
                hdr = [l for l in p.stdout.splitlines() if l.startswith("system,")][-1].split(",")
                line = [l for l in p.stdout.splitlines() if l.startswith(f"{cell},anneal,{seed},")][0]
                r = dict(zip(hdr, line.split(",")))
                best = float(r["best"])
                hit = energy_hits(best, ref)
                wall, pot = float(r["wall_s"]), float(r["pot_s"])
                out = float(r["outside_s"]) if r.get("outside_s") else max(wall - pot, 0.0)
                bits = "0x%016x" % struct.unpack(">Q", struct.pack(">d", best))[0]
                row = [cell, arm, seed, slot, int(hit), r["evals_to_hit"] if hit else r["budget"],
                       r["charged"], r["best"], bits, f"{wall:.6f}", f"{pot:.6f}", f"{out:.6f}",
                       revision, ";".join(f"{a}={b}" for a, b in extra.items()), sha, host, 0,
                       f"{started:.0f}", f"{elapsed:.3f}"]
                with rows_path.open("a", newline="") as f:
                    csv.writer(f, lineterminator="\n").writerow(row)
                print(time.strftime("%H:%M:%S"), cell, seed, arm, int(hit), row[5],
                      f"{wall:.1f}s", flush=True)


if __name__ == "__main__":
    main()
