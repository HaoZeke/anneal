# Copyright (c) anneal developers
"""Serial develop, develop again and r6 runs on one core, n=12, paper budgets."""
import csv, hashlib, os, struct, subprocess, sys, tempfile, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from target_tol import energy_hits  # noqa: E402

CELLS = [
    ("m55r6", "h2h_morse", -250.286609),
    ("lj38", "h2h_gmin_proof", -173.928427),
    ("m55r14", "h2h_morse", -220.646208),
    ("lj98", "h2h_gmin_proof", -543.665361),
    ("lj102", "h2h_gmin_proof", -569.363652),
    ("m80r14", "h2h_morse", -340.811371),
    ("lj75", "h2h_gmin_proof", -397.492331),
]
SEEDS = range(2000, 2012)
ARMS = {"A": ("/tmp/tgt6-A/release/examples", sys.argv[2]),
        "A2": ("/tmp/tgt6-A2/release/examples", sys.argv[2]),
        "R6": ("/tmp/tgt6-B/release/examples", sys.argv[3])}
ORDERS = [["A", "A2", "R6"], ["A2", "R6", "A"], ["R6", "A", "A2"]]
THREADS = ["OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "MKL_NUM_THREADS", "MKL_DOMAIN_NUM_THREADS",
           "OPENBLAS_NUM_THREADS", "RAYON_NUM_THREADS", "NUMBA_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "GOTO_NUM_THREADS", "TOKIO_WORKER_THREADS",
           "POLARS_MAX_THREADS", "TBB_NUM_THREADS"]
FIELDS = ["cell", "arm", "seed", "slot", "hit", "calls_to_hit", "charged", "best", "best_bits",
          "wall_s", "pot_s", "outside_s", "code_revision", "binary_sha256", "host", "core",
          "started_unix", "elapsed_s"]


def env():
    e = {"HOME": os.environ["HOME"], "PATH": os.environ["PATH"], "MKL_DYNAMIC": "FALSE",
         "OMP_PROC_BIND": "TRUE", "OMP_PLACES": "cores", "PYTHONHASHSEED": "0"}
    e.update({k: "1" for k in THREADS})
    return e


def main():
    rows_path = Path(sys.argv[1])
    logs = rows_path.parent / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    done = set()
    if rows_path.exists():
        done = {(r["cell"], r["arm"], r["seed"]) for r in csv.DictReader(rows_path.open())}
    else:
        with rows_path.open("w", newline="") as f:
            csv.writer(f, lineterminator="\n").writerow(FIELDS)
    host = "machine-" + hashlib.sha256(Path("/etc/machine-id").read_bytes()).hexdigest()[:16]
    shas = {(a, d): hashlib.sha256(Path(ARMS[a][0], d).read_bytes()).hexdigest()
            for a in ARMS for d in ("h2h_morse", "h2h_gmin_proof")}
    for cell, driver, ref in CELLS:
        for seed in SEEDS:
            order = ORDERS[seed % 3]
            for slot, arm in enumerate(order, 1):
                if (cell, arm, str(seed)) in done:
                    continue
                exe = Path(ARMS[arm][0], driver)
                with tempfile.TemporaryDirectory() as wd:
                    started = time.time()
                    p = subprocess.run(["taskset", "-c", "0", str(exe), "run", cell, str(seed)],
                                       cwd=wd, env=env(), capture_output=True, text=True)
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
                       ARMS[arm][1], shas[(arm, driver)], host, 0,
                       f"{started:.0f}", f"{elapsed:.3f}"]
                with rows_path.open("a", newline="") as f:
                    csv.writer(f, lineterminator="\n").writerow(row)
                print(time.strftime("%H:%M:%S"), cell, seed, arm, int(hit), row[5], f"{wall:.1f}s", flush=True)


if __name__ == "__main__":
    main()
