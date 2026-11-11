#!/usr/bin/env python3
"""One AGOX LJ38 seed. Hostname bench was set via unshare --uts."""

import os
import sys
import time
from pathlib import Path

print("wrapper_header hostname bench was set via unshare --uts", flush=True)
print(f"uts_hostname={os.uname().nodename}", flush=True)
print(
    "potential_counter=perf_counter around lj_grad; each call returns energy and gradient together",
    flush=True,
)
if os.uname().nodename != "bench":
    print("uts hostname is not bench", file=sys.stderr)
    sys.exit(2)

seed = sys.argv[1]
os.chdir("/work/anneal")
os.environ.update(
    {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMBA_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "BLIS_NUM_THREADS": "1",
        "OMP_PROC_BIND": "TRUE",
        "OMP_PLACES": "cores",
        "PYTHONUNBUFFERED": "1",
        "LJ_BOARD_ROOT": "results/lj-board",
        "HOME": "/work/home",
        "TMPDIR": "/work/scratch",
        "MPLCONFIGDIR": "/work/mpl",
        "NUMBA_CACHE_DIR": "/work/numba",
        "XDG_CACHE_HOME": "/work/cache",
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "safe.directory",
        "GIT_CONFIG_VALUE_0": "/work/anneal",
    }
)

sample = 200_000
t0 = time.perf_counter()
for _ in range(sample):
    time.perf_counter()
per_read = (time.perf_counter() - t0) / sample

sys.path.insert(0, "scripts/lj_board")
import run_arm

stats = {"calls": 0, "inside_s": 0.0}
orig = run_arm.lj_grad


def timed(x):
    started = time.perf_counter()
    result = orig(x)
    stats["inside_s"] += time.perf_counter() - started
    stats["calls"] += 1
    return result


run_arm.lj_grad = timed
sys.argv = ["scripts/lj_board/run_arm.py", "agox", "38", seed]
rc = run_arm.main()

log = Path("results/lj-board/agox") / f"lj38-seed{seed}.log"
fields = {}
for line in log.read_text().splitlines():
    for tok in line.split():
        if "=" in tok:
            key, value = tok.split("=", 1)
            fields[key] = value
wall = float(fields["wall_s"])
charged = int(fields["charged"])
inside = stats["inside_s"]
text = "\n".join(
    [
        f"energy_calls={stats['calls']}",
        f"gradient_calls={stats['calls']}",
        f"potential_calls={stats['calls']}",
        f"charged={charged}",
        f"inside_s={inside:.6f}",
        f"outside_s={wall - inside:.6f}",
        f"wall_s={wall:.6f}",
        f"perf_counter_read_s={per_read:.9e}",
        f"counter_overhead_inside_s={per_read * stats['calls']:.6f}",
        "fused=1",
        "note=lj_grad returns energy and gradient together. inside_s includes one perf_counter read per call.",
        "",
    ]
)
(Path("results/lj-board/agox") / f"lj38-seed{seed}.potential.txt").write_text(text)
print(text, end="", flush=True)
sys.exit(rc)
