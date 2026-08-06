"""Paired comparison: the two arms share a seed and a random stream.

Control and treatment diverge only at a jump, so the seeds where they disagree
are exactly the seeds the mechanism changed. That is a sharper reading than two
independent proportions, and McNemar's exact test is the one that applies.
"""

import pathlib
import re
from math import comb

BEST = re.compile(r"^\s*seed \d+: best (-?\d+\.\d+)", re.M)
REF = {75: -397.492331, 98: -543.665361}


def solved(path, ref):
    t = path.read_text(errors="replace")
    b = BEST.findall(t)
    return bool(b) and float(b[0]) < ref + 1e-4


root = pathlib.Path("sbasin_logs")
for d in sorted(root.iterdir()):
    if not d.is_dir():
        continue
    n = 75 if "75" in d.name else 98
    ref = REF[n]
    both = only_c = only_t = neither = 0
    changed = []
    for s in range(24):
        pc = d / f"base_s{s}.log"
        pt = d / f"sbasin_s{s}.log"
        if not pc.exists() or not pt.exists():
            continue
        c, t = solved(pc, ref), solved(pt, ref)
        if c and t:
            both += 1
        elif c:
            only_c += 1
            changed.append((s, "control only"))
        elif t:
            only_t += 1
            changed.append((s, "sbasin only"))
        else:
            neither += 1
    disc = only_c + only_t
    # McNemar exact: binomial on the discordant pairs.
    if disc:
        k = min(only_c, only_t)
        p = min(
            1.0,
            2 * sum(comb(disc, i) for i in range(0, k + 1)) / 2 ** disc,
        )
    else:
        p = 1.0
    print(
        f"{d.name:<11} both {both:>2}  control only {only_c:>2}  sbasin only {only_t:>2}  "
        f"neither {neither:>2}   McNemar exact p = {p:.3f}"
    )
    if changed:
        print(f"             seeds that differ: {changed}")
