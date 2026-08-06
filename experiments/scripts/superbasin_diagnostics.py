"""What the escape mechanism itself did, per arm.

A success count cannot tell a mechanism that works poorly from one that never
fired, so every quantity the refusal path produces is aggregated here.
"""

import pathlib
import re
import statistics

SB = re.compile(
    r"superbasin: (\d+) basins (\d+) transitions, (\d+) archived, "
    r"bias distortion ([-\d.naN]+)"
)
TOP = re.compile(
    r"top partition \[([^\]]*)\]\s+jumps (\d+) refused (\d+) \[([^\]]*)\] "
    r"worst revisits ([-\d.naN]+)\s+condition max ([\d.e+-]+) mean ([\w.e+-]+) "
    r"residual ([\d.e+-]+)\s+solve residual ([\d.e+-]+) exact (\d+)\s+"
    r"hops replaced (\d+)\s+improved (\d+) by ([-\d.]+)"
)
LEVEL = re.compile(
    r"level (\d+): (\d+) coarse states, largest lump (\d+), separation ([\d.naN]+)"
)


def num(v):
    try:
        f = float(v)
        return f if f == f else None
    except ValueError:
        return None


root = pathlib.Path("sbasin_logs")
for d in sorted(root.iterdir()):
    if not d.is_dir():
        continue
    for arm in ["base", "sbasin"]:
        logs = sorted(d.glob(f"{arm}_s*.log"))
        if not logs:
            continue
        nodes, edges, arch, dist = [], [], [], []
        jumps, refused, revisits, cond, condres, solveres = [], [], [], [], [], []
        hops_replaced, improved, gain, exact = [], [], [], []
        depths, top_biggest, top_share, seps = [], [], [], []
        kinds = {}
        for p in logs:
            t = p.read_text(errors="replace")
            m = SB.search(t)
            if not m:
                continue
            nodes.append(int(m.group(1)))
            edges.append(int(m.group(2)))
            arch.append(int(m.group(3)))
            v = num(m.group(4))
            if v is not None:
                dist.append(v)
            levels = LEVEL.findall(t)
            depths.append(len(levels))
            for lv in levels:
                s = num(lv[3])
                if s is not None:
                    seps.append(s)
            g = TOP.search(t)
            if g:
                parts = [int(x) for x in g.group(1).split(",") if x.strip()]
                if parts:
                    top_biggest.append(max(parts))
                    top_share.append(max(parts) / int(m.group(1)))
                jumps.append(int(g.group(2)))
                refused.append(int(g.group(3)))
                for item in g.group(4).split(","):
                    item = item.strip()
                    if not item:
                        continue
                    k, n = item.rsplit(" ", 1)
                    kinds[k] = kinds.get(k, 0) + int(n)
                r = num(g.group(5))
                if r is not None:
                    revisits.append(r)
                c = num(g.group(6))
                if c and c > 0:
                    cond.append(c)
                cr = num(g.group(8))
                if cr is not None and cr > 0:
                    condres.append(cr)
                sr = num(g.group(9))
                if sr is not None and sr > 0:
                    solveres.append(sr)
                exact.append(int(g.group(10)))
                hops_replaced.append(int(g.group(11)))
                improved.append(int(g.group(12)))
                gain.append(float(g.group(13)))

        def stat(v, f="{:.2f}"):
            if not v:
                return "-"
            return f.format(statistics.median(v)) + " med, " + f.format(max(v)) + " max"

        print(f"== {d.name} / {arm}  ({len(logs)} runs)")
        print(
            f"   graph: {stat(nodes, '{:.0f}')} basins | {stat(edges, '{:.0f}')} transitions | "
            f"archive {stat(arch, '{:.0f}')} | bias distortion {stat(dist, '{:.3f}')}"
        )
        print(
            f"   hierarchy: depth {stat(depths, '{:.0f}')} | largest top state "
            f"{stat(top_biggest, '{:.0f}')} basins | share {stat(top_share, '{:.3f}')} | "
            f"lump separation {stat(seps, '{:.1f}')}"
        )
        print(
            f"   escape: {sum(jumps)} jumps total ({stat(jumps, '{:.0f}')}) | "
            f"{sum(refused)} refusals | kinds {kinds}"
        )
        print(f"   trapping: revisits per state {stat(revisits, '{:.2f}')} against 2.00")
        print(
            f"   numerics: condition {stat(cond, '{:.4g}')} | gauss-seidel residual "
            f"{stat(condres, '{:.1e}')} | solve residual {stat(solveres, '{:.1e}')} | "
            f"{sum(exact)} exact solves"
        )
        print(
            f"   value: {sum(hops_replaced)} hops replaced ({stat(hops_replaced, '{:.0f}')}) | "
            f"{sum(improved)} jumps landed lower, by {sum(gain):.3f} total"
        )
        print()
