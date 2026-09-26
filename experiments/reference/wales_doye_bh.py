"""Wales and Doye's basin hopping, counted on the same ledger as anneal.

The point is a comparison the crate cannot flatter itself in. Every number
reported by `anneal` so far is that crate against itself, so a claim that
anything in it is an advance has no support. This is the published algorithm,
implemented from its own description, charged one unit per energy or gradient
evaluation exactly as the Rust ledger charges, so the two are comparable
without either being trusted about the other.

The algorithm, from Wales and Doye, J. Phys. Chem. A 101, 5111 (1997):

  * The transformed surface is E~(x) = E(Q(x)) with Q a local minimisation, so
    the chain moves on quenched energies.
  * Single-step: the chain carries the quenched geometry, not the geometry
    before the quench. White and Mayne (1998) measure this variant as the
    better one, and it is what "plain basin hopping" refers to.
  * Proposal: displace every coordinate uniformly by at most `step`. With
    `--angular`, a fraction of moves instead take the worst-bound atom and
    replace its radius with the largest in the cluster at a fresh random
    angle, which is the move the paper used to reach the decahedral minima at
    75 points: "choosing random theta and phi spherical polar coordinates for
    the atom in question, taking the origin at the center of mass and
    replacing the radius with the maximum value in the cluster". Comparing
    only against the plain variant would compare against the paper's weakest
    protocol.
  * Acceptance: Metropolis on the quenched energies at fixed T.
  * `step` is adjusted toward an acceptance ratio of one half, which is the
    paper's own prescription.
  * A container prevents evaporation: an atom further from the centre of mass
    than `radius` is moved back along its own radius.

The quench is SciPy's L-BFGS-B, which relaxes 400 perturbed 75-point structures
in 273 evaluations each and converges, and is therefore a stronger minimiser
than the crate's own rather than a weaker one. Nothing here is handicapped.
"""

from __future__ import annotations

import argparse
import os

# Pinned before numpy is imported. The arrays here are a few hundred elements
# wide, so a threaded BLAS spends more time synchronising than computing, and a
# campaign of concurrent seeds oversubscribes the machine several times over:
# nineteen of these drew thirty-nine cores' worth of CPU on a thirty-two core
# host, which slows every one of them.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import math
import sys

import numpy as np
from scipy.optimize import minimize

REFERENCE = {13: -44.326801, 38: -173.928427, 55: -279.248470,
             75: -397.492331, 98: -543.665361}


class Ledger:
    """One unit per energy or gradient evaluation, and a hard budget."""

    def __init__(self, budget: int):
        self.budget = budget
        self.spent = 0
        self.best = math.inf
        self.best_x = None
        self.first_hit = None

    def charge(self) -> bool:
        if self.spent >= self.budget:
            return False
        self.spent += 1
        return True

    def record(self, e: float, x: np.ndarray, target: float | None):
        if e < self.best:
            self.best = e
            self.best_x = x.copy()
            if target is not None and self.first_hit is None and e < target + 1e-4:
                self.first_hit = self.spent


class Exhausted(Exception):
    pass


def lj(x: np.ndarray):
    p = x.reshape(-1, 3)
    d = p[:, None, :] - p[None, :, :]
    r2 = (d * d).sum(-1)
    n = len(p)
    iu = np.triu_indices(n, 1)
    rr = r2[iu]
    inv6 = 1.0 / rr ** 3
    e = float((4.0 * (inv6 * inv6 - inv6)).sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        c = 24.0 * (2.0 / r2 ** 6 - 1.0 / r2 ** 3) / r2
    np.fill_diagonal(c, 0.0)
    g = -(c[:, :, None] * d).sum(1)
    return e, g.ravel()


def angular_move(x: np.ndarray, rng) -> np.ndarray:
    """Wales and Doye's angular move on the worst-bound atom.

    The atom with the highest pair energy is thrown to the far edge of the
    cluster at a random angle. A much larger step than a uniform displacement,
    and the one the 1997 paper credits with reaching the decahedral minima.
    """
    p = x.reshape(-1, 3).copy()
    n = len(p)
    c = p.mean(axis=0)
    d = p[:, None, :] - p[None, :, :]
    r2 = (d * d).sum(-1)
    np.fill_diagonal(r2, np.inf)
    inv6 = 1.0 / r2 ** 3
    per_atom = (4.0 * (inv6 * inv6 - inv6)).sum(axis=1)
    worst = int(np.argmax(per_atom))
    rmax = float(np.linalg.norm(p - c, axis=1).max())
    ct = rng.uniform(-1.0, 1.0)
    st = math.sqrt(max(0.0, 1.0 - ct * ct))
    phi = rng.uniform(0.0, 2.0 * math.pi)
    p[worst] = c + rmax * np.array([st * math.cos(phi), st * math.sin(phi), ct])
    return p.ravel()


def contain(x: np.ndarray, radius: float) -> np.ndarray:
    """Pull escaped atoms back inside the container.

    Without it a cluster evaporates: an atom that leaves feels no force and the
    run optimises a smaller cluster while reporting the larger one's count.
    """
    p = x.reshape(-1, 3).copy()
    p -= p.mean(axis=0)
    r = np.linalg.norm(p, axis=1)
    out = r > radius
    if out.any():
        p[out] *= (radius / r[out])[:, None]
    return p.ravel()


def quench(x: np.ndarray, led: Ledger, maxiter: int = 3000):
    def fg(v):
        if not led.charge():
            raise Exhausted
        return lj(v)

    try:
        r = minimize(fg, x, jac=True, method="L-BFGS-B",
                     options=dict(maxiter=maxiter, maxfun=6 * maxiter,
                                  ftol=1e-16, gtol=1e-8))
    except Exhausted:
        return None, None
    return float(r.fun), r.x


def basin_hopping(n: int, budget: int, seed: int, temperature: float = 0.8,
                  target: float | None = None, angular: float = 0.0):
    rng = np.random.default_rng(seed)
    radius = 2.5 * n ** (1.0 / 3.0)
    led = Ledger(budget)

    x = rng.uniform(-radius / 2, radius / 2, size=3 * n)
    x = contain(x, radius)
    e, x = quench(x, led, 3000)
    if e is None:
        return led, 0, 0.0
    led.record(e, x, target)

    step = 0.38
    hops = 0
    accepted = 0
    window = 0
    window_accepted = 0
    while led.spent < led.budget:
        if angular > 0.0 and rng.random() < angular:
            y = angular_move(x, rng)
        else:
            y = x + rng.uniform(-step, step, size=x.shape)
        y = contain(y, radius)
        ey, y = quench(y, led, 3000)
        if ey is None:
            break
        hops += 1
        window += 1
        led.record(ey, y, target)
        if ey < e or rng.random() < math.exp(-(ey - e) / temperature):
            # Single step: the chain carries the quenched geometry.
            e, x = ey, y
            accepted += 1
            window_accepted += 1
        # The paper's prescription: hold the acceptance ratio near one half.
        if window >= 50:
            ratio = window_accepted / window
            step *= 1.1 if ratio > 0.5 else 0.9
            step = min(max(step, 0.01), 2.0)
            window = window_accepted = 0
    return led, hops, accepted / max(1, hops)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("n", type=int)
    ap.add_argument("budget", type=int)
    ap.add_argument("seeds", type=int, default=8, nargs="?")
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--angular", type=float, default=0.0,
                    help="fraction of proposals that are angular moves")
    a = ap.parse_args()

    ref = REFERENCE.get(a.n)
    print(f"LJ{a.n}, budget {a.budget} charged evaluations, {a.seeds} seeds, "
          f"angular {a.angular}, reference {ref if ref is not None else 'none'}")
    solved = 0
    hits = []
    deepest = math.inf
    for s in range(a.seed0, a.seed0 + a.seeds):
        led, hops, acc = basin_hopping(a.n, a.budget, s, a.temperature, ref, a.angular)
        deepest = min(deepest, led.best)
        ok = ref is not None and led.best < ref + 1e-4
        solved += bool(ok)
        if led.first_hit is not None:
            hits.append(led.first_hit)
        print(f"  seed {s}: best {led.best:.6f}  hops {hops}  accept {acc:.3f}  "
              f"charged/hop {led.spent // max(1, hops)}  "
              f"first_hit {led.first_hit if led.first_hit else 'censored'}"
              f"{'  SOLVED' if ok else ''}")
        sys.stdout.flush()
    print(f"{solved}/{a.seeds} solved, deepest {deepest:.6f}")
    if len(hits) * 2 > a.seeds:
        print(f"first encounter: median {sorted(hits)[len(hits) // 2]} charged evaluations "
              f"({len(hits)} reached, {a.seeds - len(hits)} censored)")
    else:
        print(f"first encounter: no median, {a.seeds - len(hits)} of {a.seeds} runs censored")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
