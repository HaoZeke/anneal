"""Population annealing on the same Lennard-Jones ledger as basin hopping.

Hukushima and Iba, Phys. Rev. E 67, 066106 (2003), with the adaptive
temperature step used by Machta, Phys. Rev. E 82, 026704 (2010): the next
inverse temperature is the one that puts the effective sample size at half
the population. One independent population is one seed. Every energy or
gradient call is charged, including the final quenches. An unquenched
energy is not a hit. A hit is a quenched energy within 1e-4 of the
Cambridge reference.

The population is 100. Each generation gives every walker one Metropolis
trial that displaces every coordinate by at most 0.38, the Wales--Doye
step, at the current temperature. The ladder starts at T=0.8 and is not
allowed below T=0.05. Once the ladder is at that floor, generations
continue there until 500 evaluations per walker remain, and that remainder
is spent quenching the population with the same L-BFGS-B as
wales_doye_bh.py.
"""

from __future__ import annotations

import argparse
import math
import sys

import numpy as np

from wales_doye_bh import Exhausted, Ledger, REFERENCE, contain, lj, quench

N_WALKERS = 100
T_HIGH = 0.8
T_LOW = 0.05
STEP = 0.38
QUENCH_RESERVE_PER_WALKER = 500


def choose_dbeta(energies: np.ndarray, target_ess: float) -> float:
    """Smallest positive step whose effective sample size is at most the target."""
    de = energies - float(np.min(energies))

    def ess(dbeta: float) -> float:
        weights = np.exp(-dbeta * de)
        total = float(weights.sum())
        if total <= 0.0 or not math.isfinite(total):
            return 0.0
        weights /= total
        return 1.0 / float(np.sum(weights * weights))

    if ess(50.0) > target_ess:
        return 50.0
    low = 0.0
    high = 50.0
    for _ in range(40):
        mid = 0.5 * (low + high)
        if ess(mid) > target_ess:
            low = mid
        else:
            high = mid
    return high


def population_annealing(n: int, budget: int, seed: int):
    rng = np.random.default_rng(seed)
    radius = 2.5 * n ** (1.0 / 3.0)
    led = Ledger(budget)
    target = REFERENCE.get(n)
    xs = np.zeros((N_WALKERS, 3 * n))
    es = np.empty(N_WALKERS)
    alive = 0
    for i in range(N_WALKERS):
        x = contain(rng.uniform(-radius / 2, radius / 2, size=3 * n), radius)
        if not led.charge():
            break
        energy, _gradient = lj(x)
        xs[i] = x
        es[i] = energy
        alive += 1
    if alive == 0:
        return led, 0, T_HIGH
    xs = xs[:alive]
    es = es[:alive]
    beta = 1.0 / T_HIGH
    generations = 0
    while led.spent + alive < led.budget - QUENCH_RESERVE_PER_WALKER * alive:
        trial_x = np.empty_like(xs)
        trial_e = np.empty_like(es)
        for i in range(alive):
            proposal = contain(xs[i] + rng.uniform(-STEP, STEP, size=xs.shape[1]), radius)
            if not led.charge():
                trial_x[i] = xs[i]
                trial_e[i] = es[i]
                continue
            energy, _gradient = lj(proposal)
            temperature = 1.0 / beta
            if energy <= es[i] or rng.random() < math.exp(-(energy - es[i]) / temperature):
                trial_x[i] = proposal
                trial_e[i] = energy
            else:
                trial_x[i] = xs[i]
                trial_e[i] = es[i]
        xs = trial_x
        es = trial_e
        temperature = 1.0 / beta
        if temperature > T_LOW:
            dbeta = choose_dbeta(es, alive / 2.0)
            beta = min(beta + dbeta, 1.0 / T_LOW)
            weights = np.exp(-(beta - (1.0 / temperature)) * (es - float(np.min(es))))
            weights /= float(weights.sum())
            draw = rng.choice(alive, size=alive, replace=True, p=weights)
            xs = xs[draw]
            es = es[draw]
        generations += 1
    quenched = 0
    for i in range(alive):
        energy, coordinates = quench(xs[i], led, 3000)
        if energy is None or coordinates is None:
            break
        led.record(energy, coordinates, target)
        quenched += 1
    return led, generations, 1.0 / beta


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("n", type=int)
    parser.add_argument("budget", type=int)
    parser.add_argument("seeds", type=int, nargs="?", default=1)
    parser.add_argument("--seed0", type=int, default=0)
    args = parser.parse_args()
    ref = REFERENCE.get(args.n)
    print(
        f"LJ{args.n} population annealing, budget {args.budget}, "
        f"{args.seeds} seeds from {args.seed0}, walkers {N_WALKERS}, "
        f"T {T_HIGH} to {T_LOW}, step {STEP}, reference {ref}"
    )
    solved = 0
    deepest = math.inf
    for seed in range(args.seed0, args.seed0 + args.seeds):
        led, generations, final_t = population_annealing(args.n, args.budget, seed)
        deepest = min(deepest, led.best)
        ok = ref is not None and led.best < ref + 1e-4
        solved += bool(ok)
        print(
            f"  seed {seed}: best {led.best:.6f}  generations {generations}  "
            f"final_T {final_t:.4f}  charged {led.spent}  "
            f"first_hit {led.first_hit if led.first_hit else 'censored'}"
            f"{'  SOLVED' if ok else ''}"
        )
        sys.stdout.flush()
    print(f"{solved}/{args.seeds} solved, deepest {deepest:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
