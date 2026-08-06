"""Reproduces the separation table in the docs for `anneal_core::tensor_id`.

Two questions, in order.

1. The exact one. Bloom's homometric pair {0,1,4,10,12,17} and
   {0,1,8,11,13,17} on a line share the multiset of pairwise distances
   {1..13, 16, 17} and are not congruent, so no merge radius on a sorted
   distance spectrum tells them apart. What do the kernel spectra do?

2. The one a search meets. Quench random starts to Lennard-Jones minima and
   ask how far apart the closest pair of distinct minima is, in units of the
   descriptor's own response to a 0.02 displacement plus a relabelling and a
   rotation. A merge radius must sit above that response and below the closest
   pair, so the worst-case ratio is the width of the band the radius has to
   find, and a ratio below 1 means there is no band.

The Rust side is `TripletSpectrum`, with the same sigma and weight; this is
where the numbers quoted in its documentation come from.

Run: python3 experiments/tensor_id_separation.py
"""

import sys

import numpy as np

rng = np.random.default_rng(11)

# Matches TripletSpectrum::new.
SIGMA = 2.5
WEIGHT = 2.5
# Displacement scale a quench leaves behind, per coordinate.
JITTER = 0.02


def say(*a):
    print(*a)
    sys.stdout.flush()


# ------------------------------------------------------------- descriptors
def dists(X):
    return np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)


def sorted_pairs(X):
    d = dists(X)
    return np.sort(d[np.triu_indices(len(X), 1)])


def kern(X, s=SIGMA):
    return np.exp(-0.5 * (dists(X) / s) ** 2)


def triplet(A):
    """Mode-3 contraction of T_ijk = A_ij A_jk A_ik, scaled by n / tr(A^2)."""
    A2 = A @ A
    return A * A2 * (len(A) / np.trace(A2))


def spectra(X, s=SIGMA):
    A = kern(X, s)
    return np.concatenate(
        [np.sort(np.linalg.eigvalsh(A)), np.sort(np.linalg.eigvalsh(triplet(A)))]
    )


def joint(X, s=SIGMA, w=WEIGHT):
    return np.concatenate([sorted_pairs(X), w * spectra(X, s)])


def mode_sv(X, s=SIGMA):
    """Exact HOSVD mode singular values of T, order N^4. Reference only."""
    A = kern(X, s)
    n = len(A)
    T = (A[:, :, None] * A[None, :, :] * A[:, None, :]).reshape(n, n * n)
    return np.sqrt(np.maximum(np.sort(np.linalg.eigvalsh(T @ T.T)), 0.0))


def moments(X, s=SIGMA):
    """Nine power traces, the cheap compression that does not work."""
    n = len(X)
    A = kern(X, s)
    A2 = A @ A
    M = A * A2 * (n / np.trace(A2))
    M2, M3 = M @ M, (M @ M) @ M
    fr = lambda P, Q: float((P * Q).sum())
    raw = [(2, fr(A, A)), (3, fr(A2, A)), (4, fr(A2, A2))]
    raw += [
        (1, float(np.trace(M))),
        (2, fr(M, M)),
        (3, fr(M2, M)),
        (4, fr(M2, M2)),
        (5, fr(M3, M2)),
        (6, fr(M3, M3)),
    ]
    return np.array([np.sign(t) * (abs(t) / n) ** (1.0 / k) for k, t in raw])


def scramble(X, eps=JITTER):
    Y = X + rng.normal(scale=eps, size=X.shape)
    Y = Y[rng.permutation(len(Y))]
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return Y @ Q


def jitter_of(f, X, reps=16):
    b = f(X)
    return max(np.linalg.norm(f(scramble(X)) - b) for _ in range(reps))


# ---------------------------------------------------------- Lennard-Jones
def lj(X):
    d = X[:, None, :] - X[None, :, :]
    r2 = (d**2).sum(-1)
    np.fill_diagonal(r2, np.inf)
    s6 = r2 ** (-3)
    e = 2.0 * (s6 * s6 - s6).sum()
    c = (24.0 * (2.0 * s6 * s6 - s6) / r2)[:, :, None]
    return e, -2.0 * (c * d).sum(1)


def fire(X, steps=8000, gtol=1e-5):
    v, dt, alpha, npos = np.zeros_like(X), 0.002, 0.1, 0
    e, g = lj(X)
    for _ in range(steps):
        f = -g
        gn = np.linalg.norm(f)
        if gn < gtol:
            break
        if (f * v).sum() > 0:
            npos += 1
            v = (1 - alpha) * v + alpha * (f / gn) * np.linalg.norm(v)
            if npos > 5:
                dt, alpha = min(dt * 1.1, 0.02), alpha * 0.99
        else:
            npos, v, dt, alpha = 0, np.zeros_like(X), dt * 0.5, 0.1
        v = v + dt * f
        step = dt * v
        cap = np.abs(step).max()
        if cap > 0.05:
            step *= 0.05 / cap
        X = X + step
        e, g = lj(X)
    return X, e, np.linalg.norm(g)


def minima(n, want):
    """Distinct converged minima, told apart by energy."""
    out, seed = [], 0
    while len(out) < want and seed < 1500:
        r = np.random.default_rng(seed)
        seed += 1
        X = r.normal(size=(n, 3))
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        X *= r.uniform(0, 1, (n, 1)) ** (1 / 3) * 0.74 * n ** (1 / 3)
        X, e, gn = fire(X)
        if gn > 1e-3 or not np.isfinite(e):
            continue
        if all(abs(e - e2) > 1e-4 for _, e2 in out):
            out.append((X, e))
    return out


def z_stats(f, ms, reps=16):
    v = [f(X) for X, _ in ms]
    j = max(jitter_of(f, X, reps) for X, _ in ms)
    pairs = sorted(
        np.linalg.norm(v[a] - v[b])
        for a in range(len(v))
        for b in range(a + 1, len(v))
    )
    return pairs[0] / j, pairs[len(pairs) // 2] / j, j


# --------------------------------------------------------- 1. homometric
def line(v):
    X = np.zeros((len(v), 3))
    X[:, 0] = np.asarray(v, float)
    return X


HA, HB = line([0, 1, 4, 10, 12, 17]), line([0, 1, 8, 11, 13, 17])

say("=" * 78)
say("1. Bloom homometric pair, separation in units of the same descriptor's")
say(f"   response to a {JITTER} jitter. Sorted distances separate by "
    f"{np.linalg.norm(sorted_pairs(HA) - sorted_pairs(HB)):.2e}.")
say("=" * 78)
say(f"{'sigma':>6} {'spectra':>18} {'mode SV of T (N^4)':>22} {'9 traces':>18}")
for s in [1.0, 1.4, 2.0, 2.5, 3.0, 4.0, 6.0]:
    row = f"{s:6.1f}"
    for f in (spectra, mode_sv, moments):
        sep = np.linalg.norm(f(HA, s) - f(HB, s))
        jit = jitter_of(lambda X, f=f, s=s: f(X, s), HA)
        row += f" {sep:9.3e}/{sep / jit:<8.2f}"
    say(row)
say("")
say("The contraction leads the exact mode spectrum at every width, so")
say("unfolding a mode and keeping all of it buys nothing over contracting it.")
say("")

# ------------------------------------------------------- 2. LJ near misses
say("=" * 78)
say(f"2. Lennard-Jones minima at sigma {SIGMA}, weight {WEIGHT}. Worst case and")
say("   median over all pairs of distinct minima.")
say("=" * 78)
say(f"{'system':>7} {'pairs':>6} {'dist worst':>11} {'joint worst':>12} "
    f"{'trace worst':>12} {'dist med':>9} {'joint med':>10}")
for n, want in [(13, 12), (26, 18), (38, 20), (55, 14)]:
    ms = minima(n, want)
    zd, md, jd = z_stats(sorted_pairs, ms)
    zj, mj, _ = z_stats(joint, ms)
    _z, _m, jt = z_stats(moments, ms)
    zt, _mt, _ = z_stats(
        lambda X, w=jd / jt: np.concatenate([sorted_pairs(X), w * moments(X)]), ms
    )
    npairs = len(ms) * (len(ms) - 1) // 2
    say(f"LJ{n:<5} {npairs:6d} {zd:11.2f} {zj:12.2f} {zt:12.2f} "
        f"{md:9.2f} {mj:10.2f}")
say("")
say("A worst case below 1 means the closest pair of distinct minima is nearer")
say("than a jittered copy of one of them, so no merge radius separates it.")
say("The nine-trace compression is worse than having no spectral block at all.")
