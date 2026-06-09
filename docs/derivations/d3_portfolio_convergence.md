# D3: Portfolio convergence preservation

## Setup

A portfolio driver runs $K$ member algorithms (arms) by interleaving slices: a
scheduler picks an arm, runs it for a slice of $b$ steps, records any improvement
to a shared **incumbent** best, and repeats. Let $f: \mathcal{B} \to \mathbb{R}$
be the objective on a bounded box $\mathcal{B} = \prod_{j=1}^d [\ell_j, u_j]
\subset \mathbb{R}^d$, measurable and bounded below, with essential infimum

$$
f^* \;=\; \operatorname*{ess\,inf}_{x \in \mathcal{B}} f(x)
\;=\; \sup\{c : \lambda(\{f < c\}) = 0\},
$$

where $\lambda$ is Lebesgue measure on $\mathcal{B}$.

**Restart arm.** At least one arm $k_0$ performs **restarts**: it draws a fresh
point $X_m \sim \mu$ from a fixed restart measure $\mu$ on $\mathcal{B}$ with an
everywhere-positive density, $\mu(dx) = \rho(x)\, dx$ with $\rho(x) \ge \rho_0 >
0$ on $\mathcal{B}$. The implementation supplies $\mu$ as uniform, or as a
QMC/Halton design with a Cranley-Patterson shift (randomized QMC); both give
positive density (the CP-shifted Halton point, marginalized over the shift, is
uniform). Restarts are mutually independent given the schedule.

**Monotone incumbent.** The incumbent update is the law-respecting best update:
$\mathrm{best}_t = \min(\mathrm{best}_{t-1}, f(\text{candidate}_t))$. This is the
$\mathrm{Accept}$ slot's downhill boundary (L3) applied to the global best: a
strictly worse candidate never displaces the incumbent, so $t \mapsto
\mathrm{best}_t$ is non-increasing.

**Level set.** For $\varepsilon > 0$ define the level set and its restart measure

$$
L_\varepsilon \;=\; \{x \in \mathcal{B} : f(x) \le f^* + \varepsilon\},
\qquad
\mu(L_\varepsilon) \;=\; \int_{L_\varepsilon} \rho(x)\, dx \;>\; 0,
$$

where positivity holds because $\lambda(L_\varepsilon) > 0$ for every
$\varepsilon > 0$ (by definition of the essential infimum) and $\rho \ge \rho_0
> 0$.

## Part (a): almost-sure convergence to the essential infimum

### Theorem D3.1

Suppose the restart arm $k_0$ is scheduled infinitely often (in the
infinite-budget limit, the schedule selects $k_0$ for infinitely many slices
almost surely). Then

$$
\mathrm{best}_t \;\xrightarrow[t \to \infty]{}\; f^*
\qquad \text{almost surely.}
$$

### Proof

Let $\{X_m\}_{m \ge 1}$ be the sequence of restart draws produced by arm $k_0$;
since $k_0$ runs infinitely often, $m \to \infty$ as $t \to \infty$. Fix
$\varepsilon > 0$ and set $p_\varepsilon = \mu(L_\varepsilon) > 0$. The events
$\{X_m \in L_\varepsilon\}$ are independent with probability $p_\varepsilon$
each, so $\sum_m \mathbb{P}(X_m \in L_\varepsilon) = \sum_m p_\varepsilon =
\infty$. By the second Borel-Cantelli lemma, $X_m \in L_\varepsilon$ for
infinitely many $m$ almost surely. At the first such $m$, the candidate has
$f(X_m) \le f^* + \varepsilon$, and the monotone incumbent gives
$\mathrm{best}_t \le f^* + \varepsilon$ for all subsequent $t$. Hence
$\limsup_t \mathrm{best}_t \le f^* + \varepsilon$ a.s.

Take a countable sequence $\varepsilon_i \downarrow 0$; intersecting the
probability-one events gives $\limsup_t \mathrm{best}_t \le f^*$ a.s. The reverse
inequality $\mathrm{best}_t \ge f^*$ holds for every $t$ up to a $\lambda$-null
set of candidate draws: a restart lands in $\{f < f^*\}$ with probability
$\mu(\{f < f^*\}) = 0$ (null set times bounded density), and a union over the
countably many restarts is still null. Therefore $\mathrm{best}_t \to f^*$
a.s. $\qquad\blacksquare$

The hypothesis that the restart arm is scheduled **infinitely often** is
essential and stated explicitly: a Thompson scheduler (D4) that starved arm
$k_0$ would break the Borel-Cantelli step, which is exactly why D4 imposes a
probability floor.

## Part (b): finite-restart rate and QMC deterministic covering

### Theorem D3.2 (geometric tail)

After $n$ independent restarts from $\mu$,

$$
\mathbb{P}\big(\mathrm{best} \le f^* + \varepsilon \text{ after } n \text{ restarts}\big)
\;\ge\; 1 - \big(1 - \mu(L_\varepsilon)\big)^{\,n}.
\tag{D3.1}
$$

### Proof

The incumbent reaches the level set as soon as **any** of the $n$ restarts lands
in $L_\varepsilon$. The complementary event is that all $n$ restarts miss
$L_\varepsilon$, which by independence has probability $\prod_{m=1}^n (1 -
\mu(L_\varepsilon)) = (1 - \mu(L_\varepsilon))^n$. Subtracting from $1$ gives
(D3.1). Other arms can only lower the incumbent further, so the bound is a lower
bound on the true success probability. $\qquad\blacksquare$

The right-hand side reaches $1 - e^{-1} \approx 0.632$ at $n = 1/\mu(L_\varepsilon)$
and $1 - e^{-c}$ at $n = c/\mu(L_\varepsilon)$: the number of restarts needed for
target confidence scales as $1/\mu(L_\varepsilon)$, the inverse measure of the
level set.

### QMC stratification: deterministic covering

For a low-discrepancy restart design (Halton with a Cranley-Patterson shift) the
random tail (D3.1) sharpens to a **deterministic** coverage guarantee. Let $P =
\{x_1, \dots, x_n\} \subset [0,1]^d$ (mapped affinely to $\mathcal{B}$) have star
discrepancy

$$
D_n^*(P) \;=\; \sup_{B = \prod_j [0, a_j)}
\left| \frac{\#\{x_m \in B\}}{n} - \lambda(B) \right|,
$$

which for Halton sets obeys $D_n^*(P) = O\big((\log n)^d / n\big)$
(Niederreiter, 1992). For any anchored box $B$ with $\lambda(B) = \mu(L)$, the
discrepancy bound rearranges to

$$
\#\{x_m \in B\} \;\ge\; n\,\lambda(B) - n\, D_n^*(P)
\;=\; n\big(\mu(L) - D_n^*(P)\big).
\tag{D3.2}
$$

Hence once

$$
n \;>\; \frac{D_n^*(P)}{\mu(L)}
\qquad\Longleftrightarrow\qquad
\mu(L) > D_n^*(P),
\tag{D3.3}
$$

the right-hand side of (D3.2) is strictly positive: **at least one** design
point lands in every anchored box of measure $\mu(L)$, with certainty rather
than in probability. This is the QMC stratification improvement: where the iid
bound (D3.1) leaves a $(1 - \mu(L))^n$ failure probability, the low-discrepancy
design eliminates it for boxes once $n$ clears the $D_n^*/\mu(L)$ threshold. The
Cranley-Patterson shift keeps the design unbiased (each marginal uniform) while
preserving the discrepancy order, so (D3.1) holds *and* (D3.2) holds for the
same draws.

**Hypothesis caveat.** The covering (D3.2) is stated for anchored boxes, which is
exact when $L_\varepsilon$ contains an anchored box of measure $\mu(L_\varepsilon)$
(a basin near a box corner) or after the standard reduction to general boxes,
which costs a factor $2^d$ in the discrepancy (the box discrepancy is bounded by
$2^d D_n^*$). For a general level set the clean guarantee is the iid (D3.1); the
QMC bonus (D3.2) applies to the box-shaped portion of the basin and is the
mechanism behind faster coverage in low effective dimension.

## Connection to the algebra (L1-L4)

The interleaving preserves each arm's invariants through the laws:

- **(L2) support compatibility** keeps each arm's proposals inside its declared
  neighborhood (the box), so restarts stay in $\mathcal{B}$ and the measure $\mu$
  is well-defined; the bounded-support hypothesis of Theorem D3.1 is exactly
  (L2) applied per arm.
- **(L3) downhill boundary** is the monotone incumbent: the shared best update
  accepts a candidate only when it does not worsen the best, so $\mathrm{best}_t$
  is non-increasing, which both Theorems use.
- **(L1) symmetry** and **(L4) temperature monotonicity** are local to each arm's
  Move/Accept pair and are unaffected by interleaving: the portfolio composes
  law-valid arms into a law-valid driver because it only ever calls each arm's
  `step` and reads its objective value, never mutating another arm's state.

The portfolio therefore inherits the per-arm invariants (monotone best, bounded
support) by construction, and Theorem D3.1 lifts the single-arm restart
guarantee to the portfolio whenever the restart arm runs infinitely often.

## Witness

`proofs/d3_portfolio_convergence.py` verifies: (1) the geometric tail identity
$1 - (1-p)^n$ symbolically and by Monte Carlo; (2) the monotone incumbent equals
the global minimum of all candidate samples; (3) the QMC star-discrepancy
covering by a deterministic box-hit count against the Niederreiter lower bound
$n\mu(B) - n D_n^*$ on a Halton + Cranley-Patterson-shift design. Tests in
`proofs/tests/test_d3_portfolio_convergence.py`.

## References

- Niederreiter, H. (1992). Random Number Generation and Quasi-Monte Carlo Methods. SIAM CBMS-NSF 63.
- Cranley, R. and Patterson, T. N. L. (1976). Randomization of number theoretic methods for multiple integration. SIAM J. Numer. Anal. 13(6), 904-914.
- Solis, F. J. and Wets, R. J.-B. (1981). Minimization by random search techniques. Math. Oper. Res. 6(1), 19-30.
- Durrett, R. (2019). Probability: Theory and Examples, 5th ed. Cambridge University Press (Borel-Cantelli).
