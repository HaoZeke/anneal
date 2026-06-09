# D1: Metropolis-independence acceptance lower bound

## Setup

The surrogate Move slot turns a tempered surrogate density into an independence
proposal. Fix a temperature $T > 0$ and a state space $\mathcal{X} \subseteq
\mathbb{R}^d$ (the bounded box). The target is the tempered Gibbs measure

$$
\pi_T(x) \;=\; \frac{1}{Z_f(T)}\, e^{-f(x)/T},
\qquad Z_f(T) = \int_{\mathcal{X}} e^{-f(x)/T}\, dx,
$$

and the independence proposal draws $y \sim q$ from the surrogate's own tempered
density

$$
q(x) \;=\; \frac{1}{Z_s(T)}\, e^{-s(x)/T},
\qquad Z_s(T) = \int_{\mathcal{X}} e^{-s(x)/T}\, dx,
$$

where $s$ is the surrogate of $f$. For the rank-one additive (functional
tensor-train) surrogate, $s(x) = \sum_{j=1}^{d} g_j(x_j)$ separates, so $q$
factorizes and `AdditiveSurrogate::sample` draws each coordinate independently
by one-dimensional inverse-CDF (`eindir/src/additive.rs`,
`src/methods/additive_independence.rs`). The acceptance test runs against the
**true** objective $f$, not the surrogate.

Define the per-point log-surrogate error and its sup-norm

$$
r(x) \;=\; \frac{f(x) - s(x)}{T},
\qquad
\delta \;=\; \sup_{x \in \mathcal{X}} \, |r(x)| \;=\;
\frac{1}{T}\,\sup_{x \in \mathcal{X}} |f(x) - s(x)|.
\tag{D1.1}
$$

We assume $\delta < \infty$ (bounded log-error), which holds whenever $f - s$ is
bounded on the box; this is the only hypothesis the bound needs.

## Theorem D1.1 (dimension-free acceptance lower bound)

For the Metropolis-Hastings independence sampler with proposal $q$ and target
$\pi_T$ above, the acceptance probability satisfies

$$
\alpha(x, y)
\;=\; \min\!\left(1,\; \frac{\pi_T(y)\, q(x)}{\pi_T(x)\, q(y)}\right)
\;\ge\; \min\!\left(1,\; e^{-2\delta}\right)
\;=\; e^{-2\delta}
\qquad \text{for all } x, y \in \mathcal{X},
\tag{D1.2}
$$

with no dependence on the dimension $d$. The constant $2$ in the exponent is
tight.

### Proof

The Hastings ratio for an independence proposal (the proposal density $q(y)$
does not depend on the current state $x$) is

$$
R(x, y) \;=\; \frac{\pi_T(y)\, q(x)}{\pi_T(x)\, q(y)}.
$$

The normalizers $Z_f(T)$ and $Z_s(T)$ cancel between numerator and denominator:

$$
R(x, y)
= \frac{e^{-f(y)/T}}{e^{-f(x)/T}} \cdot \frac{e^{-s(x)/T}}{e^{-s(y)/T}}
= \exp\!\left(\frac{-f(y) + f(x) + s(y) - s(x)}{T}\right)
= \exp\!\big(r(x) - r(y)\big),
\tag{D1.3}
$$

where the last equality groups the four terms as $\tfrac{f(x)-s(x)}{T} -
\tfrac{f(y)-s(y)}{T} = r(x) - r(y)$. By (D1.1), $r(x) \ge -\delta$ and $r(y)
\le \delta$, hence

$$
r(x) - r(y) \;\ge\; (-\delta) - (\delta) \;=\; -2\delta,
$$

so $R(x, y) \ge e^{-2\delta}$ and $\alpha(x, y) = \min(1, R(x, y)) \ge
\min(1, e^{-2\delta}) = e^{-2\delta}$, since $\delta \ge 0$ forces $e^{-2\delta}
\le 1$. None of these steps reference $d$. $\qquad\blacksquare$

### Tightness of the constant $2$

The bound is attained, not merely valid. If there exist $x_\star, y_\star$ with
$r(x_\star) = -\delta$ and $r(y_\star) = +\delta$ (the surrogate underestimates
$f$ at $x_\star$ and overestimates it at $y_\star$, each by the full sup-norm
gap), then $R(x_\star, y_\star) = e^{-2\delta}$ exactly. Therefore the constant
in the exponent cannot be improved to anything smaller than $2$ under the
sup-norm definition (D1.1): the worst-case exponent $\min_{r_x, r_y \in
[-\delta, \delta]}(r_x - r_y) = -2\delta$.

**Remark on the manuscript constant.** The promise in
`iise_manuscript.org` (Section 7, surrogate Move paragraph; the generated
`iise_manuscript.tex` line 606) is $\alpha \ge \exp(-2\delta)$. This derivation
confirms that constant is correct and tight for the **two-sided** sup-norm
$\delta = \sup_x |(f-s)/T|$. The weaker $\exp(-\delta)$ would hold only under a
**one-sided** hypothesis (for example, $0 \le (f-s)/T \le \delta$ with the
surrogate a uniform under-estimator, giving $r(x) - r(y) \ge 0 - \delta =
-\delta$). The implemented surrogate is fit by backfitting least squares with no
sign constraint, so the two-sided $\exp(-2\delta)$ is the honest bound; the
manuscript text needs no amendment.

## Corollary D1.2 (uniform ergodicity)

The independence sampler with kernel $P$ defined by $\alpha$ above is uniformly
ergodic, and for every starting state $x$ and every $n \ge 1$,

$$
\big\| P^n(x, \cdot) - \pi_T \big\|_{\mathrm{TV}}
\;\le\; \big(1 - e^{-2\delta}\big)^{\,n}.
\tag{D1.4}
$$

### Proof

For an independence sampler the transition kernel on $y \ne x$ has density
$q(y)\,\alpha(x, y)$. Using (D1.2),

$$
P(x, dy) \;=\; q(y)\,\alpha(x, y)\, dy + \big[\text{holding mass}\big]\,\delta_x(dy)
\;\ge\; e^{-2\delta}\, q(y)\, dy
\qquad (y \ne x).
$$

The standard independence-sampler argument (Mengersen and Tweedie, 1996,
Theorem 2.1; Roberts and Rosenthal, 2004, Section 3) sharpens this: writing the
importance weight $w(x) = \pi_T(x)/q(x) = Z_s(T)/Z_f(T)\cdot e^{r(x)}$, the
acceptance bound (D1.2) is equivalent to $\inf_x \pi_T(x)/q(x) \ge \beta_\star
\sup_x \pi_T(x)/q(x)$ with $\beta_\star = e^{-2\delta}$. The whole space is then
a small set with minorization $P(x, \cdot) \ge \beta_\star\, \pi_T(\cdot)$,
because

$$
P(x, A) \;\ge\; \int_A q(y)\,\alpha(x,y)\,dy
\;\ge\; \int_A q(y)\, e^{-2\delta} \frac{\pi_T(y)}{\,q(y)\,}\!\Big/\!\sup_z\tfrac{\pi_T(z)}{q(z)} \, dy
\;\ge\; e^{-2\delta}\, \pi_T(A),
$$

where the middle step uses $\alpha(x,y) \ge \pi_T(y)q(x)/(\pi_T(x)q(y)) \wedge 1$
and the sup-/inf-weight bound. A one-step minorization with constant
$\beta_\star$ against the stationary law $\pi_T$ gives geometric TV contraction
at rate $1 - \beta_\star$ (Meyn and Tweedie, 2009, Theorem 16.0.2), which is
(D1.4). $\qquad\blacksquare$

**Reading of the rate.** The exponent $n$ in $(1 - e^{-2\delta})^n$ is the
**number of MH iterations**, not the dimension $d$. The contraction factor $1 -
e^{-2\delta}$ is itself dimension-free, inherited directly from the
dimension-free acceptance bound (D1.2). A surrogate with small sup-norm
log-error ($\delta \to 0$) gives rate $\to 0$: near-perfect surrogates mix in
one step, uniformly over $d$. This is the mechanism behind the empirical result
(`iise_manuscript.tex` line 607) that the rank-one independence sampler reaches
the global basin on $20/20$ seeds at $d = 20$ and $d = 50$ within a fixed budget
while a dimension-scaled random walk reaches only $55$-$67\%$ (reported in
`iise_manuscript.org`).

## Connection to the algebra

The construction reuses two slots without touching the others. The surrogate
enters at the $\mathrm{Move}$ slot as $q \propto e^{-s/T}$, and the existing
Metropolis rule at the $\mathrm{Accept}$ slot supplies $\alpha$. Law (L3)
(downhill boundary) is exactly the $\min(1, \cdot)$ in (D1.2); law (L4)
(temperature monotonicity) holds because $R(x,y) = e^{(r(x)-r(y))}$ with
$r = (f-s)/T$ is monotone in $T$ for a fixed energy gap. The bound (D1.2) is a
property of the $\mathrm{Move}$/$\mathrm{Accept}$ pair and is independent of the
$\mathrm{Cool}$ schedule, so it transfers unchanged to every driver that
consumes the surrogate Move.

## Witness

`proofs/d1_independence_bound.py` verifies: (1) the symbolic identity
$R(x,y) = e^{r(x)-r(y)}$ via `sympy`; (2) the tight worst-case exponent
$-2\delta$; (3) the numeric bound and its saturation by the adversarial pair;
and (4) the uniform-ergodicity TV decay on a finite-state independence chain by
exact total-variation computation. Tests in
`proofs/tests/test_d1_independence_bound.py`.

## References

- Mengersen, K. L. and Tweedie, R. L. (1996). Rates of convergence of the Hastings and Metropolis algorithms. Annals of Statistics 24(1), 101-121.
- Roberts, G. O. and Rosenthal, J. S. (2004). General state space Markov chains and MCMC algorithms. Probability Surveys 1, 20-71.
- Meyn, S. and Tweedie, R. L. (2009). Markov Chains and Stochastic Stability, 2nd ed. Cambridge University Press.
