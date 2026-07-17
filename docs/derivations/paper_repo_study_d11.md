# paper_repo_study.md — design loops from published / research code

## 1. In-repo: `proofs/d6_annealed_descent_scaling.py`

**Design loop:** model sphere Metropolis state gain → closed-form partial expectations → pairing sign → optimize_c → emit α*(θ).

Path-tagged facts:
- L32–47: `G(c,theta) = -E[Delta a]` with closed forms for negative/positive parts and `A(c,theta)`.
- L114–162: `gain`, `acceptance`, `optimize_c` (golden section on closed-form G).
- L60–61: G>0 iff θ<2 (critical window).
- L175–181: “running at theta ~ 0.5 keeps ~91% of the maximal descent rate”.
- L310–325: `WITNESS_*` gates must all hold.

**Pattern:** load-bearing identity first, then implementable constants from maximizing the *right* functional (state drift, not stationary diffusion).

## 2. In-repo: `proofs/d7_budgeted_escape_window.py`

**Design loop:** Kramers escape scaling + D6 ceiling → budgeted feasibility window.

Path-tagged facts:
- L4–16: window `b/ln B <~ T < 2 (f-f*)/d`, nonempty iff `b <~ 2 gap ln(B)/d`.
- L50–69: finite double-well birth-death chain (exact Metropolis transitions).
- L72–97: expected escape time via Thomas algorithm (no MC).
- L118–164: `kramers_scaling` and `window_grows_with_log_budget` witnesses.

**Pattern:** combine two inequalities into a *window*; when empty, no constant T works — motivates restarts/bias (stated L21–25).

## 3. External paper+code: SciPy `dual_annealing` (Xiang / Gubian GSA)

Path: SciPy `optimize/_dual_annealing.py` (Copyright Yang Xiang, Sylvain Gubian; visiting distribution from Xiang et al. GSA).

**Design loop:** published GSA visiting distribution → strategy-chain accept/reject → optional local search; temperature is a schedule parameter fed into `visit_fn` and acceptance.

Path-tagged facts:
- `VisitingDistribution` with fixed `visiting_param` (default 2.62), precomputed factors.
- `visiting` / `visit_fn(temperature, dim)` — temperature enters the visiting law (Formula Visita).
- `StrategyChain.accept_reject` — Tsallis-style acceptance with `acceptance_param` and `temperature_step`.

**Pattern:** separate *visiting law*, *acceptance law*, and *local polish* as composed slots; parameters come from the paper, not ad-hoc schedules alone.

## Gap left open (motivates D11)

GPMD/D6 alone always runs θ⋆=1/2 (descent-optimal). D7 says escape within remaining budget needs T ≳ b/ln B. **No shipped law clamps the operating temperature into the D6∩D7 window using remaining budget and a barrier proxy.** That is the new algorithm (BFWT).
