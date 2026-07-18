# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!-- towncrier release notes start -->

## [0.7.1](https://github.com/HaoZeke/anneal/tree/0.7.1) - 2026-07-18

### Added

- `amsa_optimize`: the portfolio's whitened BFWT annealed-descent machinery
  (Haario covariance whitening, budget-feasible window temperature, online
  barrier estimate, Robbins--Monro scale control, IPOP reseeds) as a
  standalone solver with per-coordinate Cauchy tail moves, stagnation
  polish bursts, a Python binding, and comparison-protocol wiring. The
  portfolio's behaviour is unchanged.


## [0.7.0](https://github.com/HaoZeke/anneal/tree/0.7.0) - 2026-07-18

### Added

- Probe-based regime routing: mid-width multimodal-global commitments now
  spend ~2% of the budget on three short projected descents from spread
  starts; a single basin depth class demotes to polish-heavy routing, and
  multi-basin landscapes where descents pay route large slices to the
  perturb-and-descend hop arm (full slice depth) instead of GSA
  (`routing_probe` module).

### Fixed

- `projected_gradient_polish` no longer plateaus on ill-conditioned
  valleys: quadratic-interpolation backtracking replaces naive halving,
  the L-BFGS curvature floor drops from sqrt(eps) to 8*eps so genuine
  high-kappa pairs survive, and a failed line search restarts the memory
  two orders finer from the best point instead of returning early. On the
  CUTEst least-squares loss set this converts repeated 4e-6 plateaus into
  cell wins (LANCZOS2LS to 2e-11, NELSONLS 52 -> 3.8) with no control
  regressions.
- `rgpot_minimize` example gated behind the `capi` feature with
  `eindir-core/capi` forwarded, restoring plain `cargo test`.

## [0.6.0](https://github.com/HaoZeke/anneal/tree/0.6.0) - 2026-07-18

### Added

- Population-controlled diffusion arm `dmc_pop`: branch/kill walker
  bookkeeping for classical box-constrained minimization, with residual
  branching, SHADE differential-evolution memory, QMC initialization,
  elitist inject, shrinking population targets, dual-style L-BFGS elite
  polish, and multi-walker batch evaluation on the CUTEst SOTA path.
- Annealed-descent arm `am_sa` implementing the D6 gap-proportional
  cooling law with running-covariance proposals and Robbins--Monro step
  control, plus the D11 budget-feasible window temperature (BFWT)
  schedule and a `MultimodalGlobal` regime with a dual-class GSA path.
- Portfolio auto policy: feature-based regime routing, Tree-structured
  Parzen density-ratio allocation, Good--Turing x record-statistics
  budget-conversion gate, D5 terminal polish reserve, Bayesian
  one-step-lookahead endgame with Luby restarts, and a bounded
  coordinate-opposition scout.
- Enhanced-sampling arms behind the same ledger: well-tempered
  metadynamics on sketch-map collective variables and transition-path
  shooting moves for continuous boxes.
- Executable derivation ledger extended to D5--D11 (SymPy, and Lean for
  the T1 window), with the GPMD design lab emitting shipped constants.
- Status-bearing SOTA census tooling: frozen CUTEst manifests, budgeted
  TuRBO and NGOpt/Py-BOBYQA baselines, solver-local restart recovery,
  and performance/data/Pareto profile plotting from the census CSV.
- Rayon fan-out for parallel tempering, QMC multi-start, polish, and
  `dmc_pop`; quickstart notebook and stand-alone portfolio script.

### Changed

- `eindir-core` requirement raised to 0.5.2 for the parallel
  `eval_batch` and Halton design paths.

## [0.5.0](https://github.com/HaoZeke/anneal/tree/0.5.0) - 2026-06-26

### Removed

- The ``anneal_core::grad`` module (and its public re-exports of ``Gradient``, ``AnalyticGradient``, and ``FiniteDiffGradient``) has been removed entirely.

  Gradient support is now provided exclusively by the ``eindir_core`` primitives crate (on which anneal is built):

  - ``eindir_core::Gradient``
  - ``eindir_core::DifferentiableObjective``
  - ``eindir_core::AnalyticGradient``
  - ``eindir_core::FiniteDiffGradient``
  - The full ``eindir_core::gradient`` module

  All drivers (HMC/NUTS, GLE-Langevin, projected-gradient polish and QMC variants, etc.) continue to work unchanged. Internal code and tests now import directly from ``eindir_core``.

  Python users should supply gradients via ``grad_fn`` when constructing ``eindir.PyObjective`` (or pass plain callables to the anneal wrappers, which continue to work).

  See the eindir documentation (cross-linked via intersphinx) for the full story, including analytic gradients on the surrogates and best practices for supplying native derivatives from Python frameworks.

### Added

- Fixed-budget benchmark drivers (``anneal_sota.qmc_annealed_hybrid`` and ``sota_cutest``) wire the shared ``eindir`` and ``anneal`` primitives through the comparison path:

  - Tensor-train, additive, and Chebyshev surrogates as Move-slot proposal sources.
  - GLE colored-noise dynamics with native gradients.
  - Native-gradient L-BFGS-B and QMC polish with objective and gradient evaluations charged through ``Counter``.
  - QMC seeding through the typed algebra components.

  The ``hybrid_de`` / anneal entry in comparisons now calls the same hybrid path used by the benchmark helpers instead of a separate reimplementation.


### Miscellaneous

- Resolve ``dlpk`` (0.1.5), ``eindir-core`` (0.5.0), and the ``rgpot-core``
  example dependency from crates.io instead of git/path pins, unblocking
  registry publication of ``anneal-core``.

## [0.4.0](https://github.com/HaoZeke/anneal/tree/0.4.0) - 2026-04-26

### Added

- Bayesian-pilot SA with Laplace fit on hyperparameters (T_init, sigma, and q_v extension to 3D/4D); multi-chain bGSA over (T, eps, L, q).
- MCMC-SA (sparse-skip + Gelman-Rubin), log-domain Metropolis as f16 default, Kahan compensated summation experiments, and adapter/laws A2/A6 scaffolds.
- bGSA (Bayesian-pilot + q-deformed HMC + parallel tempering) with Gelman-Rubin termination, MCMC-SA method, CUTEst benchmark catalog, Dolan-More/More-Wild/Pareto figures, and HMC/Bayesian-pilot drivers integrated into the typed algebra (including PyGradient + run_hmc).


## [0.3.1](https://github.com/HaoZeke/anneal/tree/0.3.1) - 2026-04-26

### Added

- Added finite-precision experiments runner plus four manuscript studies (underflow, cancellation, trajectory bias, compensated summation).


## [0.3.0](https://github.com/HaoZeke/anneal/tree/0.3.0) - 2026-04-26

### Added

- Landed typed component algebra: SaVariant, Objective traits, BSA/FSA/GSA implementations with law witnesses, sympy proofs for limit reductions, TLA+ specs + Apalache/TLC verification, and pixi verify env.
- SaVariant::checked preset constructors enforcing the L1-L4 invariants.

### Changed

- BREAKING refactor: hard-broke legacy quencher API. New pure-Rust SA driver (run_rs), criterion benches, and tests now use SaVariant + run().


## [0.2.0](https://github.com/HaoZeke/anneal/tree/0.2.0) - 2026-04-26

### Added

- Added OIDC PyPI + Zenodo release workflow and DOI/citation metadata.
- Pixi workspace with default, python, docs, verify, and gpu environments (replacing environment.yml and PDM).
- Replaced legacy MyST + furo docs with orgmode export + Sphinx (shibuya theme) pipeline, including rustdoc post-processing.
- Scaffolded anneal-core Rust crate (lib + cdylib + staticlib) plus C/C++ bindings via cargo-c, pkg-config, meson, CMake, and hand-written C++ companion.

### Changed

- Adopted cog for conventional commits + cargo-dist style releases; dropped .pre-commit-config.yaml and tbump (CI + cog bump handle lint/release).
- BREAKING: build system migrated from PDM to maturin mixed mode. Python sources moved under python/anneal/; wheel now ships Rust extension anneal._core.


## [0.1.0](https://github.com/HaoZeke/anneal/tree/0.1.0) - 17-02-2024


No significant changes.
