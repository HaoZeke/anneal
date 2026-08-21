# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!-- towncrier release notes start -->

## Unreleased

### Added

- Occupancy ArchiveHole starts from an even covering of \(S^{3N-1}\)
  around the occupied minimum (SoftSaddle `triangulation_hight_v2`),
  not a random packing-mean \(\nu=3\) kick. Plasencia Gutiérrez, M.;
  Argáez, C.; Jónsson, H. Improved Minimum Mode Following Method for
  Finding First Order Saddle Points. *J. Chem. Theory Comput.* **2017**,
  *13* (1), 125-134. <https://doi.org/10.1021/acs.jctc.5b01216>
- Occupancy Leave quenches through xtsci L-BFGS on a PES whose known
  packings (this chain's origin and previous chains in the packing
  archive) are inverted in the DECAF map: \(\mu\) is the mean of
  per-center SOAP+ACE \(\nu=3\), \(P=J_\mu^{\mathsf T}\hat u_\varphi\)
  is the Cartesian pullback, and the dimer Householder of Henkelman
  and Jónsson, *J. Chem. Phys.* **1999**, *111*, 7010
  <https://doi.org/10.1063/1.480097> is applied to \(P\) instead of
  the lowest mode or a Cartesian radius. Leftover SOAP \(p_i-\mu\)
  is not this map. A raw-\(E\) polish follows only when DECAF
  reports a new family.

## [0.9.0](https://github.com/HaoZeke/anneal/tree/v0.9.0) - 2026-08-13

### Added

- A durable cooperative-search protocol with replica identity, monotone event
  sequences, idempotent replay, snapshot-version checks, validated candidate
  ingress, descriptor-hole proposals, policy-state queries, and an append-only
  charged-work ledger.  The Cap'n Proto client and coordinator expose the same
  catalogue operations in process and over RPC.
- Cut-and-splice mixing for the CSA bank: a random plane cut between two
  members, with atom-count repair, as the ``mix_fraction`` step.
- Elja campaign drivers pin a full source object ID, verify calibrated inputs,
  record the complete run contract, require terminal markers, and seal every
  result package with a closed ``SHA256SUMS`` manifest.  Shared and private-bank
  arms use the same ensemble budget and provenance contract.
- Event-catalogue residual search (``graphkey``): local NAUTY topology keys,
  a ``log n`` event catalogue, energy-floor flicker classes with record EI,
  a remaining-drop screen predictor, a GMRF residual on the class graph, and
  ``archive_search`` (CLI token ``ras`` / ``pair``). Large-budget ras is one
  ungated polish hop for ``N >= 70`` (LJ75 Marks) and a 30/70 skip-return
  then polish split for smaller clusters (LJ38/55). Small-budget molecular
  and slab walks stay on their own branch. Archive search schedules its
  return-polish stages on cloned configurations and does not mutate the caller's
  preset.
- Persistent in-process minimum-profile adapters for molecular-cluster and slab
  searches. ``nwchemc`` and ``cpmdc`` load once, reuse the same ``ProfileEngine``
  for the complete hop loop, and require neither an RPC server nor a result
  cache. Molecular requests omit a cell while periodic slab requests carry it
  through the same request type.
- Synchronous, fixed-population Feynman--Kac reconfiguration for cooperative
  chains.  Complete epochs rank energy, basin scarcity, and GMRF residual
  uncertainty, then return a replayable systematic-resampling plan with parent,
  family-size, rejuvenation, and effective-sample-size diagnostics.

### Changed

- Cluster proposal libraries implement the general ``MoveKernel`` interface and
  ``Config::proposal_kernel`` plugs them into the same ``HoppingSampler`` used by
  ordinary bounded objectives. Length- and energy-bearing preset values,
  including restart and rigid-group repacking geometry, derive from declared
  ``length_scale`` and ``energy_scale`` values rather than Lennard-Jones units.
- Shared-bank samples cross a charged local validation boundary before they can
  affect a receiving chain.  Molecular and slab campaign drivers support one
  shared bank or one private bank per replica, record the selected topology, and
  retain per-replica validation and snapshot evidence.
- ``Config::recommended`` is the complete scientific hop: Thompson-allocated
  LeanBurst includes the analytic SOAP pullback alongside the atomic proposals,
  the return screen rejects basin returns before a full quench, and stall
  symmetrisation is enabled. ``Config::for_cluster`` remains the plain
  Wales--Doye comparison preset.

### Fixed

- Accepted coordinator sockets are normalized to blocking mode so framed RPC
  connections remain open across multiple requests on BSD, macOS, and Linux.
- The recommended SOAP pullback scales both its Cartesian RMSD and descriptor
  cutoff from the preset's declared ``length_scale`` instead of assuming
  Lennard-Jones coordinate units.


## [0.8.0](https://github.com/HaoZeke/anneal/tree/v0.8.0) - 2026-08-08

### Added

- Exact basin identity by canonical contact-graph labelling through nauty,
  vertex-coloured with per-species-pair cutoffs (``graphkey`` feature);
  vesin cell lists compiled from upstream as the neighbour backend
  (``vesin-nl``); Franzblau primitive-ring profiles; the diffusion-map
  direction of the minima archive with Nystrom extension; a joint
  multi-sweep multinomial estimator for the density of minima; and
  fractional evaluation pricing through ``Ledger::charge_frac`` with the
  staged-quench ``Settle`` hook.
- Molecular clusters: rigid group moves (shake, relocation of the
  least-bound group by inter-group contacts, composed bursts), with groups
  derived each hop from the structure's own connectivity under the
  bond-matrix rule over Cordero covalent radii, so a reactive event
  regroups the moves instead of stranding the walker. Frozen substrates,
  dynamic active regions that follow the adsorbate through neighbour
  shells, and con-file geometries through readcon; engines attach through
  the rgpot RPC server, so the driver holds no engine code.
- Python bindings for the measured cluster-search layer: ``Config.recommended`` /
  ``Config.for_cluster``, ``Ledger`` with a charged budget, and
  ``cluster_search(obj_fn, grad_fn, n, budget, seed, recommended)`` returning
  ``{best, best_energy, hops}``.
- ``Config::recommended(n)``: the measured cluster-search configuration.
  Composed surface relocations paying one acceptance test per excursion,
  Normal-Gamma Thompson allocation over move arms rewarded by depth, and
  tabu on stall, each layer confirmed against paired controls across four
  Lennard-Jones morphologies with Beta-Binomial posterior comparisons.
  Reference GMIN at matched potential-call budgets does not reach the 75-
  and 98-point systems this stack solves. ``Config::for_cluster(n)`` stays
  the plain Wales-Doye comparison baseline.

### Miscellaneous

- Mechanisms that measured null or harmful against paired controls ship as
  reproducibility evidence with explicit banners, not as advice:
  flat-histogram acceptance, statistical temperature, well-tempered energy
  bias, soft-subspace and learned-covariance perturbations, nested search
  as configured, archive-discovery acquisition, committor population, and
  the early-gated screen.


## [0.7.2](https://github.com/HaoZeke/anneal/tree/v0.7.2) - 2026-08-01

### Changed

- Lint cleanup with no behavioural change: removed the superseded endgame
  basin-grind driver and the portfolio-local unit-box helpers (the
  stall-recovering interpolated polish owns that path since 0.7.0),
  collapsed nested conditionals into let-chains, switched to
  `clamp`/`is_multiple_of`, documented `GpmdResult` fields, and promoted
  the theta-star descent-window check to a compile-time assertion.
- Commit lint in CI now checks from the latest tag; the pre-convention
  legacy history is no longer linted.

## [0.7.1](https://github.com/HaoZeke/anneal/tree/v0.7.1) - 2026-07-18

### Added

- `amsa_optimize`: the portfolio's whitened BFWT annealed-descent machinery
  (Haario covariance whitening, budget-feasible window temperature, online
  barrier estimate, Robbins--Monro scale control, IPOP reseeds) as a
  standalone solver with per-coordinate Cauchy tail moves, stagnation
  polish bursts, a Python binding, and comparison-protocol wiring. The
  portfolio's behaviour is unchanged.


## [0.7.0](https://github.com/HaoZeke/anneal/tree/v0.7.0) - 2026-07-18

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

## [0.6.0](https://github.com/HaoZeke/anneal/tree/v0.6.0) - 2026-07-18

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

## [0.5.0](https://github.com/HaoZeke/anneal/tree/v0.5.0) - 2026-06-26

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

## [0.4.0](https://github.com/HaoZeke/anneal/tree/v0.4.0) - 2026-04-26

### Added

- Bayesian-pilot SA with Laplace fit on hyperparameters (T_init, sigma, and q_v extension to 3D/4D); multi-chain bGSA over (T, eps, L, q).
- MCMC-SA (sparse-skip + Gelman-Rubin), log-domain Metropolis as f16 default, Kahan compensated summation experiments, and adapter/laws A2/A6 scaffolds.
- bGSA (Bayesian-pilot + q-deformed HMC + parallel tempering) with Gelman-Rubin termination, MCMC-SA method, CUTEst benchmark catalog, Dolan-More/More-Wild/Pareto figures, and HMC/Bayesian-pilot drivers integrated into the typed algebra (including PyGradient + run_hmc).


## [0.3.1](https://github.com/HaoZeke/anneal/tree/v0.3.1) - 2026-04-26

### Added

- Added finite-precision experiments runner plus four manuscript studies (underflow, cancellation, trajectory bias, compensated summation).


## [0.3.0](https://github.com/HaoZeke/anneal/tree/v0.3.0) - 2026-04-26

### Added

- Landed typed component algebra: SaVariant, Objective traits, BSA/FSA/GSA implementations with law witnesses, sympy proofs for limit reductions, TLA+ specs + Apalache/TLC verification, and pixi verify env.
- SaVariant::checked preset constructors enforcing the L1-L4 invariants.

### Changed

- BREAKING refactor: hard-broke legacy quencher API. New pure-Rust SA driver (run_rs), criterion benches, and tests now use SaVariant + run().


## [0.2.0](https://github.com/HaoZeke/anneal/tree/v0.2.0) - 2026-04-26

### Added

- Added OIDC PyPI + Zenodo release workflow and DOI/citation metadata.
- Pixi workspace with default, python, docs, verify, and gpu environments (replacing environment.yml and PDM).
- Replaced legacy MyST + furo docs with orgmode export + Sphinx (shibuya theme) pipeline, including rustdoc post-processing.
- Scaffolded anneal-core Rust crate (lib + cdylib + staticlib) plus C/C++ bindings via cargo-c, pkg-config, meson, CMake, and hand-written C++ companion.

### Changed

- Adopted cog for conventional commits + cargo-dist style releases; dropped .pre-commit-config.yaml and tbump (CI + cog bump handle lint/release).
- BREAKING: build system migrated from PDM to maturin mixed mode. Python sources moved under python/anneal/; wheel now ships Rust extension anneal._core.


## [0.1.0](https://github.com/HaoZeke/anneal/tree/v0.1.0) - 17-02-2024


No significant changes.
