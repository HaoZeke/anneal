# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!-- towncrier release notes start -->

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
