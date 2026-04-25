# Changelog

## [0.2.0] - 2026

### Changed

- BREAKING: build system migrated from PDM (`pdm-backend`) to `maturin` mixed mode. Python sources now live under `python/anneal/`; the wheel ships a Rust extension as `anneal._core` alongside the existing pure-Python helpers.
- BREAKING: Python tests moved from `tests/` to `pytest/`.

### Added

- Rust crate `anneal-core` (lib + cdylib + staticlib) scaffolding for the typed component algebra landing in v0.3.0.
- C/C++ bindings via `cargo-c` (pkg-config, headers, hand-written C++ companion).
- Build-system glue (`meson.build`, `meson_options.txt`, `CMakeLists.txt`) for downstream native consumers.
- Pixi workspace with `default`, `python`, and (in the next minor) `docs` environments.
- Orgmode + Sphinx (shibuya theme) documentation pipeline.
- `eindir>=0.2.0,<0.3` runtime dependency on the typed primitives crate.

### Removed

- PDM tooling (`pdm.lock`, PDM-specific sections in `pyproject.toml`).
- `environment.yml` (replaced by `pixi.toml`).
- `tbump.toml` (replaced by `cog bump`).
- `towncrier.toml` and `changelog.d/` news-fragment system (replaced by `cog changelog`).
- `.pre-commit-config.yaml` (CI handles fmt and clippy gating).

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!-- towncrier release notes start -->

## [0.1.0](https://github.com/HaoZeke/anneal/tree/0.1.0) - 17-02-2024


No significant changes.
