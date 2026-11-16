# CI workflows (Nickel + orchestrator)

## Honest status (not "fully friendly" yet)

This layout is **better than eleven hand-edited workflows with pin drift**, but it is
**not** a polished authoring UX. Treat it as an intermediate architecture.

| What is OK | What is still rough |
|------------|---------------------|
| One PR/main entrypoint (`ci-orchestrator.yml`) + `CI gate` for ruleset | Orchestrator is **hand-maintained YAML** (~650 lines), parallel to nickel mirrors |
| Nickel generates the five special-purpose workflows; `gha-drift` catches hand-edits there | Job bodies duplicated: orchestrator truth vs `build.ncl`/`prek.ncl`/… audit mirrors |
| `plan.ncl` + `export-plan` for traceability | Plan does **not** drive `needs:` (GHA cannot wire deps from JSON mid-run) |
| Ruleset lists orchestrator-era check names only | Cosmo/release/docs still separate (correct for triggers/secrets, more mental load) |

**Why the orchestrator is hand-edited (not a virtue, a tradeoff):**

1. **GHA platform limit** — only committed `.github/workflows/*.yml` at run start are
   executable. You always need *some* committed workflow file; you cannot "run only
   Nickel in prepare and execute generated workflows as new files in the same run."
2. **Speed vs ergonomics on the PR** — folding build/prek/potentials/docs into one
   workflow was done by assembling real GHA YAML first so CI could go green; lifting
   ~650 lines of multi-step `run: |` / `if:` / matrices into Nickel `m%"..."%` strings
   is tedious and easy to get wrong (see cosmo `working-directory` leak inside a run
   body).
3. **Nickel is the composition language we already ship** (`cigen` env). A second
   generator language (Nim, Elvish, …) does **not** remove the need for committed
   YAML; it adds another runtime on the authoring path unless it clearly deletes a
   whole class of bugs.

**The unfriendly part is not only "hand YAML" — it is shell-as-CI-logic.**

Long `run: |` / Nickel `m%"..."%` blocks (meson setup, torch wheel fixes, cosmo
smoke, prek/snapper installers) are hard to review, impossible to unit-test as
written, and the cosmo `working-directory` leak is exactly this class of bug.
Copying those blocks into `ci/gha/scripts/*.sh` is only a *mild* improvement
(file vs string); it is still imperative shell as the product.

**Directions that are actually nicer (pick a lane; do not stack all at once):**

| Approach | What it fixes | Cost / fit for rgpot |
|----------|---------------|----------------------|
| **A. potctl (or pixi tasks) owns steps** | GHA becomes `potctl ci meson-matrix` / `pixi r ci-build` — logic in Rust/Python/pixi, testable in normal crates | Best incremental: potctl already has `ci darwin-env`, `ci preflight`, `release assert`; extend that surface instead of more fragments |
| **B. Nix layer (flake checks / `nix run .#ci-*`)** | Hermetic envs, fewer `setup-pixi` + apt/curl/installer one-liners in GHA; one place for tool versions; optional pure builds | Real alternative to "shell + pixi in Actions"; competes with/alongside pixi unless you deliberately dual-stack or migrate. Still need *thin* GHA (or another runner) to invoke `nix build`/`nix run` — Nix does not remove the GHA platform limit |
| **C. Nickel only for workflow *graph*** | `orchestrator.ncl` → committed yml; `gha-drift` on everything | Fixes dual authoring; does **not** fix shell friendliness if bodies stay multiline strings |
| **D. scripts/*.sh only** | Slightly better diffs than inline strings | Still unfriendly; treat as interim at best |
| **E. Janet (or similar Lisp)** | One small embeddable runtime for *step/plan logic* and glue (`jpm`/`janet` in `cigen` or a tiny binary); readable loops/conditionals without bash; easy to test as normal scripts | Good middle ground if potctl shouldn't own every leg yet and you don't want full Nix. **Not** a second workflow emitter (don't compile Janet → 11 yml files). GHA still thins to `janet ci/gha/run.janet meson-job …` |

**Recommendation (opinionated):** prefer **A before more shell files**; evaluate **B**
seriously if the goal is "CI env and steps are declarative and reproducible," not
just "generate yml from nickel." If you want programmable steps without growing
Rust/potctl or a flake yet, **E (Janet)** is more plausible than Nim/Elvish-as-emitter
or a `scripts/*.sh` tree: one binary, lisp for control flow, shell only at the
edges (`os/spawn` meson/cmake). Still pair with **C** later so the *graph* is nickel
(or hand yml) and the *bodies* are `janet …` / `potctl …` / `nix run …` — one line
each.

Do **not** add Nim/Elvish/**Janet** as a third *workflow emitter* unless it
validates a plan/graph with real tests (then it's a linter, not an authoring path).

**Avoid:** nickel *and* hand orchestrator *and* shell fragments *and* a half-migrated
Nix flake *and* ad-hoc Janet — strictly worse.

## Decided architecture (pixi stays; no Nix for now)

| Layer | Owner | Role |
|-------|--------|------|
| **Workflow graph / matrices / `needs:` / triggers** | **Nickel** (`ci/gha/*.ncl` → committed yml; goal: include `orchestrator.ncl`) | Structure only — no meson/torch/prek installers in `m%"..."%` |
| **Env / packages** | **pixi** (`setup-pixi`, envs `devbld` / `metatomicbld` / …) | Unchanged; not Nix |
| **Step bodies (thin)** | **`potctl ci …`** (primary) | One line per GHA step: `potctl ci meson-test` / `cmake-test` / `build-test --sys …`. Logic + tests live in the `potctl` crate; same binary CI already stages via `setup-ci-tools` / `ensure-potctl` |
| **GHA / host** | Thin yaml | checkout, potctl artifact, pixi, `run: potctl ci …` only |
| **Janet** | *Not chosen* as the step plane | Fine as a *future* maintainer-only experiment; do not add unless potctl is deliberately kept tiny and you accept a second control-plane binary in CI |

**Why potctl over Janet here (decision):**

1. **Maintainers run it locally when CI fails** — not “end users of the library,” but *you* debugging a red matrix leg. Same entrypoint as Actions beats a CI-only Janet that only exists on runners.
2. **Already on the critical path** — orchestrator builds/restores potctl every run; darwin-env, preflight, release assert are already potctl. Growing `potctl ci` avoids a second runtime in `cigen`/PATH.
3. **Pixi stays the env plane** — potctl *invokes* pixi/meson/cargo; it does not replace pixi (and we are not doing Nix yet).
4. **Janet would fit only if** step glue must not live in Rust *and* you accept “CI/maintainer-only lisp, never expected locally except by people who install janet.” That is a valid small-team preference, but for rgpot it splits the control plane without a strong win over potctl.

**Target shape (bodies thin):**

```yaml
# nickel writes the job/matrix/needs skeleton; each step body is ~one line
- run: potctl ci preflight
- run: potctl ci meson-test --rpc ${{ matrix.rpc }} --cache ${{ matrix.cache }}
- run: potctl ci cmake-test --rpc ${{ matrix.rpc }} --cache ${{ matrix.cache }}
# not: 40 lines of meson setup / torch wheel / snapper installer in run: |
```

**Shipped orchestrator thin bodies:** `meson-test` / `cmake-test`, `ensure-torch-metatomic` +
`metatomic-test`, `xtb-tblite-test`, `xckernel-test`, `plugin-test`, `rpc-integ`, `towncrier-check` (auto-installs towncrier via
pipx/pip --user if missing), `bridge-stress-full`. Release-prepare: `release-assert` /
`towncrier-draft` / `cog-bump-dry-run` / `cog-check`. Still multi-line (honest): prepare export-plan,
prek/snapper installer curls+apt, rust nextest one-liner block, CI gate `needs.*.result` aggregator.

**Nickel does not own:** shell installers, torch wheel repair, cosmo build scripts (those move into potctl subcommands or existing cosmo paths).

Until `orchestrator.ncl` exists, edit the orchestrator directly but **prefer adding logic to potctl**, not new inline shell; keep `plan.ncl` stage ids aligned by hand.

## What is committed under `.github/workflows/`

| File | How maintained | Trigger |
|------|----------------|---------|
| **`ci-orchestrator.yml`** | **Hand-maintained** (primary PR CI; should become nickel-exported) | `push` `main` and `feat/**`, `PR` `main` and `develop`, `workflow_dispatch`, weekly `schedule` (docs hygiene only) |
| `release.yml` | Nickel `release.ncl` → `gen-gha` | `v*` tags |
| `release-prepare.yml` | Nickel `release_prepare.ncl` | path-filtered PR + `workflow_dispatch` |
| `ci_docs.yml` | Nickel `ci_docs.ncl` | docs build/deploy (write + gh-pages) |
| `ci_doc_commenter.yml` | Nickel `ci_doc_commenter.ncl` | `workflow_run` after docs |
| `cosmo-potctl.yml` | Nickel `cosmo_potctl.ncl` | path-filtered cosmo potctl |
| `isa-audit.yml` | Hand-maintained reusable workflow | `workflow_call` from a build job |
| `isa-audit-selftest.yml` | Hand-maintained | push to main and ci/** when the audit files change, and pull request; calls `isa-audit.yml` |
| `xtb.yml` | Hand-maintained | Linux `-Dwith_xtb=true`, `meson test --suite xtb` on every PR |

**Removed** (folded into orchestrator): `build.yml`, `potentials.yml`, `prek.yml`,
`docs_quality.yml`, `towncrier.yml`, `cosmo-potctl-spike.yml`.

Branch protection (ruleset `main` / id `12112837`): required contexts include
**`CI gate`**, `prepare (nickel plan)`, `prek`, `snapper semantic line breaks`,
`towncrier fragments`, `lychee link check`, `Build documentation` — not legacy
`Build Matrix` / standalone potentials/docs_quality workflow names.

## Orchestrator stages (`ci-orchestrator.yml`)

```text
prepare (export-plan → ci-plan.json + audit artifact)
    ↓
prek ∥ docs_snapper ∥ docs_lychee ∥ towncrier(PR) ∥ ci-tools
    ↓ (after ci-tools; skipped on schedule except docs_*)
build_and_test ∥ rust_tests ∥ client_bridge_stress
  ∥ potentials_metatomic ∥ potentials_uma_engine ∥ potentials_tight_binding
  ∥ potentials_tblite_06 ∥ potentials_plugins (plugin-test) ∥ xckernel_goldens
  ∥ build_windows_meson ∥ nodownload_rehearsal ∥ build_fast_nlist_off ∥ thread_limit
  ∥ eonviz_pins   (need `prepare` only; no potctl)
  (build_windows_meson: MSVC meson leg, fortran suppressed via
   ci/gha/machines/no-fortran.ini so the flang/MSVC kernel pairing stays out.
   nodownload_rehearsal: `git archive` tree built with --wrap-mode=nodownload,
   the conda-forge source-build rehearsal; fortran kernels included, since
   they are in-tree and pull no wrap.
   build_fast_nlist_off: meson with the linkcell/minimage pair list disabled.
   thread_limit: one meson build, the suite under OMP_THREAD_LIMIT=1 and =2.
   eonviz_pins: pixi task check-eonviz-pins and the byte-exact restart golden.)
  (NWChemPot frontend/stub always; energy/grad need an external libnwchemc engine.)
    ↓
ci_gate   ← single required-check aggregator
```

`prepare` always runs first. `plan.ncl` / `ci-plan.json` documents the same graph;
GHA `needs:` in the orchestrator is the executable order.

## Why some YAML remains (GHA limit)

GitHub Actions only loads workflow files **committed** under `.github/workflows/`
at the **start** of a run. Writing `build.yml` mid-run does **not** execute it.

| Approach | Feasible? |
|----------|-----------|
| Pre writes many `.yml`; GHA runs them as *new* workflows same run | **No** |
| One orchestrator + `needs:` + matrices; plan JSON for audit/trace | **Yes** (done) |
| Separate workflows for different triggers/secrets (tags, gh-pages, cosmo) | **Yes** (kept) |

## Commands (`pixi` env `cigen`)

| Command | Role |
|---------|------|
| `pixi r -e cigen export-plan` | `ci/gha/out/ci-plan.json` + `workflows-audit/` (mirrors + committed yml) |
| `pixi r -e cigen gen-gha` | Export **only** release/docs/cosmo/commenter yml from Nickel |
| `pixi r -e cigen gha-drift` | Fail if those nickel-exported yml drift (orchestrator excluded) |
| `pixi r -e cigen nickel-fmt` | Format all `ci/gha/**/*.ncl` |

Edit orchestrator jobs in **`.github/workflows/ci-orchestrator.yml`** directly; keep
`plan.ncl` stages in sync. Optional: mirror intent in `build.ncl` / `prek.ncl` /
`potentials.ncl` / … (audit-only via `export-plan`, not live workflows).

Edit **release/docs/cosmo** via Nickel (`ci/gha/*.ncl`), then `gen-gha` + commit yml.

## Layout

| Path | Role |
|------|------|
| `lib/pins.ncl` | Action / pixi / runner pins (shared by nickel modules) |
| `lib/steps.ncl` | Reusable builders for nickel-exported workflows |
| `plan.ncl` | Runtime plan stages → `export-plan` |
| `build.ncl`, `prek.ncl`, `potentials.ncl`, … | Audit/reference mirrors of orchestrator jobs |
| `release.ncl`, `ci_docs.ncl`, … | Live committed workflows via `gen-gha` |
| `gen.sh` / `export-plan.sh` / `check-drift.sh` / `format.sh` | Tooling |

## Pins

Bump in `lib/pins.ncl` for nickel modules; update matching pins in
`ci-orchestrator.yml` when changing orchestrator actions/pixi version.
