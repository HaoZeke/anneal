# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!-- towncrier release notes start -->

## [3.7.0] - 2026-11-17

[3.7.0]: https://github.com/OmniPotentRPC/rgpot/tree/v3.7.0

### Breaking

- C++ member functions of the potentials now use camelCase, like the
  other members of those classes.
  Rename the calls; no old name remains.

  - `PotentialBase::get_type()` is `PotentialBase::getType()`.
  - `PotentialBase::set_cache()` and `Potential::set_cache()` are
    `setCache()`.
    An override in a derived class must use the new name.
    The Python method stays `set_cache`.
  - `CP2KPot::probe_available()`, `CPMDPot::probe_available()`,
    `MOPACPot::probe_available()` and `NWChemPot::probe_available()` are
    `probeAvailable()`.
  - `CPMDPot::abi_available()`, `MOPACPot::abi_available()` and
    `NWChemPot::abi_available()` are `abiAvailable()`.
  - `lr::CoulombPot::set_neighbours()`,
    `lr::DispersionPot::set_neighbours()` and
    `lr::NlhPot::set_neighbours()` are `setNeighbours()`.
  - `lr::SumPot::share_neighbours()` is `shareNeighbours()`.
  - The namespace `rgpot::details` in `include/rgpot/errors.hpp` is
    `rgpot::detail`, like every other internal namespace.

- `RGPOT` no longer takes `U0`, `CUTOFF` and `PSI` in PLUMED units.
  Write `RGPOT ATOMS=... POTENTIAL=LJ OPTIONS={u0 1.0 psi 1.0 cutoff
  8.0} TYPES=Ar` instead, with the options in eV and Angstrom:
  `U0=96.48533212331002 CUTOFF=0.8 PSI=0.1` under GROMACS units is
  `u0 1.0 cutoff 8.0 psi 1.0`.
  `TYPES` is required.

### Added

- `StillingerWeberSi::forceImpl` writes the Cauchy stress and
  `caps().stress` is true.
  The stress is (1/V) dE/dε in eV/Angstrom^3, with the sign and layout
  of the `LJPot`, `MorsePot` and `ZBLPot` stress.
  The virial sums each folded bond vector times the energy gradient with
  respect to that vector.
  That sum is the many-body virial of Thompson, Plimpton and Mattson,
  J. Chem. Phys. 131, 154107 (2009).
  Its digital object identifier (DOI) is 10.1063/1.3245303.
  Energies and forces keep their bits.
  `SWSiTest` now also runs under CMake.

- The Fortran `SWPot` (`PotType::SWSi`) writes the Cauchy stress and
  `caps().stress` is true.
  It sums the same bond-vector virial as `StillingerWeberSi`, so its
  stress is the `StillingerWeberSi` stress times the epsilon ratio
  1.068176741337.
  Its energies and forces keep their bits.
  `SWPot::forceBatch`, the call eOn's engine makes, skips the virial and
  leaves `has_stress` at 0.
  In Python, `StillingerWeberSi` gains `caps()` and `evaluate()`, which
  returns the stress, so `RgpotCalculator` reports the Atomic Simulation
  Environment (ASE) stress for it and for `SWPot`.

- `rgpot_lammps_potential`, `rgpot_gromacs_potential` and
  `rgpot_openmm_potential` are the public constructors in C++, the C ABI
  and Python.
  LAMMPS (Large-scale Atomic/Molecular Massively Parallel Simulator)
  dlopens the engine plugin.
  GROMACS and OpenMM store the nanometre and kJ/mol conversion once.

- The engine drivers are `pair_style rgpot`, a PLUMED `RGPOT` collective
  variable used by `gmx mdrun -plumed`, an OpenMM shifted Lennard-Jones
  comparison, and an `rgpot_engine_force` Lennard-Jones plugin.
  Each one is a meson test when its library is present.

- UmaPot evaluates a same-composition stencil in chunks of the package
  `batch_max` and keeps the per-system loop when `batch_max` is 0 or 1.
  A static band (`--batch-style static`) is traced at exactly
  `batch_max` systems, and a short chunk repeats its last system.
  `--batch-style packed` is one disjoint graph with a dynamic atom count
  and a dynamic edge count (`batch_packed`).
  UmaPot sends the real system count and does not repeat a system.
  `scripts/export_uma_aoti.py --batch-max N --molecular-box L` traces
  either graph.
  The force gradient is traced on torch builds that gate
  `torch.autograd.grad`, and `OMP_NUM_THREADS` is applied to torch
  before the compile because the CPU general matrix multiply (GEMM)
  kernel is built for that thread count.

- `scripts/cpu_scaling.py` writes strong, one-cell and weak CPU scaling
  tables as comma-separated values (CSV).
  Each CSV header names the host, the compilers and the build options.
  Every row is checked bit for bit against a 1-thread call.
  It runs only when called and changes no library behaviour.

- Free-cluster Cambridge Morse and reduced Lennard-Jones kernels,
  `rgpot_cambridge_morse` and `rgpot_cluster_lj`.
  Both sums run over every pair: a neighbour list would drop a tail that
  is not zero.
  Energy and gradient are one pass.

- Free-cluster Cleri–Rosato kernel `rgpot_gupta`.
  The sum visits every pair: a neighbour list would drop a tail the
  Cambridge minima include.
  The density is accumulated once and the force reuses that pair
  derivative.

- Hessian-vector products for any potential, for minimum-mode searches
  (Lanczos, dimer) that apply the Hessian to many directions at one
  geometry.
  In the C ABI, `rgpot_potential_hvp` computes `H v` on a potential
  handle.
  `RGPOT_HVP_FORWARD`, the default, reuses the forces of the last centre
  the handle evaluated, so k products at one geometry cost k + 1 force
  calls instead of 2k, at first order in the step.
  `RGPOT_HVP_CENTRAL` costs two calls a product at second order.
  A potential that knows its Hessian registers an analytic product with
  `rgpot_potential_set_hvp_callback` and the products make no force
  call.
  `rgpot_potential_hvp_centre` returns the centre energy and forces from
  the same cache, `rgpot_potential_hvp_invalidate` drops it, and
  `rgpot_potential_hvp_counts` reports the calls made.
  The centre is keyed on the bits of the positions, atomic numbers and
  cell.
  In C++, `PotentialHandle` gains `hvp`, `hvp_centre`,
  `set_hvp_callback`, `invalidate_hvp` and `hvp_counts`, and
  `rgpot/HessianVector.hpp` gives the same products to potentials used
  without the Rust core.
  `MorsePot::hessianVectorImpl` is the analytic product of the shifted
  and switched Morse potential, and `PotentialHandle::from_impl`
  registers it.
  In Python, every potential gains `hvp`, `hvp_centre`,
  `invalidate_hvp`, `hvp_counts`, `reset_hvp_counts` and
  `has_analytic_hvp`.

- Long-range summands behind `-Dwith_long_range=true`: Ewald, smooth
  particle mesh Ewald and damped shifted force electrostatics, D3(BJ),
  D4, MBD and QDO dispersion, SO3LR NLH repulsion, and a sum that adds
  them to any other potential on the same `ForceInput`.

- The PLUMED action `RGPOT` evaluates any potential `pair_style rgpot`
  accepts: `POTENTIAL=Name OPTIONS={key value ...} TYPES=El,...`, with
  the names and options of `pair_coeff` (now listed once in the
  potential names reference).
  Positions, the cell, the value and its derivatives follow PLUMED's
  units, converted with its `Units` object.
  Periodic cells, including triclinic GROMACS boxes, go to the
  potential, and the box derivatives come from its stress, so
  `BIASVALUE` under `gmx mdrun -plumed` gives the right virial and
  pressure in NPT runs.
  A potential without a stress stops on a periodic cell; `NOPBC`
  evaluates an isolated cluster.
  New meson tests check the forces and box derivatives against
  `plumed driver --debug-forces`, the units, and the C and Python
  interfaces, and `gromacs_mdrun` now also checks the bias, virial and
  pressure of a short Sutton-Chen NPT run.

- The header `rgpot/Staged.hpp` lets a host that builds its own
  neighbour list, such as LAMMPS, evaluate a finite-range potential on
  that list in two stages, the scheme of `pair_style eam`:
  `densityStage` sums each atom's densities, the host completes them
  across its copies of neighbouring atoms (`Locality::densities` values
  per atom), and `forceStage` adds the forces.
  `HalfList` takes a half list in the layout of a LAMMPS `NeighList`,
  with newton on or off, and `StageTally` takes the host's share of the
  energy and virial and the per-atom split.
  `LJPot`, `MorsePot`, `SuttonChenPot` and `EAMTablePot` have the
  stages, the concept `rgpot::StagedPotential` detects them, and
  `rgpot::spec::Built` carries them for a potential built by name.
  `PotCaps` and the C++ layout are unchanged.

- `CP2KPot` runs CP2K Quickstep through `libcp2kc_rgpot`, a shim over
  libcp2k built with `-Dwith_cp2k=enabled`.
  `CP2KParams` (`PotentialConfig.cp2k`) renders a default GPW deck or
  fills `@RGPOT_COORD@` and `@RGPOT_CELL@` in a verbatim template.
  The shim keeps one force environment per orbital key, and an
  unconverged SCF raises `rgpot::CP2KScfError` on every rank instead of
  aborting the MPI world.
  potserv serves it as `CP2K`.

- `LJPot`, `LJClusterPot` and `MorsePot` gain `evaluate_per_atom`, which
  returns everything `evaluate` does plus per-atom `energies` (shape
  `(N,)`, eV, summing to the energy) and `virials` (shape `(N, 6)`, eV,
  Voigt order, summing to minus the volume times the stress).
  It exists only on potentials whose C++ kernel has `forcePerAtom`.

- `compute pe/atom`, `compute stress/atom` and
  `compute centroid/stress/atom` work with `pair_style rgpot` for LJ,
  Morse, SuttonChen and EAMTable, on any number of ranks; they match
  `lj/cut`, `morse` and `eam/alloy` atom by atom.
  `LJPot`, `LJClusterPot`, `MorsePot`, `ZBLPot`, `SuttonChenPot` and
  `EAMTablePot` gain `forcePerAtom`, which also writes per-atom energies
  (pair terms split half to each atom, EAM and Sutton-Chen embedding to
  its own atom) and per-atom virials; `rgpot::PerAtomPotential` in
  `rgpot/PerAtom.hpp` detects it.
  No virtual or data member changes, so the C++ layout is unchanged.
  Potentials without the split stop the run with an error naming the
  potential.

- `librgpot_gromacs` (`-Dwith_gromacs_dropin=enabled`, off by default)
  serves any rgpot potential through the libcp2k C API of the GROMACS
  QM/MM interface.
  A double-precision GROMACS 2024 to 2026 built with `-DGMX_CP2K=ON`
  against it (`engines/gromacs/build_gromacs.sh`) evaluates the QM group
  in `gmx mdrun` without PLUMED, with the energy in GROMACS'
  `Quantum En.` term.
  An `&RGPOT` section of the `grompp -qmi` file chooses a `pot_spec`
  name and options or an `rgpot_engine_*` library and its configuration;
  elements come from the topology.
  Periodic (including triclinic) and isolated QM regions work; there is
  no virial, so NPT is not supported, and MM charges do not act on the
  QM region.
  Bad input stops `mdrun` with an `rgpot-gromacs:` message.
  New meson tests `gromacs_dropin_api` and `gromacs_dropin_mdrun`
  compare with `rgpot-eval` through real `mdrun` runs, and
  `engines/gromacs/bench_dropin.py` measures the per-step cost.

- `pair_style rgpot` evaluates finite-range potentials by domain on more
  than one rank, the way `pair_style eam` does.
  For LJ, the `u0 psi` form, Morse, SuttonChen with a positive cutoff
  and EAMTable, each rank runs the density and force stages of the
  potential on the half neighbour list of LAMMPS, whose ghosts reach one
  cutoff plus the skin, and the densities of a many-body potential pass
  between the ranks between the two stages, so the force call no longer
  runs on rank 0 alone.
  On one rank with `mode domain`, the 16384-atom Lennard-Jones box of
  `bench_pair_rgpot.py` takes 8.27 ms per step (`lj/cut` 8.38, the
  gather path 15.10) and the 32000-atom CuNi box 46.84 ms (`eam/alloy`
  44.47, the gather path 70.10).
  Energies, forces, the global virial, `pe/atom` and `stress/atom` match
  the gather path and `lj/cut`, `eam/alloy` and `morse` on 1, 2 and 4
  ranks, under `newton on` and `off`, including ranks that own no atoms.
  Engines, potentials without a finite range or the two stages,
  `virial fd` and one-rank runs keep the gather on rank 0.
  The new `pair_style rgpot cutoff mode auto|gather|domain` keyword
  forces either path, and `init` logs which one runs and why.
  In C++, `rgpot/Locality.hpp` adds `rgpot::Locality` and the concept
  `rgpot::LocalPotential`, a non-virtual `locality()` on `LJPot`,
  `MorsePot`, `SuttonChenPot` and `EAMTablePot`; `PotCaps` and the C++
  layout are unchanged.

- `pair_style rgpot` works as a native LAMMPS pair style.
  It fills the global virial from the potential's stress, or from a
  strain difference when there is none, so pressure, NPT and
  `fix box/relax` work in orthogonal and triclinic boxes.
  `pair_coeff * * Name [key value ...] El ...` selects an rgpot
  potential by name and maps types to elements.
  MPI runs gather the box on rank 0.
  Other unit styles stop with an error.
  `rgpot-eval` evaluates a named potential on one geometry.

- `rgpot.ase.RgpotCalculator` accepts any potential class the bindings
  expose, by class name or kind string (`"morse"`, `"ljcluster"`,
  `"tersoffsic"`, ...; `rgpot.ase.available_kinds()` lists them), and
  passes keyword arguments to that class's constructor.
  It reports per-atom `energies` and `stresses` for LJ, LJ-cluster and
  Morse, offers a finite-difference `stress` for kernels without one
  (`numerical_stress=True`), rebuilds the potential on `calc.set(...)`,
  and pickles, `copy`s and `deepcopy`s by rebuilding the potential from
  its parameters.
  It matches ASE's `LennardJones`, `MorsePotential` and an analytic
  `EAM` for EAM-Al to 1e-10 relative, and runs under the ASE optimizers,
  cell filters, Velocity Verlet, NEB, `Vibrations` and `SumCalculator`.

- `rgpot::IPIDriver` and the `rgpot-ipi-driver` program are the client
  side of the i-PI socket protocol (built with `with_ipi`).
  They connect to i-PI, ASE's `SocketIOCalculator` or `IPIPot` over a
  UNIX socket or TCP and answer `STATUS`, `INIT`, `POSDATA`, `GETFORCE`
  and `EXIT` with any potential `engines/pot_spec.hpp` builds, returning
  hartree, hartree/bohr and the virial `-sigma V`.
  Open clusters get a padded box, thin periodic cells are tiled to the
  cutoff, and `--stress-step` gives a finite-difference virial for
  kernels without stress.
  The flags follow `i-pi-py_driver`.
  `rgpot-ipi-bench` times the socket round trip against a direct call;
  `docs/orgmode/howto/ipi.org` now covers both directions and lists the
  timings.

- `scripts/export_uma_aoti.py` exports a dynamic UMA package by default:
  the atom count (2 to `--max-atoms`, default 1024) and the edge count
  (1 to `--max-edges`, default 300 per atom) are symbolic, so one
  package keeps working when an NEB image or a dimer step changes the
  neighbour list, and, with the MoLE experts kept in the graph, for
  other molecules of the same task head.
  The package is kept only after it matches `FAIRChemCalculator` on the
  input, on the input with one atom moved past the cutoff, and on a
  second molecule.
  The checkpoint can come from `--checkpoint` or `RGPOT_UMA_CHECKPOINT`.
  `UmaPot` refuses an atomic number above the package's `z_max` before
  the compiled graph runs.

- `scripts/export_uma_aoti.py` resolves the checkpoint once
  (`--checkpoint`, `RGPOT_UMA_CHECKPOINT` or a Hugging Face download)
  and records it in the package: `checkpoint_source` (`cli`, `env` or
  `hf`), `checkpoint_path` or `checkpoint_hf_repo` and
  `checkpoint_hf_filename`, the resolved commit as
  `checkpoint_hf_revision`, and `checkpoint_sha256` of the file it
  loaded.
  `UmaPot::checkpointProvenance()` and the exporter's
  `checkpoint_provenance()` read them back; packages exported earlier
  load as before and report the keys absent.

### Developer

- The benchmark description states the comparison in plain words.
  The upper bound is at least the widest calibration ratio.
  The comparison that fails on a slowdown is the one to run.
  Continuous integration should use that comparison's exit status.

- The C ABI page lists the public C functions.
  The coverage check fails when one of them is missing from that page.
  The ABI baseline keeps the neighbour and fast-list exports.

- The develop continuous integration job builds a serial LAMMPS with the
  PLUGIN package from the same headers as `pair_style rgpot`.
  It then requires the LAMMPS, GROMACS, OpenMM and PLUMED tests to pass.

- Configuring the engine drivers accepts an include path that lies
  inside the source tree.
  `include_directories()` rejects that absolute path, so a pixi prefix
  or a LAMMPS checkout under the workspace could not configure.
  The path is made relative to `engines/meson.build`.
  A path outside the tree stays absolute.

- CMake's `nlist_rust` test passes the script the OpenMP libraries
  FindOpenMP chose, and the runtime's directory as an rpath, so the
  shared library the script links finds the OpenMP runtime with
  AppleClang, which has no `-fopenmp`.
  Called without them, as meson calls it, the script uses `-fopenmp`
  only if a test program links with it, so a build without OpenMP no
  longer fails where clang takes the flag but finds no libomp.

- Every CI job that runs on a runner sets `timeout-minutes`: its
  measured p95 plus a margin that covers the longest per-test timeout it
  runs, so meson or CTest still kills and names a hung test before the
  job times out.
  `ci/gha/README.md` gives the rule and the numbers, and shows how to
  tell a job no runner took, which a timeout cannot catch, from one that
  hung.
  The reusable `isa-audit.yml` takes a `timeout-minutes` input, 10 by
  default.

- Every OpenMP region in the library now opens in
  `CppCore/rgpot/Slices.hpp`, through `forEachSlice`,
  `forEachSlicePhase` or `onFullTeam`, so a result depends on its slice
  count and not on how many threads the runtime starts.
  `CppCore/tests/test_openmp_confined.py` fails the test suite when
  OpenMP appears in any other library source, `ThreadTeamTest` checks
  the threaded potentials and neighbour tables on short, dynamic and
  nested teams, and the `thread_limit` CI job runs the suite under
  `OMP_THREAD_LIMIT=1` and `OMP_THREAD_LIMIT=2`.
  The long-range summands follow the same rule, so particle mesh Ewald,
  D3 and D4 no longer stall or skip work when the runtime starts fewer
  threads than they ask for.

- The C ABI stamp of `rgpot-core` moves to 1, 2 and 1: the
  Hessian-vector entry points are additions, and `rgpot_potential_t`
  keeps its layout.
  `scripts/abi_core_baseline.txt` lists the `rgpot_*` functions the
  `rgpot-core` shared library exports.
  The meson test `abi_core_symbols` fails when one of them is no longer
  exported.
  `scripts/abi_symbols.sh` takes the name pattern with `-p`, ahead of
  the library and its baselines.

- The ISA audit proves dispatched clones from the linked library instead
  of reading them from `rgpot/isa-allow.toml`.
  A local function reached only behind a `__builtin_cpu_supports` test
  or from a `target_clones` resolver may use the features that test
  covers, so the AVX2 Cambridge Morse, cluster Lennard-Jones and Gupta
  bodies pass the wheel audit with no entry.
  AVX in any other function is still a violation, and the report lists
  each clone with its dispatcher.

- The ISA self-test links its dispatch fixtures with `-nodefaultlibs`,
  so they import libgcc's CPU model and the proof reads `__cpu_model`
  through the GOT.
  The static copy is built for the toolchain's own `-march`, and on the
  Ubuntu 26.04 runner its `__cpu_indicator_init` has two BMI2 `shlx`.
  `allow-dispatch.toml` drops its `__cpu_indicator_init` entry.

- The `abi_symbols` test picks the `nm` mode from the library it is
  given: `nm -D --defined-only` for a shared object and
  `nm -g --defined-only` for a static `lib*.a` or `*.lib` archive, which
  has no dynamic symbol table.
  macOS still uses `nm -gU`.
  A small fixture built as a shared object and as both archive kinds
  runs every mode in every `with_tests` build, and checks that a shared
  object's hidden names and an archive's missing names still fail.

- `CppCore/rgpot/rpc/Potentials.ordinals` records every field, union
  arm, enumerant and method ordinal of `Potentials.capnp` with its name
  and type, append only.
  `schema_ledger.py check` (meson suite `schema`, CI job
  `schema ordinal ledger`) holds the pinned `potentials-schema` commit,
  the local series diffs and both vendored copies to it, and refuses a
  series diff that removes or rewrites an `@N` line or adds an ordinal
  that is not a `local` ledger entry.

- `CppCore/tests/registry.txt` lists every test with the features that
  gate it, its kind and its options, and meson and CMake both register
  their tests from it.
  CMake now registers each test meson does except those only meson can
  run, whose lines give the reason: in the build the `thread_limit` CI
  job tests, that is 99 of meson's 107 tests, where CMake registered 31.
  CMake's unit tests link OpenMP as meson's do, and its ABI symbol tests
  run their script through `sh`.
  `CuH2PotTest`, `CapnpAdapterTest` and `BridgeStressTest` take meson's
  names, `CuH2test`, `RPCTest` and `RPCClientTest`.
  Each build registers `test_list_parity`, which fails when the build's
  tests or their options differ from the registry.
  `tools/check_test_parity.sh` configures the plain, the RPC and cache,
  and the client-only builds with meson and CMake, without building
  them, and fails when the two register different tests.
  The `test_parity` CI job runs it, and `ci_gate` waits on that job.

- `EAMCellVerletTest` requires a NaN wherever the direct pair loop gives
  one, whatever its sign bit, and every other value bit for bit.
  IEEE 754 does not specify the sign of a NaN result, and optimised
  Clang builds flip it for the forces between coincident atoms, so the
  test failed in the macOS CMake builds and passed in the unoptimised
  meson builds.

- `docs/doccheck.py` checks the documentation against the source.
  It requires a reference entry with Inputs, Units, Outputs and Errors
  for every public C, C++ and Python name, and a classification for
  every code block in the README and the docs.
  The C names include the Metatomic and XTB engine tables and the
  libcp2k API of the GROMACS drop-in.
  The entry of each `*Config` struct must name every field, a reference
  page must name every field the rgpot-local schema diffs add, and the
  `&RGPOT` key table must list the keys the drop-in reads.
  It runs, compiles or statically checks each block according to that
  classification, and flags option names, paths and `RGPOT_*` names that
  do not exist; paths that `.gitignore` covers, such as `docs/build`,
  count as build outputs.
  `pixi r -e docs doccheck` runs it, and the docs CI job runs it with
  the Sphinx build, which now uses `-W --keep-going`.
  It replaces `docs/check_c_symbols.py`; the engines CI job no longer
  asks for the `c_symbols` test.

- `potctl ci cmake-test` and `potctl ci bridge-client-build` give
  `cmake --build` an explicit `--parallel` count:
  `CMAKE_BUILD_PARALLEL_LEVEL` when it is set, the usable CPUs
  otherwise.
  A bare `-j` let make start every ready compile at once, and with the
  whole test registry to build, the macOS CMake legs (3 cores, 7 GB)
  spent 7 to 49 minutes compiling, where the meson legs of the same run
  built and tested in 3 to 6.

- `rgpot::simd::exp_hw_n` is declared on every target.
  MSVC, Arm and other builds without the x86-64 hardware body get one
  that calls `rgpot::simd::exp`, so a caller that picks the body at run
  time compiles everywhere.

- `sc_call_bench` (`meson test --benchmark sc_call_bench`) times
  Sutton-Chen calls on free clusters of 13 to 561 atoms with no cutoff:
  the C entry `rgpot_sutton_chen_energy_forces`, a `SuttonChenPot` built
  per call, one kept across calls in the default and the streaming
  neighbour mode, and a plain half-pair loop.
  Each mode also prints its largest energy and force gaps to that loop
  and a hash of the energy and force bits, so two builds can be compared
  bit for bit.

- `tools/check_test_parity.sh`, and so the `test_parity` CI job, also
  reads the `compile_commands.json` meson and CMake write when
  configured, and fails when a source both compile gets different
  contraction or fast-math flags from each.
  `tools/check_fp_parity.py` does the comparison.

### Changed

- A stored `nlist::Neighbours` list is kept when the caller wraps atoms
  back into the periodic cell.
  The half-skin test measures each atom's move to the nearest image of
  its reference position, so an atom that crossed a face no longer
  counts as having moved a cell vector, and the bins are redone after
  such a wrap.
  The potentials that keep such a list (`LJPot`, `LJClusterPot`,
  `MorsePot`, `ZBLPot`, `SuttonChenPot`, `EAMPot`, `StillingerWeberSi`,
  `TersoffSiC` and `EAMTablePot`) rebuild it less often on wrapped
  input: on the gather path of `pair_style rgpot`, 3 instead of 40
  builds in 112 calls for 16384 Lennard-Jones atoms.
  One thread walks the pairs in index order, so its results keep their
  bits; with `OMP_NUM_THREADS` above 1 the order follows the bins and
  results agree to rounding.
  `nlist::Builder`, whose table stores image shifts, is unchanged.

- An infinite cutoff or skin in `rgpot::nlist::Neighbours` walks every
  pair straight from the coordinates, with no bins and no stored list,
  `i` ascending and then `j` ascending, and keeps the update for any
  move while the atom count, the cell and the spec stay the same;
  `allPairs()` reports it.
  That walk takes a copy of its callback, so a caller reads back only
  the state it captured by reference.
  `SuttonChenPot` with `cutoff <= 0` searches that way.
  It used to stand in a 1e6 search, which binned a cluster into one
  cell, read `MemAvailable` twice and stored and sorted every pair on
  each call.
  Energies and forces keep their bits on one thread.
  A team of threads used to leave that one cell to one thread and now
  splits the atoms into runs, one per thread: the forces keep their
  bits, and the energy, a sum over threads, can move in the last bits.

- `EAMTablePot` keeps its pair list between force calls, with a Verlet
  skin of `EAMTableConfig::skin` (1 Angstrom by default, `skin` in
  `pair_coeff * * EAMTable ...` and the other hosts of
  `rgpot::spec::build`), instead of building a list with no skin on
  every call.
  Energies, forces, stress and per-atom terms keep their bits for any
  skin.
  The spline tables were and are made once, in the constructor.
  Because the list is per-instance scratch, `caps().reentrancy` is now
  `Reentrancy::PerInstance`: a caller that evaluates one `EAMTablePot`
  from several threads at once needs one instance per thread, as for
  `LJPot`.
  On the gather path of `pair_style rgpot` a 32000-atom CuNi step went
  from 338.63 to 70.74 ms on one rank, with one list build in 62 calls
  (`docs/orgmode/howto/engine-lammps.org`).

- `EAMTablePot` locates each pair distance on its radial grid once per
  pass.
  It located it once per table before: twice per pair in the density
  pass and three times in the force pass.
  It also keeps the three radial tables of each element pair together
  instead of indexing nested vectors for every pair.
  The arithmetic is the same, so the results do not change.
  On one rank with `mode domain`, a 32000-atom CuNi step of
  `pair_style rgpot` went from 54.13 to 47.03 ms, against 43.87 ms for
  `eam/alloy`.

- `SuttonChenPot` keeps the powers and `1/r^2` that the force of each
  pair takes from the density pass, and the force pass reads them back
  instead of raising the distance to `n` and `m` a second time.
  `setPairTermLimit` bounds the store, 32 MiB by default, and the force
  pass computes the pairs past it again.
  The pair lambdas inline into the walks and take their constants by
  value, and the density and weight arrays are kept with the instance.
  `SuttonChenPot.cc` is built with `-ffp-contract=off`, as the other
  kernels are, so reading a term back and computing it again round
  alike.
  Energies, forces, the stress and the per-atom split keep their bits
  wherever the old build fused no multiply into an add.

- `potentials-schema` is pinned at 190fee6, upstream main 7fa9046 plus
  the CP2K arm, and both vendored `Potentials.capnp` copies are that
  file byte for byte.
  `PotentialConfig` gains the upstream arms `xtb` @6, `tblite` @7, `uma`
  @8, `lj` @9, `morse` @10, `zbl` @11, `d3` @12, `d4` @13, `mopac` @14,
  `skala` @15, `expr` @16 and `cp2k` @17, with their parameter structs,
  so rgpot numbers them as gpridge and every other consumer does.
  `BuiltinParams` gains `config` @2, `PotentialSpec` gains `cp2k` @6,
  `PotSpec` gains `potential` @2 and `MetatomicParams` gains
  `lengthUnit` through `attachSystemExtras` (@13 to @18).

- `rgpot_sutton_chen_energy_forces` and `rgpot_lj_eval` keep the
  potential of their last call on each thread instead of building one
  per call, so the next call with the same parameters and atom count,
  and for `rgpot_lj_eval` the same atomic numbers, reuses its neighbour
  update and scratch.
  Any other call, and one that throws, drops it and builds a new one.
  Energies and forces keep the bits of a potential built for the call.
  `rgpot::Kept` in `rgpot/KeptPotential.hpp` holds one potential per
  thread and potential type, keyed by the bits of its parameters.
  With this and the unbounded search and kept pair terms, a Sutton-Chen
  12-6 call on a free 55-atom cluster with no cutoff went from 60.901 to
  13.690 microseconds on one core (`sc_call_bench`,
  `docs/orgmode/reference/sutton_chen.org`).

### Fixed

- UmaPot treats a package with neither `nedges_min` nor `nedges_max`
  as fixed-shape when `shapes` records `edge_index`.
  The second extent is the traced edge count, and the neighbour list
  must match it.
  A published gpridge export records that list, for example
  `edge_index` `[2, 1332]`, and it used to be refused.
  A package with only one of the two keys is still refused.

- A debug or sanitizer build no longer aborts when a vesin neighbour
  list with no pairs is reset or recomputed.
  The reset had called memset on a null pairs pointer, which is
  undefined even when the byte count is zero.
  It now poisons only the storage the list allocated, and it stops at
  capacity.

- On a cell narrower than three cutoffs, `rgpot_fast_pairs` answers a
  short buffer from n(n-1)/2 and fills the caller's buffer directly.
  For the 10k-atom bridge cell, it no longer builds a 400 MB pair list
  just to report the count.
  A short-buffer return also clears the thread-local error message.

- `RgpotCalculator` reports the stress of a non-periodic structure as
  the cluster virial divided by the caller's cell volume, so it matches
  ASE's finite-difference stress.
  A non-periodic evaluation still uses a padded cube so images stay
  outside the cutoff.
  ASE's finite-difference stress divides the same virial by the caller's
  cell, so the cube stress is rescaled by V_cube / V_caller.

- The C++ soversion row is found by index.
  A pointer into `kCxxSoversionTable` is not a constant expression under
  `-fsanitize=undefined`, so the layout static asserts failed that
  build.
  The recorded hash is unchanged.
  `tools/check_cxx_layout_ubsan.sh` configures the address and undefined
  sanitizers and runs `CxxLayoutTest` and `CxxLayoutVirtualCounts`.

- The nine doubles stored in `rgpot_potential_t` are lattice vectors as
  rows, the same layout as `ForceInput.box`.
  The header comment had called that matrix column-major.

- The LAMMPS and OpenMM configure probes ignore `engines/lammps` and
  `engines/openmm`.
  Those directories sat on the interpreter's path, so the probes
  reported the modules present and the meson suite ran tests that need
  the libraries.

- A CMake build compiles the core library sources and the reduced pair,
  Lennard-Jones, Sutton-Chen and Tersoff potentials without
  floating-point contraction, as meson does.
  On arm64, where the compiler fuses `a*b+c` into one multiply-add by
  default, the CMake library's results differed from meson's in the last
  bits, and `ReducedPairTest` and the skewed-cell case of
  `FastNlistTest` failed on macOS.

- A CMake build with the cache links the shared RocksDB library when
  there is one, as meson does.
  conda-forge's static RocksDB put jemalloc ahead of libc in every
  program that links `librgpot`, and such a program crashed in `free`
  once it loaded `libmpi`, as the MPI calculator group and collective
  RPC tests do.

- A GFN2 single point that does not converge at the configured
  electronic temperature is retried with smeared occupation and
  evaluated again at that temperature.
  The returned energy is the cooled one.

- A meson build with `with_eigen=true` configures again when neither
  pkg-config nor CMake finds `eigen3`.
  The lookup passed `dependency()` a `subproject:` keyword, which
  current meson rejects; it now names the fallback
  `['eigen', 'eigen_dep']`, so Eigen comes from
  `subprojects/eigen.wrap`.

- A neighbour walk split across more slices than the OpenMP runtime
  started threads now walks every slice.
  `nlist::Neighbours::forCellTeam`, the threaded `nlist` builds and
  `simd::fill_view` handed slice `t` only to thread `t`, so a build
  without OpenMP, a call inside another OpenMP region or a run under
  `OMP_THREAD_LIMIT` dropped the pairs of the missing threads: with
  `OMP_NUM_THREADS=3` and `OMP_THREAD_LIMIT=2`, the threaded Sutton-Chen
  energy came out wrong, and the macOS meson build, which has no OpenMP,
  failed `NeighbourLayerTest`.
  `nlist::forEachSlice` runs the slices on the threads the runtime
  starts, so a sum taken in slice order is the same for any team.
  A build inside another OpenMP region is no longer ruled out.

- A project that vendors rgpot with `subproject()` no longer installs
  `rgpot-lj` and the engine plugins from `engines/`.
  They are installed only when rgpot is the top-level project, like the
  rest of rgpot, so `meson install` in the parent no longer stops on an
  engine output the parent never built.

- A threaded pair-list `build` (`OMP_NUM_THREADS` above 1, at least 8192
  atoms) now stores the same pairs as one thread when a pair has more
  than one image inside the cutoff, as in a periodic box thinner than
  two search radii.
  Before, it kept only one image of each pair and lost the rest.

- Both vendored `Potentials.capnp` copies are again the pinned
  `potentials-schema` 190fee6 file byte for byte, so
  `schema-sync-canonical-rpc` and `schema-sync-canonical-core` pass when
  a parent such as gpridge resolves `potentials-schema` from its own
  wrap.
  The rgpot-local neighbour fields (`ForceInput.neighbors` @9,
  `PotentialResult.neighborList` @28, `Potential.neighbours` @13 and the
  `NeighborRequest`, `NeighborSystem` and `NeighborList` structs) moved
  from the wrap's `diff_files` to the add-only series
  `CppCore/rgpot/rpc/local/series`, mirrored in
  `rgpot-core/schema/local` for the crate.
  Meson, CMake and `rgpot-core/build.rs` apply the series and compile,
  install and serve that effective schema, which is unchanged on the
  wire.
  `schema_ledger.py check` now refuses a vendored copy that differs from
  the pinned file or carries a `local` ledger entry, a wrap that sets
  `diff_files`, and a series mirror that drifts;
  `schema_ledger.py apply` writes the effective schema.

- Repeating `bindCalculators` with an argument of 0 or less returns the
  stored group when it names the same split.
  Before, the repeat compared the raw argument with the resolved size
  and returned index -1.

- The C++ potentials that share the neighbour list (LJ, Morse,
  Sutton-Chen, ZBL, EAM, Stillinger-Weber, Tersoff) are several times
  faster on a sheared or otherwise non-orthogonal periodic cell, with
  bit-identical results.
  The minimum image of a pair no longer searches the nearby lattice
  images when the rounded image is provably the shortest, and the
  neighbour list drops a pair without that search when every image lies
  beyond the cutoff plus skin.
  Through `rgpot.ase.RgpotCalculator`, a sheared 32-atom copper cell,
  which the Morse cutoff tiles into 864 atoms, went from 203 to 37 ms
  for the first evaluation and from 48 to 6.1 ms for each one after a
  small move.

- The Rust RPC server rejects a `calculate` request whose positions do
  not hold three values per atom or whose box does not hold 9 values,
  and a callback result whose forces tensor has the wrong size.
  Before, the callback read past the end of a short request and the
  server read past the end of short forces.

- The headers in `include/rgpot` include each other by file name, so
  `rgpot/rgpot.hpp` compiles on a case-insensitive file system such as
  APFS or NTFS.
  There, with `CppCore` ahead of `include` on the search path,
  `rgpot/potential.hpp` also named `CppCore/rgpot/Potential.hpp`, and
  code using `rgpot.hpp` failed with "no member named
  'PotentialHandle'".

- Threaded stored EAM, `LJClusterPot`, `SWSi` and `TersoffSiC` now
  evaluate every atom when the OpenMP runtime starts fewer threads than
  `OMP_NUM_THREADS` asks for.
  Each split the atoms into one slice per requested thread and handed
  slice `t` only to thread `t`, so a call inside another OpenMP region
  or under `OMP_THREAD_LIMIT` left out the atoms of the missing threads:
  with `OMP_NUM_THREADS=2` and `OMP_THREAD_LIMIT=1`, the stored EAM
  energy of a 500-atom aluminium crystal was 151.6 eV off.
  They now run their slices through `nlist::forEachSlice`, so a fixed
  thread count gives the same bits whatever team the runtime starts.

- Two `EAMPot` objects evaluated at the same time on different threads
  no longer race on one process-wide table of element splines.
  Each potential now keeps the tables of the elements it has seen.

- `CP2KPot`, `CPMDPot`, `MOPACPot` and `NWChemPot` skip an engine
  variable that is set but empty, as the other engine-backed potentials
  do.
  Before, the empty path opened the running program and stopped the
  search with a missing-symbol error.

- `D3Pot` and the `"d3"` term of `ExprPot` accept `damping` as `"bj"`,
  `"BJ"`, `"zero"` or `"Zero"` and raise `ValueError` for any other
  string.
  Before, an unknown or misspelled name silently selected Becke-Johnson
  damping.

- `EAMPot` throws `std::invalid_argument` for an atomic number it has no
  parameters for.
  Before, it threw the bare `int` 14324, which callers catching
  `std::exception` could not handle.

- `EAMTable` builds again: its pair walk uses the `Neighbours` cell list
  after the removal of the old pair-list cache header.

- `EMTPot`, `D3Pot` and `D4Pot` rebuild their cached structure when the
  atomic numbers change.
  Before, they did this only when the atom count changed, so a reused
  instance gave the energy and forces of the previous species.

- `IPIPot` no longer fails with "timed out" just before its deadline.
  The wait it passed to `poll()` was the time left rounded down to whole
  milliseconds, so `poll()` could return while time remained; it is now
  rounded up.

- `StillingerWeberSi`, `TersoffSiC` and `IPIPot` raise
  `rgpot.NumericalError` from `__call__` for a non-finite energy, and
  `StillingerWeberSi` and `TersoffSiC` release the interpreter during
  the call, like the other potentials.
  Before, they returned the energy unchecked, and the two kernels held
  the interpreter.

- `UmaPot::caps()` reports `batched`.
  `UmaPot` already evaluated a batch of images in one AOTInductor call,
  but the flag was false, so a caller that trusts it, such as a nudged
  elastic band, evaluated the images one at a time.

- `UmaPot` and `libuma_engine` no longer unpack the `.pt2` into a new
  directory under the system temporary directory, which torch left
  behind whenever a process died and which could fill or be missing on a
  fresh machine.
  The package is unpacked once into
  `<root>/aoti/<sha256 of the .pt2>.model/` and reused by later runs,
  and concurrent processes share one unpack.
  The root is `UmaConfig::cache_dir` or the new `UmaParams.cacheDir`,
  then `RGPOT_CACHE_DIR`, then `$XDG_CACHE_HOME/rgpot`, then
  `.rgpot-cache` in the working directory.

- `XTBPot` and `XTBDlopen` no longer start an SCC from the iterate of
  one that failed, and a session's GFN2 energy on hot Si clusters
  matches a fresh start and standalone xtb.
  The new `XTBConfig::warm_min_gap` (1 eV) uses xtb's own guess when the
  HOMO-LUMO gap is too small for a unique SCC solution.
  The engine exports `rgpot_xtb_set_warm_min_gap` and
  `rgpot_xtb_warm_min_gap`.
  `TBLitePot` drops the wavefunction of a failed SCF.

- `evaluate_batch` raises `TypeError` for a positions or box entry that
  is not a C-contiguous float64 array, as the other entry points do.
  Before, it raised `RuntimeError` with the message `std::bad_cast`.

- `from rgpot import *` now also imports `LammpsPot`, `GromacsPot`,
  `OpenMMPot`, `rgpot_lammps_potential`, `rgpot_gromacs_potential` and
  `rgpot_openmm_potential`.
  They were exported but missing from `rgpot.__all__`.

- `potserv` on the Morse backend applies
  `configure(PotentialConfig.morse)`, rebuilding the pot with the sent
  `de`, `a`, `re` and `cutoff` and keeping its switch width; it used to
  refuse every configure call that was not for NWChem or CPMD.
  A value it cannot use, such as a cutoff below the switch width, gets
  `ok=false` and leaves the pot unchanged.
  On every backend `none` is a no-op, and any other arm the backend does
  not apply, including one from a newer schema, gets `ok=false` with a
  message that names the arm.

- `rgpot-eval LJ`, `Morse` and `SuttonChen` without a `cutoff` option
  use the potential's own default cutoff (15, 9.5 and 40 Angstrom)
  instead of 0.
  `pair_style rgpot` still passes its cutoff.

- `rgpot.ase.RgpotCalculator` gives the right energy, forces and stress
  for a periodic cell narrower than twice the potential's cutoff.
  The kernels take the minimum image of each pair, so before this a
  32-atom argon cell with an 8.5 Angstrom LJ cutoff came out 12% too
  high and a 4-atom EAM-Al cell gave -2.996 instead of -3.356 eV/atom.
  The calculator now tiles such a cell into a supercell that is wide
  enough and returns the energy per copy and the forces of the first
  copy.
  Structures with periodic and non-periodic axes (slabs, wires) are
  evaluated instead of raising `ValueError`.

- `rgpot.h` declares `rgpot_potential_set_peer_capabilities` only when
  `RGPOT_HAS_SCHEMA` is defined, which the meson build does with RPC on.
  Before, the header declared it in builds whose library did not define
  it, so a call failed at link time.

- `rgpot_engine_last_error` in the in-tree Lennard-Jones engine plugin
  writes the terminating NUL into a one-byte buffer, as `engine_c_abi.h`
  promises.
  Before, it left that buffer untouched.

- `rgpot_engine_set_charge_spin` on the Metatomic engine reaches the
  model: each system carries the `rgpot::charge`,
  `rgpot::spin_multiplicity` and `rgpot::spin` data the model declares
  in `requested_inputs`, and the engine reads `charge`, `spin` and
  `attachSystemExtras` from `MetatomicParams`.
  It used to return 0 and drop the values.
  For a value the model does not read, other than charge 0 or
  multiplicity 1, `rgpot_engine_set_charge_spin` now returns 3 and
  `rgpot_engine_create` fails.

- `rgpot_external_potential_energy_forces` returns 1 with its usual
  message for a null `box9` when `n` is 0, instead of reading through
  the null pointer.
  The LAMMPS, GROMACS and OpenMM constructors also reject an infinite
  `u0`, `cutoff` or `psi`, which they used to accept.

- `rgpot_tensor_owned_cpu_f64_2d` returns NULL with an error message for
  NULL data, a negative extent or an overflowing size, and
  `rgpot_tensor_device`, `rgpot_tensor_data` and `rgpot_tensor_shape`
  return a safe value for a NULL tensor.
  Before, these cases were undefined behaviour.

## [3.6.0] - 2026-11-13

[3.6.0]: https://github.com/OmniPotentRPC/rgpot/tree/v3.6.0

### Breaking

- librgpot's ELF soversion is 5.
  LJPot and MorsePot each hold a neighbour list and a cached pair table,
  so a binary built against soversion 4 does not load librgpot.so.5.
  librgpot_mpi stays at soversion 3.
- librgpot's ELF soversion is 6.
  LJ and Morse no longer keep a copied half-list, and ZBL and the open
  Lennard-Jones cluster hold a neighbour list.
  A caller built against soversion 5 does not load librgpot.so.6.
  librgpot_mpi stays at soversion 3.

### Added

- A public neighbour query returns a structure-of-arrays table or a
  chunk of a walk.
  A pair list is a half shell or a full shell.
  A cluster of four is a separate shape and stores every neighbour.
  The C ABI is version 4.
  `rgpot_nlist_request_init` sets the defaults, `bytes` is the struct
  size, and `rgpot_nlist_strerror` names a code.
  Potentials.capnp carries the same columns, and a client streams chunks
  with `cursor` and `next`.
  The tutorial is `docs/orgmode/tutorials/neighbour-list.org`.
  The query walks the same cell list as the classical potentials.
  Fewer than 8192 atoms stay on the calling thread.
  A larger build writes one shared table.
  Several independent systems are one batch, and the C builder keeps a
  Verlet list.
  A positive skin with a strict cut is rejected.
  The classical potentials call that same walk.
  `rgpot.neighbor_list` is the Python half list.
  The Fortran gather uses that list and the builder's skin cache.
  `quantities` selects columns, and `offsets=True` adds a sorted CSR row
  pointer.
- `EAMTablePot` reads embedded atom method (EAM) and Finnis-Sinclair
  potentials from the LAMMPS (Large-scale Atomic/Molecular Massively
  Parallel Simulator) `funcfl`, `setfl` and `eam/fs` tables, with
  energy, forces and an optional Cauchy stress.
  It is built only with `-Dwith_eam_table=true` (CMake
  `RGPOT_WITH_EAM_TABLE`), so the default build is unchanged.
  On `Cu_u3.eam` and `CuNi.eam.alloy` from the LAMMPS `potentials`
  directory it matches `pair_style eam` and `eam/alloy` to 3.1e-12 eV in
  the energy and 1.7e-14 eV/Angstrom in the forces.
- `LJClusterPot` and `MorsePot` are on the Python extension.
  They call the same force implementations as the C++ classes already
  linked into `librgpot`.
- `SuttonChenPot` writes the Cauchy stress when
  `SuttonChenConfig::stress` is set.
  The flag is off by default, and with it clear the energy, forces,
  capabilities and parameter key are unchanged.
  The stress matches LAMMPS `pair_style eam/fs` to 2e-13 eV/Angstrom^3
  on a cubic and a sheared platinum cell.
- `TersoffSiC::forceImpl` writes the Cauchy stress and `caps().stress`
  is true.
  The stress is (1/V) dE/dε in eV/Angstrom^3, with the sign and layout
  of the `LJPot`, `MorsePot` and `ZBLPot` stress.
  The virial sums each folded bond vector times the energy gradient with
  respect to that vector, the many-body form of Thompson, Plimpton and
  Mattson, J. Chem. Phys. 131, 154107 (2009).
  Its digital object identifier (DOI) is 10.1063/1.3245303.
  Energies and forces keep their bits.
  `TersoffPot` keeps `caps().stress` false and writes no stress on any
  call, with `has_stress` 0 and a zero tensor, since its all-silicon
  calls stay on the Fortran gather.
- `XTBPot` and `XTBDlopen` reuse the self-consistent charge (SCC)
  wavefunction across calls.
  Reuse holds while every atom stays within `warm_move` angstroms.
  The default guard is 1 angstrom.
  `reset()` drops that guess, as does a larger move or a new
  composition.
  The next single point then uses xtb's own guess.
  The engine exports `rgpot_xtb_reset` and `rgpot_xtb_set_warm_move`.
  It also exports `rgpot_xtb_warm_move` and
  `rgpot_xtb_reused_wavefunction`.
- Sutton-Chen potential, with the reduced platinum and gold 10-8
  parameters as the default.
  Energy and analytic forces, cluster and periodic evaluation, and
  `rgpot_sutton_chen_energy_forces` for callers that load the library.
- Physical Sutton-Chen presets for platinum and gold, and bulk
  nearest-neighbor and cohesive-well scales.
- Lennard-Jones, Morse, the analytical embedded-atom method (EAM)
  kernel, and the orthorhombic fold select an instruction set at run
  time through Google Highway.
  `RGPOT_SIMD` names the target and `rgpot::simd::chosen_target_name()`
  reports it.
  A product-sum uses a hardware fused multiply-add (FMA) only on a
  target the runtime check selected, and only when that target has the
  instruction.
  SSE2 and SSE4 use it when glibc reports a usable FMA, and a Dekker
  product otherwise.
  Both potentials still take their pairs from the pair-list cache.
- Unshifted Lennard-Jones and Morse cluster calls, binary Lennard-Jones,
  and TIP4P site energy.
  `rgpot_lj_gradient` returns the energy gradient.
  `rgpot_lj_curvature_pairs` evaluates a pair list in one call.
  The `rgpot-pair` crate inlines the same unit trees.
- `rgpot.LJPot` takes the `LJConfig` fields as keywords (`u0`, `cutoff`,
  `psi`, `switch_width`), with the C++ defaults, so Python can turn on
  the C^2 cutoff switch of the Lennard-Jones pair.
  An `ExprPot` child given as a dict reads the same keys for `"lj"` and
  `De`, `a`, `re`, `cutoff` and `switch_width` for `"morse"`.
  Both pair kinds were built from their defaults before.
- `PotType::TersoffSi` is one potential.
  All-silicon configurations, and a null species pointer, use the
  Fortran gather.
  Any other atomic number uses `TersoffSiC`, the 1989 silicon-carbon
  kernel.
  That kernel is from Tersoff, Phys. Rev. B 39, 5566 (1989).
  Its digital object identifier (DOI) is 10.1103/PhysRevB.39.5566.
  The erratum, Phys. Rev. B 41, 3248 (1990), sets carbon B to 346.74 eV
  and d to 4.3484.
  The DOI of the erratum is 10.1103/PhysRevB.41.3248.2.
  Carbon beta is dimensionless and is not printed by the erratum.
  An atomic number other than 6 or 14 is rejected.
  There is no separate silicon-carbide PotType.
  Calling `TersoffSiC` directly always evaluates the 1989 kernel.
  `energy_gradient` evaluates one configuration or a batch of
  configurations that share a cell and a species list.
- `StillingerWeberSi` (`PotType::SWSi1985`) is the Stillinger-Weber
  silicon potential of Stillinger and Weber, Phys. Rev. B 31, 5262
  (1985), doi:10.1103/PhysRevB.31.5262.
  Epsilon is the 3.4723e-12 erg printed in equation (2.9), 2.16723920
  eV.
  `PotType::SWSi` keeps eOn's rescaled epsilon, 2.16826 times
  1.06767385, which is 2.3149945 eV.
  The ratio of the eOn epsilon to the published value is 1.068176741337.
  The factor 1.06767385 is copied from eOn's SWFortran.f90 and is
  untraced.
  `energy_gradient` evaluates one configuration or a batch of
  configurations that share a cell and a species list.
- `TersoffSiC::setHeldSkin` keeps a skinned neighbour list across
  `forceImpl` calls.
  The list is rebuilt when any atom has moved by more than half the skin
  since the last build, or when the cell or the atom count changes.
  The skin is 0 by default, which rebuilds on every call as before.
  `TersoffPot::setHeldSkin` passes the skin to the 1989 kernel;
  all-silicon calls stay on the Fortran gather.
  With a positive skin the instance reports `Reentrancy::PerInstance`.
- MOPACPot dlopens the split
  [mopacc](https://github.com/OmniPotentRPC/mopacc) engine
  (`libmopacc.so`, OpenMOPAC AM1).
  Packed `MopacCParams`, not Cap'n Proto.
  It is not the molecular orbital package (MOPAC) engine of the
  Amsterdam Modeling Suite.
- NWChem remote procedure call (RPC) specifications accept basis,
  theory, exchange-correlation, and self-consistent field (SCF)
  iteration controls for reproducible electronic-structure requests.
- Skala XC is available as C++ ``SkalaPot``: NWChem density functional
  theory (DFT) through ``libnwchemc`` with ``dft.xc`` defaulting to
  ``skala-1.1``.
  Enable with ``-Dwith_skala=true`` (requires ``-Dwith_rpc=true``).
- UmaPot evaluates the OMol universal model for atoms (UMA) from an
  AOTInductor `.pt2` package.
  The package comes from `scripts/export_uma_aoti.py`.
  The vesin library builds the neighbor list, and the compiled graph
  returns energy and forces.
  Charge and spin are per-call tensor inputs.
  One `.pt2` is exported per composition with `merge_mole`.
  Its sidecar records cutoff, neighbors, task, charge, spin, `z_set` and
  label.
  The Baker exporter walks Baker endpoints and deduplicates them by
  `z_set`, charge and spin.
  It is `scripts/export_baker_uma_aoti.py`.
  potserv accepts `Uma:<model.pt2>` and `Uma:<model.pt2>:<task>`.
- `IPIPot` listens for an i-PI driver on a UNIX or TCP socket and
  returns energy, forces, and Cauchy stress.
  `potserv` selects it with `IPI:unix:<path>` or
  `IPI:tcp:<host>:<port>`.
  Meson `with_ipi` defaults to auto and stays off on Windows.
  CMake `RGPOT_WITH_IPI` follows the same rule.
- The Python extension builds a free-threaded cp314t wheel and declares
  that support as beta.
  `rgpot.ase.RgpotCalculator` evaluates potentials in process, including
  a batch of structures.
- `rgpot_eindir_abi_compatible(&stamp)` checks an `eindir_abi_stamp_t`
  against the eindir-core compiled into `librgpot`, so a caller that
  takes an rgpot potential as an eindir objective needs no second link
  to eindir-core.
  It returns what `eindir_core_abi_compatible` returns for the same
  argument.
  Existing calls do not change.

### Developer

- `FastNlistTest` links with `-Dwith_fortran_pots=disabled`.
  Its two water-H cases call the Fortran kernels and are compiled only
  when those kernels are built.
- The regeneration, comparison and timing scripts read the hosts they
  may run on from `RGPOT_REGEN_HOSTS`, a comma-separated list of host
  names that is empty by default.
  Until it lists the current host, `scripts/regen_dftd_goldens.py`,
  `scripts/regen_expr_goldens.py`, `scripts/regen_xckernel_goldens.py`,
  `scripts/gen_libxckernel.sh`,
  `scripts/compare_xckernel_tda_nwchemc.py` and
  `scripts/time_lj_rgpot_vs_pyeonclient.py` refuse to run.
  `DFTD_ALLOW_REGEN=1`, `EXPR_ALLOW_REGEN=1`, `XCKERNEL_ALLOW_REGEN=1`
  and `TIME_LJ_ALLOW=1` still override the check.
  `scripts/gen_libxckernel.sh` records the host in
  `third_party/libxckernel/PIN` as the label from `RGPOT_HOST_LABEL`, or
  as `unspecified` when that is unset or blank, and never writes the
  system hostname there.
- Coverage jobs stay green when the Codecov OpenID Connect (OIDC) token
  request times out.
  A failed OIDC upload is retried with the repository token.
- Pull requests into develop run the CI orchestrator, including the
  build matrix and the Windows leg.
- Documentation builds use a concurrency group per ref, so one pull
  request does not cancel another's build.
  A running GitHub Pages deploy is left to finish.
- The release version is written by hand only in the root `Cargo.toml`.
  The key is `version` in `[workspace.package]`.
  `rgpot-core` and `rgpot-pair` inherit it.
  cogapp regenerates every other copy from it.
  Copies sit in the meson, CMake, pyproject, pixi, towncrier and Sphinx
  files.
  The C header and the rgpot-core README carry copies as well.
  A prek hook fails a pull request when a copy drifts.
- The release-prepare commit check ignores merge commits.
  Cog 7 is invoked as `cog check --ignore-merge-commits <base>..HEAD`.
  Single-parent `merge:` sync commits are ignored as well.
- The shared `py-tsan` workflow sets `vm.mmap_rnd_bits=28`
  unconditionally and runs its container with `seccomp=unconfined`, and
  the instruction set architecture (ISA) self-test compiles its x86
  fixtures at `-march=x86-64`.
  Both follow the `ubuntu-latest` runner's move to Ubuntu 26.04, where
  the sysctl is root-only to read and the default compiler emits vector
  extension (VEX) scalar moves.
- py-tsan takes an optional rust-archive input for the Rust staticlib
  name.
  The default remains librust_core_demo.a.
- The tight Lennard-Jones cluster kernel's AVX2 bodies run only after
  `__builtin_cpu_supports("avx2")`.
  `rgpot/isa-allow.toml` names `lj_avx2`, `eval_quad`, and `acc_quad`
  for Advanced Vector Extensions (AVX) and AVX2.
  It also names `__cpu_indicator_init` from GCC for the `xgetbv` that
  initializer uses.
- Wheel builds keep an unstripped copy of librgpot and _core and reject
  instructions outside the x86-64 and armv8-a baselines, except vendored
  libraries and the FMA kernels behind the glibc check named in
  rgpot/isa-allow.toml.
- `librgpot` on x86-64 now carries one copy of `exact_fma_hw`, the one
  in `exp.cc`.
  The weak copy in `exact_fma_hw.cc` stayed in `.text` with no symbol,
  and the instruction set audit of the Linux x86-64 wheels failed on its
  `vfmadd132sd`.
- The Baker hydrogen cyanide (HCN) UmaPot case skips when
  ``uma-s-1p1-omol-hcn.pt2`` is absent.
  Set ``RGPOT_UMA_REQUIRE_FIXTURES`` to fail that case instead.
- `CppCore/meson.build` follows `meson format` and seven org
  documentation pages follow the semantic line breaks of `snapper`.
- The shared `py-wheels` workflow defaults to the `macos-15` runner
  image instead of `macos-14`, which GitHub Actions retires on
  November 2, 2026.
  The `wheels` workflow and the C++ layout check request `macos-15`
  as well.
- Twelve pinned GitHub Actions in the CI workflows move to releases
  that run on the Node 24 runtime, including checkout 7.0.1,
  setup-python 7.1.0 and upload-artifact 7.0.2.

### Changed

- `EAMPot` keeps a Verlet neighbour list and reuses it while every atom
  moves by at most half of one fifth of the widest cutoff.
  One thread matches the previous energy and forces bit for bit.
  `OMP_NUM_THREADS` above 1 splits atoms across threads.
  A fixed count repeats bit for bit and stays within a relative
  tolerance of 1e-9 of the one-thread result.
- macOS wheels of the main package are built on the `macos-15` runner
  and tagged `macosx_15_0`.
  Homebrew on that image builds libgfortran, libquadmath, OpenBLAS and
  libomp for macOS 15.0, and delocate rejects them under a 14.0 target.
  The bind and rust-core template wheels stay `macosx_14_0`.
- EAM, Lennard-Jones, Sutton-Chen, Stillinger-Weber and Tersoff share
  one neighbour layer.
  It bins atoms into cells, folds the minimum image, and either stores a
  Verlet list or walks the cells without storing pairs.
  The mode follows the memory the operating system reports as available,
  or a caller can select it.
  `OMP_NUM_THREADS` above 1 splits atoms across threads.
  One thread matches the previous energy and forces bit for bit.
  A fixed thread count repeats bit for bit and stays within a relative
  tolerance of 1e-9 of the one-thread result.
- Morse and ZBL take pairs from the shared neighbour layer.
  The separate pair-list cache is gone.
  Stillinger-Weber and Tersoff use that layer for every mode, including
  the default.
  One thread stays within 1e-12 of the previous ordered walk.
  A streaming evaluation keeps a cell binning and a per-atom scratch,
  and several threads walk those cells and build bonds together.
- Sutton-Chen streaming keeps the per-atom density between two cell
  walks.
  It no longer stores a partner index for every pair.
- A short cutoff reuses the shared stream stencil, keeping only cells
  that hold a pair inside the search when a 27-slot entry per cell is
  larger than the pair index.
  Morse and EAM stay on the full stencil.
  `Mode::Auto` still stores when the pair index fits.
- The CPMDPot README states how a force call sets the cell.
  The box of the call is the cell, and a cpmdc release after `0.3.0`
  replaces the `CELL` of the input deck with it.
- The Morse kernel computes `exp` with `rgpot::simd::exp_fixed`, a fixed
  sequence of IEEE operations that returns the same bits on every
  instruction set and C library (within 1.24 ulp), and evaluates each
  pair-list row in vector clones for AVX2 and AVX-512F.
  Morse energies and forces can move in the last bits, and the Morse
  cache key changes.
- The general-cell minimum image is computed in double.
  An ill-conditioned cell is Minkowski-reduced as in Nguyen and Stehle
  (2009), whose DOI is 10.1145/1597036.1597050.
  A reduced basis that is still too ill-conditioned searches the 27
  images with coefficients in {-1, 0, 1}.
  Each product-sum is a hardware fused multiply-add, inlined in the
  fold, when glibc reports FMA.
  Otherwise it is the exactly rounded TwoProduct and TwoSum emulation of
  Boldo and Muller (2011), whose DOI is 10.1109/tc.2010.139.
  That file is compiled with contraction off, so the folded bits do not
  depend on the instruction set.
- The software fused multiply-add used on CPUs without FMA is about five
  times faster in the minimum-image fold, with the same bits as
  `std::fma`.
- The rgpot-core C ABI minor version is now 1 for the new
  `rgpot_eindir_abi_compatible`, in `RGPOT_ABI_VERSION_MINOR`.
  A caller built against minor 0 still passes `rgpot_abi_compatible`.
  The ABI major and the layout revision do not change.
  The `librgpot` SONAME does: this release ships soversion 6.
- The XcKernel stage-B contraction runs on dgemm and dsyr2k, blocked
  over the grid, linked through FlexiBLAS or the generic blas
  dependency.
- The eonviz environment installs readcon 0.17.1 from PyPI.
  That wheel's RoundTrip writer matches the vendored restart golden.
  Its `atoms_to_con` writes one `.con` frame with the RoundTrip writer,
  precision omitted and canonical left off.
  The UMA exporter reads `.con` files with `atoms_from_con`.
  CI compares those writes with the byte-exact restart golden.

### Fixed

- Coincident Lennard-Jones atoms raise NumericalError again.
  Potentials that keep a neighbour scratch ask for one instance per
  thread.
  The layout-stamp job reads `kCxxSoversion` from the header.
  ThreadSanitizer leaves the fused multiply-add (FMA) resolver units
  uninstrumented so the extension can load.
  The macOS wheel links libgfortran from the path the compiler reports.
  A stored neighbour list is kept when free memory cannot be read.
  Flang is not given the MSVC OpenMP compile flag.
  Apple clang is not given `-fopenmp` by the Rust neighbour caller.
  The installed CMake package exports the neighbour archive the
  libraries link.
  The instruction set architecture (ISA) audit names the hardware
  Lennard-Jones span.
  The socket neighbour example is built only where POSIX processes
  exist.
  The stream scaling bench is too: it forks a child per row.
  The Rust neighbour caller on Windows links the COFF archives with
  link.exe and exports the C entry points.
  An empty JSON member in a UMA package is written as an empty object,
  which the AOTInductor loader can parse.
  Hypothesis checks stay serial, because the strategy cache is
  process-wide.
  The 108000-atom neighbour list is allowed 30 seconds.
  The Sutton-Chen cluster relaxation is allowed 120 seconds.
  rustc on that host uses the archive directory as its temp directory.
- The classical engine now links the water potentials when
  `-Dwith_water=true` is set, so `libclassical_engine` no longer fails
  to link.
  It carries the water kernels that its `tip4p`, `spce` and `tip4p_pt`
  names construct.
- `librgpot_cuh2` links the Fortran runtime again.
  The plugin is a C++ link, so the runtime was not inferred and the
  shared library was missing `_gfortran_*`.
- The manylinux wheel jobs retry the gfortran and OpenBLAS install.
  Each retry reads fresh metadata from the primary AlmaLinux repository.
  The mirror list could return the same broken mirror on every attempt.
  The jobs stop when the install fails, instead of failing on a missing
  OpenBLAS.
- `forceBatch` now zeroes the energy and forces of every result slot
  before the kernel runs, as it already did for the variance and stress.
  Before, a reused buffer kept an earlier call's energy and forces when
  the kernel threw, for example when a plugin reported an error or
  returned a non-finite value.
- UmaPot builds positions, the cell and cell offsets in the package
  dtype.
  The package is AOTInductor (PyTorch ahead-of-time compiled).
  It records `pos_dtype`, and `model_dtype` when `pos_dtype` is absent.
  The exporter converts the model with `hydra.double()` for `--dtype
  float64`, and calls `hydra.float()` for `--dtype float32`.
  A package with neither key stays float32.
  UmaPot logs that fallback and both dtypes at load.
- A cell narrower than three cutoffs is every unordered pair, including
  lists past 4096 atoms.
  The client bridge stress test (10k atoms, a 10 A box, Lennard-Jones
  cutoff 15 A) no longer asks linkcell for that stencil.
- Pair-list force updates each use one fused multiply-add, so a cold
  evaluation and a captured pair list agree to the last bit.
- The pair-list loops and the minimum-image fold use the hardware fused
  multiply-add on any CPU with FMA, detected from the `cpuid`
  instruction, where they took the software path on C libraries older
  than glibc 2.33 (Enterprise Linux 8).
  `RGPOT_FMA=software` selects the software path, which returns the same
  bits.
- The aarch64 Linux wheels build again.
  The minimum-image fold takes `target("fma")` on x86-64 only.
  That is the one target where `fma_hw_active()` can select the hardware
  body.
- A ThreadSanitizer build of `librgpot` compiles each
  `RGPOT_SIMD_CLONES` loop once, for the baseline, so
  `import rgpot._core` no longer crashes in the loader.
  The sanitizer had instrumented the clone resolver.
  It called the sanitizer hooks through procedure linkage table slots.
  An `RTLD_NOW` load had not bound those slots yet.
  Every other build keeps the clones, and every clone returns the same
  bits.
- `MinimumImage::fold` now finishes on thin or strongly sheared cells,
  and on long displacements along open directions; those cases search an
  LLL-reduced basis of the periodic rows.
- The H-in-water 26-image scan is one routine.
  Under Apple Clang the Fortran copy and the test copy differed by an
  ulp, so a skewed cell disagreed with the cluster folded from it.
- The linkcell and minimage pair list builds only when cargo and rustc
  1.89 or newer are available, and the MSVC archive is
  `rgpot_fastnlist.lib` with the system libraries that compiler prints.
  On a skewed cell with a longer fractional wrap, WaterH keeps the
  26-image nearest vector.
- When the fast pair-list library is built, it is linked into potserv,
  the examples, and client-only builds.
- A static build links the fast pair list as one relocatable object.
  Apple ar was taking the rust archive on its command line as the
  archive to write, so the macOS umbrella library was never created.
- The fast pair-list archive on macOS is written with ``/usr/bin/ar``.
  The conda archiver pixi exports as ``AR`` produces an archive Apple ld
  rejects.
- An eindir gradient is served from the fused energy result only when
  the positions have the same bits as the energy call.
  Before, a gradient at `-0.0` was served the result computed at `+0.0`,
  because the two compare equal.
  The pair list cache and the half-shell neighbour list now compare the
  box bit for bit as well, so a box that differs only in the sign of a
  zero gets its own build.
  Inputs without a signed zero take the same path as before.
- The MOPAC frontend stays inside `librgpot` and is not installed as its
  own archive.
  Installing that archive copied the core helpers into it and the shared
  library then failed to link.
- The NWChem and Car-Parrinello molecular dynamics (CPMD) frontends stay
  inside ``librgpot`` and are not installed as their own archives.
  Installing those archives could copy the core helpers into them and
  make the shared library fail to link.
  A client-only build, which has no umbrella library, still installs the
  archives.
- The Lennard-Jones cluster kernel no longer crashes at thread exit when
  RocksDB links jemalloc.
  The conda-forge `rocksdb` links it, and `RGPOT_WITH_CACHE=ON` uses
  that build.
  Its aligned scratch buffer comes from `posix_memalign`, which jemalloc
  replaces along with `free`.
- The open Lennard-Jones cluster kernel allocates its scratch buffer
  with ``_aligned_malloc`` on Windows.
  MSVC does not provide ``std::aligned_alloc``.
- The half-shell pair walk builds with the Visual C++ compiler again.
  That compiler gets `__forceinline`, `__declspec(noinline)` and a plain
  test, and GCC and Clang keep their attributes and `__builtin_expect`.
  The pair and Sutton-Chen headers declare their C entry points for
  export, so `cl.exe` accepts the exported definitions.
  The Tersoff neighbour-list bench prints `_MSC_FULL_VER` where the
  compiler has no `__VERSION__`.
- The Meson build compiles `ReducedPairTest` with `-ffp-contract=off`,
  as it already compiles the `ReducedPair` library.
  Apple Clang had fused the expected Morse curvature in the test into an
  `fma`, which missed the library value by one ulp.
- Publishing an AOTInductor package on Windows replaces a file a reader
  still has open.
  The reader keeps the complete package it opened, and the replace no
  longer fails with access denied.
- On Windows, UmaPot can publish a rewritten AOTInductor package while
  another reader or writer has the destination open.
  Access denied and sharing violations during the replace are retried
  until a deadline.
  The published file is one complete payload.
- UmaPot clears PT_GNU_STACK PF_X on the AOTInductor wrapper.so before
  dlopen, so Elja-traced ``.pt2`` packages load on hosts that refuse an
  executable stack.
  ``scripts/export_uma_aoti.py`` applies the same rewrite at mint time.
- The XcKernel golden pins use the stage-B bound `k*eps*|a|^T|b|` (k is
  the grid-block length, at most 256) plus half an ulp of the pin, for
  the C-versus-NumPy and random-grid Fock contractions.
  A fixed 1e-16 cutoff rejected a correct dsyr2k sum.
  CI runs XcKernelGoldens with `-Dwith_xckernel=true`.
- XcKernel links OpenBLAS when FlexiBLAS and a pkg-config file named
  blas are both absent.
  Both manylinux `openblas-devel` and Homebrew publish `openblas.pc`.
- `XcKernel.contract` still treats a missing `dchi` as omitted.
  `scal` stays required.
  The stub is valid Python: passing `None` for `scal` raises
  `TypeError`.
- ``XTBDlopen`` no longer loads a different
  ``libxtb_engine`` when ``engine_path`` is set and that file cannot be
  opened.
  Environment variables and the bare soname are used only when
  ``engine_path`` is empty.
- ``XTBDlopen`` rejects a slash-less ``engine_path``.
  A bare soname would be opened through the loader search path and could
  substitute a different ``libxtb_engine``.
  Pass an absolute or relative path.
  When that path does not open, the error names it and the loader
  diagnostic instead of environment variables that are ignored.
- `IPIPot` sets the UNIX socket address length on macOS.
  The mock driver is joined before the capture it writes is released.
- The `rgpot._core` type stub lists `StillingerWeberSi` and
  `TersoffSiC`, and carries the current `LJPot` and `ExprPot`
  docstrings.
- `ase_helpers` imports with only ASE and numpy again.
  `atoms_from_con` and `atoms_to_con` work without the compiled
  `rgpot._core`.
  `AnnealRunner` and `AnnealOptimizer` are loaded from
  `rgpot.ase.anneal` on first use.
- Metatomic-enabled build-tree executables locate Torch and metatomic
  shared libraries without an ``LD_LIBRARY_PATH`` override.
- `eam_al` stores the eOn r2549 `potinit.f` default-real literals at
  single precision.
  Keeping the extra printed digits moves the four-atom aluminium cluster
  by 1.76e-6 eV.
- `sutton_chen_scales` summed a bcc lattice.
  The nearest-neighbor distance is now the fcc value a / sqrt(2), not
  0.5634 a, and the reduced 10-8 well is 295.4645 epsilon, not 292.9794.
  `fcc_power_sum` in C++ and `rgpot-pair` changes the same way.

### Miscellaneous

- Questions and bug reports now go to the rgpot issue tracker.
  Security reports go to `SECURITY.md`.
  The Code of Conduct names rgoswami@ieee.org for enforcement reports.
  A root `CONTRIBUTING.md` points to the contributing pages.
  Links between pages no longer point at missing reStructuredText files.
  The navigation now lists the thread-safety page and the Fortran
  porting rules.
- The tutorial "Scan a potential energy surface over RPC" evaluates
  Lennard-Jones and Morse pair curves and an H2-over-copper EAM surface
  through `potserv`, and ships the scripts that draw its figures.
  The remote procedure call (RPC) model page now draws the client,
  server and backend layout.

## [3.5.1](https://github.com/OmniPotentRPC/rgpot/tree/v3.5.1) - 2026-10-11

### Breaking

- 3.5.0 changed the C++ layout of PotCaps, LJConfig, LJClusterConfig and
  MorseConfig and should have bumped the librgpot soversion.
  It kept soversion 3, so a caller built against 3.4.0 loaded 3.5.0's
  librgpot.so.3 and crashed.
  3.5.1 installs that layout as librgpot.so.4.
  Callers built against 3.4.0 fail to load it.
  Callers built against 3.5.0 need to relink.
  The C ABI stamp stays at major 1, minor 0, layout revision 1, because no C
  struct changed layout and the C API has no shared library of its own.
  librgpot_mpi stays at soversion 3, because its C ABI did not change.

### Developer

- The rgbind header is the shared nanobind policy for free-threaded extension modules, with reusable py-* workflows for wheels, free-threading, ThreadSanitizer, and typing.
- py-tsan takes an optional rust-archive input for the Rust staticlib name.
  The default remains librust_core_demo.a.

### Changed

- The documentation site uses the v2 logo and favicons, explains the RPC model,
  and lists the surrounding ecosystem in the navigation.

## [3.5.0](https://github.com/OmniPotentRPC/rgpot/tree/v3.5.0) - 2026-10-11

### Breaking

- An rgpot client from this release calls `Potential.getCapabilities` (ordinal `@12`) before the first `calculate` and refuses a server that does not implement the call or whose capabilities are incompatible.
  A server built on an older `Potentials.capnp` needs that method before such a client can use it.

- `PotCaps` gained `groupCollective` and `stress`.
  A C++ caller compiled against the 3.4.0 headers has a different `PotCaps` layout from a `librgpot` built from this release.
  `rgpot_cxx_layout_stamp()` reports the loaded layout.

- `LJConfig`, `LJClusterConfig` and `MorseConfig` gained `switch_width`, so their layout changed.
  The C++ layout stamp covers them.

- `librgpot` no longer links MPI.
  Calculator groups need `librgpot_mpi` installed next to `librgpot`, or `RGPOT_MPI_LIBRARY`.
  A consumer that took MPI flags from the `rgpot` meson or pkg-config dependency now asks for `rgpot-mpi` or for MPI itself.

- The CMake package config moved from `lib/cmake/pot` to `lib/cmake/rgpot`.
  `find_package(rgpot)` finds that directory.
  An `rgpot_DIR` that points at `lib/cmake/pot` does not.

### Deprecated

- The wheel's `libmetatomic_engine.so`, a symlink to `librgpot_metatomic_engine.so`, is kept for one release and removed after it.
  Source builds still install `libmetatomic_engine.so`.

### Added

- A `umaexport` pixi environment pins `fairchem-core` 2.23.0 and `torch` 2.13, so `pixi run -e umaexport python scripts/export_uma_aoti.py --help` is reproducible.
  The `export-uma-aoti` task runs the exporter in it.

- A standalone C plugin ABI (`include/rgpot/plugin.h`, `RGPOT_PLUGIN_API_VERSION` 1) lets a shared library export `rgpot_plugin_init()`.
  The host scans `RGPOT_PLUGIN_PATH`, checks the API version and registers the plugin.
  `RGPOT_CAP_INSTANCE_SAFE` means instances may calculate concurrently; without it the host serializes every instance of that plugin.

- Backends loaded through the potential profile are checked before dispatch.
  `ProfileSession::load` and `rgpot::abi::checked_load` read the backend's `Capabilities` and refuse a different protocol family or major, a lower protocol minor than required, another schema identity, an incompatible eindir bridge ABI, layout or feature set, another DLPack revision, or a missing energy or forces operation.
  Metadata a backend leaves unset is not checked.

- Effective medium theory and the cell-list embedded-atom potential now live in this library.
  EMTConfig.rasmussen selects the Rasmussen parameter set.
  The copper cluster energy remains 5.129167 eV.

- Every backend gets the generic `rgpot_engine_*` table through one adapter over `PotentialBase` (`rgpot/engine/EngineTable.cc`): new engine libraries `classical_engine` (lj, ljcluster, morse, zbl and the Fortran potentials), `mopac_engine`, `nwchem_engine` and `skala_engine`, and the generic table added to the UMA, Metatomic and XTB engines.
  The table adds `rgpot_engine_name`, `rgpot_engine_version` and `rgpot_engine_model_digest` (SHA-256 of the loaded model file), a stated thread-safety policy per reentrancy class, and engine ABI minor 2.
  `UmaParams` and `MetatomicParams` carry the torch thread counts (default 1) and determinism requests, `MetatomicParams` also the rotation and probe options, and the engine applies them at create.
  `potentials-schema` is repinned to the revision that adds them.

- Lennard-Jones, Morse, and ZBL write the Cauchy stress from the same pair force.
  xTB and tblite copy the virial the single point already stores.

- Metatomic evaluates a batch in one model forward and reports that capability.
  The pip wheel packs the engine as `librgpot_metatomic_engine.so`.

- TIP4P, SPC/E and TIP4P on platinum compile when `-Dwith_water=true`.
  The option defaults false.
  Each kernel keeps the 8.5 angstrom cutoff and the 1.0 angstrom switching width.

- The C++ classes that cross the library boundary have a layout stamp: `rgpot::abi::header_stamp()` is a `constexpr` of `rgpot/abi/CxxLayout.hpp`, `rgpot_cxx_layout_stamp()` reports the loaded library, and `rgpot::abi::layout_compatible()` compares them.
  It covers the force structures, `PotCaps`, the pair potential classes and their configs, the `RGPOT_HAS_CACHE` bit and the virtual-function counts of `Potential`, and exists in builds without the Rust core.

- The NWChem multi-rank force host and the collective RPC server own their MPI runtime through `MpiHost`.
  A throw on one rank is reduced across the communicator and rethrown on every rank with that rank's message, so the ranks leave the call together.

- The generic engine table gains optional symbols and a minor version: `rgpot_engine_abi_stamp` / `rgpot_engine_abi_compatible` (major, minor, layout revision), `rgpot_engine_caps`, `rgpot_engine_last_error`, `rgpot_engine_set_num_threads` and `rgpot_engine_set_charge_spin`, all exported by `libuma_engine.so`, which is now linked with `-z nodelete`.
  The `umaenginebld` pixi environment and a CI leg build the engine and run `UmaEngineTest`.
  `potserv` builds when metatomic and rpc are both enabled.

- `CPMDPot::selectOrbitals(key)` names the image or bead the next forces belong to and passes it to `cpmdc_session_select_orbitals`, so an engine that keeps orbitals per key starts each SCF from that image's or bead's own previous orbitals.

- `Capabilities` carries `buildVersion` and `buildRevision` (appended at `@18` and `@19`), filled by the Rust and C++ producers, and `rgpot_source_revision()` returns the source revision of the loaded library.
  The revision comes from `RGPOT_SOURCE_REVISION` or the git checkout and is empty when neither exists.

- `LJConfig`, `LJClusterConfig` and `MorseConfig` take `switch_width`: a positive width replaces the shifted truncation with the C^2 quintic switch `S(t) = 1 - 10t^3 + 15t^4 - 6t^5` over the last `switch_width` Angstrom below the cutoff, so the energy, force and curvature reach zero there.
  The default 0 keeps the shifted truncation and its cache keys.
  `scripts/validation/pair_kernels.py` (sympy) checks every pair kernel's force against `-dE/dx` and the switch's C^2 conditions; `scripts/validation/quintic_switch.sollya` bounds its binary64 evaluation error.

- `PotentialCache` counts computed and served evaluations per handle (`counts()`, `reset_counts()`).
  The same split is available on eindir potentials through `rgpot_potential_eval_counts` and `rgpot_potential_reset_eval_counts`, and from Python as `PotentialCache.counts()` (bound when built with the cache), so two runs under different cache states report the same computed count.

- `XcKernel` serves the `mgga_lapl` family as well as `lda`, `gga` and `mgga_tau`.
  Laplacian contractions need the AO Laplacians in `lapl_chi` and return 2 when those are missing, leaving the output unchanged.

- `pixi run check-eonviz-pins` (`scripts/check_eonviz_pins.sh`) fails when an `eonviz` dependency is a git branch or tag, lacks a full commit `rev`, or when `chemparseplot` or `rgpycrumbs` is not an exact `==` pin.

- `rgpot.h` carries `RGPOT_ABI_VERSION_MAJOR`, `RGPOT_ABI_VERSION_MINOR` and `RGPOT_ABI_LAYOUT_REVISION` with an `rgpot_abi_stamp_t` struct, and the library exports `rgpot_abi_stamp()` and `rgpot_abi_compatible(const rgpot_abi_stamp_t *)`.
  A loaded `librgpot` is compatible with a caller when the major and layout revision are equal and the library minor is at least the caller's.
  The C ABI reference page states when each number changes.

- `rgpot_potential_set_peer_capabilities` installs the peer's `Capabilities` message on a callback handle.
  An incompatible peer makes `rgpot_potential_calculate`, `eindir_objective_eval` and `eindir_objective_grad` fail before the callback runs, with `peer refused: <first mismatch>` in `rgpot_last_error()`.
  A handle that never receives a message keeps the callback contract.

- `rgpot_version()`, `rgpot_version_major()`, `rgpot_version_minor()` and `rgpot_version_patch()` report the version of the loaded library, so a C caller can check it against the `RGPOT_VERSION*` macros of the header it compiled with.
  `rgpot.h` carried 3.3.0 through the 3.4.0 release; `potctl release sync` now stamps its macros and `potctl release assert` fails when they drift.

- `scripts/export_uma_aoti.py` embeds `natoms`, per-element `counts` and the `model`, `torch_version` and `fairchem_version` that produced the package.
  UmaPot checks `natoms` and `counts` against every input, so a package for C2H2 refuses C2H4 although both share `z_set` `[1, 6]`.

- `scripts/export_uma_aoti.py` records which export path produced a package (`export_path`), documents that torch 2.13 with fairchem-core 2.23 always takes the static `make_fx` fallback, and warns when that fallback runs without `--molecular-box`.

- rgpot::BatchEvaluator evaluates many configurations in parallel: it builds one potential per worker from a factory, splits the batch into contiguous ranges and returns results in input order, with no re-entrancy requirement on the potential.
  A worker failure is rethrown after every worker has joined.

### Developer

- A plugin CI leg (potctl ci plugin-test) runs PluginTest, PluginContractTest and ExternalPluginTest on Ubuntu and macOS and is required by the CI gate.

- CI runs the check-eonviz-pins pixi task as a blocking job.

- Coverage generation fails on test failures and lcov errors; only the Codecov upload remains advisory.
  The coverage scripts accept lcov 2.

- Every GitHub Action reference is pinned to a full commit SHA with the version in a trailing comment; ci/gha/check-drift.sh rejects unpinned references.

- Pull request CI runs hold read-only permissions: docs deployment and the Codecov OIDC upload run only on pushes, in their own jobs.

- The MPI contract client runs the capability handshake against the collective RPC server before it configures or evaluates, and a second run requires a protocol major the server does not speak and must stop at the handshake with the reason, for both the serial and the MPI server layouts.

- The Python contract tests cover the recursive `ExprParams` / `PotSpec` tree: the schema names the tree, `PotentialConfig` stays a single-backend union, and a nested `0.5*inner + d3` composition survives a Cap'n Proto round trip.

- The macOS wheel job is required and validates the wheel contents, and publishing waits for it.

- The wheel test command runs `tests/test_rgpot_pip_core.py` against each repaired wheel, so the runpath, import and abi3 checks gate the wheel jobs.

- `CapabilitiesWire` decodes `Capabilities` messages across the complete pre-build-identity `Potentials.capnp` (`CppCore/tests/fixtures`) and requires every member that file declares to exist in the current schema with the same ordinal and type.

- `RPCSchemaSync` regenerates the Cap'n Proto artifacts from both vendored `Potentials.capnp` copies, requires the output and the code generator request to match, and requires the build's generated header and source to equal a fresh regeneration.

- tests/check_wheel_contents.py fails when a built wheel contains the xckernel generator files (catalog.py, cbackend.py) or depends on sympy; both wheel jobs run it.

- The Ubuntu Meson job passes -Dwith_mpi=enabled and runs calculator_group_test under mpirun with four ranks.

### Changed

- Every Fortran potential (SW, EDIP, Lenosky, Tersoff, EAM Al, FeHe, CuH2, Water-H) keeps its neighbour table and error message in a per-instance workspace and reports `Reentrancy::PerInstance` with `perImageInstances`, replacing `ProcessSerial`.
  Separate instances evaluate on separate threads, and a NEB that keeps one instance per image no longer rebuilds one shared table as the images alternate (220 of 240 calls rebuilt it in a 601-atom EAM aluminium NEB, 27.5 percent of its samples).
  The C entry points without a workspace remain for the CuH2 plugin and stay process-serial.
  eOn 3.6.0's EAM_AL needs one instance per image, which this provides ([#95](https://github.com/OmniPotentRPC/rgpot/issues/95)).

- LJPot, LJClusterPot, MorsePot and ZBLPot search pairs with linked cells once the box holds 64 or more cells of one cutoff plus skin, so a fresh search costs O(n) instead of O(n^2), and fully periodic boxes past 20000 atoms keep their cached pair list.
  The cached force loop runs over a sorted CSR half list with the periodic image recorded at build when cutoff + skin is below half the box, and keeps each atom's force in registers across its row.
  `PairListCache::accumulate` takes a radial kernel returning a `PairTerm` (energy and `-V'(r)/r`) and returns the energy.
  `PairListCache::evaluate` and `CachedPairList::forEach` keep their 3.4.0 signatures.

- The EAM aluminium kernel evaluates four exponentials per neighbour entry instead of up to eighteen: the density pass keeps rho(r) and rho'(r) for the force pass, the shifts at the cutoff are computed once per call, and integer powers and the embedding polynomial run without `pow`.
  `scripts/validation/eam_al.py` (sympy) checks the rearranged forms and the gather force against the closed form ([#95](https://github.com/OmniPotentRPC/rgpot/issues/95)).

- The Fortran potentials ask vesin for unsorted neighbour lists and order each atom's row by partner index with two linear counting sorts, which replaces vesin's global sort (10.6 percent of an EAM aluminium NEB) and keeps every force sum in one order whether vesin reused its cached topology or rebuilt it ([#95](https://github.com/OmniPotentRPC/rgpot/issues/95)).

- The Rust `RpcClient` and the `pot_calculate` bridge call `Potential.getCapabilities` before the first `calculate` and refuse a server with incompatible capabilities.
  A server that does not implement the call is refused with an error naming `Potential.getCapabilities` and the upgrade path; the rgpot servers (`potserv`, `rgpot_rpc_server_start`) implement it.
  Servers built on `Potentials.capnp` elsewhere need the call before an rgpot client built from this release can use them.

- The `eonviz` pixi environment pins `chemparseplot` and `rgpycrumbs` to the 1.11.0 releases instead of git branches, and raises the `readcon` floor to 0.14.7.

- `librgpot` no longer links MPI.
  With MPI found, the build adds `librgpot_mpi` (meson dependency and pkg-config name `rgpot-mpi`) holding every MPI call of the calculator groups, and `bindCalculators` or `calculatorsUseMpi` load it with `dlopen` on first use through one C entry point, `rgpot_mpi_api_v1` (`rgpot/calculator_mpi_abi.h`).
  A host that never asks for calculator groups no longer maps libmpi, whose UCX load-time hooks cost 0.4 s per process start.
  `loadCalculatorMpi`, `calculatorMpiLoaded` and `calculatorMpiLoadError` let a host load it from a chosen path (or `RGPOT_MPI_LIBRARY`) and report failures.

- `uma_ase_parity` requires the `.pt2` model path as its argument and prints a usage line without it; the former default was a TorchScript path `UmaPot` rejects.

- Energies and forces of the pair potentials and the Fortran potentials can differ from 3.4.0 in the last bits, because sums now run in sorted CSR and partner order.

### Fixed

- TBLitePot builds against tblite 0.6.0.
  The GFN calculator constructors take an optional config pointer there, and a null config keeps the previous behaviour.
  Older tblite keeps the two-argument call.

- The UmaPot README and `UmaConfig::cutoff` described a `model.pt2.json` sidecar that overrides the config.
  UmaPot reads only the metadata the exporter embeds in the `.pt2`; the docs say so and list the keys it uses.

- The eindir callbacks validate what a force callback returns with `RGPOT_SUCCESS` before reading it: a null forces tensor or data pointer, a wrong dtype, rank or element count, a non-finite force and a non-finite energy are refused with `EINDIR_INVALID_PARAMETER` and a `rgpot_last_error()` message, and the gradient output is left untouched.
  `n_atoms * 3` overflow is refused at construction.

- The result cache no longer deadlocks a bound MPI world.
  A potential that reports the new `PotCaps::groupCollective` (`CPMDPot` does) has its cache hit or miss decided jointly over its calculator through the new `calculatorAgree`, so a geometry one rank holds and another misses is computed on every rank of the calculator instead of leaving the missing rank alone in the engine's collective.

- UmaPot checks the charge, spin, task and composition embedded in an AOTI package against its `UmaConfig` and the input atoms on every force call, and throws `rgpot::UmaContractError` on a mismatch.
  Without the check, a config whose charge or spin differs from the package's evaluates a different potential energy surface and reports no error.

- `CPMDPot::forceImpl` exchanges force errors on its calculator's communicator instead of `MPI_COMM_WORLD`.
  A host that hands calculator groups an uneven batch (two NEB endpoints on four groups) left the computing groups in that world exchange while the idle groups waited in `shareFromCalculator`, and every grouped run hung at its first such batch.
  A failure still prints on every rank of the failing calculator and aborts the whole world.

- `UmaPot` refuses an edge count outside the range recorded in the `.pt2` metadata (`nedges_min`, `nedges_max`) with an `UmaContractError` on field `nedges`, instead of passing the compiled graph a shape it was not exported for.
  `scripts/export_uma_aoti.py` writes both keys: 1 to 65536 for a dynamic export, the traced edge count for a static one.

- `scripts/export_baker_uma_aoti.py` deduplicates Baker systems by exact composition, charge and spin.
  Keying by `z_set` merged systems that share an element set but not their atom counts, and their packages cannot evaluate each other.

- In 3.4.0 a periodic pair list assumed an orthogonal box, and LJPot warned that the box is assumed to be orthogonal.
  Periodic images now use the full-lattice minimum image, and an ill-conditioned cell throws.
  Energies and forces on a non-orthogonal cell change.

- A cache hit restores the stored variance and stress.
  Records keep the energy and force prefix and append versioned variance and stress metadata.
  A record from an older build still reads, and a potential that reports stress treats a record without that metadata as a miss.

- The CMake install exports the neighbor-list library, xxHash, the Fortran runtime, the public Fortran potentials and their kernels, the CuH2 plugin entry points, the calculator schemas, the engine ABI headers, and the optional MPI calculator companion.
  A static core is usable from a shared consumer, and static kernels stay hidden at that link.
  MPI host targets are enabled only when mpi.h compiles.
  On Apple the Fortran Coulomb plugin links its C driver.

- Core potentials compile with their feature definitions.
  Engine plugins link the core archive on demand.
  potserv takes the torch dependencies when metatomic and rpc are both enabled.

- The UMA exporter rewrites a strided force-slice write so a snapshot of the base stays independent of later writes.
  Molecular inputs are centred before they are converted to the model's precision.
  A rewritten AOTI archive is published without clobbering a package another reader still has open.
  An export candidate is kept only when its tensors run and match the reference.
  Package publication and reads retry a transient Windows sharing error.

- An LDA XcKernel accepts basis values that have no gradients.
  The Python bindings check XC operand arrays and keep them alive for the call.

- The CPMD engine is given the calculator group's communicator.
  Communicator adopters are retained and deduplicated across binding orders.
  A refused calculator binding stops the next CPMD force evaluation.
  Registered CPMD callbacks stay on one process-lifetime engine handle.
  NWChem keeps one engine handle for each resolved function table for the life of the process.

- Pair sums of one geometry now agree bit for bit between a cold scan and a cached CSR list.
  In 3.4.0 those two orders could differ in the last bits, and repeated eOn NEB runs differed at 1e-12 eV.

## [3.4.0](https://github.com/OmniPotentRPC/rgpot/tree/3.4.0) - 2026-09-30

### Added

- Calculator groups report how many there are and broadcast a result from one group to every rank; the meson build takes MPI through `-Dwith_mpi=enabled`.

### Developer

- The twelve-rank refused calculator split calls finalizeMpiAtExit, so mpirun can exit 0 after MPI_Init.
- The two-rank calculator share that matches calls finalizeMpiAtExit, so mpirun can exit 0 after MPI_Init.
- mpirun -n 4 calculator_group_test checks that rank 3 holds the buffer rank 0 filled. An out-of-range calculator owner aborts every rank.

### Changed

- Read the engine documentation at https://cpmdc.rgoswami.me.

### Fixed

- A failed CPMD engine call sets an abort flag, and the exit handler calls `MPI_Abort` instead of a collective `MPI_Finalize` that peers stuck in a broadcast would never reach.
- A force that throws on one rank is printed on every rank in that call, then MPI_Abort runs on MPI_COMM_WORLD. MPI_Finalize at exit runs only when this library called MPI_Init.
- A release tag no longer fails when `publish.yml` has already uploaded `rgpot-core` to crates.io: `release.yml` skips `cargo publish` for a version that exists, so its GitHub release job still runs.
- A shareFromCalculator call that cannot enter the broadcast aborts every bound rank after those ranks print the error.
- The HTML sitemap lists each page at the site root.
- The Morse cutoff-shift test compares `energyShift()` with an absolute tolerance at the operands' rounding scale instead of a 1e-14 relative pin on a cancelled difference, so the suite passes on aarch64 (EESSI 2026.06, Neoverse-N2), where `exp` and multiply-add contraction differ from x86 by an ulp.
- The calculator namespace page names the rank that fills ForceOut and the first rank of a share broadcast.
- The one-shot CPMD gradient path keeps a nine-component stress tensor when the engine computed one.
- `CPMDPot::setParams` with byte-identical parameters keeps the engine session and its stored wavefunction; the engine loaded by `bindCalculators` stays loaded for the process; `paramsKey` fingerprints the CPMD parameters for the result cache.
- calculatorsUseMpi follows MPI_Initialized. A refused calculator split reports the MPI_COMM_WORLD size and a negative group index.


## [3.3.0](https://github.com/OmniPotentRPC/rgpot/tree/3.3.0) - 2026-09-28

### Added

- CPMD ranks can be split into one calculator group per image.
- The new `rgpot_eindir_abi_stamp()` returns the eindir-core ABI stamp of the objective base in `rgpot_potential_t`. The generated `rgpot.h` stays valid C, with the fused evaluation cache behind an opaque pointer.

### Changed

- Always compile the xTB dlopen frontend (`XTBDlopen`) without `-Dwith_xtb`, matching Metatomic/NWChem: hosts ship without NEEDED `libxtb` and load `libxtb_engine.so` via `RGPOT_XTB_ENGINE` at runtime. Linked `XTBPot` and the engine plugin still require `with_xtb`. Config types moved to `XTBConfig.hpp` (no `xtb.h`). ([#53](https://github.com/OmniPotentRPC/rgpot/issues/53))

### Fixed

- `CPMDPot` copies `PotentialResult.stress` into `ForceOut` when the engine returns nine components.
- RocksDB 11 builds of the cache work: `PotentialCache` opens the database through the `std::unique_ptr` `DB::Open` overload on RocksDB 10.4 and later. ([#54](https://github.com/OmniPotentRPC/rgpot/issues/54))
- A `-Dwith_cache=true` build lists `-DRGPOT_HAS_CACHE=TRUE` and `rocksdb` in `rgpot.pc`, so consumers of the cache API find its flags and headers through pkg-config. ([#55](https://github.com/OmniPotentRPC/rgpot/issues/55))
- Configuring rgpot, on its own or as a Meson subproject, no longer warns that `add_languages` is missing `native:`.
- Linux wheels import without a system OpenBLAS. auditwheel repaired the real librgpot library but not the librgpot.so.3 and librgpot.so copies that _core loads, so those still named libopenblas.so.0.
- Linux wheels vendor the OpenMP runtime: `import rgpot` no longer needs a system `libgomp.so.1`.
- rgpot configures offline: the `potentials-schema` subproject is optional, and the two schema-sync tests run only when it is available, so `--wrap-mode=nodownload` builds no longer fail at `meson setup`.

### Miscellaneous

- OIDC trusted publishing on tag v* and workflow_dispatch. ([#76](https://github.com/OmniPotentRPC/rgpot/issues/76))


## [3.2.0](https://github.com/OmniPotentRPC/rgpot/tree/3.2.0) - 2026-09-13

### Added

- MOPACPot dlopens the split [mopacc](https://github.com/OmniPotentRPC/mopacc)
  engine (`libmopacc.so`, OpenMOPAC). Packed `MopacCParams`.
  Default model is AM1.
- NWChem RPC specifications accept basis, theory, exchange-correlation, and SCF
  iteration controls for reproducible electronic-structure requests.
- Skala XC is available as C++ ``SkalaPot``: NWChem DFT through ``libnwchemc``
  with ``dft.xc`` defaulting to ``skala-1.1``. Enable with ``-Dwith_skala=true``
  (requires ``-Dwith_rpc=true``).
- UmaPot evaluates UMA / OMol from an AOTInductor ``.pt2`` (``scripts/export_uma_aoti.py``): vesin builds the neighbor list, the compiled graph returns energy and forces. Charge and spin are per-call tensor inputs. One ``.pt2`` is exported per composition (``merge_mole``); the sidecar records cutoff, neighbors, task, charge, spin, ``z_set``, and label. ``scripts/export_baker_uma_aoti.py`` walks Baker endpoints and deduplicates by ``(z_set, charge, spin)``. potserv accepts ``Uma:<model.pt2>`` and ``Uma:<model.pt2>:<task>``.

### Fixed

- CI pins snapper-fmt to v0.10.0. Latest 0.11.2 panics on UTF-8 in org files.
- Metatomic-enabled build-tree executables locate Torch and metatomic shared
  libraries without an ``LD_LIBRARY_PATH`` override.
- UmaPot clears PT_GNU_STACK PF_X on AOTI wrapper.so before dlopen, so Elja-traced ``.pt2`` packages load on hosts that refuse an executable stack. ``scripts/export_uma_aoti.py`` applies the same rewrite at mint time.


## [3.1.2](https://github.com/OmniPotentRPC/rgpot/tree/3.1.2) - 2026-09-04

### Fixed

- D3/D4 in the wheel are built from pinned meson wraps (static s-dftd3 / dftd4), not a conda prefix. The dftd4 wrap uses OpenBLAS as its LAPACK. ``pip install`` no longer needs vendored ``libblas`` from micromamba.


## [3.1.1](https://github.com/OmniPotentRPC/rgpot/tree/3.1.1) - 2026-09-04

### Added

- The PyPI wheel compiles ExprPot, D3Pot, D4Pot, and XcKernel and vendors ``s-dftd3`` / ``dftd4``. ``import rgpot`` exposes those classes (``has_expr`` / ``has_dftd3`` / ``has_dftd4`` / ``has_xckernel``).


## [3.1.0](https://github.com/OmniPotentRPC/rgpot/tree/3.1.0) - 2026-09-04

### Added

- A wheels workflow builds and publishes the Python package. rgpot had none:
  2.5.4 reached PyPI hand-built from a branch, with no tag and no
  main-branch source behind it, which is how the version surfaces drifted
  apart in the first place. A `v*` tag now builds manylinux and macOS
  wheels plus an sdist with cibuildwheel and publishes them; a manual run
  defaults to TestPyPI so an upload can be rehearsed before the
  irreversible one.

  The build step refuses a wheel whose `librgpot` carries fewer than the
  eight Fortran `bind(c)` entry points, so the potentials cannot silently
  fall out of the artifact again.
- Add `scripts/time_lj_rgpot_vs_pyeonclient.py` to time in-process `LJPot`
  against pyeonclient `Matter` on the ExprPotTest two-atom LJ fixture.
- D3Pot wraps the s-dftd3 ISO C API (``-Dwith_dftd3=true``, pixi env ``dftd3``). Config selects BJ/zero damping, a functional key, and an explicit ATM three-body flag. In-process Potential summand; no ``PotentialConfig.d3`` arm.
- D4Pot wraps the dftd4 ISO C API (``-Dwith_dftd4=true``, pixi env ``dftd4``). Config selects a functional key, total charge, and an explicit ATM/many-body flag. In-process Potential summand; no ``PotentialConfig.d4`` arm.
- ExprPot algebra goldens pin identity ``lj``, ``0.5*lj + morse``, ``2*lj - lj``, and ``0.5*(lj+morse)`` under ``CppCore/tests/data/expr/`` (energy 1e-14 eV, forces 1e-12 eV/A). Missing fixtures fail closed. Optional ``0.5*lj + d3`` when ``-Dwith_dftd3=true``.
- ExprPot compiles a Lepton energy string over named ``unique_ptr<PotentialBase>`` children (``-Dwith_expr=true``). Construct-time name checks fail closed. No PotentialConfig arm, no XcKernel term.
- ExprPot forces use analytic Lepton ``differentiate(name)`` weights compiled at construct: ``F = sum_i (df/dE_i) F_i``. No finite difference, no muParser.
- ExprPot wraps through ``PotentialHandle::from_impl`` / ``rgpot_potential_new_eindir`` as one eindir objective. A single ``rgpot_potential_calculate`` returns ``0.5*lj + morse``.
- Golden masters pin D3 BJ ATM on/off and D4 PBE energy and forces against s-dftd3 / dftd4 on a water-octamer cube (``meson test --suite dftd``).
- In-process `XcKernel` contracts Libxc derivative arrays into AO XC Fock and fxc matrices via the libxckernel C ABI (`-Dwith_xckernel=true`, off by default). This is not a `Potential` and has no `PotentialConfig` arm.
- Optional conda-forge s-dftd3 / dftd4 gates (``-Dwith_dftd3`` / ``-Dwith_dftd4``, pixi features ``dftd3`` / ``dftd4``). Default workspace and wheel stay free of those libraries.
- Optional vendored OpenMM Lepton gate (``-Dwith_expr``, ``third_party/lepton``). Default workspace and wheel stay free of Lepton object code.
- RPC/config can name a recursive ``ExprParams`` / ``PotSpec`` tree (for example ``0.5*lj + d3``). ``PotentialConfig`` stays a single-backend configure union. The C++ ``ExprPot`` constructor remains the first-slice API.
- TDA pin operator (H2O/sto-3g singlet roots) is also compared to libnwchemc ``task tddft`` / TDA via ``scripts/compare_xckernel_tda_nwchemc.py``. Exclusive sigma contraction stays 1e-17; the nwchemc root residual is the measured engine value (1e-6).
- `Potential::forceBatch` evaluates several independent systems in one call.
  The default `forceBatchImpl` loops over `forceImpl`, so every existing kernel answers a batch without an override.
  When the result cache is on, hits are served per system and only the misses reach the kernel as one compact batch.
- `XcKernel` assembles TDA and RPA sigma over the singlet `st_o2_p` C kernels plus host Coulomb, matching the PySCF `lib.einsum` contraction path and `eval_rho` (`ao @ dm`). The meson golden and live `TDA.gen_vind` / `gen_tdhf_operation` both meet exclusive 1e-17 against those committed sigma pins.
- `rgpot-core` can load any minimum-profile backend by prefix, keep one persistent session, and exchange the shared `PotentialConfig`, `ForceInput`, and `PotentialResult` carriers in-process.

### Changed

- Docs credit Antics and load only `antics.js`; Umami tracker and label are gone.
- The 1 MiB large-file CI gate now allowlists ``CppCore/tests/data/``, ``third_party/``, and ``pixi.lock`` (XcKernel goldens and the lockfile). Other tracked files still fail at 1 MiB.

### Fixed

- D3Pot now passes a sigma buffer to ``dftd3_get_dispersion`` so s-dftd3 fills the gradient (forces were previously left at zero).
- GGA TDA/RPA ``applyFxc`` keeps the generated 7-term monomials on the host long-double evaluator so C matches live ``gen_vind`` at exclusive 1e-17.
- LDA TDA/RPA fxc forms `wv = w * rho * (v2rho2_0 + v2rho2_1)` and tiles
  stage B at PySCF `BLKSIZE` (128). `--tda-rpa` pins host J from the
  `get_j` call inside `gen_vind` / `gen_tdhf_operation`.
- The cp312 wheel now carries the `abi3` tag: `tool.meson-python.limited-api`
  declares the stable-ABI build to meson-python, so one cp312-abi3 wheel
  installs on every CPython >= 3.12 instead of 3.12 alone.
- The metatomic torch dependency links on macOS. Its link arguments carried
  `-rpath-link` and `--as-needed`, which are GNU ld only: Apple's linker
  rejects them outright, so any Darwin build with `-Dwith_metatomic=true`
  died with "unknown options" while producing `librgpot.3.dylib`. Those
  flags now apply on GNU linkers only; Apple's resolves the dylibs from the
  library path and their install names.
- Wheel publish falls back to the `pypi-rgpot` / `testpypi-rgpot`
  `PYPI_TOKEN` until a matching trusted publisher is registered.

  cibuildwheel `build` must be a space-separated string so cp310, cp311,
  and the cp312 abi3 wheel all ship (a TOML array only built cp310).
- `--pyscf` Fock live-PySCF regen now fails above exclusive 1e-15 after long-double stage A/B (mGGA-tau float64 einsum was 1.50e-15).
- `--tda-rpa` keeps committed TDA/RPA host-J pins. Live `get_j` jitters
  about 1e-15 on this case and is reported, not written over the Coulomb
  pin the exclusive 1e-17 C compare uses.
- `--tda-rpa` now writes same-SCF extras and MANIFEST before the exclusive 1e-17 live-vs-committed gate; sigma/J pins are the MO-replay `gen_vind` / `get_j` (a fresh RKS is not bit-stable at that scale).
- `--tda-rpa` regen pins PySCF to one OpenMP thread and replays committed MOs so live `gen_vind` / `gen_tdhf_operation` can meet exclusive 1e-17 against the sigma pins.


## [3.0.2](https://github.com/OmniPotentRPC/rgpot/tree/3.0.2) - 2026-08-08

### Fixed

- Windows MSVC builds Cap'n Proto with lean windows.h defines, and links Flang pot kernels through MSVC lib.exe archives so llvm-ar no longer triggers LNK1107. ([#61](https://github.com/OmniPotentRPC/rgpot/issues/61))
- MetatomicPot no longer fails to load exported PET-MAD (and similar) models on metatensor-torch 0.10.3: scripted modules without `_mts_buffer_names` no longer trip the mixed-dict `.to()` walk.
- Python 3.x wheels import without torch: fat MetatomicPot is no longer
  `link_whole`'d into `librgpot` when `with_python` (engine stays dlopen-only),
  and the repair step ships `librgpot.so.3` (mesonpy zip dropped the soname link)
  plus a RUNPATH into site-packages torch/metatensor/vesin.


## [3.0.1](https://github.com/OmniPotentRPC/rgpot/tree/3.0.1) - 2026-07-26

### Fixed

- The Python wheel builds the Fortran 2018 potentials. `[tool.meson-python.args]`
  never passed `-Dwith_fortran_pots`, so a wheel carried a `librgpot` with no
  Fortran-backed potential in it -- SW, EDIP, Lenosky, Tersoff, EAM-Al, FeHe,
  CuH2, and TIP4P-H were all missing from the one artifact most users install.
  The cibuildwheel configuration installs a Fortran compiler in the build
  image to match, and the repair step vendors `libgfortran` as it would any
  other dependency.


## [3.0.0](https://github.com/OmniPotentRPC/rgpot/tree/3.0.0) - 2026-07-26

### Added

- Pot-hosting groundwork for absorbing eOn's potential kernels:
  `PotCaps` capability descriptors (`PotentialBase::caps()`) replace
  caller-side thread-safety lists; `paramsKey()` parameter fingerprints
  (FNV-1a + kernel-version salt) join the result-cache key so instances
  with different parameters never share entries; `LJConfig` establishes
  the plain-aggregate parameter convention; a `pot_bench` Catch2
  microbenchmark harness gates future pot migrations; `potctl` locksteps
  `pyproject.toml`. ([#57](https://github.com/OmniPotentRPC/rgpot/issues/57))
- Eight potentials absorbed from eOn arrive as Fortran 2018 kernels under
  `CppCore/rgpot/fortran/`: Stillinger-Weber, EDIP, Lenosky, Tersoff, EAM
  aluminium, FeHe, CuH2, and TIP4P-H. Each is a rewrite rather than a
  wrapper -- modules with `implicit none`, derived-type parameters in place
  of COMMON blocks, kinds from `iso_fortran_env` asserted against the C
  types at compile time, `intent` on every argument, `pure` kernels,
  structured control flow, and status returns instead of `stop`.

  Neighbours come from vesin through `rgpot_neighbors`, a CSR full-list
  table carrying pair vectors and distances, and the vendored vesin Fortran
  interface gained the C API's Verlet `skin` option so the kernels keep the
  list caching their predecessors had. Pair sums are restated as gathers:
  each atom accumulates the whole force acting on it and writes one column,
  so the atom loops run under `do concurrent`. Bond-order and embedding
  kernels reach the same form in three passes, since the environment
  derivative is not known until its sum completes.

  Each kernel carries a Fortran-only test (translation invariance, zero net
  force, analytic forces against central differences of the energy) and is
  pinned in `FortranPotsTest` to the reference energies of the kernels it
  replaces. The kernel archives link under `--exclude-libs`, so no Fortran
  symbol reaches the library interface; a test fails the suite if one ever
  does.
- Three classical potentials absorbed from eOn, always built: `MorsePot`
  (pairwise Morse with the shifted cutoff, platinum defaults through
  `MorseConfig`), `LJClusterPot` (12-6 Lennard-Jones on free boundaries,
  `LJClusterConfig`), and `ZBLPot` (screened nuclear repulsion with the
  LAMMPS switching function, `ZBLConfig{cut_inner, cut_global}`, which
  rejects a cutoff pair outside `0 < cut_inner < cut_global`). All three
  take their pairs from the shared `PairListCache`, carry `paramsKey()`
  fingerprints, and appear in `pot_bench`. `PotType` gains `Morse`,
  `LJCluster` and `ZBL`.

### Changed

- `LJPot` pair search goes through `rgpot::nlist::PairListCache` (header-only
  port of eOn's PairListCache): Verlet-skin cached candidate lists in a
  process-global proximity-matched pool, lazy list capture so one-shot
  evaluations run a single fused scan, exact per-call MIC folding, and the
  inverse-r2 pair math instead of `pow()`/`sqrt()`. Boxes where MIC caching
  is unsound keep the historical per-call scan. ([#56](https://github.com/OmniPotentRPC/rgpot/issues/56))
- One versioned `librgpot.so.3` umbrella: every pot builds as a PIC static
  convenience library folded in via `link_whole`; only dlopen'd engine
  plugins ship as separate shared objects. `rgpot.pc` always installs when
  rgpot is the top-level project, exporting the umbrella with
  `RGPOT_HAS_*` feature cflags. vesin resolves once at top level
  (installed >=0.6 preferred, vendored `third_party/vesin` fallback with
  the Fortran interface), and the new `with_fortran_pots` feature
  decouples Fortran-backed potentials from the RPC role options.
  Statistics counters in `registry<T>` are atomic, and public headers no
  longer inject `AtomMatrix` at namespace scope. CI gains the first
  Windows leg and an offline `--wrap-mode=nodownload` rehearsal of the
  release-tarball build. ([#57](https://github.com/OmniPotentRPC/rgpot/issues/57))
- The Cap'n Proto schema ships inside `librgpot` instead of a separate
  `libptlrpc.so`. Consumers that bundle the umbrella alone -- pip wheels
  especially -- no longer need a second shared object beside it, and
  `rgpot.pc` correspondingly stops listing `-lptlrpc`; the symbols are
  unchanged and still exported from `librgpot`. `potserv` and
  `pot_client_bridge` take the generated translation unit from the shared
  archive rather than compiling their own copy.
- The CuH2 potential is the in-tree Fortran 2018 kernel
  (`CppCore/rgpot/fortran/rgpot_cuh2.f90`), reached through
  `rgpot::fortranpots::CuH2Pot` from `rgpot/fortran/FortranPots.hpp`. It
  matches the energy and forces of the kernel it replaces to 1e-6 on the
  geometry `CuH2PotTest` pins. `rgpot::CuH2Pot` and
  `rgpot/CuH2/CuH2Pot.hpp` are gone, and with them the `fortcuh2`
  subproject and its wrap-git entry, so a source build needs no download
  for the Fortran pots and the offline `--wrap-mode=nodownload` CI leg
  covers them. `rgpot/CuH2/cuh2Utils.hpp` keeps the slab-geometry helpers
  unchanged. In-tree sources gate the Fortran pots on
  `RGPOT_HAS_FORTRAN_POTS`, the same cflag `rgpot.pc` exports to
  consumers.

### Fixed

- `potctl`'s lockstep check now covers `pyproject.toml`, so the Python
  package version cannot drift away from meson, CMake, cargo, pixi, and
  towncrier again. It had: 2.5.4 went to PyPI as a multi-ABI wheel set
  (torch 2.7 through 2.13) while every other surface stayed at 2.5.3. All
  six now read the same version, and the check fails the build when they
  do not. ([#52](https://github.com/OmniPotentRPC/rgpot/issues/52))
- Four link-layout defects in the umbrella build, each of which only
  surfaced in a profile that combines the RPC stack with the Fortran
  potentials:

  - Every convenience archive carried its own copy of the vendored vesin
    translation unit, because meson copies a static library's objects into
    each static library that links it, so the umbrella saw multiple
    definitions of vesin's thread-local error state. The archives take a
    headers-only view now and the objects enter `librgpot` -- and each leaf
    binary -- exactly once.
  - `librgpot` linked without the Fortran runtime: it is a C++ target that
    takes the kernels through `link_whole`, which leaves meson no Fortran
    source to infer the runtime from. It is now named per compiler id.
  - `ptlrpc_dep` exported the generated capnp `.cpp` as a dependency source,
    compiling a second copy of that translation unit into every consumer.
    Only the generated header propagates.
  - The umbrella exported the RPC entry points only when something inside
    `librgpot` happened to reference them; they are folded in with
    `link_whole` now, since it is consumers that call them.
- The RPC integration test compares the CuH2 energy and forces at the same
  1e-6 tolerance `CuH2PotTest` uses, instead of asserting exact float
  equality against values from the kernel the Fortran 2018 rewrite replaced.
  Its error path also kills the server before draining its output: reading a
  live server's stderr blocks until that process exits, so an assertion
  failure did not report itself, it sat until the six-hour CI timeout. The
  handler now prints the exception type and traceback as well, since
  `AssertionError` stringifies to nothing.
- The strict-determinism metatomic test no longer requires two matched force
  calls to agree bit for bit, because the provider does not offer that.
  Instrumenting the path shows rgpot handing the model identical inputs --
  positions, cell, and neighbour list all hash the same on every SO3 pass of
  both calls -- and getting energies back that differ by one to two ulp on a
  call chosen at random. It reproduces with `OMP_NUM_THREADS=1`, with the
  TorchScript profiling executor frozen, and with the pair vectors copied
  into torch-owned storage, so it is neither thread-count reassociation, nor
  graph specialization, nor alignment of the buffers rgpot passes in. The
  test now bounds the energy at four ulp and the force components at an
  absolute floor scaled to the largest component, which still fails on any
  drift larger than the provider's own noise.
- The vendored vesin Fortran interface builds under LLVM Flang on Windows.
  It bound `vesin_neighbors` directly, which takes the device and options
  structs by value; Flang cannot lower a by-value `BIND(C)` derived type on
  the `x86_64-pc-windows-msvc` target and aborts the compiler outright
  ("not yet implemented: passing VALUE BIND(C) derived type for this
  target"), so no Fortran consumer of vesin could be built there at all.
  vesin gains `vesin_neighbors_byref`, which takes both structs through
  pointers and forwards them by value, and the Fortran module binds to that.
  Marked as an upstream candidate alongside the other local vesin patches.


## [2.5.3](https://github.com/OmniPotentRPC/rgpot/tree/2.5.3) - 2026-07-20

### Fixed

- Fix ``-Dwith_rpc_client_only=true -Dwith_tests=true``: use ``pot_bridge_dep``
  (was undefined ``rgpot_bridge_dep``), link ``units.cc`` into unit tests when
  ``rgpot_core`` is not built, and soft-skip bridge stress cases when potserv is
  not running (lazy client connect). Catch2 configure/link/test succeed for the
  client frontend product without a live server.


## [2.5.2](https://github.com/OmniPotentRPC/rgpot/tree/2.5.2) - 2026-07-18

### Fixed

- Rebuild Metatomic engines against vesin 0.6+ ``VesinOptions`` (``skin`` /
  ``n_threads``) so ``vesin_neighbors`` no longer fails with a bare error when
  runtime libvesin is 0.6 and the engine was built for 0.5.


## [2.5.0](https://github.com/OmniPotentRPC/rgpot/tree/2.5.0) - 2026-07-17

### Added

- Multi-language Codecov coverage (Rust, C++, Python, Fortran) with OIDC uploads. ([#47](https://github.com/OmniPotentRPC/rgpot/issues/47))
- Install headers and ``rgpot.pc`` (``nwchempot`` / ``cpmdpot`` / ``ptlrpc``; no
  torch or xTB at link time) so eOn and other hosts can prefer
  ``dependency('rgpot')`` over the Meson subproject wrap. Engines stay runtime
  dlopen. See ``docs/eon_pkgconfig.md``.
- Keep Metatomic **dlopen** product on pip: portable `libmetatomic_engine.so` plugin (no eonclib) + `evaluate_metatomic` frontend (not LJ-only wheels).
- Metatomic dual path: linked ``MetatomicPot`` (fast) and ``MetatomicDlopen``
  frontend plus optional ``libmetatomic_engine.so`` C ABI (slow/plugin path).
- Metatomic engines are built for each supported torch major
  (``rgpot/lib/torch-X.Y/libmetatomic_engine.so``) and selected at runtime from
  the installed torch version — same multi-ABI model as metatomic-torch itself.
  The pip product covers **torch 2.7 and newer**; earlier majors are out of scope.
- MetatomicConfig gains an explicit ``torch_determinism`` policy
  (``TorchDeterminismPolicy::Fast`` default, ``Strict`` opt-in). Strict mode
  enables deterministic LibTorch algorithms, math-only scaled-dot-product
  attention (flash / memory-efficient / cuDNN SDP disabled), deterministic
  cuDNN with benchmarking off, deterministic fill of uninitialized memory, and
  disables TF32 for cuBLAS and cuDNN. These flags are process-global via
  ``at::globalContext()``; Fast never mutates them. CUDA hosts still need
  ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` (or ``:16:8``) before the first cuBLAS
  call for bit-stable matmuls.
- Portable Metatomic engine wheels: `$ORIGIN` RUNPATH to site-packages torch/metatomic
  (single build-time torch major; multi-ABI package dirs cannot be listed oldest-first).
  `scripts/rgpot_build_wheel.sh` always runs RPATH repair after `python -m build`.
- Python bindings use **nanobind** with **stable ABI** (abi3 / Py_LIMITED_API 3.12)
  when built on Python >= 3.12 (same policy as pyeonclient). Metatomic engines are
  packed multi-ABI under ``rgpot/lib/torch-X.Y/`` and selected from the installed
  torch major at runtime. Supported libtorch majors start at **2.7** (engines for
  2.7–2.13 ship in the manylinux wheel); torch 2.6 and older are not bundled.
- Python package `rgpot` is pip-installable: core Lennard-Jones bindings via
  meson-python wheels (`import rgpot; rgpot.evaluate_lj(...)`), plus optional
  Metatomic multi-ABI engines for **torch 2.7+**.
- xTB dual backends: keep linked ``XTBPot`` and add ``XTBDlopen`` +
  ``libxtb_engine.so`` C ABI plugin (same pattern as metatomic engine).

### Fixed

- Metatomic C ABI engines soft-release the GIL so pyeonclient Job.run can evaluate forces under torch autograd without a fat metatomic link.
- SoftGilRelease in the metatomic C ABI only calls PyEval_SaveThread when PyGILState_Check is true, so nested GIL release from pyeonclient Job.run is safe.


## [2.2.1](https://github.com/OmniPotentRPC/rgpot/tree/2.2.1) - 2026-07-06

### Fixed

- Windows/MSVC builds no longer fail on unused ``cxxabi.h`` includes or nested
  ``std::array`` box flattening in ``Potential`` (``C1083`` / ``C2676``).


## [2.2.0](https://github.com/OmniPotentRPC/rgpot/tree/2.2.0) - 2026-07-05

### Added

- A profile-driven ABI loader (`rgpot::abi::ProfileLoader`) resolves the minimum
  potential ABI available from a single install prefix.

### Changed

- The canonical potentials-schema contract is pinned at v1.13.0, adding the
  Capabilities discovery surface; the in-tree `Potentials.capnp` copies track it
  byte-for-byte.

### Fixed

- `rgpot-core` publishes to crates.io again: `dlpk` (0.1.5) and `eindir-core`
  (0.5.0, `capi`) now resolve from the registry instead of git pins, and the
  `publish = false` guard is gone.


## [2.1.0](https://github.com/OmniPotentRPC/rgpot/tree/2.1.0) - 2026-07-03

### Added

- eOn integration: eOn ships an in-process `RgpotPot` potential (`-Dwith_rgpot=true`) that consumes rgpot's `NWChemPot` / `CPMDPot` frontends as a Meson subproject and `dlopen`s `libnwchemc` / `libcpmdc` directly, with `potserv` remaining available for out-of-process RPC. See `docs/orgmode/howto/eon-rgpot.org`.

### Changed

- The Cap'n Proto schema is pinned to canonical [potentials-schema](https://github.com/OmniPotentRPC/potentials-schema) v1.12.0: `MetatomicParams` arm (union ordinal 4) with the upstream requested-outputs surface, typed NWChem and CPMD parity batches (dplot/esp, prop/linres/pimd/path/tddft), `CommonMethodSpec` overlay, and a schema-sync CI gate that fails when the vendored copies diverge from the pinned release.

### Fixed

- Release-prepare CI no longer runs `cargo publish --dry-run` for `rgpot-core` while it depends on git-only `dlpk` / `eindir-core` (not on crates.io). The job uses `cargo check -p rgpot-core --locked`, and `package.publish` is `false` until registry deps exist. ([#42](https://github.com/OmniPotentRPC/rgpot/issues/42))


## [2.0.0](https://github.com/OmniPotentRPC/rgpot/tree/2.0.0) - 2026-06-26

> Channel note: `v2.0.0` was prepared on `main` but never tagged or published;
> the content below first ships in `v2.1.0`.

### Added

- NWChemPot backend: stable message-based C ABI (`nwchem_c_abi.h`), always-built frontend with `dlopen` of optional `libnwchemc`, stub ABI for CI without NWChem, Cap'n Proto `NWChemParams`/`configure @1`, and `potserv ... NWChem`.
- CPMDPot backend: always-built frontend with `dlopen` of optional `libcpmdc` (split [`cpmdc`](https://github.com/OmniPotentRPC/cpmdc) engine), Cap'n Proto `CPMDParams` / `PotentialConfig.cpmd` / `configure`, structured `CPMDInputSection` arms, in-tree `cpmdc_fake_engine` for CI without CPMD, and `potserv ... CPMD`. Engine lookup: explicit `enginePath`, then `CPMDC_LIBRARY`, `RGPOT_CPMDC_ENGINE`, `RGPOT_CPMD_ENGINE`, then `libcpmdc.so` and the other default names.
- rgpot potentials are now eindir objectives: `rgpot_potential_t` embeds eindir's `eindir_objective_t` as its first member (zero-cost IS-A), with the embedded eval/grad callbacks routed through the rgpot force callback (gradient = -force). rgpot-core consumes `eindir-core` as a shared Cargo crate rather than a prebuilt static lib, so downstream Rust consumers (e.g. `anneal-core`) can minimize an rgpot potential through `eindir_core::Objective<f64>` without a two-Rust-runtime conflict. See `docs/orgmode/howto/eindir-anneal.org`.

### Developer

- CI authoring uses a Nickel `ci/gha/` library with a single hand-maintained
  `ci-orchestrator.yml` for PR/main jobs (prepare/plan, hygiene, build/rust/bridge,
  potentials, CI gate), and nickel-exported workflows only for release, docs,
  cosmo potctl, and doc-commenter. Removed redundant `build`/`prek`/`potentials`/
  `docs_quality`/`towncrier`/`cosmo-potctl-spike` committed workflows in favor of
  the orchestrator plus `gen-gha`/`gha-drift` for the remaining exports. ([#40](https://github.com/OmniPotentRPC/rgpot/issues/40))

### Changed

- NWChemPot is now a pure consumer of the split [`nwchemc`](https://github.com/OmniPotentRPC/nwchemc) engine: the in-tree NWChem embed (`nwchem_c_abi.c`, `nwchem_embed_c_api.f90`, `nwchem_embed_legacy.F`) and the `-Dwith_nwchem`/`-Dnwchem_root`/`-Dnwchem_target` build options are removed. The frontend always builds and `dlopen`s `libnwchemc.so`; the capnp schema is synced byte-for-byte to nwchemc's canonical superset so a flat `NWChemParams` round-trips with no field loss.


## [1.2.0](https://github.com/OmniPotentRPC/rgpot/tree/1.2.0) - 2026-06-24

### Added

- XTBPot: GFN tight-binding via the xtb C API (GFNFF, GFN0/1/2-xTB). Feature-gated with ``-Dwith_xtb=true``. RPC names ``XTB``, ``GFNFF``, ``GFN0xTB``, ``GFN1xTB``. ([#35](https://github.com/OmniPotentRPC/rgpot/issues/35))
- TBLitePot: GFN tight-binding via the tblite C API (GFN1, GFN2, IPEA1). Feature-gated with ``-Dwith_tblite=true``. ([#36](https://github.com/OmniPotentRPC/rgpot/issues/36))
- MetatomicPot: load metatomic TorchScript models directly in C++ (vesin neighbor lists, autograd forces). Feature-gated with ``-Dwith_metatomic=true``. Requires vesin 0.5+ (``VesinDevice{VesinCPU, 0}``). ([#37](https://github.com/OmniPotentRPC/rgpot/issues/37))
- Units module (``rgpot/units.hpp``): CODATA 2018 constants plus a runtime unit expression parser, with Cap'n Proto ``lengthUnit``/``energyUnit`` negotiation on the RPC boundary. ([#38](https://github.com/OmniPotentRPC/rgpot/issues/38))

### Developer

- Lockstep monorepo release via cocogitto 7 + towncrier + ``potctl`` (``potctl/``,
  not published): ``cog bump`` runs ``potctl release sync`` then towncrier then
  ``potctl release assert --require-changelog``; ``release.yml`` publishes
  ``rgpot-core`` on stable ``v*`` tags. CI ``potentials.yml`` builds xtb/tblite and
  metatomic/vesin backends.
- Release tooling in ``potctl`` (Rust workspace crate): lockstep assert
  (meson/CMake/cargo/towncrier/pixi), fail-fast CHANGELOG gate, ``release sync``
  writes all surfaces, cargo publish --locked only, RC tags skip crates.io,
  release-prepare cargo dry-run, SECURITY.md and CODEOWNERS for release surfaces.

### Fixed

- MetatomicPot: target vesin 0.5+ ``VesinDevice`` struct (``{VesinDeviceKind, device_id}``, pass ``VesinDevice{VesinCPU, 0}``) and zero-initialized ``VesinOptions`` (``sorted`` / ``algorithm``).


## [1.1.0](https://github.com/OmniPotentRPC/rgpot/tree/1.1.0) - 2026-03-29

### Added

- XTBPot: GFN tight-binding potential via xtb (GFNFF, GFN0, GFN1, GFN2). Feature-gated with ``-Dwith_xtb=true``. ([#35](https://github.com/OmniPotentRPC/rgpot/issues/35))
- TBLitePot: GFN tight-binding potential via tblite (GFN1, GFN2, IPEA1). Feature-gated with ``-Dwith_tblite=true``. ([#36](https://github.com/OmniPotentRPC/rgpot/issues/36))
- MetatomicPot: ML atomistic models via metatomic/PyTorch with autograd forces and vesin neighbor lists. Feature-gated with ``-Dwith_metatomic=true``. ([#37](https://github.com/OmniPotentRPC/rgpot/issues/37))
- Shared unit conversion header (``rgpot/units.hpp``) with CODATA 2018 physical constants for Angstrom/Bohr, Hartree/eV, and Boltzmann conversions. ([#38](https://github.com/OmniPotentRPC/rgpot/issues/38))


## [1.0.3](https://github.com/OmniPotentRPC/rgpot/tree/1.0.3) - 2026-03-01

### Fixed

- MSVC/clang-cl compatibility: use ``/W3`` instead of GCC/Clang warning flags and skip ``-lstdc++`` link arg on Windows ([#31](https://github.com/OmniPotentRPC/rgpot/issues/31))


## [1.0.2](https://github.com/OmniPotentRPC/rgpot/tree/1.0.2) - 2026-03-01

### Added

- External integration guide covering namespace collision mitigation when embedding rgpot as a subproject ([#30](https://github.com/OmniPotentRPC/rgpot/issues/30))

### Changed

- CI dependency bumps and cleanup to prevent trailing whitespace in generated headers ([#29](https://github.com/OmniPotentRPC/rgpot/issues/29))


## [1.0.0](https://github.com/OmniPotentRPC/rgpot/tree/1.0.0) - 2026-02-15

### Added

- doc(arch): add architecture guide covering layer diagram, error conventions, and how to register new potentials
- feat(build): add release CI workflow, meson subproject install guards, CMake FetchContent readiness, and package version config
- feat(cache): add a RocksDB integration
- feat(ci): add `rust_tests` job using cargo-nextest on Ubuntu and macOS with default and `--all-features` configurations
- feat(cpp): add C++ RAII wrappers (`include/rgpot/`) — `PotentialHandle`, `InputSpec`, `CalcResult`, `RpcClient`, `Error`, with full Doxygen documentation
- feat(rpc): add feature-gated Cap'n Proto RPC client and server in Rust, sharing the existing `Potentials.capnp` schema with the C++ side
- feat(rpc): initialize a C style integration to the server
- feat(rpc): initialize a server component
- feat(rust): add Rust core library (`rgpot-core/`) with `#[repr(C)]` types, callback-based potential dispatch, status codes, thread-local error handling, and auto-generated C header via cbindgen
- feat(rust): integrate DLPack tensor exchange protocol via `dlpk` crate — core types now use `DLManagedTensorVersioned*` for device-agnostic data exchange, with borrowed (non-owning) and owned tensor helpers in new `tensor` module
- test(rust): add 39 unit tests covering types, status codes, potential lifecycle, C API, null-pointer handling, error propagation, and `free_fn` invocation

### Changed

- chore(build): add Meson `with_rust_core` option, `rust-test` / `rust-test-all` pixi tasks, and `cargo-nextest` dependency

### Fixed

- fix(cpp): remove incorrect `extern "C"` trampolines from `LJPot.hpp` and `CuH2Pot.hpp` that used raw `int` returns and `void*` params; use typed `PotentialHandle::from_impl<>()` template instead


## [0.0.1](https://github.com/OmniPotentRPC/rgpot/tree/v0.0.1) - 2024-01-26

### Added

- Initial release with C++ core: Lennard-Jones and CuH2 EAM potentials.
- CRTP-based `Potential<Derived>` template with optional caching.
- Cap'n Proto RPC server and client bridge.
- Meson and CMake build systems.
- CI build matrix (Meson/CMake x Linux/macOS x RPC/Cache feature flags).
- RPC integration tests and client bridge stress tests.
