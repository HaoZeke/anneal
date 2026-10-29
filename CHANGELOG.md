# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!-- towncrier release notes start -->

## [0.12.0](https://github.com/HaoZeke/anneal/tree/v0.12.0)

### Added

- ``paper_controls`` runs the registered control arms: ``recommended``,
  ``for_cluster`` (screened basin hopping, and the in-tree control while
  GMIN stays on the optbench side), ``derived``, and soap on against
  soap off. The log header carries the commit, config digest, engine
  digest, seed, force count, wall time and CPU model, and each run
  writes a ``RunManifest``. LJ38 uses the first 72 seeds of 2000-2199
  at 400000 evaluations. LJ75 and LJ98 use the first 48 at 4000000.
  Held-out landscapes use 2200-2247 and do not invent a budget. A run
  launched before the v0.12.0 tag is a smoke and is labelled smoke.
- Catalogue selection uses four ranks. The default log weight is
  ``-r_E + 0.8 r_N + 0.6 r_S + 0.4 r_U``, clipped at 4 nats. ``r_U`` is
  the rank of the graph-field score ``variance / (1 + effort)``. Scores
  within one part in 1e8 share that rank.
  ``lj38_catalogue_epoch`` logs one four-replica LJ38 epoch on seeds
  3100-3103. The paper force total is 400000, shared by the four
  replicas. A run launched before the v0.12.0 tag is a smoke and is
  labelled smoke.
- ``water_hexamer`` reads deposited isomer geometries from
  ``WATER_ISOMER_DIR`` and sets the assignment gate to half the minimum
  species-aware IRA RMSD before any seed. ``ASE_PYTHON`` selects the
  deposited calculators: tblite 0.7.0 GFN2-xTB, PySCF 2.14.0 RKS
  PBE/6-31G* grid level 3, and pet-mad 1.4.4 checkpoint 1.0.2 on CPU,
  through ASE 3.29.0. The helper keeps one ASE atoms object and
  updates positions. The gate is printed even when the engine does
  not load. A non-finite opening quench is reported and does not drop
  the gate or the reference energies.
- ``water_hexamer`` searches (H2O)6 under GFN2-xTB, NWChem PBE/6-31G*, and
  PET-MAD. The PBE arm records basis 6-31G*, theory dft, functional PBE,
  charge 0, multiplicity 1, task gradient, and no named dispersion
  correction. Recommended, communicating, derived, and Good-Turing arms
  keep the molecular groups. Integrity requires exactly two hydrogens
  within 1.2 angstrom of each oxygen. A dissociated monomer is
  unassigned. Isomer names wait on same-level reference geometries.
  The isolated-water template is not that set. Seeds are 0-31 and the
  force budget is 4000 on every engine. A run launched before the
  v0.12.0 tag is a smoke and is labelled smoke.
- Lattice-ensemble ablation arms on ``Plan``. ``retire = 0`` never frees a
  bank slot. ``Resemblance::LeeCsa`` uses eq. (2) of Lee, Lee and Lee,
  Phys. Rev. Lett. 91, 080201 (2003): the weighted absolute difference of
  the first and second shell coordination histograms, radii 1.35 and 1.70
  at sigma = 1, scaled by the pair minimum on any other pair.
  ``CutoffSchedule::LeeFixedRatio`` reduces the merge distance by a fixed
  ratio from D_ave/2 to D_ave/5 over 10000 local minimizations and then
  holds it. ``banked = false`` runs Shao, Cheng and Cai's lattice move
  from a fresh start, with no bank and no splice. The defaults stay the
  radial key, the held cutoff, retirement after 20 draws, and the bank.
  ``lj_lattice_ensemble`` runs its private arm at two slots unless
  ``SLOTS`` is set, and its default seed is the registered block.
  ``lattice_ensemble_ablation`` reports the T1 columns and the paired
  difference against the full ensemble. The quench and the lattice site
  energy follow ``PairKind``, so a Morse ensemble is Morse.
- ``pt111_heptamer`` quenches the OptBench Pt(111) heptamer with rgmin and
  certifies edges with an rgsaddle dimer from each quenched endpoint pair.
  The hop and the RMSD use the smallest free layer split at 1.0 angstrom.
  Other free layers keep a live gradient. Morse parameters and reference
  geometries are read from the OptBench problem file and ``.con`` tree.
  ``mid.con`` is not treated as a saddle. A match is an energy within
  0.05 eV and an RMSD within 0.1 angstrom.
- ``paper/seeds.toml`` registers the paper seed ranges before any run.
  R1, R12 and R15 use 2000-2199. R2 uses 2200-2247. The lattice
  ensembles use 3000-3029 on LJ75 and LJ98 and 3100-3199 on LJ38,
  including the shared arm with splice 0 (``nosplice``). The Pt(111)
  heptamer uses 4000-4047 and the soap pairs use 5000-5063.
  ``Plan::default`` is frozen at the field values pinned by
  ``plan_default_is_frozen``.
- Binary Lennard-Jones and rigid TIP4P on the stock cluster presets.
  ``cluster_search`` takes ``species``, ``sigmas``, ``rigid``, ``layout``,
  and ``target``. Two or more species labels add a swap move. A wide
  diameter ratio leaves the occupied basin by one softened constant-energy
  (NVE) step and one inner-outer exchange, then quenches that point in full.
  That step needs the ``rgsaddle`` feature.
  The same chart also tries one unlike-pair exchange from a side
  generator and keeps it only when the quench is downhill.
  A rigid chart orients every body and keeps the centres fixed. The
  acceptance temperature of a rigid chart is scaled inside the hop. A
  chart with more than ten bodies holds that factor at 4 and scales the
  orientation amplitude toward an acceptance ratio of one half. A chart
  of ten bodies or fewer is unchanged. Temperature, screen length,
  relax length, and record gradient stay on the preset. A rigid quench
  is that same L-BFGS descent, and it returns when the energy crosses
  the run target. A rigid record accepts the root-mean-square force.
  A rigid chain that goes 1000 hops and records no new lowest minimum
  returns to that minimum when it lies within one hopping temperature
  of the run target. Farther above the target the walker is replaced
  by a fresh draw and quenched, and the ledger keeps the lowest
  minimum. That 1000 is the restart-file interval in
  the source. The compiled reseed window is zero. The C header is unchanged.
- ``BinaryLennardJones`` is a two-species Lennard-Jones cluster with no
  cutoff. Species A is the first ``n_a`` atoms and the unlike diameter is
  the value the caller passes. ``GminTip4p`` is rigid TIP4P on the GMIN
  DEFTIP4 coefficients, stored as centres followed by Rodrigues rotations.
  Neither potential adds a C or Python parameter.
- ``cluster_search`` takes ``water_sites``, the body-frame O, H, H positions of
  a rigid water, and then searches ``n`` molecules as centres followed by
  exponential-map rotations, on ``Config.communicating_tip4p``,
  ``Config.recommended_tip4p`` or ``Config.for_tip4p``. The gradient-sign
  probe runs at the start on a rigid state. ``Config`` gains the two TIP4P
  constructors and the ``rigid_body``, ``screen_margin`` and
  ``calibrated_screen`` getters.

### Developer

- A half turn composed on the TIP4P Rodrigues chart stays on its own axis.
  The composition helper is what a rigid orientation step calls, and it
  lives with the potential so the hop and the energy share one product.
  The C header and the Python bindings are unchanged.
- Coordinate-length checks on the GMIN TIP4P helpers use
  ``is_multiple_of``. The C header and the Python bindings are unchanged.
- The stock binary and rigid helpers compile under the current clippy
  denies. Species and diameter lists use ``repeat_n``. The rigid move
  list is returned directly, and the wide-exchange guard is one
  condition. Behaviour is unchanged. The C header and the Python
  bindings are unchanged.
- The normal-quantile table and the format strings that name their
  values satisfy clippy's deny set. The quantile values are unchanged.
  The TIP4P profile walks the sites it indexes. The C header and the
  Python bindings are unchanged.
- ``examples/globalopt_profile.rs`` profiles the cluster presets on optbench
  archive starts under harness-style counting, and
  ``scripts/optbench/anneal_globalopt.py`` scores them through optbench's
  lane-B harness with native kernels checked against optbench's own.

### Changed

- The Pt(111) heptamer dimer stops translation on the maximum force
  component. ``rgsaddle`` 0.3.0 has no force-gate field on
  ``MinModeConfig``.

- ``Plan::start`` is ``StartRule::PresetSphere`` or ``StartRule::Density``.
  The default is the preset sphere of ``Config::recommended``, radius
  ``0.9 * L0 * N^(1/3)`` with ``L0`` the preset length scale. A fresh
  trial used to be ``random_cluster`` at density 0.7; that draw is
  ``StartRule::Density(0.7)``. The serial arm at seed ``s`` quenches to
  the same first energy, bit for bit, as the recommended chain at that
  seed. Changing the default does change existing output. The 13-atom
  ensemble tests still pass and still finish on the icosahedron, but
  the best-energy bits and the trace are not the density-0.7 values.
  ``lj_lattice_ensemble`` prints different energies unless ``DENSITY``
  is set, which selects ``StartRule::Density``.
- ``Lattice::for_pair`` sets bond, hollow, site, clearance and merge
  distances to 1.2, 1.47, 1, 0.85 and 0.25 times ``PairKind::r_min``.
  ``interior_coordination`` stays a field. ``Lattice::default`` is
  ``for_pair(LennardJones)`` and matches the previous hard-coded well
  bit for bit. ``lattice_ensemble::run`` takes that pair kind.
- The optional ``rgsaddle`` feature is off by default. It compiles
  minimum-saddle-minimum sessions and stationary-index checks. It
  compiles the constant-energy (NVE) minima-hopping escape and the
  dense symmetric eigensolve for a connected landscape split. A default
  build omits those paths. Connected spectral splits return
  ``GraphError::FeatureDisabled``. Saddle sessions and stationary-index
  checks return ``PesExplorationError::SaddleUnavailable``. An NVE
  escape returns ``NveError::FeatureDisabled``. ``md_escape`` and the
  binary wide-diameter-ratio escape stop the run and set
  ``Outcome::nve_unavailable``. ``IrcKind`` and ``NveError`` each have
  one definition. ``python`` and ``bank-rpc`` enable the feature.
  PyPI wheels and in-repo feature builds keep the sessions. The
  feature compiles rgsaddle 0.3.0 from crates.io against rgmin 0.5.0.
  ``rgsaddle``, ``python``, and ``bank-rpc`` build from crates.io.
  The lockfile records rgmin 0.5.0, rgsaddle 0.3.0, eindir-core 0.6.0,
  and linkcell 0.3.11. The activation relaxation leaves ``ftol_rel`` and
  ``noise_floor`` unset.
- ``nve_escape`` and ``nve_escape_seeded`` return ``NveError``.
  ``GraphError::FeatureDisabled``,
  ``PesExplorationError::SaddleUnavailable``, and
  ``NveError::FeatureDisabled`` are new. ``Outcome``, ``ArchiveOutcome``,
  ``CommittorOutcome``, ``BankOutcome``, and ``StockSearch`` gain
  ``nve_unavailable``.
- The cluster hop's energy screen credits each trial with the descent its
  screening pass leaves undone, the upper decile of screened minus full over the
  last 256 full relaxations, and relaxes every eighth refusal in full so the
  estimate also samples the trials it refuses (``Config::calibrated_screen``).
  A full relaxation that ends on its step cap below the incumbent is continued
  until it stops descending (``Config::finish_improving``). Both are on in
  every preset. ``Config.calibrated_screen`` and ``Config.finish_improving``
  read the two flags from Python. On 256 TIP4P10 archive starts at
  development seeds and 1e6 force calls, PAR-2 went from 78681 to 15870 on
  ``Config::for_tip4p`` and from 92738 to 16369 on
  ``Config::communicating_tip4p``.
- The cluster-search quench stops at ``Config::record_gradient``, the gradient
  at which a minimum is recordable. The stop sits above rgmin's 1e-6.
  On 32 LJ38 archive starts at development seeds the communicating preset's
  PAR-2 fell from 5792 to 4269 force calls and its median from 5726 to 3253.
- ``Config::recommended`` completes the core's orbit on every new basin and,
  below 98 points, draws between the plain and the two-phase relative 0.7
  surface, the rule ``Config::communicating`` already used. On optbench archive
  starts at development seeds the preset went from a median 1.4e5 force calls
  on 32 LJ38 starts to 4.9e3, and from 2 to 16 of 16 LJ75 starts solved in 1e6.
- ``cluster_search`` keeps both chart keyword sets on one call. ``species``,
  ``sigmas``, ``rigid``, ``layout`` and ``target`` select a stock preset and
  then a binary or rigid chart. ``water_sites`` selects the TIP4P presets and
  a caller body frame. The two sets exclude each other. A callback that raises
  ``KeyboardInterrupt`` still stops the run. The C header is unchanged.

### Fixed

- ``Config::recommended`` at 38 points grows the named lattices (face-centred,
  hexagonal, icosahedral, simple cubic, and the observed cloud) instead of
  the LeanBurst displacement stack. That stack left 17 of 72 seeds on the
  Mackay minimum ``-173.252378`` at 400000 calls. At 75 and 98 points the
  global minimum is not one of those packings, and ``recommended`` stays
  LeanBurst. A stall takes the recorded basin trail, the climb, and the
  restart. ``for_cluster`` turns the Bayes screen on at
  ``(R-S)/(2R-S)``, which is ``7/15`` at 25 and 200 steps. A call plateau
  with the climb enabled is the directed leave, not a fresh Mackay draw.
  ``paper_controls`` passes the potential gradient, which is what the
  climb reads.
- The LJ75 Mackay opening is read in the deposited fixture's atom order.
  Sorted distance signatures put a search minimum at that geometry back
  into that order before the fivefold residual, which otherwise follows
  storage order and misses the axis at length 0.9018. The quench is
  adopted only when the current energy is the shelf and the gap above it
  sits in ``(1.02, 1.12)``. The climb stays the gate, so ``for_cluster``
  does not take the opening.
- The seam restart banks a structure only when it is within two epsilon
  of the previous floor and its icosahedral template share is strictly
  below that floor. A DECAF gap and an energy doorway are not the
  broadcast. ``communicating`` leaves ``restart_on_stall`` off.
- The census IPC address, the q-Gaussian bound check, the q-Gaussian
  error, and the MD work directory name the values they test. The address,
  the accepted ``(q_v, q_a)`` interval, the error text, and the directory
  name are unchanged.
- ``rgpot-ex`` links one ``eindir_abi_stamp_t``. ``rgpot-core`` is crates.io
  3.5.1, which depends on ``eindir-core`` 0.6.0, the same crate the rest of
  the workspace uses. The git pin at 3.0.2 compiled a second eindir, and
  packaging rewrote that pin to registry 3.0.2, which depends on
  ``eindir-core`` 0.5.2.
- The certified-constructor UI tests record the current compiler wording.
  Each uncertified combination still fails to compile: an acceptance rule
  is not a certified acceptance, a downstream crate cannot implement the
  sealed acceptance trait, and a full-space move is not certified for a
  box. The UI build repeats the path patches the parent resolve used, so
  the check runs offline.
- The Rust API page is generated again. ``sphinx-rustdocgen`` 1.3.0 joined
  a ``#[path]`` module onto the file that declared it, then failed to read
  that path. The docs build installs 1.3.0 with that join made from the
  file's directory. Generated pages join a wrapped code span onto one line,
  keep a literal closed when a word follows it, and escape LaTeX bars.
  The cluster-search title line matches the title. The method catalog, the
  composition walk, the communication and regime notes, and the box
  Langevin escape are in the contents. The eindir inventory is no longer
  fetched at build time.
- Water isomer names use species-aware IRA. The gate is half the
  minimum gap between distinct deposited references and is computed
  before any seed. A dissociated monomer is unassigned. Search
  geometries and deposited GOFEE geometries use that same classifier.
  The Kabsch helper is not the label. A build without the ``ira``
  feature leaves the gate unset instead of substituting that helper.
- A search whose first quench is not finite returns
  ``SearchError::NoFiniteStart`` instead of continuing from that value.
  The ledger redraws a start, up to ``DEFAULT_START_ATTEMPTS`` (8), and
  records each redraw. A finite first quench is unchanged. A caller stop
  during that opening quench returns the outcome recorded so far.
- Slab hop groups use a species-pair cutoff. Cu-Cu is 2.36 angstrom,
  above the Cu2 bond length 2.21927(3) angstrom (Ram, Jarman and
  Bernath, J. Mol. Spectrosc. 156, 468, 1992,
  doi:10.1016/0022-2852(92)90247-L) and below the 2.5 angstrom fixture
  contact. Other metal pairs use the unscaled Cordero sum, so Pt at
  2.77 angstrom stays separate and a shorter metal dimer stays one
  body. Light pairs stay at 1.25 times the Cordero sum, which bonds H2.
  A rule that never bonds two radii of at least 1 angstrom splits that
  Cu2 bond, so it is not used.
- An in-process census cohort dials neighbours only after every publisher
  is bound. A dial that landed on a port reservation completed the TCP
  handshake there and waited for a protocol header the reservation never
  sends. That wait is ten seconds. Closing the reservation on macOS leaves
  the unaccepted connection open. A five second connectivity check then
  expired with replicas missing.
- Search drivers return as soon as the caller stops evaluation and keep the
  best state found. They no longer spend the rest of the budget after the
  potential stops answering. A ledger slice keeps the caller's stop flag.
  ``ANNEAL_STOPPED`` from a C callback ends
  ``anneal_box_minimize`` the same way and writes the best point. A flag
  that stays down leaves the force-call count unchanged. Stock binary
  Lennard-Jones and rigid TIP4P searches, including a batch of them, return
  on that stop. So does ``cluster_search`` on a binary chart, a rigid chart,
  or a water-site chart.
- Stopping a catalog coordinator closes its listen socket before stop
  returns, so a new coordinator can bind that same address. The thread
  that owned the listener kept running after stop, and the listen fd
  was closed only after stop had already returned, so a restart on that
  port failed with the address already in use.
- The JCC LJ causal pair runner runs under bash 3.2, the bash GitHub puts
  on macOS images. Protocol values are assignment exports. Replica indexes
  use a shell loop. The dynamic-loader check runs where ``ldd`` is present.
  Run artifacts are sealed with a portable read loop and a plain ``sort``.
- The cargo-c install places ``dlpack/dlpack.h`` (DLPack 1.3, the header
  ``dlpk`` 0.1.5 was translated from) beside ``anneal-core.h``. The C header
  includes that file. ``pkg-config --cflags anneal-core`` compiles a
  consumer, including ``DLManagedTensorVersioned``. Meson and CMake install
  the same header.
- Two-choice restarts read the occupied state a peer published at the
  same checkpoint. A peer that has already finished contributes the last
  state it published. The mailbox returned the scheduler's stored state.
  An empty store made one run restart from a different state than its pair.
  A private repeat then disagreed by a hop, 1250 against 1251.
- ``cluster_search`` ends at its next evaluation when a callback raises a
  ``BaseException`` outside ``Exception``, such as ``KeyboardInterrupt``,
  and re-raises it. ``Ledger::with_stop`` closes a ledger on a caller-owned
  flag.

### Miscellaneous

- CI runs on pushes to ``develop`` and to ``main``. ``cargo fmt`` checks
  the workspace member. Workspace exclude covers ``nng`` and ``nng-sys``.
  It also covers ``metatensor-sys``. Stable rustfmt formats the workspace
  member. The vendored crates' ``rustfmt.toml`` asks for nightly options.
  The ``ignore`` key is unstable on stable. ``cargo fmt --all`` would still
  walk a path dependency.


## [0.11.1](https://github.com/HaoZeke/anneal/tree/v0.11.1) - 2026-10-11

### Fixed

- The box ensemble drivers keep the callback error record the other
  drivers already keep, and a callback exception ends the call the same
  way.


## [0.11.0](https://github.com/HaoZeke/anneal/tree/v0.11.0) - 2026-10-11

### Deprecated

- The ChemFit bridges still run these calls anneal 0.10.0 took, and give a
  ``FutureWarning`` that says what to pass instead; each will raise in a future
  release. Preset keywords under the portfolio are ignored: ``preset_kwargs`` in
  ``fit_anneal`` and ``run_fitter``, the preset keywords of ``fit_chemfit``, and
  ``preset`` in ``run_benchmark`` and ``run_fitter``. So is a ``fit_chemfit``
  preset keyword of another method, so one set of keywords still sweeps
  ``boltzmann``, ``fast`` and ``gsa``, and a ``run_fitter`` preset of another
  kind than its classical ``method`` (``method="sa"`` runs the preset).
  ``run_benchmark`` runs a preset of another method, or one given with
  ``method="sa"``, as that preset's own method. A numeric string given for a
  parameter value, a bound, ``bound_span``, ``default_span`` or a preset keyword
  a ``fit_chemfit`` method takes, such as a bounds pair PyYAML reads from
  ``[1e-3, 1e1]`` or an item of an object array, is read as a number, with one
  warning per call, and a string parameter comes back as the float it reads as.
  An ``x0`` dict leaf with its parameter's size but another shape is read in C
  order, and one ``x0`` value for the fitter's only parameter fills it where
  ``fitter.bounds`` gives both of its sides and ``low`` and ``high`` are
  omitted. ``fit_chemfit`` reads a fitter's ``initial_parameters`` given as
  ``(key, value)`` pairs, which ChemFit 3.1 keeps as given, as the dict they
  make, whether a list, a tuple, ``d.items()``, a zip or a generator; as in
  0.10.0 they are read once, so a zip or a generator is spent after one fit.
  ``run_benchmark`` drives a fitter that has ``evaluate`` but no ``step``, or
  ``ask`` but no ``tell``, with its ``tell`` or ``step`` as the step notice, or
  with no step notices when it has neither.

### Added

- `tensor_id`: basin identity from a three-body tensor invariant.
  `SortedPairs` identifies a basin by the multiset of pairwise distances, which
  fixes `tr A^2` of the Gaussian kernel `A` and leaves `tr A^3`, the weighted
  triangle sum, free to differ; Bloom's homometric pair `{0,1,4,10,12,17}` and
  `{0,1,8,11,13,17}` share the multiset and are not congruent. `TripletSpectrum`
  appends the spectra of `A` and of the mode-3 contraction
  `M = A .* (A A)` of the triangle tensor `T_ijk = A_ij A_jk A_ik`, both exactly
  invariant to relabelling and to rigid motions, to the sorted distances it
  leaves unchanged. Worst-case separation over all pairs of distinct quenched
  Lennard-Jones minima, in units of the descriptor's own response to a 0.02
  jitter, goes from 0.85 to 1.17 at 38 points and from 0.87 to 1.11 at 55, both
  from below 1 where no merge radius separates the closest pair at all, and the
  homometric pair from exactly 0 to 10.3. At 13 points, where the distance
  descriptor was not failing, it does not move. The cost is 91.1 us at 38 points
  and 1.05 ms at 98, against 2.45 us and 17.8 us for one Lennard-Jones gradient
  of which a hop spends thirty-one, and no potential evaluations at all. The
  Tucker core is not a descriptor: it is invariant only up to a block-orthogonal
  gauge, and the full `N^4` mode spectrum separates no better than the `N^3`
  contraction. ([#tensor_id](https://github.com/HaoZeke/anneal/issues/tensor_id))
- A ``cma`` arm for values-only portfolio runs (``policy="auto"``, no
  ``grad_fn``, no ``noise_sigma``): resumable (mu/mu_w, lambda)-CMA-ES with
  cumulative step-size adaptation, rank-one plus rank-mu covariance updates, and
  BIPOP restarts. Every run is centred on the incumbent, which is ``x0`` until a
  lower point is found; the first run is local, and the first restart after it
  has the default population and a box-wide step size. A run ends once its
  recent best values span less than the portfolio's success threshold in force
  when the arm last took a slice, which ``CmaEs::set_tol_fun_hist`` sets between
  generations (``CmaEs::tol_fun_hist`` reads it). Above 100 dimensions, or when
  the budget is shorter than the covariance's learning horizon, runs keep only
  the diagonal (sep-CMA-ES), so a candidate costs linear time and memory.
  Candidates are mirror-reflected into the box and each one is a single charged
  evaluation.
- A ``qn`` arm for values-only portfolio runs: projected BFGS on
  finite-difference gradients from the incumbent, with a projected backtracking
  line search and stencils that step inward at the bounds. The initial inverse
  Hessian is the box metric ``diag(w^2)``, so sides that span decades descend as
  fast as unit ones. A descent that begins at an incumbent already evaluated,
  ``x0`` included, reuses its value. Forward differences give way to central ones
  when they stall, a converged descent is kicked from the incumbent with draws
  independent of the CMA-ES arm's, and every stencil point and trial is a single
  charged evaluation.
- A rigid Jorgensen TIP4P water objective, with an exponential-map
  rotation chart and a Wales--Hodges translation/rotation move library,
  so water clusters can be searched against the published putative
  minima.
- Added ``from anneal.ase import Anneal``. Same ``run(fmax=...)`` call site
  as ASE ``BFGS`` / ``LBFGS`` / ``FIRE``. Default ``mode="local"`` is a
  quench only. ``mode="search"`` hops from the current geometry
  (``search_from``, length scale from nearest neighbours) then quenches;
  ``mode="auto"`` hops when there are four or more free atoms. Not a local-optimizer rename: hops leave the
  starting basin. Hop ``seed`` and ``budget`` are explicit. Periodic
  cells and cell filters are refused on hop.
- Box-constrained communicating hops share a Euclidean minimum history
  and Goedecker escape feedback. The move is a reflected Gaussian plus
  quench, not a 3N cluster kick.
- Chains share a motif-class table. A chain restarts when its own best
  in the class has not improved for ``CORE_PATIENCE`` charged calls, or
  when a finished trial sits above the class median. The in-process
  cooperative catalog and the Cap'n Proto coordinator expose the same
  ``report`` operation.
- Cu(111) multi-adsorbate fixture: ``scripts/make_cuh2_fcc_slab.py --hydrogens 6``
  writes ``examples/fixtures/cuh2_fcc_slab_h6.con``, a 4x4, 4-layer slab with
  the bottom two layers frozen and six randomly placed hydrogens.
- Dynamic lattice search (``methods::lattice_search``) keeps two readers of a
  pair cluster. One builds hollow sites over the triangles and squares of the
  current structure, grows an interior lattice that carries both stackings of
  every facet, places the surface around the kept interior, and walks
  occupations until no relocation lowers a point. The ``dls`` driver token
  uses it, and the restoration tests cover the LJ98 and LJ104 minima.
  The other is a variant of the dynamic lattice search of Shao, Cheng and Cai,
  J. Comput. Chem. 25, 1693 (2004), doi 10.1002/jcc.20096. Its sites sit at the
  pair-well distance over the cluster's own triangles and are not relaxed;
  every atom with fewer than twelve bonds is movable; one deterministic greedy
  pass weighs the highest-energy movable atoms against all sites; and one
  L-BFGS quench in the step-capped form used by GMIN follows each search. The
  lattice needs a pair potential with one well distance. Every value-and-gradient
  call is charged as one call. Lattice pair terms are charged at their fraction
  of the n(n-1)/2 pairs of a full evaluation and settled in whole calls. Under
  one call per chain can be outstanding at a first hit or at the end of a run,
  while choosing a move costs at most one call. ``examples/lj_lattice_probe.rs``
  reports the cost and end energies of random starts taken down the descent.
- Leave extra starts are SHS downward distortions on the active-volume sphere and farthest-point packing covers.
- Occupancy Leave searches an active volume of packing leftover, several times, and discards unused starts.
- Python ``ensemble_optimize`` is the box search entry: with a
  gradient it hops and quenches; without one it is the values-only
  portfolio. Each Python eval and grad callback spends one ledger unit.
  The design vector is never treated as a point set.
- Reoccupation move inside the chain (``Config::reoccupy`` and
  ``reoccupy_interval``, driver token ``reocc``): every interval of charged
  calls the chain rebuilds its surface on the lattice grown from its interior,
  relaxes it and adopts the result when lower, charging all of the work to the
  ledger. Takes LJ104 from 0 to 23 of 32 seeds and LJ102 from 59 to 117 of 128
  chains at 400k evaluations.
- ``examples/lj_lattice_ensemble.rs`` compares the shared and private lattice
  ensembles with the serial recommended chain and forty independent recommended
  chains at one ensemble force total, on paired seeds and at most the requested
  number of threads. Its recommended arms reproduce ``lj_ensemble_splice`` in
  ``indep`` mode bit for bit.
- ``global_optimize`` and ``global_optimize_objective`` accept ``x0``, a start
  point inside the box that is evaluated first and becomes the first incumbent.
  ``portfolio_optimize_from`` is the Rust entry point. An invalid start raises
  ``ValueError``.
- ``methods::lattice_ensemble`` runs chains that split one force budget and pool
  the minima their lattice descents reach in a bank held apart by a radial
  resemblance cutoff, under the bank rule of conformational space annealing
  (Lee, Scheraga and Rackovsky, J. Comput. Chem. 18, 1222 (1997)) with the
  D_ave/2 merge distance of Lee, Lee and Lee, Phys. Rev. Lett. 91, 080201
  (2003), arXiv cond-mat/0307690. Trials start from random clusters, from members
  with surface atoms moved onto vacant hollows, or from cut-and-splice pairs of
  members (Deaven and Ho, Phys. Rev. Lett. 75, 288 (1995)). A member drawn twenty
  times without improving gives its slot to the next random start that resembles
  no member, so the bank keeps taking in new funnels after its first members stop
  descending. The retirement rule, the radial resemblance key, the held merge
  distance and the counted accounting are this work's. ``Sharing::Private``
  gives each chain its own bank. A one-slot private bank never splices and never
  draws the splice coin, so its random streams part from the shared arm's at the
  first bank draw, if not sooner, and the private ablation removes cross-chain
  splicing together with the exchange of members. Chains advance in synchronous
  generations with parents drawn and offers admitted in chain order, so a run
  replays bit for bit at any thread count.

### Changed

- Central differences in the values-only finite-difference descents (the ``qn``
  arm and the closing polish) adapt their interval to the curvature each stencil
  measures. After the first central stencil at a coordinate, its next interval is
  a tenth of the distance ``|g / c|`` to the minimum of the quadratic through the
  stencil, never above the ``eps^(1/3)`` rule and never below ``eps^(2/3)`` times
  the coordinate's scale or the rounding limit ``sqrt(eps |f| / |c|)``. A failed
  central line search re-differences at the same point when an interval has
  moved by more than a factor of two, at most twice per iterate. Fits whose
  valleys are narrower than the ``eps^(1/3)`` interval, such as
  ``binary_lj_fit``, now descend to the noise floor instead of stalling where the
  stencil straddles the valley.
- In values-only portfolio runs the GSA arm runs one chain from 10 dimensions,
  and below that one for every eight evaluations of its first slice, between two
  and four. The first chain starts at the incumbent, as dual_annealing starts
  from ``x0``, and the others at seeded low-discrepancy points; once another arm
  has lowered the incumbent below every chain, the worst chain goes on from
  there. Chains run quenched, without a local search: each temperature step
  visits every coordinate once, with dual_annealing's visiting distribution
  cooling from box-wide steps to local ones, and keeps only improvements.
  Without a local search, Metropolis moves at dual_annealing's temperatures
  spent most of a short budget uphill, and on separable multimodal objectives
  (shifted Rastrigin, Styblinski-Tang, Levy) a one-coordinate visit drops that
  coordinate into a lower basin where a visit to every coordinate at once almost
  never improves a good point. The DE arm's population starts with the
  incumbent, as SciPy's differential_evolution places ``x0``, and its first slice
  evolves the population once it is seeded.
- In values-only portfolio runs the additive surrogate and the QMC restart arm
  take values in the run's unit, the power of two at or below the success
  threshold, where their temperatures and floors were absolute. The surrogate
  fits values, and sets its temperatures and their floors, in that unit, and the
  restart arm's annealing chains weigh each rise in it against a temperature
  that starts at one, so they all but never take a rise as large as the last
  gain. Multiplying the objective by a power of two now leaves a whole
  values-only run unchanged, short of overflow and underflow, where a run used
  to depart from the unscaled one once the surrogate or the restart arm played.
- In values-only portfolio runs the success threshold, which decides whether the
  opening descent goes on, whether a descent keeps its turn, whether a GSA or
  CMA-ES phase has gained and when a CMA-ES run stops, is ``1e-4 max(g, s)`` in
  place of ``1e-4 max(|f|, 1)``, which turned absolute below ``|f| = 1`` and
  coarsened when a constant was added to the objective. ``g`` is the gain of the
  last slice that lowered the incumbent from below the start's value, and ``s``
  the run's resolution: ``eps |f|`` of the first value the run finds below the
  start's, but never less than the smallest positive normal double, which it is
  until such a value is found (when the start's value is not finite, every
  finite value is below it). Gains are differences of values and a slice from
  the start's value records none, so the start's value sets no threshold, and a
  constant added to the objective changes one only through the resolution of the
  shifted values: the rounding of each gain, and ``s``. ``s`` changes at most
  once and never falls below that positive floor, so a bounded objective still
  allows only a bounded number of successes, as the restart guarantee needs.
  The values-only descent likewise stops on the resolution of its values, an
  iteration that lowers its value by less than ``8 eps`` of it being slow, and
  until a curvature pair scales the inverse Hessian each line search of a
  finite-difference descent starts with a trial whose largest move is 5% of a
  side whatever the gradient's size, where a small gradient used to shorten it.
  Multiplying the objective by a power of two multiplies every threshold
  exactly. Measured over 40 paired seeds against the rules they replaced
  (medians, "A against B" with A's wins/ties/losses, rows named
  ``problem@budget`` from ``x0`` as in the regime notes), the descent's
  resolution test trails a fixed ``1e-12`` tolerance on ``fit6@1000``, -4.78
  against -4.89 (3/20/17) and without ``x0`` -3.334 against -3.334 (1/32/7),
  with its x1e-6 and x2^-20 copies and its +1e2 and x1e-6 controls, on
  ``ellR30@20000`` and ``ellR50@20000``, where five and six seeds stop short of
  0 (0/35/5, 0/34/6), and on the x1e-6 control of ``ellR30@5000`` (1.2e-6
  against 0, 2/19/19). With the tolerance these come level, but the offset
  copies of ``fit6@5000`` fall back, +1e2 to -11.6 against -12.8 (9/9/22),
  +1e4 to -9.33 against -12.6 (0/17/23) and -1e4 to -8.15 against -12.8
  (0/24/16), and ``fit6@5000`` itself is -12.78 against -12.83 (12/10/18); a
  slow test at a fraction of the threshold costs more (``fit6@5000`` -6.79).
  ``ellR30@5000``, 3.4e-8 against 0 (0/19/21), and its x1e-6 and x2^-20 copies
  come level only with the tolerance and the opening without its bar on recent
  gains together (0, 0/38/2). The threshold on the last gain trails
  ``1e-4 max(|f|, s)`` on ``rastR100@20000``, 870 against 833 (4/18/18), and on
  the +1e2 copy of ``levy30@5000``, 6.4e-11 against 4.3e-12 (5/22/13); the
  ``|f|``-scaled threshold brings both level but costs the offset copies: +1e4
  ``rosen10@1000`` is 5.40 against 1.2e-10 (0/0/40), +1e4 ``fit6@1000`` is
  -0.847 against -4.67 (1/0/39) and at 5000 -6.16 against -12.6 (1/1/38), and
  +1e4 ``ellR30@5000`` is 4.4e-3 against 2.9e-4 (0/6/34). Of the 88 offset and
  x1e-6 copies, six trail their rounding-only control at p < 0.05, about as many
  as chance gives and none past a correction for the 88 tests: +1e2
  ``fit6@1000`` (-4.82 against -4.79, 16/0/24), +1e2 ``ackS30@5000`` (1.05e-6
  against 6.9e-7, 11/8/21), +1e4 ``ackley10@5000`` (1.6e-10 against 2.1e-11,
  2/32/6), -1e4 ``ellR30@1000`` (5589 against 5589, 12/0/28) and x1e-6
  ``rosen30`` at 1000 and 5000 (74.1 against 74.1, 12/1/27; 22.1 against 22.0,
  14/0/26). Every x2^-20 copy repeats its plain row run for run.
- Periodic cutoff lists keep every linkcell image inside the cutoff, after
  dividing coordinates and the cell by the descriptor length scale. Where one
  direction is not periodic, the list keeps one shortest image and does not
  wrap that direction. Free-cluster shell counts use that unwrapped cutoff.
- The ChemFit bridges now raise, before ``fitter.init()``, on arguments anneal
  0.10.0 coerced or ignored without a word. A fractional ``budget`` or ``seed``
  raises instead of passing through ``int()`` (a budget of 60.5 ran 60
  evaluations), and a negative seed, or one of ``2**64`` or more, raises
  ``ValueError`` where it raised ``OverflowError`` after ``init``. A string or
  bool ``budget``, ``seed``, ``steps_per_epoch`` or ``tell_every``, a bool
  ``bound_span`` or ``default_span``, and a string one that is not a number,
  raise ``TypeError``, where 0.10.0 passed them through ``int()`` or
  ``float()``, reading ``"60"`` as 60 and ``True`` as 1, or did not read them.
  Each of these settings, and each preset keyword of ``fit_chemfit``, may be a
  ``Decimal`` or a one-element array, read as the number it holds, as 0.10.0's
  ``int()`` and ``float()`` read it; a one-element ``array_api_strict`` array
  with dimensions, which they refused, is read the same way. ``tell_every`` and
  ``steps_per_epoch`` must be positive whole numbers under every method: where
  the method uses one, 0 ran as 1 and 2.5 as 2, and where it does not
  (``tell_every`` under the classical methods, ``steps_per_epoch`` under the
  portfolio), any value ran. ``bound_span`` must be positive and finite beside
  ``low`` and ``high`` too, and ``default_span`` when the fitter bounds every
  parameter, where 0.10.0 did not read them. ``fit_chemfit`` raises
  ``TypeError`` on a keyword no preset takes, such as ``t_inti`` or
  ``tell_evry``, which it ignored, and on a preset keyword of its method that is
  not a number, such as ``True`` or ``"hot"``, which it passed through
  ``float()`` after ``init``. A ``preset_kwargs`` that is not a dict, and a
  ``preset`` that is not ``Boltzmann()``, ``Fast()`` or ``Gsa()``, raise
  ``TypeError`` under the portfolio too. ``run_benchmark`` raises ``ValueError``
  on an unknown method, where it ran the preset it was given or raised
  ``KeyError`` after ``init``. A bounds entry that is not one ``(lower, upper)``
  pair, such as a 3-tuple or a list of three per-element pairs, or that holds a
  non-numeric string, raises in ``fit_chemfit`` and ``chemfit_box``, which
  ignored it in favour of ``default_span``; a two-item list is that pair, each
  side a scalar or an array of the leaf's shape. ``fit_chemfit`` and
  ``chemfit_box`` now read bounds pairs they ignored in favour of
  ``default_span``: one with per-element sides, such as a NumPy array of two
  rows, and a NumPy pair that does not hold numbers, such as
  ``np.array(["0", "0.8"])``, which is read as ``[0, 0.8]`` with the
  ``FutureWarning`` for numeric strings. An entry that is not one pair raises in
  ``run_benchmark``'s context bounds and the ``context_bounds`` of
  ``resolve_bounds`` too, where 0.10.0 fell back to ``fitter.bounds``, and in
  ``bounds_from_fitter``, which returned ``None``. A NumPy pair there is now
  read where 0.10.0 ignored it, and so are bounds held in a mapping other than a
  dict, such as a ``MappingProxyType``, there and in ``fit_anneal`` and
  ``run_fitter``. A parameter that is not finite or holds ``None``, which 0.10.0
  read as NaN, raises in ``ChemFitVector``, in ``fit_chemfit``, whose portfolio
  ignored it, and in ``run_benchmark``, which moved an infinite start to the
  nearer bound and raised on NaN after ``init``. A complex leaf, or a NumPy
  complex number in an object array, raises instead of losing its imaginary
  part. Each ``x0`` dict leaf must have its parameter's size, apart from one
  value for the fitter's only parameter where ``fitter.bounds`` gives both of
  its sides and ``low`` and ``high`` are omitted: 0.10.0 read the leaves flat,
  so leaves of the wrong sizes with the right total moved values across
  parameters (``{"a": 0.3, "b": [0.6, 0.7]}`` started ``a`` at ``[0.3, 0.6]``).
  ``resolve_bounds`` and ``bounds_from_fitter`` raise on an infinite or NaN
  bound, a box too wide for a float, or a lower bound above the upper, and
  ``chemfit_box`` on an infinite bound or a box too wide for a float, where each
  returned the box. ``fit_chemfit`` and ``run_fitter`` now read method names in
  any case, as ``fit_anneal`` and ``run_benchmark`` did.
- With ``policy="auto"``, no ``grad_fn`` and no ``noise_sigma``,
  ``global_optimize`` runs one values-only loop whatever the box width, in place
  of the regime-routed Thompson allocation and its DE/GSA front-load. Without
  ``x0`` it starts from the best of a seeded design of ``dim + 1`` points, so
  seeds vary the start. When the budget lets it converge, five finite-difference
  gradients per coordinate, a descent from the start opens the run and goes on
  while each slice lowers the incumbent by more than the success threshold and
  by more than ``1e-4`` times the largest of the opening's last three gains,
  and while its last four slices have gained more, together, than ``1e-4``
  times its largest gain; the first slice from below the start is held to the
  gain of the slice from the start. Until that descent has
  lowered its own value from below the start's, a slice in which it stalls short
  of convergence, as when its first curvature pair spans a steep wall, or stops
  at the start's value, as behind a sentinel, gives CMA-ES one slice, once, and
  the opening goes on from the incumbent CMA-ES leaves. With less, as for a
  30-dimensional problem under 4650 evaluations, CMA-ES takes one slice from the
  start, the descent keeps the turn until a slice gains less per evaluation than
  the one before, and GSA then keeps it while each of its slices gains at least
  as fast as the descent would have gone on to (its last gain per evaluation
  times the ratio to the one before). Until GSA has played, the descent also
  hands it the turn once fewer than three slices are left beyond the closing
  reserve; if GSA's first slice falls short, the descent keeps the turn until
  the run has used twice the evaluations it had. From there GSA and then CMA-ES
  each keep the turn while they pay: a phase ends once it has gone eight slices
  (or as long as its last such gain took) without lowering the incumbent by
  more than the success threshold and by more than ``1e-4`` times the phase's
  largest gain, GSA's first phase also once its gain per evaluation since it
  began is no more than the descent's last, and a CMA-ES phase that follows it
  without taking the turn likewise from its eighth slice. From its fifth slice
  on, a GSA slice that gains less than an eighth of the phase's best per
  evaluation lends CMA-ES a slice, which takes the turn if it gains faster than
  both that slice and an eighth of the phase's gain per evaluation so far; GSA
  lends no other slice until the run has used twice the evaluations it had, and
  none after a lent slice that gained nothing. A phase CMA-ES took lends GSA a
  slice when its last two slices gain less per evaluation than its lent slice
  had to beat, and hands the turn back if GSA's slice gains faster; the two
  alternate for up to eight phases. If the bar on the first slice from below
  the start ended the opening before the descent converged and GSA's first
  phase ends at its first slice, the descent takes a turn before CMA-ES's
  phase. After an opening the phases, and that turn, stop short of the closing
  reserve plus a slice for each other arm not yet played, counted again at
  every slice, and without one short of a slice for DE while it has not played.
  CMA-ES, the descent, GSA,
  DE, the additive surrogate and the QMC restart arm then each take a turn if
  they have not yet and the budget left holds eight evaluations beyond the
  closing reserve; later rounds pick uniformly with probability ``1/round``,
  otherwise replay an arm whose last turn lowered the incumbent, and otherwise
  draw by discounted Thompson sampling. A turn is one slice, but the descent
  keeps the turn while each slice lowers its value by more than the success
  threshold, until it converges, so a kicked descent, or a long one from a poor
  start, is scored on the basin it reaches. The descent closes the run with a
  reserve of the ten gradients' worth of evaluations a descent needs to converge
  (at most 40% of the budget, and none when that leaves fewer than three
  gradients' worth), going on with its current descent unless another arm has
  lowered the incumbent since it last played, in which case it restarts there.
  A slice is four gradients' worth of evaluations or a 48th of the budget,
  whichever is larger, but at most the geometric mean of the budget and four
  gradients' worth, so past 48 times 48 four-gradient slices the number of
  slices keeps growing with the budget. The floor the restart guarantee needs is
  asymptotic in the budget: the opening, the phases and the descent's turns go
  on only while they lower the incumbent, or the descent's value, by more than
  the success threshold, whose positive floor changes at most once, so on a
  bounded objective they last a number of slices that does not grow with the
  budget, and every arm keeps a uniform share and is pulled infinitely often as
  the budget grows. Measured over 40 paired seeds against the rules they
  replaced (medians, "A against B" with A's wins/ties/losses, rows named
  ``problem@budget`` from ``x0`` as in the regime notes), these rules trail on
  the following rows, and putting a replaced rule back costs the rows named with
  it. The rate hand-over short of the opening budget, against two fixed GSA
  slices after the descent's first: ``ellS30@1000`` 1673 against 711 (7/0/33),
  ``ellS50@5000`` 6.88 against 1.52 (2/0/38), ``ellS100@5000`` 1839 against 1110
  (2/0/38), ``ellS100@20000`` 1.17 against 0.0282 (0/0/40), ``ellR100@20000``
  2.48 against 2.41 (19/0/21), ``rosen100@20000`` 74.9 against 72.0 (17/0/23),
  ``stybR30@1000`` -1057 against -1057 (4/18/18), ``levy100@20000`` 0.0416
  against 1.0e-11 (5/15/20), ``rast100_shift@20000`` 9.95 against 7.96 (9/3/28)
  and ``rastS100@20000`` 10.9 against 7.96 (8/3/29); the two slices back cost
  ``ackS50@5000`` 4.31 against 2.94 and ``ackS100@5000`` 13.2 against 5.94
  (1/0/39 each), ``ellR30@1000`` 12557 against 5589 (3/0/37), ``ellR100@5000``
  20110 against 15750 (5/0/35) and ``rosen100@5000`` 252 against 168 (7/0/33).
  The opening's bar on its recent gains, against an opening that goes on while
  each slice is a success: ``ellS30@5000`` 0.0626 against 0 (0/13/27),
  ``rosen30@5000`` 22.2 against 16.2 (2/1/37) with its copies and controls (22.0
  to 22.5 against 15.9 to 16.7), and the ``ellR30@5000`` controls for +1e2, +1e4
  and -1e4 (2.8e-11, 2.3e-4 and 8.7e-4 against 3.2e-12, 1.4e-5 and 1.2e-5);
  dropping the bar costs ``ackS30@5000`` 3.8e-6 against 6.2e-7 (1/18/21),
  ``levyR30@5000`` 3.51 against 0.544 (5/0/35) and ``levy30@5000`` 7.2e-7
  against 4.5e-10 (6/17/17). ``ellR30@5000``, 3.4e-8 against 0 (0/19/21), and
  its x1e-6 and x2^-20 copies come level only with both the opening without its
  bar and the descent's ``1e-12`` tolerance (0, 0/38/2), at the costs of each.
  Lending from a GSA phase's fifth slice: ``mich10@5000`` -9.655 against -9.66
  (3/28/9, p 0.021), which no tried rule brings level without putting other rows
  behind: lending from any slow slice puts ``mich10@5000`` without ``x0``,
  ``rast10_shift@1000`` and ``styb10_45@1000`` without ``x0`` behind, from the
  fourth slice ``rast10_fixed@1000`` and ``schwefel10_cec@1000``, a lent slice
  held to a quarter of the phase's pace ``ackley10@1000`` (1.16 against 0.0694),
  and ending a phase CMA-ES took once its slices slow ``ackS30@5000``. With
  phases that keep the turn while they still find lower basins, the fifth-slice
  lend also costs ``ackS30@20000``, 8.6e-7 against 6.3e-7 (12/1/27, p 0.048),
  and ``ackley10@5000`` without ``x0``, 4.1e-11 against 3.2e-11 (6/23/11), with
  its -1e4 copy and two controls. Rotated ellipsoids short of the opening budget
  trail an earlier form of the loop: ``ellR50@1000`` 1.38e5 against 4.79e4,
  ``ellR50@5000`` 37.7 against 9.20, ``ellR100@20000`` 2.48 against 0.633 and
  ``ellR30@1000`` 5589 against 2987; letting the descent keep the turn while it
  pays brings ``ellR50@5000`` to 17.4 and ``ellR100@20000`` to 0.890 but costs
  ``ellR30@1000`` (20880), ``ellR100@5000`` (41370), ``ackS50@5000`` (3.26),
  ``ackS100@5000`` (9.86), ``levy100@5000``, ``rosen50@5000`` and
  ``stybS100@5000``. Against 0.10.0 the loop trails on rotated Rastrigin and
  Levy: ``rastR30@1000`` 281 against 173 (3/0/37), ``rastR50@5000`` 427 against
  296 (1/0/39), ``rastR100@5000`` 893 against 650 (9/0/31), ``rastR100@20000``
  870 against 470 (1/0/39), ``levyR50@5000`` 98.8 against 69.8 (8/0/32) and
  ``levyR100@20000`` 203 against 108 (0/0/40); the diffusion population in DE's
  place and a rotation-invariant DE leave each of them behind, and CMA-ES
  restarting each phase with four times its population brings ``rastR50@5000``
  level (313) and ``rastR100@20000`` to 749 but costs ``ackS30@5000`` (2.01
  against 6.2e-7).
- ``run_fitter`` now runs the ``preset`` it is given and passes ``x0``,
  ``low``, ``high``, ``bound_span``, ``steps_per_epoch`` and
  ``preset_kwargs`` on to ``fit_anneal``. anneal 0.10.0 dropped them all, so
  the same call can give a different fit: ``method="sa"`` with
  ``preset=Fast()`` runs Fast where 0.10.0 ran Boltzmann, a classical
  ``method`` runs a preset of its own kind, and ``x0``, ``low`` and ``high``
  set the start and the box. A forwarded value that ``fit_anneal`` refuses
  raises, and any other keyword, such as ``n_epochs`` or ``tell_every``,
  raises a ``TypeError`` that names ``run_fitter``.

### Fixed

- Continuous population search does not round candidates to absolute integers
  or select integer-neighborhood jumps. Its scalar refinement uses box-relative
  opposition and continuous axial and diffusion moves, without a caller gradient.
- Occupancy ArchiveHole aims in the landfold book and quenches on a compacted surface, then names the landing by landfold.
- Occupancy Leave is a catalog draw or a walk. It is not a covering or a min-mode climb of the occupied structure.
- Scalar portfolio warmup reserves exploration work for untried active arms
  before applying preferred slice sizes. Enlarged slices cannot consume those
  shares or the protected scalar-refinement reserve.
- The ChemFit bridges ``fit_anneal``, ``fit_chemfit``, ``run_benchmark`` and
  ``run_fitter`` drive ``evaluate`` / ``step`` when the fitter has them and
  ``ask`` / ``tell`` otherwise, and refuse a fitter with neither pair before
  ``init``; only ``run_benchmark`` still drives ``evaluate`` or ``ask`` without
  its partner, with a ``FutureWarning``. On current ChemFit, ``fit_anneal``,
  ``fit_chemfit`` and ``run_fitter`` called the missing ``ask``, scored every
  candidate as ``+inf`` with the ``AttributeError`` swallowed, and handed
  ``finish`` a point that was never evaluated. The first exception the fitter
  raises, or a loss that is not one real number, now ends the fit: the fitter is
  not called again, ``finish`` is skipped, and the exception reaches the caller.
  anneal 0.10.0 read a numeric string, a bool or the real part of a NumPy
  complex number as the loss, and scored a loss it could not read, such as an
  array of two values, as the worst value. Every bridge reads a ``Decimal`` loss
  as its value, and a one-element list, tuple or array of any shape, an
  ``np.matrix`` or another library's array included, as its element. 0.10.0 read
  such a list only from ``ask`` in ``fit_chemfit`` and ``run_benchmark``, and
  such an array only where ``float()`` reads it, which for a NumPy array of one
  or more dimensions gives NumPy's ``DeprecationWarning``.
- The ChemFit bridges evaluate the start first, bit for bit, count it against
  the budget, and never exceed the budget. Every parameter leaf comes back in
  its own type: a Python number, a ``Decimal`` included, as a ``float``, a NumPy
  scalar or array with its shape and floating dtype (integer, bool and object
  leaves as float64), and a list or tuple as a list or tuple nested the same
  way. The bounds of a ``float32`` or ``float16`` leaf are rounded inward so the
  cast candidate stays inside them, and a floating leaf wider than float64, such
  as an x86 ``longdouble``, raises ``TypeError`` naming it. Bounds may be NumPy
  pairs or per-element arrays, and a bound that is not finite, is inverted, or
  is too wide for a float raises ``ValueError`` naming the parameter element. A
  parameter whose two bounds are equal is held fixed, and every argument is
  checked before ``fitter.init()``.
- The documentation of ``Ledger::charge_frac`` says what it charges. Fractions
  accumulate and are charged as a whole call each time their sum crosses one;
  a residue under one call is never charged, so a ledger can report up to one
  call less than the work done, mid-run and at the end, and a charge that runs
  into the budget leaves the rest of its work unpaid. It had said the residue
  was still owed when the run ended, and nothing collected it.
- The portfolio population-diffusion arm applies shared peer geometry to global
  proposals before evaluation and stores the corrected walker coordinates.
  Scalar objectives need no gradient callback; local refinement remains raw.
  Private and zero-height controls bypass the proposal adapter.
- With ``replicas`` above one, every rung of the cluster-hopping ladder hopped
  at ``temperature``, so the ladder's temperatures reached only the swap test.
  Rung ``k`` of ``R`` now hops at ``ladder_top^(k/(R-1))`` times the temperature
  a single chain would hop at. Under ``budget_window`` the law sets the coldest
  rung's temperature, and a rise a rung declines enters the barrier estimate
  divided by that rung's ratio. Under ``statistical_temperature`` the estimate is
  clamped to its band around ``temperature`` before the ratio multiplies it, and
  only the rungs at ratio one feed the density of states it is read from. The
  swap factor reads each rung's temperature from the state it holds and weighs
  the funnel bias, the packing pile and the energy bias every rung shares by the
  difference of the two inverse temperatures. Under ``flat_histogram`` it weighs
  the biases alone, since the flat-histogram cost is the same on every rung and
  cancels from the factor with the energies. Under ``energy_bias`` the bias's
  tempering factor and its deposits read the temperature the coldest rung would
  hop at from the state it holds, so which rung fills its first sample or
  deposits no longer sets them. A state keeps its basin and its validation
  gradient as it moves between rungs, so the first step a rung takes after a
  switch is recorded from the state that rung holds, and every rung's starting
  quench is validated before it is recorded. A ladder refuses
  ``delayed_acceptance``: a hop its surrogate decides is tested on the bare
  energy and one it abstains on with the biases, so no swap factor can balance an
  exchange between rungs that hop by two weights. A single chain is unchanged.
- ``Reflected`` mirrors a proposal whose offset or sum overflows ``f64``
  instead of placing it on a wall. The step comes from
  ``MoveKernel::displacement``, which ``Gaussian`` and ``Cauchy`` implement.
  ``run_rs_qmc_variant`` starts its chains on the finite wall of an axis with
  an infinite wall. In the Rust sampler a NaN objective value counts as worse
  than every number, ``+inf`` included, so a chain no longer stalls at a NaN
  start and a NaN never beats a number. A Python objective that returns NaN
  is still scored as ``+inf``, the same value an ordinary exception receives.
  The unbounded ``boltzmann``, ``fast`` and ``gsa`` constructors stay on
  ``R^dim``; the boxed presets are ``boltzmann_in_box``, ``fast_in_box`` and
  ``gsa_in_box``.
- ``global_optimize``, and with it the default ChemFit portfolio, no longer
  panics on a box narrower than about ``2e-7`` in some coordinate. The gradient
  arm bounded its central-difference step with ``clamp(1e-8, 0.05 * width)``,
  whose floor exceeds its cap on such a coordinate; the step now stays within a
  twentieth of the width.
- ``methods::cluster_search::median_encounter`` decides the Kaplan-Meier median
  in exact integer arithmetic. The survival is exactly a half once half the runs
  are found before any is censored, and the floating-point product it replaces
  rounded above a half at some even run counts, 24, 28, 30 and 38 among them, so
  the median came back one encounter late: the sixteenth first encounter of
  thirty instead of the fifteenth. A run censored at the time of an encounter
  is still at risk at it, so the median no longer depends on the order the runs
  are given in.
- ``occupancy_retire_at`` honours leftover dwell unless the packing
  book has one family, matching ``Hop.leftover_unsaturated_does_not_retire``.


## [0.10.0](https://github.com/HaoZeke/anneal/tree/v0.10.0) - 2026-10-09

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
  is not this map. A raw-\(E\) polish and leftover-SOAP requench are
  projectors onto the occupied packing and do not run on that walk.
- A packing is a single-linkage community of DECAF book cells at
  ``PACKING_LINK`` \(=0.35\), and the cell grain ``PACKING_MERGE``
  \(=0.20\) names a cell rather than a packing. Measured on quenched
  isomers within \(8\varepsilon\) of each icosahedral floor
  (``examples/decaf_packing_separator``): the LJ75 shelf reaches L1
  \(0.56\) from the ico reference while ico-Marks is \(0.4267\), so no
  radius around one reference separates them, while under single linkage
  the shelf chains into one community and Marks stands alone from
  \(0.10\) through \(0.40\). One atom carries \(2/N\) of the
  histogram, so the same L1 is a tighter grain on a smaller cluster: the
  LJ38 shelf chains 148 of 154 at \(0.30\), 153 at \(0.35\) and 155 at
  \(0.40\), with Oh alone throughout at an ico-Oh L1 of \(1.1579\).
  \(0.35\) is the middle of what both sizes allow. Leave adopts on that
  community, OtherFamily draws on it, and the certificate counts live
  packings rather than cells.
- The packing angular band stays at ``l_max`` 3 with ``PACKING_LINK``
  \(0.35\). On 72 LJ75 points (70 shelf isomers within \(8\varepsilon\)
  of the icosahedral floor, plus the two references; seed 20260821)
  the book gap is \(0.4267\) at ``l_max`` 3, with Marks alone at link
  \(0.35\) (71 of 72 in the icosahedral component) and the join at
  \(0.45\). At ``l_max`` 4 the gap is \(0.4533\) and the join is at
  \(0.50\). At ``l_max`` 6 the gap is \(1.6267\), Marks stays alone
  through radius \(1.20\), and the link \(0.35\) component holds 64 of
  72. On 159 LJ38 points (157 shelf isomers plus the two references)
  the same link holds 152 of 159 at ``l_max`` 3 and 60 of 159 at
  ``l_max`` 6. ``DECAF_LMAX`` selects the band in
  ``examples/decaf_packing_separator``. Production ``PACKING_SPEC``
  does not change. The class-histogram icosahedron-Marks distance
  \(0.6933\) is a different codebook.
- The Leave start is a rung of a ladder in the packing map
  (``known_basin::leave_packing_rung``): a covering direction of the
  DECAF feature with the \(\mu\) component stripped, pointed away from
  the packings on file and pulled back through \(J_\mu\). A rung whose
  quench lands back in the same packing walks the next rung
  (``known_basin::leave_packing_ladder``) with the invert armed, instead
  of drawing another hole of the same size. Wales, D. J.; Doye, J. P. K.
  *J. Phys. Chem. A* **1997**, *101*, 5111
  <https://doi.org/10.1021/jp970984n> put the LJ75 ico-Marks barriers at
  8.69 and 7.48 \(\varepsilon\), which a Cartesian 0.35 cap cannot reach
  in any direction.
- Occupancy Leave climbs the covering or ladder start up the local
  ridge (Quapp gradient extremal / Barkema--Mousseau ART / SoftSaddle
  MMF via [`activation`]) until the force along the mode flips, then
  quenches from the overshoot. A packing-map kick plus a quench is
  still inside the occupied well; the ridge walk is the crossing.

- New ``anneal.chemfit.fit_anneal`` drives the gradient-free optimizers from a
  ChemFit fitter through ``init`` / ``ask`` / ``tell`` / ``finish``. The default
  driver is the Thompson-allocated portfolio. ``boltzmann``, ``fast``, and
  ``gsa`` are the single-chain ablations. Nested parameter dictionaries are
  flattened to vectors and rebuilt on return. The fitter's initial parameters
  are the start unless ``x0`` is passed, and every evaluation stays in the box.

- New ``anneal.chemfit.fit_chemfit`` and ``ChemFitVector`` flatten a fitter
  with dotted keys, build a finite box from ``fitter.bounds`` (unbounded
  parameters fall back to ``init +/- default_span``), and close the run with
  ``fitter.finish``. ``run_benchmark`` accepts the same session when the fitter
  speaks ``evaluate`` / ``step`` instead of ``ask`` / ``tell``.

- ``CooperativeRun::flush`` waits for a replica's posted work to reach the
  coordinator, and ``CooperativeRun::set_on_published_prize`` reports a
  published reference the coordinator cannot know about.

- Inverted Gelman--Rubin on the cooperative catalog: explore-role chains
  that mix are forced to leave, and a sampled incumbent is marked dominant
  only when occupant chains have mixed onto it and it is strictly more
  occupied than every observed competing packing. This is a convergence
  diagnostic, not a global-optimality certificate.

- Occupancy visit floor, hatch-stable dwell, Cheeger Fiedler
  witness, and Gelman--Rubin certificate length are derived
  bounds with Lean and SymPy identities, not free constants.

- Occupancy retires only on a compact putative: one connected
  first-neighbour component with a cycle. A path or a split
  fragment is not a cluster.

- Leftover dwell uses the Esty (1983) upper bound. Occupant
  mixing is Vehtari split-R-hat at 1.01. Leave accepts a new
  Franzblau ring class as well as a new DECAF family.

- Symbolic identities for inverted Gelman--Rubin certification, the
  Good--Turing occupancy stop, keep-fraction ranking, Chatterjee--Voter
  frequent-hop refusal, and same-packing Exploit versus different-packing
  Explore.

- Live catalog roster with Attach/Detach/Tick/Scale, a deterministic
  coordinator clock, and successive-halving spawn/retire decisions.
  The supervisor launches workers for pending spawns; workers Attach
  themselves and honour a retired policy reply.

- Same-system cooperative searches enact Raft-committed spectral segments by
  retrieving one coordinator-validated basin anchor per changed assignment;
  stale replies and repeated snapshots cannot restart a live chain.

- Two-phase relaxation on a molecular library penalizes group centroids
  rather than atoms, so the diameter and compression terms cannot squeeze
  intramolecular bonds.

- ``global_optimize``, ``global_optimize_objective``,
  ``qmc_gsa_global_search`` and ``qmc_gsa_global_search_objective`` take
  ``x0``. The portfolio charges it as its first evaluation and records it as
  the first incumbent, so arms that read the incumbent, such as the
  trust-region poll, HMC and the population arm, start from it until a lower
  point is found; the GSA search uses it as its first chain's start. A
  starting point outside the box, of the wrong size, or not finite raises
  ``ValueError``. While no evaluation is finite the incumbent is the first
  point evaluated, so ``global_optimize`` returns ``x0`` rather than a point it
  never evaluated. ``portfolio_optimize_from`` and
  ``qmc_gsa_global_search_from`` are the Rust entry points.

### Changed

- The cooperative catalog speaks Cap'n Proto RPC. A replica attaches
  with a subscriber capability, calls through a session bound to its
  identity, and receives epoch-close, roster, and retire events. The
  existing CatalogRequest/CatalogReply payload, journal, and operation
  handlers are unchanged. Observe needs no identity.

- Plain-exchange population replacement anneals its distance cutoff from
  half the mean pairwise distance of the first bank to one fifth, and
  draws mix partners from that frozen bank.

### Fixed

- A shared catalog checkpoint requests one observed boundary crossing
  on the probe interval. The proposal is the aligned displacement at
  the parsed transport noise and radius. A private catalog, an empty
  region, a rejection, and a transport that does not fit keep the fixed
  probe. The recorded destination is not adopted.
- ``SurfacePortfolio::sharing`` records the occupied source before the
  shared book is attached. The draw and the block credit use that
  source. A book with no source is not attached.
- Occupancy extras idled against a one-packing book. A Leave trades a
  replica's walk for coverage, and that trade is only worth making when
  the Leave has somewhere to go: with one community on the book the
  ArchiveHole it draws has no measured yield, since a packing-ladder rung
  sized by bisection to spend exactly its barrier quenches back to the
  floor it started on at every rung from \(1.32\) to \(42.3\)
  \(\varepsilon\), five times the LJ75 ico-Marks barrier in one
  displacement. ``OccupancyLeaveTarget::Walk`` is what a one-packing book
  returns, and the extra keeps walking with the shared bias holding it off
  the basins the others are on. The walk is what crosses: 3 of 64
  independent LJ75 walks at 400k evaluations reached \(-397.492331\), at
  hops 4160, 6226 and 4411 of about 11000.
- A Leave fired on the checkpoint slice, which is 500 hops against a
  crossing that costs of order four thousand. It now waits out the longest
  run of quiet checkpoints the replica has itself come back from, which is
  that replica's own history rather than a constant.
- The packing census certified completeness from a single merged bin,
  where Chao1 has \(n_1=0\) by construction: a run reported
  ``packing_sat`` and ``stop`` while the raw book carried \(n_1=99\) and
  \(\hat p_0=0.54\). Communities now come from single linkage rather than
  a forced 2-means bipartition, and a one-community book is answered by
  the Good--Turing sample of the cells it folds.
- ``arm_leave`` read its references from the leftover-SOAP archive, which
  holds cloud means rather than structures, so the length check dropped
  every one and the Householder inverted nothing but the well it started
  in. References cross as structures.

- ``run`` and ``run_qmc`` keep every evaluation inside ``[low, high]`` and take
  ``x0``. The ``Boltzmann``, ``Fast`` and ``Gsa`` presets now pair their
  Gaussian, Cauchy and Tsallis moves with mirror reflection into the box and the
  box neighbourhood (``boltzmann_in_box``, ``fast_in_box``, ``gsa_in_box``), a
  pairing the composition laws certify. Before, ``low`` and ``high`` only drew
  the start, and a 13-atom walk in a ``[-3, 3]`` box evaluated 99.95% of its
  points outside the box. ``run`` starts its walk at ``x0`` and ``run_qmc``
  uses it as the first start; a starting point outside the box, of the wrong
  length, or not finite raises ``ValueError``, and so do invalid preset
  parameters, which used to panic. ``low``, ``high`` and ``x0`` may be any
  array-like of one shape, an ``(n_atoms, 3)`` positions array included; they
  are flattened in C order and ``obj_fn`` receives the flat vector.

- Python callbacks no longer fail silently. ``KeyboardInterrupt`` and
  ``SystemExit`` raised inside ``obj_fn`` or ``grad_fn`` end the run and are
  re-raised when the driver returns, where they used to be scored as ``+inf``
  while the run went on. A return value that is not a number raises
  ``TypeError``. An ordinary exception is still scored as the worst value, so
  budget counters can stop a driver by raising, and the driver now reports how
  many were scored, with the first message, as one ``RuntimeWarning``. A NaN
  objective value is scored as ``+inf``, and in ``run`` and ``run_qmc`` two
  infeasible points count as level, so a walk started where the objective is
  NaN or ``+inf`` moves until it finds the feasible region instead of freezing
  there. ``cluster_search`` follows the same rules for exceptions and return
  values; its hop chain still skips quenches whose energy is not finite.
  Gradients may be returned as any array-like (``jax.grad``, torch, lists).

- Universal descriptor schema version 2 applies the nonexpansive map
  ``x / sqrt(1 + ||x||^2)`` to every dimensionless block. Near-zero chiral
  signals retain their amplitude instead of being inflated to unit scale, while
  large radial, angular, graph, SOAP, and ACE blocks remain bounded.

- Family extras of a crowded DECAF packing Leave it. The packing
  champion stays and walks isomers.

- ``global_optimize`` only calls ``grad_fn`` inside the box. A point an arm
  proposes outside is mirror-reflected before the objective sees it, and the
  gradient is now taken at the same reflected point, its sign flipped along
  every coordinate the fold reverses, which is the gradient of the function
  the portfolio actually evaluates. Before, the gradient was called at the
  unreflected point, up to 0.34 outside a ``[-3, 3]`` box. No arm hands the
  caller's objective or gradient a non-finite coordinate: the population arm's
  success-history mean weighed a success from a walker at ``+inf`` as
  infinite, its scale factors became NaN, and up to 15% of a run's calls went
  to NaN points when part of the box was infeasible.

- An occupancy Leave that keeps the unquenched hole step records that
  geometry's own energy, updates the basin the chain stands in, and is
  reported as unvalidated rather than validated.

- A TIS extra seated on a leftover-SOAP interface Leaves once its path
  maximum crosses that interface, and interface seats persist across
  policy requests so the RETIS exchange and the one-sided promotion have
  an ordering to repair. Leave-path frames carry their own lambda rather
  than the replica's running maximum, so the shoot returns to the frame
  furthest from the occupied well.

- Occupancy Leave waits for the measured Marks crossing floor before interrupting a quiet walk.

- Occupancy retire honors leftover-SOAP dwell unless the packing
  book holds one family. MixingCertified plus packing saturation
  does not retire extras while leftover is still hatching.

- Occupancy extras walk while leftover SOAP is unsaturated, so Marks is a long chain not a Leave.

- MixingCertified on a two-family book waits for leftover-SOAP
  hatch-stable dwell. Mixed ico plus a mixed shallower competitor
  is not MixingCertified on that pair alone.

- Occupancy workers skip policy RPC during the Leave crossing floor and print hops per core-hour.

- Occupancy Leave takes a catalog representative of a different DECAF
  family, or a hole of the shared occupied-packing archive. Occupancy
  extras do not draw a random cluster.

- Occupancy OtherFamily walks when FunnelModel EI on seen packings is exhausted.

- A packing-book query can no longer credit a visit. Histograms built for
  a read fold unseen environments into one shared bin, and taking one on
  the visit path merged structurally distinct packings into a single
  family.

- Mirror reflection folds a box whose period ``2 (high - low)`` overflows,
  and a box with one infinite wall, across the wall the point crossed.
  An axis with equal bounds stays at that value. The Python drivers still
  refuse a box whose width is not finite.

- Reflection into a box returns a point already inside unchanged, so a point
  on the upper wall is no longer folded to ``high`` plus an ulp and evaluated
  outside the box. The Tsallis visiting step is formed from logarithms: near
  ``q_v = 3`` its scale underflowed to zero while ``|y|^{-e}`` overflowed, the
  product was NaN, and ``Gsa`` walks stalled with fewer evaluations than the
  schedule promised; below ``q_v`` of about 1.007, ``Gamma`` in the scale
  overflowed and every step was zero or a tail redraw. ``run`` and ``run_qmc`` take ``max_evals`` to spend an
  exact number of objective calls.


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
