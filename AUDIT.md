# Search-code motif, size, and species audit

Scope: every Rust source file under `src/` at
`bc16d35cf13e71c2edaf343b80fd8c41368631d4`. I searched identifiers, comments,
numeric comparisons, named constants, preset tables, move selection, catalog
admission, candidate construction, and descriptor consumers. Test-only
fixtures were checked but are not search behaviour.

Classifications:

- **Behaviour**: changes a proposal, parameter, admission, identity, restart, or
  candidate from a size, species, named motif, template, or builder.
- **Diagnostic only**: labels or reports a structure without steering search.
- **General geometry**: a cardinality precondition or a construction derived
  only from the current coordinates. It is listed for review but is not a named
  motif/size tuning branch.
- **Physical model**: constants defining the requested objective, not search
  policy.
- **In-flight removal**: behaviour already assigned to one of the two named
  follow-up changes. This audit does not edit those paths.

## Findings

| Location | Hit | Classification | Proposed general replacement |
| --- | --- | --- | --- |
| `src/methods/cluster_hopping/config.rs:1538`, `:1593` | `n_points < 98` enables two surface modes. | **In-flight removal / behaviour** (`bc-a84a44f0`). | Activate from measured surface participation or move acceptance, independent of atom count. |
| `src/methods/cluster_hopping/preset.rs:11` | `ALL_POINTS_STEP = 0.38` is justified by an LJ38/Wales–Doye band, although it is applied at every size. | **Behaviour**, but not an atom-count branch. | Retain only as a dimensionless length-scaled default or adapt it from acceptance; do not add per-size rows. |
| `src/methods/cluster_hopping/moves.rs:229-232`; `src/methods/cluster_hopping/config.rs:2361-2365` | A lone flexible group of at least four atoms disables rigid group moves and SOAP. | **Behaviour** from a numeric group-size comparison. | Decide from declared flexible degrees of freedom and measured move cost, not group cardinality four. |
| `src/soap.rs:1208`, `:1253`, `:1257` | `ACTIVE_VOLUME_MIN = 8` fixes the minimum active region. | **In-flight removal / behaviour** (`bc-250d428e`). | Select active support from descriptor sensitivity or a cumulative displacement/uncertainty target. |
| `src/soap.rs:1346`, `:1348` | FCC and icosahedral `Template::points()` are SOAP prototypes. | **In-flight removal / behaviour** (`bc-250d428e`). | Learn prototypes from encountered environments or compare directly in descriptor space. |
| `src/soap.rs:1459-1461`, `:1671-1762`, `:2290-2668` | Fivefold plateau, axes, probes, residuals, openings, and steps; size gates at `n < 7`, `n < 5`, `n > 13`, and `n <= 13`. | **In-flight removal / behaviour** (`bc-250d428e`). | Generate directions from generic descriptor gradients, unstable modes, or under-sampled local environments. |
| `src/soap.rs:1459-1499` | `fivefold_plateau`, `extended_scale`, and `motif_fractions` have no production callers. | **Dead motif API**, not current runtime steering. | Delete them or expose them only through an explicitly diagnostic interface. |
| `src/soap.rs:247-257` | Atomic-number branches provide hard-coded covalent radii. | **In-flight removal / species behaviour** (`bc-250d428e`). | Accept pair scales or a bond graph as problem data; do not map elements inside search. |
| `src/methods/cluster_hopping/moves.rs:248` | A group count above the water-hexamer constant selects `large_water` behaviour. | **Behaviour**. | Use measured rigid-body extent, cost, or acceptance statistics rather than a named system size. |
| `src/methods/cluster_hopping/moves.rs:1538-1559` | A 97-element covalent-radius table controls active regions and connectivity. | **Species behaviour**. | Require caller-supplied radii, pair scales, or connectivity. |
| `src/methods/cluster_hopping/moves.rs:3145-3167`; `src/methods/cluster_hopping.rs:3679-3694` | A declared diameter ratio of `1.2` changes species swaps, homotop escape, and exchange paths. | **Composition behaviour**, but uses caller-supplied scales rather than element IDs. | Make the ratio a problem/config parameter or infer it from exchange acceptance. |
| `src/methods/cluster_hopping/moves.rs:2650`, `:2658`; `src/featomic_hop.rs:1313` | Move implementations call the fivefold-specific SOAP step. | **In-flight removal / behaviour** (`bc-250d428e`). | Call a generic descriptor-gradient escape move. |
| `src/methods/cluster_hopping.rs:8010-8012`; `src/catalog/packing.rs:1370-1401`; `src/catalog.rs:50`, `:83-89` | An icosahedral PTM share and fixed fivefold window select a channel; `ICOSAHEDRAL_TEMPLATE` hard-codes the PTM slot. | **In-flight removal / behaviour** (`bc-250d428e`). | Allocate channels by observed transition/acceptance evidence over unnamed descriptor regions. |
| `src/catalog/leave_learn.rs:29`; `src/known_basin.rs:2162-2173` | A dedicated fivefold leave arm selects a fivefold move. | **In-flight removal / behaviour** (`bc-250d428e`). | Allocate among generic learned escape directions. |
| `src/methods/activation.rs:231-288`, `:1463`, `:1573`, `:1939-1944`, `:2582`, `:2790`, `:2895-2914` | Activation adds fivefold openings and regularizes/quench-selects pentagons. | **Behaviour**. | Form candidates from Hessian/force residual modes and generic local-symmetry error, without a fixed polygon order. |
| `src/csm.rs:87-158`; `src/methods/cluster_hopping/config.rs:1552-1560`, `:1595-1599`; `src/methods/cluster_hopping.rs:6162-6227` | The continuous-symmetry move chooses among fixed `C_i`, `C_5`, and `D_5h` groups and projects the candidate; both stock presets enable it. | **Behaviour** (fixed fivefold groups and structure projector). | Supply groups as experiment input or derive candidate operations from symmetries measured on the incumbent. |
| `src/continuous_symmetry.rs:20-67`; `src/methods/cluster_hopping.rs:6179` | The alternate continuous-symmetry move constructs an inversion-symmetrised candidate for the published LJ38 benchmark. | **Behaviour** (fixed-group structure projector). | Use detected incumbent symmetry or generic residual modes rather than a built-in benchmark group. |
| `src/catalog/occupancy.rs:1267-1301` | Five-ring incidence receives extra displacement weight. | **Behaviour**. | Weight atoms by generic ring novelty, local strain, or learned leave success without preferring ring order five. |
| `src/corekey.rs:303-415`; `src/coreclass.rs:1-11`, `:70-139`; `src/catalog_rpc/mod.rs:390` | Fixed 555/421/422 counts and fivefold-density thresholds define motif classes that key cooperative restart policy. | **Behaviour**. | Cluster generic continuous local descriptors online and key restarts by data-derived regions. |
| `src/methods/dls_published.rs:86-188`, `:218` | Published Table 1 selects move count, primary sites, starts, and retained candidates from atom-count breakpoints 49 through 309. | **Behaviour** (published-baseline reproduction). | Put reproduction values in external experiment input; use budget, acceptance, and coverage criteria in the general search. |
| `src/methods/dls_published.rs:386-450`, `:872-934` | The published DLS path builds primary sites from Mackay/icosahedral shells. | **Behaviour** (named lattice template). | Cover directions inferred from the current structure or a generic hypersphere design. |
| `src/methods/gmin_hop.rs:149-179` | The TIP4P preset hard-codes water move scales, temperature (`2.9` below 18 molecules and `2.2` otherwise), quench/pool tolerances, patience, container, and separation. | **Species/system behaviour** (published-baseline reproduction). | Move reproduction values to external experiment input; adapt generic search controls from acceptance, cost, and geometry. |
| `src/bank_rpc/server.rs:133` | `kmax_factor` changes at `n >= 55`. | **Behaviour**. | Derive the IRA search bound from geometric convergence/uncertainty or use one size-independent setting. |
| `src/methods/cluster_hopping.rs:3214-3215`, `:3602`, `:7893`, `:7930` | `RIGID_HOT_BODIES = 10` switches rigid temperature and hot/cold handling. | **Behaviour**. | Adapt from measured rigid move cost and acceptance, not body count. |
| `src/catalog/packing.rs:68`, `:483`, `:507`, `:1028`, `:1155`; `src/known_basin.rs:959` | `MINIMUM_PACKING_ATOMS = 13` changes descriptor availability and known-basin identity. | **Behaviour**. | Detect descriptor sufficiency from available neighbours and fall back to a size-independent metric. |
| `src/catalog/molecular.rs:117`, `:130-136`, `:173-188`, `:461-529`, `:583-705`; `src/methods/cluster_hopping.rs:4422`, `:5504-5524` | Water geometry, species order, hexamer size, named isomers, budgets, rewiring prior, and water-specific basin gates live in or feed the catalog search preset. | **Species/system behaviour**. | Move the molecule schema, seed geometry, campaign budget, priors, and validation rules to external problem/experiment input. |
| `src/methods/surrogate_level.rs:95-186` | The level search reconstructs TIP4P and keys memory only by oxygen coordinates. | **Species behaviour**. | Supply body grouping, representative sites, reconstruction, and key functions through the problem interface. |
| `src/methods/cluster_search.rs:1321-1338` | The generic reporting path recognizes `WATER_POLISH_EXTERNAL` and skips polishing for the water driver. | **Species/system behaviour**. | Replace the water-named environment branch with a caller-supplied polishing policy. |
| `src/structure.rs:543-644`, `:991-1021` | `Template::points()` contains ideal FCC/HCP/icosahedral/simple-cubic coordinates; PTM maps structures to those names. The labels are diagnostic, but the point templates are consumed by builders and SOAP. | **Behaviour plus diagnostic use**. | Retain labels only behind diagnostics; remove ideal coordinates from search and use observed neighbourhoods. |
| `src/lattice.rs:45-92`, `:151-166`, `:243-277`, `:292-463`; `src/methods/cluster_hopping/moves.rs:2049` | `Source::library`, `decahedral_cut`, pentagonal shells, growth, and candidate construction build named lattice structures and are exposed as a hop move. | **Behaviour** (template and structure builder). | Restrict proposals to perturbations/regrowth derived from the incumbent's measured local geometry. |
| `src/construct.rs:40-56`, `:142-218`; `src/methods/cluster_hopping.rs:4456` | A posterior chooses among `lattice::Source::library()` structure-building recipes. | **Behaviour** (structure-builder selector). | Learn over generic current-state perturbations, not complete constructed candidates. |
| `src/methods/lattice_search.rs:105-415`, `:433-710`; `src/methods/cluster_hopping.rs:5536` | Hollow-site generation, occupation optimization, reoccupation, and successive placement construct full candidates. | **Behaviour** (geometry-derived structure builder). | Use the same geometric scores for bounded local moves on the incumbent rather than rebuilding it. |
| `src/methods/cluster_hopping/general.rs:188-237`; `src/methods/cluster_hopping.rs:6466` | `regrow_by_energy` starts from a simplex/tetrahedral holes and successively rebuilds a structure; `n < 6` is its feasibility gate. | **Behaviour** (general structure builder; the size check itself is a geometric precondition). | Reseed/perturb a subset of existing coordinates using measured local voids without constructing the whole cluster. |
| `src/symmetrise.rs:862-1118`, `:1355-1515`; `src/packing.rs:22-73`; `src/methods/cluster_hopping.rs:8441-8558` | Core symmetrisation and orbit completion detect a point group, project atoms, and construct completed candidates on new basins. They are motif-independent but still structure builders. | **Behaviour** (geometry-derived structure builder). | Use detected symmetry only as a diagnostic/continuous feature, or perturb local residuals without completing a whole structure. |
| `src/methods/cluster_hopping.rs:8715-8752` | Stall symmetrisation tries fixed orders `[2, 3, 4, 5, 6]`; the nearby design comment promises an icosahedral/555 withholding gate that the runtime conditions do not implement. | **Behaviour / documentation drift**. | Select detected operations from generic residual evidence and make any exclusion an explicit, tested predicate. |
| `src/methods/splice.rs:35-155`; `src/methods/cluster_hopping.rs:9619-9753` and their callers | Plane splice builds a child only from two parent structures; random starts/restarts draw unlabeled points from a radius/density rule. Neither contains a motif, lattice, element, or benchmark-size choice. | **General geometry / initialization**, not a hidden template. | No replacement needed under the motif rule; if “structure builder” is meant literally for every full-coordinate proposal, these APIs must also move outside search. |
| `src/packing.rs:77`, `:106`; `src/twin.rs:347-350`, `:529-532` | Four-point surface and eight-point two-layer minimum guards. | **General geometry**, not benchmark-size tuning. | No replacement needed; keep the reasoned allowlist if these generic moves remain. |
| `src/twin.rs:318-585` | Layer slides and stacking-fault slips use detected planes, not a named lattice template. | **General geometry**; motif language is descriptive. | No replacement required; a stricter policy could use generic local strain modes. |
| `src/structure.rs:1008-1021`; `src/methods/cluster_hopping.rs:7609-7629`, `:8319-8344` | PTM FCC/HCP/icosahedral fractions and trace counters report morphology. | **Diagnostic only**; trace comments state that no downstream code steers by them. | Keep isolated from move/config decisions and explicitly allowlist. |
| `src/morphology.rs:160-167` | Ideal icosahedral/FCC/HCP/simple-cubic descriptor coordinates. Production descriptors are generic; these named values are reference/reporting constants and test expectations. | **Diagnostic only**. | Keep out of search configuration; compare generic descriptor coordinates when steering bias. |
| `src/shape/mod.rs:35-48`; `src/symmetrise.rs:704-707`, `:858-861`, `:1212-1214`; `src/sym_hessian.rs:38-57`, `:287-290` | Generic symmetry measures/detection mention fivefold or icosahedral examples, but operate on supplied/detected operations; named structures otherwise occur in tests. | **General geometry / diagnostic**. | No replacement needed while search does not select a named operation from these labels. |
| `src/potentials/*.rs` element, TIP4P, Sutton-Chen, Gupta, and pair-potential constants | Parameters define an explicitly requested objective rather than select search policy. | **Physical model**, outside this rule. | Pass a different potential to change the physics; do not duplicate these constants in search moves. |
| `src/binary_water.rs:33-136` | Species IDs, pair scales, and rigid geometry are supplied from the selected binary/TIP4P problem to generic configuration. | **Problem adapter**, not a hidden species branch. | Keep values caller/problem supplied; avoid moving them into generic moves. |
| `src/methods/cluster_hopping/config.rs:187-188`, `:978-998`, `:1557-1560`, `:1596-1599`, `:1934-1938` | `prior_reseed`, `prior_growth`, `prior_core`, `prior_handover`, `prior_motif`, and `ContinuousSymmetry::Measure::reseed_hops` are motif-named/defaulted configuration debt with no hop-loop reads; only the interval is consumed. | **Inert configuration / documentation drift**. | Remove unused fields or wire a replacement only to generic measured stall evidence. |

## Added by the review of the lint

| Location | Hit | Classification | Proposed general replacement |
| --- | --- | --- | --- |
| `src/assignment.rs:23`, `:38`; `src/csm.rs:366` | `SHORTEST_PATH_FROM = 98` picks the assignment solver by matrix size, and the continuous-symmetry projection repeated the literal `98`. The constant has no size word in its name, so the first scanner missed both. | **Behaviour** (solver crossover, size-keyed). The duplicated literal in `csm.rs` is removed: it now asks `assignment::uses_shortest_path`. | Time both solvers on the first projection and choose by measured cost. |
| `src/potentials.rs:247-256` | `ccd_morse_minimum` keys published Morse minima by `(rho, n_points)` match arms over 38, 55, 75. Only examples read it, for hit reporting. | **Diagnostic only** (reference data). | Move the table to the benchmark input. |
| `src/methods/dls_published.rs:87` | `ROWS` is a table keyed by atom-count breakpoints; the first scanner saw no comparison against a literal. | **Behaviour** (published-baseline reproduction), as the DLS row above. | As above. |
| `src/methods/cluster_hopping/moves.rs:248`; `src/methods/cluster_hopping.rs:3602`, `:7893`, `:7929`; `src/catalog/packing.rs:483`, `:507`, `:1026`, `:1155`; `src/known_basin.rs:959`; `src/soap.rs:1251`, `:1257` | Readers of `WATER_HEXAMER_MOLECULES`, `RIGID_HOT_BODIES`, `MINIMUM_PACKING_ATOMS` and `ACTIVE_VOLUME_MIN` compare an atom count with the constant, so they carry the same size switch. | **Behaviour**, as the declarations above. | As above. |
| `src/methods/cluster_hopping/moves.rs:1542`; `src/soap.rs:249` | A 97-entry covalent-radius table and a six-arm match on atomic number. | **Species behaviour**, as the table row above. | Require caller-supplied radii. |
| `src/history.rs:101`, `src/methods/portfolio.rs:4403`, `src/methods/cluster_search.rs:2101`, `src/methods/bayesian_pilot.rs:286` | Counters named `n` compared with or fixed to a literal of five or more. | **Lookalike**, not a violation. The variables are renamed (`samples`, `archived`, `observations`, `grid`) so they no longer need an allowlist entry. | None needed. |

## Scanner policy

`tests/no_motif_constants.rs` lexes production Rust under `src/` and
`crates/` and drops comments. Items under `#[cfg(test)]`, `#[test]` and
`#![cfg(test)]` are skipped by brace matching, so code after a test module is
still scanned. The remaining code is cut into statements at `;`, `{` and `}`
outside brackets, so a constant, comparison or match arm split over several
lines is one statement. A statement is a hit when it:

1. names a motif (fivefold, icosahedral, decahedral, Mackay, Marks, Leary,
   FCC, HCP, BCC, pentagon, `d5h`, motif classes), including camel-case and
   snake-case spellings;
2. embeds a benchmark size in an identifier (`lj38`, `ico55`, `known_n104`);
3. spells a benchmark cluster size (13, 38, 55, 75, 98, 104, 147, 150) beside
   a comparison, a range, a match arm, an assignment, in a constant, or twice
   in one list, in any integer form (`0x26`, `38usize`, `3_8`);
4. compares an atom, point, molecule or body count with a literal of five or
   more, on either side, or fixes such a variable to one;
5. declares an integer constant whose name carries a size word, a table of
   integer-keyed rows, or a table of forty or more numbers;
6. is a match arm keyed by an integer when the scrutinee is a size or an
   atomic number; or
7. reads a constant that rules 3 or 5 flagged, wherever it is declared.

Python is scanned under `python/` with all of these rules, and under
`scripts/`, `experiments/` and `bench/` for benchmark-size defaults and
comparisons (`n=38`, `n == 98`, `N = 75`). Test files are not scanned.

Every hit must be claimed by an entry that names a path, a needle, a statement
count and a digest of the claimed statements. Editing, adding or removing a
claimed statement changes the digest and fails the test, so a broad needle
cannot hide a swap. An entry that claims nothing is stale. The two in-flight
removals must delete their entries when they land.

The following lookalikes were deliberately excluded: dimension and empty-input
checks; two-/three-/four-point geometric validity checks; shift amounts such
as `s << 13`; physical-potential constants; and motif words confined to
comments or test-only items. Counters that were named `n` were renamed
instead of allowlisted.
