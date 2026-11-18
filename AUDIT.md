# Search-code motif, size, and species audit

Baseline audited: every Rust source file under `src/` at
`bc16d35cf13e71c2edaf343b80fd8c41368631d4`. Resolution annotations reflect
the no-size-constants series applied above that baseline. I searched
identifiers, comments, numeric comparisons, named constants, preset tables,
move selection, catalog admission, candidate construction, and descriptor
consumers. Test-only fixtures were checked but are not search behaviour.

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
| `src/methods/cluster_hopping/config.rs` surface portfolios | **Resolved:** both enabled surface modes are offered at every size. | **Size-independent measured behaviour.** | The surface portfolio allocates from observed rewards; no atom-count branch remains. |
| `src/methods/cluster_hopping/preset.rs:11` | **Resolved:** `ALL_POINTS_STEP_RATIO` is a dimensionless default multiplied by the problem length scale. | **Size-independent setting.** | Keep one ratio for every potential and do not add per-size rows. |
| `src/methods/cluster_hopping/moves.rs`; `src/methods/cluster_hopping/config.rs` | **Resolved:** lone flexible groups retain every declared molecular arm. | **Measured behaviour.** | Molecular presets enable depth allocation, which learns arm value without a group-cardinality cut. |
| `src/soap.rs:1208`, `:1253`, `:1257` | `ACTIVE_VOLUME_MIN = 8` fixes the minimum active region. | **In-flight removal / behaviour** (`bc-250d428e`). | Select active support from descriptor sensitivity or a cumulative displacement/uncertainty target. |
| `src/soap.rs:1346`, `:1348` | FCC and icosahedral `Template::points()` are SOAP prototypes. | **In-flight removal / behaviour** (`bc-250d428e`). | Learn prototypes from encountered environments or compare directly in descriptor space. |
| `src/soap.rs:1459-1461`, `:1671-1762`, `:2290-2668` | Fivefold plateau, axes, probes, residuals, openings, and steps; size gates at `n < 7`, `n < 5`, `n > 13`, and `n <= 13`. | **In-flight removal / behaviour** (`bc-250d428e`). | Generate directions from generic descriptor gradients, unstable modes, or under-sampled local environments. |
| `src/soap.rs:1459-1499` | `fivefold_plateau`, `extended_scale`, and `motif_fractions` have no production callers. | **Dead motif API**, not current runtime steering. | Delete them or expose them only through an explicitly diagnostic interface. |
| `src/soap.rs:247-257` | Atomic-number branches provide hard-coded covalent radii. | **In-flight removal / species behaviour** (`bc-250d428e`). | Accept pair scales or a bond graph as problem data; do not map elements inside search. |
| `src/methods/cluster_hopping/moves.rs` cycle arms | **Resolved:** SOAP, cycle reversal, and rewiring no longer switch at a named group count. | **Measured behaviour.** | Every compatible three-site molecular chart exposes the same arms; the allocator learns their value. |
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
| `src/bank_rpc/server.rs` IRA metric | **Resolved:** the server uses one `IraMetric::default()` search bound. | **Size-independent setting.** | No atom-count branch remains. |
| `src/methods/cluster_hopping.rs` rigid acceptance | **Resolved:** every rigid chart uses one temperature conversion and adapts orientation amplitude from accepted basin-changing proposals. | **Measured behaviour.** | Same-basin returns are vetoed before feedback and do not enter its trial count. |
| `src/catalog/packing.rs`; `src/known_basin.rs` | **Resolved:** packing admission is based on measured local geometric rank. | **Descriptor sufficiency.** | Full-rank neighbour clouds use packing descriptors; planar and collinear clouds fall back without an atom-count floor. |
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
| `src/methods/cluster_hopping/config.rs` retired priors | **Resolved:** `prior_reseed`, `prior_growth`, `prior_core`, `prior_handover`, `prior_motif`, and `ContinuousSymmetry::Measure::reseed_hops` were removed. | **Inert configuration removed.** | Only the consumed continuous-symmetry interval remains. |

## Scanner policy

`tests/no_motif_constants.rs` scans production `src/**/*.rs` lines for:

1. named motif identifiers (including fivefold, icosahedral, decahedral,
   Mackay, Marks, Leary, FCC, pentagon, and motif-class spellings);
2. direct comparisons of `n`, `n_points`, `n_atoms`, `atom_count`, or `natoms`
   against integer thresholds of five or more; and
3. integer constants of five or more whose names identify atom, point,
   molecule, rigid-body, or SOAP active-volume counts.

Every existing match is tied to a path, source needle, and one-line reason.
The expected match count for each exception is locked: another occurrence of
an already allowed needle also fails. The test likewise fails when an
allowlist entry becomes stale, so the two in-flight removals must delete their
corresponding exceptions when they land.

The following lookalikes were deliberately excluded: dimension and empty-input
checks; two-/three-/four-point geometric validity checks; sample, visit,
optimizer, and archive counters named `n`; physical-potential constants; and
motif words confined to comments or `#[cfg(test)]` fixture modules. No source
path owned by `bc-250d428e` or `bc-a84a44f0` was edited.
