<p align="center">
  <img src="./branding/logo/anneal_logo.png" alt="Anneal" width="280">
</p>

# Anneal

**Start here.** Bound-constrained global optimization with a single budget knob, or classical simulated-annealing presets you can swap without rewriting a driver.

Simulated-annealing components on the [eindir](https://github.com/HaoZeke/eindir) typed primitives. One surface, many drivers: classical presets, Bayesian pilot+mixer, generalized Langevin equation (GLE) colored noise, rank-1 additive independence, quasi-Monte Carlo (QMC) polish, device/ensemble scale. All obey the same five-component algebra (Obj / Cool / Neigh / Move / Accept) and four composition laws checked at construction.

| | |
|---|---|
| Docs | https://anneal.rgoswami.me |
| License | MIT |
| Software DOI | https://zenodo.org/doi/10.5281/zenodo.10672746 |
| Paper reproducibility | https://github.com/HaoZeke/anneal_repro — Zenodo [10.5281/zenodo.20672620](https://doi.org/10.5281/zenodo.20672620) |
| History | Continuous development since **2023-02** (see git log); multi-author `CITATION.cff` |

## Generic global minimization

For a scalar objective on a finite box, `minimize` is the common Python entry.
Every callback receives the declared design dimension, including fixed
coordinates. A vector length divisible by three does not select atomic geometry.

```python
import anneal

result = anneal.minimize(
    lambda x: float(x @ x),
    x0=[1.0, 0.0, 1.0, 1.0, 1.0],
    bounds=[(-2.0, 2.0), (0.0, 0.0), (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)],
    jac=lambda x: 2.0 * x,
    budget=256,
    replicas=4,
    seed=7,
)
assert result.charged == result.nfev + result.njev <= 256
assert result.x[1] == 0.0
coverage = result.diagnostics["coverage_regions_per_chain"]
```

With a gradient, replicas propose and locally improve candidates. Without a
gradient, multiple replicas use values-only hop chains; one replica uses the
values-only portfolio. Shared coverage records explored regions without
requiring stationary points. Certified-minimum history is separate information,
not the definition of exploration. The returned incumbent uses the original
objective, independently of the exploration penalty and occupied chain state.

`nfev` and `njev` count actual objective and gradient calls; `charged` is their
sum. `diagnostics` preserves the underlying work, history, and coverage fields.
`success` means a finite feasible candidate was returned, not that global
optimality was certified. An optional parameter store archives results and
supplies a starting candidate; it does not serialize chain or coverage state.

The [communication contract](docs/orgmode/explanation/communication.org) maps
these interfaces to their actual channels. The Rust box configuration also
selects [persistent white or colored-noise escape](docs/orgmode/howto/box-langevin-escape.org)
through the same box search. Atomic symmetry-aware proposals and constrained
geometry use explicit specialized entry points; box clipping does not provide
a manifold retraction.

## Cluster search and cooperative production

`Config::recommended(n)` composes surface relocations that pay one acceptance
test for a whole excursion, Normal-Gamma Thompson allocation over move arms,
and tabu response to a stalled walk. `Config::for_cluster(n)` retains the plain
Wales-Doye protocol as a comparison baseline. Accuracy and efficiency claims
come from sealed, evaluation-matched ensembles rather than reference energies
or morphology labels supplied to the search.

Large-cluster production uses four synchronously cooperating replicas. Each
replica spends an independently auditable charged-work sequence and submits a
freshly validated quenched representative. The coordinator updates an exact
basin census and bounded descriptor catalogue, then closes a population epoch
only after all replicas submit. A target-free Feynman--Kac potential ranks
energy, descriptor novelty, census scarcity, and latent-Gaussian transition
uncertainty. Replayable systematic resampling assigns parents at fixed
population size; family caps and distinct descriptor-space rejuvenation keep
one funnel from consuming every processor element.

This population layer borrows fixed-population bookkeeping from diffusion
Monte Carlo, not imaginary-time quantum propagation or fixed-node physics.
The latent transition field is the Gaussian part of an INLA-style model; its
Gaussian posterior is solved directly, so no Laplace approximation or R-INLA
runtime is involved. Bayesian move allocation and quench screening retain
their own evidence, while nested sampling remains a matched-budget comparison
with separate live-point weights. Shared-catalogue and one-private-catalogue-
per-replica ensembles form the causal communication comparison.

The optional census bus exchanges current validated minima directly between
same-node replicas, independently of coordinator-mediated adoption:

| Setting | Effect |
| --- | --- |
| `CENSUS_BUS_BASE` | Enables the bus and selects its endpoint namespace. Use a distinct value for each concurrent ensemble. |
| `CENSUS_BUS_NEIGHBORS=0` | Subscribes to every peer; the default. |
| `CENSUS_BUS_NEIGHBORS=k` | Subscribes only to cyclic neighbours within positive ring distance `k`. |
| `CENSUS_BUS_IPC=1` | Uses Unix-domain IPC; the default transport is loopback TCP. |
| `CATALOG_SHARED_BIAS=1` | Enables shared repulsive bias deposits. |
| `CENSUS_BUS_UNBOUNDED=1` | Admits distant bus minima into that bias; the default admits only nearby packings. |

Changed minima publish immediately at eligible checkpoints; unchanged minima
refresh every eight eligible checkpoints. Hop-only refreshes update retained
state without duplicating deposits. Checkpoint spacing is controlled by
`CATALOG_SLICE`, in charged objective calls, not physical time.

The bus retains direct neighbours without forwarding their messages. A ring
therefore supplies a local census, not a globally mixed gossip aggregate, and
does not restrict the coordinator's parent selection or hearing. The packing
gate uses `nearby_packing` with histogram L1 distance at most `PACKING_LINK`
(0.35); it is distinct from exact-basin merge radii and supplies no guarantee
of one population cluster per energy funnel. The bounded/unbounded switch
applies to bus observations, not population-parent or own-visit history.

Packing comparisons reuse descriptor rows only when every ordered coordinate
bit matches. A process-wide cache shared by validation and request threads
retains at most 8 MiB of coordinate-key and descriptor payloads; bookkeeping
and caller-held references are additional. Pairwise
codebooks and histograms remain separate, so reuse does not introduce a shared
classification map or delay census invalidation when an occupied minimum moves.

The [communication contracts](docs/orgmode/explanation/communication.org)
map the atomic, box-gradient and values-only entry points to their history,
bias, census and coordinate-adoption channels, including certificate and
charged-work boundaries.

The [persistent box escape guide](docs/orgmode/howto/box-langevin-escape.org)
connects colored or scalar-white Langevin proposals to that same quench and
history path, with per-chain noise memory and combined callback accounting.

```rust
use anneal_core::methods::cluster_hopping::{optimize, Config, Ledger};

let cfg = Config::recommended(38);
let mut ledger = Ledger::new(400_000);
// supply `relax` closing over your objective; see examples/lj_cluster_search.rs
```

The `lj_joint_optimum` release example also provides a controlled NVE
minima-hopping communication comparison. With the `ira` feature enabled,
`mh-communication` runs both private-history and shared-history ensembles:

```bash
ANNEAL_MH_REPLICAS=4 ./target/release/examples/lj_joint_optimum \
  75 200000 2 mh-communication
```

Here 200,000 is the total objective-call budget for each ensemble, divided
among four replicas; two seeds give two paired comparisons. Both arms use
matched starting coordinates, replica seeds, and escape controls. Shared
history changes escape effort on rediscovery without copying coordinates
or pooling acceptance thresholds. Descriptors order exact minimum-identity
checks, and only freshly validated quenches enter the history. Trajectory
integration constructs optimization proposals, not kinetic rates or physical
residence times. JSON reports include aggregate first-discovery work,
per-replica work by stage, history overhead, and ensemble wall time. Private
minimum counts sum replica-local identities; shared counts describe one
common history. These counts are not interchangeable measures of coverage.

The structural archive retains rejected proposals, but the default exclusion
history contains only accepted minima. An energy-rejected proposal remains eligible
for another threshold trial. Shared classification and acceptance publication
are atomic; an unresolved quench contributes charged work, not a basin visit.

For an alternative exploration policy, set
`ANNEAL_MH_HISTORY_POLICY=observed-exclusion` on an ensemble run. Every certified
observation then enters exclusion history, including energy-rejected proposals,
and rediscovery feedback uses total observation counts. This policy is distinct
from accepted-minimum history; its output arm names carry `-observed-exclusion`.
`accepted` is the default, and unrecognized policy names are rejected.

Exact matching uses a per-ensemble cache of immutable pair-distance spectra,
keyed by complete coordinates. It only prunes impossible matches; survivors
retain native matching and identity-context checks. The default coordinate-key
and spectrum payload limit is 128 MiB, with map metadata additional. Set
`ANNEAL_MH_PAIR_CACHE_BYTES=0` to disable storage for a controlled comparison.
Each relation applies its pair screen once; rejection stops at the first
sufficient distance discrepancy, and survivors reuse the prepared screening
work without changing the native identity decision.
Ensemble records retain each replica's best coordinates for independent audits.

For a controlled escape probe, `ANNEAL_START_COORDINATES` selects a plain
coordinate file containing one finite `x y z` triplet per atom, without an
XYZ header or element labels. Every replica starts from that structure;
the seed still controls its proposals. The configuration record reports
`start_protocol: "fixed-coordinate-file"` and embeds the input coordinates.
This input cannot be combined with `ANNEAL_OPTBENCH_STARTS`: a diagnostic
shelf structure is not a published random-start archive. The `mh-private-soft`
and `mh-shared-soft` selectors add velocity softening to the same comparison.

External potentials use the same optimizer driver. The molecular-cluster and
slab examples share one persistent in-process profile adapter; selecting
`nwchemc` loads `libnwchemc` once and serves the complete hop loop without an
RPC server or a result cache. Molecular requests omit a simulation cell, while
the slab driver sends the periodic cell through the same adapter.

```bash
POTENTIAL_CONFIG=/path/to/PotentialConfig.bin \
POTENTIAL_LIBRARY=/path/to/libnwchemc.so \
cargo run --locked --release --features rgpot-ex \
  --example molecular_cluster -- 6 1200 8 nwchemc
```

The shared adapter is
[`examples/common/profile_engine.rs`](examples/common/profile_engine.rs); the
two consumers are
[`examples/molecular_cluster.rs`](examples/molecular_cluster.rs) and
[`examples/slab_adsorption.rs`](examples/slab_adsorption.rs).

Rigid TIP4P water is a first-class objective, not an example-local potential.
Each molecule is a rigid body (centre of mass plus an exponential-map
rotation vector) with Jorgensen parameters, no cutoff, and an analytic
gradient assembled from site forces and torques. Basin hopping uses a
Wales--Hodges translation/rotation kernel rather than the atomic Cartesian
moves. Compare putative global minima against Wales and Hodges,
*Chem. Phys. Lett.* **286**, 65 (1998):

```bash
cargo run --locked --release --example water_tip4p -- 6 20000 4
```

The `vesin-nl` feature compiles vesin's own cell-list sources into the
neighbour list instead of the crate's reimplementation. Its build script
reads `VESIN_SRC`, a checkout of [vesin](https://github.com/Luthaf/vesin),
so `cargo test --all-features` needs it set:

```bash
VESIN_SRC=/path/to/vesin cargo test --locked --all-features
```

## Install

```bash
pip install anneal
```

Full stack (pinned Rust + Python + docs):

```bash
pixi install
```

## Start here (budget-only portfolio)

The intended stand-alone tool for most users: pass an objective, box bounds, and a work-unit budget (objective and gradient evaluations share the counter).

```python
import numpy as np
from anneal import global_optimize

def rastrigin(x):
    return 10.0 * len(x) + np.sum(x * x - 10.0 * np.cos(2.0 * np.pi * x))

low, high = np.full(5, -5.0), np.full(5, 5.0)
out = global_optimize(rastrigin, low, high, budget=4000, seed=0)
print(out["best_val"], out["best_pos"])
```

Runnable copies:

- Script: [`examples/quickstart_portfolio.py`](examples/quickstart_portfolio.py)
- Notebook: [`examples/notebooks/01_quickstart.ipynb`](examples/notebooks/01_quickstart.ipynb)
- Website quickstart + four tutorials: https://anneal.rgoswami.me

## Classical presets (same driver, different slots)

```python
from anneal import Boltzmann, Fast, Gsa, run

h = run(rastrigin, low, high, Boltzmann(t_init=5.0, sigma=0.5),
        n_epochs=40, steps_per_epoch=50, seed=1)
print(h.best_val)
```

## Optional arms (additive independence + QMC polish)

```python
import numpy as np
from anneal import additive_independence, qmc_polish

def rastrigin(x):
    return 10.0 * len(x) + np.sum(x*x - 10.0 * np.cos(2.0 * np.pi * x))

def grad_rastrigin(x):
    return 2.0 * x + 20.0 * np.pi * np.sin(2.0 * np.pi * x)

low = np.full(5, -5.0)
high = np.full(5, 5.0)

# Values-only rank-1 independence (no gradient)
res = additive_independence(rastrigin, low, high, max_fevals=3000, seed=7)

# Polish with gradient
refined = qmc_polish(rastrigin, grad_rastrigin, low, high,
                     n_starts=32, max_fevals_per_start=50, seed=0, top_k=1)
print(refined["best_val"])
```

Full docs, tutorials (classical, Bayesian pilot+mixer, GLE, polish+device), algebra, how-tos, and reference at https://anneal.rgoswami.me .

## Method catalog

Every algorithmic mechanism in the crate, with file:line references,
constants, and how the pieces feed each other:
[`docs/orgmode/methods/catalog.org`](docs/orgmode/methods/catalog.org).
A narrative walk of one cooperative run end to end, naming each
mechanism as it fires, plus a dataflow diagram:
[`docs/orgmode/methods/composition.org`](docs/orgmode/methods/composition.org).

| Group | Covers |
|---|---|
| Local search | Quenching, screening, biased hopping, move kernels, stall escapes |
| Learned allocation | Thompson/depth/contextual allocators, budget-window temperature, regime selection |
| Descriptors and identity | SOAP/ACE spaces, featomic, IRA/SOFI shape matching, census calibration |
| Cooperative layer | Catalog/census, policy, boundary transport, descriptor holes, Feynman-Kac population epochs, spectral referee, umbrella bridges, the Cap'n Proto protocol |
| Population methods | CSA bank, archive search, splice, portfolio, replica exchange, warm L-BFGS |
| Statistics and accounting | Cooperative and local ledgers, Good-Turing census accounting, paired-seed evaluation methodology |

## Development

```bash
pixi install
pixi run -e python python-test
pixi run -e docs docs-export
pixi run -e docs docs-build
```

See `pixi.toml` and `docs/export.el` (modeled on rgpycrumbs/rsx-rs patterns).

## License and citation

MIT (see `LICENSE.txt`). Citation: `CITATION.cff` or the software Zenodo DOI. Multi-author software citation lists six authors. Project history since February 2023. Reproducibility package for paper tables and figures: [HaoZeke/anneal_repro](https://github.com/HaoZeke/anneal_repro) (Zenodo [10.5281/zenodo.20672620](https://doi.org/10.5281/zenodo.20672620)).
