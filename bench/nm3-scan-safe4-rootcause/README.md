# SAFE4 BLJ130N38 trajectory root cause

## Revisions

The comparison used develop
`61fa67cf380e46876b569da45e7cc091647a775b`, tree
`cff796c507a8fa1d0bfb28888c3570b61988055e`, and SAFE4
`7d5bd6924075fbc66c33da3632d86f2e7eb40b76`, tree
`578a12095ab5e0e6ac89c805173db78a9d1578b7`.

The binary-Lennard-Jones driver came from blob
`85916cc5e6f83d0bb1854bebf8c9bb6c27bd9b50` on
`origin/bench/blj-gmin`. The diagnostic copy calls the existing public
`BinaryLennardJones::new` constructor, prints `best.to_bits()`, and can
write charged-call position and energy bits. Those changes do not alter
the configuration, potential, random stream, or search.

Every run was a separate process pinned with `taskset -c 0`. Rayon,
OpenBLAS, OpenMP, and MKL thread counts were all one. Builds used Rust
1.88.0 and the release profile.

## Endpoint reproduction

The 12 seeds reproduce the reported result:

| revision | hits |
| --- | ---: |
| develop | 11/12 |
| SAFE4 | 6/12 |

Develop hits and SAFE4 misses on seeds 2003, 2006, 2007, 2009, and
2011. The complete rows are in `blj130n38-endpoints.csv`.

## Seven-commit bisect

The table below covers those five seeds. `best bits` and `charged` count
rows identical to develop.

| revision | hits | best bits | charged |
| --- | ---: | ---: | ---: |
| `61fa67cf380e46876b569da45e7cc091647a775b` | 5/5 | 5/5 | 5/5 |
| `8f349907f671fdb83209d105e83509e0a938f62f` | 4/5 | 2/5 | 0/5 |
| `3b4d0d5d73559cc5b6d1fa968fc989ab44c04d3e` | 1/5 | 0/5 | 0/5 |
| `52c2937fc59bb22dc78b443326a1b54c7cde816d` | 0/5 | 0/5 | 0/5 |
| `838390775c91cfa717c5bc848e3ee6dddd9f3837` | 0/5 | 0/5 | 0/5 |
| `df2f1a8132655b835f48e7adaa1a3c8656c22ed1` | 0/5 | 0/5 | 0/5 |
| `39e00683cd3caebdf15d48ba00f0f996e1cffeec` | 0/5 | 0/5 | 0/5 |
| `7d5bd6924075fbc66c33da3632d86f2e7eb40b76` | 0/5 | 0/5 | 0/5 |

The first trajectory-changing commit is therefore
`8f349907f671fdb83209d105e83509e0a938f62f`, not one of the four
scan-safe commits. The per-seed evidence is in
`blj130n38-bisect.csv`.

## First divergence at `8f349907`

Seed 2003 was traced at every charged potential or gradient call.
Develop and the culprit are bit-identical through call 31,864. That
call is the validation gradient of the state at energy
`-183.53238924878482`, with position hash
`3148051099797865224`.

Call 31,865 is the first difference. It begins the scheduled
continuous-symmetry quench at hop 9:

| value | develop | culprit |
| --- | ---: | ---: |
| projected-position hash | 17327473028948167674 | 689677982348609016 |
| initial projected energy | 188012.607419281 | 37581.30722774891 |
| first x coordinate | -0.22561521676315882 | -0.36916724148833724 |
| first y coordinate | 0.90437013147905843 | 0.98027910650773686 |
| first z coordinate | -0.55892283988650127 | -0.40658280955563109 |

All 114 coordinate bit patterns differ at that call. The exact
coordinate and energy bits are in `divergence-trace.json`.

Develop's `csm::propose` chooses the lowest-measure member of its fixed
`C_i`, `C_5`, and `D_5h` menu. On this state it chooses `D_5h`, measure
`0.037389704963383626`. The quench lowers the energy to
`-187.74690577809855`, so the strict improvement rule adopts it.

The culprit replaces that menu in `csm::detected_ops`. No operation is
inside the detector's 0.25 threshold, so the fallback unbounded scan
selects a proper order-3 rotation with deviation
`0.2898208225054814` and axis
`[-0.8217311851504174, 0.5623864081191248,
-0.09208358873414686]`. Closing that generator gives three
operations. Its quench reaches `-179.58922194386113`, above the
occupied energy, so it is rejected. The exact detected candidate and
both transition records are in `projection-verdict.json`.

There is no RNG draw at the divergence. `csm::propose` takes no RNG,
and the scheduled projection branch runs directly between the
bit-identical validation call and call 31,865. This is also not a
Metropolis tie: the old candidate is strictly adopted and the new
candidate is strictly rejected. The detected order changes before the
first differing potential evaluation.

This is the direct execution of `8f349907`'s intended detector policy.
It is not a wrong-neighbour result, tie-order change, or floating-point
reduction-order change. On this seed the policy change is unlucky.

## First divergence at `3b4d0d5d`

Seed 2003 was traced between
`8f349907f671fdb83209d105e83509e0a938f62f` and
`3b4d0d5d73559cc5b6d1fa968fc989ab44c04d3e`. The runs are
bit-identical through charged call 30,006. Call 30,007 is the first
difference:

| value | parent | `3b4d0d5d` |
| --- | ---: | ---: |
| position hash | 5859068800520237662 | 8830317320304641679 |
| initial energy | -25.149179424480955 | -185.26831907025812 |
| differing coordinate words | 114/114 | 114/114 |

The call is the relaxation of a random `Symmetrise` proposal through
`Symmetrise::propose` in `src/movekernel.rs`. The axis, blend, and
random stream are the same:

```text
axis = [-0.6051250433003629, -0.7506517677567397,
        0.2652274599931455]
blend = 0.60250195493291836
```

The parent draws order 6 from five orders. Removing the caps enlarges
the draw set to 37 orders, so the same random variate selects order 34.
That high-order, near-self-mapping turn is accepted at hop 4:

```text
from_bits=13864096739452260888
to_bits=13864096739429522462
action=sym adopted=true
```

The cap removal is an intended policy change. However, this path
exposes a general proposal bug: `Symmetrise` samples an order without
checking whether that turn exchanges any sites. It is not specific to
BLJ130N38, a species, or a motif. The exact call records and proposal
fields are in `divergence-trace-r2.json`.

## First divergence at `52c2937f`

Seed 2009 was traced between
`3b4d0d5d73559cc5b6d1fa968fc989ab44c04d3e` and
`52c2937fc59bb22dc78b443326a1b54c7cde816d`. The runs are
bit-identical through charged call 25,459. Call 25,460 is the first
difference:

| value | parent | `52c2937f` |
| --- | ---: | ---: |
| position hash | 14925541266737733085 | 16325367837784085924 |
| initial energy | 966.9674897837749 | -161.346802398689 |
| differing coordinate words | 114/114 | 114/114 |

This is the orbit-completion block in
`src/methods/cluster_hopping.rs`, fed by
`packing::orbit_candidates`, after the same accepted hop-2 symmetry
move. On the same 23-point core, the parent detector returns 242
candidates. Its first three generators all have order 23 and closure
reaches the 48-operation limit. No orbit transition is accepted. The
child rejects those self mappings and returns nine candidates:

```text
orders = {1: 2, 2: 1, 3: 2, 5: 2, 6: 2}
selected generator orders = [3, 5, 1]
group size = 48
```

The resulting orbit completion is a strict improvement and is accepted
at hop 3. Thus `52c2937f` is a correctness fix behaving as intended.
It does not reject the valid order-3 or order-5 turns; it exposes them
after removing spurious high-order self mappings. The later miss is an
unlucky trajectory outcome, not evidence that the self-mapping test is
wrong.

## General fix experiment

A general prototype reused the self-mapping detector in the random
`Symmetrise` move. It kept the sampled order when that turn exchanged
sites. Otherwise, it selected the lowest-residual valid order from the
same caller-supplied order set. If no order exchanged sites, it kept the
old proposal. The prototype introduced no motif, size, species, or
tuned numeric constant.

At the paper budget, it improved BLJ130N38 from SAFE4's 6/12 to 10/12.
It recovered all five SAFE4 misses that had hit on develop, and gained
seed 2002, but lost SAFE4 hits on seeds 2000 and 2010. The complete
rows are in `rejected-prototype-blj130n38.csv`.

The no-loss smoke rejected the prototype:

| system | SAFE4 | prototype |
| --- | ---: | ---: |
| LJ38 | not rerun | 12/12 |
| LJ75 | 4/12 reported panel | 7/12 |
| LJ98 | 9/12 | 7/12 |

LJ98 was rerun on SAFE4 in the same serial setup; its 9/12 rows are in
`safe4-lj98.csv`. All prototype smoke rows are in
`rejected-prototype-lj-smoke.csv`. The rejected arm is commit
`1546b40a8af6bb2fa0a83e9474de07f513be9a0b`, tree
`548055ecb3464d2c9bce64f596e2f13481254ad4`. It is a reproducibility
record and is not proposed for develop. The BLJ executable SHA256 was
`65e90eb6019c86f4c7cbf8f1e8fa2a567b13826995bbc893b1862de9f5eeb074`;
the LJ executable SHA256 was
`28428d223118ec60851e1db5041671011bc0253bece35c8e4726a078e0d69c60`.
The committed LJ rebuild did not match the earlier
`086fd9cc44683ccb08f8f4ded8f568e72449d32425673ee41651436e2f35cf46`
executable, so the prototype rows below were rerun rather than relabelled.
Every scientific field reproduced exactly; only the committed revision,
tree, executable hash, and timing fields changed.

The exact prototype rule was:

```rust
let cloud = Cloud::from_view(x, n);
if cloud
    .symmetry_deviation(axis, sampled, f64::INFINITY)
    .is_some()
{
    return sampled;
}
let mut best_order = None;
let mut best = f64::INFINITY;
for &order in orders {
    let Some(deviation) =
        cloud.symmetry_deviation(axis, order, f64::INFINITY)
    else {
        continue;
    };
    if deviation < best {
        best = deviation;
        best_order = Some(order);
    }
}
best_order.unwrap_or(sampled)
```

Four other general exploratory policies were screened on BLJ130N38:
project every valid detected fallback, combine that projection with the
best valid random order, reject an invalid sampled order as a no-op, and
sample uniformly among valid orders. None restored develop's 11/12.
Those screens were not candidates for the no-loss panel.

## Verdict

The first trajectory change is the intended continuous-symmetry
detector policy in
`8f349907f671fdb83209d105e83509e0a938f62f`. The larger hit drops then
come from two separately traced paths:

1. `3b4d0d5d73559cc5b6d1fa968fc989ab44c04d3e` changes the random
   order mapping and exposes the general unchecked-self-mapping bug in
   `Symmetrise`.
2. `52c2937fc59bb22dc78b443326a1b54c7cde816d` correctly removes
   self-mapping detector candidates. Its changed orbit completion is
   intended and unlucky on the traced seed.

The general prototype partially recovers BLJ130N38 but regresses LJ98
from 9/12 to 7/12. No general candidate both restored the BLJ result and
passed the no-loss screen. The committed fallback and its regression
test exist only to identify the rejected benchmark arm; they are not a
production recommendation and have no towncrier fragment. The
conditional three-arm final panel was not run because there is no
accepted fix arm.

## Reproduction

The endpoint command shape was:

```text
taskset -c 0 <revision-binary> run <system> <seed>
```

The bisect used seeds 2003, 2006, 2007, 2009, and 2011 at the paper
budget of 400000 charged calls for every revision on the first-parent
path. The divergence runs set `H2H_TRACE`. Separate diagnostics printed
the selected projection, random symmetry proposal, detector candidates,
and transition records without changing search decisions. All panel
runs were serial, separate processes under `taskset -c 0`.
