# UmaPot: UMA / OMol AOTI frontend

`UmaPot` loads an AOTI (AOTInductor, PyTorch ahead-of-time
compiled) package (`.pt2`) that `scripts/export_uma_aoti.py`
wrote from a UMA checkpoint.

vesin builds the neighbor list. The graph returns energy and forces.
Charge and spin are per-call tensors. The compiled graph does not
call fairchem at evaluate time.

## Package

One `.pt2` per composition, charge and spin. The exporter embeds the
runtime contract in the package itself
(`aot_inductor.metadata`). `UmaPot` reads it with
`AOTIModelPackageLoader::get_metadata()`; no file next to the
package is read.

| Key | Use at runtime |
| --- | --- |
| `cutoff`, `max_neighbors` | vesin neighbor list |
| `molecular_box` | cube side length; molecular positions are centred at zero (0: caller's cell) |
| `batch_max` | systems per forward (0 or 1: one system at a time; above 1: chunks of that size) |
| `batch_packed` | 1: the chunk is the real system count (dynamic atom and edge counts). Absent or 0: a short chunk repeats its last system up to `batch_max` |
| `pos_dtype` | input dtype of positions, the cell and cell offsets (`float32` or `float64`) |
| `model_dtype` | weight dtype recorded by the exporter (`float32` or `float64`) |
| `task_name`, `charge`, `spin` | must equal `UmaConfig` |
| `z_set`, `natoms`, `counts` | must match the input atoms |
| `natoms_min`, `natoms_max`, `nedges_min`, `nedges_max` | atom and edge counts the graph accepts |
| `z_max` | largest atomic number a composition-free package accepts |
| `label`, `shapes`, `inputs`, `outputs` | recorded |
| `model`, `torch_version`, `fairchem_version`, `export_path` | recorded |

`UmaConfig.cutoff` / `max_neighbors` are defaults. The embedded
values win when present.

`scripts/export_uma_aoti.py --dtype float64` converts the model with
`hydra.double()` before the AOTI trace and records both dtype keys as
`float64`. `--dtype float32` calls `hydra.float()` and records
`float32`. A package with neither key falls back to float32, and the
load line keeps a missing key as `unset` rather than inventing it.
`pos_dtype` and `model_dtype` must name the same dtype; a disagreement
throws before any force call.

## Contract checks

`merge_mole` folds one composition, charge and spin into the graph.
A different charge or spin evaluates another potential energy
surface without any error; a different composition aborts inside
the graph. Every force call therefore checks the config and the
input against the embedded keys and throws `rgpot::UmaContractError`
(a `std::runtime_error`; `field()` names the key) on a mismatch:

```
UmaPot: config charge is -1, the package was exported for 0
UmaPot: input counts is {1: 4, 6: 2}, the package was exported for {1: 2, 6: 2}
UmaPot: edge_index shape is [2, 42], the package was exported for [2, 56]
```

Positions and the edge list are checked against the shape the package
declares before the compiled graph is called. A static package names
the traced sizes (`[8, 3]` and `[2, 56]` for an eight-atom complete
graph). A package that was exported with a symbolic edge count names
that axis and its range. A mismatch throws `rgpot::UmaShapeError`.
An edge index outside the position buffer throws the same error and
is not gathered. `pos_shape` and `edge_index_shape` record the
declaration; when they are absent the atom count and the edge range
are used. A package that records neither `nedges_min` nor
`nedges_max` is fixed-shape: the edge count is the `edge_index`
length in `shapes` (Grignard is `[2, 1332]`). The neighbour list is
built, and a short band is padded, to that count. A list that cannot
fit throws `UmaShapeError` and names both shapes. A package with only
one of the two keys, or with neither key and no fixed `edge_index`
in `shapes`, is refused.

A molecular box recenters the atoms in a cube and is traced on every
ordered pair, `n(n-1)`. Any other edge count throws `UmaShapeError`
before the compiled graph runs. The single path and the band path
use that same check. A band whose edge range is not a single number
still checks each system against the range. Any nonzero cell offset
is a periodic image, so the call throws `UmaContractError` on
`cell_offsets` (`nonzero`).

`natoms` and `counts` exist only in packages from the current
exporter. Older packages get the `z_set` check, which cannot tell
C2H2 from C2H4.

`scripts/export_baker_uma_aoti.py` walks Baker endpoints and
deduplicates by exact composition (atom count per element), charge
and spin.

## Export path

Without `--molecular-box` and `--batch-max`, `export_uma_aoti.py`
writes a dynamic package (`export_path` `dynamic`). The atom count
and the edge count are `torch.export.Dim` symbols, by default
`natoms` in [2, 1024] and `nedges` in [1, 1024 * 300]
(`--max-atoms`, `--max-edges`). The package records `natoms_min`,
`natoms_max`, `nedges_min`, `nedges_max`, `pos_shape` `natoms,3` and
`edge_index_shape` `2,nedges`, and UmaPot checks each call against
those ranges. The MoLE experts stay in the graph, so the package
covers every composition of its task head: it records no `z_set`,
`natoms` or `counts`, and `z_max` bounds the atomic numbers.
`--merge-mole` folds one composition into the weights instead. The
package is kept only after it matches FAIRChemCalculator on the input,
on the input with one atom moved past the cutoff, and on a second
molecule. `docs/orgmode/howto/uma-export.org` lists the bounds and
limits.

With `--molecular-box` the exporter tries a dynamic export, then a
static one, then `make_fx(tracing_mode="real")`, and records the
traced edge count as both `nedges_min` and `nedges_max`, plus the
molecular box and a molecular flag.

## Band export

A Cartesian stencil is many geometries of one molecule. Export that
molecule once with `--batch-max` greater than 1 and UmaPot evaluates
the stencil through `forceBatch`, one compiled forward per chunk.
`--molecular-box` recenters every geometry in the same cube and widens
the neighbour cutoff to `0.4 * L`, so the edge count stays
`batch_max * n * (n - 1)`. The graph is traced at exactly `batch_max`
systems. A chunk with fewer systems repeats its last system and the
padded results are discarded. `batch_max` of 0 or 1 keeps the
single-system graph, and `forceBatch` then calls the per-system loop.

```
python scripts/export_uma_aoti.py \
  --atoms path/to/geometry.xyz \
  --molecular-box 30 \
  --batch-max 8 \
  --dtype float32 \
  --task omol \
  --charge 0 \
  --spin 1 \
  --out /tmp/geometry-b8.pt2
```

`pixi run -e umaexport export-uma-aoti -- --molecular-box 30 --batch-max 8`
is the same command inside the exporter environment. The `.pt2` is a
compiled graph for one composition. Do not commit it. The checkpoint
weights stay where fairchem loaded them.

`--batch-style packed` is the other graph. It concatenates the systems
(one batch index, block-diagonal edges) and exports a dynamic atom
count and a dynamic edge count, up to `--batch-max` systems. UmaPot
then sends the real count and does not repeat a system to fill the
chunk. A same-composition pack shares one expert mix, so that mix is
one GEMM rather than a per-system segment loop.

```
python scripts/export_uma_aoti.py \
  --atoms path/to/geometry.xyz \
  --molecular-box 30 \
  --batch-style packed \
  --batch-max 8 \
  --dtype float32 \
  --out /tmp/geometry-packed.pt2
```

The CPU GEMM kernel is compiled for `torch.get_num_threads()` at export
time. Set `OMP_NUM_THREADS` (and the same `MKL_NUM_THREADS` and
`OPENBLAS_NUM_THREADS`) to the count the package will run with. The
exporter applies `OMP_NUM_THREADS` to torch before the compile.

## potserv

```
Uma:<model.pt2>
Uma:<model.pt2>:<task>
```

Task default is `omol`. Device default is `cpu`.

## Build

Needs torch and vesin (`-Dwith_uma=true`, pixi env that has both).
`libuma_engine.so` is the dlopen plugin for eonclient: the host
never links torch. Config is a Cap'n Proto `UmaParams` message.

Molecular graphs use a large constant box so the neighbor list
shape is stable. Do not treat that box as a periodic crystal.

Elja / some HPC loaders need the AOTI execstack rewrite in
`aoti_execstack.hpp` (`docs/newsfragments/+uma-elja-execstack.fixed.md`).
