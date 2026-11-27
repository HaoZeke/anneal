# ASE provenance rerun

This record uses anneal commit
`9837358ae1027067ff654fac8b48969020853452`, whose tree is
`15032ef401fb681e230d27c7b315d89f7ac53e4e`. The release binaries are
built with Rust 1.88.0 from that detached revision.

ASE is pinned to exactly 3.29.0. The wheel
`ase-3.29.0-py3-none-any.whl` has SHA-256
`7b9dd103f007810339c24acfee2f6b677c0c48443b21d3c98e52959246cf4ebf`.
Each row repeats the version and wheel digest.

The six systems use the paper budgets: 400,000 calls for LJ38;
1,000,000 for LJ55 and Au55; and 4,000,000 for LJ98, LJ104 and LJ150.
Every system runs seeds 2000--2011 for the cluster preset, ASE basin
hopping, ASE minima hopping and the recommended preset. All cluster
preset seeds run first. Three times their slowest wall time is then
recorded on every row and applied to the other three arms.

Every potential evaluation returns an energy and its gradient together.
The separate energy and gradient columns therefore have the same
nonzero count. A row also records its stopping reason, best energy,
reference classification, source commit and tree, and a SHA-256 machine
identifier. Runs are serial, bound to one core, and set each supported
numeric thread count to one.

Use a detached source tree and separate release target:

```sh
REV=9837358ae1027067ff654fac8b48969020853452
git worktree add --detach "$SOURCE_ROOT" "$REV"
cargo +1.88.0 build --release --locked \
  --manifest-path "$SOURCE_ROOT/Cargo.toml" \
  --example ase_panel --example asg_eval \
  --target-dir "$TARGET_ROOT"
```

Install the pinned wheel under `ASE_PYTHON_ROOT`, verify its digest, and
run:

```sh
python3 bench/ase-rerun-9837358a/run_campaign.py \
  --python-root "$ASE_PYTHON_ROOT" \
  --panel "$TARGET_ROOT/release/examples/ase_panel" \
  --asg "$TARGET_ROOT/release/examples/asg_eval" \
  --work "$RUN_ROOT" \
  --rows bench/ase-rerun-9837358a/rows.csv \
  --logs "$LOG_ROOT"
```

The driver is append-only and skips complete rows when resumed. It
writes each sanitized row log through a temporary file, syncs it,
renames it, and verifies its SHA-256 from the final file.
