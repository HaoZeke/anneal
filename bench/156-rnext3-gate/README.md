# 156 r-next3 performance and parity gate

Arm `A` is `develop` `5abb4dedec6cd2e95e7b17472639529fd139b278`. Arm `A2`
is a second independent release build of `5abb4ded` for the A/A noise
floor band. Arm `B` is `5abb4ded` with `anneal-156-rnext3.mbox` applied
(`5ebffa4a29a20e71bb3e265f27bec8dd0b321c89`, tree
`4a05a694d5882a89ae8776a4efac93d4b6ec6d48`). Arm `R` is `r-next2`
(`pull/156/head` `52c2937fc59bb22dc78b443326a1b54c7cde816d`) merged with
`5abb4ded` (`2042b1b6d70be2adaf577f2ddf886aee1937cd48`, tree
`5ddfb3f4cc5ae1c07d9055b7d87860f0681d6448`).

The schedule, cell parameters, statistical definitions, and verdict rules
are declared in `PROTOCOL.md`.

## Builds

Each arm is a release build from its own worktree and target directory:

```
cargo build --release --locked --features rgpot-ex \
  --example h2h_gmin_proof --example h2h_seed --example h2h_metal \
  --example h2h_morse --example h2h_sc
```

The Sutton-Chen cells load `rgpot_sutton_chen_energy_forces` from one
shared library built from `rgpot`
`8efdd246322d1f32bc1bcbd7e5dfdd7a4237960e` with
`g++ -std=c++20 -O2 -fPIC -shared -ffp-contract=off -ICppCore` over
`SuttonChen/SuttonChenPot.cc`, `SuttonChen/SuttonChenScales.cc`,
`nlist/HalfShell.cc`, `nlist/HalfShellWalk.cc`, `nlist/NeighbourList.cc`,
`nlist/Neighbours.cc`, `nlist/MinimumImage.cpp`, `simd/exact_fma_hw.cc`,
and `simd/exp.cc` under `CppCore/rgpot`, without OpenMP. All binary
sha256 digests are recorded in `binary-sha256.txt`.

## Run

One process at a time, pinned with `taskset -c 0`. Every thread-count
environment variable is `1`, `OMP_DYNAMIC=FALSE`, `MKL_DYNAMIC=FALSE`,
and `H2H_*` variables are unset. No warm-up is discarded and no `--quick`
flag is passed.

Even seeds run `A, B, R, A2`; odd seeds run `A2, R, B, A`. No GMIN
executable or input is used.

## Probe

`probe.c` is compiled with `cc -O2 -fPIC -shared -Wall -Wextra -Werror`
and loaded via `LD_PRELOAD`. It activates only when `/proc/self/exe`
matches `GATE_PROBE_EXE`. It arms `ITIMER_PROF` at 1000 microseconds,
samples `RssAnon` from `/proc/self/status` on every tick as well as at
process start and exit, and counts ticks whose instruction pointer falls
inside `anneal_core` `#[target_feature]` symbol ranges.

## Record

Rows are appended to `<cell>/rows.csv` with LF (`\n`) line endings and
never rewritten. `<cell>/summary.csv` and top-level `summary.csv` and
`SUMMARY.md` summarize the completed rows per cell and overall.
