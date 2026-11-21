# Cap-Turn Default-On Gate (`bench/cap-turn-gate`)

Pre-registered gate deciding whether `Config::cap_twist` may be switched
from opt-in (`false`) to default-on (`true`) on top of PR #204
(`feat/cap-turn`).

## Code Revisions and Trees

- Candidate revision (`A`, `B`, `PA`, `A2`, `P1..P5`):
  - commit: `16ef5cb869f68c46e71b1eae2cbf7cee9321d8d4`
  - code tree: `0e08e2e2086180ca3b186f977191f6c1a2d12cd1`
- Reference revision (`D`, `develop` baseline):
  - commit: `9837358ae1027067ff654fac8b48969020853452`
  - code tree: `15032ef401fb681e230d27c7b315d89f7ac53e4e`

`anneal-core` (`src/`) is unmodified across all candidate builds so the
measured library tree is strictly `0e08e2e2086180ca3b186f977191f6c1a2d12cd1`.

## Flag Mechanics in `anneal-core`

- `Config::cap_twist: bool` (`src/methods/cluster_hopping/config.rs`)
  defaults to `false` in `Config::with_scales`. When `true`,
  `Config::resolved_json()` emits `"cap_twist":true`; when `false`, the
  field is skipped so existing preset hashes are preserved.
- `stall_arms(cfg)` (`src/methods/cluster_hopping.rs`) adds `"cap"` iff
  `cfg.cap_twist && !cfg.move_library.is_rigid_body() &&
  cfg.move_library.declared_groups().is_none()`.
- On atomic clusters (`LJ`, `BLJ`, `Au`, `Morse`), `"cap"` joins
  `stall_arms` and `leave_arms`, invoking `crate::cap_twist::propose` on
  call-plateau leaves (`restart_patience = 4_000`) and stuck stalls, and
  incrementing `Outcome::cap_escapes` on each cap escape.
- On rigid-body water (`TIP4P`), `cfg.move_library.is_rigid_body()` is
  `true`; `stall_arms(cfg)` reads `cfg.cap_twist` and withholds `"cap"`,
  keeping `Outcome::cap_escapes == 0` and preserving rigid-body
  trajectories bit-for-bit.
- In the unified gate driver (`bench/cap-turn-gate/driver`), setting
  `H2H_CAP_TWIST=1` sets `cfg.cap_twist = true` on every cell's `Config`
  and emits both `cap_twist_flag` (`1` iff `Config::resolved_json()`
  contains `"cap_twist":true`) and `cap_escapes` (`Outcome::cap_escapes`,
  plus `cap_arm_draws` from `Outcome::arms`).

## Arms and Build Matrix

All binaries are built with `cargo build --release --locked` and
`anneal-core/rgpot-ex` kernels active:

1. `A` (`16ef5cb8`, `GATE_PAD_KB=0`, `H2H_CAP_TWIST` unset): `cap_twist`
   off on the primary candidate release binary.
2. `B` (`16ef5cb8`, `GATE_PAD_KB=0`, `H2H_CAP_TWIST=1`): **exact same
   compiled release binary** (`binary_sha256(B) == binary_sha256(A)`)
   with `cap_twist` on.
3. `D` (`9837358a`, `GATE_PAD_KB=0`, `H2H_CAP_TWIST` unset): `develop`
   reference release binary.
4. `PA` (`16ef5cb8`, `GATE_PAD_KB=0`, `H2H_CAP_TWIST` unset): primary
   candidate binary rerun in the calibration band (paired against `A` for
   same-binary `A/A` ratios and against `A2, P1..P5` for rebuild and
   placement ratios).
5. `A2` (`16ef5cb8`, `GATE_PAD_KB=0`, `H2H_CAP_TWIST` unset): independent
   release rebuild in a separate target directory (`A/A` rebuild).
6. `P1..P5` (`16ef5cb8`, `GATE_PAD_KB = 4, 16, 64, 256, 1024`,
   `H2H_CAP_TWIST` unset): `A` rebuilt 5 times in separate target
   directories with only `.rodata` padding (`driver/build.rs`) changed to
   shift binary virtual-page placement across 4 KiB..1 MiB boundaries.

## Execution Hygiene and Schedule

- Single CPU core (`taskset -c 0` on every child; orchestrator pinned to
  disjoint cores `1..N-1`).
- Single-threaded execution (`OMP_NUM_THREADS=1`,
  `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `RAYON_NUM_THREADS=1`,
  `VECLIB_MAXIMUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`, verified in-process
  via `Threads: 1` in `/proc/self/status`).
- Paper budgets everywhere (`no --quick`), identical seeds `2000..=2011`
  (`n = 12`) and budgets across all arms (`cells.py`).
- Panel (`34` cells across `5` potential families):
  - `LJ` (`8`): `lj38`, `lj55`, `lj75`, `lj98`, `lj102`, `lj103`,
    `lj104`, `lj150`
  - `BLJ` (`4`, Doye-Meyer): `blj105n38`, `blj105n55`, `blj115n38`,
    `blj115n55`
  - `Au` (`2`, Gupta): `au38`, `au55`
  - `Morse` (`6`, `rho=6` and `rho=14`): `m38r6`, `m38r14`, `m55r6`,
    `m55r14`, `m80r6`, `m80r14`
  - `Water / TIP4P` (`14`): `tip4p8` through `tip4p21`
- Main schedule (`band = "main"`): balanced 12-seed 3-arm schedule over
  `(A, B, D)` where each of the 6 permutations appears twice across the
  12 seeds and the `(A, B)` pair projection is exact `ABBA` repeated
  three times (`AB, BA, BA, AB, ...`).
- Calibration / placement schedule (`band = "place"`): 7-arm Latin
  rotation over `(PA, A2, P1, P2, P3, P4, P5)` across 12 representative
  cells spanning all 5 families (`lj38`, `lj55`, `blj105n38`,
  `blj105n55`, `blj115n38`, `au38`, `m38r6`, `m38r14`, `m55r6`, `m80r6`,
  `tip4p8`, `tip4p12`), yielding `84` cell-level median calibration
  ratios (`12` same-binary `PA/A`, `12` independent-rebuild `A2/PA`, and
  `60` padded-placement `P1..P5/PA`).

## In-Process Memory and Timing Measurement

- Peak `RssAnon` (`rss_anon_kb`) and `VmHWM` (`vm_hwm_kb`) are sampled
  in-process from the benchmark child's own `/proc/self/status` after
  `exec` using a pre-allocated buffer (zero heap allocation during
  sampling): before search start, every `1,024` objective calls outside
  the potential timer, and immediately after search return while
  `Outcome`, `Ledger`, optimiser, and objective remain live.
- Sampling overhead (`status_s`) is tracked separately so
  `outside_s = max(0, wall_s - pot_s - status_s)` isolates pure
  outside-potential search time.
- Parent `wait4` `ru_maxrss_kb` and in-process `vm_hwm_kb` are recorded
  and reported for diagnostics only; neither gates.

## Pre-Registered Gate Bar

Replacement cap is `2` per cell. Any failed or environment-contaminated
attempt is kept in `rows.csv` with `superseded=1` and a reason string. If
a cell requires more than `2` replacements or lacks `12` active rows per
arm, its verdict is `INCONCLUSIVE`.

Otherwise, a cell verdict is `PASS` iff all four criteria hold (`FAIL` if
any fails):

1. **No hit loss**: `hits_B >= hits_A`, where a hit is re-derived as
   `best <= reference + 0.001`. Two-sided Fisher exact `p` and paired
   exact McNemar `p` are reported for every cell alongside reference arm
   `D` hits.
2. **Outside-time ratio**: Type-7 median outside-potential time per
   charged call (`outside_s / max(1, charged)`) satisfies
   `median(B) / median(A) <= 1.10` (paired seed-bootstrap 95% CI over
   `10,000` draws with seed `20261121`, raw `outside_s` ratio, and pooled
   `A/A + placement` band reported alongside).
3. **In-process peak `RssAnon`**: Peak `rss_anon_kb` (`Type-7` cell
   median) passes iff:
   - `|median(B) - median(A)| <= 8.0` kB, **OR**
   - `median(B) / median(A)` lies within the empirical pooled
     `A/A + placement` cell-median ratio band `[band_lo, band_hi]`, **OR**
   - `median(B) < median(A)` with `median(charged_B) < median(charged_A)`
     (strict memory reduction due to earlier global-minimum stop in `B`,
     never memory growth; any `median(B) > median(A)` must satisfy
     `|median(B) - median(A)| <= 8.0` kB or
     `median(B) / median(A) <= band_hi`).
4. **Timed `B` `cap_twist` counter**:
   - Every `B` cell (`34/34`) emits `sum(cap_twist_flag) == 12` (`1` on
     every `B` run).
   - Every atomic `B` cell (`20/20`: `lj*`, `blj*`, `au*`, `m*`) emits
     `sum(cap_escapes) > 0` and `sum(cap_arm_draws) > 0`.
   - Every rigid-body water `B` cell (`14/14`: `tip4p8..tip4p21`) emits
     `sum(cap_twist_flag) == 12`, `sum(cap_escapes) == 0`, and seed-wise
     identical `(charged, best_bits)` to `A`.
   - Every `A`, `PA`, `A2`, `P1..P5`, and `D` cell emits `0` on all
     `cap_twist` counters.

Overall gate verdict is `PASS` iff all `34` cells are `PASS` (zero `FAIL`
and zero `INCONCLUSIVE`).
