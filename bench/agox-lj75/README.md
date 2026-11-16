Copyright (c) anneal developers

LJ75, seeds 6400-6405, budget 4,000,000, host bench.
The reference minimum is -397.492331, the Marks decahedron.
The rows were measured in one fresh session. No log was copied or reused
from an earlier kit. This record is based on develop
98dcfb3106f191baf7381fecd5624426831094fe.

Current AGOX and documented AGOX are version 3.11.1, source
0a5136605161aaff2610584cd93ae6e9564fda46. The wheel that produced these
rows has sha256
5ecb3640fa1701c56007ed267573a8c62fdc3c6c69e15c77428b62516aad3eff.
The tree pin
f8b4f49b0e5a053c1b69205633c8b19523cd188b4001a12f093aaaa7f6250457
was not the file on that machine. The documented arm keeps the stock
box-force adjustment.
Anneal code_revision ccc22cee607b18463b3891dd2f9048810d671ac0, built
with rustc 1.88.0 (6b00bc388 2025-06-23) and cargo 1.88.0 (873a06493
2025-05-10). That binary is `lj_cluster_search` on `Config::for_cluster`.
A rerun on this tree is `h2h_arm3 lj75 default`, the same preset.
`h2h_arm3` does not time the potential, so a newly appended anneal row
leaves the inside and outside columns empty. The rows below keep the
times from the measured binary.

| arm | hits | median wall s | median best | gap to reference | AGOX/anneal | anneal/AGOX |
| --- | --- | --- | --- | --- | --- | --- |
| current AGOX | 0/6 | 5655.108823 | -392.980553 | 4.511777 | 103.663412 | 0.009647 |
| documented AGOX | 0/6 | 14493.759020 | -372.106043 | 25.386287 | 263.511422 | 0.003795 |
| anneal | 0/6 | 55.299841 | -396.282249 | 1.210082 | 1.000000 | 1.000000 |

Medians are type 7, the mean of the 3rd and 4th of the six ordered values.
Wall ratios are paired by seed. The gap is the median best minus
-397.492331.

## Why anneal is 0/6

The system is LJ75. The budget is 4,000,000 charged evaluations. The
reference is -397.492331. Anneal's median best is -396.282249, which is
1.210082 above that minimum. Current AGOX's median best is -392.980553,
4.511777 above it. Documented AGOX's median best is -372.106043,
25.386287 above it. Every arm is 0/6.

The anneal arm is the plain Wales-Doye basin hop, `Config::for_cluster`.
A hop displaces atoms and quenches. That search has depth one: from the
structure a 75-point chain settles into, a single displacement does not
reach a lower packing. Wales and Doye put the barriers between the
icosahedral funnel and the Marks funnel at 8.69 and 7.48 epsilon. Four of
the six anneal seeds stop on -396.282249, the icosahedral floor named in
the hopper. The other two stop higher, at -396.191649 and -394.628650.
The chain spends the budget inside that funnel.

The general exit from an occupied packing is the Leave walk already in
the crate. It is not switched on by `Config::for_cluster`, and this
record does not switch it on. That change was not gated here, serially,
at n greater than or equal to 6, against a same-session develop baseline.

`scripts/agox_lj75/summarize.py` reprints the medians and the hit counts.
`scripts/agox_lj75/enter_bench.sh` appends one new seed and leaves an
existing row untouched.
