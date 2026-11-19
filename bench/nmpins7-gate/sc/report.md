# nmpins7 gate, shard sc

Same session, one core, one thread. Seeds 2000-2011. Hit window +0.001.
Wall cap 7200 s. A capped seed is a miss. Medians and IQRs are Hyndman-Fan type 7.
B versus A loses on hits when the one-sided Fisher probability that B has fewer hits,
multiplied by 30, is below 0.05; on calls when the one-sided Mann-Whitney
probability that B's calls-to-hit are higher, multiplied by 30, is below 0.05;
and on outside time when the median outside-potential time per call of B exceeds 1.1 times A.
C versus A is the same arithmetic with no pass or fail.

BLJ7 is not a name in the panel tables. The cell measured here is blj105n38,
budget 400000, reference -177.2090846748. sc126-55 and sc96-38 use the budgets
and references in `scripts/metal_h2h/run_sc.py`.

## Provenance

Base `5eb79cdcb0a986984dc7b487e5a3cc93224246f9`. Arm A source `5eb79cdcb0a986984dc7b487e5a3cc93224246f9`.
Kit source `4cd811004c67f26b0e216664cb51352f04654c09`, tree `51c1a7e0180585d078c3457b5550b2963d1ab752`.
Kit build `0d14de6b013729062b7b679b6e4bccf853565d55` reads `general_symmetry` and is the binary for B and C.
Binary Lennard-Jones driver commit `8de2a95c3ff3eb2b173b88b4a8cb48c144cbfe19`, blob `85916cc5e6f83d0bb1854bebf8c9bb6c27bd9b50`.
On this commit `BinaryLennardJones::doye_meyer` is `cfg(test)`, so the built driver calls `BinaryLennardJones::new` with that constructor's arguments.
rgpot `8efdd246322d1f32bc1bcbd7e5dfdd7a4237960e`, librgpot.so.3.7.0 sha256 `4afdef0a0c9eefcc4f9b9f3c7466a71caf2db98e4615603502731d6014d8e200`.
h2h_sc A sha256 `bb281aca4b03caca3c37475a78830ea8f78edbe62c99c9b281067d11582dc0e0`.
h2h_sc kit sha256 `b2ce470f64b4d6e7634453e35eb9f25323a9595294c1e66d7c6c7e8d373f3608`.
h2h_blj A sha256 `fa63f2b9601f85ae4a7e02d337264abedf14ed2d807908f1c8ec47f010c4b217`.
h2h_blj kit sha256 `3b7077b6a1953044f5088e3b6f4096e1189617801281350f44e707c42463a35d`.
rustc `rustc 1.88.0 (6b00bc388 2025-06-23)`.
Packages: anneal-core 0.12.0, readcon-core 0.14.8, rgmin 0.5.0, rgpot-core 3.5.1, rgpot-pair 3.7.0.
Machine id sha256 `3b11c160088e7af5c4f7ebd9da95e0bfe019a34698416701bb057c15fea10873`. Core 0.
Thread variables set to 1: OMP_NUM_THREADS, OMP_THREAD_LIMIT, MKL_NUM_THREADS, MKL_DOMAIN_NUM_THREADS, OPENBLAS_NUM_THREADS, GOTO_NUM_THREADS, NUMBA_NUM_THREADS, VECLIB_MAXIMUM_THREADS, BLIS_NUM_THREADS, RAYON_NUM_THREADS, NUMEXPR_NUM_THREADS, TBB_NUM_THREADS, JULIA_NUM_THREADS, KMP_NUM_THREADS.
Session `3900fa53758e368d`.

## Results

| cell | arm | hits | median calls (IQR) | median outside/call (IQR) |
| --- | --- | --- | --- | --- |
| sc126-55 | A | 12/12 | 18379.5 (32640.8) | 4.768e-06 (8.59e-07) |
| sc126-55 | B | 12/12 | 18379.5 (32640.8) | 4.7395e-06 (9.2725e-07) |
| sc126-55 | C | 12/12 | 18379.5 (32658.8) | 4.8455e-06 (1.0165e-06) |
| sc96-38 | A | 12/12 | 50230 (33454.5) | 2.373e-06 (4.0075e-07) |
| sc96-38 | B | 12/12 | 50230 (33454.5) | 2.423e-06 (3.6975e-07) |
| sc96-38 | C | 12/12 | 50230 (33454.5) | 2.439e-06 (4.425e-07) |
| BLJ7 | A | 2/12 | 400000 (0) | 1.1397e-05 (4.71725e-06) |
| BLJ7 | B | 2/12 | 400000 (0) | 1.14285e-05 (4.44475e-06) |
| BLJ7 | C | 2/12 | 400000 (0) | 1.15685e-05 (5.22075e-06) |

## B versus A

B matched A on calls and on the hit bit for every seed. The call samples are tied, so the Mann-Whitney probability is 1.

sc126-55: hits 12/12 against 12/12, Fisher one-sided 1, Bonferroni 1, two-sided 1. Calls one-sided 1, Bonferroni 1, two-sided 1. Outside median ratio 0.994023, Mann-Whitney two-sided 0.885234, Bonferroni 1, one-sided higher 0.442617, Bonferroni 1. The ratio is inside the A/A interquartile range. Verdict: passes.
sc96-38: hits 12/12 against 12/12, Fisher one-sided 1, Bonferroni 1, two-sided 1. Calls one-sided 1, Bonferroni 1, two-sided 1. Outside median ratio 1.02107, Mann-Whitney two-sided 0.930987, Bonferroni 1, one-sided higher 0.465494, Bonferroni 1. The ratio is inside the A/A interquartile range. Verdict: passes.
BLJ7: hits 2/12 against 2/12, Fisher one-sided 0.704969, Bonferroni 1, two-sided 1. Calls one-sided 1, Bonferroni 1, two-sided 1. Outside median ratio 1.00276, Mann-Whitney two-sided 0.97697, Bonferroni 1, one-sided higher 0.488485, Bonferroni 1. The ratio is inside the A/A interquartile range. Verdict: passes.

## C versus A

sc126-55: hits 12/12 against 12/12, Fisher one-sided 1, Bonferroni 1, two-sided 1. Calls one-sided 0.5, Bonferroni 1, two-sided 1. Outside median ratio 1.01625, Mann-Whitney two-sided 0.97697, Bonferroni 1, one-sided higher 0.511515, Bonferroni 1.
sc96-38: hits 12/12 against 12/12, Fisher one-sided 1, Bonferroni 1, two-sided 1. Calls one-sided 0.5, Bonferroni 1, two-sided 1. Outside median ratio 1.02781, Mann-Whitney two-sided 0.885234, Bonferroni 1, one-sided higher 0.442617, Bonferroni 1.
BLJ7: hits 2/12 against 2/12, Fisher one-sided 0.704969, Bonferroni 1, two-sided 1. Calls one-sided 0.5, Bonferroni 1, two-sided 1. Outside median ratio 1.01505, Mann-Whitney two-sided 0.795012, Bonferroni 1, one-sided higher 0.397506, Bonferroni 1.

## A/A noise band on BLJ7

Twelve pairs of the arm A binary, order ABBA. Per-seed ratio of outside time per call, label A2 over label A: median 1.01167, IQR 0.977516 to 1.02721, range 0.935685 to 1.11598. The ratio is outside time per call of label A2 divided by label A.

## Slot counts

Each of A, B and C occupies each of the three slots four times on every cell.
On the BLJ7 A/A series each of A and A2 occupies each of the two slots six times.
