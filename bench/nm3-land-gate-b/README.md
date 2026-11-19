# nm3-land no-loss gate (part B)

**Verdict: PASS**

## Provenance and Protocol

- Arm A: `commit:5eb79cdcb0a986984dc7b487e5a3cc93224246f9` (tree `7a2ba49b37241a8fe42b2c45c226374de6d92274`)
- Arm B: `tree:2ba43832b540c6e9005cfb6be31243b36ab086a7` (tree `2ba43832b540c6e9005cfb6be31243b36ab086a7`)
- Sessions: 1 (first `20261119T124036Z`)
- Machine ID (SHA-256): `3b11c160088e7af5c4f7ebd9da95e0bfe019a34698416701bb057c15fea10873`
- CPU: `Intel(R) Xeon(R) Processor` (`x86_64`, kernel `6.12.113+`, affinity `0`)
- Toolchains: `rustc 1.88.0 (6b00bc388 2025-06-23)`, `cargo 1.88.0 (873a06493 2025-05-10)`, Python `3.12.3`
- Seeds: `2000-2011` (n=12 per system, `PAPER_BUDGETS`, `H2H_BUDGET` unset)
- Order: `ABBA_balanced_per_seed`; `(slot, arm)` counts across 576 rows: slot 1 arm A=144, slot 1 arm B=144, slot 2 arm A=144, slot 2 arm B=144
- Provenance verification (`git cat-file -e`): 9/9 recorded library and harness blob ids resolved.

### Release binary digests

| arm | binary | sha256 |
| --- | --- | --- |
| A | `h2h_sc` | `d3ace2a2588d163a3316337afff6c976b7dd2ac40a36d3b5184179e2eb02669b` |
| A | `h2h_seed` | `29277f9423bcbbc2d81e5fac38d66751216a329c11cc9aa891a9747f30a3dfe3` |
| A | `nm3-gate-driver` | `c7ff1fed7c1192af8e38673fa5c7e6a21cd2c13dd37a76d0b8c20f905ae674f7` |
| B | `h2h_sc` | `f04e0ba48c9ed78733e4269cad1795bda2121dd4a8535dd1a170217cdf02cb83` |
| B | `h2h_seed` | `33e12dbef8e50152d7bcaf38fff07c1463ae5660639d13bbacba6d3cc876ec47` |
| B | `nm3-gate-driver` | `a4bdc1a76256a19fea05f30ccd51f9722f44c3c45eedcb75e758b8750814feb8` |

## Panels

| panel | systems | hits A | hits B | Fisher / strat Fisher p | MW / van Elteren p (calls) | calls direction | wall A (s) | wall B (s) | inside A (s) | inside B (s) | outside A (s) | outside B (s) | outside/call A (us) | outside/call B (us) | outside/call B/A | same | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Sutton-Chen | 2 | 24/24 | 24/24 | 1 | 0.838256 | B fewer calls | 65.954 | 67.136 | 61.839 | 62.666 | 4.115 | 4.470 | 3.3093 | 3.5429 | 1.0706 | 0/24 | PASS |
| Morse M75 rho14 | 1 | 9/12 | 11/12 | 0.590062 | 0.224336 | B more calls | 579.907 | 896.453 | 420.952 | 654.193 | 158.955 | 242.260 | 8.8802 | 8.6352 | 0.9724 | 1/12 | PASS |
| BLJ | 7 | 46/84 | 42/84 | 0.368081 | 0.247425 | B more calls | 2920.852 | 2912.203 | 1529.004 | 1542.686 | 1391.848 | 1369.517 | 17.8453 | 17.2710 | 0.9678 | 6/84 | PASS |
| water/TIP4P rigid | 14 | 113/168 | 113/168 | 1 | 1 | equal | 1343.457 | 1341.117 | 1014.238 | 1011.573 | 329.219 | 329.544 | 2.4017 | 2.4041 | 1.0010 | 168/168 | PASS |

Multi-system panels report the two-sided stratified exact Fisher p on hits and the two-sided tie-corrected van Elteren test on calls to hit (misses entered at budget). Single-system panels report the system's exact Fisher p and tie-corrected Mann-Whitney p. Wall, inside, and outside are panel sums in seconds; outside/call is total outside-potential seconds divided by total charged calls in microseconds.

## Systems

| panel | system | budget | hits A | hits B | Fisher p | calls A | calls B | MW p calls | wall A (s) | wall B (s) | inside A (s) | inside B (s) | outside A (s) | outside B (s) | out/call A (us) | out/call B (us) | out/call B/A | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Sutton-Chen | sc96-38 | 400000 | 12/12 | 12/12 | 1 | 50230 | 47151.5 | 0.623605 | 2.075535 | 2.001406 | 1.953232 | 1.889416 | 0.134248 | 0.110846 | 2.3329 | 2.6418 | 1.1324 | PASS |
| Sutton-Chen | sc126-55 | 1000000 | 12/12 | 12/12 | 1 | 18379.5 | 25691.5 | 0.83986 | 1.258367 | 1.750164 | 1.184236 | 1.631088 | 0.075383 | 0.132953 | 4.3441 | 5.3945 | 1.2418 | PASS |
| Morse M75 rho14 | m75r14 | 4000000 | 9/12 | 11/12 | 0.590062 | 963277 | 2.6171e+06 | 0.224336 | 29.819341 | 85.231048 | 22.525011 | 60.906943 | 7.294331 | 23.596123 | 7.5048 | 8.8406 | 1.1780 | PASS |
| BLJ | blj105n38 | 400000 | 2/12 | 1/12 | 1 | 400000 | 400000 | 0.546494 | 5.628615 | 6.665383 | 1.640219 | 1.646166 | 3.962851 | 5.019345 | 9.9071 | 12.5484 | 1.2666 | PASS |
| BLJ | blj115n38 | 400000 | 12/12 | 11/12 | 1 | 82300.5 | 127812 | 0.470293 | 0.844399 | 1.243958 | 0.365752 | 0.588681 | 0.294886 | 0.582943 | 2.4071 | 4.6373 | 1.9265 | PASS |
| BLJ | blj130n38 | 400000 | 11/12 | 11/12 | 1 | 134072 | 145942 | 0.930957 | 1.122121 | 1.639161 | 0.547768 | 0.599069 | 0.676382 | 0.858648 | 6.5338 | 5.7228 | 0.8759 | PASS |
| BLJ | blj105n55 | 1000000 | 12/12 | 12/12 | 1 | 158055 | 227252 | 0.326243 | 2.244794 | 3.387493 | 1.442842 | 2.052123 | 0.641833 | 1.453636 | 3.2467 | 5.9724 | 1.8395 | PASS |
| BLJ | blj115n55 | 1000000 | 0/12 | 0/12 | 1 | 1e+06 | 1e+06 | 1 | 33.600386 | 29.868730 | 8.617968 | 8.628861 | 24.989308 | 21.242669 | 24.9893 | 21.2427 | 0.8501 | PASS |
| BLJ | blj130n55 | 1000000 | 9/12 | 7/12 | 0.666846 | 633042 | 682433 | 0.906354 | 15.121082 | 16.660163 | 5.517617 | 5.892382 | 9.037931 | 9.616551 | 17.6817 | 11.9017 | 0.6731 | PASS |
| BLJ | blj130n98 | 4000000 | 0/12 | 0/12 | 1 | 4e+06 | 4e+06 | 1 | 168.323101 | 174.350075 | 108.569695 | 109.230618 | 59.471700 | 65.630568 | 14.8679 | 16.4076 | 1.1036 | PASS |
| water/TIP4P rigid | tip4p8 | 2000000 | 12/12 | 12/12 | 1 | 6339 | 6339 | 1 | 0.020431 | 0.020125 | 0.012303 | 0.012214 | 0.008128 | 0.007912 | 1.3217 | 1.2578 | 0.9516 | PASS |
| water/TIP4P rigid | tip4p9 | 2000000 | 12/12 | 12/12 | 1 | 5839.5 | 5839.5 | 1 | 0.022512 | 0.021664 | 0.014309 | 0.013706 | 0.008203 | 0.007957 | 1.3699 | 1.3421 | 0.9797 | PASS |
| water/TIP4P rigid | tip4p10 | 2000000 | 12/12 | 12/12 | 1 | 13138.5 | 13138.5 | 1 | 0.060404 | 0.055808 | 0.039120 | 0.036181 | 0.021284 | 0.019627 | 1.5315 | 1.5030 | 0.9814 | PASS |
| water/TIP4P rigid | tip4p11 | 2000000 | 12/12 | 12/12 | 1 | 109288 | 109288 | 1 | 0.523104 | 0.515371 | 0.352610 | 0.347206 | 0.170493 | 0.168165 | 1.5919 | 1.5898 | 0.9987 | PASS |
| water/TIP4P rigid | tip4p12 | 2000000 | 12/12 | 12/12 | 1 | 41418.5 | 41418.5 | 1 | 0.251314 | 0.243106 | 0.170813 | 0.164765 | 0.080502 | 0.078341 | 1.9762 | 1.9750 | 0.9994 | PASS |
| water/TIP4P rigid | tip4p13 | 2000000 | 12/12 | 12/12 | 1 | 216690 | 216690 | 1 | 1.530682 | 1.466759 | 1.048331 | 1.006718 | 0.482352 | 0.460040 | 2.0596 | 2.0252 | 0.9833 | PASS |
| water/TIP4P rigid | tip4p14 | 2000000 | 12/12 | 12/12 | 1 | 135853 | 135853 | 1 | 0.939785 | 0.916955 | 0.659613 | 0.644267 | 0.280042 | 0.272686 | 2.0467 | 2.0501 | 1.0017 | PASS |
| water/TIP4P rigid | tip4p15 | 2000000 | 7/12 | 7/12 | 1 | 1.48162e+06 | 1.48162e+06 | 1 | 10.734158 | 10.827908 | 7.960613 | 7.881890 | 2.773546 | 2.946017 | 1.9741 | 1.9643 | 0.9950 | PASS |
| water/TIP4P rigid | tip4p16 | 2000000 | 2/12 | 2/12 | 1 | 2e+06 | 2e+06 | 1 | 16.312230 | 16.268200 | 12.007455 | 11.932368 | 4.290555 | 4.317557 | 2.1616 | 2.1641 | 1.0012 | PASS |
| water/TIP4P rigid | tip4p17 | 2000000 | 5/12 | 5/12 | 1 | 2e+06 | 2e+06 | 1 | 17.453712 | 17.394579 | 13.315139 | 13.275651 | 4.115275 | 4.117398 | 2.2488 | 2.2625 | 1.0061 | PASS |
| water/TIP4P rigid | tip4p18 | 2000000 | 2/12 | 2/12 | 1 | 2e+06 | 2e+06 | 1 | 19.255792 | 19.160418 | 14.706989 | 14.650002 | 4.466432 | 4.435299 | 2.2779 | 2.2210 | 0.9750 | PASS |
| water/TIP4P rigid | tip4p19 | 2000000 | 7/12 | 7/12 | 1 | 1.06084e+06 | 1.06084e+06 | 1 | 10.999913 | 11.021097 | 8.573443 | 8.585099 | 2.439920 | 2.435999 | 2.8266 | 2.6216 | 0.9275 | PASS |
| water/TIP4P rigid | tip4p20 | 2000000 | 6/12 | 6/12 | 1 | 1.68625e+06 | 1.68625e+06 | 1 | 19.065134 | 19.091693 | 14.935545 | 14.908524 | 4.129589 | 4.169735 | 2.8012 | 2.8711 | 1.0249 | PASS |
| water/TIP4P rigid | tip4p21 | 2000000 | 0/12 | 0/12 | 1 | 2e+06 | 2e+06 | 1 | 25.662902 | 25.670816 | 19.372422 | 19.372965 | 6.232231 | 6.282841 | 3.1161 | 3.1414 | 1.0081 | PASS |

## TIP4P bit-identity

For every rigid TIP4P `(H2O)n` size (`n = 8..21`) and seed (`2000-2011`), both arms start from identical coordinates and compare `.trace.csv` bytes, `.xyz`/`.gmin` geometry bytes, best energy value/bits, and call counts (`evals_to_hit` and `charged`).

| system | trajectory byte-identical | best energy identical | calls identical | all bit-identical |
| --- | --- | --- | --- | --- |
| tip4p8 | 12/12 | 12/12 | 12/12 | 12/12 |
| tip4p9 | 12/12 | 12/12 | 12/12 | 12/12 |
| tip4p10 | 12/12 | 12/12 | 12/12 | 12/12 |
| tip4p11 | 12/12 | 12/12 | 12/12 | 12/12 |
| tip4p12 | 12/12 | 12/12 | 12/12 | 12/12 |
| tip4p13 | 12/12 | 12/12 | 12/12 | 12/12 |
| tip4p14 | 12/12 | 12/12 | 12/12 | 12/12 |
| tip4p15 | 12/12 | 12/12 | 12/12 | 12/12 |
| tip4p16 | 12/12 | 12/12 | 12/12 | 12/12 |
| tip4p17 | 12/12 | 12/12 | 12/12 | 12/12 |
| tip4p18 | 12/12 | 12/12 | 12/12 | 12/12 |
| tip4p19 | 12/12 | 12/12 | 12/12 | 12/12 |
| tip4p20 | 12/12 | 12/12 | 12/12 | 12/12 |
| tip4p21 | 12/12 | 12/12 | 12/12 | 12/12 |

### TIP4P per-seed bit-identity matrix (`Y` = trajectory, best energy, and calls all byte-identical; `D` = differs)

| system | 2000 | 2001 | 2002 | 2003 | 2004 | 2005 | 2006 | 2007 | 2008 | 2009 | 2010 | 2011 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tip4p8 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| tip4p9 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| tip4p10 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| tip4p11 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| tip4p12 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| tip4p13 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| tip4p14 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| tip4p15 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| tip4p16 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| tip4p17 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| tip4p18 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| tip4p19 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| tip4p20 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| tip4p21 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |

Total TIP4P pairs all byte-identical: 168/168.

## Gate Rule

Any loss at `p < 0.05` on hits (exact Fisher per system, stratified exact Fisher per multi-system panel) or calls to hit (tie-corrected Mann-Whitney per system, tie-corrected van Elteren per multi-system panel), or any B-only failure, returns `FIX` naming the failing cells; otherwise `PASS`.
