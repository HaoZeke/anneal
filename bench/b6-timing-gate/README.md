# b6 timing gate

PASS if no cell loses hits (B below A) and every cell's outside-per-call ratio is at most 1.10. Otherwise FIX, naming the cells.

Verdict: PASS

Seeds 2000-2011. TIP4P budget 20000 calls. lj38 budget 400000 calls. One serial session, core 0, one thread in each pool. Arms alternate ABBA. Misses rank at budget + 1 for the calls test. Outside time per call is outside seconds divided by charged calls. The ratio is the type-7 median on B over the type-7 median on A. The interval is a paired bootstrap of 10000 replicates, seed 20261120, type-7 quantiles at 0.025 and 0.975. The pooled TIP4P p-value is the exact two-sided test of the sum of arm-A hits conditional on each size's margins. The pooled ratio uses every TIP4P row, and its bootstrap resamples inside each size.

## Verdict

| cell | hits A | hits B | Fisher p | calls A | calls B | MW U | MW p | outside/call A | outside/call B | ratio | CI low | CI high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tip4p8 | 12/12 | 12/12 | 1 | 6339 | 6339 | 72 | 1 | 1.35264e-06 | 1.31167e-06 | 0.969708 | 0.955792 | 0.98893 |
| tip4p9 | 12/12 | 12/12 | 1 | 5839.5 | 5839.5 | 72 | 1 | 1.55888e-06 | 1.5091e-06 | 0.96807 | 0.938622 | 1.02023 |
| tip4p10 | 8/12 | 8/12 | 1 | 13138.5 | 13138.5 | 72 | 1 | 1.61474e-06 | 1.57142e-06 | 0.973173 | 0.947053 | 0.993883 |
| tip4p11 | 3/12 | 3/12 | 1 | 20001 | 20001 | 72 | 1 | 1.58382e-06 | 1.58372e-06 | 0.999939 | 0.981547 | 1.01917 |
| tip4p12 | 4/12 | 4/12 | 1 | 20001 | 20001 | 72 | 1 | 1.93117e-06 | 1.97405e-06 | 1.02221 | 0.936039 | 1.06309 |
| tip4p13 | 0/12 | 0/12 | 1 | 20001 | 20001 | 72 | 1 | 2.17435e-06 | 2.17555e-06 | 1.00055 | 0.961314 | 1.07344 |
| tip4p14 | 0/12 | 0/12 | 1 | 20001 | 20001 | 72 | 1 | 2.25282e-06 | 2.23835e-06 | 0.993575 | 0.973788 | 1.00589 |
| tip4p15 | 0/12 | 0/12 | 1 | 20001 | 20001 | 72 | 1 | 2.6126e-06 | 2.5878e-06 | 0.990508 | 0.962745 | 1.04139 |
| tip4p16 | 0/12 | 0/12 | 1 | 20001 | 20001 | 72 | 1 | 2.78838e-06 | 2.6458e-06 | 0.948868 | 0.889257 | 0.984885 |
| tip4p17 | 0/12 | 0/12 | 1 | 20001 | 20001 | 72 | 1 | 2.97258e-06 | 3.02903e-06 | 1.01899 | 0.971709 | 1.03239 |
| tip4p18 | 0/12 | 0/12 | 1 | 20001 | 20001 | 72 | 1 | 3.01847e-06 | 3.02715e-06 | 1.00287 | 0.963597 | 1.03951 |
| tip4p19 | 0/12 | 0/12 | 1 | 20001 | 20001 | 72 | 1 | 3.45592e-06 | 3.2922e-06 | 0.952625 | 0.935857 | 1.05941 |
| tip4p20 | 0/12 | 0/12 | 1 | 20001 | 20001 | 72 | 1 | 3.69308e-06 | 3.56285e-06 | 0.964738 | 0.92577 | 1.05702 |
| tip4p21 | 0/12 | 0/12 | 1 | 20001 | 20001 | 72 | 1 | 3.97107e-06 | 3.96498e-06 | 0.998464 | 0.981038 | 1.03588 |
| lj38 | 12/12 | 12/12 | 1 | 127840 | 127840 | 72 | 1 | 2.74952e-06 | 2.78436e-06 | 1.01267 | 0.951814 | 1.05713 |

## Time

| cell | wall A | wall B | pot A | pot B | outside A | outside B |
| --- | --- | --- | --- | --- | --- | --- |
| tip4p8 | 0.021282 | 0.021548 | 0.012703 | 0.012674 | 0.008578 | 0.008874 |
| tip4p9 | 0.023194 | 0.022390 | 0.014253 | 0.013813 | 0.008941 | 0.008576 |
| tip4p10 | 0.058578 | 0.056338 | 0.037596 | 0.036283 | 0.020984 | 0.020055 |
| tip4p11 | 0.097142 | 0.097811 | 0.065445 | 0.066286 | 0.031474 | 0.031525 |
| tip4p12 | 0.117109 | 0.117826 | 0.079671 | 0.079999 | 0.037126 | 0.037475 |
| tip4p13 | 0.137271 | 0.138396 | 0.093775 | 0.094911 | 0.043487 | 0.043511 |
| tip4p14 | 0.147405 | 0.147179 | 0.102754 | 0.102894 | 0.045056 | 0.044767 |
| tip4p15 | 0.177695 | 0.177199 | 0.125276 | 0.125443 | 0.052252 | 0.051756 |
| tip4p16 | 0.191535 | 0.183306 | 0.135769 | 0.130852 | 0.055767 | 0.052916 |
| tip4p17 | 0.212900 | 0.214425 | 0.153429 | 0.154036 | 0.059452 | 0.060581 |
| tip4p18 | 0.218404 | 0.221209 | 0.158444 | 0.160666 | 0.060369 | 0.060543 |
| tip4p19 | 0.250011 | 0.241370 | 0.179700 | 0.175602 | 0.069118 | 0.065844 |
| tip4p20 | 0.273355 | 0.264364 | 0.198550 | 0.193371 | 0.073861 | 0.071257 |
| tip4p21 | 0.295542 | 0.294245 | 0.215556 | 0.214629 | 0.079422 | 0.079299 |
| lj38 | 0.681257 | 0.694994 | 0.319843 | 0.325708 | 0.361414 | 0.369285 |

## Pooled TIP4P

Stratified Fisher p 1. Hits 39/168 on A and 39/168 on B. Outside per call median 2.44895e-06 on A and 2.44763e-06 on B. Ratio 0.999459 with 95% interval 0.98197 to 1.02436.

## Hit status changes

No seed changed hit status in either direction.

## lj38

Best energy is identical on every lj38 seed.
Charged call count is identical on every lj38 seed.
Calls to the hit are identical on every lj38 seed.
seed 2000: best -173.92842659 vs -173.92842659 (identical), charged 124155 vs 124155 (identical), evals_to_hit 124155 vs 124155 (identical)
seed 2001: best -173.92842659 vs -173.92842659 (identical), charged 111509 vs 111509 (identical), evals_to_hit 111509 vs 111509 (identical)
seed 2002: best -173.92842659 vs -173.92842659 (identical), charged 293837 vs 293837 (identical), evals_to_hit 293837 vs 293837 (identical)
seed 2003: best -173.92842659 vs -173.92842659 (identical), charged 174442 vs 174442 (identical), evals_to_hit 174442 vs 174442 (identical)
seed 2004: best -173.92842659 vs -173.92842659 (identical), charged 131524 vs 131524 (identical), evals_to_hit 131524 vs 131524 (identical)
seed 2005: best -173.92842659 vs -173.92842659 (identical), charged 69728 vs 69728 (identical), evals_to_hit 69728 vs 69728 (identical)
seed 2006: best -173.92842659 vs -173.92842659 (identical), charged 99994 vs 99994 (identical), evals_to_hit 99994 vs 99994 (identical)
seed 2007: best -173.92842659 vs -173.92842659 (identical), charged 110660 vs 110660 (identical), evals_to_hit 110660 vs 110660 (identical)
seed 2008: best -173.92842659 vs -173.92842659 (identical), charged 325364 vs 325364 (identical), evals_to_hit 325364 vs 325364 (identical)
seed 2009: best -173.92842659 vs -173.92842659 (identical), charged 317864 vs 317864 (identical), evals_to_hit 317864 vs 317864 (identical)
seed 2010: best -173.92842659 vs -173.92842659 (identical), charged 231706 vs 231706 (identical), evals_to_hit 231706 vs 231706 (identical)
seed 2011: best -173.92842659 vs -173.92842659 (identical), charged 49500 vs 49500 (identical), evals_to_hit 49500 vs 49500 (identical)

## Schedule and build

Counted (slot, arm) pairs: (0, A) = 90, (0, B) = 90, (1, A) = 90, (1, B) = 90.

Code trees:
- A `cff796c507a8fa1d0bfb28888c3570b61988055e`
- B `14b537fd9b73b8fe21e01721954b41a278c5d223`

Binary sha256:
- A h2h_gmin_proof `4c3b76681380a124eb20379ee161b5f5474725cee01a366f185ae9337860b954`
- A h2h_seed `f25f3f054ea1dc9b06f9bd8fa08f0e77882f137ded20c1ab970c655fac4db296`
- B h2h_gmin_proof `771c35016cece95eb49b4edb1a25058c5620f0cfcd7fcc287f375832a3b72b3e`
- B h2h_seed `939f46d9c60b57f0e11f2c8f3988993f9a56f131771c8333b4bef49c324f2329`

Hashed machine id: `3b11c160088e7af5c4f7ebd9da95e0bfe019a34698416701bb057c15fea10873`.
