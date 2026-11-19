# Shelf gate

Verdict: **LOSES (opt-in only)**.

A, B and C are respectively develop, the shelf branch, and the shelf branch plus the four supplied patches.
C's tree after applying the patches is `b17735cfa8c81727c596d6e20a696b5b9678fb50` (expected `b17735cfa8c81727c596d6e20a696b5b9678fb50`).
Revisions: A `21cd57b6eb0b4d79ff86a5769632b65f3e6ccd82`, B `849d97b8cee6fda8a7b211923e93926033a3c074`, C `3b6d12b64acd4fa9c070e9cc6719ec832d02edcb`.

All systems use seeds 2000–2011 (n=12), paper call budgets, the recommended preset, one process at a time on core 0, and one thread. Arms rotate A/B/C, B/C/A, C/A/B by seed. Every row was recorded under the synthetic host `bench`.

## Per-arm results

| system | A hits | A calls | A inside / outside (s) | B hits | B calls | B inside / outside (s) | C hits | C calls | C inside / outside (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lj38 | 12/12 | 6357.0 | 0.014427 / 0.037672 | 12/12 | 6357.0 | 0.015073 / 0.038943 | 12/12 | 6357.0 | 0.014714 / 0.038939 |
| lj75 | 12/12 | 44176.0 | 0.394331 / 0.649282 | 12/12 | 36102.0 | 0.315140 / 0.607561 | 12/12 | 46107.5 | 0.400964 / 0.633186 |
| lj76 | 12/12 | 65986.5 | 0.605276 / 0.998497 | 12/12 | 37509.0 | 0.329817 / 0.715024 | 12/12 | 86795.0 | 0.757017 / 1.048149 |
| lj77 | 12/12 | 760807.5 | 6.905847 / 8.118464 | 9/12 | 2256480.0 | 21.562616 / 24.705744 | 10/12 | 1706995.5 | 16.052793 / 19.495705 |
| lj98 | 10/12 | 1321410.0 | 20.979368 / 20.393151 | 10/12 | 1515555.5 | 23.121685 / 22.065043 | 11/12 | 2061244.5 | 30.072907 / 29.503764 |
| lj102 | 8/12 | 1377927.0 | 21.952983 / 21.947289 | 11/12 | 1163409.5 | 18.369163 / 18.959206 | 5/12 | 4000000.0 | 61.984742 / 59.038096 |
| lj103 | 8/12 | 2581470.5 | 48.966858 / 47.165538 | 10/12 | 1163818.0 | 23.966193 / 24.558829 | 7/12 | 2819038.5 | 49.263368 / 57.032998 |
| lj104 | 9/12 | 1136745.5 | 18.632576 / 20.229913 | 5/12 | 4000000.0 | 65.066853 / 64.366541 | 8/12 | 3129200.0 | 52.769085 / 60.638734 |
| binary-lj | 0/12 | 2000000.0 | 49.354859 / 49.091936 | 0/12 | 2000000.0 | 48.434186 / 48.130418 | 0/12 | 2000000.0 | 49.337653 / 48.872462 |
| m80r6 | 12/12 | 238328.0 | 6.368269 / 2.929898 | 12/12 | 142298.0 | 3.800185 / 2.052279 | 12/12 | 171542.0 | 4.590058 / 2.052821 |
| m80r14 | 8/12 | 3477864.0 | 92.342202 / 44.102588 | 3/12 | 4000000.0 | 105.004638 / 47.573110 | 3/12 | 4000000.0 | 105.031643 / 48.093901 |

## Pairwise gate

| system | comparison | hits C / other | Fisher p | median calls C / other | Mann–Whitney p | hit drop | call increase |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| lj38 | C-vs-A | 12 / 12 | 1.000000 | 6357.0 / 6357.0 | 0.976920 | false | false |
| lj38 | C-vs-B | 12 / 12 | 1.000000 | 6357.0 / 6357.0 | 0.976920 | false | false |
| lj75 | C-vs-A | 12 / 12 | 1.000000 | 46107.5 / 44176.0 | 0.750517 | false | true |
| lj75 | C-vs-B | 12 / 12 | 1.000000 | 46107.5 / 36102.0 | 0.563108 | false | true |
| lj76 | C-vs-A | 12 / 12 | 1.000000 | 86795.0 / 65986.5 | 0.418620 | false | true |
| lj76 | C-vs-B | 12 / 12 | 1.000000 | 86795.0 / 37509.0 | 0.105741 | false | true |
| lj77 | C-vs-A | 10 / 12 | 0.478261 | 1706995.5 / 760807.5 | 0.088465 | true | true |
| lj77 | C-vs-B | 10 / 9 | 1.000000 | 1706995.5 / 2256480.0 | 0.234539 | false | false |
| lj98 | C-vs-A | 11 / 10 | 1.000000 | 2061244.5 / 1321410.0 | 0.340358 | false | true |
| lj98 | C-vs-B | 11 / 10 | 1.000000 | 2061244.5 / 1515555.5 | 0.435331 | false | true |
| lj102 | C-vs-A | 5 / 8 | 0.413649 | 4000000.0 / 1377927.0 | 0.447915 | true | true |
| lj102 | C-vs-B | 5 / 11 | 0.027191 | 4000000.0 / 1163409.5 | 0.263754 | true | true |
| lj103 | C-vs-A | 7 / 8 | 1.000000 | 2819038.5 / 2581470.5 | 0.789575 | true | true |
| lj103 | C-vs-B | 7 / 10 | 0.370709 | 2819038.5 / 1163818.0 | 0.188460 | true | true |
| lj104 | C-vs-A | 8 / 9 | 1.000000 | 3129200.0 / 1136745.5 | 0.254372 | true | true |
| lj104 | C-vs-B | 8 / 5 | 0.413649 | 3129200.0 / 4000000.0 | 0.693119 | false | false |
| binary-lj | C-vs-A | 0 / 0 | 1.000000 | 2000000.0 / 2000000.0 | 1.000000 | false | false |
| binary-lj | C-vs-B | 0 / 0 | 1.000000 | 2000000.0 / 2000000.0 | 1.000000 | false | false |
| m80r6 | C-vs-A | 12 / 12 | 1.000000 | 171542.0 / 238328.0 | 1.000000 | false | false |
| m80r6 | C-vs-B | 12 / 12 | 1.000000 | 171542.0 / 142298.0 | 0.772493 | false | true |
| m80r14 | C-vs-A | 3 / 8 | 0.099533 | 4000000.0 / 3477864.0 | 0.095392 | true | true |
| m80r14 | C-vs-B | 3 / 3 | 1.000000 | 4000000.0 / 4000000.0 | 0.939502 | false | false |

## Every regression direction

- lj75 C-vs-A: median calls 46107.5 against 44176.0 (Mann–Whitney p 0.750517).
- lj75 C-vs-B: median calls 46107.5 against 36102.0 (Mann–Whitney p 0.563108).
- lj76 C-vs-A: median calls 86795.0 against 65986.5 (Mann–Whitney p 0.418620).
- lj76 C-vs-B: median calls 86795.0 against 37509.0 (Mann–Whitney p 0.105741).
- lj77 C-vs-A: hits 10 against 12 (Fisher p 0.478261); median calls 1706995.5 against 760807.5 (Mann–Whitney p 0.088465).
- lj98 C-vs-A: median calls 2061244.5 against 1321410.0 (Mann–Whitney p 0.340358).
- lj98 C-vs-B: median calls 2061244.5 against 1515555.5 (Mann–Whitney p 0.435331).
- lj102 C-vs-A: hits 5 against 8 (Fisher p 0.413649); median calls 4000000.0 against 1377927.0 (Mann–Whitney p 0.447915).
- lj102 C-vs-B: hits 5 against 11 (Fisher p 0.027191); median calls 4000000.0 against 1163409.5 (Mann–Whitney p 0.263754).
- lj103 C-vs-A: hits 7 against 8 (Fisher p 1.000000); median calls 2819038.5 against 2581470.5 (Mann–Whitney p 0.789575).
- lj103 C-vs-B: hits 7 against 10 (Fisher p 0.370709); median calls 2819038.5 against 1163818.0 (Mann–Whitney p 0.188460).
- lj104 C-vs-A: hits 8 against 9 (Fisher p 1.000000); median calls 3129200.0 against 1136745.5 (Mann–Whitney p 0.254372).
- m80r6 C-vs-B: median calls 171542.0 against 142298.0 (Mann–Whitney p 0.772493).
- m80r14 C-vs-A: hits 3 against 8 (Fisher p 0.099533); median calls 4000000.0 against 3477864.0 (Mann–Whitney p 0.095392).

The verdict changes to LOSES only for a hit drop with two-sided Fisher p < 0.05 or a median call increase with tie-corrected two-sided Mann–Whitney p < 0.05.
