# CSM assignment gate

Arm A is `21cd57b6eb0b4d79ff86a5769632b65f3e6ccd82`. Arm B is that commit plus the two patches, and its tree is `31cc2b6048d8e11e4ee8fc10d36dd0588f059b2e`.

The first patch solves a forced Kuhn-Munkres assignment, below 98 atoms, by successive shortest paths, and falls back to the Hungarian solver when the two answers are not separated. It is compiled in. The second patch keeps going in the ACE factorial once the product passes 20!, which a Wigner 3j symbol needs when its three angular momenta sum to 20 or more. That is angular momentum 7 or 8. The SOAP hop on the recommended preset uses `l_max = 6`, and the rigid TIP4P runs leave SOAP off. No panel sets angular momentum 7 or 8, so the factorial change is not on these walks.

Seeds 2000-2011. The order is binary Lennard-Jones, Morse, then rigid TIP4P at 8, 12, 16, 20, and 21 molecules. Each seed runs one arm and then the other. The arm that runs first flips with the sum of the panel, system, and seed indices. One process, and one thread in every pool.

A pair matches when the hit, the evaluations to the hit, the charged count, the call count, and both energy bit patterns are equal. There are 264 pairs. Every pair matches. `rows.csv` is one row per run. `summary.csv` holds type-7 medians and the tie-corrected Mann-Whitney normal approximation with the continuity correction, per system and pooled per panel. A loss probability is the one-sided test that B is slower than A. The two-sided probabilities are the `mw_p_wall`, `mw_p_pot`, and `mw_p_outside` columns.

The smallest loss probability is 0.185422, on time inside the potential for `blj115n55`. No system and no pooled panel is below 0.05 on wall time, time inside the potential, or time outside it. The walks match and B is not slower, so the first patch stays on by default.

Pooled panels. Medians are seconds. A is the base, B is the patched tree.

| panel | n | wall A | wall B | wall loss | pot A | pot B | pot loss | outside A | outside B | outside loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| blj | 84 | 49.761104 | 49.150802 | 0.48292 | 1.608693 | 1.594185 | 0.495571 | 46.124635 | 47.641869 | 0.486714 |
| morse | 120 | 16.257816 | 16.182476 | 0.60162 | 11.591219 | 11.625465 | 0.472203 | 4.714766 | 4.436894 | 0.7379 |
| rigid | 60 | 8.590050 | 8.580583 | 0.5272 | 6.631351 | 6.624190 | 0.54076 | 1.958699 | 1.956393 | 0.507328 |

Per system. Every row matched.

| panel | system | wall A | wall B | wall loss | pot A | pot B | pot loss | outside A | outside B | outside loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| blj | blj105n38 | 35.483333 | 35.593654 | 0.534506 | 1.496951 | 1.483145 | 0.727815 | 33.981643 | 33.981531 | 0.534506 |
| blj | blj105n55 | 15.100261 | 14.975537 | 0.465494 | 1.715102 | 1.680889 | 0.465494 | 12.856943 | 12.914093 | 0.511515 |
| blj | blj115n38 | 2.208830 | 2.207775 | 0.41993 | 0.511024 | 0.512715 | 0.41993 | 1.649369 | 1.656680 | 0.397506 |
| blj | blj115n55 | 282.885108 | 291.211748 | 0.41993 | 8.075101 | 8.176844 | 0.185422 | 274.843628 | 282.919561 | 0.442617 |
| blj | blj130n38 | 21.691928 | 21.179968 | 0.488485 | 1.454560 | 1.467790 | 0.58007 | 20.215117 | 19.705340 | 0.488485 |
| blj | blj130n55 | 116.160072 | 117.260829 | 0.557383 | 7.264231 | 7.274694 | 0.353727 | 108.688699 | 109.691803 | 0.534506 |
| blj | blj130n98 | 1399.204328 | 1392.950703 | 0.557383 | 91.126145 | 90.956703 | 0.843839 | 1307.563866 | 1301.636493 | 0.557383 |
| morse | m55r6 | 0.447867 | 0.404932 | 0.74664 | 0.284838 | 0.285163 | 0.442617 | 0.166217 | 0.121009 | 0.829611 |
| morse | m55r14 | 7.500773 | 7.380097 | 0.667497 | 5.660234 | 5.660884 | 0.465494 | 1.840539 | 1.719213 | 0.727815 |
| morse | m70r3 | 10.582240 | 10.519279 | 0.58007 | 7.488536 | 7.496099 | 0.442617 | 3.093704 | 3.023179 | 0.624584 |
| morse | m70r6 | 1.973115 | 1.798418 | 0.646273 | 1.330064 | 1.331829 | 0.488485 | 0.648883 | 0.466589 | 0.843839 |
| morse | m70r10 | 27.252071 | 26.972024 | 0.764757 | 19.933872 | 20.012049 | 0.397506 | 7.140649 | 6.796648 | 0.798748 |
| morse | m70r14 | 10.464230 | 10.092426 | 0.667497 | 7.461756 | 7.462012 | 0.511515 | 3.002474 | 2.630414 | 0.74664 |
| morse | m80r3 | 33.167054 | 33.088618 | 0.602494 | 23.907500 | 23.909745 | 0.442617 | 9.259555 | 9.178873 | 0.624584 |
| morse | m80r6 | 2.625941 | 2.267347 | 0.688198 | 1.663769 | 1.665399 | 0.511515 | 0.962171 | 0.601947 | 0.814578 |
| morse | m80r10 | 124.698747 | 123.930464 | 0.602494 | 92.344688 | 92.488894 | 0.332503 | 32.299920 | 31.441571 | 0.667497 |
| morse | m80r14 | 138.825092 | 138.794429 | 0.764757 | 103.006336 | 103.168320 | 0.375416 | 35.057868 | 33.996112 | 0.782135 |
| rigid | tip4p8 | 0.019817 | 0.019891 | 0.56877 | 0.011933 | 0.011957 | 0.534506 | 0.007885 | 0.007935 | 0.557383 |
| rigid | tip4p12 | 0.222220 | 0.221506 | 0.511515 | 0.150874 | 0.150362 | 0.534506 | 0.071346 | 0.071144 | 0.442617 |
| rigid | tip4p16 | 15.935695 | 15.988739 | 0.602494 | 11.788243 | 11.776177 | 0.74664 | 4.155288 | 4.204282 | 0.557383 |
| rigid | tip4p20 | 18.922044 | 18.874634 | 0.534506 | 14.798741 | 14.771353 | 0.70832 | 4.095197 | 4.086923 | 0.511515 |
| rigid | tip4p21 | 25.156150 | 25.069953 | 0.624584 | 19.168171 | 19.157753 | 0.442617 | 5.958557 | 5.939668 | 0.557383 |
