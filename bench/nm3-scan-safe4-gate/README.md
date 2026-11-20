# nm3-scan-safe4 Gate Report

**Overall Verdict: PASS**

- Part 1 (Symmetry Detection Microbench): **PASS**
- Part 2 (Default-On Panel, `PAPER_BUDGETS`): **PASS**

## Provenance

- Hashed machine id(s) (SHA-256): `3b11c160088e7af5c4f7ebd9da95e0bfe019a34698416701bb057c15fea10873`
- Part 1 arm trees: `A':e7cae35817aaaf404e103b7995b12c9d60db5bcd`, `BASE:e7cae35817aaaf404e103b7995b12c9d60db5bcd`, `SAFE2:93029bd7aa0113eb943dac7844e72fc5525b8fec`, `SAFE4:578a12095ab5e0e6ac89c805173db78a9d1578b7`
- Part 2 arm trees: `A':e7cae35817aaaf404e103b7995b12c9d60db5bcd`, `BASE:e7cae35817aaaf404e103b7995b12c9d60db5bcd`, `SAFE4:578a12095ab5e0e6ac89c805173db78a9d1578b7`
- Part 1 microbench binary SHA-256s:
  - `A':70a67351bdb84d53a572e51b724de0b7d30f2014f6da261f43097cdda0a3fbae`
  - `BASE:70a67351bdb84d53a572e51b724de0b7d30f2014f6da261f43097cdda0a3fbae`
  - `SAFE2:bbcb6f23f61cdbfff74c633265a2ac0cca533fb5739a981668c9bc006a587e5b`
  - `SAFE4:d38d97d124915021458685a4018915d8bc9c5e1aaf2fa2647d1ad1838a9178a3`
- Part 2 binary SHA-256s:
  - `A':h2h_gmin_proof:2a75b59d9cc6fa101d4330913c25a74bfb1a4cf2cbb6ba4ffe8d4fb1fc732027`
  - `A':h2h_metal:1ecde305398bea47fdc94be71153a692111223e535aee671032d9599c6045753`
  - `A':h2h_morse:1c7ad248e8eb741ad2d6f1caf724d7a8ac4d28abb85003aa20f914c0418efef3`
  - `A':h2h_sc:ed951abd389f15923e9236eb087a12c1e8aa4b8299dfd43c91b11df33f60dfc1`
  - `A':h2h_seed:8a467ad22205410ddf35f9b480d3774adf9001122d2fd32b7a8d1ab638a5ed82`
  - `A':nm3-gate-driver:1f0a6d60a6e71804ee7e9b9af47d6dfcd9559e76f8c4b84424e5b66098ca075b`
  - `BASE:h2h_gmin_proof:e5c22d2e5ed58ad92d8d0de33f6db2b5ca2ad995156d522ed6e9858db3c0cf07`
  - `BASE:h2h_metal:0e99fdb9395b61427da07de799821a1c5e575770ebc4ae3915159f803d11c8b5`
  - `BASE:h2h_morse:0b584c524dbb7c8fcc5fac10178197b72d6e21397f283b01166a78e351b48052`
  - `BASE:h2h_sc:f139617452a1052f713a04dbd954c97b7e7830b597b0a9a41abb7743affd099f`
  - `BASE:h2h_seed:8955899ad38d5cce6b0ac1cae7a5dc994c0da79865a974fff938835bb8ae1b96`
  - `BASE:nm3-gate-driver:95a776fc7cc037045941331dc9a79d2b3e802418ce9ace65d7bf9b31774dd716`
  - `SAFE4:h2h_gmin_proof:3326422a30690d9fe61ac601e5bffa5c05014394e85e1f8e340f5da22958b63a`
  - `SAFE4:h2h_metal:37f219f5c97adab9a7f97687b1df9c3c7c85ed9e1f8309a0a853f3d4de43c4d3`
  - `SAFE4:h2h_morse:60195d928bf00515168b16f86fc522dad56b35bea5e0365ef59f137cb5b29b51`
  - `SAFE4:h2h_sc:ba2d5b8f773aeb5368a1b24b46e7cec22a65e54f86a8f99edcf2a34630ba8c19`
  - `SAFE4:h2h_seed:899e635354d11af36ac8afad11b9ee78cecc56f3e84e22ef79f33e2f8a16d08d`
  - `SAFE4:nm3-gate-driver:886e60cba4f5a0c21c54599b99f80470bc7e7efec9400af5268e8a563ff5537f`

## Part 1: Symmetry Detection Microbench

Cases: `detect_all` and `detect_orders` on `random33`, `random55`, `random38` (`orders 2..=38`), `ring55`, and `packed80` (`orders 2..=12` for the others, `tolerance = 0.1`).
Rounds: `12` interleaved rounds across `BASE`, `A'`, `SAFE2`, `SAFE4` (`480` rows total; `(slot, arm)` count = `[30]`).

| case | BASE (us) | A' (us) | SAFE2 (us) | SAFE4 (us) | SAFE4/BASE | SAFE2/BASE | SAFE4/SAFE2 | A'/BASE | bit-identical | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | :---: |
| `detect_all:random33` | 18.267 | 18.363 | 21.561 | 16.367 | 0.8960 | 1.1803 | 0.7591 | 1.0052 | YES | PASS |
| `detect_all:random55` | 35.585 | 35.605 | 42.642 | 33.881 | 0.9521 | 1.1983 | 0.7945 | 1.0006 | YES | PASS |
| `detect_all:random38` | 80.720 | 80.895 | 93.892 | 76.441 | 0.9470 | 1.1632 | 0.8141 | 1.0022 | YES | PASS |
| `detect_all:ring55` | 609.154 | 611.183 | 401.830 | 386.808 | 0.6350 | 0.6597 | 0.9626 | 1.0033 | YES | PASS |
| `detect_all:packed80` | 949.954 | 949.600 | 894.167 | 878.361 | 0.9246 | 0.9413 | 0.9823 | 0.9996 | YES | PASS |
| `detect_orders:random33` | 19.753 | 19.521 | 20.816 | 18.419 | 0.9325 | 1.0538 | 0.8849 | 0.9883 | YES | PASS |
| `detect_orders:random55` | 38.279 | 38.228 | 40.968 | 36.438 | 0.9519 | 1.0702 | 0.8894 | 0.9987 | YES | PASS |
| `detect_orders:random38` | 92.748 | 92.782 | 95.086 | 88.696 | 0.9563 | 1.0252 | 0.9328 | 1.0004 | YES | PASS |
| `detect_orders:ring55` | 566.563 | 568.893 | 351.672 | 346.075 | 0.6108 | 0.6207 | 0.9841 | 1.0041 | YES | PASS |
| `detect_orders:packed80` | 899.395 | 898.488 | 831.775 | 818.569 | 0.9101 | 0.9248 | 0.9841 | 0.9990 | YES | PASS |

## Part 2: Default-On Panel (`PAPER_BUDGETS`, `H2H_*` unset)

Cells: `33` systems across LJ, Metal/Au, SC, Morse, BLJ, and TIP4P (`1188` rows total; `(slot, arm)` count = `[132]`).
Hit criterion: finite `best <= reference + 1e-3`. Misses ranked at `budget` for Mann-Whitney.
`out/call` is computed per row as `outside_s / charged`; ratios compare per-cell medians (`SAFE4/BASE` vs `A'/BASE`).
Note: `for_tip4p` disables `symmetrise_on_stall` (`symmetrise_on_stall = false`), so `tip4p8..21` serve as bit-identical panel controls alongside the active symmetrise cells.

| cell | hits BASE / SAFE4 / A' | Fisher p | med calls BASE / SAFE4 / A' | MW p (calls) | out/call med BASE / SAFE4 / A' (us) | out/call SAFE4/BASE | out/call A'/BASE | bit-id (best_bits & charged) | verdict |
| --- | :---: | ---: | :---: | ---: | :---: | ---: | ---: | :---: | :---: |
| `lj38` | 12/12/12 | 1 | 164252/164252/164252 | 1 | 3.434/2.729/3.358 | 0.7946 | 0.9777 | 12/12 | PASS |
| `lj55` | 12/12/12 | 1 | 6862/6862/6862 | 1 | 7.899/6.521/8.395 | 0.8256 | 1.0628 | 12/12 | PASS |
| `au38` | 10/10/10 | 1 | 166896/166896/166896 | 1 | 5.161/4.993/5.046 | 0.9673 | 0.9777 | 12/12 | PASS |
| `sc96-38` | 12/12/12 | 1 | 78840/78840/78840 | 1 | 3.160/3.049/3.377 | 0.9650 | 1.0687 | 12/12 | PASS |
| `sc126-55` | 12/12/12 | 1 | 57371/57371/57371 | 1 | 6.762/5.850/6.763 | 0.8652 | 1.0002 | 12/12 | PASS |
| `m80r6` | 12/12/12 | 1 | 164834/164834/164834 | 1 | 21.116/13.843/21.843 | 0.6556 | 1.0344 | 12/12 | PASS |
| `blj115n38` | 11/11/11 | 1 | 94588/94588/94588 | 1 | 3.469/3.467/3.495 | 0.9995 | 1.0076 | 12/12 | PASS |
| `blj130n38` | 6/6/6 | 1 | 307644/307644/307644 | 1 | 6.025/6.039/5.804 | 1.0023 | 0.9633 | 12/12 | PASS |
| `blj105n38` | 2/2/2 | 1 | 400000/400000/400000 | 1 | 16.117/16.300/16.256 | 1.0114 | 1.0087 | 12/12 | PASS |
| `blj105n55` | 12/12/12 | 1 | 240536/240536/240536 | 1 | 10.217/10.437/10.626 | 1.0216 | 1.0401 | 12/12 | PASS |
| `blj130n55` | 11/11/11 | 1 | 514816/514816/514816 | 1 | 20.779/20.267/19.939 | 0.9754 | 0.9596 | 12/12 | PASS |
| `blj115n55` | 0/0/0 | 1 | 1000000/1000000/1000000 | 1 | 28.226/29.703/28.670 | 1.0523 | 1.0157 | 12/12 | PASS |
| `tip4p8` | 12/12/12 | 1 | 6339/6339/6339 | 1 | 1.365/1.376/1.363 | 1.0080 | 0.9990 | 12/12 | PASS |
| `tip4p9` | 12/12/12 | 1 | 5840/5840/5840 | 1 | 1.409/1.408/1.384 | 0.9992 | 0.9822 | 12/12 | PASS |
| `tip4p10` | 12/12/12 | 1 | 13138/13138/13138 | 1 | 1.475/1.503/1.475 | 1.0192 | 1.0001 | 12/12 | PASS |
| `tip4p11` | 12/12/12 | 1 | 109288/109288/109288 | 1 | 1.566/1.588/1.571 | 1.0145 | 1.0033 | 12/12 | PASS |
| `tip4p12` | 12/12/12 | 1 | 41418/41418/41418 | 1 | 1.862/1.860/1.883 | 0.9986 | 1.0113 | 12/12 | PASS |
| `tip4p13` | 12/12/12 | 1 | 216690/216690/216690 | 1 | 1.972/1.960/1.919 | 0.9944 | 0.9735 | 12/12 | PASS |
| `tip4p14` | 12/12/12 | 1 | 135853/135853/135853 | 1 | 2.047/2.072/2.068 | 1.0124 | 1.0101 | 12/12 | PASS |
| `tip4p15` | 7/7/7 | 1 | 1481620/1481620/1481620 | 1 | 1.974/2.005/1.971 | 1.0158 | 0.9982 | 12/12 | PASS |
| `tip4p16` | 2/2/2 | 1 | 2000000/2000000/2000000 | 1 | 2.313/2.292/2.277 | 0.9910 | 0.9845 | 12/12 | PASS |
| `tip4p17` | 5/5/5 | 1 | 2000000/2000000/2000000 | 1 | 2.387/2.442/2.439 | 1.0230 | 1.0217 | 12/12 | PASS |
| `tip4p18` | 2/2/2 | 1 | 2000000/2000000/2000000 | 1 | 2.380/2.296/2.351 | 0.9647 | 0.9879 | 12/12 | PASS |
| `tip4p19` | 7/7/7 | 1 | 1060845/1060845/1060845 | 1 | 2.624/2.580/2.621 | 0.9831 | 0.9988 | 12/12 | PASS |
| `tip4p20` | 6/6/6 | 1 | 1686250/1686250/1686250 | 1 | 2.853/2.983/2.908 | 1.0456 | 1.0194 | 12/12 | PASS |
| `tip4p21` | 0/0/0 | 1 | 2000000/2000000/2000000 | 1 | 3.258/3.162/3.254 | 0.9705 | 0.9989 | 12/12 | PASS |
| `lj75` | 4/4/4 | 1 | 4000000/4000000/4000000 | 1 | 12.413/8.216/12.614 | 0.6619 | 1.0163 | 12/12 | PASS |
| `m75r14` | 11/11/11 | 1 | 1543818/1543818/1543818 | 1 | 21.644/12.577/20.822 | 0.5811 | 0.9620 | 12/12 | PASS |
| `lj98` | 9/9/9 | 1 | 2058098/2058098/2058098 | 1 | 22.040/13.375/22.510 | 0.6068 | 1.0213 | 12/12 | PASS |
| `lj104` | 6/6/6 | 1 | 3289394/3289394/3289394 | 1 | 29.380/17.494/29.644 | 0.5954 | 1.0090 | 12/12 | PASS |
| `m80r14` | 7/7/7 | 1 | 3271302/3271302/3271302 | 1 | 22.366/13.180/22.290 | 0.5893 | 0.9966 | 12/12 | PASS |
| `lj103` | 5/5/5 | 1 | 4000000/4000000/4000000 | 1 | 26.367/15.502/26.924 | 0.5879 | 1.0211 | 12/12 | PASS |
| `blj130n98` | 0/0/0 | 1 | 4000000/4000000/4000000 | 1 | 16.764/14.844/16.851 | 0.8855 | 1.0052 | 12/12 | PASS |

### Bit-Identity Audit (`SAFE4` vs `BASE`)

All `(system, seed)` pairs are bit-identical (`best_bits` and `charged` match 12/12 on every cell).

