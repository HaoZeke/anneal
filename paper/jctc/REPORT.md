# Report

The library head `90c898dd2971c8753268b153d96b9b54429b081c` on `lib/recommended-no-size-switches` removes both atom-count branches from `Config::recommended` and `Config::communicating`.
Main `46c8c867d23d6dba75a878a7614144687fb31542` still has `if n_points == 38`.
The 38-atom numbers in the abstract, the results and the conclusions are the six fresh runs of that library head.
They are not a campaign row.

## LJ38 six-run

Sorted penalised costs: 10249, 55324, 65670, 69471, 91412, 208674.
Median `(65670+69471)/2 = 67570.5`.
Interquartile range `91412-55324 = 36088`.
Hits 6 of 6.
Hardware header on every log, with no hostname: `uname -m` x86_64, `nproc` 8, CPU model Intel Xeon Processor.

## Cited commits

The article cites `90c898dd2971c8753268b153d96b9b54429b081c` and names it as the head of `lib/recommended-no-size-switches`.
`git merge-base --is-ancestor` of that commit against main exits 1.
The same command against the library head exits 0.
`b7f4b6e1acbc02cf7d11fc38cbc2cda91bb78f37`, `19523548a2fb43c4c0ad7ee53db0a2fdf6fe193d` and `645bc9386838e4951dd3f90bf63f010103ff8c5d` are not cited.
Each fails `git merge-base --is-ancestor` against main and against the library head.
`645bc938` is only on `fix/paper-controls-warm-relax`.
The command output is in `RECONCILE.md`.

## Disclosure

`e7d59c15` is not an object in this clone.
The article uses this sentence and no other disclosure:

Generative AI tools assisted with drafting and editing the text; the authors reviewed and edited all content and take full responsibility for it.

That sentence names no model, vendor or company.

## Seed table

`data/lj55_lj98/seeds.csv` has 512 rows and no `develop_sha` column.
The Wales-group basin-hopping program (GMIN) rows carry the same constant.
The retired table wrote `f29dedc0e379564ac31caf8eb2c6c9cb94f515eb` on every row, including those GMIN rows.
Nothing in this archive reads that column.

## Tests

Library unit tests: 1067 passed.
Integration tests that ran before `compile_contracts`: passed.
`compile_contracts` failed with permission denied while creating `/agent/repos/tmp`, the target directory named in `.cargo/config.toml`.
That directory is not writable here.
The failure is not a change in the preset.
The integration tests that cargo did not reach, and the result line, are in `logs/rest-test-summary.txt` and `logs/full-test-summary.txt`.

## Prose

`proseguard` on the manuscript and the SI, profile `rohit-paper`, reports `B.V.` in the affiliation.
That abbreviation is the legal name and is a known false positive.
`CRediT` did not fire.

## Not pushed

No push, no tag, and no pull request.
The library head waits for Inky after this audit.
