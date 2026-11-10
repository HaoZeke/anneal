# Lennard-Jones landscape figure inputs

These dump selections come from `anneal` commit
`76952552116a7bd76b4e4b153073759d59d16377`. Every source run used the
release example binary with SHA-256
`40bbebf5102fe9118b4328024441db078f6ea8772eae95ab90b195ae457993d2`.
The per-arm logs and provenance files record the system, budget, seed, and
search configuration.

`selection.csv` records the decompressed full-dump SHA-256, its row count,
the selected-file SHA-256, and the selection parameters. The landscape
script runs separately for LJ38, LJ75, and LJ98, so the NumPy random
generator is reset to seed 7 for each figure. Each arm contributes at most
800 rows, and its deepest row is retained even when random capping omits it.

Regenerate the selected inputs from checksum-matched full dumps with:

```sh
python3 rewrite_2026/scripts/materialize_landscape_inputs.py \
  --input-dir PATH_TO_DECOMPRESSED_DUMPS \
  --output-dir rewrite_2026/data/landscapes_7695255 \
  --cap 800 \
  --seed 7
```

The matching lossless archives are named `<arm>.min.zst`; verify their
decompressed content against the `source_sha256` field before selection.
