# Review notes

The prose changes keep the same facts and the same numbers.
Quoted lines below are the exact sentences that were flagged.

## Notes and reproduce

Both files used this sentence.

```
python3 -B scripts/lj104_sha_map.py --check reads that row.
```

The bare flag was read as an em dash.
The command also omitted the kit and the repo, so it exited 2.
The new sentence quotes the full command.

```
`python3 -B scripts/lj104_sha_map.py --check <kit> <repo>` reads it.
```

The notes also gained one sentence.
The arm opened at 14:41:04 and closed at 14:57:28.
The receipt tree is 7af429c6270ec5c00d3b6536fbaefeb03563e1ac.
The receipt head is f724a4c413f0460d1b6551f883da9ae3664e6330.

## Commit 560e1989

These five lines stay in that commit.
History is not rewritten.
The new message uses the plain forms.

```
--check reads logs/sha-map.tsv and does not rewrite it.
```

The bare flag was read as an em dash.
Plain form: the check mode reads the map and does not rewrite it.

```
It compares every row with the file, the SUMS line, and the tip blob.
```

The short name of the sums file was read as an acronym.
The plain sentence compares every row with the file and the sums line.
It also compares the tip blob.

```
--refresh is the only mode that rewrites the map.
```

The bare flag was read as an em dash.
Plain form: the refresh mode is the only mode that rewrites the map.

```
The script gains the copyright and MIT header.
```

The licence abbreviation was read as an acronym.
Plain form: the script gains the copyright and licence header.

```
The notes and the reproduce file name the corrected row and --check.
```

The bare flag was read as an em dash.
The notes name the corrected row and the check mode.
The reproduce file names them too.

## Copyright

scripts/h2h_receipt.sh gained the copyright line.
examples/h2h_gmin_proof.rs gained the copyright line.
tests/stall_growth_hold.rs gained the copyright line.
The kit copy of scripts/rmsd.py gained the same line.
That file is not in git.
