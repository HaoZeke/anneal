# NH3 octamer Pareto rows

Twelve seeds, 2000 through 2011. Each seed has an anneal arm and a crest arm. The four recorded tables are `shards/rows-A.csv`, `shards/rows-B.csv`, `shards/rows-C.csv` and `shards/rows-D.csv`. Those files are the shard records and are not edited. `run.py` reads them and writes `rows.csv`. `stats.py` reads `rows.csv` and writes `stats.json`.

## Column mapping

`rows.csv` has one schema. The shard headers are not the same.

| merged field | shard A | shard B | shard C | shard D |
| --- | --- | --- | --- | --- |
| seed | seed | seed | seed | seed |
| arm | method | method | method | method |
| hit | yes is 1 and no is 0 | yes is 1 and no is 0 | yes is 1 and no is 0 | 0 and 1 are kept |
| calls | calls_to_first_hit, and a miss is budget plus 1 | calls_to_first_hit, and a miss is budget plus 1 | calls_to_first_hit when that cell is filled; an empty miss is budget plus 1 | calls_to_first_hit; a miss is already the charged count plus 1 |
| wall_s | wall_s | wall_s | wall_s | wall_s |
| best_energy | best_energy_Eh | best_energy_Eh | best_energy_Eh | best_energy_Eh |
| revision | the revision cell, copied | the revision cell, copied | the revision cell, copied | the revision cell, copied |
| shard | A | B | C | D |
| machine_id | unrecorded; host_id is not copied | unrecorded; host_id_sha256 is not copied | unrecorded; host_id_hash is not copied | unrecorded; host_id is not copied |

`hit` is the shard flag. A yes becomes 1 and a no becomes 0. Shard D already stores 0 and 1.

`calls` is `calls_to_first_hit` when `hit` is 1. When `hit` is 0 and `calls_to_first_hit` is empty, `calls` is the numeric budget plus 1. When `hit` is 0 and `calls_to_first_hit` is already the numeric budget plus 1, that recorded value is kept. Shard C anneal seed 2006 and crest seed 2007 leave `calls_to_first_hit` empty. Their budget cells are 332675 and 316368. Shard D anneal seeds 2009, 2010 and 2011 store 332676, and their budget cell is 332675. No shard B row has hit no, and those crest budget cells are the word uncapped.

`machine_id` is the literal unrecorded on every row. Shard A, B and C host columns are 64 hex digits. Shard D `host_id` is 16 hex digits. None of those values are written into `rows.csv`.

## Revision cells

The revision cell is copied as recorded.

| shard | arm | seeds | revision cell |
| --- | --- | --- | --- |
| A | anneal | 2000 2001 2002 | d3c3e351e2bfd8c26a279943a26e5a8230467284 |
| A | crest | 2000 2001 2002 | 6914a25 |
| B | anneal | 2003 | d3c3e351e2bfd8c26a279943a26e5a8230467284 |
| B | anneal | 2004 2005 | e8fb2c5b51f6f7e545ae63cdf5b77d3e6a32fdfa |
| B | crest | 2003 2004 2005 | crest 3.0.2 6914a25; xtb 6.7.1 edcfbbe |
| C | anneal | 2006 2007 2008 | f02ff93e5503cee295378add6d38cd186f786050 |
| C | crest | 2006 2007 2008 | 3.0.2+6914a25 |
| D | anneal | 2009 2010 2011 | d3c3e351e2bfd8c26a279943a26e5a8230467284 engine:8efdd246322d1f32bc1bcbd7e5dfdd7a4237960e xtb:6.7.1(edcfbbe) |
| D | crest | 2009 2010 2011 | crest:3.0.2(af7eb99) xtb:6.7.1(edcfbbe) |

Shard A anneal records d3c3e351e2bfd8c26a279943a26e5a8230467284. Shard B anneal does not use one string. Seed 2003 records that same string, and seeds 2004 and 2005 record e8fb2c5b51f6f7e545ae63cdf5b77d3e6a32fdfa. Shard C anneal records f02ff93e5503cee295378add6d38cd186f786050 on seeds 2006, 2007 and 2008. Shard D anneal records the d3c3e351e2bfd8c26a279943a26e5a8230467284 string with an engine token and an xtb token in the same cell.

Crest cells are not the same string across shards. Shard A crest is 6914a25. Shard B crest is crest 3.0.2 6914a25; xtb 6.7.1 edcfbbe. Shard C crest is 3.0.2+6914a25. Shard D crest is crest:3.0.2(af7eb99) xtb:6.7.1(edcfbbe).

## Summary

`stats.py` compares anneal with crest. The two-sided Fisher exact test uses the hit counts. The two-sided Mann-Whitney test uses `calls`, with a miss stored as budget plus 1. The median ratio is the crest median of `calls` divided by the anneal median of `calls`. The interval is a percentile bootstrap of 10000 resamples from seed 20261123, linear quantiles, drawing the anneal indices and then the crest indices.
