``run_capped.py`` runs anneal and CREST on (H2O)16 seeds 2000-2011 from
the same start, on GFN2 plus the O-H wall, with a cap of 280305 calls
and a stop at the first call within 0.001 eV of wdbase-1. Each row
records hits, calls, time inside and outside the potential, the anneal
revision, the host and the binary hashes. The Snakemake rule writes a
stamp under ``/work``.
