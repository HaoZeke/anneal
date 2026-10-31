``organic_bench`` prints ``record_polish``, ``BEST_GMAX`` (the largest
gradient component at the best, uncharged), a ``POLISH`` line with the reserve,
the search's and the polish's calls and the calls left unspent, and a
``POLISH_RECORD`` line per polished record. Its ``polish`` arm runs the final
polish alone on the xyz structure. Under ``WATER_GAP_ARM``, ``water_hexamer``
prints the polish's reserve, calls and records per seed.
``experiments/hessopt_rerun.py`` runs the methanol smoke test and the
campaign's jobs and scores them by the graph rule; ``experiments/hessopt_rerun.sh``
runs the jobs.
