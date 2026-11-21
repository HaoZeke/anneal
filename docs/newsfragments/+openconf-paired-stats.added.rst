``scripts/openconf_h2h/paired_stats.py`` compares one arm of a ``rows.csv`` with a base arm on the same seeds.
It reports the exact McNemar test on the discordant seeds, the invalid candidates of each arm, the median time per call outside the potential, and the Wilcoxon signed-rank test of those times.
The Wilcoxon p is exact for up to 50 pairs, with equal pairs dropped first, and the normal approximation past that.
``scripts/openconf_h2h/rescore_diff.py`` lists the scored cells that differ between two ``rows.csv`` files of the same runs.
