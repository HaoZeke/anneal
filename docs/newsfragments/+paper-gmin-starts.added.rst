``paper_controls`` starts each hop arm from a GMIN ``coords`` file when
``PAPER_STARTS`` names a directory of them. Seed ``seed_start + k`` takes
``coords.<k + 1>`` centred at the origin, and the log records the file name
and its SHA-256. The walk still draws from the seed. The 200 LJ75 starts
are ``lj75_clusters.tgz`` from
https://theory.cm.utexas.edu/benchmarks/globalopt/lj75/lj75_clusters.tgz,
which is not vendored.
