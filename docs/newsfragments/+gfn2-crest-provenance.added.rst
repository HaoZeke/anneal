``gfn2_crest.py`` appends each CREST row with the anneal revision, the
host, the date and the CREST, xtb and engine sha256, and refuses a file
with another header. ``--summary`` adds hits, the median and IQR of
calls to the hit, time per call, and Fisher's exact and Mann-Whitney p
against anneal.
