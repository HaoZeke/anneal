The (H2O)16 CREST driver wrappers and the socket sit in a short
temporary directory. xtb 6.7.1 aborts when the ``--driver`` command is
about 75 characters. A CREST run whose driver saw no call raises instead
of appending a row.
