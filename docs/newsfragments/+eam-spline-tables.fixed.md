Two `EAMPot` objects evaluated at the same time on different threads no longer race on one process-wide table of element splines.
Each potential now keeps the tables of the elements it has seen.
