``run_qmc`` laid out the design points of all its starts before its first
call, however few of them ``max_evals`` left a call, and its check of the
room for its histories did not count them. In two dimensions they took
some 200 bytes a start at peak, so under an address-space limit a run of
10**6 starts could abort the interpreter before any call, and with
``n_starts=10**12`` and ``max_evals=5`` the run set out to lay out 10**12
points before its five calls. Each start's point is now made as the start
begins, so the starts take no room before they run and those past the
last that ``max_evals`` leaves a call are never laid out. Each point is the
one the design laid out at once gave its start, bit for bit, so histories
are unchanged.
