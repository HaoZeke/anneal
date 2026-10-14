``run_qmc`` laid out the design points of all its starts before its first
call, however few of them ``max_evals`` left a call. In two dimensions they
took some 200 bytes a start at peak, so under an address-space limit a run
of 10**6 starts could abort the interpreter before any call, and with
``n_starts=10**12`` and ``max_evals=5`` the run set out to lay out 10**12
points before its five calls. Each start's point is now made as the start
begins, so the starts take no room before they run and those past the
last that ``max_evals`` leaves a call are never laid out. A point made as
its start begins is, bit for bit, the one laying out the whole design at
once gives that start, so making the points one at a time changes no
history.
