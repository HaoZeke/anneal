``run`` and ``run_qmc`` reserved room in the history for every one of
``n_epochs`` epochs before the first step, even when ``max_evals`` ended the
run long before, so a huge ``n_epochs`` aborted the interpreter: with
``n_epochs=10**12`` and ``max_evals=5`` the allocation of 40 TB failed. They
reserve only the epochs the run records, and that call returns after five
evaluations.
