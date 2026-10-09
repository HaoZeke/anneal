Python callbacks no longer fail silently. ``KeyboardInterrupt`` and
``SystemExit`` raised inside ``obj_fn`` or ``grad_fn`` end the run and are
re-raised when the driver returns, where they used to be scored as ``+inf``
while the run went on. A return value that is not a number raises
``TypeError``. An ordinary exception is still scored as the worst value, so
budget counters can stop a driver by raising, and the driver now reports how
many were scored, with the first message, as one ``RuntimeWarning``. A NaN
objective value is scored as ``+inf``, and two infeasible points count as
level, so a walk started where the objective is NaN or ``+inf`` moves until
it finds the feasible region instead of freezing there. ``cluster_search``
follows the same rules.
Gradients may be returned as any array-like (``jax.grad``, torch, lists).
