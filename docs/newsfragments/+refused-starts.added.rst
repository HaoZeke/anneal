``RelaxStats::refused_starts`` and ``WarmLbfgs::refused_starts`` count the
relaxations that ended at their start because the first evaluation was
refused. The GFN2 conformer example answers a failed xtb call with ``+inf``
and a NaN gradient, so the guard rejects it, instead of a finite ``1e6``
with a zero gradient that a quench start read as converged. It prints the
failed and total engine calls of the search.
