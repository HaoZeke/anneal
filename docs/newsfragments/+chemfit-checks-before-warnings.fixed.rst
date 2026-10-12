The ChemFit bridges gave their ``FutureWarning`` for a call anneal 0.10.0
took, such as a preset under the portfolio or a fitter without ``step``,
before checking the rest of the call. A call refused for another argument,
such as an ``x0`` of the wrong size, empty bounds or a budget of 0, so
warned first about a fit that never ran, and under
``warnings.simplefilter("error")`` raised the warning in place of the error.
``fit_anneal``, ``fit_chemfit``, ``run_benchmark`` and ``run_fitter`` now
check every argument they refuse before giving any such warning, so a
refused call raises its error alone and a call that runs warns as before,
except where ``run_fitter`` hands a fitter to its own ``fit_anneal``. There
``run_fitter`` refuses only what it cannot pass to that method and warns of
a preset that it ignores just before the call; the budget, the seed, the
fitter's bounds and the rest are the method's to check, so a refusal inside
it, such as ``run_benchmark``'s of a budget of 0, follows the warning, and
under ``warnings.simplefilter("error")`` the warning is raised instead and
the method is not called.
