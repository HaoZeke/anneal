The ChemFit bridges gave their ``FutureWarning`` for a call anneal 0.10.0
took, such as a preset under the portfolio or a fitter without ``step``,
before checking the rest of the call. A call refused for another argument,
such as an ``x0`` of the wrong size, empty bounds or a budget of 0, so
warned first about a fit that never ran, and under
``warnings.simplefilter("error")`` raised the warning in place of the error.
``fit_anneal``, ``fit_chemfit``, ``run_benchmark`` and ``run_fitter`` now
check every argument they refuse before giving any such warning, so a
refused call raises its error alone and a call that runs warns as before.
Where ``run_fitter`` hands a fitter to its own ``fit_anneal``, it refuses
only what it cannot pass to that method, and the budget, the seed, the
fitter's bounds and the rest are the method's to check; the warning of a
preset that ``run_fitter`` ignores waits for the calls the method makes to
``fit_anneal``, ``fit_chemfit`` and ``run_benchmark``, and comes once, from
the first call whose checks pass, before ``init``. So a refusal there, such
as ``run_benchmark``'s of a budget of 0 or a seed of -1, raises its error
alone too, and a call made after the refusal gets the warning once its
checks pass. A call the method makes on another thread with the fitter, or
with an object that holds the fitter or one of the fitter's methods as an
attribute, such as a session that wraps the fitter, gives the warning on
that thread, before its ``init``, so under ``warnings.simplefilter("error")``
it is raised there in place of the fit. A call on another thread given any
other object does not, and the warning then comes once the method returns,
after the fit, as it does when no call passes its checks; it does not come
when the method raises. In each case it points at the line that called
``run_fitter``, in a module, a script or a Jupyter cell alike.
