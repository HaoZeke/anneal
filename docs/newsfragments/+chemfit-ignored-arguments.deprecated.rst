The ChemFit bridges still run these calls anneal 0.10.0 took, and give a
``FutureWarning`` that says what to pass instead; each will raise in a future
release. Preset keywords under the portfolio are ignored: ``preset_kwargs`` in
``fit_anneal`` and ``run_fitter``, the preset keywords of ``fit_chemfit``, and
``preset`` in ``run_benchmark`` and ``run_fitter``. So is a ``fit_chemfit``
preset keyword of another method, so one set of keywords still sweeps
``boltzmann``, ``fast`` and ``gsa``, and a ``run_fitter`` preset of another
kind than its classical ``method`` (``method="sa"`` runs the preset).
``run_benchmark`` runs a preset of another method, or one given with
``method="sa"``, as that preset's own method. A numeric string where a
parameter value or a bound goes, such as a bounds pair PyYAML reads from
``[1e-3, 1e1]``, is read as a number, with one warning per call, and a string
parameter comes back as the float it reads as. An ``x0`` dict leaf with its
parameter's size but another shape is read in C order, and one ``x0`` value
for the fitter's only parameter fills it where ``fitter.bounds`` gives both of
its sides and ``low`` and ``high`` are omitted. ``fit_chemfit`` reads a
fitter's ``initial_parameters`` given as a list or tuple of ``(key, value)``
pairs, which ChemFit 3.1 keeps as given, as the dict they make.
``run_benchmark`` drives a fitter that has ``evaluate`` or ``ask`` but no
``step`` or ``tell`` without step notices.
