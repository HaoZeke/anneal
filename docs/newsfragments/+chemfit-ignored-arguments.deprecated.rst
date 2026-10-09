The ChemFit bridges still run the calls anneal 0.10.0 took while ignoring
part of them, and give a ``FutureWarning`` that says what to pass instead;
each will raise in a future release. Preset keywords under the portfolio
are ignored: ``preset_kwargs`` in ``fit_anneal`` and ``run_fitter``, the
preset keywords of ``fit_chemfit``, and ``preset`` in ``run_benchmark`` and
``run_fitter``. So is a ``fit_chemfit`` preset keyword of another method,
so one set of keywords still sweeps ``boltzmann``, ``fast`` and ``gsa``,
and a ``run_fitter`` preset of another kind than its classical ``method``
(``method="sa"`` runs the preset). ``run_benchmark`` runs a preset of
another method, or one given with ``method="sa"``, as that preset's own
method. An ``x0`` dict leaf with its parameter's size but another shape is
read in C order, and ``run_benchmark`` drives a fitter that has
``evaluate`` or ``ask`` but no ``step`` or ``tell`` without step notices.
