``run_fitter`` now runs the ``preset`` it is given and passes ``x0``,
``low``, ``high``, ``bound_span``, ``steps_per_epoch`` and
``preset_kwargs`` on to ``fit_anneal``. anneal 0.10.0 dropped them all, so
the same call can give a different fit: ``method="sa"`` with
``preset=Fast()`` runs Fast where 0.10.0 ran Boltzmann, a classical
``method`` runs a preset of its own kind, and ``x0``, ``low`` and ``high``
set the start and the box. A forwarded value that ``fit_anneal`` refuses
raises, and any other keyword, such as ``n_epochs`` or ``tell_every``,
raises a ``TypeError`` that names ``run_fitter``.
