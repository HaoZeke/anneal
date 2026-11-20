``Config::recommended_molecular`` now defaults ``local_step`` to
``LocalStep::Lbfgs`` instead of ``LocalStep::Auto`` so fixed-evaluation
campaigns do not silently add coordinate-transform overhead. Automatic
and unconditional TRIC remain opt-in via ``LocalStep::Auto`` and
``LocalStep::Tric`` (or ``local_step="auto"`` / ``"tric"`` in Python).
