``length_scale`` and ``energy_scale`` now change ``ras=True``,
``species``, and ``rigid`` searches. Those runs used to keep the
unit-scale start, temperature, and merge radius when a scale was
passed. A species or rigid geometry is drawn at ``length_scale``, and
its energies follow ``energy_scale``. An archive search uses
``recommended_with_scales``.
