``run`` and ``run_qmc`` keep every evaluation inside ``[low, high]`` and take
``x0``. The ``Boltzmann``, ``Fast`` and ``Gsa`` presets now pair their
Gaussian, Cauchy and Tsallis moves with mirror reflection into the box and the
box neighbourhood (``boltzmann_in_box``, ``fast_in_box``, ``gsa_in_box``), a
pairing the composition laws certify. Before, ``low`` and ``high`` only drew
the start, and a 13-atom walk in a ``[-3, 3]`` box evaluated 99.95% of its
points outside the box. ``run`` starts its walk at ``x0`` and ``run_qmc``
uses it as the first start; a starting point outside the box, of the wrong
length, or not finite raises ``ValueError``, and so do invalid preset
parameters, which used to panic.
