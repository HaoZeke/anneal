``catalog::molecular::HARTREE_EV``, ``BOHR_ANGSTROM`` and ``WATER_RECORD_GRADIENT`` changed from constants to functions: ``hartree_ev()``, ``bohr_angstrom()`` and ``water_record_gradient()``.
Callers replace the path with a call.
The factors now come from ``readcon_core::units`` (CODATA 2018), as in rgpot, instead of a typed numeric literal, and the Python scripts read the same factors from ``scripts/codata2018.py``.
The values agree with the old literals to the digits the literals carried.
