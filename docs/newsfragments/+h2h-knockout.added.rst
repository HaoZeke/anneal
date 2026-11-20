``Knockouts`` and ``Config::apply_knockouts`` switch off named parts of
a preset: ``prior_soap`` turns off the fivefold SOAP proposal and
``move_choice`` draws the move arm uniformly. The ``h2h_gmin_proof``,
``h2h_morse`` and ``h2h_metal`` drivers read a comma list of these names
from ``H2H_KNOCKOUT``. Unset or empty, a run is bit-identical to the
preset, and an unknown name stops the driver.
