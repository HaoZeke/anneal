``Config::capped_lindh_quench`` quenches Cartesian molecules with a Lindh-preconditioned step cut to a trust radius, in place of L-BFGS.
A cold L-BFGS step is one coordinate unit long, which climbs a bond, and the line search then spends evaluations walking it back.
The option is off by default, applies only to a Cartesian molecular library with species declared, and is ignored for rigid-body charts.
``methods::anc_quench`` exposes the solver as ``anc_minimize`` and ``anc_minimize_watched``.
``methods::surrogate_level::polish_tip4p_minima`` hops on TIP4P from a water cluster and polishes the lowest distinct cages on the caller's objective, so the ledger is charged only for the polish.
