``LocalStep::Auto`` times the opening Cartesian quench of a molecular search per potential call and one TRIC step outside the potential, and quenches with TRIC from then on only when the call costs more.
``Config::recommended_molecular`` sets it, so a cheap force field keeps L-BFGS and an electronic-structure gradient gets TRIC; its resolved digest changes.
``cluster_search(local_step=None)`` keeps the preset's quench, accepts ``"auto"``, and reports the quench in force and the two timings.
