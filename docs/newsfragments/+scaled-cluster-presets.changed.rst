``cluster_search(..., communicating=True)`` and
``cluster_search(..., derived=True)`` now honour ``length_scale`` and
``energy_scale``. Both presets used to ignore those arguments and run
at scale 1. A communicating run starts from
``for_cluster_with_scales`` and keeps the orbit, recognition, and
surface schedule; the surface beta is
``energy_scale / length_scale**4``. A derived run starts from
``recommended_with_scales`` and still turns on the budget window and
the cost-asymmetric Bayes screen.
