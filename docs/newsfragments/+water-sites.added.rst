``cluster_search`` takes ``water_sites``, the body-frame O, H, H positions of
a rigid water, and then searches ``n`` molecules as centres followed by
exponential-map rotations, on ``Config.communicating_tip4p``,
``Config.recommended_tip4p`` or ``Config.for_tip4p``. The gradient-sign
probe runs at the start on a rigid state. ``Config`` gains the two TIP4P
constructors and the ``rigid_body``, ``screen_margin`` and
``calibrated_screen`` getters.
