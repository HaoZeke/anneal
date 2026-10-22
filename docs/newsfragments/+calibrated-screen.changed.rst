The cluster hop's energy screen credits each trial with the descent its
screening pass leaves undone, the upper decile of screened minus full over the
last 256 full relaxations, and relaxes every eighth refusal in full so the
estimate also samples the trials it refuses (``Config::calibrated_screen``).
A full relaxation that ends on its step cap below the incumbent is continued
until it stops descending (``Config::finish_improving``). Both are on in
every preset. ``Config.calibrated_screen`` and ``Config.finish_improving``
read the two flags from Python. On 256 TIP4P10 archive starts at
development seeds and 1e6 force calls, PAR-2 went from 78681 to 15870 on
``Config::for_tip4p`` and from 92738 to 16369 on
``Config::communicating_tip4p``.
