The generic packing probe screens each candidate before its full
relaxation, with the walk's own energy screen. A candidate is relaxed for
Config.screen_steps first and finished only if that energy, less the
descent the walk has measured such a pass to leave undone, can still come
below the lower of the walk's best and the best candidate already held. A
finished candidate is relaxed again from its placement, so it reaches the
minimum it reached without the screen. Screening passes are charged like
every other call.
