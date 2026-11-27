The generic packing probe screens each candidate before its full
relaxation, with the walk's own energy screen. A candidate is relaxed for
Config.screen_steps first and finished only if that energy, less the
descent the walk has measured such a pass to leave undone, can still come
below the lower of the walk's best and the best candidate already held. A
finished candidate is relaxed again from its placement, so it reaches the
minimum it reached without the screen. Screening passes are charged like
every other call. The fixed probe costs quoted with the restored packings
are those of the unscreened probe. Screened, on the paper drivers at seeds
2000 to 2011, the probe's charge per walk fell from 188 to 101 calls on
Lennard-Jones 38, from 2069 to a median of 536 on 75, from 2668 to 847.5
on 98, from 1952 to 850 on 102, and from 2584 to 1052 on Morse 55 at rho
14.
