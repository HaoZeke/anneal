``Config::recommended`` no longer sets ``restart_on_stall``.
``Config::derived`` and the binary Lennard-Jones and rigid-water stock presets inherit the change.
In those presets the random stall restart was never drawn.
The climb on each 5000-call plateau resets the count of quiet hops that the stall arms wait for.
Seeded searches follow the same trajectories as before, and the resolved configuration digest changes.
A caller that turns ``escape_on_stall`` off on these presets no longer gets a random redraw every ``restart_patience`` calls.
