``cluster_hopping::Config::for_tip4p`` and the presets built on it add a third move arm, ``rcycle``.
It reverses a directed hydrogen-bond cycle.
It borrows the loop move for proton arrangements in ice.
It has no proposal ratio, so it is a minimiser's move.
``MoveLibrary::RigidBody`` turns it on and off with ``reverse_cycles``.
