``examples/sc_h2h.rs`` runs arm 3 on a Sutton-Chen cluster from a start file.
The runner lets the head-to-head with Wales's global minimisation program (GMIN) start both codes at the same random point.
Its ``iso`` arm is one chain.
In the ``ens`` arm, chains split the budget and share one exact minimum history.
That ensemble has no shared bias, no gossip and no restart.
The chains take turns at each checkpoint, so a run uses one thread and repeats exactly.
A hit stops every chain once a validated quench reaches the reference energy plus 0.001.
