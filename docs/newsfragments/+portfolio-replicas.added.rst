New ``portfolio_replicas`` runs several values-only portfolio loops from one
start point as one batched search under one total budget. The start is
evaluated once; at each step every replica still running asks for one point,
and those no replica has evaluated before go to the objective's batch
evaluation together, the others answered from their evaluations. With a shared
archive the replicas keep one incumbent, a point another replica has
evaluated is answered from the archive without a call or a charge, and
replicas that ask for one point in the same batch share its row, paid for by
the one with the most of its share left. Each replica's finite-difference
descent differences its gradients from a different coordinate, so replicas
that descend together from one incumbent ask for different stencil points
and a gradient in ``n`` dimensions takes about ``n / R`` batches. A private
archive per replica runs them as independent runs batched together, each
paying for its own rows while the objective is called once at a point. One
replica repeats the solo run from the start exactly.
