The superbasin tracker now records a proposal that reaches a known basin as a transition.
It looked the proposal up by its coordinates in an index keyed by descriptors, so no lookup matched and every proposal counted as leaving the graph.
``superbasin_report`` therefore showed no transitions, and ``superbasin_escape`` had no graph to solve.
Runs with ``superbasin_escape`` off follow the same trajectories as before.
With it on, a chain that reaches ``superbasin_period`` hops can now take a jump.
