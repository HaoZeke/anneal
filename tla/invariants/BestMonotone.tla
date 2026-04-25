---- MODULE BestMonotone ----
EXTENDS Workflow

\* Re-export of the best-monotonicity invariant: Cost(best) is non-increasing
\* over the trajectory and is at most Cost(cur) at every reachable state.
BestMonotoneInv == BestMonotone

====
