---- MODULE MonotoneCooling ----
EXTENDS Workflow

\* Re-export of the L4 (temperature-monotonicity) invariant. The cooling
\* schedule never raises the temperature; checked as an action property
\* on every reachable transition.
MonotoneCoolingInv == MonotoneCooling

====
