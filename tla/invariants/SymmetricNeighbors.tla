---- MODULE SymmetricNeighbors ----
EXTENDS Workflow

\* Re-export of the L1 (symmetric-neighborhood) invariant lifted to the
\* trajectory: every consecutive transition between distinct states moves
\* between mutual neighbors.
SymmetricNeighborsInv == SymmetricNeighbors

====
