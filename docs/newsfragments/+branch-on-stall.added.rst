``Config::branch_on_stall`` spends part of a stalled stretch on independent descents from fresh random clusters.
A branch starts once the walk has gone longer without a lower minimum than it took to reach the one it holds, longer than twice its longest stretch between two lower minima, and longer than ``restart_patience`` calls.
It stops when it has gone as long without a lower quench as it took to reach its lowest, or when its lowest is a minimum already held.
The walk never sees a branch and keeps its own trajectory, later by the calls the branches spent.
A branch that reaches the ledger's stop target ends the run at the call that found it.
The flag is off by default and is left out of the resolved configuration while off.
