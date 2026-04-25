-------------------------- MODULE Workflow --------------------------
(***************************************************************************)
(* TLA+ specification of the SA workflow class from the IISE manuscript.   *)
(* State variables track the current point, best-seen point, temperature,  *)
(* epoch counter, and trajectory. Transitions are non-deterministic at the *)
(* acceptance rule (TLA+ has no native probability); any probabilistic     *)
(* implementation refines this specification.                               *)
(***************************************************************************)
EXTENDS Naturals, Sequences, FiniteSets, TLC

CONSTANTS
    S,        \* Finite state space
    T_0,      \* Initial temperature
    Cool(_),  \* Cooling schedule: epoch -> temp
    Neigh(_), \* Neighborhood: state -> set of states
    Cost(_),  \* Objective: state -> integer (TLA+ has no reals)
    MaxSteps  \* Liveness bound on the epoch counter

VARIABLES cur, best, temp, epoch, history

vars == <<cur, best, temp, epoch, history>>

----------------------------------------------------------------------------
(*                              Init                                        *)
Init ==
    /\ cur \in S
    /\ best = cur
    /\ temp = T_0
    /\ epoch = 0
    /\ history = <<cur>>

----------------------------------------------------------------------------
(*                              Step relations                              *)
\* Propose action: pick a neighbor and either improve or accept-uphill.
\* AcceptUphill is left unconstrained: TLA+ has no probability.
Propose ==
    \E c \in Neigh(cur):
        /\ cur' = c
        /\ best' = IF Cost(c) < Cost(best) THEN c ELSE best
        /\ history' = Append(history, c)
        /\ UNCHANGED <<temp, epoch>>

\* CoolStep action: advance the epoch counter and update temperature.
\* The cooling schedule operator Cool(_) is required by L4 to be non-increasing.
\* Action name is CoolStep to avoid colliding with the Cool constant operator.
CoolStep ==
    /\ epoch < MaxSteps
    /\ temp' = Cool(epoch + 1)
    /\ epoch' = epoch + 1
    /\ UNCHANGED <<cur, best, history>>

Step == Propose \/ CoolStep

Spec == Init /\ [][Step]_vars /\ WF_vars(Step)

----------------------------------------------------------------------------
(*                              Invariants                                  *)
TypeOK ==
    /\ cur \in S
    /\ best \in S
    /\ temp \in Nat
    /\ epoch \in Nat
    /\ \A i \in DOMAIN history: history[i] \in S

BestMonotone ==
    Cost(best) <= Cost(cur)

\* SymmetricNeighbors lifted to a property of the relation: every step
\* moves between mutual neighbors per L1.
SymmetricNeighbors ==
    \A i \in DOMAIN history:
        \A j \in DOMAIN history:
            (j = i + 1 /\ history[i] # history[j]) =>
                history[j] \in Neigh(history[i])
                /\ history[i] \in Neigh(history[j])

MonotoneCooling ==
    [][temp' <= temp]_vars

----------------------------------------------------------------------------
(*                              Liveness                                    *)
EventualTermination == <>(epoch = MaxSteps)

EventualCooling == \A k \in 0..MaxSteps: <>(temp <= Cool(k))

----------------------------------------------------------------------------
(*                  Concrete operator definitions                           *)
(* Used by small.cfg / apalache.cfg as substitutions for the abstract       *)
(* CONSTANTS Cool(_), Neigh(_), Cost(_).                                    *)
\* StepCool: a simple non-increasing schedule. Drops by 1 each epoch,
\* clamped at 0. Witnesses L4 (non-increasing in epoch).
StepCool(e) == IF T_0 - e > 0 THEN T_0 - e ELSE 0

\* TrivialNeigh: every state is its own neighbor plus all other states.
\* Witnesses L1 (symmetric) trivially.
TrivialNeigh(s) == S

\* TrivialCost: zero cost everywhere. Witnesses BestMonotone trivially
\* (every transition keeps Cost(best) = Cost(cur) = 0).
TrivialCost(s) == 0

============================================================================
