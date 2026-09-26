-------------------------- MODULE WorkflowApalache --------------------------
(***************************************************************************)
(* Apalache-friendly instance of the Workflow specification.               *)
(* Apalache's ConfigurationPass cannot do TLC-style operator substitutions *)
(* (Cool <- StepCool); it requires the concrete operators to be in scope.  *)
(* This module pins S, T_0, MaxSteps to small literals and supplies        *)
(* StepCool/TrivialNeigh/TrivialCost directly, then INSTANCE-imports the   *)
(* Workflow Spec, invariants, and BoundedHistory constraint.               *)
(***************************************************************************)
EXTENDS Naturals, Sequences, FiniteSets

CONSTANTS
    \* @type: Set(Str);
    S,
    \* @type: Int;
    T_0,
    \* @type: Int;
    MaxSteps

\* @type: Int => Int;
StepCool(e) == IF T_0 - e > 0 THEN T_0 - e ELSE 0

\* @type: Str => Set(Str);
TrivialNeigh(s) == S

\* @type: Str => Int;
TrivialCost(s) == 0

VARIABLES
    \* @type: Str;
    cur,
    \* @type: Str;
    best,
    \* @type: Int;
    temp,
    \* @type: Int;
    epoch,
    \* @type: Seq(Str);
    history

W == INSTANCE Workflow WITH Cool <- StepCool,
                            Neigh <- TrivialNeigh,
                            Cost <- TrivialCost

Spec               == W!Spec
Init               == W!Init
Next               == W!Step
TypeOK             == W!TypeOK
BestMonotone       == W!BestMonotone
SymmetricNeighbors == W!SymmetricNeighbors
MonotoneCooling    == W!MonotoneCooling
EventualCooling    == W!EventualCooling
EventualTermination == W!EventualTermination
BoundedHistory     == W!BoundedHistory

============================================================================
