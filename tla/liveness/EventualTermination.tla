---- MODULE EventualTermination ----
EXTENDS Workflow

\* Re-export of the eventual-termination property: the epoch counter
\* eventually reaches MaxSteps under weak fairness of Step.
EventualTerminationProp == EventualTermination

====
