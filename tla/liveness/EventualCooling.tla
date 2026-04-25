---- MODULE EventualCooling ----
EXTENDS Workflow

\* Re-export of the eventual-cooling property: under WF_vars(Step) and
\* a strictly monotone cooler, every temperature in the schedule's range
\* is eventually crossed.
EventualCoolingProp == EventualCooling

====
