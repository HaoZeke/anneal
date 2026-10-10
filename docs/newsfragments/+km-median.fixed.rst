``methods::cluster_search::median_encounter`` decides the Kaplan-Meier median
in exact integer arithmetic. The survival is exactly a half once half the runs
are found before any is censored, and the floating-point product it replaces
rounded above a half at some even run counts, 24, 28, 30 and 38 among them, so
the median came back one encounter late: the sixteenth first encounter of
thirty instead of the fifteenth. A run censored at the time of an encounter
is still at risk at it, so the median no longer depends on the order the runs
are given in.
