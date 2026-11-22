# Builder claim checks

The claim checks use the preregistered hit tolerance and outside-cost
analysis.

- TIP4P19 reproduced 3/6 B hits.
- TIP4P19 seed 2001 reproduced a hit at 836428 charged calls.
- TIP4P15 reproduced 4/6 B hits.
- TIP4P17 seeds 9200--9211 reproduced 4/12 B hits.
- TIP4P21 did not reproduce the claimed paired outside ratio. The observed
  B/A median ratio was 0.998 with bootstrap interval [0.965, 1.044], versus
  the claimed 0.979 [0.932, 0.993].

The overall gate failed only the outside-cost clause. TIP4P18, LJ150, and
M80 radius 14 had B/A interval upper bounds of 1.150, 1.135, and 1.146.
Every hit, memory, and rigid-diagnostic clause passed.

Across B's rigid-water panel rows, the rigid diagnostics reported 51 shell
events and 267400 recorded events.
