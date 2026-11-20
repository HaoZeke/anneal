# b11 stalled rigid walk diagnosis

This diagnosis compares develop
`61fa67cf380e46876b569da45e7cc091647a775b` with the generalized walk at
`f911c2ecb4983f4d2de143e81210f0ea87e04613`. Runs use TIP4P17,
seeds 2000--2011, and a budget of 2,000,000 calls.

## Decision path

Develop starts the shell path after 40 restart-patience windows and the
stored-cage path after four shell horizons. With the default restart
patience, these horizons are 200,000 and 800,000 calls.

The generalized revision replaces both horizons with restart patience
times six generalized coordinates times the body count. At 17 bodies
this is 8,670,000 calls, beyond the run budget.

The call trace confirms the consequence on every seed. Develop has one
600,629--600,793-call gap in its main-hop trace for the shell path.
Eleven seeds then finish with a 94,419--510,829-call archive tail.
The generalized revision has no large gap and only 9--59 untraced calls.
It executes 25,339--28,883 main hops, compared with
11,255--19,068 on develop.

The missing cold path leaves every charged call in the full main-chain
bookkeeping path. This accounts for the outside-time increase and
removes the stored-cage escapes that produce develop hits.

Develop reaches the reference cage on seeds 2002, 2003, 2006, 2008, and
2011. The generalized revision reaches it on seeds 2002, 2008, and 2011.
Seed 2006 first separates when develop enters the shell path. Seed 2003
separates earlier, at hop 501, because the same commit also replaces the
water cycle reversal with a generic contact-cycle construction. Seed
2010 first separates in that construction at hop 2,986.

## Candidate result

Revision `8e3dfaf042df4296e50091daf2808e8a87dc1df6` restores the two
established horizons while retaining the generalized contact and archive
walks. On seeds 2000--2011 it records 8 of 12 TIP4P15 hits and 7 of 12
TIP4P17 hits, but only 4 of 12 TIP4P20 hits. Develop records 7, 5, and 6
hits in those cells. Its aggregate outside times are 2.442931 and
3.520747 microseconds per call on TIP4P15 and TIP4P20, ratios of 1.043
and 0.916 against develop.

The horizon repair therefore meets the time limit but not the no-loss
condition. Restoring the older cycle implementation recovers the missing
TIP4P20 rows, but that implementation selects water by atomic numbers
and uses its fixed contact cutoff. Shipping that fallback would violate
the requirement against species and motif conditions.

The generalized walk cannot meet the stated bar without removing its
generic contact behavior. Search code is consequently unchanged.
