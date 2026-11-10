# Verdict FIX-14, next to F1-F13

The comparison is a tie. Development seeds 3000-3029
give 20 of 30 and 20 of 30 against the reference 19 of
30. The verdict's range was 20 to 22 of 30 against 19.
That is not a win. The registered test is 5000-5029
and was not run.

Code tip before these notes:
e6d763ffd347b3a503222e0190d22aabeeacc42e.

## Disposition

1. Seal. SUMS, OUTER and STORE_READY are different
   seconds, ready last. The ready ref matches the
   bundle ref. receipts/verify.txt records bundle
   verify and fsck --strict. Both exited 0.
2. History. 841769c3, f0b4bbb7 and 8ee94082 were
   replayed onto -0600. Their trees are unchanged.
   REPLAY.txt is the map. Archive refs are never
   pushed. The lattice-search commit's parent is the
   replayed v1 tip. It does not fork from 87a1c409.
   ed5849f6 is not in this clone. 87a1c409 has tree
   b8c8d0bc0b0684229c1c7b5862b15e983ffc16c9.
3. Leak. logs/fail-before.txt has no machine path.
   The failure line is the relabel assertion.
4. Serial runner. bench/lj104-lattice/v3/run_serial.sh
   is committed and runs one pinned cell at a time.
   The recorded cells were one process on cpu 0.
   They were started before this commit. The script
   is that loop, with the work directory supplied
   by the caller.
5. Identity. receipts/identity.txt names the code
   tip and tree the binary was built from. The
   per-draw logs do not repeat that line.
6. Reference logs. gmin/<seed>/ holds row.txt,
   stdout.txt, stderr.txt and lowest. The hit and
   evaluation rows match the sealed GMIN_BEST table,
   outer
   0e343f9f37c25d4567a586817710aa64baa2b308b7ced17a783949d13d06d2cc.
   They were not copied from that table alone.
7. One threshold. Both sides use a re-quenched
   energy at or below -582.086641, one millionth
   below the minimum -582.086642. The reference
   re-quench is 19 of 30, the same seeds as its
   0.001 flags. No seed lies between the stops.
8. Uniform cut. The tie pool is exact float
   equality only. On these energies the pool has
   one member, so the cut does not test near-ties.
   The two development counts differ by the random
   stream. This pass does not rerun them: a search
   is still on cpu 0.
9. Charge. 48e3beeb charges the full atom pair sum
   as one evaluation and does not charge the
   constant core energy. The serial recount is
   still 20 and 20 of 30. Median charged cost of
   a lowest-index hit is 941310. The uniform
   median is 532260.5.
10. Propose arm. 1c05ca8a makes that arm
    unreachable. The test
    energy_lattice_propose_is_unreachable passes.
11. Fail before. vacant_sites_ignore_atom_labels
    fails on 87a1c409 and passes on this tip.
12. Message. eb604328 names the five changed files
    as well as the files it adds. The tree matches
    the previous object. The re-quench program
    prints the achieved norm. The hit flag is the
    energy test, not that norm.
13. Copyright. The line is on dls_published.rs,
    lj_dls.rs, gmin_run.sh, gmin_sha.sh and
    h2h_cap.f90.
14. Prose. proseguard: NOT_RUN (binary absent).
    receipts/lint.txt holds the line-width check
    and the banned-token check under those names.

## Development numbers

Fisher exact p on [[20, 10], [19, 11]] is 1 for
both cuts. The seeds pair only the first start, so
the sign test is nominal. It is p = 1 for both.
Lowest-index wall sum 2175.000 s, from 10:51:38 to
11:37:48 -0600. Uniform wall sum 1887.260 s, from
11:37:48 to 12:24:26 -0600. First load was
3.01 2.84 2.40. Later load was 1.00. Pin was cpu 0.

## Clock

Receipt times are -0600. A paired reading was
20:01:32 +0000 and 14:01:32 -0600, six hours
apart, same second. The runner TZ is
America/Chicago. No second namespace clock was
read; that shell was busy.

## Still open

temperature-1.5 and the angular arm are not a full
set of 30. Finished temperature seeds stay in the
logs. Unfinished seeds are pending. No rate for
30 is stated. Seeds 5000-5029 were not run.
