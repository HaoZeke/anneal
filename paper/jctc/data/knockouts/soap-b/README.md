# SOAP prior knockout, shard soap-b

Five Lennard-Jones systems, lj98, lj102, lj103, lj104 and lj150, seeds
2000-2011, budget 4000000 charged calls. Arm A is the stock driver,
`H2H_KNOCKOUT` unset. Arm B sets `H2H_KNOCKOUT=prior_soap`. Both arms
ran the same `h2h_gmin_proof` binary. The `revision` column is the
tree id `b12293b9d637dd32417be3aa482302e0f30dd594`, not a commit id.

The driver was `h2h_gmin_proof`, including the `budget lj103` entry.
A hit is a quenched energy at or below the reference plus 0.001. A
miss is recorded at the full budget. `best_bits` is the IEEE-754
binary64 encoding, as little-endian hex, of the printed `best` token.
The driver prints `best` with eight digits after the decimal point.

Each seed's start was written once by the stock binary's `coords`
command and passed to both arms as the fourth argument. On this
driver that argument receives the seed start: the run writes the
generated start there and quenches it. The file bytes after each
step-0 run matched the stock `coords` file, and the develop binary's
`coords` file matched it as well. Even seeds ran stock then the
knockout. Odd seeds ran the knockout then stock. One process at a
time, pinned to CPU 0, with one thread in each numeric library.
There was no warm-up exclusion and no retry.

Step 0, before any measured row, compared seed 2000 of each system
on the develop binary and on this tree with the switch unset. The
printed best, its bits and the charged calls matched on all five
systems. `cargo test --release --test knockouts` passed 3 tests.
`pytest` of `test_knockout_drivers.py` passed 1 test.

No run crashed.

The paired counts, the two-sided exact McNemar test and the
Newcombe method-10 interval are in `paired_stats.txt`.
