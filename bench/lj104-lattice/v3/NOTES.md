# Lattice v3.2

The development comparison is a tie with the reference.
Seeds 3000-3029 are the development set. Seeds
5000-5029 stay the registered test, outside this seal.

The charge and its fail-before test are one commit.
logs/fail-before-charge.txt names base 0a79aadb and
the panic spent 653. receipts/tests.txt records the
pass of the_atom_pair_sum_is_charged at the tip.

## Disposition

1. Seal. REPORT, SHA256SUMS, OUTER, the bundle touch
   and STORE_READY fall in later seconds. STORE_READY
   lists OUTER, the tip, the tree, BUNDLE_SHA256, the
   stat times and the full refs/heads name.
2. History. The replay script is replay.py and its
   log is replay.log. Shared seconds are real.
   Archive refs are never pushed. The bundle carries
   refs/archive for 0a79aadb, e5037f12 and the
   v3.2-draft e947d022.
3. Paths. The fail-before logs name the base SHA and
   the panic line. The compile lines use a neutral
   mount.
4. Serial runner. run_development.sh runs seeds
   3000-3029 one cell at a time on cpu 0. Time T is
   the first start, on or after the charge commit.
5. Identity. receipts/identity.txt names the tip and
   the tree that produced the binary.
6. Reference logs. gmin/seed holds row.txt,
   stdout.txt, stderr.txt and lowest. The rows match
   the sealed reference table, outer
   0e343f9f37c25d4567a586817710aa64baa2b308b7ced17a783949d13d06d2cc.
7. One threshold. Both sides use a re-quenched
   energy at or below -582.086641. The reference
   re-quench stays 19 of 30.
8. Uniform cut. The tie pool keeps exact float
   equality. These energies leave a one-member pool.
9. Charge. The development cuts below are the serial
   run of the tip that charges the atom pair sum.
10. Propose arm. The exchange propose match stays
    closed. energy_lattice_propose_is_unreachable
    passes.
11. Fail before. the_atom_pair_sum_is_charged fails
    on 0a79aadb and passes on the charge commit.
12. Message. The lattice-search commit names the
    files it adds and the five files it changes.
    The hit flag is the energy test.
13. Copyright. The developer line is on
    dls_published.rs, lj_dls.rs, gmin_run.sh,
    gmin_sha.sh and h2h_cap.f90.
14. Prose. proseguard: NOT_RUN (binary absent).
    receipts/lint.txt holds line-width and
    banned-token under those names.

## Development recount

The cut tables in data/ are rebuilt from the raw
logs by recount.py. The headline counts and medians
are the ones that script prints for this run.

## Clock

Receipt times are -0600. A paired reading was
20:01:32 +0000 and 14:01:32 -0600, six hours apart
in the same second. The runner TZ is
America/Chicago.

## Fork

This clone holds 87a1c409 in place of ed5849f6.
Both name tree
b8c8d0bc0b0684229c1c7b5862b15e983ffc16c9.
The lattice-search commit follows the replayed v1
tip.
