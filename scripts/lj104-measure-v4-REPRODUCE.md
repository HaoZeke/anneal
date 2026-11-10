The measured arm reruns seeds 2000 through 2005, one seed at a time.

The engine is commit f724a4c4 and tree 7af429c6.

The binary is the release example h2h_gmin_proof from that tree.

Every timed arm shares one private mount namespace.

It also shares one hostname namespace.
That namespace sets the nodename.

The receipt script opens that namespace with unshare.

A private file that reads bench is mounted on the hostname file.

uname -n returns bench.

The host copy of the hostname file is left untouched.

Each receipt records uname -n and the hash of that hostname file.

Package hashes cover rustc, cargo, and the standard library archive.

They also cover libgcc, libm, and libc.

The receipt records boot_id and uptime at the start and at the end.

It records the cpuinfo megahertz value and the raw cpu listing.

The binary digest is taken before the first seed.

That digest is copied into every receipt.

The command text and the program line are stored as printed.

No token is removed and no line is dropped.

taskset selects processor 0.

Rayon and OpenMP each use one thread.

The budget is 2000000 evaluations.

The traced panel is carried over unchanged from the v4.2 kit.

The reference logs are the v4.2 bytes.

The regression logs and the prefix trace are those same bytes.

The end geometry is the lowest row of the minima file.

That row is written as one atom per line.
The corrected v4.6 row scripts/lj104_v4_prediction.txt reads 9c18b867.
`python3 -B scripts/lj104_sha_map.py --check <kit> <repo>` reads it.
