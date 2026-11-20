# Pairscale knockout shard: Length unit block

- Base commit: dd6ff9273be30ab317bfd5508596f4beb19cb3a4
- Revision recorded in rows: b12293b9d637dd32417be3aa482302e0f30dd594 (tree id)
- Driver: h2h_morse
- Arms: A is stock (H2H_PAIR_SCALE unset), B is H2H_PAIR_SCALE=1
- Systems: m55r6, m55r14, m80r6, m80r14
- Seeds: 2000-2011
- Order: ABBA by seed (even seed A then B, odd seed B then A)
- Starts: shared per system and seed via coords output, then passed as run arg 4
- Execution: one process at a time with taskset -c 0 and one thread
- Crashes: none

See session.txt for step-0 checks and binary/start hashes.
See paired-stats.txt for paired hit tables and exact tests.
