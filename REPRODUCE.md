The measured arm reruns seeds 2000 through 2005, one seed at a time.

The engine is commit f724a4c413f0 and tree 7af429c6270e.
Engine commit f724a4c413f0460d1b6551f883da9ae3664e6330.
Engine tree 7af429c6270ec5c00d3b6536fbaefeb03563e1ac.
The binary digest follows.
2b07c440a23ec5c5ba0db4df4efecd4494963f590958ca07482f8994c433659e.

The receipt script opens one private mount namespace.
It also opens one hostname namespace.
That namespace sets the nodename.
A private file that reads bench is mounted on the hostname file.
The host copy of the hostname file is left untouched.
Each receipt records uname -n and the hash of that hostname file.
The binary digest is taken before the first seed.
taskset selects processor 0.
The budget is 2000000 evaluations.
