The measured arm reruns seeds 2000 through 2005, one seed at a time.

The timed commit is 7a05b51318dad29e03780d981dd3543aa8cd0457.
The timed tree is e2120c08e91f9698e121d0c38e79ce64de8a74e3.
The binary digest follows.
565b531d948819eac74b7a223a4fa6ae4ba7d8dcf70163d62b4b14ae8fae37a0.

The receipt script opens one private mount namespace.
It also opens one hostname namespace.
That namespace sets the nodename.
A private file that reads bench is mounted on the hostname file.
The host copy of the hostname file is left untouched.
Each receipt records uname -n and the hash of that hostname file.
The binary digest is taken before the first seed.
taskset selects processor 0.
The budget is 2000000 evaluations.
