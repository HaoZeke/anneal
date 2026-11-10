The measured arm reruns seeds 2000 through 2005, one seed at a time.

The engine is commit d9eec16a32df and tree 801ff2bc6cb1.
Engine commit d9eec16a32dfe85e30c06837dc4e2a2eebc8fdd1.
Engine tree 801ff2bc6cb1ecce3b8321a2c99a2ec24eec1c5a.
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
