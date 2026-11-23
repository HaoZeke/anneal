The GitHub release gains ``anneal-core-linux-aarch64.tar.gz``, a native
library archive for aarch64 Linux. It is cross-compiled on the x86_64
runner with the distribution's ``gcc-aarch64-linux-gnu``, and the
release job links a consumer against it and runs that consumer under
``qemu-user`` before the archive is uploaded.
