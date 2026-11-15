`UmaPot` and `libuma_engine` no longer unpack the `.pt2` into a new directory under the system temporary directory, which torch left behind whenever a process died and which could fill or be missing on a fresh machine.
The package is unpacked once into `<root>/aoti/<sha256 of the .pt2>.model/` and reused by later runs, and concurrent processes share one unpack.
The root is `UmaConfig::cache_dir` or the new `UmaParams.cacheDir`, then `RGPOT_CACHE_DIR`, then `$XDG_CACHE_HOME/rgpot`, then `.rgpot-cache` in the working directory.
