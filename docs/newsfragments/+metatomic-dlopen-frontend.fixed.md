The Python extension exports `evaluate_metatomic_dlopen` in every build.
`has_metatomic_dlopen` is true whether or not torch is linked, and the
engine library is still opened at runtime.
Installing the package includes the `librgpot` SONAME as a regular file,
so the extension can load the library from a wheel.
