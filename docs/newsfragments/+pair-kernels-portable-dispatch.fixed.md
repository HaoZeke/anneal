<!-- Copyright 2023--present rgpot developers -->

The `rgpot-pair` x86 kernels now select their AVX2/FMA clones with CPUID and
XGETBV instead of compiler runtime tables. This keeps accelerated kernels
available when Rust links the static library with `-nodefaultlibs`, including
on x86_64 macOS, while retaining the scalar fallback.
