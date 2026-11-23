<!-- Copyright 2023--present rgpot developers -->

The Meson configure step reads the libc++ version again on Windows toolchains built on clang (clang, clang-cl, llvm-mingw, MSYS2 CLANG64). A libc++ older than 19 once more gets the `__atomic_fetch_or` fallback check, and configure stops with a clear error when that fallback cannot link. MSVC is still skipped, and the probe passes the C++20 standard flag again.
