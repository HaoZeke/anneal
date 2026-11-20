The native library archives on the GitHub release carry
``lib/pkgconfig/anneal-core.pc``. Its paths are relative to the file, so
after extracting, ``PKG_CONFIG_PATH=<archive>/lib/pkgconfig`` is enough
for ``pkg-config --cflags --libs anneal-core``. The version, description
and library name come from the cargo-c metadata in ``Cargo.toml``, and
``Libs.private`` lists what rustc says the static library needs. The
release job builds a consumer through pkg-config before it uploads each
archive.
