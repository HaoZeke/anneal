``ANNEAL_REPRODUCIBLE_ARCHIVE=1`` builds the Meson cargo static archive with
remapped source, target, and ``CARGO_HOME`` paths. The archive written after
``ld -r`` has a zero mtime, uid, and gid, including when the archiver has no
deterministic mode. The default build is unchanged.
