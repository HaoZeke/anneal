# Pinned GMIN binary hashes

The 2026-11-19 tarball `ec5730d5e27fd5c1ac142360278abc62bd3c7dbea8686c9a2dc9de817b58b776` has SVNREV `9cbec1a446352bc9ef064bd2b1a5767194655919`. The panel recipe at `eeec8e2c216ae1506768bca8e29846c1c039ca3b` (`scripts/gmin_panel/build_gmin.sh`, gfortran 13.3.0, the three panel patches, serial CMake, `make -j`) builds that tree. The tarball does not determine the GMIN sha256. `check_gmin_build.sh` repeats the builds below.

## Three builds

| Build | Directory | sha256 |
| --- | --- | --- |
| first | A | `542009f8ceb80e97a03ac633ea4843f4b99a8111e2bc140a07bd80dde738f98e` |
| second | B, a different absolute path of the same length | `c48d9e21e49792936c6e9d7e57f62288c2743dbc9ce9b3c7928c776a9a2833b6` |
| third | A again, 120 seconds after the first build finished | `e28f9aa6e20f033ff2935779d3d0058a62ebb0005d9495d3244ad9aff8ed4575` |

Shard a was `24939393456c1dbe9c20d515b981f33c86d43b79b01de369db231edc8dbe5e68`, shard b `d3542eca295d7e1343860cd20a780943fcd269bb5293344f5371cf81d68fa629`, shard c `b2d448da772879c57dfcde5002a4cf991e4c4903e684e562daa65e3bdbf6216e`. The brief expected the prefix `dcd3ce8c`. That prefix is the hash of one earlier build.

## Sections

`objcopy -O binary -j` on the three builds:

| Section | Verdict | sha256 |
| --- | --- | --- |
| `.text` | identical | `f1532af4c39028d3f8705eb75a9dbdee1d06d63363ca8da660171d0661452e39` |
| `.rodata` | differs | A `f426f702b58c651a9caf9e4a817dbaafc7a7f052cb8badac6ebfa795456b6dbe`, B `70aa19f6cfccb2e264b6c2e674a3f0b1bc4b5fc7ac75f98605ccf44ad8e840db`, A again `be630ed5a358f6f4f4eea277b62acdf73a0b3868166ad0a76f8decdb8ffa9924` |
| `.data` | identical | `fa7d3cc6e95345051ec4ee8bed4ed0beb72e4d51ba3552ab7c9931766e798fc1` |
| `.data.rel.ro` | identical | `f54f40044bd72fe4bbb44032e3446f5cf5b518b64d07640abd47c0935d5bc756` |
| `.eh_frame` | identical | `fdf959a8bbee4f75e7952b51d328d773e87667e76d865d16bde97ae75c14d99a` |
| `.eh_frame_hdr` | identical | `4cb804b1fbd5e59a110c6b95b3b48f54333eec8277ce0b94a8d397eef6e50fca` |

`.text` is byte-identical, so the three binaries are the same code. No function and no object in `.text` differs. There is no debug section. `.comment` is `GCC: (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0` in every binary (`readelf -p .comment`). The strings diff has no assembler temporary name.

Same directory, later rebuild: `.rodata` differs in four bytes, the minute and the second of `Compilation time:`. The three lines were `Thu Nov 19 05:44:06 PM UTC 2026`, `Thu Nov 19 05:49:27 PM UTC 2026` and `Thu Nov 19 05:53:44 PM UTC 2026`. Different directory, same length: the same clock line moves, and gfortran's runtime locus strings move with it (`At line N of file <absolute path>`, `In file '<absolute path>'`). CMake passes those absolute paths. `-D_SVN_ROOT_` is the absolute `wales` directory. Equal length keeps every `lea disp32(%rip)` in place, so `.text` matches while `.rodata` does not.

## Cause

`wales/SCRIPTS/all/display_version.sh` is run while GMIN links and writes `display_version.f90`. The subroutine contains `` `date` ``, `$USER@$HOSTNAME`, and `uname -o` / `uname -m`. GNU `date` and gfortran 13.3.0 both ignore `SOURCE_DATE_EPOCH` (the Fortran compiler's `f951` has no such string; the C compiler does). Each build therefore stamps the wall clock, and a different absolute work directory stamps a different path into `.rodata`. The expected prefix `dcd3ce8c` is the stamp of one such build. Shard a used one work directory; shards b and c rebuilt in another directory at different times. The machine code did not change.

## Flag pair

Two further builds in one directory passed `SOURCE_DATE_EPOCH=1763510400` and `-ffile-prefix-map=$PWD=.` on the compile (`$PWD` is the build directory), then `strip -s`. gfortran 13.3.0 still embedded the absolute source path. The clock lines were still the wall clock, `Thu Nov 19 06:01:12 PM UTC 2026` and `Thu Nov 19 06:04:41 PM UTC 2026`. `.text` stayed `f1532af4c39028d3f8705eb75a9dbdee1d06d63363ca8da660171d0661452e39`.

| Binary | sha256 |
| --- | --- |
| first | `96a3feb2b1488058ef03dce8d8acdfbed9247f52b3eed6985786bad5e455d294` |
| second | `62e4f0a231e8fd20c3b19f6704a50055b16fa717328902b5078d94cac43f49bb` |
| first, `strip -s` | `41b05b98ab2448168b827711702c9d4533e22d78af846a2eec4a2809075e241c` |
| second, `strip -s` | `540c265b51e568d63021614049165f2238327d8244c72877e1184fd042698d1d` |

`strip -s` leaves the clock bytes in `.rodata`. The stripped hashes differ.

## Stable hash

Hold one absolute work directory. Set `SOURCE_DATE_EPOCH`. Put a `date` first on `PATH` that runs `/bin/date -u -d "@$SOURCE_DATE_EPOCH"`, so the generator's plain `date` and its `date +%d%m` probe both follow the epoch. Export `USER` and `HOSTNAME` as constants before that generator runs. The prefix map can stay on the compile line; this gfortran does not use it to rewrite the locus strings, and the fixed directory is what keeps those strings identical. Two builds then match before `strip`:

`04054eef6ee1c91179b7b0fbe177b2f6d9fd1af388d1d4a7557f86ce3c035d74`

and after `strip -s`:

`d20fcf52092dff669f4e05dfccad235c1e8a116b2bf0926cd61973816a3306a0`

The generated lines were `Compilation time: Wed Nov 19 00:00:00 UTC 2025` and `Compiled by repro@repro on GNU/Linux x86_64`. That clock line is shorter than the wall-clock line, so `.text` of this pair is `de034cea87cdc40797c5cde15709666d916b475279bc3f8bec42e38329a91c3b` rather than `f1532af4…`. The bytes that move are the displacements in `lea disp32(%rip)` (for example `sock_connect_c.cold`). The two frozen builds match each other, including `.text`. The unstripped hash is already stable; `strip -s` is optional once the clock, the account line, and the work directory are fixed.
