CI and the release workflow install cbindgen with
``cargo install cbindgen --version 0.29.4 --locked`` and not with a
prebuilt-binary installer that takes the latest release. A new cbindgen
can no longer change ``anneal-core.h`` between two tags. A contract test
requires one exact version, built from its lockfile, in every workflow.
