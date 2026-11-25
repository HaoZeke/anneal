# Benchmark row provenance

Every benchmark row written by a current paper or workflow harness carries
these provenance columns:

- `code_revision` (or the harness-specific revision columns) identifies the
  code that produced the row.
- `gmin_build_key` on GMIN-bearing tables records the 64-hex SHA-256 build
  key from the built `gmin-source.txt` (empty on non-GMIN rows).  The build
  note records the inputs feeding that key plus the built binary digest:
  `cap=`, `patch=` digests, compiler binaries (`fc=`, `cc=`, `cxx=`,
  `gfortran=`), compiler and CMake options (`cmake=`, `fflags=`, `fcflags=`,
  `cflags=`, `cxxflags=`, `cppflags=`, `ldflags=`), `blas=` with
  `blas_sha256=` (the OpenBLAS library the built binary loads, found before
  the key is computed), and `binary_sha256=`.  Saved GMIN rows are reused only when both
  `code_revision` and `gmin_build_key` match the current build; any mismatch
  is rejected naming the mismatched field.
- `host` is written through `host_label` in `scripts/machine_provenance.py`.
  The UTS name `bench`, which every bench run sets and which is not a
  physical-machine identity, is recorded as is.  Any other hostname is
  recorded as `host-` and the first sixteen hexadecimal digits of the
  SHA-256 of the public salt `anneal-bench-host-label-v1\0` followed by
  the name.  No script records a raw hostname, and a test fails when one
  reads it another way.
- A kernel boot id is written through `boot_label` in the same file, as
  `boot-` and the first sixteen hexadecimal digits of the SHA-256 of the
  public salt `anneal-bench-boot-label-v1\0` followed by the id, so two
  measurements still show whether they share a boot.  Shell scripts call
  `machine_provenance.py boot-label`, and the Rust examples carry the same
  rule in `examples/common/host_label.rs`.  The raw id is read in those two
  files only, the first for Python and shell and the second for Rust, and a
  test fails on any other read.  Where the kernel gives no host name or boot
  id, a Rust example writes `unknown`, never a digest of a stand-in.
- A raw host name is read by Python scripts only as the argument of
  `host_label`, by shell scripts only through `machine_provenance.py`, and by
  Rust only in `examples/common/host_label.rs`.  A job still dials the real
  address of a bank or catalog server.  Anything it writes down shows
  `endpoint_label` instead, which keeps the port and replaces the host by its
  label.  The test reads `.py`, `.sh`, `.sbatch` and `.rs` files, in
  `experiments/` too, and fails when a value derived from a label is used as
  a network address.
- `machine_id` is lowercase hexadecimal SHA-256 of the public salt
  `anneal-bench-machine-id-v1\0` followed by the machine's identifier.
  On Linux the identifier is the stripped contents of `/etc/machine-id`;
  the stripped DMI `product_uuid` is used only when `/etc/machine-id` is
  absent or empty.  On macOS neither file exists, so the identifier is the
  hardware UUID (`IOPlatformUUID`) that `ioreg -rd1 -c
  IOPlatformExpertDevice` prints.  A machine that offers none of these
  raises an error; a row never records an invented identifier.
- `cpu_model` is the first available CPU model string, with whitespace
  normalized.  On Linux it is the first of the `model name`, `Model`,
  `Processor` and `Hardware` entries of `/proc/cpuinfo` that is not empty.
  On macOS, where there is no `/proc/cpuinfo`, it is the brand string of
  `sysctl -n machdep.cpu.brand_string`.  A machine with neither raises an
  error.
- A process started by a campaign may inherit both values through
  `ANNEAL_MACHINE_ID` and `ANNEAL_CPU_MODEL`.  Every row writer reads them
  through `machine_columns()`, the one path that stamps a row.  The
  inherited `ANNEAL_MACHINE_ID` must be a salted SHA-256 digest like the
  one this machine computes, and a variable that is empty counts as unset.
  Setting only one of the two, or an `ANNEAL_MACHINE_ID` that is not a
  digest, is an error.  With neither variable set the columns describe the
  machine that runs the row.

The machine fields describe the physical machine that ran the measured row.
Summaries preserve the distinct source values, separated by semicolons.
In `metal_h2h`, reused legacy `.row` files that lack provenance fields
(`anneal_head`, `code_revision`, `host`, `machine_id`, or `cpu_model`) are
marked `unknown` in summaries rather than attributing the current revision or
machine to them.
Harnesses refuse to append under legacy headers rather than rewriting or
backfilling provisional rows; panel runners also refuse in-place overwrites
of legacy tables unless directed to an alternate `--out` directory or given
the explicit `--overwrite-legacy` opt-in to regenerate every row with full
provenance.
