# Benchmark row provenance

Every benchmark row written by a current paper or workflow harness carries
these provenance columns:

- `code_revision` (or the harness-specific revision columns) identifies the
  code that produced the row.
- `gmin_build_key` on GMIN-bearing tables records the 64-hex SHA-256 build
  key from the built `gmin-source.txt` (empty on non-GMIN rows).  Saved GMIN
  rows are reused only when both `code_revision` and `gmin_build_key` match
  the current build; any mismatch is rejected naming the mismatched field.
- `host` remains the UTS hostname for compatibility.  Bench runs normally
  record `bench`; it is not a physical-machine identity.
- `machine_id` is lowercase hexadecimal SHA-256 of the public salt
  `anneal-bench-machine-id-v1\0` followed by `/etc/machine-id`.  The DMI
  `product_uuid` is used only when `/etc/machine-id` is unavailable.
- `cpu_model` is the first available CPU model string from `/proc/cpuinfo`,
  with whitespace normalized.

The machine fields describe the physical machine that ran the measured row.
Summaries preserve the distinct source values, separated by semicolons.
Harnesses refuse to append under legacy headers rather than rewriting or
backfilling provisional rows.
