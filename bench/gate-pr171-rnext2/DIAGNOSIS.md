# Missing baseline counters

The first gate built D and D2 from the gate checkout, not from the
instrumented A checkout. The gate checkout has the same production source
as A, but its working tree did not contain the temporary counter patch.
The B build changed into the instrumented candidate checkout first, so G
and G2 contained the patch.

The relevant D command was:

```text
CARGO_TARGET_DIR="$RUN/build/d" cargo +1.88.0 build --release --locked \
  --example h2h_seed --example h2h_gmin_proof -j 1
```

It ran with the gate checkout as its current directory. Setting
`CARGO_TARGET_DIR` changes only the output directory; it does not select a
source checkout. No feature or environment flag enables the counter.

The parser was not asking D for a candidate-only field. The temporary A
instrumentation writes a valid neutral counter:

```text
offered=0
hit=0
calls=0
```

Because that instrumentation was absent from the compiled D and D2
sources, their runs never created the counter file. The parser then
retained each row as an error. Representative harness output was:

```text
done m38r14d D 2000 exit=2 hit=0
done m38r14d G 2000 exit=0 hit=1
```

For comparison, candidate seed 2001 recorded:

```text
offered=1
hit=0
calls=14
```

The repair is a build-step correction: build both independent baseline
binaries while the instrumented A checkout is the current directory.
The temporary patch writes only the neutral counter after the run and
does not alter calls, charged calls, energies, or the committed A tree.
Production source remains the exact develop revision and tree registered
by the protocol.
