# Morse data-file rule

One rule generates every data file. The non-comment keyword lines from
`examples/h2h_gmin_morse.data` are copied in order and unchanged. The
generator then writes:

```text
MORSE <rho>
RADIUS <radius>
```

The driver appends one `RANSEED` line for a run. The seed is therefore
not part of the recorded data file or its digest.

At GMIN revision
`9cbec1a446352bc9ef064bd2b1a5767194655919`, `source/io1.f90` first
assigns an unset radius as

```text
2 + (3 * N / 17.77153175) ** (1/3)
```

Its `PMORSE` case leaves that value unchanged. In particular, Morse
does not apply the `2 ** (1/6)` factor used by the default-potential
branch. The generator writes the native Morse default explicitly with
ten digits after the decimal point.

`scripts/morse_g1_panel/datafile.py` generates all files from the atom
count and rho. No value is selected or edited for an individual
cluster.
