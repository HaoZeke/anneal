# GFN2 SCC on a broken monomer

Seeds 17001 and 17003–17005 of (NH3)4, and an (H2O)8 seed, spent their
budget inside xtb. The log line is `scc_rejected` and the closest pair
is usually a normal bond (N–H near 1.01 Å, O–H near 0.96 Å), repeated
about eighteen times. The geometry behind the first ammonia abort is
not a compressed contact and not an exploded start: one nitrogen has
no hydrogen inside 1.3 Å, two others keep a single hydrogen, and the
cluster still fits in 11 Å. xtb 6.7.1 then stops with

```
scf: Self consistent charge iterator did not converge
```

after the iteration budget. The same geometry at `--etemp 1000` or
`--etemp 3000` converges in a few milliseconds. The gap on that
structure is 0.29 eV, which is why 300 K does not.

Two changes, neither of them a budget cap.

The search refuses a declared group in which any atom has lost every
partner inside the bond tolerance (1.25 times the summed covalent
radii) before the potential is called, and steps back toward the last
finite geometry. A compressed contact is still the overlap wall. Atoms
in different groups are not compared.

The search does not close the descent. A failed potential call is
already closed by the stall on a repeated geometry. This gate only
keeps the call from being made.

The wavefunction work is in rgpot `8efdd246322d1f32bc1bcbd7e5dfdd7a4237960e`
(`XTBPot::rescueUnconvergedScc`). The first single point stops at 48
iterations, several times a converged GFN2 count of about ten, so a
geometry that was going to finish is unchanged. Past that, the charge
is smeared at 1000 K, then 3000 K, then 5000 K, and a second single
point at the configured temperature (300 K, accuracy 0.01) is what
gets returned. The hot energy is not the result. A healthy tetramer
still evaluates to −17.729616186942 Eh. Anneal only names that engine
revision.
