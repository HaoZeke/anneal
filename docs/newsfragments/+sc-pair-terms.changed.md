`SuttonChenPot` keeps the powers and `1/r^2` that the force of each pair takes from the density pass, and the force pass reads them back instead of raising the distance to `n` and `m` a second time.
`setPairTermLimit` bounds the store, 32 MiB by default, and the force pass computes the pairs past it again.
The pair lambdas inline into the walks and take their constants by value, and the density and weight arrays are kept with the instance.
`SuttonChenPot.cc` is built with `-ffp-contract=off`, as the other kernels are, so reading a term back and computing it again round alike.
Energies, forces, the stress and the per-atom split keep their bits wherever the old build fused no multiply into an add.
