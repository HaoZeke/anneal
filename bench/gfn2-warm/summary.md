# GFN2 warm against fresh

Engine `4193327dd994b2a9a58f65df6579ded221eeffc7`. Code `caf6f326`.
Host `bench`.

Frame comparison, 300 K, engine gap policy. Max |E_warm − E_fresh| is
0 eV on Si13 (6 finite frames; frame 1 unconverged on both), 0 eV on
Si10, and 6.594e-12 eV on C10.

Search, seeds 2000–2011, budget 100000, 3000 K. Warm gap 1 eV, fresh
gap 1e9. Hit within 1e-3 eV of Si10 −460.68373809 eV and C10
−571.32273783 eV.

| system | warm hits | fresh hits | median calls warm | median calls fresh | Fisher p | Mann–Whitney p |
|---|---|---|---|---|---|---|
| Si10 | 11/12 | 11/12 | 954 | 954 | 1 | 1 |
| C10 | 11/12 | 12/12 | 1174 | 784 | 1 | 0.273 |

Si10 is the same trajectory in both modes. C10 warm loses seed 2003.
Median time inside one call is 0.06185 s warm against 0.06151 s fresh
on Si10, and 0.00500 s against 0.00603 s on C10. Median time outside
one call is 1.62e-5 s on Si10 in both modes, and 8.45e-6 s warm against
1.11e-5 s fresh on C10.

The setter is not default-on.
