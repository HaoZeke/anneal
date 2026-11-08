# Dead code the long-range series removed

Each entry quotes one deletion in the series diff.

## Mesh factor table

The commit that runs a capped particle mesh Ewald request as that team
removes the cached factor table from Electro.cc.
The float kernel builds its own scale.
The stress kernel recomputes the factor from the reciprocal vector.
It does that while it accumulates the virial.
No other code read the table or `apply_cached`.

```
-    std::vector<double> factor;
-  auto apply_cached = [&](int tid) {
-          const double factor = kfac[at];
-          rec_plain += factor * (re * re + im * im);
-          const double scale = two_n * factor;
-    run_kernel(apply_cached);
```

## Serial flag on the charge matvec

The same commit removes a thread-local flag from FastD4.cc.
Factorizing the electronegativity equilibration (EEQ) matrix calls
apply_one directly.
`apply` never observed that flag.
The conjugate gradient still calls `apply` for the parallel matvec.

```
-thread_local int g_serial_apply = 0;
-    if (g_serial_apply) {
-      apply_one(q, y);
-      g_serial_apply = 1;
-      g_serial_apply = 0;
```

## Order-5 branch of the spline derivative

The same commit also removes one branch from bspline_derivative in
Electro.cc.
`weights_order5` supplies the order-5 spread, and `bspline_derivative`
does not.

```
-  if (order == 5) {
```

## Unreachable team clamps

The commit that drops mesh branches no input can reach removes two
clamps from Team.hpp.
A grid longer than 32 points already has at least four planes, so the
plane cap cannot be zero.
The mesh cap and the atom cap never exceed the request.

```
-    if (cap < 1) {
-      cap = 1;
-    }
-  if (team > requested) {
-    team = requested;
-  }
```
