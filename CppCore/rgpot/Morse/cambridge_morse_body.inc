/*
 * Pair walk shared by the generic and AVX2 copies. Included inside a
 * function whose arguments are x, n, rho, energy and gradient. Products
 * are not contracted: each fma is the one rounding the force update does.
 */
{
  size_t k;
  size_t i;
  size_t j;
  double e = 0.0;
  for (k = 0; k < 3 * n; k++) {
    gradient[k] = 0.0;
  }
  for (i = 0; i < n; i++) {
    const size_t base_i = 3 * i;
    for (j = i + 1; j < n; j++) {
      const size_t base_j = 3 * j;
      const double d0 = x[base_i] - x[base_j];
      const double d1 = x[base_i + 1] - x[base_j + 1];
      const double d2 = x[base_i + 2] - x[base_j + 2];
      const double r2 = d0 * d0 + d1 * d1 + d2 * d2;
      double r;
      double a;
      double coef;
      if (!(r2 > 0.0)) {
        continue;
      }
      r = sqrt(r2);
      a = exp(rho * (1.0 - r));
      e += a * (a - 2.0);
      coef = 2.0 * rho * (a - a * a) / r;
      gradient[base_i] = fma(coef, d0, gradient[base_i]);
      gradient[base_i + 1] = fma(coef, d1, gradient[base_i + 1]);
      gradient[base_i + 2] = fma(coef, d2, gradient[base_i + 2]);
      gradient[base_j] = fma(-coef, d0, gradient[base_j]);
      gradient[base_j + 1] = fma(-coef, d1, gradient[base_j + 1]);
      gradient[base_j + 2] = fma(-coef, d2, gradient[base_j + 2]);
    }
  }
  *energy = e;
}
