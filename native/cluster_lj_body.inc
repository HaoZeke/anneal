/* Reduced 12-6 walk. Included like cambridge_morse_body.inc. No rho. */
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
      double inv2;
      double inv6;
      double inv12;
      double coef;
      if (r2 <= 0.0) {
        continue;
      }
      inv2 = 1.0 / r2;
      inv6 = inv2 * inv2 * inv2;
      inv12 = inv6 * inv6;
      e += 4.0 * (inv12 - inv6);
      coef = -24.0 * inv2 * (2.0 * inv12 - inv6);
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
