/* Cleri–Rosato sum, every pair, no cutoff. Included in a function whose
 * arguments are x, n, slot, n_species, a, xi, p, q, r0, energy, gradient.
 * The first pass stores the pair derivative the second pass consumes, in
 * the same i < j order, so the density and the force match that walk.
 */
{
  size_t pairs = 0;
  if (n >= 2) {
    pairs = n * (n - 1) / 2;
  }
  if (rgpot_gupta_ensure(n, pairs) != 0) {
    *energy = INFINITY;
    for (size_t k = 0; k < 3 * n; k++) {
      gradient[k] = 0.0;
    }
    return;
  }
  double *rho = rgpot_gupta_rho();
  double *pack = rgpot_gupta_pack();
  for (size_t i = 0; i < n; i++) {
    rho[i] = 0.0;
  }
  for (size_t k = 0; k < 3 * n; k++) {
    gradient[k] = 0.0;
  }
  double total = 0.0;
  size_t m = 0;
  for (size_t i = 0; i < n; i++) {
    const size_t base_i = 3 * i;
    const size_t si = (size_t)slot[i];
    for (size_t j = i + 1; j < n; j++) {
      const size_t base_j = 3 * j;
      const double d0 = x[base_i] - x[base_j];
      const double d1 = x[base_i + 1] - x[base_j + 1];
      const double d2 = x[base_i + 2] - x[base_j + 2];
      const double r2 = d0 * d0 + d1 * d1 + d2 * d2;
      if (!(r2 > 0.0)) {
        *energy = INFINITY;
        return;
      }
      const size_t key = si * n_species + (size_t)slot[j];
      const double r = sqrt(r2);
      const double u = r / r0[key] - 1.0;
      const double phi = 2.0 * a[key] * exp(-p[key] * u);
      const double psi = xi[key] * xi[key] * exp(-2.0 * q[key] * u);
      total += phi;
      rho[i] += psi;
      rho[j] += psi;
      pack[6 * m] = d0;
      pack[6 * m + 1] = d1;
      pack[6 * m + 2] = d2;
      pack[6 * m + 3] = r;
      pack[6 * m + 4] = -(p[key] / r0[key]) * phi;
      pack[6 * m + 5] = -(2.0 * q[key] / r0[key]) * psi;
      m++;
    }
  }
  for (size_t i = 0; i < n; i++) {
    total -= sqrt(rho[i]);
  }
  m = 0;
  for (size_t i = 0; i < n; i++) {
    const double inv_i = 1.0 / sqrt(rho[i]);
    for (size_t j = i + 1; j < n; j++) {
      const double d0 = pack[6 * m];
      const double d1 = pack[6 * m + 1];
      const double d2 = pack[6 * m + 2];
      const double r = pack[6 * m + 3];
      const double dphi = pack[6 * m + 4];
      const double dpsi = pack[6 * m + 5];
      const double embed = -0.5 * (inv_i + 1.0 / sqrt(rho[j]));
      const double coef = (dphi + embed * dpsi) / r;
      gradient[3 * i] += coef * d0;
      gradient[3 * i + 1] += coef * d1;
      gradient[3 * i + 2] += coef * d2;
      gradient[3 * j] -= coef * d0;
      gradient[3 * j + 1] -= coef * d1;
      gradient[3 * j + 2] -= coef * d2;
      m++;
    }
  }
  *energy = total;
}
