#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @file PairHessian.hpp
 * @brief The Hessian block of one pair term and its place in a dense Hessian.
 *
 * A pair energy @f$\phi(r)@f$ of @f$d = x_i - x_j@f$, @f$r = |d|@f$,
 * contributes the symmetric 3 x 3 block
 * @f[
 *   K = \phi''\,\hat d \hat d^T + \frac{\phi'}{r}\,(I - \hat d \hat d^T)
 * @f]
 * to the diagonal blocks @f$(i, i)@f$ and @f$(j, j)@f$ of the energy Hessian
 * and @f$-K@f$ to @f$(i, j)@f$ and @f$(j, i)@f$. Written in the squared
 * distance, @f$\phi(r) = g(s)@f$ with @f$s = r^2@f$, the same block is
 * @f[
 *   K = 2 g_s\,I + 4 g_{ss}\,d d^T ,
 * @f]
 * which is the form a dual-number evaluation of @f$g@f$ supplies.
 *
 * The six distinct entries are computed once and mirrored, so a Hessian
 * assembled from these blocks is symmetric to the last bit.
 */

#include <cstddef>

namespace rgpot::pair_hessian {

/// The upper triangle of a symmetric 3 x 3 block.
struct Block {
  double xx, xy, xz, yy, yz, zz;
};

/// @f$K@f$ from the radial derivatives @f$\phi'@f$ (@a d1) and
/// @f$\phi''@f$ (@a d2) at @f$d@f$, with @a inv_r @f$= 1 / r@f$.
inline Block radial(double d1, double d2, double dx, double dy, double dz,
                    double inv_r) noexcept {
  const double ux = dx * inv_r;
  const double uy = dy * inv_r;
  const double uz = dz * inv_r;
  const double across = d1 * inv_r;
  const double along = d2 - across;
  return {along * ux * ux + across, along * ux * uy, along * ux * uz,
          along * uy * uy + across, along * uy * uz, along * uz * uz + across};
}

/// @f$K@f$ from @f$g_s@f$ (@a gs) and @f$g_{ss}@f$ (@a gss) at @f$d@f$.
inline Block squared(double gs, double gss, double dx, double dy,
                     double dz) noexcept {
  const double iso = 2.0 * gs;
  const double outer = 4.0 * gss;
  return {outer * dx * dx + iso, outer * dx * dy, outer * dx * dz,
          outer * dy * dy + iso, outer * dy * dz, outer * dz * dz + iso};
}

/// Add @f$K (v_i - v_j)@f$ to atom @a i of @a hv and subtract it from
/// atom @a j, the pair's share of @f$H v@f$.
inline void apply(const Block &k, std::size_t i, std::size_t j, const double *v,
                  double *hv) noexcept {
  const double *vi = v + 3 * i;
  const double *vj = v + 3 * j;
  const double wx = vi[0] - vj[0];
  const double wy = vi[1] - vj[1];
  const double wz = vi[2] - vj[2];
  const double kx = k.xx * wx + k.xy * wy + k.xz * wz;
  const double ky = k.xy * wx + k.yy * wy + k.yz * wz;
  const double kz = k.xz * wx + k.yz * wy + k.zz * wz;
  double *hi = hv + 3 * i;
  double *hj = hv + 3 * j;
  hi[0] += kx;
  hi[1] += ky;
  hi[2] += kz;
  hj[0] -= kx;
  hj[1] -= ky;
  hj[2] -= kz;
}

/// Add @f$K@f$ to the blocks @f$(i, i)@f$ and @f$(j, j)@f$ of the
/// row-major @a dim x @a dim matrix @a h and @f$-K@f$ to @f$(i, j)@f$ and
/// @f$(j, i)@f$.
inline void scatter(const Block &k, std::size_t i, std::size_t j,
                    std::size_t dim, double *h) noexcept {
  const double m[3][3] = {
      {k.xx, k.xy, k.xz}, {k.xy, k.yy, k.yz}, {k.xz, k.yz, k.zz}};
  for (std::size_t a = 0; a < 3; ++a) {
    double *rowI = h + (3 * i + a) * dim;
    double *rowJ = h + (3 * j + a) * dim;
    for (std::size_t b = 0; b < 3; ++b) {
      rowI[3 * i + b] += m[a][b];
      rowJ[3 * j + b] += m[a][b];
      rowI[3 * j + b] -= m[a][b];
      rowJ[3 * i + b] -= m[a][b];
    }
  }
}

} // namespace rgpot::pair_hessian
