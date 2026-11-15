// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/nlist/MinimumImage.hpp"
#include "rgpot/simd/exact_fma.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

// The paired software fma is used where the unit has no fused instruction.
#if defined(__GNUC__) && !defined(__FMA__) && !defined(__ARM_FEATURE_FMA)
#define RGPOT_FOLD_X2 1
#else
#define RGPOT_FOLD_X2 0
#endif

namespace rgpot::nlist {
namespace {

using rgpot::simd::mul_add;

constexpr double kTol = 1e-12;
constexpr int kMaxIt = 10000;
// search_bounds enumerates its box only up to this many candidates.
constexpr double kExhaustiveCap = 4096.0;
// Guard on the reduced-basis box. A reduced basis of at most three rows
// keeps it far below this; reaching it means the input left double range.
constexpr double kReducedCap = 1048576.0;
// Guard on LLL steps for at most three rows.
constexpr int kLllCap = 100000;
// Largest j by k slab of cached partial images kept on the stack.
constexpr std::uint64_t kPartialCap = 256;
// Relative slack on the image length bounds. It covers the rounding in the
// dual norms, about 1e-8 for a cell that passes well_conditioned.
constexpr double kBoundSlack = 1e-6;

thread_local std::uint64_t searches_run = 0;

// Hardware instantiations are compiled inside a target("fma") function, so
// __builtin_fma becomes vfmadd there. That builtin and the target attribute
// are GCC and Clang. MSVC uses std::fma in the same instantiation. The other
// instantiation is mul_add: the same instruction when this unit was built
// with FMA, and the Dekker TwoProduct plus TwoSum otherwise. Both match
// std::fma.
template <bool Hardware>
double product_add(double a, double b, double c) {
  if constexpr (Hardware) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_fma(a, b, c);
#else
    return std::fma(a, b, c);
#endif
  } else {
    return mul_add(a, b, c);
  }
}

// a*b + c*d + e*f. The last product rounds on its own; each earlier
// product is fused into the sum. Both forms match std::fma.
template <bool Hardware>
double fdot3(double a, double b, double c, double d, double e, double f) {
  return product_add<Hardware>(a, b, product_add<Hardware>(c, d, e * f));
}

template <bool Hardware>
double frac_of(double x, double y, double z, const double inv[9], int k) {
  return fdot3<Hardware>(x, inv[k], y, inv[3 + k], z, inv[6 + k]);
}

// reduced - (i*c0 + j*c1 + k*c2), each product fused into the sum.
template <bool Hardware>
double sub_image(double reduced, double i, double c0, double j, double c1,
                 double k, double c2) {
  return product_add<Hardware>(
      -i, c0,
      product_add<Hardware>(-j, c1, product_add<Hardware>(-k, c2, reduced)));
}

double n2(const double v[3]) {
  return fdot3<false>(v[0], v[0], v[1], v[1], v[2], v[2]);
}

double vdot(const double u[3], const double v[3]) {
  return fdot3<false>(u[0], v[0], u[1], v[1], u[2], v[2]);
}

void lin3(const double c[3], const double a[3], const double b[3],
          const double d[3], double out[3]) {
  for (int i = 0; i < 3; ++i)
    out[i] = fdot3<false>(c[0], a[i], c[1], b[i], c[2], d[i]);
}

void sort_by_length(double basis[3][3]) {
  auto swap_row = [&](int i, int j) {
    for (int c = 0; c < 3; ++c) {
      const double t = basis[i][c];
      basis[i][c] = basis[j][c];
      basis[j][c] = t;
    }
  };
  if (n2(basis[0]) > n2(basis[1])) {
    swap_row(0, 1);
  }
  if (n2(basis[1]) > n2(basis[2])) {
    swap_row(1, 2);
  }
  if (n2(basis[0]) > n2(basis[1])) {
    swap_row(0, 1);
  }
}

void gauss_pair(double u[3], double v[3]) {
  for (int it = 0; it < kMaxIt; ++it) {
    if (n2(v) + kTol < n2(u)) {
      for (int c = 0; c < 3; ++c) {
        const double t = u[c];
        u[c] = v[c];
        v[c] = t;
      }
    }
    const double uu = n2(u);
    if (uu < kTol) {
      return;
    }
    const double n = std::round(vdot(u, v) / uu);
    if (n == 0.0) {
      return;
    }
    for (int c = 0; c < 3; ++c)
      v[c] = mul_add(-n, u[c], v[c]);
  }
}

void subtract_plane(double basis[3][3]) {
  const double *u = basis[0];
  const double *v = basis[1];
  double w[3] = {basis[2][0], basis[2][1], basis[2][2]};
  const double uu = n2(u);
  const double vv = n2(v);
  if (uu < kTol || vv < kTol) {
    return;
  }
  const double nu = std::round(vdot(w, u) / uu);
  double w1[3];
  for (int c = 0; c < 3; ++c)
    w1[c] = mul_add(-nu, u[c], w[c]);
  const double nv = std::round(vdot(w1, v) / vv);
  for (int c = 0; c < 3; ++c)
    w[c] = mul_add(-nv, v[c], w1[c]);
  double best[3] = {w[0], w[1], w[2]};
  double best2 = n2(w);
  for (int i = -1; i <= 1; ++i) {
    for (int j = -1; j <= 1; ++j) {
      if (i == 0 && j == 0) {
        continue;
      }
      double t[3];
      for (int c = 0; c < 3; ++c)
        t[c] = mul_add(static_cast<double>(i), u[c],
                        mul_add(static_cast<double>(j), v[c], w[c]));
      const double t2 = n2(t);
      if (t2 < best2) {
        best2 = t2;
        best[0] = t[0];
        best[1] = t[1];
        best[2] = t[2];
      }
    }
  }
  basis[2][0] = best[0];
  basis[2][1] = best[1];
  basis[2][2] = best[2];
}

bool is_minkowski_reduced(const double basis[3][3]) {
  const double na = n2(basis[0]);
  const double nb = n2(basis[1]);
  const double nc = n2(basis[2]);
  if (!(na - nb <= kTol && nb - nc <= kTol)) {
    return false;
  }
  const int rhs_idx[12] = {0, 1, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2};
  const double rows[12][3] = {
      {0, 1, 0}, {0, 0, 1}, {1, 1, 0}, {1, 0, 1}, {0, 1, 1}, {1, -1, 0},
      {1, 0, -1}, {0, 1, -1}, {1, 1, 1}, {1, -1, 1}, {1, 1, -1}, {1, -1, -1},
  };
  const double norms[3] = {std::sqrt(na), std::sqrt(nb), std::sqrt(nc)};
  for (int i = 0; i < 12; ++i) {
    double image[3];
    lin3(rows[i], basis[0], basis[1], basis[2], image);
    if (std::sqrt(n2(image)) + kTol < norms[rhs_idx[i]]) {
      return false;
    }
  }
  return true;
}

void minkowski_reduce(double basis[3][3]) {
  if (is_minkowski_reduced(basis)) {
    return;
  }
  for (int it = 0; it < kMaxIt; ++it) {
    sort_by_length(basis);
    gauss_pair(basis[0], basis[1]);
    subtract_plane(basis);
    sort_by_length(basis);
    if (is_minkowski_reduced(basis)) {
      return;
    }
  }
}

double det_of(const double a[9]) {
  const double m0 = mul_add(-a[5], a[7], a[4] * a[8]);
  const double m1 = mul_add(-a[5], a[6], a[3] * a[8]);
  const double m2 = mul_add(-a[4], a[6], a[3] * a[7]);
  return fdot3<false>(a[0], m0, -a[1], m1, a[2], m2);
}

bool invert(const double a[9], double inv[9]) {
  const double det = det_of(a);
  if (!std::isfinite(det) || det == 0.0) {
    return false;
  }
  inv[0] = mul_add(-a[5], a[7], a[4] * a[8]) / det;
  inv[1] = mul_add(-a[1], a[8], a[2] * a[7]) / det;
  inv[2] = mul_add(-a[2], a[4], a[1] * a[5]) / det;
  inv[3] = mul_add(-a[3], a[8], a[5] * a[6]) / det;
  inv[4] = mul_add(-a[2], a[6], a[0] * a[8]) / det;
  inv[5] = mul_add(-a[0], a[5], a[2] * a[3]) / det;
  inv[6] = mul_add(-a[4], a[6], a[3] * a[7]) / det;
  inv[7] = mul_add(-a[0], a[7], a[1] * a[6]) / det;
  inv[8] = mul_add(-a[1], a[3], a[0] * a[4]) / det;
  for (int k = 0; k < 9; ++k) {
    if (!std::isfinite(inv[k])) {
      return false;
    }
  }
  return true;
}

double inf_norm(const double m[9]) {
  double norm = 0.0;
  for (int row = 0; row < 3; ++row) {
    const double a = std::abs(m[3 * row]);
    const double b = std::abs(m[3 * row + 1]);
    const double c = std::abs(m[3 * row + 2]);
    const double sum = (a + b) + c;
    norm = std::max(norm, sum);
  }
  return norm;
}

bool well_conditioned(const double cell[9], const double inv[9]) {
  const double eps = std::numeric_limits<double>::epsilon();
  const double cond = (inf_norm(cell) * inf_norm(inv)) * eps;
  return cond <= std::sqrt(eps);
}

std::int64_t checked_index(double value) {
  const double limit = std::ldexp(1.0, 63);
  if (!std::isfinite(value) || value <= -limit || value >= limit) {
    throw std::overflow_error("periodic image index is out of range");
  }
  return static_cast<std::int64_t>(value);
}

template <bool Hardware>
double squared(double x, double y, double z) {
  return fdot3<Hardware>(x, x, y, y, z, z);
}

// Periodic rows of the cell, LLL-reduced. b holds the reduced rows and
// u the integer change of basis, b[r] = sum_j u[r][j] * cell row rows[j].
struct ReducedRows {
  int n = 0;
  int rows[3] = {};
  double b[3][3] = {};
  double u[3][3] = {};
};

double dot3(const double x[3], const double y[3]) {
  return fdot3<false>(x[0], y[0], x[1], y[1], x[2], y[2]);
}

// Gram-Schmidt of the first n rows: bstar and the squared norms.
void gram_schmidt(const ReducedRows &r, double bstar[3][3], double norm2[3],
                  double mu[3][3]) {
  for (int i = 0; i < r.n; ++i) {
    for (int c = 0; c < 3; ++c)
      bstar[i][c] = r.b[i][c];
    for (int j = 0; j < i; ++j) {
      mu[i][j] = dot3(r.b[i], bstar[j]) / norm2[j];
      for (int c = 0; c < 3; ++c)
        bstar[i][c] = mul_add(-mu[i][j], bstar[j][c], bstar[i][c]);
    }
    norm2[i] = dot3(bstar[i], bstar[i]);
  }
}

// LLL with delta = 0.99 on at most three rows. Each swap multiplies the
// potential prod_i |b*_i|^(2(n-i)) by less than delta, and the potential is
// bounded below by the lattice, so the loop terminates. kLllCap is a guard
// on top of that argument, not a substitute: it throws instead of spinning.
ReducedRows lll_rows(const double cell[9], const bool periodic[3]) {
  ReducedRows r;
  for (int k = 0; k < 3; ++k) {
    if (!periodic[k]) {
      continue;
    }
    r.rows[r.n] = k;
    for (int c = 0; c < 3; ++c)
      r.b[r.n][c] = cell[3 * k + c];
    r.u[r.n][r.n] = 1.0;
    ++r.n;
  }
  double bstar[3][3] = {};
  double norm2[3] = {};
  double mu[3][3] = {};
  gram_schmidt(r, bstar, norm2, mu);
  int i = 1;
  int steps = 0;
  while (i < r.n) {
    if (++steps > kLllCap) {
      throw std::overflow_error("periodic cell reduction did not converge");
    }
    for (int j = i - 1; j >= 0; --j) {
      const double q = std::round(mu[i][j]);
      if (q == 0.0) {
        continue;
      }
      if (!(std::fabs(q) < 0x1p52)) {
        throw std::overflow_error("periodic cell reduction is out of range");
      }
      for (int c = 0; c < 3; ++c) {
        r.b[i][c] = mul_add(-q, r.b[j][c], r.b[i][c]);
        r.u[i][c] = mul_add(-q, r.u[j][c], r.u[i][c]);
      }
      gram_schmidt(r, bstar, norm2, mu);
    }
    if (norm2[i] >= (0.99 - mu[i][i - 1] * mu[i][i - 1]) * norm2[i - 1]) {
      ++i;
      continue;
    }
    for (int c = 0; c < 3; ++c) {
      std::swap(r.b[i][c], r.b[i - 1][c]);
      std::swap(r.u[i][c], r.u[i - 1][c]);
    }
    gram_schmidt(r, bstar, norm2, mu);
    i = std::max(i - 1, 1);
  }
  return r;
}

// Minimum image over the periodic rows through an LLL-reduced basis. A
// lattice point v within distance R of the in-span part of the target has
// reduced coordinates m_i = d_i . v, so |m_i - d_i . p| <= |d_i| R, where
// d_i are the dual rows inside the span. For a reduced basis that box has
// a small constant size. Candidates are evaluated with the same sub_image
// and squared as search_bounds, in cell-row coordinates, and ties go to the
// lexicographically smallest index triple, the one search_bounds keeps.
template <bool Hardware>
void search_reduced(const double cell[9], const bool periodic[3],
                    const double reduced[3], double &dx, double &dy,
                    double &dz) {
  const ReducedRows r = lll_rows(cell, periodic);
  const int n = r.n;
  double g[3][3] = {};
  for (int a = 0; a < n; ++a)
    for (int b = 0; b < n; ++b)
      g[a][b] = dot3(r.b[a], r.b[b]);
  double gi[3][3] = {};
  if (n == 1) {
    gi[0][0] = 1.0 / g[0][0];
  } else if (n == 2) {
    const double det = mul_add(-g[0][1], g[1][0], g[0][0] * g[1][1]);
    gi[0][0] = g[1][1] / det;
    gi[1][1] = g[0][0] / det;
    gi[0][1] = -g[0][1] / det;
    gi[1][0] = -g[1][0] / det;
  } else {
    double flat[9];
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b)
        flat[3 * a + b] = g[a][b];
    double finv[9];
    if (!invert(flat, finv)) {
      throw std::invalid_argument("periodic cell must be nonsingular");
    }
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b)
        gi[a][b] = finv[3 * a + b];
  }
  double dual_rows[3][3] = {};
  double f[3] = {};
  for (int a = 0; a < n; ++a) {
    for (int c = 0; c < 3; ++c) {
      double acc = 0.0;
      for (int b = 0; b < n; ++b)
        acc = mul_add(gi[a][b], r.b[b][c], acc);
      dual_rows[a][c] = acc;
    }
    f[a] = dot3(dual_rows[a], reduced);
  }
  // In-span residual of the rounded (Babai) point bounds the search radius.
  double e[3] = {};
  for (int c = 0; c < 3; ++c) {
    double acc = 0.0;
    for (int a = 0; a < n; ++a)
      acc = mul_add(f[a] - std::round(f[a]), r.b[a][c], acc);
    e[c] = acc;
  }
  const double rin = std::sqrt(dot3(e, e)) * (1.0 + 0x1p-20);
  double lo[3] = {};
  double hi[3] = {};
  double box = 1.0;
  for (int a = 0; a < n; ++a) {
    const double ext = rin * std::sqrt(dot3(dual_rows[a], dual_rows[a]));
    lo[a] = std::floor(f[a] - ext) - 1.0;
    hi[a] = std::ceil(f[a] + ext) + 1.0;
    box *= hi[a] - lo[a] + 1.0;
  }
  if (!(box <= kReducedCap)) {
    throw std::overflow_error("periodic image search count is too large");
  }
  double best[3] = {reduced[0], reduced[1], reduced[2]};
  double best2 = squared<Hardware>(best[0], best[1], best[2]);
  double best_idx[3] = {0.0, 0.0, 0.0};
  bool have = false;
  double m[3] = {};
  const auto visit = [&] {
    double idx[3] = {0.0, 0.0, 0.0};
    for (int a = 0; a < n; ++a)
      for (int j = 0; j < n; ++j)
        idx[r.rows[j]] = mul_add(m[a], r.u[a][j], idx[r.rows[j]]);
    double cand[3];
    for (int c = 0; c < 3; ++c)
      cand[c] = sub_image<Hardware>(reduced[c], idx[0], cell[c], idx[1],
                                    cell[3 + c], idx[2], cell[6 + c]);
    const double r2 = squared<Hardware>(cand[0], cand[1], cand[2]);
    const bool lex_first =
        idx[0] < best_idx[0] ||
        (idx[0] == best_idx[0] &&
         (idx[1] < best_idx[1] ||
          (idx[1] == best_idx[1] && idx[2] < best_idx[2])));
    if (!have || r2 < best2 || (r2 == best2 && lex_first)) {
      have = true;
      best2 = r2;
      for (int c = 0; c < 3; ++c) {
        best[c] = cand[c];
        best_idx[c] = idx[c];
      }
    }
  };
  for (m[0] = lo[0]; m[0] <= hi[0]; m[0] += 1.0) {
    if (n == 1) {
      visit();
      continue;
    }
    for (m[1] = lo[1]; m[1] <= hi[1]; m[1] += 1.0) {
      if (n == 2) {
        visit();
        continue;
      }
      for (m[2] = lo[2]; m[2] <= hi[2]; m[2] += 1.0)
        visit();
    }
  }
  dx = best[0];
  dy = best[1];
  dz = best[2];
}

template <bool Hardware>
bool search_bounds(const double cell[9], const double inv[9],
                   const double dual[3], const bool periodic[3], double within2,
                   double &dx, double &dy, double &dz) {
  double reduced[3] = {dx, dy, dz};
  double initial[3] = {};
  for (int k = 0; k < 3; ++k) {
    if (!periodic[k]) {
      continue;
    }
    initial[k] = std::round(
        frac_of<Hardware>(reduced[0], reduced[1], reduced[2], inv, k));
    checked_index(initial[k]);
  }
  for (int c = 0; c < 3; ++c) {
    reduced[c] = sub_image<Hardware>(reduced[c], initial[0], cell[c], initial[1],
                                    cell[3 + c], initial[2], cell[6 + c]);
  }
  double best[3] = {reduced[0], reduced[1], reduced[2]};
  double best2 = squared<Hardware>(best[0], best[1], best[2]);
  // Any other image is this one less a lattice vector with a nonzero
  // integer n_k on some periodic axis k. Its fractional coordinate there is
  // f_k - n_k, so it is at least (1 - |f_k|) / |d_k| long. Rounding left
  // |f_k| <= 1/2 unless the displacement outran double precision, and the
  // bound is then clamped to zero. An image shorter than the bound on every
  // periodic axis is the only one that short, and the search below would
  // return its bits unchanged. Half the smallest plane spacing,
  // 1 / (2 |d_k|), is the same test without the fractional coordinates.
  double densest = 0.0;
  for (int k = 0; k < 3; ++k) {
    if (periodic[k]) {
      densest = std::max(densest, dual[k]);
    }
  }
  if (best2 * (densest * densest) < 0.25 * (1.0 - kBoundSlack)) {
    dx = best[0];
    dy = best[1];
    dz = best[2];
    return true;
  }
  double f[3] = {};
  bool alone = true;
  for (int k = 0; k < 3; ++k) {
    if (periodic[k]) {
      f[k] = frac_of<Hardware>(reduced[0], reduced[1], reduced[2], inv, k);
      const double gap = std::max(1.0 - std::fabs(f[k]), 0.0);
      alone = alone &&
              best2 * (dual[k] * dual[k]) < (1.0 - kBoundSlack) * (gap * gap);
    }
  }
  if (alone) {
    dx = best[0];
    dy = best[1];
    dz = best[2];
    return true;
  }
  const double radius = std::sqrt(best2);
  std::int64_t low[3] = {};
  std::int64_t high[3] = {};
  double span[3][2] = {};
  double box = 1.0;
  for (int k = 0; k < 3; ++k) {
    if (periodic[k]) {
      const double extent = radius * dual[k];
      span[k][0] = std::floor(f[k] - extent) - 1.0;
      span[k][1] = std::ceil(f[k] + extent) + 1.0;
      box *= span[k][1] - span[k][0] + 1.0;
    }
  }
  // The box comes from the dual norms of the full cell and the whole
  // displacement length. A thin or strongly sheared cell, or a long
  // displacement along an open direction, makes it astronomically large.
  // Those cases go to a reduced-basis search over the periodic rows.
  if (!(box <= kExhaustiveCap)) {
    ++searches_run;
    search_reduced<Hardware>(cell, periodic, reduced, dx, dy, dz);
    return true;
  }
  // The same bound with the caller's radius: when the rounded image and
  // every other image are longer than that radius, no search is needed.
  // The box above is small enough that the search would not have thrown.
  const double reach2 = within2 * (1.0 + kBoundSlack);
  if (best2 > reach2) {
    bool beyond = true;
    for (int k = 0; k < 3; ++k) {
      if (periodic[k]) {
        const double gap = std::max(1.0 - std::fabs(f[k]), 0.0);
        beyond = beyond && reach2 * (dual[k] * dual[k]) <
                               (1.0 - kBoundSlack) * (gap * gap);
      }
    }
    if (beyond) {
      return false;
    }
  }
  ++searches_run;
  std::uint64_t count = 1;
  for (int k = 0; k < 3; ++k) {
    if (periodic[k]) {
      low[k] = checked_index(span[k][0]);
      high[k] = checked_index(span[k][1]);
    }
    const auto width = static_cast<std::uint64_t>(high[k]) -
                       static_cast<std::uint64_t>(low[k]) + 1;
    if (width == 0 ||
        count > static_cast<std::uint64_t>(
                    std::numeric_limits<std::int64_t>::max()) /
                    width) {
      throw std::overflow_error("periodic image search count is too large");
    }
    count *= width;
  }
  // sub_image fuses k first, then j, then i. The k and j partials do not
  // depend on i, so they are computed once and reused for every i. The
  // operations and the visiting order are unchanged, so the bits are too.
  const auto wj = static_cast<std::uint64_t>(high[1] - low[1]) + 1;
  const auto wk = static_cast<std::uint64_t>(high[2] - low[2]) + 1;
  if (wj * wk <= kPartialCap) {
    double qk[kPartialCap][3];
    for (std::uint64_t kk = 0; kk < wk; ++kk) {
      const double k = static_cast<double>(low[2] + static_cast<std::int64_t>(kk));
      for (int c = 0; c < 3; ++c)
        qk[kk][c] = product_add<Hardware>(-k, cell[6 + c], reduced[c]);
    }
    double qjk[kPartialCap][3];
    for (std::uint64_t jj = 0; jj < wj; ++jj) {
      const double j = static_cast<double>(low[1] + static_cast<std::int64_t>(jj));
      for (std::uint64_t kk = 0; kk < wk; ++kk) {
        for (int c = 0; c < 3; ++c)
          qjk[jj * wk + kk][c] = product_add<Hardware>(-j, cell[3 + c], qk[kk][c]);
      }
    }
    const std::uint64_t njk = wj * wk;
    for (auto i = low[0]; i <= high[0]; ++i) {
      const double ni = -static_cast<double>(i);
      std::uint64_t q = 0;
#if RGPOT_FOLD_X2
      if constexpr (!Hardware) {
        // Two candidates per step through the paired software fma. Lane 0
        // is compared before lane 1, which keeps the scalar visiting order.
        using rgpot::simd::exact_fma_x2;
        using rgpot::simd::f64x2;
        const f64x2 vni = {ni, ni};
        for (; q + 1 < njk; q += 2) {
          f64x2 cand[3];
          for (int c = 0; c < 3; ++c) {
            const f64x2 base = {qjk[q][c], qjk[q + 1][c]};
            const f64x2 cc = {cell[c], cell[c]};
            cand[c] = exact_fma_x2(vni, cc, base);
          }
          const f64x2 r2 = exact_fma_x2(
              cand[0], cand[0], exact_fma_x2(cand[1], cand[1], cand[2] * cand[2]));
          for (int lane = 0; lane < 2; ++lane) {
            if (r2[lane] < best2) {
              best[0] = cand[0][lane];
              best[1] = cand[1][lane];
              best[2] = cand[2][lane];
              best2 = r2[lane];
            }
          }
        }
      }
#endif
      for (; q < njk; ++q) {
        double cand[3];
        for (int c = 0; c < 3; ++c)
          cand[c] = product_add<Hardware>(ni, cell[c], qjk[q][c]);
        const double r2 = squared<Hardware>(cand[0], cand[1], cand[2]);
        if (r2 < best2) {
          best[0] = cand[0];
          best[1] = cand[1];
          best[2] = cand[2];
          best2 = r2;
        }
      }
    }
    dx = best[0];
    dy = best[1];
    dz = best[2];
    return true;
  }
  for (auto i = low[0]; i <= high[0]; ++i) {
    for (auto j = low[1]; j <= high[1]; ++j) {
      for (auto k = low[2]; k <= high[2]; ++k) {
        double cand[3];
        for (int c = 0; c < 3; ++c) {
          cand[c] = sub_image<Hardware>(reduced[c], static_cast<double>(i), cell[c],
                                       static_cast<double>(j), cell[3 + c],
                                       static_cast<double>(k), cell[6 + c]);
        }
        const double r2 = squared<Hardware>(cand[0], cand[1], cand[2]);
        if (r2 < best2) {
          best[0] = cand[0];
          best[1] = cand[1];
          best[2] = cand[2];
          best2 = r2;
        }
      }
    }
  }
  dx = best[0];
  dy = best[1];
  dz = best[2];
  return true;
}

template <bool Hardware>
void search_narrow(const double cell[9], const double inv[9],
                   const bool periodic[3], double &dx, double &dy,
                   double &dz) {
  int base[3] = {};
  for (int k = 0; k < 3; ++k) {
    if (!periodic[k]) {
      continue;
    }
    const double f = frac_of<Hardware>(dx, dy, dz, inv, k);
    if (!std::isfinite(f)) {
      throw std::overflow_error("periodic image index is out of range");
    }
    base[k] = static_cast<int>(checked_index(std::round(f)));
  }
  double reduced[3] = {dx, dy, dz};
  for (int c = 0; c < 3; ++c) {
    reduced[c] =
        sub_image<Hardware>(reduced[c], static_cast<double>(base[0]), cell[c],
                            static_cast<double>(base[1]), cell[3 + c],
                            static_cast<double>(base[2]), cell[6 + c]);
  }
  double best[3] = {reduced[0], reduced[1], reduced[2]};
  double best2 = squared<Hardware>(best[0], best[1], best[2]);
  for (int i = -1; i <= 1; ++i) {
    if (!periodic[0] && i != 0) {
      continue;
    }
    for (int j = -1; j <= 1; ++j) {
      if (!periodic[1] && j != 0) {
        continue;
      }
      for (int k = -1; k <= 1; ++k) {
        if (!periodic[2] && k != 0) {
          continue;
        }
        double cand[3];
        for (int c = 0; c < 3; ++c) {
          cand[c] = sub_image<Hardware>(reduced[c], static_cast<double>(i), cell[c],
                                       static_cast<double>(j), cell[3 + c],
                                       static_cast<double>(k), cell[6 + c]);
        }
        const double r2 = squared<Hardware>(cand[0], cand[1], cand[2]);
        if (r2 < best2) {
          best2 = r2;
          best[0] = cand[0];
          best[1] = cand[1];
          best[2] = cand[2];
        }
      }
    }
  }
  dx = best[0];
  dy = best[1];
  dz = best[2];
}

void dual_of(const double inv[9], double dual[3]) {
  for (int k = 0; k < 3; ++k) {
    dual[k] = std::hypot(inv[k], inv[3 + k], inv[6 + k]);
    if (!std::isfinite(dual[k])) {
      throw std::overflow_error("periodic cell inverse is not finite");
    }
  }
}

} // namespace

MinimumImage::MinimumImage(const double *cell,
                           const std::array<bool, 3> &periodic)
    : periodic_(periodic) {
  for (int k = 0; k < 9; ++k) {
    if (!std::isfinite(cell[k])) {
      throw std::invalid_argument("periodic cell must be finite");
    }
    cell_[static_cast<std::size_t>(k)] = cell[k];
  }
  double raw[9];
  for (int k = 0; k < 9; ++k) {
    raw[k] = cell_[static_cast<std::size_t>(k)];
  }
  double inv[9];
  if (!invert(raw, inv)) {
    throw std::invalid_argument("periodic cell must be nonsingular");
  }
  double basis[3][3] = {
      {raw[0], raw[1], raw[2]},
      {raw[3], raw[4], raw[5]},
      {raw[6], raw[7], raw[8]},
  };
  bool narrow = !well_conditioned(raw, inv);
  if (narrow) {
    minkowski_reduce(basis);
    double reduced[9];
    for (int row = 0; row < 3; ++row) {
      for (int col = 0; col < 3; ++col) {
        reduced[3 * row + col] = basis[row][col];
      }
    }
    double rinv[9];
    if (!invert(reduced, rinv)) {
      throw std::invalid_argument("periodic cell must be nonsingular");
    }
    for (int k = 0; k < 9; ++k) {
      cell_[static_cast<std::size_t>(k)] = reduced[k];
      inverse_[static_cast<std::size_t>(k)] = rinv[k];
    }
    narrow = !well_conditioned(reduced, rinv);
    if (!narrow) {
      dual_of(rinv, dual_norm_.data());
    }
  } else {
    for (int k = 0; k < 9; ++k) {
      inverse_[static_cast<std::size_t>(k)] = inv[k];
    }
    dual_of(inv, dual_norm_.data());
  }
  narrow_ = narrow;
}

namespace {

template <bool Hardware>
bool fold_search(bool narrow, const double cell[9], const double inv[9],
                 const double dual[3], const bool periodic[3], double within2,
                 double &dx, double &dy, double &dz) {
  if (narrow) {
    ++searches_run;
    search_narrow<Hardware>(cell, inv, periodic, dx, dy, dz);
    return true;
  }
  return search_bounds<Hardware>(cell, inv, dual, periodic, within2, dx, dy,
                                 dz);
}

// The attribute keeps vfmadd in this body. The caller enters it only after
// simd::detail::fma_hw_active() finds a usable FMA, and the guard is the
// one that function uses: target("fma") is an x86 target name, which GCC
// rejects on aarch64. Elsewhere fma_hw_active() is false, so this body is
// never entered and the caller keeps the portable path.
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("fma"), noinline, flatten))
#endif
bool fold_hardware(bool narrow, const double cell[9], const double inv[9],
                   const double dual[3], const bool periodic[3], double within2,
                   double &dx, double &dy, double &dz) {
  return fold_search<true>(narrow, cell, inv, dual, periodic, within2, dx, dy,
                           dz);
}

} // namespace

void MinimumImage::fold(double &dx, double &dy, double &dz) const {
  (void)foldWithin(dx, dy, dz, std::numeric_limits<double>::infinity());
}

std::uint64_t MinimumImage::searches() noexcept { return searches_run; }

bool MinimumImage::foldWithin(double &dx, double &dy, double &dz,
                              double within2) const {
  if (!std::isfinite(dx) || !std::isfinite(dy) || !std::isfinite(dz)) {
    throw std::invalid_argument("pair displacement must be finite");
  }
  bool periodic[3] = {periodic_[0], periodic_[1], periodic_[2]};
  double cell[9];
  double inv[9];
  for (int k = 0; k < 9; ++k) {
    cell[k] = cell_[static_cast<std::size_t>(k)];
    inv[k] = inverse_[static_cast<std::size_t>(k)];
  }
  double dual[3] = {dual_norm_[0], dual_norm_[1], dual_norm_[2]};
  if (rgpot::simd::detail::fma_hw_active()) {
    return fold_hardware(narrow_, cell, inv, dual, periodic, within2, dx, dy,
                         dz);
  }
  return fold_search<false>(narrow_, cell, inv, dual, periodic, within2, dx, dy,
                            dz);
}

} // namespace rgpot::nlist
