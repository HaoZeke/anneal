// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/LennardJones/LJClusterKernel.hpp"

#include <cstddef>
#include <cstdlib>
#include <new>

#if defined(_WIN32)
#include <malloc.h>
#endif
#if defined(__x86_64__)
#include <immintrin.h>
#endif
#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace rgpot {
namespace {

// MSVC does not provide std::aligned_alloc. The CRT pair is
// _aligned_malloc / _aligned_free; free() cannot release that pointer.
// Elsewhere posix_memalign, not aligned_alloc: a malloc linked ahead of libc,
// such as the jemalloc that conda's RocksDB pulls in, can replace free and
// posix_memalign but not aligned_alloc. free then gets a foreign pointer.
void *alloc_aligned(std::size_t alignment, std::size_t bytes) {
#if defined(_WIN32)
  return _aligned_malloc(bytes, alignment);
#else
  void *mem = nullptr;
  return posix_memalign(&mem, alignment, bytes) == 0 ? mem : nullptr;
#endif
}

void free_aligned(void *mem) {
#if defined(_WIN32)
  _aligned_free(mem);
#else
  std::free(mem);
#endif
}

struct Soa {
  double *x{nullptr};
  double *y{nullptr};
  double *z{nullptr};
  std::size_t cap{0};

  Soa() = default;
  Soa(const Soa &) = delete;
  Soa &operator=(const Soa &) = delete;
  ~Soa() { free_aligned(x); }

  void ensure(std::size_t n) {
    if (n <= cap) {
      return;
    }
    std::size_t nc = cap == 0 ? 32 : cap * 2;
    if (nc < n) {
      nc = n;
    }
    nc = (nc + 7u) & ~std::size_t{7};
    const std::size_t bytes = nc * 3 * sizeof(double);
    void *mem = alloc_aligned(64, bytes);
    if (mem == nullptr) {
      throw std::bad_alloc();
    }
    free_aligned(x);
    x = static_cast<double *>(mem);
    y = x + nc;
    z = y + nc;
    cap = nc;
  }
};

Soa &soa_scratch() {
  thread_local Soa soa;
  return soa;
}

inline void load_soa(const double *pos, std::size_t n, double *x, double *y, double *z) {
  for (std::size_t i = 0; i < n; ++i) {
    x[i] = pos[3 * i];
    y[i] = pos[3 * i + 1];
    z[i] = pos[3 * i + 2];
  }
}

inline void acc_one(double dx, double dy, double dz, double r2, double cutoff2, double psi2,
                    double four_u0, double shift, double &energy, double *forces, std::size_t i,
                    std::size_t j) {
  if (!(r2 <= cutoff2)) {
    return;
  }
  const double inv = 1.0 / r2;
  const double sr2 = psi2 * inv;
  const double a = (sr2 * sr2) * sr2;
  const double b = four_u0 * a;
  energy += (b * (a - 1.0)) - shift;
  const double scale = ((6.0 * b) * inv) * ((2.0 * a) - 1.0);
  const double fx = scale * dx;
  const double fy = scale * dy;
  const double fz = scale * dz;
  forces[3 * i] += fx;
  forces[3 * i + 1] += fy;
  forces[3 * i + 2] += fz;
  forces[3 * j] -= fx;
  forces[3 * j + 1] -= fy;
  forces[3 * j + 2] -= fz;
}

void lj_scalar(const double *x, const double *y, const double *z, std::size_t n, double cutoff2,
               double psi2, double four_u0, double shift, double *forces, double &energy) {
  for (std::size_t i = 0; i < n; ++i) {
    const double xi = x[i];
    const double yi = y[i];
    const double zi = z[i];
    for (std::size_t j = i + 1; j < n; ++j) {
      const double dx = xi - x[j];
      const double dy = yi - y[j];
      const double dz = zi - z[j];
      const double r2 = (dx * dx + dy * dy) + dz * dz;
      acc_one(dx, dy, dz, r2, cutoff2, psi2, four_u0, shift, energy, forces, i, j);
    }
  }
}

#if defined(__x86_64__)
// target("avx2") also emits AVX encodings. lj_cluster_direct calls lj_avx2
// only after __builtin_cpu_supports("avx2"), and the ISA audit proves that
// from the library: lj_avx2, eval_quad, and acc_quad must stay internal and
// reached only from behind that check. The target attribute keeps that body
// out of the baseline caller.
struct Quad {
  alignas(32) double r2[4];
  alignas(32) double e[4];
  alignas(32) double fx[4];
  alignas(32) double fy[4];
  alignas(32) double fz[4];
};

__attribute__((target("avx2"))) void eval_quad(double xi, double yi, double zi, const double *x,
                                               const double *y, const double *z, std::size_t j,
                                               __m256d vpsi2, __m256d vfour, __m256d vone,
                                               __m256d vshift, __m256d vsix, __m256d vtwo,
                                               Quad &q) {
  const __m256d dx = _mm256_sub_pd(_mm256_set1_pd(xi), _mm256_loadu_pd(x + j));
  const __m256d dy = _mm256_sub_pd(_mm256_set1_pd(yi), _mm256_loadu_pd(y + j));
  const __m256d dz = _mm256_sub_pd(_mm256_set1_pd(zi), _mm256_loadu_pd(z + j));
  const __m256d r2 = _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(dx, dx), _mm256_mul_pd(dy, dy)),
                                   _mm256_mul_pd(dz, dz));
  const __m256d inv = _mm256_div_pd(vone, r2);
  const __m256d sr2 = _mm256_mul_pd(vpsi2, inv);
  const __m256d a = _mm256_mul_pd(_mm256_mul_pd(sr2, sr2), sr2);
  const __m256d b = _mm256_mul_pd(vfour, a);
  const __m256d term =
      _mm256_sub_pd(_mm256_mul_pd(b, _mm256_sub_pd(a, vone)), vshift);
  const __m256d scale = _mm256_mul_pd(_mm256_mul_pd(_mm256_mul_pd(vsix, b), inv),
                                      _mm256_sub_pd(_mm256_mul_pd(vtwo, a), vone));
  _mm256_store_pd(q.r2, r2);
  _mm256_store_pd(q.e, term);
  _mm256_store_pd(q.fx, _mm256_mul_pd(scale, dx));
  _mm256_store_pd(q.fy, _mm256_mul_pd(scale, dy));
  _mm256_store_pd(q.fz, _mm256_mul_pd(scale, dz));
}

__attribute__((target("avx2"))) void acc_quad(const Quad &q, double cutoff2, double &energy,
                                              double *forces, std::size_t i, std::size_t j) {
  for (int k = 0; k < 4; ++k) {
    if (!(q.r2[k] <= cutoff2)) {
      continue;
    }
    energy += q.e[k];
    const double fx = q.fx[k];
    const double fy = q.fy[k];
    const double fz = q.fz[k];
    forces[3 * i] += fx;
    forces[3 * i + 1] += fy;
    forces[3 * i + 2] += fz;
    const std::size_t jj = j + static_cast<std::size_t>(k);
    forces[3 * jj] -= fx;
    forces[3 * jj + 1] -= fy;
    forces[3 * jj + 2] -= fz;
  }
}

__attribute__((target("avx2"))) void lj_avx2(const double *x, const double *y, const double *z,
                                             std::size_t n, double cutoff2, double psi2,
                                             double four_u0, double shift, double *forces,
                                             double &energy) {
  const __m256d vpsi2 = _mm256_set1_pd(psi2);
  const __m256d vfour = _mm256_set1_pd(four_u0);
  const __m256d vone = _mm256_set1_pd(1.0);
  const __m256d vshift = _mm256_set1_pd(shift);
  const __m256d vsix = _mm256_set1_pd(6.0);
  const __m256d vtwo = _mm256_set1_pd(2.0);
  for (std::size_t i = 0; i < n; ++i) {
    const double xi = x[i];
    const double yi = y[i];
    const double zi = z[i];
    std::size_t j = i + 1;
    Quad a;
    Quad b;
    for (; j + 7 < n; j += 8) {
      eval_quad(xi, yi, zi, x, y, z, j, vpsi2, vfour, vone, vshift, vsix, vtwo, a);
      eval_quad(xi, yi, zi, x, y, z, j + 4, vpsi2, vfour, vone, vshift, vsix, vtwo, b);
      acc_quad(a, cutoff2, energy, forces, i, j);
      acc_quad(b, cutoff2, energy, forces, i, j + 4);
    }
    for (; j + 3 < n; j += 4) {
      eval_quad(xi, yi, zi, x, y, z, j, vpsi2, vfour, vone, vshift, vsix, vtwo, a);
      acc_quad(a, cutoff2, energy, forces, i, j);
    }
    for (; j < n; ++j) {
      const double dx = xi - x[j];
      const double dy = yi - y[j];
      const double dz = zi - z[j];
      const double r2 = (dx * dx + dy * dy) + dz * dz;
      acc_one(dx, dy, dz, r2, cutoff2, psi2, four_u0, shift, energy, forces, i, j);
    }
  }
}
#endif

#if defined(__aarch64__)
struct Pair2 {
  alignas(16) double r2[2];
  alignas(16) double e[2];
  alignas(16) double fx[2];
  alignas(16) double fy[2];
  alignas(16) double fz[2];
};

void eval_pair2(double xi, double yi, double zi, const double *x, const double *y, const double *z,
                std::size_t j, float64x2_t vpsi2, float64x2_t vfour, float64x2_t vone,
                float64x2_t vshift, float64x2_t vsix, float64x2_t vtwo, Pair2 &q) {
  const float64x2_t dx = vsubq_f64(vdupq_n_f64(xi), vld1q_f64(x + j));
  const float64x2_t dy = vsubq_f64(vdupq_n_f64(yi), vld1q_f64(y + j));
  const float64x2_t dz = vsubq_f64(vdupq_n_f64(zi), vld1q_f64(z + j));
  const float64x2_t r2 =
      vaddq_f64(vaddq_f64(vmulq_f64(dx, dx), vmulq_f64(dy, dy)), vmulq_f64(dz, dz));
  const float64x2_t inv = vdivq_f64(vone, r2);
  const float64x2_t sr2 = vmulq_f64(vpsi2, inv);
  const float64x2_t a = vmulq_f64(vmulq_f64(sr2, sr2), sr2);
  const float64x2_t b = vmulq_f64(vfour, a);
  const float64x2_t term = vsubq_f64(vmulq_f64(b, vsubq_f64(a, vone)), vshift);
  const float64x2_t scale =
      vmulq_f64(vmulq_f64(vmulq_f64(vsix, b), inv), vsubq_f64(vmulq_f64(vtwo, a), vone));
  vst1q_f64(q.r2, r2);
  vst1q_f64(q.e, term);
  vst1q_f64(q.fx, vmulq_f64(scale, dx));
  vst1q_f64(q.fy, vmulq_f64(scale, dy));
  vst1q_f64(q.fz, vmulq_f64(scale, dz));
}

void acc_pair2(const Pair2 &q, double cutoff2, double &energy, double *forces, std::size_t i,
               std::size_t j) {
  for (int k = 0; k < 2; ++k) {
    if (!(q.r2[k] <= cutoff2)) {
      continue;
    }
    energy += q.e[k];
    const double fx = q.fx[k];
    const double fy = q.fy[k];
    const double fz = q.fz[k];
    forces[3 * i] += fx;
    forces[3 * i + 1] += fy;
    forces[3 * i + 2] += fz;
    const std::size_t jj = j + static_cast<std::size_t>(k);
    forces[3 * jj] -= fx;
    forces[3 * jj + 1] -= fy;
    forces[3 * jj + 2] -= fz;
  }
}

void lj_neon(const double *x, const double *y, const double *z, std::size_t n, double cutoff2,
             double psi2, double four_u0, double shift, double *forces, double &energy) {
  const float64x2_t vpsi2 = vdupq_n_f64(psi2);
  const float64x2_t vfour = vdupq_n_f64(four_u0);
  const float64x2_t vone = vdupq_n_f64(1.0);
  const float64x2_t vshift = vdupq_n_f64(shift);
  const float64x2_t vsix = vdupq_n_f64(6.0);
  const float64x2_t vtwo = vdupq_n_f64(2.0);
  for (std::size_t i = 0; i < n; ++i) {
    const double xi = x[i];
    const double yi = y[i];
    const double zi = z[i];
    std::size_t j = i + 1;
    Pair2 a;
    Pair2 b;
    for (; j + 3 < n; j += 4) {
      eval_pair2(xi, yi, zi, x, y, z, j, vpsi2, vfour, vone, vshift, vsix, vtwo, a);
      eval_pair2(xi, yi, zi, x, y, z, j + 2, vpsi2, vfour, vone, vshift, vsix, vtwo, b);
      acc_pair2(a, cutoff2, energy, forces, i, j);
      acc_pair2(b, cutoff2, energy, forces, i, j + 2);
    }
    for (; j + 1 < n; j += 2) {
      eval_pair2(xi, yi, zi, x, y, z, j, vpsi2, vfour, vone, vshift, vsix, vtwo, a);
      acc_pair2(a, cutoff2, energy, forces, i, j);
    }
    if (j < n) {
      const double dx = xi - x[j];
      const double dy = yi - y[j];
      const double dz = zi - z[j];
      const double r2 = (dx * dx + dy * dy) + dz * dz;
      acc_one(dx, dy, dz, r2, cutoff2, psi2, four_u0, shift, energy, forces, i, j);
    }
  }
}
#endif

} // namespace

void lj_cluster_direct(const double *pos, std::size_t n, double cutoff, double u0, double psi,
                       double cutoff_u, double *forces, double *energy) {
  if (n < 2 || energy == nullptr) {
    return;
  }
  Soa &soa = soa_scratch();
  soa.ensure(n);
  load_soa(pos, n, soa.x, soa.y, soa.z);
  const double cutoff2 = cutoff * cutoff;
  const double psi2 = psi * psi;
  const double four_u0 = 4.0 * u0;
  double acc = *energy;
#if defined(__x86_64__)
  if (__builtin_cpu_supports("avx2")) {
    lj_avx2(soa.x, soa.y, soa.z, n, cutoff2, psi2, four_u0, cutoff_u, forces, acc);
    *energy = acc;
    return;
  }
#endif
#if defined(__aarch64__)
  lj_neon(soa.x, soa.y, soa.z, n, cutoff2, psi2, four_u0, cutoff_u, forces, acc);
  *energy = acc;
  return;
#endif
  lj_scalar(soa.x, soa.y, soa.z, n, cutoff2, psi2, four_u0, cutoff_u, forces, acc);
  *energy = acc;
}

} // namespace rgpot
