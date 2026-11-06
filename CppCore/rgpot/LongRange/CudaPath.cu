// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/LongRange/CudaPath.hpp"

#include "rgpot/LongRange/CudaMath.cuh"
#include "rgpot/LongRange/Lattice.hpp"
#include "rgpot/units.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <vector>

namespace rgpot::lr {
namespace {

constexpr double kKe = units::HARTREE_TO_EV * units::BOHR_TO_ANGSTROM;

__global__ void pair_sum_kernel(const double *pair, int n, double *out) {
  double local = 0.0;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
    local += pair[i];
  }
  __shared__ double buf[128];
  buf[threadIdx.x] = local;
  __syncthreads();
  if (threadIdx.x == 0) {
    double s = 0.0;
    for (int i = 0; i < blockDim.x; ++i) {
      s += buf[i];
    }
    atomicAdd(out, s);
  }
}

__global__ void damp_kernel(const double *c6, const double *r0, const double *r, int n, double a1,
                            double a2, double *out) {
  double local = 0.0;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
    local += cuda::bj_pair(c6[i], r0[i], a1, a2, r[i]);
  }
  __shared__ double buf[128];
  buf[threadIdx.x] = local;
  __syncthreads();
  if (threadIdx.x == 0) {
    double s = 0.0;
    for (int i = 0; i < blockDim.x; ++i) {
      s += buf[i];
    }
    atomicAdd(out, s);
  }
}

__global__ void ewald_real_kernel(const int *pi, const int *pj, const double *px, const double *py,
                                  const double *pz, const double *pr, const double *q, int npair,
                                  double alpha, double cutoff, double *energy, double *F) {
  for (int t = blockIdx.x * blockDim.x + threadIdx.x; t < npair; t += blockDim.x * gridDim.x) {
    const double r = pr[t];
    if (!(r > 0.0) || r >= cutoff) {
      continue;
    }
    double dedr = 0.0;
    const double e = cuda::real_pair(kKe, alpha, q[pi[t]], q[pj[t]], r, &dedr);
    atomicAdd(energy, e);
    if (F != nullptr) {
      const double invr = 1.0 / r;
      const int i = pi[t];
      const int j = pj[t];
      atomicAdd(F + 3 * i, dedr * px[t] * invr);
      atomicAdd(F + 3 * i + 1, dedr * py[t] * invr);
      atomicAdd(F + 3 * i + 2, dedr * pz[t] * invr);
      atomicAdd(F + 3 * j, -dedr * px[t] * invr);
      atomicAdd(F + 3 * j + 1, -dedr * py[t] * invr);
      atomicAdd(F + 3 * j + 2, -dedr * pz[t] * invr);
    }
  }
}

template <class T>
T *dev_copy(const T *src, std::size_t n) {
  if (n == 0 || src == nullptr) {
    return nullptr;
  }
  T *dst = nullptr;
  if (cudaMalloc(&dst, n * sizeof(T)) != cudaSuccess) {
    return nullptr;
  }
  if (cudaMemcpy(dst, src, n * sizeof(T), cudaMemcpyHostToDevice) != cudaSuccess) {
    cudaFree(dst);
    return nullptr;
  }
  return dst;
}

bool launch_ewald(const CudaCoulombIn &in, double *energy, double *F) {
  int *pi = dev_copy(in.pair_i, in.npair);
  int *pj = dev_copy(in.pair_j, in.npair);
  double *px = dev_copy(in.pair_x, in.npair);
  double *py = dev_copy(in.pair_y, in.npair);
  double *pz = dev_copy(in.pair_z, in.npair);
  double *pr = dev_copy(in.pair_r, in.npair);
  double *q = dev_copy(in.q, in.n);
  double *dE = nullptr;
  double *dF = nullptr;
  bool ok = q != nullptr && cudaMalloc(&dE, sizeof(double)) == cudaSuccess;
  if (ok && F != nullptr) {
    ok = cudaMalloc(&dF, 3 * in.n * sizeof(double)) == cudaSuccess &&
         cudaMemset(dF, 0, 3 * in.n * sizeof(double)) == cudaSuccess;
  }
  if (ok) {
    ok = cudaMemset(dE, 0, sizeof(double)) == cudaSuccess;
  }
  if (ok && in.npair > 0 && (pi == nullptr || pr == nullptr)) {
    ok = false;
  }
  if (ok && in.npair > 0) {
    const int threads = 128;
    const int blocks = static_cast<int>((in.npair + threads - 1) / threads);
    ewald_real_kernel<<<blocks, threads>>>(pi, pj, px, py, pz, pr, q, static_cast<int>(in.npair),
                                           in.alpha, in.cutoff, dE, dF);
    ok = cudaDeviceSynchronize() == cudaSuccess;
  }
  double real_e = 0.0;
  if (ok) {
    ok = cudaMemcpy(&real_e, dE, sizeof(double), cudaMemcpyDeviceToHost) == cudaSuccess;
  }
  if (ok && F != nullptr && dF != nullptr) {
    ok = cudaMemcpy(F, dF, 3 * in.n * sizeof(double), cudaMemcpyDeviceToHost) == cudaSuccess;
  }
  cudaFree(pi);
  cudaFree(pj);
  cudaFree(px);
  cudaFree(py);
  cudaFree(pz);
  cudaFree(pr);
  cudaFree(q);
  cudaFree(dE);
  cudaFree(dF);
  if (!ok) {
    return false;
  }
  // Reciprocal space and the self term stay on the host copy so the
  // k-shell matches the serial sum exactly. The pair kernel is the
  // part that moves.
  CudaCoulombIn host = in;
  host.npair = 0;
  double rest = 0.0;
  std::vector<double> restF;
  double *frest = nullptr;
  if (F != nullptr) {
    restF.assign(3 * in.n, 0.0);
    frest = restF.data();
  }
  coulomb_host(host, &rest, frest);
  // coulomb_host with no pairs still adds self and reciprocal. The real
  // pair energy is real_e. Forces add.
  *energy = rest + real_e;
  if (F != nullptr) {
    for (std::size_t i = 0; i < 3 * in.n; ++i) {
      F[i] += restF[i];
    }
  }
  return true;
}

} // namespace

int cuda_device_count() {
  int n = 0;
  if (cudaGetDeviceCount(&n) != cudaSuccess) {
    cudaGetLastError();
    return 0;
  }
  return n < 0 ? 0 : n;
}

void coulomb_cuda(const CudaCoulombIn &in, double *energy, double *F, int *used_device) {
  if (used_device == nullptr) {
    throw std::invalid_argument("cuda status pointer is null");
  }
  *used_device = 0;
  if (cuda_device_count() > 0 && in.method == 0 && launch_ewald(in, energy, F)) {
    *used_device = 1;
    return;
  }
  coulomb_host(in, energy, F);
}

double pair_sum_cuda(const double *pair, std::size_t n, int *used_device) {
  if (used_device == nullptr) {
    throw std::invalid_argument("cuda status pointer is null");
  }
  *used_device = 0;
  if (cuda_device_count() > 0 && n > 0) {
    double *src = nullptr;
    double *dst = nullptr;
    if (cudaMalloc(&src, n * sizeof(double)) == cudaSuccess &&
        cudaMalloc(&dst, sizeof(double)) == cudaSuccess &&
        cudaMemcpy(src, pair, n * sizeof(double), cudaMemcpyHostToDevice) == cudaSuccess &&
        cudaMemset(dst, 0, sizeof(double)) == cudaSuccess) {
      const int threads = 128;
      const int blocks = static_cast<int>((n + threads - 1) / threads);
      pair_sum_kernel<<<blocks, threads>>>(src, static_cast<int>(n), dst);
      double sum = 0.0;
      if (cudaDeviceSynchronize() == cudaSuccess &&
          cudaMemcpy(&sum, dst, sizeof(double), cudaMemcpyDeviceToHost) == cudaSuccess) {
        cudaFree(src);
        cudaFree(dst);
        *used_device = 1;
        return sum;
      }
    }
    cudaFree(src);
    cudaFree(dst);
  }
  return pair_sum_host(pair, n);
}

double damp_sum_cuda(const double *c6, const double *r0, const double *r, std::size_t n, double a1,
                     double a2, int *used_device) {
  if (used_device == nullptr) {
    throw std::invalid_argument("cuda status pointer is null");
  }
  *used_device = 0;
  if (cuda_device_count() > 0 && n > 0) {
    double *dc6 = nullptr;
    double *dr0 = nullptr;
    double *dr = nullptr;
    double *dst = nullptr;
    const bool alloc = cudaMalloc(&dc6, n * sizeof(double)) == cudaSuccess &&
                       cudaMalloc(&dr0, n * sizeof(double)) == cudaSuccess &&
                       cudaMalloc(&dr, n * sizeof(double)) == cudaSuccess &&
                       cudaMalloc(&dst, sizeof(double)) == cudaSuccess;
    if (alloc && cudaMemcpy(dc6, c6, n * sizeof(double), cudaMemcpyHostToDevice) == cudaSuccess &&
        cudaMemcpy(dr0, r0, n * sizeof(double), cudaMemcpyHostToDevice) == cudaSuccess &&
        cudaMemcpy(dr, r, n * sizeof(double), cudaMemcpyHostToDevice) == cudaSuccess &&
        cudaMemset(dst, 0, sizeof(double)) == cudaSuccess) {
      const int threads = 128;
      damp_kernel<<<1, threads>>>(dc6, dr0, dr, static_cast<int>(n), a1, a2, dst);
      double sum = 0.0;
      if (cudaDeviceSynchronize() == cudaSuccess &&
          cudaMemcpy(&sum, dst, sizeof(double), cudaMemcpyDeviceToHost) == cudaSuccess) {
        cudaFree(dc6);
        cudaFree(dr0);
        cudaFree(dr);
        cudaFree(dst);
        *used_device = 1;
        return sum;
      }
    }
    cudaFree(dc6);
    cudaFree(dr0);
    cudaFree(dr);
    cudaFree(dst);
  }
  return damp_sum_host(c6, r0, r, n, a1, a2);
}

} // namespace rgpot::lr
