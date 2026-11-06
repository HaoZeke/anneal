#pragma once
// Optional CUDA path for Ewald and PME, and for a pair reduction.
// With no device the same kernels run on the host. The host result is
// what the tests compare to the serial implementation.

#include <cstddef>

namespace rgpot::lr {

struct CudaCoulombIn {
  int method = 0; // 0 Ewald, 1 PME
  const double *pos = nullptr;
  const double *q = nullptr;
  std::size_t n = 0;
  double total = 0.0;
  const double *box = nullptr;
  const int *pair_i = nullptr;
  const int *pair_j = nullptr;
  const double *pair_x = nullptr;
  const double *pair_y = nullptr;
  const double *pair_z = nullptr;
  const double *pair_r = nullptr;
  std::size_t npair = 0;
  double alpha = 0.3;
  double cutoff = 9.0;
  int kmax = 4;
  int grid = 16;
  int order = 4;
};

// Writes the energy and, when F is not null, the forces (length 3 n).
// used_device is 1 only when a kernel was launched.
void coulomb_cuda(const CudaCoulombIn &in, double *energy, double *F, int *used_device);

// Sum of a pairwise dispersion matrix. The host and the device use the
// same sequential reduction so the two agree to roundoff.
double pair_sum_cuda(const double *pair, std::size_t n, int *used_device);

// Rational-damping pair sum used by the D3/D4 CUDA kernel.
double damp_sum_host(const double *c6, const double *r0, const double *r, std::size_t n, double a1,
                     double a2);
double damp_sum_cuda(const double *c6, const double *r0, const double *r, std::size_t n, double a1,
                     double a2, int *used_device);

// Host-only copies of the kernels. Tests call these when no GPU is present.
double pair_sum_host(const double *pair, std::size_t n);
void coulomb_host(const CudaCoulombIn &in, double *energy, double *F);

int cuda_device_count();

} // namespace rgpot::lr
