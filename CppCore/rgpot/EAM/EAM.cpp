/*
** This file is part of eOn.
**
** SPDX-License-Identifier: BSD-3-Clause
**
** Copyright (c) 2010--present, eOn Development Team
** All rights reserved.
**
** Repo:
** https://github.com/TheochemUI/eOn
*/
#include "EAM.h"
#include "rgpot/nlist/NeighbourList.hpp"
#include "Parameters.h"
#include "StreamSpline.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

void EAM::cleanMemory() {
  std::vector<double>().swap(wrapped_);
  std::vector<Delta>().swap(deltas_);
  std::vector<std::int32_t>().swap(js_);
  std::vector<double>().swap(rho_);
  std::vector<double>().swap(dF_);
  std::vector<double>().swap(fscale_);
  std::vector<double>().swap(morse_);
  std::vector<double>().swap(gdx_);
  std::vector<double>().swap(gdy_);
  std::vector<double>().swap(gdz_);
  std::vector<double>().swap(ginv_);
  std::vector<std::int32_t>().swap(rev_);
  std::vector<double>().swap(thread_energy_);
  std::vector<double>().swap(partial_f_);
  std::vector<double>().swap(stream_rho_);
  std::vector<double>().swap(stream_force_);
  parts_.clear();
  part_n_ = -1;
  part_nt_ = -1;
  rev_stamp_ = 0;
  n_atoms_ = 0;
}

EAM::element_parameters EAM::get_element_parameters(int atomic_number) const {
  for (int i = 0; i < NPARAMS; i++) {
    if (el_params[i].Z == atomic_number) {
      return el_params[i];
    }
  }
  throw std::invalid_argument("EAM: no parameters for atomic number " +
                              std::to_string(atomic_number));
}

void EAM::prepare(long N, const double *R, const double *box) {
  wrapped_.resize(static_cast<std::size_t>(3 * N));
  double xmin = std::numeric_limits<double>::max();
  double ymin = xmin;
  double zmin = xmin;
  for (long i = 0; i < 3 * N; i += 3) {
    wrapped_[static_cast<std::size_t>(i)] = R[i];
    wrapped_[static_cast<std::size_t>(i + 1)] = R[i + 1];
    wrapped_[static_cast<std::size_t>(i + 2)] = R[i + 2];
    xmin = std::min(xmin, R[i]);
    ymin = std::min(ymin, R[i + 1]);
    zmin = std::min(zmin, R[i + 2]);
  }
  if (xmin > 0) {
    xmin = 0;
  }
  if (ymin > 0) {
    ymin = 0;
  }
  if (zmin > 0) {
    zmin = 0;
  }
  for (long i = 0; i < 3 * N; i += 3) {
    wrapped_[static_cast<std::size_t>(i)] += std::abs(xmin);
    wrapped_[static_cast<std::size_t>(i + 1)] += std::abs(ymin);
    wrapped_[static_cast<std::size_t>(i + 2)] += std::abs(zmin);
  }
  for (long i = 0; i < 3 * N; i++) {
    while (wrapped_[static_cast<std::size_t>(i)] > box[i % 3]) {
      wrapped_[static_cast<std::size_t>(i)] -= box[i % 3];
    }
  }
  box_len_[0] = box[0];
  box_len_[1] = box[1];
  box_len_[2] = box[2];
}

void EAM::buildList(long N) {
  double widest = 0.0;
  for (int i = 0; i < NPARAMS; i++) {
    widest = std::max(widest, el_params[i].r_cut);
  }
  double box9[9] = {box_len_[0], 0, 0, 0, box_len_[1], 0, 0, 0, box_len_[2]};
  // A stream does not reuse bins unless the coordinates match, so the
  // skin only adds pairs the kernel then rejects.
  const double skin = mode_ == rgpot::nlist::Mode::Stream ? 0.0 : kEamSkinOverCutoff * widest;
  rgpot::nlist::updateCells(neighbours_, wrapped_.data(), static_cast<std::size_t>(N), box9,
                            widest, skin, rgpot::nlist::Image::Half, mode_, {true, true, true});
  n_atoms_ = N;
  if (neighbours_.streaming()) {
    rev_.clear();
    rev_stamp_ = 0;
    return;
  }
  if (rev_stamp_ != neighbours_.builds()) {
    ensureRev();
    rev_stamp_ = neighbours_.builds();
  }
}

void EAM::ensureRev() {
  const auto &off = neighbours_.offsets();
  const auto &nbr = neighbours_.neighbours();
  rev_.assign(nbr.size(), 0);
  for (std::size_t i = 0; i + 1 < off.size(); ++i) {
    for (std::int32_t p = off[i]; p < off[i + 1]; ++p) {
      const auto j = static_cast<std::size_t>(nbr[static_cast<std::size_t>(p)]);
      const auto jlo = off[j];
      const auto jhi = off[j + 1];
      const auto *row = nbr.data();
      const auto *found =
          std::lower_bound(row + jlo, row + jhi, static_cast<std::int32_t>(i));
      if (found == row + jhi || *found != static_cast<std::int32_t>(i)) {
        rev_[static_cast<std::size_t>(p)] = -1;
      } else {
        rev_[static_cast<std::size_t>(p)] =
            static_cast<std::int32_t>(found - row);
      }
    }
  }
}

void EAM::ensureParts(long N, int nt) {
  if (part_n_ == N && part_nt_ == nt &&
      static_cast<int>(parts_.size()) == nt) {
    return;
  }
  parts_.resize(static_cast<std::size_t>(nt));
  for (int t = 0; t < nt; ++t) {
    parts_[static_cast<std::size_t>(t)] = rgpot::nlist::partition(N, t, nt);
  }
  part_n_ = N;
  part_nt_ = nt;
}

void EAM::accumulate(long i, const int *atomicNrs, double *F, double &U,
                     std::vector<Delta> &deltas, const std::int32_t *stream_ids,
                     long stream_n) const {
  deltas.clear();
  const element_parameters epar = get_element_parameters(atomicNrs[i]);
  const double cutoff2 = epar.r_cut * epar.r_cut;
  const double half0 = box_len_[0] * 0.5;
  const double half1 = box_len_[1] * 0.5;
  const double half2 = box_len_[2] * 0.5;
  const double xi = wrapped_[static_cast<std::size_t>(3 * i)];
  const double yi = wrapped_[static_cast<std::size_t>(3 * i + 1)];
  const double zi = wrapped_[static_cast<std::size_t>(3 * i + 2)];
  double dens = 0.0;
  double dix = 0.0;
  double diy = 0.0;
  double diz = 0.0;
  const auto &off = neighbours_.offsets();
  const auto &nbr = neighbours_.neighbours();
  const bool stream = neighbours_.streaming();
  const long begin = stream ? 0 : static_cast<long>(off[static_cast<std::size_t>(i)]);
  const long end =
      stream ? stream_n
             : static_cast<long>(off[static_cast<std::size_t>(i) + 1]);
  for (long p = begin; p < end; ++p) {
    const long j =
        stream ? stream_ids[static_cast<std::size_t>(p)]
               : nbr[static_cast<std::size_t>(p)];
    double dx = xi - wrapped_[static_cast<std::size_t>(3 * j)];
    double dy = yi - wrapped_[static_cast<std::size_t>(3 * j + 1)];
    double dz = zi - wrapped_[static_cast<std::size_t>(3 * j + 2)];
    if (dx > half0) {
      dx -= box_len_[0];
    } else if (dx < -half0) {
      dx += box_len_[0];
    }
    if (dy > half1) {
      dy -= box_len_[1];
    } else if (dy < -half1) {
      dy += box_len_[1];
    }
    if (dz > half2) {
      dz -= box_len_[2];
    } else if (dz < -half2) {
      dz += box_len_[2];
    }
    const double r2 = dx * dx + dy * dy + dz * dz;
    if (r2 > cutoff2) {
      continue;
    }
    const double r = std::sqrt(r2);
    const double r6 = r2 * r2 * r2;
    const double exp_b1 = std::exp(-epar.beta1 * r);
    const double exp_b2 = std::exp(-2.0 * epar.beta2 * r);
    const double rho_pair = exp_b1 + 512.0 * exp_b2;
    dens += r6 * rho_pair;
    const double r5 = r2 * r2 * r;
    const double mag_force_den =
        6.0 * r5 * rho_pair +
        r6 * (-epar.beta1 * exp_b1 - 1024.0 * epar.beta2 * exp_b2);
    const double invR = 1.0 / r;
    const double fscale = -mag_force_den * invR;
    dix += fscale * dx;
    diy += fscale * dy;
    diz += fscale * dz;
    deltas.push_back(
        Delta{j, -fscale * dx, -fscale * dy, -fscale * dz});
    if (j > i) {
      const double expArg = std::exp(-epar.alphaM * (r - epar.Rm));
      const double d = 1.0 - expArg;
      const double phi_r = epar.Dm * d * d - epar.Dm;
      // d(phi)/dr = -mag_force, so the force on i is
      // -(d phi/dr) (r_i - r_j) / r = mag_force (r_i - r_j) / r.
      const double mag_force = 2.0 * epar.alphaM * epar.Dm * d * (d - 1.0);
      const double fcomp_scale = mag_force * invR;
      F[3 * i] += fcomp_scale * dx;
      F[3 * i + 1] += fcomp_scale * dy;
      F[3 * i + 2] += fcomp_scale * dz;
      F[3 * j] -= fcomp_scale * dx;
      F[3 * j + 1] -= fcomp_scale * dy;
      F[3 * j + 2] -= fcomp_scale * dz;
      U += phi_r;
    }
  }
  U += embedding_function(epar.func_coeff, dens);
  const double dF = embedding_force(epar.func_coeff, dens);
  std::size_t slot = 0;
  for (; slot < deltas.size() && deltas[slot].atom < i; ++slot) {
    const Delta &d = deltas[slot];
    F[3 * d.atom] -= dF * d.x;
    F[3 * d.atom + 1] -= dF * d.y;
    F[3 * d.atom + 2] -= dF * d.z;
  }
  F[3 * i] -= dF * dix;
  F[3 * i + 1] -= dF * diy;
  F[3 * i + 2] -= dF * diz;
  for (; slot < deltas.size(); ++slot) {
    const Delta &d = deltas[slot];
    F[3 * d.atom] -= dF * d.x;
    F[3 * d.atom + 1] -= dF * d.y;
    F[3 * d.atom + 2] -= dF * d.z;
  }
}

int EAM::threadsRequested() const { return rgpot::nlist::threadsRequested(); }

void EAM::geometry(long i, long j, double &dx, double &dy, double &dz,
                   double &r2) const {
  const double half0 = box_len_[0] * 0.5;
  const double half1 = box_len_[1] * 0.5;
  const double half2 = box_len_[2] * 0.5;
  const double xi = wrapped_[static_cast<std::size_t>(3 * i)];
  const double yi = wrapped_[static_cast<std::size_t>(3 * i + 1)];
  const double zi = wrapped_[static_cast<std::size_t>(3 * i + 2)];
  dx = xi - wrapped_[static_cast<std::size_t>(3 * j)];
  dy = yi - wrapped_[static_cast<std::size_t>(3 * j + 1)];
  dz = zi - wrapped_[static_cast<std::size_t>(3 * j + 2)];
  if (dx > half0) {
    dx -= box_len_[0];
  } else if (dx < -half0) {
    dx += box_len_[0];
  }
  if (dy > half1) {
    dy -= box_len_[1];
  } else if (dy < -half1) {
    dy += box_len_[1];
  }
  if (dz > half2) {
    dz -= box_len_[2];
  } else if (dz < -half2) {
    dz += box_len_[2];
  }
  r2 = dx * dx + dy * dy + dz * dz;
}

void EAM::evaluateStoredParallel(long N, const int *atomicNrs, double *F,
                                  double *U) {
  const int nt = threads_used_;
  rho_.resize(static_cast<std::size_t>(N));
  dF_.resize(static_cast<std::size_t>(N));
  partial_f_.assign(static_cast<std::size_t>(3 * N), 0.0);
  thread_energy_.assign(static_cast<std::size_t>(nt), 0.0);
  ensureParts(N, nt);
  const auto nslot = neighbours_.slots();
  if (fscale_.size() != nslot) {
    fscale_.assign(nslot, 0.0);
    morse_.assign(nslot, 0.0);
    gdx_.assign(nslot, 0.0);
    gdy_.assign(nslot, 0.0);
    gdz_.assign(nslot, 0.0);
  }
  const auto &off = neighbours_.offsets();
  const auto &nbr = neighbours_.neighbours();
  rgpot::nlist::forEachSlice(nt, [&](int t) {
    const long begin = static_cast<long>(parts_[static_cast<std::size_t>(t)].begin);
    const long end = static_cast<long>(parts_[static_cast<std::size_t>(t)].end);
    for (long i = begin; i < end; ++i) {
      const element_parameters epar = get_element_parameters(atomicNrs[i]);
      const double cutoff2 = epar.r_cut * epar.r_cut;
      double dens = 0.0;
      double fx = 0.0;
      double fy = 0.0;
      double fz = 0.0;
      double energy = 0.0;
      const auto lo = off[static_cast<std::size_t>(i)];
      const auto hi = off[static_cast<std::size_t>(i) + 1];
      for (std::int32_t p = lo; p < hi; ++p) {
        const auto s = static_cast<std::size_t>(p);
        const long j = nbr[s];
        double dx = 0.0;
        double dy = 0.0;
        double dz = 0.0;
        double r2 = 0.0;
        geometry(i, j, dx, dy, dz, r2);
        gdx_[s] = dx;
        gdy_[s] = dy;
        gdz_[s] = dz;
        if (!(r2 > 0.0)) {
          fscale_[s] = 0.0;
          morse_[s] = 0.0;
          continue;
        }
        const double r = std::sqrt(r2);
        const double invR = 1.0 / r;
        if (r2 <= cutoff2) {
          const double r6 = r2 * r2 * r2;
          const double exp_b1 = std::exp(-epar.beta1 * r);
          const double exp_b2 = std::exp(-2.0 * epar.beta2 * r);
          const double rho_pair = exp_b1 + 512.0 * exp_b2;
          dens += r6 * rho_pair;
          const double r5 = r2 * r2 * r;
          const double mag =
              6.0 * r5 * rho_pair +
              r6 * (-epar.beta1 * exp_b1 - 1024.0 * epar.beta2 * exp_b2);
          fscale_[s] = -mag * invR;
          if (j > i) {
            const double expArg = std::exp(-epar.alphaM * (r - epar.Rm));
            const double d = 1.0 - expArg;
            energy += epar.Dm * d * d - epar.Dm;
            const double fcomp =
                2.0 * epar.alphaM * epar.Dm * d * (d - 1.0) * invR;
            morse_[s] = fcomp;
            fx += fcomp * dx;
            fy += fcomp * dy;
            fz += fcomp * dz;
          } else {
            morse_[s] = 0.0;
          }
        } else {
          fscale_[s] = 0.0;
          morse_[s] = 0.0;
        }
      }
      const double dFi = embedding_force(epar.func_coeff, dens);
      rho_[static_cast<std::size_t>(i)] = dens;
      dF_[static_cast<std::size_t>(i)] = dFi;
      for (std::int32_t p = lo; p < hi; ++p) {
        const auto s = static_cast<std::size_t>(p);
        const double fscale = fscale_[s];
        if (fscale == 0.0) {
          continue;
        }
        fx -= dFi * fscale * gdx_[s];
        fy -= dFi * fscale * gdy_[s];
        fz -= dFi * fscale * gdz_[s];
      }
      energy += embedding_function(epar.func_coeff, dens);
      partial_f_[static_cast<std::size_t>(3 * i)] = fx;
      partial_f_[static_cast<std::size_t>(3 * i + 1)] = fy;
      partial_f_[static_cast<std::size_t>(3 * i + 2)] = fz;
      thread_energy_[static_cast<std::size_t>(t)] += energy;
    }
  });
  // The second pass reads the dF and pair terms of every slice.
  rgpot::nlist::forEachSlice(nt, [&](int t) {
    const long begin = static_cast<long>(parts_[static_cast<std::size_t>(t)].begin);
    const long end = static_cast<long>(parts_[static_cast<std::size_t>(t)].end);
    for (long i = begin; i < end; ++i) {
      double fx = partial_f_[static_cast<std::size_t>(3 * i)];
      double fy = partial_f_[static_cast<std::size_t>(3 * i + 1)];
      double fz = partial_f_[static_cast<std::size_t>(3 * i + 2)];
      const auto lo = off[static_cast<std::size_t>(i)];
      const auto hi = off[static_cast<std::size_t>(i) + 1];
      for (std::int32_t p = lo; p < hi; ++p) {
        const auto s = static_cast<std::size_t>(p);
        const long j = nbr[s];
        const auto rev_slot = rev_[s];
        const double fnei =
            rev_slot >= 0 ? fscale_[static_cast<std::size_t>(rev_slot)] : 0.0;
        const double fcomp =
            (rev_slot >= 0 && j < i)
                ? morse_[static_cast<std::size_t>(rev_slot)]
                : 0.0;
        if (fnei == 0.0 && fcomp == 0.0) {
          continue;
        }
        // fcomp is (d phi/dr) with a minus already in mag_force. The
        // neighbour's pair force is +fcomp * dx, while the embedding
        // term stays -dF * fnei * dx.
        const double scale = dF_[static_cast<std::size_t>(j)] * fnei - fcomp;
        const double dx = gdx_[s];
        fx -= scale * dx;
        fy -= scale * gdy_[s];
        fz -= scale * gdz_[s];
      }
      partial_f_[static_cast<std::size_t>(3 * i)] = fx;
      partial_f_[static_cast<std::size_t>(3 * i + 1)] = fy;
      partial_f_[static_cast<std::size_t>(3 * i + 2)] = fz;
    }
  });
  for (long i = 0; i < N; ++i) {
    F[3 * i] = partial_f_[static_cast<std::size_t>(3 * i)];
    F[3 * i + 1] = partial_f_[static_cast<std::size_t>(3 * i + 1)];
    F[3 * i + 2] = partial_f_[static_cast<std::size_t>(3 * i + 2)];
  }
  double energy = 0.0;
  for (int t = 0; t < nt; ++t) {
    energy += thread_energy_[static_cast<std::size_t>(t)];
  }
  *U = energy;
}

const rgpot::eam::ElementSpline &EAM::splineFor(int Z) {
  static std::vector<rgpot::eam::ElementSpline> tables;
  if (tables.capacity() < 8) {
    tables.reserve(8);
  }
  for (const auto &table : tables) {
    if (table.Z == Z) {
      return table;
    }
  }
  const element_parameters *raw = nullptr;
  for (int i = 0; i < NPARAMS; ++i) {
    if (el_params[i].Z == Z) {
      raw = &el_params[i];
      break;
    }
  }
  if (raw == nullptr) {
    throw std::invalid_argument("EAM: no parameters for atomic number " +
                                std::to_string(Z));
  }
  tables.push_back(rgpot::eam::makeElementSpline(raw->Z, raw->Dm, raw->alphaM, raw->Rm,
                                                 raw->beta1, raw->beta2, raw->r_cut,
                                                 raw->func_coeff));
  return tables.back();
}

void EAM::evaluateDensityStream(long N, const int *atomicNrs, double *F,
                                 double *U) {
  // One half pair, tabulated. The density walk and the force walk
  // share the cell bins. Each thread keeps its own rows, so the
  // partner atom needs no lock and no pair index.
  js_.clear();
  const int nt = threads_used_;
  std::vector<const rgpot::eam::ElementSpline *> kind(
      static_cast<std::size_t>(N));
  for (long i = 0; i < N; ++i) {
    kind[static_cast<std::size_t>(i)] = &splineFor(atomicNrs[i]);
  }
  const rgpot::eam::ElementSpline *only = kind.empty() ? nullptr : kind[0];
  bool uniform = true;
  for (const auto *k : kind) {
    if (k != only) {
      uniform = false;
      break;
    }
  }
  const auto rows = static_cast<std::size_t>(nt) * static_cast<std::size_t>(N);
  stream_rho_.assign(rows, 0.0);
  std::vector<double> pair_energy(static_cast<std::size_t>(nt), 0.0);
  auto density_row = [&](int t) {
    return stream_rho_.data() + static_cast<std::size_t>(t) * static_cast<std::size_t>(N);
  };
  rho_.assign(static_cast<std::size_t>(N), 0.0);
  dF_.resize(static_cast<std::size_t>(N));
  stream_force_.assign(rows * 3, 0.0);
  double embedded = 0.0;
  auto absorb = [&](int thread) {
    if (thread != 0) {
      return;
    }
    for (long i = 0; i < N; ++i) {
      double dens = 0.0;
      for (int t = 0; t < nt; ++t) {
        dens += stream_rho_[static_cast<std::size_t>(t) * static_cast<std::size_t>(N) +
                            static_cast<std::size_t>(i)];
      }
      rho_[static_cast<std::size_t>(i)] = dens;
      const auto *s = kind[static_cast<std::size_t>(i)];
      dF_[static_cast<std::size_t>(i)] = embedding_force(s->embed, dens);
      embedded += embedding_function(s->embed, dens);
    }
  };
  auto force_row = [&](int t) {
    return stream_force_.data() +
           static_cast<std::size_t>(t) * static_cast<std::size_t>(N) * 3;
  };
  auto add_pair = [&](double *row, std::int32_t i, std::int32_t j, double scale, double dx,
                      double dy, double dz) {
    row[3 * i] += scale * dx;
    row[3 * i + 1] += scale * dy;
    row[3 * i + 2] += scale * dz;
    row[3 * j] -= scale * dx;
    row[3 * j + 1] -= scale * dy;
    row[3 * j + 2] -= scale * dz;
  };
  auto density_of = [&](int t, std::int32_t i, std::int32_t j, double, double, double, double r2) {
        if (!(r2 > 0.0)) {
          return;
        }
        if (uniform && only != nullptr) {
          if (r2 > only->cutoff2) {
            return;
          }
          const double r = std::sqrt(r2);
          const double rho = only->density(r);
          double *row = density_row(t);
          row[i] += rho;
          row[j] += rho;
          pair_energy[static_cast<std::size_t>(t)] += only->pairEnergy(r);
          return;
        }
        const auto *si = kind[static_cast<std::size_t>(i)];
        const auto *sj = kind[static_cast<std::size_t>(j)];
        const double r = std::sqrt(r2);
        double *row = density_row(t);
        // j > i on a half walk, so the pair term belongs to i.
        if (r2 <= si->cutoff2) {
          row[i] += si->density(r);
          pair_energy[static_cast<std::size_t>(t)] += si->pairEnergy(r);
        }
        if (r2 <= sj->cutoff2) {
          row[j] += sj->density(r);
        }
  };
  auto force_of = [&](int t, std::int32_t i, std::int32_t j, double dx, double dy, double dz,
                      double r2) {
        if (!(r2 > 0.0)) {
          return;
        }
        const double r = std::sqrt(r2);
        double scale = 0.0;
        if (uniform && only != nullptr) {
          if (r2 > only->cutoff2) {
            return;
          }
          const double fs = only->densityForce(r);
          scale = only->pairForce(r) -
                  (dF_[static_cast<std::size_t>(i)] + dF_[static_cast<std::size_t>(j)]) * fs;
        } else {
          const auto *si = kind[static_cast<std::size_t>(i)];
          const auto *sj = kind[static_cast<std::size_t>(j)];
          if (r2 <= si->cutoff2) {
            scale += si->pairForce(r);
            scale -= dF_[static_cast<std::size_t>(i)] * si->densityForce(r);
          }
          if (r2 <= sj->cutoff2) {
            scale -= dF_[static_cast<std::size_t>(j)] * sj->densityForce(r);
          }
        }
        if (scale != 0.0) {
          add_pair(force_row(t), i, j, scale, dx, dy, dz);
        }
  };
  neighbours_.forCellTeam(true, nt, density_of);
  absorb(0);
  neighbours_.forCellTeam(true, nt, force_of);
  for (long i = 0; i < N; ++i) {
    double fx = 0.0;
    double fy = 0.0;
    double fz = 0.0;
    for (int t = 0; t < nt; ++t) {
      const double *row = stream_force_.data() +
                          static_cast<std::size_t>(t) * static_cast<std::size_t>(N) * 3;
      fx += row[3 * i];
      fy += row[3 * i + 1];
      fz += row[3 * i + 2];
    }
    F[3 * i] = fx;
    F[3 * i + 1] = fy;
    F[3 * i + 2] = fz;
  }
  double energy = embedded;
  for (double part : pair_energy) {
    energy += part;
  }
  *U = energy;
}

void EAM::evaluate(long N, const int *atomicNrs, double *F, double *U) {
  const int nt = threadsRequested();
  threads_used_ = nt;
  if (neighbours_.streaming()) {
    evaluateDensityStream(N, atomicNrs, F, U);
    return;
  }
  if (nt == 1) {
    double energy = 0.0;
    for (long i = 0; i < N; ++i) {
      accumulate(i, atomicNrs, F, energy, deltas_, nullptr, 0);
    }
    *U = energy;
    return;
  }
  evaluateStoredParallel(N, atomicNrs, F, U);
}
void EAM::force(long N, const double *R, const int *atomicNrs, double *F,
                double *U, double *variance, const double *fullbox) {
  (void)variance;
  const double box[3] = {fullbox[0], fullbox[4], fullbox[8]};
  for (int axis = 0; axis < 3; ++axis) {
    if (!(box[axis] > 0.0)) {
      throw std::invalid_argument(
          "EAM::force: box diagonal and cutoff must be positive");
    }
  }
  *U = 0;
  for (long k = 0; k < 3 * N; k++) {
    F[k] = 0.0;
  }
  if (N < 1) {
    return;
  }
  for (long i = 0; i < N; ++i) {
    (void)get_element_parameters(atomicNrs[i]);
  }
  prepare(N, R, box);
  buildList(N);
  evaluate(N, atomicNrs, F, U);
}

double EAM::embedding_function(const double *func_coeff, double rho) {
  double result = func_coeff[8];
  for (int i = 7; i >= 0; i--) {
    result = result * rho + func_coeff[i];
  }
  return result;
}

double EAM::embedding_force(const double *func_coeff, double rho) {
  double result = func_coeff[8] * 8;
  for (int i = 7; i >= 1; i--) {
    result = result * rho + i * func_coeff[i];
  }
  return -result;
}
