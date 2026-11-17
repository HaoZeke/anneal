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
#pragma once

#include "rgpot/EAM/StreamSpline.hpp"
#include "rgpot/nlist/Neighbours.hpp"

#include <array>
#include <cstdint>
#include <deque>
#include <vector>

#ifndef EAM_STANDALONE
#include "eon/Potential.h"
#endif

/// Fraction of the widest element cutoff added to the neighbour search.
/// A later call reuses the list while every atom has moved by at most half
/// of that distance. One fraction for every geometry.
inline constexpr double kEamSkinOverCutoff = 1.0 / 5.0;

/// Embedded-atom aluminium, palladium and gold.
///
/// The neighbour search is a linked-cell list with a Verlet skin. Below
/// rgpot::kTeamMinAtoms atoms the call runs on one thread and matches the
/// direct pair loop bit for bit: the same pairs in the same atom order,
/// with unchanged arithmetic. From there up the energy of each atom's row
/// is summed on its own and the rows are added in index order, and
/// `OMP_NUM_THREADS` above 1 splits the atoms across threads. A slice sums
/// the force on each of its atoms in the order of the one-thread loop, so
/// every thread count, and every team the runtime starts, gives the same
/// bits. The variable is read on each call.
class EAM
#ifndef EAM_STANDALONE
    : public eonc::Potential
#endif
{
public:
#ifdef EAM_STANDALONE
  EAM() = default;
  ~EAM() = default;
#else
  explicit EAM(const eonc::Parameters &params)
      : eonc::Potential(eonc::PotType::EAM_AL, params) {}

  ~EAM() override = default;
#endif

  void cleanMemory();
  void force(long N, const double *R, const int *atomicNrs, double *F,
             double *U, double *variance, const double *fullbox)
#ifndef EAM_STANDALONE
      override
#endif
      ;

  /// Neighbour-list builds since this object was created.
  [[nodiscard]] std::uint64_t listBuilds() const noexcept {
    return neighbours_.builds();
  }

  /// Stored neighbour indices. Zero while the layer is streaming.
  [[nodiscard]] std::size_t neighborSlots() const noexcept {
    return neighbours_.slots();
  }

  /// True when the last build kept the Verlet list.
  [[nodiscard]] bool listReused() const noexcept { return neighbours_.reused(); }

  /// Partner indices retained between the density pass and the force pass.
  /// A density-only stream keeps none.
  [[nodiscard]] std::size_t streamPartners() const noexcept { return js_.size(); }

  /// Ask for a stored list or a streaming walk. Auto stores when the
  /// candidate index array fits in the memory reported as available.
  void setNeighbourMode(rgpot::nlist::Mode mode) noexcept { mode_ = mode; }

  /// Threads used by the most recent force call. 1 when OpenMP is off,
  /// `OMP_NUM_THREADS` is unset, or the call had fewer than
  /// rgpot::kTeamMinAtoms atoms.
  [[nodiscard]] int threadsUsed() const noexcept { return threads_used_; }

#ifndef EAM_STANDALONE
  /// The list and the scratch live on the instance.
  [[nodiscard]] bool isSharedInstanceThreadSafe() const noexcept override {
    return false;
  }
#endif

private:
  struct element_parameters {
    const int Z;
    const double Dm;
    const double alphaM;
    const double Rm;
    const double beta1;
    const double beta2;
    const double r_cut;
    const double func_coeff[9];
  };
  static const element_parameters el_params[];

  /// One neighbour's density derivative, in atom-index order.
  struct Delta {
    long atom;
    double x;
    double y;
    double z;
  };

  std::vector<double> wrapped_;
  std::vector<Delta> deltas_;
  std::vector<std::int32_t> js_;
  std::vector<double> rho_;
  std::vector<double> dF_;
  std::vector<double> fscale_;
  std::vector<double> morse_;
  std::vector<double> gdx_;
  std::vector<double> gdy_;
  std::vector<double> gdz_;
  std::vector<double> ginv_;
  std::vector<std::int32_t> rev_;
  /// Energy of each atom's row, and the sum of fscale * d over its row.
  std::vector<double> row_energy_;
  std::vector<double> row_pull_;
  /// Density and force rows for the stream, one per slice, so a half
  /// pair can update both atoms without a lock.
  std::vector<double> stream_rho_;
  std::vector<double> stream_force_;
  std::vector<rgpot::nlist::Range> parts_;
  /// Tables for the elements seen so far. A deque keeps every table in
  /// place as it grows, so the per-atom pointers into it stay valid.
  std::deque<rgpot::eam::ElementSpline> splines_;
  rgpot::nlist::Neighbours neighbours_;

  double box_len_[3]{};
  long n_atoms_ = 0;
  long part_n_ = -1;
  int part_nt_ = -1;
  int threads_used_ = 1;
  std::uint64_t rev_stamp_ = 0;
  rgpot::nlist::Mode mode_ = rgpot::nlist::Mode::Auto;

  void prepare(long N, const double *R, const double *box);
  void buildList(long N);
  void ensureRev();
  void ensureParts(long N, int nt);
  void evaluate(long N, const int *atomicNrs, double *F, double *U);
  void evaluateStoredParallel(long N, const int *atomicNrs, double *F,
                              double *U);
  void evaluateDensityStream(long N, const int *atomicNrs, double *F,
                             double *U);
  void evaluateStreamRows(long N, const int *atomicNrs, double *F, double *U);
  [[nodiscard]] const rgpot::eam::ElementSpline &splineFor(int Z);
  void accumulate(long i, const int *atomicNrs, double *F, double &U,
                  std::vector<Delta> &deltas, const std::int32_t *stream_ids,
                  long stream_n) const;
  void geometry(long i, long j, double &dx, double &dy, double &dz,
                double &r2) const;

  [[nodiscard]] element_parameters
  get_element_parameters(int atomic_number) const;
  [[nodiscard]] static double embedding_function(const double *func_coeff,
                                                 double rho);
  [[nodiscard]] static double embedding_force(const double *func_coeff,
                                              double rho);
};
