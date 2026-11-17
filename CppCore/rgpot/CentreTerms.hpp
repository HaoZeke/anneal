#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief The terms of a three-body potential, centre by centre, with the
 * same bits on any number of threads.
 *
 * A centre i with bonds to partners j_0 < j_1 < ... adds pair terms on
 * (i, j_a) and triplet terms on (i, j_a, j_b). Below kTeamMinAtoms each
 * term goes straight into the totals through CentreDirect. From
 * kTeamMinAtoms up, sumCentres has each centre sum its terms into one force
 * on itself, one on each partner, an energy and a strain (CentreTerms), and
 * adds the centres to the totals in index order: as they finish on one
 * thread, and from one ledger per slice on a team.
 */

#include "rgpot/Slices.hpp"
#include "rgpot/stress.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace rgpot {

namespace detail {

/// One centre into the totals: its energy, its strain, then the force of
/// each row, the centre's own first.
inline void addCentre(double *force, double &energy, double *strain, double e, const double *s,
                      std::size_t rows, const std::int32_t *atoms, const double *f) {
  energy += e;
  if (strain != nullptr) {
    for (std::size_t c = 0; c < 6; ++c) {
      strain[c] += s[c];
    }
  }
  for (std::size_t r = 0; r < rows; ++r) {
    double *to = force + 3 * static_cast<std::size_t>(atoms[r]);
    to[0] += f[3 * r];
    to[1] += f[3 * r + 1];
    to[2] += f[3 * r + 2];
  }
}

} // namespace detail

/// The terms of centre `i`, added to the totals as they come. `strain` may
/// be null.
class CentreDirect {
public:
  CentreDirect(std::int32_t i, double *force, double *energy, double *strain)
      : i_{static_cast<std::size_t>(i)}, force_{force}, energy_{energy}, strain_{strain} {}

  [[nodiscard]] bool strained() const noexcept { return strain_ != nullptr; }
  void energy(double e) { *energy_ += e; }
  /// Force s on the centre and -s on partner `j` of bond `a`.
  void pair(std::size_t, std::int32_t j, double sx, double sy, double sz) {
    double *fi = force_ + 3 * i_;
    double *fj = force_ + 3 * static_cast<std::size_t>(j);
    fi[0] += sx;
    fi[1] += sy;
    fi[2] += sz;
    fj[0] -= sx;
    fj[1] -= sy;
    fj[2] -= sz;
  }
  /// Force -scale * g1 on `j`, -scale * g2 on `k` and their opposite on
  /// the centre, for bonds `a` and `b`.
  void triplet(std::size_t, std::int32_t j, std::size_t, std::int32_t k, double scale, double g1x,
               double g1y, double g1z, double g2x, double g2y, double g2z) {
    double *fj = force_ + 3 * static_cast<std::size_t>(j);
    double *fk = force_ + 3 * static_cast<std::size_t>(k);
    double *fi = force_ + 3 * i_;
    fj[0] -= scale * g1x;
    fj[1] -= scale * g1y;
    fj[2] -= scale * g1z;
    fk[0] -= scale * g2x;
    fk[1] -= scale * g2y;
    fk[2] -= scale * g2z;
    fi[0] += scale * (g1x + g2x);
    fi[1] += scale * (g1y + g2y);
    fi[2] += scale * (g1z + g2z);
  }
  void pairStrain(double fscale, double dx, double dy, double dz) {
    accumulatePairStrain(strain_, fscale, dx, dy, dz);
  }
  void bondStrain(double dx, double dy, double dz, double gx, double gy, double gz) {
    accumulateBondStrain(strain_, dx, dy, dz, gx, gy, gz);
  }

private:
  std::size_t i_;
  double *force_;
  double *energy_;
  double *strain_;
};

class CentreLedger;

/// The terms of one centre, summed per atom. The same calls as CentreDirect.
class CentreTerms {
public:
  explicit CentreTerms(bool strained) : strained_{strained} {}

  /// Starts centre `i` with one row per bond, in the order of `bonds`,
  /// whose entries have the partner in `.j`.
  template <typename Bonds>
  void reset(std::int32_t i, const Bonds &bonds) {
    atoms_.clear();
    atoms_.push_back(i);
    for (const auto &b : bonds) {
      atoms_.push_back(static_cast<std::int32_t>(b.j));
    }
    force_.assign(3 * atoms_.size(), 0.0);
    energy_ = 0.0;
    std::fill(strain_, strain_ + 6, 0.0);
  }

  [[nodiscard]] bool strained() const noexcept { return strained_; }
  void energy(double e) { energy_ += e; }
  void pair(std::size_t a, std::int32_t, double sx, double sy, double sz) {
    double *fi = force_.data();
    double *fj = force_.data() + 3 * (a + 1);
    fi[0] += sx;
    fi[1] += sy;
    fi[2] += sz;
    fj[0] -= sx;
    fj[1] -= sy;
    fj[2] -= sz;
  }
  void triplet(std::size_t a, std::int32_t, std::size_t b, std::int32_t, double scale, double g1x,
               double g1y, double g1z, double g2x, double g2y, double g2z) {
    double *fj = force_.data() + 3 * (a + 1);
    double *fk = force_.data() + 3 * (b + 1);
    double *fi = force_.data();
    fj[0] -= scale * g1x;
    fj[1] -= scale * g1y;
    fj[2] -= scale * g1z;
    fk[0] -= scale * g2x;
    fk[1] -= scale * g2y;
    fk[2] -= scale * g2z;
    fi[0] += scale * (g1x + g2x);
    fi[1] += scale * (g1y + g2y);
    fi[2] += scale * (g1z + g2z);
  }
  void pairStrain(double fscale, double dx, double dy, double dz) {
    accumulatePairStrain(strain_, fscale, dx, dy, dz);
  }
  void bondStrain(double dx, double dy, double dz, double gx, double gy, double gz) {
    accumulateBondStrain(strain_, dx, dy, dz, gx, gy, gz);
  }

  void addTo(double *force, double &energy, double *strain) const {
    detail::addCentre(force, energy, strain, energy_, strain_, atoms_.size(), atoms_.data(),
                      force_.data());
  }

private:
  friend class CentreLedger;
  bool strained_;
  std::vector<std::int32_t> atoms_;
  std::vector<double> force_;
  double energy_ = 0.0;
  double strain_[6] = {};
};

/// Finished centres of one slice, in the order they finished.
class CentreLedger {
public:
  void clear() {
    rows_.clear();
    atoms_.clear();
    force_.clear();
    energy_.clear();
    strain_.clear();
  }

  void keep(const CentreTerms &t) {
    rows_.push_back(static_cast<std::int32_t>(t.atoms_.size()));
    atoms_.insert(atoms_.end(), t.atoms_.begin(), t.atoms_.end());
    force_.insert(force_.end(), t.force_.begin(), t.force_.end());
    energy_.push_back(t.energy_);
    if (t.strained_) {
      strain_.insert(strain_.end(), t.strain_, t.strain_ + 6);
    }
  }

  /// The centres into the totals, each as CentreTerms::addTo would add it.
  void addTo(double *force, double &energy, double *strain) const {
    std::size_t row = 0;
    for (std::size_t c = 0; c < rows_.size(); ++c) {
      const auto rows = static_cast<std::size_t>(rows_[c]);
      detail::addCentre(force, energy, strain, energy_[c],
                        strain != nullptr ? strain_.data() + 6 * c : nullptr, rows,
                        atoms_.data() + row, force_.data() + 3 * row);
      row += rows;
    }
  }

private:
  std::vector<std::int32_t> rows_;
  std::vector<std::int32_t> atoms_;
  std::vector<double> force_;
  std::vector<double> energy_;
  std::vector<double> strain_;
};

/**
 * Runs `centre(i, terms, scratch)` for every atom i below `n` and adds the
 * centres to `force`, `energy` and, when it is set, `strain`, in index
 * order. `centre` resets `terms` for i and then adds i's terms to it.
 * Each slice has its own `Scratch`. A team runs the centres in blocks, so
 * the ledgers stay a block long.
 */
template <typename Scratch, typename Centre>
void sumCentres(std::size_t n, const Centre &centre, double *force, double &energy,
                double *strain) {
  const bool strained = strain != nullptr;
  const int slices = teamSize(n);
  if (slices == 1) {
    CentreTerms terms(strained);
    Scratch scratch{};
    for (std::size_t i = 0; i < n; ++i) {
      centre(static_cast<std::int32_t>(i), terms, scratch);
      terms.addTo(force, energy, strain);
    }
    return;
  }
  constexpr std::size_t kBlock = std::size_t{1} << 16;
  std::vector<CentreLedger> ledgers(static_cast<std::size_t>(slices));
  for (std::size_t base = 0; base < n; base += kBlock) {
    const auto len = static_cast<std::int64_t>(std::min(kBlock, n - base));
    forEachSlice(slices, [&](int s) {
      CentreLedger &ledger = ledgers[static_cast<std::size_t>(s)];
      ledger.clear();
      CentreTerms terms(strained);
      Scratch scratch{};
      const Range r = partition(len, s, slices);
      for (std::int64_t k = r.begin; k < r.end; ++k) {
        centre(static_cast<std::int32_t>(base + static_cast<std::size_t>(k)), terms, scratch);
        ledger.keep(terms);
      }
    });
    for (const CentreLedger &ledger : ledgers) {
      ledger.addTo(force, energy, strain);
    }
  }
}

} // namespace rgpot
