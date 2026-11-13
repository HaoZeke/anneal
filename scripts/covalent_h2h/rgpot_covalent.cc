// C entry points for two rgpot many-body kernels on an isolated cluster.
//
// rgpot exports Stillinger-Weber and Tersoff as C++ classes with a periodic
// cell. A cluster has no cell, so each connected part of the cluster is
// evaluated in a cube that holds it plus more than one cutoff of vacuum on
// each side. No image is then inside the cutoff of any atom. Coordinates
// are Angstrom, energies eV, and the gradient is dE/dx.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "rgpot/StillingerWeber/SWSi.hpp"
#include "rgpot/Tersoff/TersoffSiC.hpp"

namespace {

// Larger than both cutoffs: SW is 1.8 sigma = 3.77 A, Tersoff Si-Si 3.0 A.
constexpr double kVacuum = 8.0;

struct Cell {
  std::vector<std::size_t> atoms;
  std::vector<double> shifted;
  double box[9];
};

// Atoms closer than kVacuum share a part. Parts do not interact, so each
// is evaluated in its own cube and a fragment far from the rest does not
// grow the cell.
std::vector<Cell> isolate(const double *positions, std::size_t n) {
  std::vector<std::size_t> part(n);
  for (std::size_t atom = 0; atom < n; ++atom) {
    part[atom] = atom;
  }
  auto root = [&part](std::size_t atom) {
    while (part[atom] != atom) {
      atom = part[atom] = part[part[atom]];
    }
    return atom;
  };
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = i + 1; j < n; ++j) {
      double r2 = 0.0;
      for (int axis = 0; axis < 3; ++axis) {
        const double d = positions[3 * i + axis] - positions[3 * j + axis];
        r2 += d * d;
      }
      if (r2 < kVacuum * kVacuum) {
        part[root(i)] = root(j);
      }
    }
  }
  std::vector<Cell> cells;
  std::vector<std::size_t> slot(n, n);
  for (std::size_t atom = 0; atom < n; ++atom) {
    const std::size_t top = root(atom);
    if (slot[top] == n) {
      slot[top] = cells.size();
      cells.emplace_back();
    }
    cells[slot[top]].atoms.push_back(atom);
  }
  for (Cell &cell : cells) {
    double lo[3], hi[3];
    for (int axis = 0; axis < 3; ++axis) {
      lo[axis] = std::numeric_limits<double>::infinity();
      hi[axis] = -std::numeric_limits<double>::infinity();
    }
    for (std::size_t atom : cell.atoms) {
      for (int axis = 0; axis < 3; ++axis) {
        lo[axis] = std::min(lo[axis], positions[3 * atom + axis]);
        hi[axis] = std::max(hi[axis], positions[3 * atom + axis]);
      }
    }
    double side = 0.0;
    for (int axis = 0; axis < 3; ++axis) {
      side = std::max(side, hi[axis] - lo[axis]);
    }
    side += 2.0 * kVacuum;
    for (std::size_t atom : cell.atoms) {
      for (int axis = 0; axis < 3; ++axis) {
        cell.shifted.push_back(positions[3 * atom + axis] + kVacuum - lo[axis]);
      }
    }
    std::fill(cell.box, cell.box + 9, 0.0);
    cell.box[0] = cell.box[4] = cell.box[8] = side;
  }
  return cells;
}

template <typename Kernel>
void evaluate(Kernel &kernel, int z, const std::vector<Cell> &cells, double *energy,
              double *gradient, std::size_t n) {
  *energy = 0.0;
  std::fill(gradient, gradient + 3 * n, 0.0);
  std::vector<double> part;
  for (const Cell &cell : cells) {
    const std::size_t m = cell.atoms.size();
    if (m < 2) {
      continue;
    }
    const std::vector<int> numbers(m, z);
    part.assign(3 * m, 0.0);
    double e = 0.0;
    kernel.energyAndGradient(cell.shifted.data(), numbers.data(), cell.box, m, &e, part.data());
    *energy += e;
    for (std::size_t k = 0; k < m; ++k) {
      for (int axis = 0; axis < 3; ++axis) {
        gradient[3 * cell.atoms[k] + axis] = part[3 * k + axis];
      }
    }
  }
}

} // namespace

extern "C" {

/// kind 0 is Stillinger-Weber silicon, 1985 parameters. kind 1 is Tersoff
/// carbon, the 1989 parameters with the 1990 erratum. Returns 0 on success,
/// 1 for an unknown kind and 2 when rgpot throws or a value is not finite.
int rgpot_covalent_energy_gradient(int kind, long n_atoms, const double *positions,
                                   double *energy, double *gradient) {
  try {
    const auto n = static_cast<std::size_t>(n_atoms);
    for (std::size_t k = 0; k < 3 * n; ++k) {
      if (!std::isfinite(positions[k])) {
        return 2;
      }
    }
    const std::vector<Cell> cells = isolate(positions, n);
    if (kind == 0) {
      thread_local rgpot::StillingerWeberSi sw;
      evaluate(sw, 14, cells, energy, gradient, n);
    } else if (kind == 1) {
      thread_local rgpot::TersoffSiC tersoff;
      evaluate(tersoff, 6, cells, energy, gradient, n);
    } else {
      return 1;
    }
    return std::isfinite(*energy) ? 0 : 2;
  } catch (...) {
    return 2;
  }
}

/// The Stillinger-Weber length and energy units, Angstrom and eV.
void rgpot_covalent_sw_units(double *sigma, double *epsilon) {
  *sigma = rgpot::StillingerWeberSi::kSigma;
  *epsilon = rgpot::StillingerWeberSi::kEpsilon;
}

} // extern "C"
