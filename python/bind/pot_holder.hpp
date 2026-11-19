// MIT License
// Copyright 2023--present rgpot developers
//
// Per-call evaluation helpers for the free-threaded module (F2, F4, F6).

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include "rgbind/rgbind.hpp"

#include "rgpot/BatchEvaluator.hpp"
#include "rgpot/ForceStructs.hpp"
#include "rgpot/HessianVector.hpp"
#include "rgpot/PerAtom.hpp"
#include "rgpot/Potential.hpp"
#include "rgpot/pot_caps.hpp"
#include "rgpot/types/AtomMatrix.hpp"

namespace nb = nanobind;

namespace rgpot::pybind {

using F64 = rgbind::F64Array;
using I32 = nb::ndarray<const int32_t, nb::c_contig, nb::device::cpu>;
using I64 = nb::ndarray<const int64_t, nb::c_contig, nb::device::cpu>;
using AtomMatrix = types::AtomMatrix;

/// One geometry copied out of NumPy before the caller detaches (F4, F6).
struct Geom {
  AtomMatrix pos;
  std::vector<int> types;
  std::array<std::array<double, 3>, 3> box{};
};

inline std::string shape_of(const F64 &arr) {
  std::string text = "(";
  for (size_t i = 0; i < arr.ndim(); ++i) {
    if (i != 0) {
      text += ", ";
    }
    text += std::to_string(arr.shape(i));
  }
  if (arr.ndim() == 1) {
    text += ",";
  }
  text += ")";
  return text;
}

/// F6: positions are (N, 3) float64, C contiguous. N is not fixed.
inline void check_positions(const F64 &arr) {
  rgbind::check_f64_rank(arr, 2, "positions");
  if (arr.shape(1) != 3) {
    const std::string msg = "positions: expected shape (N, 3), got " +
                            shape_of(arr);
    throw nb::value_error(msg.c_str());
  }
}

inline void append_atom_type(std::vector<int> &types, long long value) {
  if (value < static_cast<long long>(std::numeric_limits<int>::min()) ||
      value > static_cast<long long>(std::numeric_limits<int>::max())) {
    throw nb::value_error(
        "atom_types: value does not fit in the internal integer dtype");
  }
  types.push_back(static_cast<int>(value));
}

inline std::string atom_types_shape_message(size_t n, size_t got) {
  return "atom_types: expected shape (" + std::to_string(n) + ",), got (" +
         std::to_string(got) + ",)";
}

/// int32 is the internal dtype. int64 arrays and plain integer lists are
/// copied in after a range check. noconvert still rejects other dtypes.
inline std::vector<int> read_types(nb::handle atom_types, size_t n) {
  std::vector<int> types;
  types.reserve(n);
  if (nb::isinstance<nb::list>(atom_types) ||
      nb::isinstance<nb::tuple>(atom_types)) {
    const size_t got = static_cast<size_t>(nb::len(atom_types));
    if (got != n) {
      throw nb::value_error(atom_types_shape_message(n, got).c_str());
    }
    for (nb::handle item : atom_types) {
      if (nb::isinstance<nb::bool_>(item) || !nb::isinstance<nb::int_>(item)) {
        throw nb::type_error("atom_types: expected a sequence of integers");
      }
      append_atom_type(types, nb::cast<long long>(item));
    }
    return types;
  }

  try {
    const I32 arr = nb::cast<I32>(atom_types, false);
    if (arr.ndim() != 1 || static_cast<size_t>(arr.shape(0)) != n) {
      const size_t got =
          arr.ndim() == 1 ? static_cast<size_t>(arr.shape(0)) : 0;
      throw nb::value_error(atom_types_shape_message(n, got).c_str());
    }
    for (size_t i = 0; i < n; ++i) {
      append_atom_type(types, static_cast<long long>(arr.data()[i]));
    }
    return types;
  } catch (const nb::cast_error &) {
    PyErr_Clear();
  }

  try {
    const I64 arr = nb::cast<I64>(atom_types, false);
    if (arr.ndim() != 1 || static_cast<size_t>(arr.shape(0)) != n) {
      const size_t got =
          arr.ndim() == 1 ? static_cast<size_t>(arr.shape(0)) : 0;
      throw nb::value_error(atom_types_shape_message(n, got).c_str());
    }
    for (size_t i = 0; i < n; ++i) {
      append_atom_type(types, static_cast<long long>(arr.data()[i]));
    }
    return types;
  } catch (const nb::cast_error &) {
    PyErr_Clear();
  }

  throw nb::type_error(
      "atom_types: expected a C-contiguous int32 or int64 array, or a list "
      "of integers");
}

inline Geom read_geom(const F64 &positions, nb::handle atom_types,
                      const F64 &box) {
  check_positions(positions);
  const size_t n = static_cast<size_t>(positions.shape(0));
  rgbind::check_f64_2d(box, 3, 3, "box");
  Geom g;
  g.pos = AtomMatrix(n, 3);
  std::memcpy(g.pos.data(), positions.data(), n * 3 * sizeof(double));
  g.types = read_types(atom_types, n);
  const double *p = box.data();
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      g.box[i][j] = p[i * 3 + j];
    }
  }
  return g;
}

inline nb::ndarray<nb::numpy, double, nb::c_contig>
matrix_to_numpy(const AtomMatrix &mat) {
  const size_t rows = mat.rows();
  const size_t cols = mat.cols();
  const size_t n = rows * cols;
  double *buf = new double[n == 0 ? 1 : n];
  if (n != 0) {
    std::memcpy(buf, mat.data(), n * sizeof(double));
  }
  nb::capsule owner(buf, [](void *p) noexcept { delete[] static_cast<double *>(p); });
  return nb::ndarray<nb::numpy, double, nb::c_contig>(buf, {rows, cols}, owner);
}

inline nb::ndarray<nb::numpy, double, nb::c_contig>
stress_to_numpy(const double stress[9]) {
  double *buf = new double[9];
  std::memcpy(buf, stress, 9 * sizeof(double));
  nb::capsule owner(buf, [](void *p) noexcept { delete[] static_cast<double *>(p); });
  return nb::ndarray<nb::numpy, double, nb::c_contig>(buf, {3, 3}, owner);
}

inline nb::ndarray<nb::numpy, double, nb::c_contig>
vector_to_numpy(const std::vector<double> &values,
                std::initializer_list<size_t> shape) {
  const size_t n = values.size();
  double *buf = new double[n == 0 ? 1 : n];
  if (n != 0) {
    std::memcpy(buf, values.data(), n * sizeof(double));
  }
  nb::capsule owner(buf, [](void *p) noexcept { delete[] static_cast<double *>(p); });
  return nb::ndarray<nb::numpy, double, nb::c_contig>(buf, shape, owner);
}

inline nb::dict caps_dict(const PotCaps &caps) {
  nb::dict out;
  const char *reentrancy = "shared";
  if (caps.reentrancy == Reentrancy::PerInstance) {
    reentrancy = "per_instance";
  } else if (caps.reentrancy == Reentrancy::ProcessSerial) {
    reentrancy = "process_serial";
  }
  out["reentrancy"] = reentrancy;
  out["per_image"] = caps.perImageInstances;
  out["batched"] = caps.batched;
  out["periodic"] = caps.periodic;
  out["group_collective"] = caps.groupCollective;
  out["stress"] = caps.stress;
  // PotCaps does not report per-atom energies. Do not invent the flag.
  return out;
}

struct FullResult {
  double energy = 0.0;
  AtomMatrix forces;
  double variance = 0.0;
  bool has_stress = false;
  double stress[9]{};
};

/// The cell of @a g, row-major.
inline std::array<double, 9> flat_box(const Geom &g) {
  std::array<double, 9> flat{};
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      flat[i * 3 + j] = g.box[i][j];
    }
  }
  return flat;
}

/// forceImpl while the caller is already detached (F4).
template <typename Pot>
FullResult run_force(Pot &pot, const Geom &g) {
  AtomMatrix forces = AtomMatrix::Zero(g.pos.rows(), 3);
  const std::array<double, 9> flat = flat_box(g);
  ForceOut fo{.F = forces.data(),
              .energy = 0.0,
              .variance = 0.0,
              .stress = {},
              .has_stress = 0};
  ForceInput in{.nAtoms = g.pos.rows(),
                .pos = g.pos.data(),
                .atmnrs = g.types.data(),
                .box = flat.data()};
  pot.forceImpl(in, &fo);
  if (!std::isfinite(fo.energy)) {
    throw rgbind::NumericalError("energy is not finite");
  }
  FullResult result;
  result.energy = fo.energy;
  result.forces = std::move(forces);
  result.variance = fo.variance;
  result.has_stress = fo.has_stress != 0;
  std::memcpy(result.stress, fo.stress, sizeof(result.stress));
  return result;
}

struct PerAtomResult {
  FullResult total;
  std::vector<double> energies;
  std::vector<double> virials;
};

/// forcePerAtom while the caller is already detached (F4).
template <PerAtomPotential Pot>
PerAtomResult run_force_per_atom(Pot &pot, const Geom &g) {
  const auto n = static_cast<size_t>(g.pos.rows());
  PerAtomResult result;
  result.energies.assign(n, 0.0);
  result.virials.assign(6 * n, 0.0);
  AtomMatrix forces = AtomMatrix::Zero(g.pos.rows(), 3);
  const std::array<double, 9> flat = flat_box(g);
  ForceOut fo{.F = forces.data(),
              .energy = 0.0,
              .variance = 0.0,
              .stress = {},
              .has_stress = 0};
  ForceInput in{.nAtoms = g.pos.rows(),
                .pos = g.pos.data(),
                .atmnrs = g.types.data(),
                .box = flat.data()};
  pot.forcePerAtom(in, &fo,
                   PerAtomOut{result.energies.data(), result.virials.data()});
  if (!std::isfinite(fo.energy)) {
    throw rgbind::NumericalError("energy is not finite");
  }
  result.total.energy = fo.energy;
  result.total.forces = std::move(forces);
  result.total.variance = fo.variance;
  result.total.has_stress = fo.has_stress != 0;
  std::memcpy(result.total.stress, fo.stress, sizeof(result.total.stress));
  return result;
}

inline nb::dict result_dict(const FullResult &result) {
  nb::dict out;
  out["energy"] = result.energy;
  out["forces"] = matrix_to_numpy(result.forces);
  out["variance"] = result.variance;
  if (result.has_stress) {
    out["stress"] = stress_to_numpy(result.stress);
  } else {
    out["stress"] = nb::none();
  }
  out["energies"] = nb::none();
  return out;
}

/**
 * One potential object, reentrant (F2, F3).
 *
 * A kernel whose caps() report Reentrancy::SharedInstance keeps a single
 * instance: concurrent calls are the same calls the GIL used to serialise.
 * Every other kernel gets one instance per calling thread, created on
 * first use under mu_ while the caller is detached, and destroyed with
 * this object.
 *
 * The Calculator path does not use this as a session claim. Only the
 * RPC client claims its connection.
 *
 * extra_ keeps one instance per caller thread id other than the
 * constructing thread. A thread that has exited does not remove its
 * entry, so the map grows with the number of distinct threads that have
 * called this object and is released in clear() when the object is
 * destroyed. BatchEvaluator does not use the map. It builds a fixed
 * set of instances, one per worker.
 */
template <typename Pot> class PotHolder {
public:
  using Factory = std::function<std::unique_ptr<Pot>()>;

  explicit PotHolder(Factory factory)
      : make_(std::move(factory)), primary_(make_()),
        owner_(std::this_thread::get_id()) {
    if (!primary_) {
      throw std::invalid_argument("potential factory returned null");
    }
    shared_ = primary_->caps().reentrancy == Reentrancy::SharedInstance;
  }

  PotHolder(const PotHolder &) = delete;
  PotHolder &operator=(const PotHolder &) = delete;

  ~PotHolder() noexcept {
    try {
      if (Py_IsInitialized() != 0) {
        nb::gil_scoped_release release;
        clear();
      } else {
        clear();
      }
    } catch (...) {
      try {
        clear();
      } catch (...) {
      }
    }
  }

  /// Caller is detached. The mutex covers lookup and creation only.
  Pot &slot() {
    if (shared_ || std::this_thread::get_id() == owner_) {
      return *primary_;
    }
    std::lock_guard<std::mutex> lock(mu_);
    auto &extra = extra_[std::this_thread::get_id()];
    if (!extra) {
      extra = make_();
      if (!extra) {
        throw std::invalid_argument("potential factory returned null");
      }
    }
    return *extra;
  }

  Pot &primary() { return *primary_; }
  const Pot &primary() const { return *primary_; }

  std::unique_ptr<Pot> make() const { return make_(); }

  /// Centre and counters of hvp(). Every instance comes from one factory,
  /// so a centre computed on one thread's instance serves the others.
  HessianVectorCache &hvp() { return hvp_; }

private:
  void clear() noexcept {
    std::lock_guard<std::mutex> lock(mu_);
    extra_.clear();
    primary_.reset();
  }

  Factory make_;
  std::unique_ptr<Pot> primary_;
  std::thread::id owner_;
  bool shared_ = false;
  std::mutex mu_;
  std::unordered_map<std::thread::id, std::unique_ptr<Pot>> extra_;
  HessianVectorCache hvp_;
};

inline HvpScheme parse_hvp_scheme(const std::string &name) {
  if (name == "auto") {
    return HvpScheme::Auto;
  }
  if (name == "forward") {
    return HvpScheme::Forward;
  }
  if (name == "central") {
    return HvpScheme::Central;
  }
  if (name == "analytic") {
    return HvpScheme::Analytic;
  }
  const std::string msg = "scheme: expected 'auto', 'forward', 'central' or "
                          "'analytic', got '" + name + "'";
  throw nb::value_error(msg.c_str());
}

/// Run a potential call while the caller is detached (F4). A domain_error
/// from the kernel or HessianVectorCache is a NumericalError, as for an
/// energy that is not finite.
template <typename Fn> auto run_detached(Fn &&fn) {
  nb::gil_scoped_release release;
  try {
    return fn();
  } catch (const std::domain_error &e) {
    throw rgbind::NumericalError(e.what());
  }
}

/// (energy, forces, variance) of one call on a bare potential, made
/// detached (F4). A non-finite energy is a NumericalError, as for the
/// PotHolder __call__.
template <typename Pot> nb::tuple call_detached(Pot &pot, const Geom &geom) {
  auto [energy, forces, variance] =
      run_detached([&] { return pot(geom.pos, geom.types, geom.box); });
  if (!std::isfinite(energy)) {
    throw rgbind::NumericalError("energy is not finite");
  }
  return nb::make_tuple(energy, matrix_to_numpy(forces), variance);
}

template <typename Pot>
void bind_calls(nb::class_<PotHolder<Pot>> &cls) {
  cls.def("caps", [](const PotHolder<Pot> &self) {
    return caps_dict(self.primary().caps());
  });
  cls.def(
      "__call__",
      [](PotHolder<Pot> &self, const F64 &positions, nb::handle atom_types,
         const F64 &box) {
        Geom geom = read_geom(positions, atom_types, box);
        auto [energy, forces, variance] = run_detached([&] {
          Pot &pot = self.slot();
          return pot(geom.pos, geom.types, geom.box);
        });
        if (!std::isfinite(energy)) {
          throw rgbind::NumericalError("energy is not finite");
        }
        return nb::make_tuple(energy, matrix_to_numpy(forces), variance);
      },
      nb::arg("positions").noconvert(), nb::arg("atom_types"),
      nb::arg("box").noconvert());
  cls.def(
      "evaluate",
      [](PotHolder<Pot> &self, const F64 &positions, nb::handle atom_types,
         const F64 &box) {
        Geom geom = read_geom(positions, atom_types, box);
        FullResult result =
            run_detached([&] { return run_force(self.slot(), geom); });
        return result_dict(result);
      },
      nb::arg("positions").noconvert(), nb::arg("atom_types"),
      nb::arg("box").noconvert(),
      "Energy, forces, variance, and stress when the kernel writes it. "
      "The kernel runs with the interpreter released.");
  if constexpr (PerAtomPotential<Pot>) {
    cls.def(
        "evaluate_per_atom",
        [](PotHolder<Pot> &self, const F64 &positions, nb::handle atom_types,
           const F64 &box) {
          Geom geom = read_geom(positions, atom_types, box);
          PerAtomResult result = run_detached(
              [&] { return run_force_per_atom(self.slot(), geom); });
          const auto n = static_cast<size_t>(geom.pos.rows());
          nb::dict out = result_dict(result.total);
          out["energies"] = vector_to_numpy(result.energies, {n});
          out["virials"] = vector_to_numpy(result.virials, {n, 6});
          return out;
        },
        nb::arg("positions").noconvert(), nb::arg("atom_types"),
        nb::arg("box").noconvert(),
        "evaluate() plus energies, shape (N,), eV, summing to energy, and "
        "virials, shape (N, 6), eV, Voigt xx yy zz yz xz xy, summing to "
        "-volume * stress. A pair term goes half to each atom. The kernel "
        "runs with the interpreter released.");
  }
  cls.def(
      "evaluate_batch",
      [](PotHolder<Pot> &self, nb::list positions, nb::list atom_types,
         nb::list boxes, unsigned n_threads) {
        if (positions.size() != atom_types.size() ||
            positions.size() != boxes.size()) {
          throw nb::value_error(
              "evaluate_batch: positions, atom_types, and box lengths differ");
        }
        if (positions.size() == 0) {
          return nb::list();
        }
        std::vector<BatchInput> inputs;
        inputs.reserve(positions.size());
        // F6. A wrong dtype or order is a TypeError, not nb::cast_error.
        auto f64_item = [](nb::handle item, const char *what, size_t i) {
          try {
            return nb::cast<F64>(item, false);
          } catch (const nb::cast_error &) {
            PyErr_Clear();
            throw nb::type_error(
                ("evaluate_batch: " + std::string(what) + "[" +
                 std::to_string(i) +
                 "] must be a C-contiguous float64 array")
                    .c_str());
          }
        };
        for (size_t i = 0; i < positions.size(); ++i) {
          auto pos = f64_item(positions[i], "positions", i);
          auto box = f64_item(boxes[i], "box", i);
          Geom geom = read_geom(pos, atom_types[i], box);
          BatchInput item;
          item.positions = std::move(geom.pos);
          item.atomic_numbers = std::move(geom.types);
          item.box = geom.box;
          inputs.push_back(std::move(item));
        }
        std::vector<BatchResult> results = run_detached([&] {
          // F4. One BatchEvaluator call. Each worker owns one instance.
          // Do not construct more instances than there are configurations:
          // a single large cell would otherwise pay for unused workspaces.
          unsigned width = n_threads == 0
                               ? std::max(1u, std::thread::hardware_concurrency())
                               : n_threads;
          if (width > inputs.size()) {
            width = static_cast<unsigned>(inputs.size());
          }
          BatchEvaluator eval([&] { return self.make(); }, width);
          return eval.evaluate(inputs);
        });
        nb::list out;
        for (const BatchResult &item : results) {
          if (!std::isfinite(item.energy)) {
            throw rgbind::NumericalError("energy is not finite");
          }
          nb::dict row;
          row["energy"] = item.energy;
          row["forces"] = matrix_to_numpy(item.forces);
          row["variance"] = item.variance;
          out.append(row);
        }
        return out;
      },
      nb::arg("positions"), nb::arg("atom_types"), nb::arg("box"),
      nb::arg("n_threads") = 0,
      "One detached BatchEvaluator call. results[i] matches a serial "
      "call on inputs[i], bit for bit. n_threads 0 uses the hardware "
      "concurrency.");
  cls.def(
      "hvp",
      [](PotHolder<Pot> &self, const F64 &positions, nb::handle atom_types,
         const F64 &box, const F64 &direction, double step,
         const std::string &scheme) {
        Geom geom = read_geom(positions, atom_types, box);
        const size_t n = static_cast<size_t>(geom.pos.rows());
        rgbind::check_f64_2d(direction, n, 3, "direction");
        const HvpScheme parsed = parse_hvp_scheme(scheme);
        std::vector<double> v(direction.data(), direction.data() + 3 * n);
        AtomMatrix hv = AtomMatrix::Zero(geom.pos.rows(), 3);
        run_detached([&] {
          const std::array<double, 9> flat = flat_box(geom);
          const ForceInput in{.nAtoms = n,
                              .pos = geom.pos.data(),
                              .atmnrs = geom.types.data(),
                              .box = flat.data()};
          self.hvp().apply(self.slot(), in, v.data(), hv.data(), step, parsed);
          return 0;
        });
        return matrix_to_numpy(hv);
      },
      nb::arg("positions").noconvert(), nb::arg("atom_types"),
      nb::arg("box").noconvert(), nb::arg("direction").noconvert(),
      nb::arg("step") = 0.0, nb::arg("scheme") = "auto",
      "Hessian-vector product H @ direction, shape (N, 3). 'forward' "
      "reuses the forces of the last geometry this object evaluated, so k "
      "products at one geometry cost k + 1 force calls; 'central' costs 2 "
      "a product and is second order in step; 'analytic' uses the "
      "kernel's own product; 'auto' is analytic where the kernel has one, "
      "otherwise forward. step 0 is 1e-3 Angstrom. The kernel runs with the "
      "interpreter released.");
  cls.def(
      "hvp_centre",
      [](PotHolder<Pot> &self, const F64 &positions, nb::handle atom_types,
         const F64 &box) {
        Geom geom = read_geom(positions, atom_types, box);
        const HvpCentre centre = run_detached([&] {
          const std::array<double, 9> flat = flat_box(geom);
          const ForceInput in{.nAtoms = static_cast<size_t>(geom.pos.rows()),
                              .pos = geom.pos.data(),
                              .atmnrs = geom.types.data(),
                              .box = flat.data()};
          return self.hvp().centre(self.slot(), in);
        });
        AtomMatrix forces(geom.pos.rows(), 3);
        std::copy(centre.forces.begin(), centre.forces.end(), forces.data());
        return nb::make_tuple(centre.energy, matrix_to_numpy(forces),
                              centre.variance);
      },
      nb::arg("positions").noconvert(), nb::arg("atom_types"),
      nb::arg("box").noconvert(),
      "(energy, forces, variance) at the geometry, from the centre that "
      "hvp() reuses. A search that needs the centre forces asks here so "
      "the forward products that follow make one force call each.");
  cls.def(
      "invalidate_hvp",
      [](PotHolder<Pot> &self) { self.hvp().invalidate(); },
      "Drop the centre hvp() reuses.");
  cls.def(
      "hvp_counts",
      [](PotHolder<Pot> &self) {
        const HvpCounts c = self.hvp().counts();
        nb::dict out;
        out["products"] = c.products;
        out["force_calls"] = c.forceCalls;
        out["centre_reused"] = c.centreReused;
        out["analytic"] = c.analytic;
        return out;
      },
      "Products, force calls, centre reuses and analytic products of "
      "hvp() and hvp_centre() so far.");
  cls.def(
      "reset_hvp_counts", [](PotHolder<Pot> &self) { self.hvp().resetCounts(); },
      "Zero hvp_counts(). The centre stays.");
  cls.def(
      "has_analytic_hvp",
      [](const PotHolder<Pot> &self) {
        return self.primary().caps().analyticHvp;
      },
      "Whether hvp(scheme='auto') uses the kernel's own product.");
}

} // namespace rgpot::pybind
