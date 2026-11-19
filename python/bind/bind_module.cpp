// MIT License
// Copyright 2023--present rgpot developers
//
// Python surface: nanobind. F1 free-threaded when the interpreter is
// built that way (see python/meson.build). F4 releases the thread
// around every potential evaluation. F9 exports only PyInit__core.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include "rgbind/rgbind.hpp"

#include "pot_holder.hpp"

#include "rgpot/ExternalPot/ExternalPot.hpp"
#include "rgpot/LennardJones/LJClusterPot.hpp"
#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/Morse/MorsePot.hpp"
#include "rgpot/StillingerWeber/SWSi.hpp"
#include "rgpot/Tersoff/TersoffSiC.hpp"
#include "rgpot/XcKernel/XcKernel.hpp"
#include "rgpot/nlist/NeighbourList.hpp"
#include "rgpot/types/AtomMatrix.hpp"

#ifdef RGPOT_HAS_METATOMIC
#include "rgpot/MetatomicPot/MetatomicConfig.hpp"
#include "rgpot/MetatomicPot/MetatomicDlopen.hpp"
#endif

#ifdef RGPOT_HAS_CACHE
#include "rgpot/PotentialCache.hpp"
#endif

#ifdef RGPOT_HAS_DFTD3
#include "rgpot/D3Pot/D3Pot.hpp"
#endif
#ifdef RGPOT_HAS_DFTD4
#include "rgpot/D4Pot/D4Pot.hpp"
#endif
#ifdef RGPOT_HAS_EXPR
#include "rgpot/ExprPot/ExprPot.hpp"
#endif
#ifdef RGPOT_HAS_FORTRAN_POTS
#include "rgpot/fortran/FortranPots.hpp"
#endif
#ifdef RGPOT_HAS_IPI
#include "rgpot/IPIPot/IPIPot.hpp"
#endif

#ifndef RGPOT_PY_VERSION
#define RGPOT_PY_VERSION "0.0.0"
#endif

namespace nb = nanobind;
using rgpot::pybind::F64;
using rgpot::pybind::PotHolder;
using rgpot::types::AtomMatrix;

namespace {

void translate_invalid_argument(const std::exception_ptr &ptr, void *) {
  try {
    std::rethrow_exception(ptr);
  } catch (const std::invalid_argument &err) {
    PyErr_SetString(PyExc_ValueError, err.what());
  }
}

#ifdef RGPOT_HAS_DFTD3
rgpot::D3Damping d3_damping(const std::string &name) {
  if (name == "bj" || name == "BJ") {
    return rgpot::D3Damping::BJ;
  }
  if (name == "zero" || name == "Zero") {
    return rgpot::D3Damping::Zero;
  }
  throw std::invalid_argument("D3 damping must be 'bj' or 'zero', got '" +
                              name + "'");
}
#endif

#ifdef RGPOT_HAS_EXPR
struct ChildSpec {
  std::string name;
  std::string kind;
  std::string functional = "pbe";
  std::string damping = "bj";
  bool atm = true;
  double charge = 0.0;
  // LJ and Morse dict fields. Absent keys keep the C++ config defaults,
  // which differ (LJ cutoff is 15 A, Morse cutoff is 9.5 A).
  bool has_u0 = false;
  double u0 = 0.0;
  bool has_cutoff = false;
  double cutoff = 0.0;
  bool has_psi = false;
  double psi = 0.0;
  bool has_switch_width = false;
  double switch_width = 0.0;
  bool has_De = false;
  double De = 0.0;
  bool has_a = false;
  double a = 0.0;
  bool has_re = false;
  double re = 0.0;
};

std::string child_kind(nb::handle spec) {
  if (nb::isinstance<nb::str>(spec)) {
    return nb::cast<std::string>(spec);
  }
  if (nb::isinstance<nb::dict>(spec)) {
    auto d = nb::cast<nb::dict>(spec);
    if (d.contains("kind")) {
      return nb::cast<std::string>(d["kind"]);
    }
    if (d.contains("type")) {
      return nb::cast<std::string>(d["type"]);
    }
  }
  throw nb::value_error(
      "ExprPot child spec must be a kind string or a dict with 'kind'");
}

ChildSpec child_spec_from(const std::string &name, nb::handle spec) {
  ChildSpec out;
  out.name = name;
  out.kind = child_kind(spec);
  if (nb::isinstance<nb::dict>(spec)) {
    auto d = nb::cast<nb::dict>(spec);
    if (d.contains("functional")) {
      out.functional = nb::cast<std::string>(d["functional"]);
    }
    if (d.contains("atm")) {
      out.atm = nb::cast<bool>(d["atm"]);
    }
    if (d.contains("damping")) {
      out.damping = nb::cast<std::string>(d["damping"]);
    }
    if (d.contains("charge")) {
      out.charge = nb::cast<double>(d["charge"]);
    }
    if (d.contains("u0")) {
      out.has_u0 = true;
      out.u0 = nb::cast<double>(d["u0"]);
    }
    if (d.contains("cutoff")) {
      out.has_cutoff = true;
      out.cutoff = nb::cast<double>(d["cutoff"]);
    }
    if (d.contains("psi")) {
      out.has_psi = true;
      out.psi = nb::cast<double>(d["psi"]);
    }
    if (d.contains("switch_width")) {
      out.has_switch_width = true;
      out.switch_width = nb::cast<double>(d["switch_width"]);
    }
    if (d.contains("De")) {
      out.has_De = true;
      out.De = nb::cast<double>(d["De"]);
    }
    if (d.contains("a")) {
      out.has_a = true;
      out.a = nb::cast<double>(d["a"]);
    }
    if (d.contains("re")) {
      out.has_re = true;
      out.re = nb::cast<double>(d["re"]);
    }
  }
  return out;
}

std::unique_ptr<rgpot::PotentialBase> child_from_spec(const ChildSpec &spec) {
  if (spec.kind == "lj") {
    rgpot::LJConfig cfg;
    if (spec.has_u0) {
      cfg.u0 = spec.u0;
    }
    if (spec.has_cutoff) {
      cfg.cutoff = spec.cutoff;
    }
    if (spec.has_psi) {
      cfg.psi = spec.psi;
    }
    if (spec.has_switch_width) {
      cfg.switch_width = spec.switch_width;
    }
    return std::make_unique<rgpot::LJPot>(cfg);
  }
  if (spec.kind == "morse") {
    rgpot::MorseConfig cfg;
    if (spec.has_De) {
      cfg.De = spec.De;
    }
    if (spec.has_a) {
      cfg.a = spec.a;
    }
    if (spec.has_re) {
      cfg.re = spec.re;
    }
    if (spec.has_cutoff) {
      cfg.cutoff = spec.cutoff;
    }
    if (spec.has_switch_width) {
      cfg.switch_width = spec.switch_width;
    }
    return std::make_unique<rgpot::MorsePot>(cfg);
  }
#ifdef RGPOT_HAS_DFTD3
  if (spec.kind == "d3") {
    rgpot::D3Config cfg;
    cfg.functional = spec.functional;
    cfg.atm = spec.atm;
    cfg.damping = d3_damping(spec.damping);
    return std::make_unique<rgpot::D3Pot>(cfg);
  }
#endif
#ifdef RGPOT_HAS_DFTD4
  if (spec.kind == "d4") {
    rgpot::D4Config cfg;
    cfg.functional = spec.functional;
    cfg.charge = spec.charge;
    cfg.atm = spec.atm;
    return std::make_unique<rgpot::D4Pot>(cfg);
  }
#endif
  throw std::invalid_argument("unknown ExprPot child kind: " + spec.kind);
}

std::unique_ptr<rgpot::ExprPot>
expr_from_specs(const std::string &expression,
                const std::vector<ChildSpec> &specs) {
  std::vector<rgpot::ExprPot::Term> children;
  children.reserve(specs.size());
  for (const ChildSpec &spec : specs) {
    children.emplace_back(spec.name, child_from_spec(spec));
  }
  return std::make_unique<rgpot::ExprPot>(expression, std::move(children));
}
#endif

#ifdef RGPOT_HAS_XCKERNEL
struct NumpyXcGrid {
  rgpot::XcGrid grid;
  F64 dchi, lapl, hess;
};

NumpyXcGrid grid_from_numpy(const F64 &chi, nb::handle dchi_obj,
                            nb::handle lapl_obj, nb::handle hess_obj) {
  rgbind::check_f64_rank(chi, 2, "chi");
  NumpyXcGrid storage;
  auto &grid = storage.grid;
  grid.nbf = static_cast<std::int64_t>(chi.shape(0));
  grid.npts = static_cast<std::int64_t>(chi.shape(1));
  grid.chi = chi.data();
  if (!dchi_obj.is_none()) {
    storage.dchi = nb::cast<F64>(dchi_obj, false);
    const auto &dchi = storage.dchi;
    if (dchi.ndim() != 3 || dchi.shape(0) != 3 ||
        dchi.shape(1) != chi.shape(0) || dchi.shape(2) != chi.shape(1)) {
      throw nb::value_error("dchi: expected shape (3, nbf, npts)");
    }
    grid.dchi = dchi.data();
  }
  if (!lapl_obj.is_none()) {
    storage.lapl = nb::cast<F64>(lapl_obj, false);
    const auto &lapl = storage.lapl;
    if (lapl.ndim() != 2 || lapl.shape(0) != chi.shape(0) ||
        lapl.shape(1) != chi.shape(1)) {
      throw nb::value_error("lapl_chi: expected shape (nbf, npts)");
    }
    grid.lapl_chi = lapl.data();
  }
  if (!hess_obj.is_none()) {
    storage.hess = nb::cast<F64>(hess_obj, false);
    const auto &hess = storage.hess;
    if (hess.ndim() != 3 || hess.shape(0) != 6 ||
        hess.shape(1) != chi.shape(0) || hess.shape(2) != chi.shape(1)) {
      throw nb::value_error("hess_chi: expected shape (6, nbf, npts)");
    }
    grid.hess_chi = hess.data();
  }
  return storage;
}

struct NumpyXcScalars {
  std::map<std::string, const double *> pointers;
  std::vector<F64> arrays;
};

NumpyXcScalars scal_from_dict(const nb::dict &scal, std::size_t npts) {
  NumpyXcScalars storage;
  storage.arrays.reserve(scal.size());
  for (auto item : scal) {
    auto name = nb::cast<std::string>(item.first);
    auto arr = nb::cast<F64>(item.second, false);
    if (!((arr.ndim() == 1 && arr.shape(0) == npts) ||
          (arr.ndim() == 2 && arr.shape(0) == npts && arr.shape(1) == 1))) {
      throw nb::value_error(
          ("scalar " + name + ": expected shape (npts,) or (npts, 1)").c_str());
    }
    storage.pointers.emplace(std::move(name), arr.data());
    storage.arrays.push_back(std::move(arr));
  }
  return storage;
}
#endif

nb::ndarray<nb::numpy, double, nb::c_contig>
buffer_to_numpy(const double *src, size_t rows, size_t cols) {
  const size_t n = rows * cols;
  double *buf = new double[n == 0 ? 1 : n];
  if (n != 0 && src != nullptr) {
    std::memcpy(buf, src, n * sizeof(double));
  }
  nb::capsule owner(
      buf, [](void *p) noexcept { delete[] static_cast<double *>(p); });
  return nb::ndarray<nb::numpy, double, nb::c_contig>(buf, {rows, cols}, owner);
}

#ifdef RGPOT_HAS_FORTRAN_POTS
template <typename Pot>
void bind_default_pot(nb::module_ &m, const char *name) {
  auto cls = nb::class_<PotHolder<Pot>>(m, name);
  cls.def("__init__", [](PotHolder<Pot> *self) {
    new (self) PotHolder<Pot>([] { return std::make_unique<Pot>(); });
  });
  rgpot::pybind::bind_calls(cls);
}
#endif

nb::tuple stillinger_weber_energy_gradient(rgpot::StillingerWeberSi &self,
                                           const F64 &positions,
                                           nb::handle atom_types,
                                           const F64 &box) {
  if (box.ndim() != 2 || box.shape(0) != 3 || box.shape(1) != 3) {
    throw std::invalid_argument("box must have shape (3, 3)");
  }
  const double *cell = box.data();
  auto own = [](double *buf) {
    return nb::capsule(
        buf, [](void *p) noexcept { delete[] static_cast<double *>(p); });
  };
  if (positions.ndim() == 2) {
    if (positions.shape(1) != 3) {
      throw std::invalid_argument("positions must have shape (n_atoms, 3)");
    }
    const auto n = static_cast<std::size_t>(positions.shape(0));
    const std::vector<int> species = rgpot::pybind::read_types(atom_types, n);
    auto *grad = new double[std::max<std::size_t>(n * 3, 1)]();
    double energy = 0.0;
    rgpot::pybind::run_detached([&] {
      self.energyAndGradient(positions.data(), species.data(), cell, n, &energy,
                             grad);
      return 0;
    });
    auto arr =
        nb::ndarray<nb::numpy, double, nb::c_contig>(grad, {n, 3}, own(grad));
    return nb::make_tuple(energy, arr);
  }
  if (positions.ndim() == 3) {
    if (positions.shape(2) != 3) {
      throw std::invalid_argument(
          "positions must have shape (n_config, n_atoms, 3)");
    }
    const auto n_config = static_cast<std::size_t>(positions.shape(0));
    const auto n = static_cast<std::size_t>(positions.shape(1));
    const std::vector<int> species = rgpot::pybind::read_types(atom_types, n);
    auto *energies = new double[std::max<std::size_t>(n_config, 1)]();
    auto *grads = new double[std::max<std::size_t>(n_config * n * 3, 1)]();
    rgpot::pybind::run_detached([&] {
      self.energyAndGradientBatch(n_config, positions.data(), species.data(),
                                  cell, n, energies, grads);
      return 0;
    });
    auto earr = nb::ndarray<nb::numpy, double, nb::c_contig>(
        energies, {n_config}, own(energies));
    auto garr = nb::ndarray<nb::numpy, double, nb::c_contig>(
        grads, {n_config, n, 3}, own(grads));
    return nb::make_tuple(earr, garr);
  }
  throw std::invalid_argument(
      "positions must have shape (n_atoms, 3) or (n_config, n_atoms, 3)");
}

nb::tuple tersoff_sic_energy_gradient(rgpot::TersoffSiC &self,
                                      const F64 &positions,
                                      nb::handle atom_types, const F64 &box) {
  if (box.ndim() != 2 || box.shape(0) != 3 || box.shape(1) != 3) {
    throw std::invalid_argument("box must have shape (3, 3)");
  }
  const double *cell = box.data();
  auto own = [](double *buf) {
    return nb::capsule(
        buf, [](void *p) noexcept { delete[] static_cast<double *>(p); });
  };
  if (positions.ndim() == 2) {
    if (positions.shape(1) != 3) {
      throw std::invalid_argument("positions must have shape (n_atoms, 3)");
    }
    const auto n = static_cast<std::size_t>(positions.shape(0));
    const std::vector<int> species = rgpot::pybind::read_types(atom_types, n);
    auto *grad = new double[std::max<std::size_t>(n * 3, 1)]();
    double energy = 0.0;
    rgpot::pybind::run_detached([&] {
      self.energyAndGradient(positions.data(), species.data(), cell, n, &energy,
                             grad);
      return 0;
    });
    auto arr =
        nb::ndarray<nb::numpy, double, nb::c_contig>(grad, {n, 3}, own(grad));
    return nb::make_tuple(energy, arr);
  }
  if (positions.ndim() == 3) {
    if (positions.shape(2) != 3) {
      throw std::invalid_argument(
          "positions must have shape (n_config, n_atoms, 3)");
    }
    const auto n_config = static_cast<std::size_t>(positions.shape(0));
    const auto n = static_cast<std::size_t>(positions.shape(1));
    const std::vector<int> species = rgpot::pybind::read_types(atom_types, n);
    auto *energies = new double[std::max<std::size_t>(n_config, 1)]();
    auto *grads = new double[std::max<std::size_t>(n_config * n * 3, 1)]();
    rgpot::pybind::run_detached([&] {
      self.energyAndGradientBatch(n_config, positions.data(), species.data(),
                                  cell, n, energies, grads);
      return 0;
    });
    auto earr = nb::ndarray<nb::numpy, double, nb::c_contig>(
        energies, {n_config}, own(energies));
    auto garr = nb::ndarray<nb::numpy, double, nb::c_contig>(
        grads, {n_config, n, 3}, own(grads));
    return nb::make_tuple(earr, garr);
  }
  throw std::invalid_argument(
      "positions must have shape (n_atoms, 3) or (n_config, n_atoms, 3)");
}

} // namespace

NB_MODULE(_core, m) {
  // F1, F3, F7. install_policy registers NumericalError and the
  // in-use translator. A second translator maps invalid_argument to
  // ValueError (F7).
  rgbind::install_policy(m);
  nb::register_exception_translator(translate_invalid_argument);

  m.doc() = "rgpot Python bindings (nanobind). Classes that need a build "
            "option, such as ExprPot, D3Pot, D4Pot, XcKernel and the Fortran "
            "kernels, are absent when the build left them out. Evaluations "
            "release the interpreter while the kernel runs.";
  m.attr("__version__") = RGPOT_PY_VERSION;
#ifdef RGPOT_HAS_METATOMIC
  m.attr("has_metatomic_dlopen") = true;
#else
  m.attr("has_metatomic_dlopen") = false;
#endif
#ifdef RGPOT_HAS_CACHE
  m.attr("has_cache") = true;
#else
  m.attr("has_cache") = false;
#endif
#ifdef RGPOT_HAS_EXPR
  m.attr("has_expr") = true;
#else
  m.attr("has_expr") = false;
#endif
#ifdef RGPOT_HAS_DFTD3
  m.attr("has_dftd3") = true;
#else
  m.attr("has_dftd3") = false;
#endif
#ifdef RGPOT_HAS_DFTD4
  m.attr("has_dftd4") = true;
#else
  m.attr("has_dftd4") = false;
#endif
#ifdef RGPOT_HAS_XCKERNEL
  m.attr("has_xckernel") = true;
#else
  m.attr("has_xckernel") = false;
#endif
#ifdef RGPOT_HAS_FORTRAN_POTS
  m.attr("has_fortran_pots") = true;
#else
  m.attr("has_fortran_pots") = false;
#endif
#ifdef RGPOT_HAS_IPI
  m.attr("has_ipi") = true;
#else
  m.attr("has_ipi") = false;
#endif

  // F6 round trip. -0.0, NaN, and subnormals are copied unchanged.
  m.def(
      "echo_f64", [](const F64 &values) { return rgbind::echo_f64(values); },
      nb::arg("values").noconvert());

  m.def(
      "evaluate_lj",
      [](const F64 &positions, nb::handle atom_types, const F64 &box) {
        auto geom = rgpot::pybind::read_geom(positions, atom_types, box);
        rgpot::LJPot pot;
        return rgpot::pybind::call_detached(pot, geom);
      },
      nb::arg("positions").noconvert(), nb::arg("atom_types"),
      nb::arg("box").noconvert());

#ifdef RGPOT_HAS_METATOMIC
  m.def(
      "evaluate_metatomic_dlopen",
      [](const F64 &positions, nb::handle atom_types, const F64 &box,
         const std::string &model_path, const std::string &engine_path,
         const std::string &device) {
        if (model_path.empty()) {
          throw nb::value_error("model_path is required");
        }
        auto geom = rgpot::pybind::read_geom(positions, atom_types, box);
        rgpot::MetatomicConfig cfg;
        cfg.model_path = model_path;
        cfg.engine_path = engine_path;
        cfg.device = device.empty() ? "cpu" : device;
        rgpot::MetatomicDlopen front(cfg);
        if (!front.available()) {
          throw std::runtime_error("MetatomicDlopen: engine not available");
        }
        const size_t n = geom.pos.rows();
        AtomMatrix forces = AtomMatrix::Zero(n, 3);
        double flat[9];
        for (int i = 0; i < 3; ++i) {
          for (int j = 0; j < 3; ++j) {
            flat[i * 3 + j] =
                geom.box[static_cast<size_t>(i)][static_cast<size_t>(j)];
          }
        }
        rgpot::ForceInput in{n, geom.pos.data(), geom.types.data(), flat};
        rgpot::ForceOut out{forces.data(), 0.0, 0.0, {}, 0};
        rgpot::pybind::run_detached([&] {
          front.forceImpl(in, &out);
          return 0;
        });
        if (!std::isfinite(out.energy)) {
          throw rgbind::NumericalError("energy is not finite");
        }
        return nb::make_tuple(
            out.energy, rgpot::pybind::matrix_to_numpy(forces), out.variance);
      },
      nb::arg("positions").noconvert(), nb::arg("atom_types"),
      nb::arg("box").noconvert(), nb::arg("model_path"),
      nb::arg("engine_path") = "", nb::arg("device") = "cpu",
      "Evaluate forces through a Metatomic engine opened with dlopen. With "
      "an empty engine_path the search is RGPOT_METATOMIC_ENGINE, "
      "METATOMIC_ENGINE, libmetatomic_engine.so on the loader path, then "
      "the directories in EON_POTENTIALS_PATH and RGPOT_ENGINE_PATH. "
      "rgpot.evaluate_metatomic also tries the engines bundled with the "
      "package. Each call builds its own frontend.");
#endif

#ifdef RGPOT_HAS_CACHE
  nb::class_<rgpot::cache::PotentialCache>(m, "PotentialCache")
      .def(nb::init<const std::string &, bool>(), nb::arg("path"),
           nb::arg("create_if_missing") = true)
      .def(
          "counts",
          [](const rgpot::cache::PotentialCache &self) {
            const auto c = self.counts();
            return nb::make_tuple(c.computed, c.served);
          },
          "Return (computed, served): results the kernel produced against "
          "results answered from this cache since construction or the last "
          "reset_counts().")
      .def("reset_counts", &rgpot::cache::PotentialCache::reset_counts);
#endif

  auto lj = nb::class_<PotHolder<rgpot::LJPot>>(m, "LJPot");
  lj.def(
      "__init__",
      [](PotHolder<rgpot::LJPot> *self, double u0, double cutoff, double psi,
         double switch_width) {
        new (self) PotHolder<rgpot::LJPot>([=] {
          rgpot::LJConfig cfg;
          cfg.u0 = u0;
          cfg.cutoff = cutoff;
          cfg.psi = psi;
          cfg.switch_width = switch_width;
          return std::make_unique<rgpot::LJPot>(cfg);
        });
      },
      nb::arg("u0") = 1.0, nb::arg("cutoff") = 15.0, nb::arg("psi") = 1.0,
      nb::arg("switch_width") = 0.0,
      "Shifted 12-6 Lennard-Jones. u0 is the well depth (eV), cutoff "
      "and psi are Angstrom. A positive switch_width (Angstrom) "
      "replaces the shift with the C^2 quintic switch over "
      "[cutoff - switch_width, cutoff]; it must not exceed cutoff.");
  lj.def("config", [](const PotHolder<rgpot::LJPot> &self) {
    const rgpot::LJConfig &cfg = self.primary().config();
    nb::dict out;
    out["u0"] = cfg.u0;
    out["cutoff"] = cfg.cutoff;
    out["psi"] = cfg.psi;
    out["switch_width"] = cfg.switch_width;
    return out;
  });
#ifdef RGPOT_HAS_CACHE
  lj.def(
      "set_cache",
      [](PotHolder<rgpot::LJPot> &self, rgpot::cache::PotentialCache *cache) {
        self.primary().setCache(cache);
      },
      nb::arg("cache").none(), nb::keep_alive<1, 2>());
#endif
  rgpot::pybind::bind_calls(lj);

  // Same call surface as LJPot. Both classes are already in librgpot;
  // this is the binding a caller of the wheel uses.
  auto cluster = nb::class_<PotHolder<rgpot::LJClusterPot>>(m, "LJClusterPot");
  cluster.def(
      "__init__",
      [](PotHolder<rgpot::LJClusterPot> *self, double u0, double cutoff,
         double psi, double switch_width) {
        new (self) PotHolder<rgpot::LJClusterPot>([=] {
          rgpot::LJClusterConfig cfg;
          cfg.u0 = u0;
          cfg.cutoff = cutoff;
          cfg.psi = psi;
          cfg.switch_width = switch_width;
          return std::make_unique<rgpot::LJClusterPot>(cfg);
        });
      },
      nb::arg("u0") = 1.0, nb::arg("cutoff") = 15.0, nb::arg("psi") = 1.0,
      nb::arg("switch_width") = 0.0,
      "Free-boundary shifted 12-6 Lennard-Jones. u0 is the well depth "
      "(eV), cutoff and psi are Angstrom. An input cell does not fold "
      "pair vectors. A positive switch_width (Angstrom) replaces the "
      "shift with the C^2 quintic switch over "
      "[cutoff - switch_width, cutoff]; it must not exceed cutoff.");
  cluster.def("config", [](const PotHolder<rgpot::LJClusterPot> &self) {
    const rgpot::LJClusterConfig &cfg = self.primary().config();
    nb::dict out;
    out["u0"] = cfg.u0;
    out["cutoff"] = cfg.cutoff;
    out["psi"] = cfg.psi;
    out["switch_width"] = cfg.switch_width;
    return out;
  });
  rgpot::pybind::bind_calls(cluster);

  auto morse_cls = nb::class_<PotHolder<rgpot::MorsePot>>(m, "MorsePot");
  morse_cls.def(
      "__init__",
      [](PotHolder<rgpot::MorsePot> *self, double De, double a, double re,
         double cutoff, double switch_width) {
        new (self) PotHolder<rgpot::MorsePot>([=] {
          rgpot::MorseConfig cfg;
          cfg.De = De;
          cfg.a = a;
          cfg.re = re;
          cfg.cutoff = cutoff;
          cfg.switch_width = switch_width;
          return std::make_unique<rgpot::MorsePot>(cfg);
        });
      },
      nb::arg("De") = 0.7102, nb::arg("a") = 1.6047, nb::arg("re") = 2.8970,
      nb::arg("cutoff") = 9.5, nb::arg("switch_width") = 0.0,
      "Shifted Morse pair potential. De is the well depth (eV), a is "
      "1/Angstrom, re and cutoff are Angstrom. Defaults are the platinum "
      "parameters. A positive switch_width (Angstrom) replaces the shift "
      "with the C^2 quintic switch over [cutoff - switch_width, cutoff]; "
      "it must not exceed cutoff.");
  morse_cls.def("config", [](const PotHolder<rgpot::MorsePot> &self) {
    const rgpot::MorseConfig &cfg = self.primary().config();
    nb::dict out;
    out["De"] = cfg.De;
    out["a"] = cfg.a;
    out["re"] = cfg.re;
    out["cutoff"] = cfg.cutoff;
    out["switch_width"] = cfg.switch_width;
    return out;
  });
  rgpot::pybind::bind_calls(morse_cls);

#ifdef RGPOT_HAS_FORTRAN_POTS
  bind_default_pot<rgpot::fortranpots::SWPot>(m, "SWPot");
  bind_default_pot<rgpot::fortranpots::EDIPPot>(m, "EDIPPot");
  bind_default_pot<rgpot::fortranpots::LenoskyPot>(m, "LenoskyPot");
  bind_default_pot<rgpot::fortranpots::TersoffPot>(m, "TersoffPot");
  bind_default_pot<rgpot::fortranpots::EAMAlPot>(m, "EAMAlPot");
  bind_default_pot<rgpot::fortranpots::FeHePot>(m, "FeHePot");
  bind_default_pot<rgpot::fortranpots::CuH2Pot>(m, "CuH2Pot");
  bind_default_pot<rgpot::fortranpots::WaterHPot>(m, "WaterHPot");
#endif

  nb::class_<rgpot::StillingerWeberSi>(m, "StillingerWeberSi")
      .def(nb::init<>(),
           "Stillinger-Weber silicon, Stillinger and Weber, Phys. Rev. B "
           "31, 5262 (1985). Epsilon is the published 2.16723920 eV, "
           "not eOn's 2.3149945 eV (ratio 1.068176741337). Energy is "
           "eV, lengths are Angstrom. The gradient is dE/dx.")
      .def(
          "__call__",
          [](rgpot::StillingerWeberSi &self, const F64 &positions,
             nb::handle atom_types, const F64 &box) {
            auto geom = rgpot::pybind::read_geom(positions, atom_types, box);
            return rgpot::pybind::call_detached(self, geom);
          },
          nb::arg("positions").noconvert(), nb::arg("atom_types"),
          nb::arg("box").noconvert(),
          "Return (energy, forces, variance) for one configuration. "
          "Forces are -dE/dx.")
      .def("caps",
           [](const rgpot::StillingerWeberSi &self) {
             return rgpot::pybind::caps_dict(self.caps());
           })
      .def(
          "evaluate",
          [](rgpot::StillingerWeberSi &self, const F64 &positions,
             nb::handle atom_types, const F64 &box) {
            auto geom = rgpot::pybind::read_geom(positions, atom_types, box);
            rgpot::pybind::FullResult result = rgpot::pybind::run_detached(
                [&] { return rgpot::pybind::run_force(self, geom); });
            return rgpot::pybind::result_dict(result);
          },
          nb::arg("positions").noconvert(), nb::arg("atom_types"),
          nb::arg("box").noconvert(),
          "Energy, forces, variance, and the Cauchy stress, shape (3, 3), "
          "in eV/Angstrom^3. The kernel runs detached.")
      .def("energy_gradient", &stillinger_weber_energy_gradient,
           nb::arg("positions").noconvert(), nb::arg("atom_types"),
           nb::arg("box").noconvert(),
           "Energy and dE/dx. positions is (n_atoms, 3) or "
           "(n_config, n_atoms, 3). A batch shares atom_types and box, "
           "and crosses into the library once.");

  nb::class_<rgpot::TersoffSiC>(m, "TersoffSiC")
      .def(nb::init<>(),
           "Tersoff silicon-carbon kernel, Tersoff, Phys. Rev. B 39, 5566 "
           "(1989). The 1990 erratum sets carbon B to 346.74 eV and d to "
           "4.3484. Carbon beta is dimensionless. Atomic numbers 14 and "
           "6. This class always evaluates the 1989 kernel. The engine "
           "name tersoff uses the Fortran silicon gather when every atom "
           "is silicon. Energy is eV, lengths are Angstrom. The gradient "
           "is dE/dx.")
      .def(
          "__call__",
          [](rgpot::TersoffSiC &self, const F64 &positions,
             nb::handle atom_types, const F64 &box) {
            auto geom = rgpot::pybind::read_geom(positions, atom_types, box);
            return rgpot::pybind::call_detached(self, geom);
          },
          nb::arg("positions").noconvert(), nb::arg("atom_types"),
          nb::arg("box").noconvert(),
          "Return (energy, forces, variance) for one configuration. "
          "Forces are -dE/dx.")
      .def("energy_gradient", &tersoff_sic_energy_gradient,
           nb::arg("positions").noconvert(), nb::arg("atom_types"),
           nb::arg("box").noconvert(),
           "Energy and dE/dx. positions is (n_atoms, 3) or "
           "(n_config, n_atoms, 3). A batch shares atom_types and box, "
           "and crosses into the library once.");

#ifdef RGPOT_HAS_DFTD3
  auto d3 = nb::class_<PotHolder<rgpot::D3Pot>>(m, "D3Pot");
  d3.def(
      "__init__",
      [](PotHolder<rgpot::D3Pot> *self, const std::string &damping,
         const std::string &functional, bool atm) {
        const rgpot::D3Damping form = d3_damping(damping);
        new (self) PotHolder<rgpot::D3Pot>([=] {
          rgpot::D3Config cfg;
          cfg.functional = functional;
          cfg.atm = atm;
          cfg.damping = form;
          return std::make_unique<rgpot::D3Pot>(cfg);
        });
      },
      nb::arg("damping") = "bj", nb::arg("functional") = "pbe",
      nb::arg("atm") = true);
  rgpot::pybind::bind_calls(d3);
#endif

#ifdef RGPOT_HAS_DFTD4
  auto d4 = nb::class_<PotHolder<rgpot::D4Pot>>(m, "D4Pot");
  d4.def(
      "__init__",
      [](PotHolder<rgpot::D4Pot> *self, const std::string &functional,
         double charge, bool atm) {
        new (self) PotHolder<rgpot::D4Pot>([=] {
          rgpot::D4Config cfg;
          cfg.functional = functional;
          cfg.charge = charge;
          cfg.atm = atm;
          return std::make_unique<rgpot::D4Pot>(cfg);
        });
      },
      nb::arg("functional") = "pbe", nb::arg("charge") = 0.0,
      nb::arg("atm") = true);
  rgpot::pybind::bind_calls(d4);
#endif

#ifdef RGPOT_HAS_EXPR
  auto expr = nb::class_<PotHolder<rgpot::ExprPot>>(m, "ExprPot");
  expr.def(
      "__init__",
      [](PotHolder<rgpot::ExprPot> *self, const std::string &expression,
         const nb::dict &terms) {
        std::vector<ChildSpec> specs;
        specs.reserve(terms.size());
        for (auto item : terms) {
          auto name = nb::cast<std::string>(item.first);
          specs.push_back(child_spec_from(name, item.second));
        }
        new (self) PotHolder<rgpot::ExprPot>(
            [=] { return expr_from_specs(expression, specs); });
      },
      nb::arg("expression"), nb::arg("terms"),
      "expression is a Lepton string; terms maps names to 'lj'/'morse'/"
      "'d3'/'d4' or a dict with kind plus that kind's fields: u0, "
      "cutoff, psi, switch_width (lj); De, a, re, cutoff, switch_width "
      "(morse); the D3/D4 fields.");
  expr.def_prop_ro("expression", [](const PotHolder<rgpot::ExprPot> &self) {
    return self.primary().expression();
  });
  expr.def(
      "d_energy_d_term",
      [](PotHolder<rgpot::ExprPot> &self, const std::string &name) {
        nb::gil_scoped_release release;
        return self.slot().dEnergyDTerm(name);
      },
      nb::arg("name"));
  rgpot::pybind::bind_calls(expr);
#endif

#ifdef RGPOT_HAS_IPI
  nb::class_<rgpot::IPIPot>(m, "IPIPot")
      .def(
          "__init__",
          [](rgpot::IPIPot *self, const std::string &address,
             double timeout_s) {
            rgpot::IPIConfig cfg;
            cfg.address = address;
            cfg.timeout_s = timeout_s;
            new (self) rgpot::IPIPot(cfg);
          },
          nb::arg("address"), nb::arg("timeout_s") = 30.0,
          "i-PI socket server. address is unix:/absolute/path, unix:<name> "
          "(binds /tmp/ipi_<name>), tcp:<host>:<port>, or tcp:<port>. The "
          "driver connects and returns forces. Wire units are bohr and "
          "hartree. timeout_s covers connecting and one force exchange.")
      .def(
          "__call__",
          [](rgpot::IPIPot &self, const F64 &positions, nb::handle atom_types,
             const F64 &box) {
            auto geom = rgpot::pybind::read_geom(positions, atom_types, box);
            return rgpot::pybind::call_detached(self, geom);
          },
          nb::arg("positions").noconvert(), nb::arg("atom_types"),
          nb::arg("box").noconvert())
      .def(
          "stress",
          [](const rgpot::IPIPot &self) -> nb::object {
            double stress[9];
            if (!self.copyStress(stress)) {
              return nb::none();
            }
            return nb::cast(buffer_to_numpy(stress, 3, 3));
          },
          "Cauchy stress from the last call, shape (3, 3), in eV/Angstrom^3. "
          "None before a successful call.")
      .def_prop_ro("endpoint", &rgpot::IPIPot::endpoint);
#endif

#ifdef RGPOT_HAS_XCKERNEL
  nb::class_<rgpot::XcKernel>(m, "XcKernel")
      .def(nb::init<std::string>(), nb::arg("name"))
      .def_static("catalog", &rgpot::XcKernel::catalog)
      .def_prop_ro("name", &rgpot::XcKernel::name)
      .def("scal_names", &rgpot::XcKernel::scalNames)
      .def(
          "contract",
          [](const rgpot::XcKernel &self, const F64 &chi, nb::handle dchi,
             const nb::dict &scal, nb::handle lapl_chi, nb::handle hess_chi) {
            auto storage = grid_from_numpy(chi, dchi, lapl_chi, hess_chi);
            const auto &grid = storage.grid;
            auto scalars = scal_from_dict(scal, static_cast<size_t>(grid.npts));
            const auto nbf = static_cast<size_t>(grid.nbf);
            std::vector<double> out(nbf * nbf, 0.0);
            int rc = 0;
            {
              nb::gil_scoped_release release;
              rc = self.contract(grid, scalars.pointers, out.data());
            }
            if (rc != 0) {
              throw std::runtime_error("XcKernel.contract rc=" +
                                       std::to_string(rc));
            }
            return buffer_to_numpy(out.data(), nbf, nbf);
          },
          // dchi defaults to None so an LDA call can omit it. scal stays
          // required, but it also defaults to None: a non-default parameter
          // cannot follow a default in the stub, and nanobind rejects None
          // when casting to dict.
          nb::arg("chi").noconvert(), nb::arg("dchi") = nb::none(),
          nb::arg("scal") = nb::none(), nb::arg("lapl_chi") = nb::none(),
          nb::arg("hess_chi") = nb::none())
      .def_static(
          "fields_from_density",
          [](const F64 &chi, nb::handle dchi, const F64 &density,
             nb::handle lapl_chi, nb::handle hess_chi) {
            auto storage = grid_from_numpy(chi, dchi, lapl_chi, hess_chi);
            const auto &grid = storage.grid;
            rgbind::check_f64_2d(density, static_cast<size_t>(chi.shape(0)),
                                 static_cast<size_t>(chi.shape(0)), "density");
            std::vector<double> density_copy(density.data(),
                                             density.data() + density.size());
            rgpot::XcFields fields;
            {
              nb::gil_scoped_release release;
              fields =
                  rgpot::XcKernel::fieldsFromDensity(grid, density_copy.data());
            }
            nb::dict out;
            auto vec1 = [](const std::vector<double> &v, size_t n) {
              return buffer_to_numpy(v.data(), n, 1);
            };
            const auto npts = static_cast<size_t>(grid.npts);
            out["rho"] = vec1(fields.rho, npts);
            out["sigma"] = vec1(fields.sigma, npts);
            out["tau"] = vec1(fields.tau, npts);
            out["lapl"] = vec1(fields.lapl, npts);
            if (!fields.grad_rho.empty()) {
              out["grad_rho"] =
                  buffer_to_numpy(fields.grad_rho.data(), 3, npts);
            }
            return out;
          },
          nb::arg("chi").noconvert(), nb::arg("dchi"),
          nb::arg("density").noconvert(), nb::arg("lapl_chi") = nb::none(),
          nb::arg("hess_chi") = nb::none());
#endif

  m.def(
      "neighbor_list",
      [](const F64 &positions, const F64 &cell, double cutoff,
         const std::string &quantities, bool full, bool offsets) {
        if (positions.ndim() != 2 || positions.shape(1) != 3) {
          throw std::invalid_argument("positions must have shape (natoms, 3)");
        }
        if (cell.ndim() != 2 || cell.shape(0) != 3 || cell.shape(1) != 3) {
          throw std::invalid_argument("cell must have shape (3, 3)");
        }
        for (char q : quantities) {
          if (q != 'i' && q != 'j' && q != 'd' && q != 'D' && q != 'S') {
            throw std::invalid_argument(
                "quantities must be a subset of i, j, d, D and S");
          }
        }
        rgpot::nlist::Query query;
        query.cutoff = cutoff;
        query.shape = rgpot::nlist::Pairs{full ? rgpot::nlist::Shell::Full
                                               : rgpot::nlist::Shell::Half};
        const bool want_d = quantities.find('d') != std::string::npos;
        const bool want_D = quantities.find('D') != std::string::npos;
        const bool want_S = quantities.find('S') != std::string::npos;
        query.fields.distances = want_d || offsets;
        query.fields.vectors = want_D;
        query.fields.shifts = want_S;
        const auto natoms = static_cast<std::size_t>(positions.shape(0));
        struct Owned {
          rgpot::nlist::Table table;
          std::vector<std::int32_t> offsets;
        };
        auto owned = std::make_shared<Owned>();
        try {
          nb::gil_scoped_release release;
          rgpot::nlist::build(owned->table, positions.data(), natoms,
                              cell.data(), query);
        } catch (const rgpot::nlist::Error &err) {
          throw std::invalid_argument(err.what());
        }
        if (offsets) {
          auto &table = owned->table;
          const auto np = table.size();
          std::vector<std::size_t> order(np);
          for (std::size_t p = 0; p < np; ++p) {
            order[p] = p;
          }
          std::sort(order.begin(), order.end(),
                    [&](std::size_t a, std::size_t b) {
                      if (table.first[a] != table.first[b]) {
                        return table.first[a] < table.first[b];
                      }
                      return table.second[a] < table.second[b];
                    });
          auto permute_i = [&](std::vector<std::int32_t> &col, int width) {
            if (col.empty()) {
              return;
            }
            std::vector<std::int32_t> next(col.size());
            for (std::size_t p = 0; p < np; ++p) {
              for (int k = 0; k < width; ++k) {
                next[static_cast<std::size_t>(width) * p +
                     static_cast<std::size_t>(k)] =
                    col[static_cast<std::size_t>(width) * order[p] +
                        static_cast<std::size_t>(k)];
              }
            }
            col.swap(next);
          };
          auto permute_d = [&](std::vector<double> &col, int width) {
            if (col.empty()) {
              return;
            }
            std::vector<double> next(col.size());
            for (std::size_t p = 0; p < np; ++p) {
              for (int k = 0; k < width; ++k) {
                next[static_cast<std::size_t>(width) * p +
                     static_cast<std::size_t>(k)] =
                    col[static_cast<std::size_t>(width) * order[p] +
                        static_cast<std::size_t>(k)];
              }
            }
            col.swap(next);
          };
          permute_i(table.first, 1);
          permute_i(table.second, 1);
          permute_i(table.shift, 3);
          permute_d(table.vector, 3);
          permute_d(table.distance, 1);
          owned->offsets.assign(natoms + 1, 0);
          for (std::size_t p = 0; p < np; ++p) {
            const auto i = static_cast<std::size_t>(table.first[p]);
            if (i < natoms) {
              owned->offsets[i + 1] += 1;
            }
          }
          for (std::size_t i = 0; i < natoms; ++i) {
            owned->offsets[i + 1] += owned->offsets[i];
          }
        }
        auto keep = [owned]() {
          auto *heap = new std::shared_ptr<Owned>(owned);
          return nb::capsule(heap, [](void *p) noexcept {
            delete static_cast<std::shared_ptr<Owned> *>(p);
          });
        };
        const auto npairs = owned->table.size();
        nb::list items;
        for (char q : quantities) {
          if (q == 'i') {
            items.append(nb::ndarray<nb::numpy, std::int32_t>(
                owned->table.first.data(), {npairs}, keep()));
          } else if (q == 'j') {
            items.append(nb::ndarray<nb::numpy, std::int32_t>(
                owned->table.second.data(), {npairs}, keep()));
          } else if (q == 'd') {
            items.append(nb::ndarray<nb::numpy, double>(
                owned->table.distance.data(), {npairs}, keep()));
          } else if (q == 'D') {
            items.append(nb::ndarray<nb::numpy, double>(
                owned->table.vector.data(), {npairs, 3}, keep()));
          } else if (q == 'S') {
            items.append(nb::ndarray<nb::numpy, std::int32_t>(
                owned->table.shift.data(), {npairs, 3}, keep()));
          }
        }
        if (offsets) {
          items.append(nb::ndarray<nb::numpy, std::int32_t>(
              owned->offsets.data(), {owned->offsets.size()}, keep()));
        }
        if (items.size() == 1) {
          return nb::object(items[0]);
        }
        PyObject *raw = PyTuple_New(static_cast<Py_ssize_t>(items.size()));
        for (Py_ssize_t k = 0; k < static_cast<Py_ssize_t>(items.size()); ++k) {
          nb::handle item = items[static_cast<size_t>(k)];
          Py_INCREF(item.ptr());
          PyTuple_SetItem(raw, k, item.ptr());
        }
        return nb::steal<nb::object>(raw);
      },
      nb::arg("positions").noconvert(), nb::arg("cell").noconvert(),
      nb::arg("cutoff"), nb::arg("quantities") = "ijd", nb::arg("full") = false,
      nb::arg("offsets") = false,
      "Neighbour arrays selected by quantities. i and j are indices, d is the "
      "distance, D is r_j - r_i with shape (npairs, 3), S is the integer shift "
      "with that shape. offsets=True appends a CSR row pointer of length "
      "natoms + 1 after sorting pairs by i, then j. "
      "`i, j, d = rgpot.neighbor_list(positions, cell, cutoff)`.");

  auto lammps = nb::class_<rgpot::LammpsPot>(m, "LammpsPot");
  lammps.def(
      "__call__",
      [](rgpot::LammpsPot &self, const F64 &positions, nb::handle atom_types,
         const F64 &box) {
        auto geom = rgpot::pybind::read_geom(positions, atom_types, box);
        return rgpot::pybind::call_detached(self, geom);
      },
      nb::arg("positions").noconvert(), nb::arg("atom_types"),
      nb::arg("box").noconvert(),
      "Energy, forces and variance. Positions are Angstrom, energy is eV.");
  auto gromacs = nb::class_<rgpot::GromacsPot>(m, "GromacsPot");
  gromacs.def(
      "__call__",
      [](rgpot::GromacsPot &self, const F64 &positions, nb::handle atom_types,
         const F64 &box) {
        auto geom = rgpot::pybind::read_geom(positions, atom_types, box);
        return rgpot::pybind::call_detached(self, geom);
      },
      nb::arg("positions").noconvert(), nb::arg("atom_types"),
      nb::arg("box").noconvert(),
      "Energy, forces and variance. Positions are Angstrom, energy is eV.");
  gromacs.def("native_u0", &rgpot::GromacsPot::nativeU0,
              "Well depth stored for GROMACS, kJ/mol.");
  gromacs.def("native_cutoff", &rgpot::GromacsPot::nativeCutoff,
              "Cutoff stored for GROMACS, nanometres.");
  gromacs.def("native_psi", &rgpot::GromacsPot::nativePsi,
              "Zero crossing stored for GROMACS, nanometres.");
  auto openmm = nb::class_<rgpot::OpenMMPot>(m, "OpenMMPot");
  openmm.def(
      "__call__",
      [](rgpot::OpenMMPot &self, const F64 &positions, nb::handle atom_types,
         const F64 &box) {
        auto geom = rgpot::pybind::read_geom(positions, atom_types, box);
        return rgpot::pybind::call_detached(self, geom);
      },
      nb::arg("positions").noconvert(), nb::arg("atom_types"),
      nb::arg("box").noconvert(),
      "Energy, forces and variance. Positions are Angstrom, energy is eV.");
  openmm.def("native_u0", &rgpot::OpenMMPot::nativeU0,
             "Well depth stored for OpenMM, kJ/mol.");
  openmm.def("native_cutoff", &rgpot::OpenMMPot::nativeCutoff,
             "Cutoff stored for OpenMM, nanometres.");
  openmm.def("native_psi", &rgpot::OpenMMPot::nativePsi,
             "Zero crossing stored for OpenMM, nanometres.");

  auto make_cfg = [](double u0, double cutoff, double psi,
                     const std::string &engine_path) {
    rgpot::ExternalLjConfig cfg;
    cfg.u0 = u0;
    cfg.cutoff = cutoff;
    cfg.psi = psi;
    cfg.engine_path = engine_path;
    return cfg;
  };
  m.def(
      "rgpot_lammps_potential",
      [&](double u0, double cutoff, double psi,
          const std::string &engine_path) {
        return rgpot::rgpot_lammps_potential(
                   make_cfg(u0, cutoff, psi, engine_path))
            .release();
      },
      nb::rv_policy::take_ownership,
      nb::arg("u0") = 1.0, nb::arg("cutoff") = 8.0, nb::arg("psi") = 1.0,
      nb::arg("engine_path") = "",
      "Shifted 12-6 Lennard-Jones. The constructor dlopens an engine "
      "plugin, which stays the backend. engine_path empty searches "
      "RGPOT_LAMMPS_ENGINE, LAMMPS_ENGINE, liblammps_engine.so, then "
      "liblj_engine.so. u0 is eV; cutoff and psi are Angstrom.");
  m.def(
      "rgpot_gromacs_potential",
      [&](double u0, double cutoff, double psi,
          const std::string &engine_path) {
        return rgpot::rgpot_gromacs_potential(
                   make_cfg(u0, cutoff, psi, engine_path))
            .release();
      },
      nb::rv_policy::take_ownership,
      nb::arg("u0") = 1.0, nb::arg("cutoff") = 8.0, nb::arg("psi") = 1.0,
      nb::arg("engine_path") = "",
      "Shifted 12-6 Lennard-Jones. Nanometres and kJ/mol are stored once. "
      "An empty engine_path evaluates in process. A path is a library "
      "whose rgpot_md_native_force uses those units.");
  m.def(
      "rgpot_openmm_potential",
      [&](double u0, double cutoff, double psi,
          const std::string &engine_path) {
        return rgpot::rgpot_openmm_potential(
                   make_cfg(u0, cutoff, psi, engine_path))
            .release();
      },
      nb::rv_policy::take_ownership,
      nb::arg("u0") = 1.0, nb::arg("cutoff") = 8.0, nb::arg("psi") = 1.0,
      nb::arg("engine_path") = "",
      "Shifted 12-6 Lennard-Jones. Nanometres and kJ/mol are stored once. "
      "An empty engine_path evaluates in process. A path is a library "
      "whose rgpot_md_native_force uses those units, such as "
      "libopenmm_engine.so.");
}
