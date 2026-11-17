// MIT License
// Copyright 2023--present rgpot developers

#include "pot_spec.hpp"

#include "rgpot/EAM/EAMPot.hpp"
#include "rgpot/EMT/EMTPot.hpp"
#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/Morse/MorsePot.hpp"
#include "rgpot/StillingerWeber/SWSi.hpp"
#include "rgpot/SuttonChen/SuttonChenScales.hpp"
#include "rgpot/Tersoff/TersoffSiC.hpp"
#include "rgpot/elements.hpp"
#ifdef RGPOT_HAS_EAM_TABLE
#include "rgpot/EAMTable/EAMTablePot.hpp"
#endif
#ifdef RGPOT_HAS_FORTRAN_POTS
#include "rgpot/fortran/FortranPots.hpp"
#endif

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace rgpot::spec {

namespace {

std::string lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return s;
}

/// Options of one potential. Every key must be read once, so a key the
/// potential does not take is reported instead of ignored.
class Reader {
public:
  Reader(const std::string &name, const Options &options)
      : name_(name), options_(options), used_(options.size(), false) {}

  const std::string *find(const std::string &key) {
    const std::string *hit = nullptr;
    for (std::size_t k = 0; k < options_.size(); ++k) {
      if (lower(options_[k].first) == key) {
        if (hit != nullptr) {
          throw std::invalid_argument(name_ + ": " + key + " is given twice");
        }
        hit = &options_[k].second;
        used_[k] = true;
      }
    }
    return hit;
  }

  double real(const std::string &key, double fallback) {
    const std::string *v = find(key);
    if (v == nullptr) {
      return fallback;
    }
    std::size_t end = 0;
    double out = 0.0;
    try {
      out = std::stod(*v, &end);
    } catch (const std::exception &) {
      end = 0;
    }
    if (end == 0 || end != v->size()) {
      throw std::invalid_argument(name_ + ": " + key +
                                  " is not a number: " + *v);
    }
    return out;
  }

  int integer(const std::string &key, int fallback) {
    const double v = real(key, fallback);
    if (v != static_cast<double>(static_cast<int>(v))) {
      throw std::invalid_argument(name_ + ": " + key + " must be an integer");
    }
    return static_cast<int>(v);
  }

  std::string text(const std::string &key, const std::string &fallback) {
    const std::string *v = find(key);
    return v == nullptr ? fallback : *v;
  }

  void finish() const {
    for (std::size_t k = 0; k < options_.size(); ++k) {
      if (!used_[k]) {
        throw std::invalid_argument(name_ + " takes no option " +
                                    options_[k].first);
      }
    }
  }

private:
  std::string name_;
  const Options &options_;
  std::vector<bool> used_;
};

#ifdef RGPOT_HAS_EAM_TABLE
EAMTableFormat tableFormat(const std::string &format, const std::string &path) {
  const std::string f = lower(format);
  if (f == "funcfl" || f == "eam") {
    return EAMTableFormat::Funcfl;
  }
  if (f == "setfl" || f == "alloy" || f == "eam/alloy") {
    return EAMTableFormat::Setfl;
  }
  if (f == "fs" || f == "eam/fs") {
    return EAMTableFormat::FinnisSinclair;
  }
  if (!f.empty()) {
    throw std::invalid_argument(
        "EAMTable: format is funcfl, setfl or fs, not " + format);
  }
  auto ends = [&](const std::string &tail) {
    return path.size() >= tail.size() &&
           lower(path.substr(path.size() - tail.size())) == tail;
  };
  if (ends(".eam.alloy") || ends(".setfl")) {
    return EAMTableFormat::Setfl;
  }
  if (ends(".eam.fs")) {
    return EAMTableFormat::FinnisSinclair;
  }
  if (ends(".eam")) {
    return EAMTableFormat::Funcfl;
  }
  throw std::invalid_argument("EAMTable: give format funcfl, setfl or fs for " +
                              path);
}
#endif

template <typename P, typename... A> Built own(A &&...args) {
  auto pot = std::make_unique<P>(std::forward<A>(args)...);
  const P *kernel = pot.get();
  Built out;
  out.eval = [kernel](const ForceInput &in, ForceOut *res) {
    kernel->forceImpl(in, res);
  };
  if constexpr (PerAtomPotential<P>) {
    out.perAtom = [kernel](const ForceInput &in, ForceOut *res,
                           const PerAtomOut &atoms) {
      kernel->forcePerAtom(in, res, atoms);
    };
  }
  if constexpr (LocalPotential<P>) {
    out.locality = kernel->locality();
  }
  if constexpr (StagedPotential<P>) {
    out.densityStage = [kernel](const HalfList &list, double *densities) {
      kernel->densityStage(list, densities);
    };
    out.forceStage = [kernel](const HalfList &list, const double *densities,
                              double *forces, StageTally *tally) {
      kernel->forceStage(list, densities, forces, tally);
    };
  }
  out.pot = std::move(pot);
  return out;
}

template <typename P, typename C> Built ownCut(const C &c) {
  Built out = own<P>(c);
  out.cutoff = c.cutoff;
  return out;
}

Built make(const std::string &key, Reader &r, double fallback) {
  if (key == "lj") {
    LJConfig c;
    c.u0 = r.real("u0", c.u0);
    c.psi = r.real("psi", c.psi);
    c.cutoff = r.real("cutoff", fallback > 0.0 ? fallback : c.cutoff);
    c.switch_width = r.real("switch_width", c.switch_width);
    return ownCut<LJPot>(c);
  }
  if (key == "morse") {
    MorseConfig c;
    c.De = r.real("de", c.De);
    c.a = r.real("a", c.a);
    c.re = r.real("re", c.re);
    c.cutoff = r.real("cutoff", fallback > 0.0 ? fallback : c.cutoff);
    c.switch_width = r.real("switch_width", c.switch_width);
    return ownCut<MorsePot>(c);
  }
  if (key == "suttonchen") {
    const std::string preset = lower(r.text("preset", ""));
    SuttonChenConfig c;
    if (preset == "platinum" || preset == "pt") {
      c = sutton_chen_platinum();
    } else if (preset == "gold" || preset == "au") {
      c = sutton_chen_gold();
    } else if (!preset.empty()) {
      throw std::invalid_argument(
          "SuttonChen: preset is platinum or gold, not " + preset);
    }
    c.epsilon = r.real("epsilon", c.epsilon);
    c.c = r.real("c", c.c);
    c.a = r.real("a", c.a);
    c.n = r.integer("n", c.n);
    c.m = r.integer("m", c.m);
    c.cutoff = r.real("cutoff", fallback > 0.0 ? fallback : c.cutoff);
    c.periodic = true;
    c.stress = true;
    return ownCut<SuttonChenPot>(c);
  }
#ifdef RGPOT_HAS_EAM_TABLE
  if (key == "eamtable") {
    EAMTableConfig c;
    c.path = r.text("file", "");
    if (c.path.empty()) {
      throw std::invalid_argument("EAMTable needs file <path>");
    }
    c.format = tableFormat(r.text("format", ""), c.path);
    c.skin = r.real("skin", c.skin);
    c.periodic = true;
    c.stress = true;
    return own<EAMTablePot>(c);
  }
#endif
  if (key == "swsi1985") {
    return own<StillingerWeberSi>();
  }
  if (key == "tersoffsi") {
    return own<TersoffSiC>();
  }
  if (key == "emt") {
    EMTConfig c;
    const std::string params = lower(r.text("parameters", "asap"));
    if (params != "asap" && params != "rasmussen") {
      throw std::invalid_argument("EMT: parameters is asap or rasmussen");
    }
    c.rasmussen = params == "rasmussen";
    return own<EMTPot>(c);
  }
  if (key == "eamcell") {
    return own<EAMPot>();
  }
#ifdef RGPOT_HAS_FORTRAN_POTS
  if (key == "swsi") {
    return own<fortranpots::SWPot>();
  }
  if (key == "edip") {
    return own<fortranpots::EDIPPot>();
  }
  if (key == "lenoskysi") {
    return own<fortranpots::LenoskyPot>();
  }
  if (key == "eamal") {
    return own<fortranpots::EAMAlPot>();
  }
  if (key == "fehe") {
    return own<fortranpots::FeHePot>();
  }
  if (key == "cuh2") {
    return own<fortranpots::CuH2Pot>();
  }
#endif
  return {};
}

} // namespace

std::string names() {
  std::string out = "LJ Morse SuttonChen";
#ifdef RGPOT_HAS_EAM_TABLE
  out += " EAMTable";
#endif
  out += " SWSi1985 TersoffSi EMT EAMCell";
#ifdef RGPOT_HAS_FORTRAN_POTS
  out += " SWSi EDIP LenoskySi EAMAl FeHe CuH2";
#endif
  return out;
}

Built build(const std::string &name, const Options &options, double cutoff) {
  Built out;
  try {
    Reader reader(name, options);
    out = make(lower(name), reader, cutoff);
    if (out.pot == nullptr) {
      out.error =
          "unknown rgpot potential " + name + "; this build has " + names();
      return out;
    }
    reader.finish();
  } catch (const std::exception &e) {
    out = Built{};
    out.error = e.what();
  }
  return out;
}

int atomicNumber(const std::string &symbol) {
  return rgpot::elements::atomicNumber(symbol);
}

} // namespace rgpot::spec
