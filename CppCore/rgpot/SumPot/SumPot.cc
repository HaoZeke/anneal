// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/SumPot/SumPot.hpp"

#include "rgpot/ParamHash.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <stdexcept>
#include <string>

namespace rgpot {

namespace {

bool env_log_enabled() {
  const char *v = std::getenv("RGPOT_SUM_LOG");
  return v != nullptr && *v != '\0' && std::string(v) != "0";
}

Reentrancy weaker(Reentrancy a, Reentrancy b) {
  return static_cast<uint8_t>(a) >= static_cast<uint8_t>(b) ? a : b;
}

} // namespace

SumPot::SumPot() : Potential(PotType::LrSum), env_log_(env_log_enabled()) {}

void SumPot::add(std::unique_ptr<PotentialBase> term) {
  add(SumTerm{.name = "term" + std::to_string(terms_.size())}, std::move(term));
}

void SumPot::add(SumTerm spec, std::unique_ptr<PotentialBase> term) {
  if (!term && spec.enabled) {
    throw std::invalid_argument("SumPot cannot take a null term unless it is disabled");
  }
  if (!std::isfinite(spec.weight)) {
    throw std::invalid_argument("SumPot term weight must be finite");
  }
  if (spec.name.empty()) {
    spec.name = "term" + std::to_string(terms_.size());
  }
  terms_.push_back(Entry{std::move(spec), std::move(term)});
  resetComponents();
}

void SumPot::setEnabled(std::size_t index, bool enabled) {
  if (index >= terms_.size()) {
    throw std::out_of_range("SumPot term index out of range");
  }
  if (enabled && !terms_[index].pot) {
    throw std::logic_error("SumPot term '" + terms_[index].spec.name +
                           "' was added disabled without a potential");
  }
  terms_[index].spec.enabled = enabled;
  resetComponents();
}

void SumPot::setWeight(std::size_t index, double weight) {
  if (index >= terms_.size()) {
    throw std::out_of_range("SumPot term index out of range");
  }
  if (!std::isfinite(weight)) {
    throw std::invalid_argument("SumPot term weight must be finite");
  }
  terms_[index].spec.weight = weight;
  resetComponents();
}

const SumTerm &SumPot::term(std::size_t index) const {
  if (index >= terms_.size()) {
    throw std::out_of_range("SumPot term index out of range");
  }
  return terms_[index].spec;
}

PotentialBase *SumPot::child(std::size_t index) const {
  if (index >= terms_.size()) {
    throw std::out_of_range("SumPot term index out of range");
  }
  return terms_[index].pot.get();
}

void SumPot::setLogSink(LogSink sink) { sink_ = std::move(sink); }

void SumPot::log(std::string_view line) const {
  if (sink_) {
    sink_(line);
  } else if (env_log_) {
    std::fprintf(stderr, "%.*s\n", static_cast<int>(line.size()), line.data());
  }
}

void SumPot::resetComponents() const {
  components_.clear();
  components_.reserve(terms_.size());
  for (const auto &t : terms_) {
    SumComponent c;
    c.name = t.spec.name;
    c.weight = t.spec.weight;
    c.enabled = t.spec.enabled;
    components_.push_back(std::move(c));
  }
}

void SumPot::forceImpl(const ForceInput &in, ForceOut *out) const {
  const std::size_t n3 = 3 * in.nAtoms;
  std::fill(out->F, out->F + n3, 0.0);
  out->energy = 0.0;
  out->variance = 0.0;
  out->has_stress = 0;
  std::fill(out->stress, out->stress + 9, 0.0);
  resetComponents();
  const bool logging = sink_ || env_log_;
  bool all_stress = true;
  std::size_t evaluated = 0;
  for (std::size_t t = 0; t < terms_.size(); ++t) {
    const Entry &e = terms_[t];
    SumComponent &c = components_[t];
    if (!e.spec.enabled) {
      if (logging) {
        log("rgpot sum: " + e.spec.name + " disabled");
      }
      continue;
    }
    c.forces.assign(n3, 0.0);
    ForceOut part{};
    part.F = c.forces.data();
    try {
      e.pot->forceBatch(ForceBatch{.nSystems = 1, .in = &in, .out = &part});
    } catch (const std::exception &ex) {
      c.forces.clear();
      log("rgpot sum: " + e.spec.name + " failed: " + ex.what());
      throw;
    } catch (...) {
      c.forces.clear();
      log("rgpot sum: " + e.spec.name + " failed");
      throw;
    }
    c.evaluated = true;
    c.energy = part.energy;
    c.variance = part.variance;
    c.has_stress = part.has_stress != 0;
    std::copy(part.stress, part.stress + 9, c.stress.begin());
    ++evaluated;

    const double w = e.spec.weight;
    out->energy += w * part.energy;
    out->variance += w * w * part.variance;
    for (std::size_t i = 0; i < n3; ++i) {
      out->F[i] += w * c.forces[i];
    }
    if (part.has_stress) {
      for (int k = 0; k < 9; ++k) {
        out->stress[k] += w * part.stress[k];
      }
    } else {
      all_stress = false;
    }
    if (logging) {
      double fmax = 0.0;
      for (std::size_t a = 0; a < in.nAtoms; ++a) {
        const double *f = c.forces.data() + 3 * a;
        fmax = std::max(fmax, std::sqrt(f[0] * f[0] + f[1] * f[1] + f[2] * f[2]));
      }
      char buf[256];
      std::snprintf(buf, sizeof(buf),
                    " weight=%.17g energy=%.17g eV fmax=%.17g eV/A stress=%d",
                    w, part.energy, fmax, part.has_stress ? 1 : 0);
      log("rgpot sum: " + e.spec.name + buf);
    }
  }
  if (evaluated > 0 && all_stress) {
    out->has_stress = 1;
  } else {
    std::fill(out->stress, out->stress + 9, 0.0);
  }
  if (logging) {
    char buf[128];
    std::snprintf(buf, sizeof(buf), "rgpot sum: total energy=%.17g eV",
                  out->energy);
    log(buf);
  }
}

PotCaps SumPot::caps() const noexcept {
  PotCaps c{.reentrancy = Reentrancy::PerInstance,
            .perImageInstances = false,
            .batched = false,
            .periodic = true,
            .groupCollective = false,
            .stress = true};
  bool enabled = false;
  for (const auto &t : terms_) {
    if (!t.spec.enabled) {
      continue;
    }
    enabled = true;
    const PotCaps k = t.pot->caps();
    c.reentrancy = weaker(c.reentrancy, k.reentrancy);
    c.perImageInstances = c.perImageInstances || k.perImageInstances;
    c.periodic = c.periodic && k.periodic;
    c.groupCollective = c.groupCollective || k.groupCollective;
    c.stress = c.stress && k.stress;
  }
  // No enabled term means the call reports no stress. Advertise the same.
  if (!enabled) {
    c.stress = false;
  }
  return c;
}

uint64_t SumPot::paramsKey() const noexcept {
  Fnv1a h;
  h.str("rgpot::SumPot");
  for (const auto &t : terms_) {
    h.str(t.spec.name);
    h.f64(t.spec.weight);
    h.b8(t.spec.enabled);
    if (t.pot) {
      h.u64(static_cast<uint64_t>(t.pot->getType()));
      h.u64(t.pot->paramsKey());
    }
  }
  return h.h;
}

} // namespace rgpot
