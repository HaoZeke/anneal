// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/ExprPot/ExprPot.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <map>
#include <set>
#include <stdexcept>
#include <string_view>
#include <utility>

#include "Lepton.h"
#include "lepton/Exception.h"

#include "rgpot/ParamHash.hpp"

namespace rgpot {
namespace {

bool isAsciiIdentChar(char c, bool first) noexcept {
  if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == '_') {
    return true;
  }
  return !first && (c >= '0' && c <= '9');
}

bool isLeptonIdentifier(std::string_view name) noexcept {
  if (name.empty() || !isAsciiIdentChar(name[0], true)) {
    return false;
  }
  for (size_t i = 1; i < name.size(); ++i) {
    if (!isAsciiIdentChar(name[i], false)) {
      return false;
    }
  }
  return true;
}

bool isBlank(std::string_view s) noexcept {
  for (char c : s) {
    if (!std::isspace(static_cast<unsigned char>(c))) {
      return false;
    }
  }
  return true;
}

Reentrancy worseReentrancy(Reentrancy a, Reentrancy b) noexcept {
  return static_cast<Reentrancy>(
      std::max(static_cast<uint8_t>(a), static_cast<uint8_t>(b)));
}

} // namespace

struct ExprPot::Impl {
  struct NamedChild {
    std::string name;
    std::unique_ptr<PotentialBase> child;
    double energyValue = 0.0;
    Lepton::CompiledExpression dEnergy;
  };

  std::string expression;
  std::vector<NamedChild> terms;
  Lepton::CompiledExpression energy;
  uint64_t paramsKey = 0;
  PotCaps caps{};

  void bindVars() {
    std::map<std::string, double *> locs;
    for (auto &term : terms) {
      locs[term.name] = &term.energyValue;
    }
    energy.setVariableLocations(locs);
    for (auto &term : terms) {
      term.dEnergy.setVariableLocations(locs);
    }
  }
};

ExprPot::ExprPot(std::string expression, std::vector<Term> terms)
    : Potential(PotType::Expr), m_impl(std::make_unique<Impl>()) {
  if (isBlank(expression)) {
    throw std::invalid_argument("ExprPot expression is empty");
  }
  if (terms.empty()) {
    throw std::invalid_argument("ExprPot has zero terms");
  }

  std::set<std::string> seen;
  m_impl->terms.reserve(terms.size());
  for (auto &term : terms) {
    if (!isLeptonIdentifier(term.first)) {
      throw std::invalid_argument("ExprPot name '" + term.first +
                                  "' is not a Lepton identifier");
    }
    if (!seen.insert(term.first).second) {
      throw std::invalid_argument("ExprPot duplicate name '" + term.first +
                                  "'");
    }
    if (!term.second) {
      throw std::invalid_argument("ExprPot term '" + term.first + "' is null");
    }
    m_impl->terms.push_back(
        Impl::NamedChild{std::move(term.first), std::move(term.second), 0.0,
                         Lepton::CompiledExpression{}});
  }

  Lepton::ParsedExpression parsed;
  try {
    parsed = Lepton::Parser::parse(expression);
  } catch (const Lepton::Exception &ex) {
    throw std::invalid_argument(std::string("ExprPot parse failed: ") +
                                ex.what());
  }

  try {
    m_impl->energy = parsed.createCompiledExpression();
  } catch (const Lepton::Exception &ex) {
    throw std::invalid_argument(std::string("ExprPot compile failed: ") +
                                ex.what());
  }

  const std::set<std::string> &vars = m_impl->energy.getVariables();
  for (const auto &term : m_impl->terms) {
    if (vars.find(term.name) == vars.end()) {
      throw std::invalid_argument("ExprPot name '" + term.name +
                                  "' is missing from the expression");
    }
  }
  for (const auto &var : vars) {
    if (seen.find(var) == seen.end()) {
      throw std::invalid_argument("ExprPot identifier '" + var +
                                  "' has no term");
    }
  }

  for (auto &term : m_impl->terms) {
    try {
      term.dEnergy =
          parsed.differentiate(term.name).createCompiledExpression();
    } catch (const Lepton::Exception &ex) {
      throw std::invalid_argument(std::string("ExprPot differentiate('") +
                                  term.name + "') failed: " + ex.what());
    }
  }

  m_impl->bindVars();
  m_impl->expression = std::move(expression);

  Fnv1a fp;
  fp.u64(/*kKernelVersion=*/1);
  fp.str(m_impl->expression);
  PotCaps caps;
  caps.reentrancy = Reentrancy::PerInstance;
  caps.batched = false;
  caps.periodic = true;
  for (const auto &term : m_impl->terms) {
    fp.str(term.name);
    fp.u64(term.child->paramsKey());
    const PotCaps child = term.child->caps();
    caps.reentrancy = worseReentrancy(caps.reentrancy, child.reentrancy);
    caps.periodic = caps.periodic && child.periodic;
    caps.perImageInstances =
        caps.perImageInstances || child.perImageInstances;
  }
  m_impl->paramsKey = fp.h;
  m_impl->caps = caps;
}

ExprPot::~ExprPot() = default;

ExprPot::ExprPot(ExprPot &&other) noexcept
    : Potential(PotType::Expr), m_impl(std::move(other.m_impl)) {}

ExprPot &ExprPot::operator=(ExprPot &&other) noexcept {
  if (this != &other) {
    m_impl = std::move(other.m_impl);
  }
  return *this;
}

void ExprPot::forceImpl(const ForceInput &in, ForceOut *out) const {
  const size_t n3 = 3 * in.nAtoms;
  if (out->F != nullptr) {
    for (size_t i = 0; i < n3; ++i) {
      out->F[i] = 0.0;
    }
  }
  out->energy = 0.0;
  out->variance = 0.0;

  const size_t nTerms = m_impl->terms.size();
  std::vector<double> childForces(nTerms * n3, 0.0);
  std::vector<double> childVars(nTerms, 0.0);
  for (size_t i = 0; i < nTerms; ++i) {
    auto &term = m_impl->terms[i];
    ForceOut childOut{.F = n3 == 0 ? nullptr : childForces.data() + i * n3,
                      .energy = 0.0,
                      .variance = 0.0,
                      .stress = {},
                      .has_stress = 0};
    ForceBatch batch{.nSystems = 1, .in = &in, .out = &childOut};
    term.child->forceBatch(batch);
    term.energyValue = childOut.energy;
    childVars[i] = childOut.variance;
  }

  out->energy = m_impl->energy.evaluate();

  double accVar = 0.0;
  bool allFiniteVar = true;
  for (size_t i = 0; i < nTerms; ++i) {
    const double weight = m_impl->terms[i].dEnergy.evaluate();
    if (out->F != nullptr && n3 > 0) {
      const double *Fi = childForces.data() + i * n3;
      for (size_t k = 0; k < n3; ++k) {
        out->F[k] += weight * Fi[k];
      }
    }
    if (!std::isfinite(childVars[i])) {
      allFiniteVar = false;
    } else {
      accVar += weight * weight * childVars[i];
    }
  }
  out->variance = allFiniteVar ? accVar : 0.0;
}

PotCaps ExprPot::caps() const noexcept { return m_impl->caps; }

uint64_t ExprPot::paramsKey() const noexcept { return m_impl->paramsKey; }

const std::string &ExprPot::expression() const noexcept {
  return m_impl->expression;
}

double ExprPot::dEnergyDTerm(const std::string &name) const {
  for (const auto &term : m_impl->terms) {
    if (term.name == name) {
      return term.dEnergy.evaluate();
    }
  }
  throw std::invalid_argument("ExprPot name '" + name + "' is not a term");
}

} // namespace rgpot
