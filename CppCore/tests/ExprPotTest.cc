// MIT License
// Copyright 2023--present rgpot developers
//
// ExprPot construct-time name checks, identity energy vs LJPot, and
// analytic Lepton chain-rule forces.

#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <catch2/catch_all.hpp>

#include "TestFixtures.hpp"
#include "rgpot/ExprPot/ExprPot.hpp"
#include "rgpot/LennardJones/LJClusterPot.hpp"
#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/Morse/MorsePot.hpp"
#include "rgpot/types/AtomMatrix.hpp"

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using rgpot::test_fixtures::nebMorseCell;
using rgpot::test_fixtures::nebMorseCluster;
using rgpot::types::AtomMatrix;

namespace {

AtomMatrix twoAtomPositions() {
  return AtomMatrix{{0.0, 0.0, 0.0}, {1.5, 0.0, 0.0}};
}

std::vector<int> twoAtomTypes() { return {18, 18}; }

std::array<std::array<double, 3>, 3> wideCell() {
  return {{{40.0, 0.0, 0.0}, {0.0, 40.0, 0.0}, {0.0, 0.0, 40.0}}};
}

std::vector<rgpot::ExprPot::Term> oneLj(const char *name = "lj") {
  std::vector<rgpot::ExprPot::Term> terms;
  terms.emplace_back(name, std::make_unique<rgpot::LJPot>());
  return terms;
}

std::vector<rgpot::ExprPot::Term> ljAndMorse() {
  std::vector<rgpot::ExprPot::Term> terms;
  terms.emplace_back("lj", std::make_unique<rgpot::LJPot>());
  terms.emplace_back("morse", std::make_unique<rgpot::MorsePot>());
  return terms;
}

double maxAbsForceDiff(const AtomMatrix &a, const AtomMatrix &b) {
  REQUIRE(a.rows() == b.rows());
  REQUIRE(a.cols() == b.cols());
  double worst = 0.0;
  for (size_t i = 0; i < a.rows(); ++i) {
    for (size_t j = 0; j < a.cols(); ++j) {
      worst = std::max(worst, std::abs(a(i, j) - b(i, j)));
    }
  }
  return worst;
}

AtomMatrix scaleAdd(double wa, const AtomMatrix &a, double wb,
                    const AtomMatrix &b) {
  AtomMatrix out(a.rows(), a.cols());
  for (size_t i = 0; i < a.rows(); ++i) {
    for (size_t j = 0; j < a.cols(); ++j) {
      out(i, j) = wa * a(i, j) + wb * b(i, j);
    }
  }
  return out;
}

class StubPot : public rgpot::Potential<StubPot> {
public:
  StubPot(double energy, double variance, AtomMatrix forces)
      : Potential(rgpot::PotType::UNKNOWN),
        energy_(energy),
        variance_(variance),
        forces_(std::move(forces)) {}

  void forceImpl(const rgpot::ForceInput &in, rgpot::ForceOut *out) const override {
    out->energy = energy_;
    out->variance = variance_;
    if (out->F != nullptr && in.nAtoms > 0) {
      std::memcpy(out->F, forces_.data(), 3 * in.nAtoms * sizeof(double));
    }
  }

  [[nodiscard]] uint64_t paramsKey() const noexcept override { return key_; }
  void setParamsKey(uint64_t k) { key_ = k; }

private:
  double energy_;
  double variance_;
  AtomMatrix forces_;
  uint64_t key_{0};
};

} // namespace

TEST_CASE("ExprPot identity energy matches LJPot on two atoms", "[expr]") {
  rgpot::LJPot lj;
  const auto positions = twoAtomPositions();
  const auto atmtypes = twoAtomTypes();
  const auto box = wideCell();

  auto [e_lj, f_lj, v_lj] = lj(positions, atmtypes, box);
  (void)v_lj;

  rgpot::ExprPot pot("lj", oneLj());
  REQUIRE(pot.getType() == rgpot::PotType::Expr);
  REQUIRE(pot.expression() == "lj");

  auto [e_expr, f_expr, v_expr] = pot(positions, atmtypes, box);
  (void)v_expr;

  REQUIRE_THAT(e_expr, WithinAbs(e_lj, 1e-14));
  REQUIRE_THAT(e_expr, WithinRel(e_lj, 1e-14));
  REQUIRE_THAT(maxAbsForceDiff(f_expr, f_lj), WithinAbs(0.0, 1e-12));
}

TEST_CASE("ExprPot rejects bad names before any force call", "[expr]") {
  SECTION("empty expression") {
    REQUIRE_THROWS_AS(rgpot::ExprPot("", oneLj()), std::invalid_argument);
    REQUIRE_THROWS_AS(rgpot::ExprPot("   ", oneLj()), std::invalid_argument);
  }
  SECTION("zero terms") {
    REQUIRE_THROWS_AS(rgpot::ExprPot("lj", {}), std::invalid_argument);
  }
  SECTION("duplicate name") {
    std::vector<rgpot::ExprPot::Term> terms;
    terms.emplace_back("lj", std::make_unique<rgpot::LJPot>());
    terms.emplace_back("lj", std::make_unique<rgpot::LJPot>());
    REQUIRE_THROWS_AS(rgpot::ExprPot("lj", std::move(terms)),
                      std::invalid_argument);
  }
  SECTION("name is not a Lepton identifier") {
    REQUIRE_THROWS_AS(rgpot::ExprPot("x", oneLj("1lj")), std::invalid_argument);
    REQUIRE_THROWS_AS(rgpot::ExprPot("x", oneLj("lj-x")), std::invalid_argument);
    REQUIRE_THROWS_AS(rgpot::ExprPot("x", oneLj("lj.x")), std::invalid_argument);
  }
  SECTION("name missing from the expression") {
    std::vector<rgpot::ExprPot::Term> terms = oneLj("lj");
    terms.emplace_back("unused", std::make_unique<rgpot::LJPot>());
    REQUIRE_THROWS_AS(rgpot::ExprPot("lj", std::move(terms)),
                      std::invalid_argument);
  }
  SECTION("identifier in the expression with no term") {
    REQUIRE_THROWS_AS(rgpot::ExprPot("lj + morse", oneLj()),
                      std::invalid_argument);
  }
}

TEST_CASE("ExprPot nested identity still matches LJPot energy", "[expr]") {
  std::vector<rgpot::ExprPot::Term> inner_terms = oneLj();
  auto inner = std::make_unique<rgpot::ExprPot>("lj", std::move(inner_terms));
  std::vector<rgpot::ExprPot::Term> outer_terms;
  outer_terms.emplace_back("inner", std::move(inner));
  rgpot::ExprPot pot("inner", std::move(outer_terms));

  rgpot::LJPot lj;
  const auto positions = twoAtomPositions();
  const auto atmtypes = twoAtomTypes();
  const auto box = wideCell();
  auto [e_lj, f_lj, v_lj] = lj(positions, atmtypes, box);
  auto [e_expr, f_expr, v_expr] = pot(positions, atmtypes, box);
  (void)v_lj;
  (void)v_expr;
  REQUIRE_THAT(e_expr, WithinAbs(e_lj, 1e-14));
  REQUIRE_THAT(maxAbsForceDiff(f_expr, f_lj), WithinAbs(0.0, 1e-12));
}

TEST_CASE("ExprPot 0.5*lj + morse matches independent chain-rule forces",
          "[expr]") {
  rgpot::LJPot lj;
  rgpot::MorsePot morse;
  rgpot::ExprPot pot("0.5*lj + morse", ljAndMorse());

  REQUIRE_THAT(pot.dEnergyDTerm("lj"), WithinAbs(0.5, 0.0));
  REQUIRE_THAT(pot.dEnergyDTerm("morse"), WithinAbs(1.0, 0.0));

  const auto check = [&](const AtomMatrix &positions,
                         const std::vector<int> &atmtypes,
                         const std::array<std::array<double, 3>, 3> &box) {
    auto [e_lj, f_lj, v_lj] = lj(positions, atmtypes, box);
    auto [e_m, f_m, v_m] = morse(positions, atmtypes, box);
    auto [e_expr, f_expr, v_expr] = pot(positions, atmtypes, box);
    (void)v_lj;
    (void)v_m;
    (void)v_expr;

    REQUIRE_THAT(e_expr, WithinAbs(0.5 * e_lj + e_m, 1e-14));
    const AtomMatrix expected = scaleAdd(0.5, f_lj, 1.0, f_m);
    REQUIRE_THAT(maxAbsForceDiff(f_expr, expected), WithinAbs(0.0, 1e-12));
    REQUIRE_THAT(pot.dEnergyDTerm("lj"), WithinAbs(0.5, 0.0));
  };

  SECTION("two-atom LJ unit fixture") {
    check(twoAtomPositions(), twoAtomTypes(), wideCell());
  }
  SECTION("neb_morse Morse/LJ unit fixture") {
    check(nebMorseCluster(), std::vector<int>(13, 1), nebMorseCell());
  }
}

TEST_CASE("ExprPot 2*lj - lj forces match LJPot", "[expr]") {
  rgpot::LJPot lj;
  const auto positions = twoAtomPositions();
  const auto atmtypes = twoAtomTypes();
  const auto box = wideCell();

  auto [e_lj, f_lj, v_lj] = lj(positions, atmtypes, box);
  (void)v_lj;

  rgpot::ExprPot pot("2*lj - lj", oneLj());
  auto [e_expr, f_expr, v_expr] = pot(positions, atmtypes, box);
  (void)v_expr;

  REQUIRE_THAT(e_expr, WithinAbs(e_lj, 1e-14));
  REQUIRE_THAT(maxAbsForceDiff(f_expr, f_lj), WithinAbs(0.0, 1e-12));
  REQUIRE_THAT(pot.dEnergyDTerm("lj"), WithinAbs(1.0, 0.0));
}

TEST_CASE("ExprPot product uses child energies in analytic weights", "[expr]") {
  rgpot::LJPot lj;
  rgpot::MorsePot morse;
  rgpot::ExprPot pot("lj * morse", ljAndMorse());

  const auto positions = twoAtomPositions();
  const auto atmtypes = twoAtomTypes();
  const auto box = wideCell();
  auto [e_lj, f_lj, v_lj] = lj(positions, atmtypes, box);
  auto [e_m, f_m, v_m] = morse(positions, atmtypes, box);
  auto [e_expr, f_expr, v_expr] = pot(positions, atmtypes, box);
  (void)v_lj;
  (void)v_m;
  (void)v_expr;

  REQUIRE_THAT(e_expr, WithinAbs(e_lj * e_m, 1e-14));
  REQUIRE_THAT(pot.dEnergyDTerm("lj"), WithinAbs(e_m, 1e-14));
  REQUIRE_THAT(pot.dEnergyDTerm("morse"), WithinAbs(e_lj, 1e-14));
  const AtomMatrix expected = scaleAdd(e_m, f_lj, e_lj, f_m);
  REQUIRE_THAT(maxAbsForceDiff(f_expr, expected), WithinAbs(0.0, 1e-12));
}

TEST_CASE("ExprPot variance scales with the square of the analytic weights",
          "[expr]") {
  const AtomMatrix f_a{{1.0, 0.0, 0.0}, {-1.0, 0.0, 0.0}};
  const AtomMatrix f_b{{0.0, 2.0, 0.0}, {0.0, -2.0, 0.0}};
  const auto positions = twoAtomPositions();
  const auto atmtypes = twoAtomTypes();
  const auto box = wideCell();

  SECTION("finite child variances") {
    std::vector<rgpot::ExprPot::Term> terms;
    terms.emplace_back("a", std::make_unique<StubPot>(1.0, 1.5, f_a));
    terms.emplace_back("b", std::make_unique<StubPot>(2.0, 4.0, f_b));
    rgpot::ExprPot pot("2*a + b", std::move(terms));
    auto [e, f, v] = pot(positions, atmtypes, box);
    REQUIRE_THAT(e, WithinAbs(4.0, 0.0));
    REQUIRE_THAT(v, WithinAbs(2.0 * 2.0 * 1.5 + 1.0 * 1.0 * 4.0, 0.0));
    REQUIRE_THAT(f(0, 0), WithinAbs(2.0, 0.0));
    REQUIRE_THAT(f(0, 1), WithinAbs(2.0, 0.0));
  }
  SECTION("a fractional and a negative weight") {
    std::vector<rgpot::ExprPot::Term> terms;
    terms.emplace_back("a", std::make_unique<StubPot>(1.0, 1.5, f_a));
    terms.emplace_back("b", std::make_unique<StubPot>(2.0, 4.0, f_b));
    rgpot::ExprPot pot("0.5*a - 3*b", std::move(terms));
    auto [e, f, v] = pot(positions, atmtypes, box);
    (void)e;
    (void)f;
    REQUIRE_THAT(v, WithinAbs(0.5 * 0.5 * 1.5 + 3.0 * 3.0 * 4.0, 1e-12));
    REQUIRE(v > 0.0);
  }
  SECTION("a nonlinear expression weights by the derivative at the energies") {
    std::vector<rgpot::ExprPot::Term> terms;
    terms.emplace_back("a", std::make_unique<StubPot>(3.0, 1.5, f_a));
    terms.emplace_back("b", std::make_unique<StubPot>(2.0, 4.0, f_b));
    rgpot::ExprPot pot("a*b", std::move(terms));
    auto [e, f, v] = pot(positions, atmtypes, box);
    (void)e;
    (void)f;
    REQUIRE_THAT(pot.dEnergyDTerm("a"), WithinAbs(2.0, 0.0));
    REQUIRE_THAT(pot.dEnergyDTerm("b"), WithinAbs(3.0, 0.0));
    REQUIRE_THAT(v, WithinAbs(2.0 * 2.0 * 1.5 + 3.0 * 3.0 * 4.0, 1e-12));
  }
  SECTION("non-finite child variance is zero") {
    std::vector<rgpot::ExprPot::Term> terms;
    terms.emplace_back("a", std::make_unique<StubPot>(1.0, 1.5, f_a));
    terms.emplace_back(
        "b", std::make_unique<StubPot>(
                 2.0, std::numeric_limits<double>::quiet_NaN(), f_b));
    rgpot::ExprPot pot("2*a + b", std::move(terms));
    auto [e, f, v] = pot(positions, atmtypes, box);
    (void)e;
    (void)f;
    REQUIRE_THAT(v, WithinAbs(0.0, 0.0));
  }
}

TEST_CASE("ExprPot caps stay at least PerInstance and AND periodic", "[expr]") {
  rgpot::ExprPot periodic("0.5*lj + morse", ljAndMorse());
  const auto caps = periodic.caps();
  REQUIRE(caps.reentrancy == rgpot::Reentrancy::PerInstance);
  REQUIRE(caps.periodic);
  REQUIRE_FALSE(caps.batched);

  std::vector<rgpot::ExprPot::Term> mixed;
  mixed.emplace_back("lj", std::make_unique<rgpot::LJPot>());
  mixed.emplace_back("clus", std::make_unique<rgpot::LJClusterPot>());
  rgpot::ExprPot mixedPot("lj + clus", std::move(mixed));
  REQUIRE_FALSE(mixedPot.caps().periodic);
  REQUIRE(mixedPot.caps().reentrancy == rgpot::Reentrancy::PerInstance);
  REQUIRE_FALSE(mixedPot.caps().batched);
}

TEST_CASE("ExprPot paramsKey hashes expression and child keys", "[expr]") {
  rgpot::ExprPot a("0.5*lj + morse", ljAndMorse());
  rgpot::ExprPot b("0.5*lj + morse", ljAndMorse());
  REQUIRE(a.paramsKey() == b.paramsKey());

  rgpot::ExprPot scaled("lj + morse", ljAndMorse());
  REQUIRE(a.paramsKey() != scaled.paramsKey());

  std::vector<rgpot::ExprPot::Term> deep;
  deep.emplace_back("lj", std::make_unique<rgpot::LJPot>());
  deep.emplace_back("morse",
                    std::make_unique<rgpot::MorsePot>(rgpot::MorseConfig{.De = 0.8}));
  rgpot::ExprPot deeper("0.5*lj + morse", std::move(deep));
  REQUIRE(a.paramsKey() != deeper.paramsKey());
}
