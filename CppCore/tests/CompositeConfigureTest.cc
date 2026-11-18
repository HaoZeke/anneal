// MIT License
// Copyright 2023--present rgpot developers
//
// PotentialConfig.composite: the wire arm builds a SumPot whose terms carry
// their names, weights and switches, a disabled term is never constructed,
// configure on a composite backend rebuilds the sum, and every other arm is
// refused by name.

#include <array>
#include <cmath>
#include <string>
#include <vector>

#include <capnp/message.h>
#include <capnp/schema.h>
#include <capnp/serialize.h>
#include <catch2/catch_all.hpp>

#include "rgpot/D2Pot/D2CpmdPot.hpp"
#include "rgpot/D2Pot/D2Pot.hpp"
#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/Morse/MorsePot.hpp"
#include "rgpot/rpc/CompositeConfigure.hpp"
#include "rgpot/types/AtomMatrix.hpp"

using Catch::Matchers::ContainsSubstring;
using rgpot::types::AtomMatrix;

namespace {

const AtomMatrix kPositions{{0.0, 0.0, 0.0}, {1.1, 0.0, 0.0}, {0.0, 1.2, 0.0}};
const std::vector<int> kZ{1, 1, 1};
const std::array<std::array<double, 3>, 3> kBox{
    {{20.0, 0.0, 0.0}, {0.0, 20.0, 0.0}, {0.0, 0.0, 20.0}}};

void setLj(::CompositeTerm::Builder term, const std::string &name, double weight,
           double u0) {
  term.setName(name);
  term.setWeight(weight);
  auto b = term.initPot().initBuiltin();
  b.setName("lj");
  auto lj = b.initConfig().initLj();
  lj.setU0(u0);
  lj.setPsi(1.0);
  lj.setCutoff(6.0);
}

double ljEnergy(double u0) {
  rgpot::LJPot pot(rgpot::LJConfig{.u0 = u0, .cutoff = 6.0, .psi = 1.0});
  auto [e, f, v] = pot(kPositions, kZ, kBox);
  (void)f;
  (void)v;
  return e;
}

} // namespace

TEST_CASE("PotentialConfig.composite sits at ordinal 18", "[composite][rpc]") {
  const auto field = capnp::Schema::from<::PotentialConfig>().getFieldByName("composite");
  REQUIRE(field.getProto().getOrdinal().getExplicit() == 18);
  capnp::MallocMessageBuilder builder;
  auto term = builder.initRoot<::CompositeTerm>();
  REQUIRE(term.getEnabled());
  REQUIRE(term.getWeight() == 1.0);
}

TEST_CASE("the composite arm builds the weighted sum it names", "[composite][rpc]") {
  capnp::MallocMessageBuilder builder;
  auto cfg = builder.initRoot<::PotentialConfig>();
  auto terms = cfg.initComposite().initTerms(3);
  setLj(terms[0], "strong", 2.0, 1.0);
  setLj(terms[1], "weak", -1.0, 0.25);
  terms[2].setName("engine");
  terms[2].setEnabled(false);
  // A disabled engine term is recorded but never built, so no cpmdc library
  // is needed here.
  terms[2].initPot().initCpmd().setEnginePath("absent-libcpmdc.so");

  auto result = rgpot::rpc::configureComposite(cfg.asReader());
  INFO(result.message);
  REQUIRE(result.ok);
  REQUIRE(result.replacement);
  REQUIRE_THAT(result.message, ContainsSubstring("strong(weight=2)"));
  REQUIRE_THAT(result.message, ContainsSubstring("engine(weight=1, disabled)"));
  rgpot::SumPot &sum = *result.replacement;
  REQUIRE(sum.size() == 3);
  REQUIRE(sum.child(2) == nullptr);
  REQUIRE(sum.term(1).weight == -1.0);

  auto [e, f, v] = sum(kPositions, kZ, kBox);
  (void)f;
  (void)v;
  const double expected = 2.0 * ljEnergy(1.0) - ljEnergy(0.25);
  REQUIRE(e == Catch::Approx(expected).margin(1e-12 * std::abs(expected)));
  const auto &parts = sum.lastComponents();
  REQUIRE(parts.size() == 3);
  REQUIRE(parts[0].name == "strong");
  REQUIRE(parts[1].energy == Catch::Approx(ljEnergy(0.25)));
  REQUIRE_FALSE(parts[2].evaluated);
  REQUIRE_THROWS_AS(sum.setEnabled(2, true), std::logic_error);
}

TEST_CASE("composite terms nest and take the other builtins", "[composite][rpc]") {
  capnp::MallocMessageBuilder builder;
  auto cfg = builder.initRoot<::PotentialConfig>();
  auto terms = cfg.initComposite().initTerms(2);
  terms[0].setName("morse");
  auto morse = terms[0].initPot().initBuiltin();
  morse.setName("Morse");
  morse.initConfig().initMorse().setDe(0.5);
  terms[1].setName("inner");
  terms[1].setWeight(0.5);
  auto inner = terms[1].initPot().initBuiltin();
  inner.setName("composite");
  setLj(inner.initConfig().initComposite().initTerms(1)[0], "lj", 1.0, 1.0);

  auto result = rgpot::rpc::configureComposite(cfg.asReader());
  INFO(result.message);
  REQUIRE(result.ok);
  auto [e, f, v] = (*result.replacement)(kPositions, kZ, kBox);
  (void)f;
  (void)v;
  rgpot::MorseConfig mc;
  mc.De = 0.5;
  rgpot::MorsePot direct(mc);
  auto [me, mf, mv] = direct(kPositions, kZ, kBox);
  (void)mf;
  (void)mv;
  const double expected = me + 0.5 * ljEnergy(1.0);
  REQUIRE(e == Catch::Approx(expected).margin(1e-12 * std::abs(expected)));
}

TEST_CASE("the composite backend refuses what it cannot build", "[composite][rpc]") {
  {
    capnp::MallocMessageBuilder builder;
    auto cfg = builder.initRoot<::PotentialConfig>();
    REQUIRE(rgpot::rpc::configureComposite(cfg.asReader()).ok);
    REQUIRE_FALSE(rgpot::rpc::configureComposite(cfg.asReader()).replacement);
    cfg.initMorse();
    auto refused = rgpot::rpc::configureComposite(cfg.asReader());
    REQUIRE_FALSE(refused.ok);
    REQUIRE_THAT(refused.message, ContainsSubstring("morse"));
  }
  {
    capnp::MallocMessageBuilder builder;
    auto cfg = builder.initRoot<::PotentialConfig>();
    auto term = cfg.initComposite().initTerms(1)[0];
    term.setName("bad");
    auto b = term.initPot().initBuiltin();
    b.setName("lj");
    b.initConfig().initMorse();
    auto refused = rgpot::rpc::configureComposite(cfg.asReader());
    REQUIRE_FALSE(refused.ok);
    REQUIRE_THAT(refused.message, ContainsSubstring("'bad'"));
    REQUIRE_THAT(refused.message, ContainsSubstring("morse config arm"));
  }
  {
    capnp::MallocMessageBuilder builder;
    auto cfg = builder.initRoot<::PotentialConfig>();
    cfg.initComposite().initTerms(1)[0].setName("empty");
    auto refused = rgpot::rpc::configureComposite(cfg.asReader());
    REQUIRE_FALSE(refused.ok);
    REQUIRE_THAT(refused.message, ContainsSubstring("'empty' has no pot"));
  }
  {
    capnp::MallocMessageBuilder builder;
    auto cfg = builder.initRoot<::PotentialConfig>();
    auto term = cfg.initComposite().initTerms(1)[0];
    term.setName("nan");
    term.setWeight(std::nan(""));
    term.initPot().initBuiltin().setName("lj");
    REQUIRE_FALSE(rgpot::rpc::configureComposite(cfg.asReader()).ok);
  }
}

TEST_CASE("an LJ plus D2 composite is built from the config path", "[composite][rpc][d2]") {
  capnp::MallocMessageBuilder builder;
  auto cfg = builder.initRoot<::PotentialConfig>();
  auto terms = cfg.initComposite().initTerms(3);
  setLj(terms[0], "lj", 1.0, 1.0);
  terms[1].setName("d2");
  auto d2 = terms[1].initPot().initBuiltin();
  d2.setName("d2");
  d2.initConfig().initD2().setCutoff(40.0);
  terms[2].setName("d2 again");
  terms[2].setWeight(-1.0);
  auto again = terms[2].initPot().initBuiltin();
  again.setName("d2");
  again.initConfig().initD2().setCutoff(40.0);

  auto result = rgpot::rpc::configureComposite(cfg.asReader());
  INFO(result.message);
  REQUIRE(result.ok);
  rgpot::SumPot &sum = *result.replacement;
  REQUIRE(sum.caps().stress);
  auto [e, f, v] = sum(kPositions, kZ, kBox);
  (void)f;
  (void)v;
  REQUIRE(e == Catch::Approx(ljEnergy(1.0)).margin(1e-12));

  rgpot::D2Config dc;
  dc.cutoff = 40.0;
  rgpot::D2Pot direct(dc);
  auto [de, df, dv] = direct(kPositions, kZ, kBox);
  (void)df;
  (void)dv;
  const auto &parts = sum.lastComponents();
  REQUIRE(de < 0.0);
  REQUIRE(parts[1].energy == Catch::Approx(de).margin(1e-14));
  REQUIRE(parts[2].energy == Catch::Approx(de).margin(1e-14));
}

TEST_CASE("the d2 arm configures a D2 backend", "[rpc][d2]") {
  REQUIRE(capnp::Schema::from<::PotentialConfig>().getFieldByName("d2").getProto().getOrdinal().getExplicit() == 19);
  capnp::MallocMessageBuilder builder;
  auto cfg = builder.initRoot<::PotentialConfig>();
  auto params = cfg.initD2();
  REQUIRE(params.getS6() == 1.2);
  REQUIRE(params.getD() == 20.0);
  REQUIRE(params.getSr() == 1.1);
  REQUIRE(params.getCutoff() == rgpot::D2Config{}.cutoff);
  params.setFunctional("pbe");
  auto applied = rgpot::rpc::configureD2(cfg.asReader());
  INFO(applied.message);
  REQUIRE(applied.ok);
  REQUIRE(applied.replacement->config().s6 == rgpot::d2_config_for("pbe").s6);

  params.setFunctional("no-such-functional");
  REQUIRE_FALSE(rgpot::rpc::configureD2(cfg.asReader()).ok);
  params.setFunctional("");
  params.setCutoff(-1.0);
  REQUIRE_FALSE(rgpot::rpc::configureD2(cfg.asReader()).ok);
  cfg.initMorse();
  auto refused = rgpot::rpc::configureD2(cfg.asReader());
  REQUIRE_FALSE(refused.ok);
  REQUIRE_THAT(refused.message, ContainsSubstring("morse"));
}

TEST_CASE("the d2cpmd arm configures CPMD's D2 and a composite term", "[rpc][d2cpmd]") {
  REQUIRE(capnp::Schema::from<::PotentialConfig>().getFieldByName("d2cpmd").getProto().getOrdinal().getExplicit() == 20);
  capnp::MallocMessageBuilder builder;
  auto cfg = builder.initRoot<::PotentialConfig>();
  auto params = cfg.initD2cpmd();
  REQUIRE(params.getS6() == rgpot::D2CpmdConfig{}.s6);
  REQUIRE(params.getAlpha() == rgpot::D2CpmdConfig{}.alpha);
  REQUIRE(params.getVdweps() == rgpot::D2CpmdConfig{}.vdweps);
  params.setFunctional("PBE");
  auto applied = rgpot::rpc::configureD2Cpmd(cfg.asReader());
  INFO(applied.message);
  REQUIRE(applied.ok);
  REQUIRE(applied.replacement->config().s6 == 0.75);
  params.setFunctional("b97-d");
  REQUIRE_FALSE(rgpot::rpc::configureD2Cpmd(cfg.asReader()).ok);
  cfg.initD2();
  auto refused = rgpot::rpc::configureD2Cpmd(cfg.asReader());
  REQUIRE_FALSE(refused.ok);
  REQUIRE_THAT(refused.message, ContainsSubstring("d2"));

  capnp::MallocMessageBuilder sum_builder;
  auto sum_cfg = sum_builder.initRoot<::PotentialConfig>();
  auto terms = sum_cfg.initComposite().initTerms(2);
  terms[0].setName("undo cpmd");
  terms[0].setWeight(-1.0);
  terms[0].initPot().initBuiltin().setName("d2cpmd");
  terms[1].setName("d2");
  terms[1].initPot().initBuiltin().setName("d2");
  auto result = rgpot::rpc::configureComposite(sum_cfg.asReader());
  INFO(result.message);
  REQUIRE(result.ok);
  auto [e, f, v] = (*result.replacement)(kPositions, kZ, kBox);
  (void)f;
  (void)v;
  rgpot::D2Pot lattice;
  rgpot::D2CpmdPot cpmd;
  const double expected = std::get<0>(lattice(kPositions, kZ, kBox)) -
                          std::get<0>(cpmd(kPositions, kZ, kBox));
  REQUIRE(e == Catch::Approx(expected).margin(1e-14));
}

#ifdef RGPOT_HAS_DFTD4
TEST_CASE("the d4 arm configures a D4 backend and a composite term", "[rpc][d4]") {
  capnp::MallocMessageBuilder builder;
  auto cfg = builder.initRoot<::PotentialConfig>();
  auto params = cfg.initD4();
  REQUIRE_FALSE(params.getPeriodic());
  params.setFunctional("blyp");
  params.setPeriodic(true);
  auto applied = rgpot::rpc::configureD4(cfg.asReader());
  INFO(applied.message);
  REQUIRE(applied.ok);
  REQUIRE(applied.replacement->config().functional == "blyp");
  REQUIRE(applied.replacement->caps().stress);
  params.setFunctional("nonesuch");
  auto unknown = rgpot::rpc::configureD4(cfg.asReader());
  REQUIRE_FALSE(unknown.ok);
  REQUIRE_THAT(unknown.message, ContainsSubstring("d4 arm rejected"));
  cfg.initD2();
  REQUIRE_FALSE(rgpot::rpc::configureD4(cfg.asReader()).ok);

  capnp::MallocMessageBuilder sum_builder;
  auto sum_cfg = sum_builder.initRoot<::PotentialConfig>();
  auto terms = sum_cfg.initComposite().initTerms(2);
  terms[0].setName("d2");
  terms[0].setWeight(-1.0);
  terms[0].initPot().initBuiltin().setName("d2");
  terms[1].setName("d4");
  auto d4 = terms[1].initPot().initBuiltin();
  d4.setName("d4");
  auto d4p = d4.initConfig().initD4();
  d4p.setFunctional("blyp");
  d4p.setPeriodic(true);
  auto result = rgpot::rpc::configureComposite(sum_cfg.asReader());
  INFO(result.message);
  REQUIRE(result.ok);
  REQUIRE(result.replacement->caps().stress);
  auto [e, f, v] = (*result.replacement)(kPositions, kZ, kBox);
  (void)f;
  (void)v;
  rgpot::D2Pot lattice;
  rgpot::D4Pot direct(rgpot::D4Config{.functional = "blyp", .periodic = true});
  const double expected = std::get<0>(direct(kPositions, kZ, kBox)) -
                          std::get<0>(lattice(kPositions, kZ, kBox));
  REQUIRE(e == Catch::Approx(expected).margin(1e-14));
}
#endif
