// MIT License
// Copyright 2023--present rgpot developers

#include <catch2/catch_test_macros.hpp>

#include <capnp/message.h>

#include <stdexcept>
#include <string>

#include "rgpot/CP2KPot/CP2KDeck.hpp"
#include "rgpot/rpc/Potentials.capnp.h"

namespace {

// Two silicon atoms of the diamond basis in a cubic 5.43 Angstrom cell.
constexpr double kPos[6] = {0.0, 0.0, 0.0, 1.3575, 1.3575, 1.3575};
constexpr int kZ[2] = {14, 14};
constexpr double kBox[9] = {5.43, 0.0, 0.0, 0.0, 5.43, 0.0, 0.0, 0.0, 5.43};

bool has(const std::string &text, const std::string &needle) {
  return text.find(needle) != std::string::npos;
}

} // namespace

TEST_CASE("CP2K default deck matches the golden text", "[cp2k][deck]") {
  ::capnp::MallocMessageBuilder msg;
  auto p = msg.initRoot<::CP2KParams>();
  const std::string deck =
      rgpot::cp2k::renderDeck(p.asReader(), 2, kPos, kZ, kBox);
  const std::string golden = "&GLOBAL\n"
                             "  PROJECT rgpot\n"
                             "  PRINT_LEVEL LOW\n"
                             "&END GLOBAL\n"
                             "&FORCE_EVAL\n"
                             "  METHOD Quickstep\n"
                             "  &DFT\n"
                             "    BASIS_SET_FILE_NAME BASIS_MOLOPT\n"
                             "    POTENTIAL_FILE_NAME GTH_POTENTIALS\n"
                             "    CHARGE 0\n"
                             "    MULTIPLICITY 1\n"
                             "    &MGRID\n"
                             "      CUTOFF 650\n"
                             "      REL_CUTOFF 70\n"
                             "    &END MGRID\n"
                             "    &SCF\n"
                             "      SCF_GUESS ATOMIC\n"
                             "      EPS_SCF 1e-06\n"
                             "      MAX_SCF 50\n"
                             "      IGNORE_CONVERGENCE_FAILURE .TRUE.\n"
                             "      &OT\n"
                             "        MINIMIZER DIIS\n"
                             "        PRECONDITIONER FULL_SINGLE_INVERSE\n"
                             "      &END OT\n"
                             "      &PRINT\n"
                             "        &RESTART OFF\n"
                             "        &END RESTART\n"
                             "      &END PRINT\n"
                             "    &END SCF\n"
                             "    &XC\n"
                             "      &XC_FUNCTIONAL PBE\n"
                             "      &END XC_FUNCTIONAL\n"
                             "    &END XC\n"
                             "  &END DFT\n"
                             "  &SUBSYS\n"
                             "    &CELL\n"
                             "A 5.430000000000 0.000000000000 0.000000000000\n"
                             "B 0.000000000000 5.430000000000 0.000000000000\n"
                             "C 0.000000000000 0.000000000000 5.430000000000\n"
                             "      PERIODIC XYZ\n"
                             "    &END CELL\n"
                             "    &COORD\n"
                             "Si 0.000000000000 0.000000000000 0.000000000000\n"
                             "Si 1.357500000000 1.357500000000 1.357500000000\n"
                             "    &END COORD\n"
                             "    &KIND Si\n"
                             "      BASIS_SET DZVP-MOLOPT-SR-GTH\n"
                             "      POTENTIAL GTH-PBE\n"
                             "    &END KIND\n"
                             "  &END SUBSYS\n"
                             "&END FORCE_EVAL\n";
  REQUIRE(deck == golden);
}

TEST_CASE("CP2K deck carries the Leiden spin-polarised settings",
          "[cp2k][deck]") {
  ::capnp::MallocMessageBuilder msg;
  auto p = msg.initRoot<::CP2KParams>();
  p.setUks(true);
  p.setEpsScf(1.0e-9);
  p.setOuterMaxScf(10);
  p.setMaxScf(30);
  auto dft = p.initExtraDftSections(1);
  dft.set(0, "    &QS\n      EPS_DEFAULT 1.0E-12\n    &END QS");
  auto fe = p.initExtraForceEvalSections(1);
  fe.set(0, "  &PRINT\n    &FORCES ON\n    &END FORCES\n  &END PRINT");
  const int z[3] = {14, 8, 8};
  const double pos[9] = {0, 0, 0, 1.6, 0, 0, -1.6, 0, 0};
  const std::string deck =
      rgpot::cp2k::renderDeck(p.asReader(), 3, pos, z, kBox);
  REQUIRE(has(deck, "    UKS .TRUE.\n"));
  REQUIRE(has(deck, "      EPS_SCF 1e-09\n      MAX_SCF 30\n"));
  REQUIRE(has(deck, "      &OUTER_SCF\n        EPS_SCF 1e-09\n"
                    "        MAX_SCF 10\n      &END OUTER_SCF\n"));
  REQUIRE(has(deck, "&END QS\n  &END DFT\n"));
  REQUIRE(has(deck, "&END PRINT\n&END FORCE_EVAL\n"));
  // One &KIND per element, in order of first appearance.
  const auto si = deck.find("&KIND Si\n");
  const auto o = deck.find("&KIND O\n");
  REQUIRE(si != std::string::npos);
  REQUIRE(o != std::string::npos);
  REQUIRE(si < o);
  REQUIRE(deck.find("&KIND O\n", o + 1) == std::string::npos);
}

TEST_CASE("CP2K template gets the geometry in place of its tokens",
          "[cp2k][deck]") {
  ::capnp::MallocMessageBuilder msg;
  auto p = msg.initRoot<::CP2KParams>();
  p.setInputTemplate("&FORCE_EVAL\n  &SUBSYS\n    &CELL\n@RGPOT_CELL@\n"
                     "    &END CELL\n    &COORD\n@RGPOT_COORD@\n"
                     "    &END COORD\n  &END SUBSYS\n&END FORCE_EVAL\n");
  const std::string deck =
      rgpot::cp2k::renderDeck(p.asReader(), 2, kPos, kZ, kBox);
  REQUIRE(deck == "&FORCE_EVAL\n  &SUBSYS\n    &CELL\n"
                  "A 5.430000000000 0.000000000000 0.000000000000\n"
                  "B 0.000000000000 5.430000000000 0.000000000000\n"
                  "C 0.000000000000 0.000000000000 5.430000000000\n"
                  "    &END CELL\n    &COORD\n"
                  "Si 0.000000000000 0.000000000000 0.000000000000\n"
                  "Si 1.357500000000 1.357500000000 1.357500000000\n"
                  "    &END COORD\n  &END SUBSYS\n&END FORCE_EVAL\n");
}

TEST_CASE("CP2K template without a geometry token is refused", "[cp2k][deck]") {
  ::capnp::MallocMessageBuilder msg;
  auto p = msg.initRoot<::CP2KParams>();
  p.setInputTemplate("&FORCE_EVAL\n  &SUBSYS\n    &COORD\n@RGPOT_COORD@\n"
                     "    &END COORD\n  &END SUBSYS\n&END FORCE_EVAL\n");
  REQUIRE_THROWS_AS(rgpot::cp2k::renderDeck(p.asReader(), 2, kPos, kZ, kBox),
                    std::invalid_argument);
  REQUIRE(rgpot::cp2k::checkTemplate(p.getInputTemplate().cStr()) ==
          "CP2K inputTemplate lacks @RGPOT_CELL@");
  REQUIRE(rgpot::cp2k::checkTemplate("&COORD\n&END COORD\n") ==
          "CP2K inputTemplate lacks @RGPOT_COORD@ and @RGPOT_CELL@");
}

TEST_CASE("CP2K element symbols cover the periodic table", "[cp2k][deck]") {
  REQUIRE(rgpot::cp2k::elementSymbol(1) == "H");
  REQUIRE(rgpot::cp2k::elementSymbol(14) == "Si");
  REQUIRE(rgpot::cp2k::elementSymbol(118) == "Og");
  REQUIRE_THROWS_AS(rgpot::cp2k::elementSymbol(0), std::invalid_argument);
  REQUIRE_THROWS_AS(rgpot::cp2k::elementSymbol(119), std::invalid_argument);
}
