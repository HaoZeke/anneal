// MIT License
// Copyright 2023--present rgpot developers

#include <catch2/catch_test_macros.hpp>

#include <string>

#include "rgpot/CPMDPot/ConfigMessage.hpp"

TEST_CASE("CPMD config messages keep stable refusal wording", "[cpmd]") {
  const std::string summary = "functional=BLYP cutoffRy=70.000000";

  REQUIRE(rgpot::formatCpmdConfigMessage(summary, true, true, "") ==
          "rgpot params applied (cpmd arm): " + summary);
  REQUIRE(rgpot::formatCpmdConfigMessage(summary, false, false, "") ==
          "rgpot params cpmd arm failed (engine not loaded): " + summary);
  REQUIRE(rgpot::formatCpmdConfigMessage(summary, false, true, "") ==
          "rgpot params cpmd arm rejected by embed: " + summary);
  REQUIRE(rgpot::formatCpmdConfigMessage(summary, false, true,
                                         "CPMD engine refused the session "
                                         "create: fake refused params") ==
          "rgpot params cpmd arm rejected by embed: " + summary +
              " (CPMD engine refused the session create: fake refused params)");
}
