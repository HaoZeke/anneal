// MIT License — task-aware UMA spin validation in UmaConfig.hpp

#include <stdexcept>
#include <string>

#include <catch2/catch_all.hpp>

#include "rgpot/UmaPot/UmaConfig.hpp"

TEST_CASE("UmaConfig defaults stay omol and spin 1", "[uma][spin]") {
  const rgpot::UmaConfig cfg;
  REQUIRE(cfg.task_name == "omol");
  REQUIRE(cfg.spin == 1);
}

TEST_CASE("validate_uma_spin_for_task accepts 0..100 for non-omol tasks",
          "[uma][spin]") {
  for (const char *task :
       {"omat", "oc20", "oc22", "oc25", "odac", "omc", "", "custom"}) {
    REQUIRE_NOTHROW(rgpot::validate_uma_spin_for_task(task, 0, "test"));
    REQUIRE_NOTHROW(rgpot::validate_uma_spin_for_task(task, 1, "test"));
    REQUIRE_NOTHROW(
        rgpot::validate_uma_spin_for_task(task, rgpot::kUmaSpinMax, "test"));
  }
}

TEST_CASE("validate_uma_spin_for_task accepts 0..100 for omol",
          "[uma][spin]") {
  REQUIRE_NOTHROW(rgpot::validate_uma_spin_for_task("omol", 0, "test"));
  REQUIRE_NOTHROW(rgpot::validate_uma_spin_for_task("omol", 1, "test"));
  REQUIRE_NOTHROW(
      rgpot::validate_uma_spin_for_task("omol", rgpot::kUmaSpinMax, "test"));
}

TEST_CASE("validate_uma_spin_for_task rejects out-of-range omol spins",
          "[uma][spin]") {
  for (const int bad : {-1, rgpot::kUmaSpinMax + 1}) {
    REQUIRE_THROWS_AS(rgpot::validate_uma_spin_for_task("omol", bad, "caller"),
                      std::invalid_argument);
    REQUIRE_THROWS_WITH(
        rgpot::validate_uma_spin_for_task("omol", bad, "caller"),
        Catch::Matchers::ContainsSubstring("caller:") &&
            Catch::Matchers::ContainsSubstring("omol") &&
            Catch::Matchers::ContainsSubstring("0 to 100") &&
            Catch::Matchers::ContainsSubstring(std::to_string(bad)));
  }
}

TEST_CASE("validate_uma_spin_for_task rejects out-of-range non-omol spins",
          "[uma][spin]") {
  for (const int bad : {-1, rgpot::kUmaSpinMax + 1}) {
    REQUIRE_THROWS_AS(rgpot::validate_uma_spin_for_task("omat", bad, "caller"),
                      std::invalid_argument);
    REQUIRE_THROWS_WITH(
        rgpot::validate_uma_spin_for_task("omat", bad, "caller"),
        Catch::Matchers::ContainsSubstring("caller:") &&
            Catch::Matchers::ContainsSubstring("omat") &&
            Catch::Matchers::ContainsSubstring("0 to 100") &&
            Catch::Matchers::ContainsSubstring(std::to_string(bad)));
  }
  REQUIRE_THROWS_WITH(rgpot::validate_uma_spin_for_task("omat", -1, ""),
                      !Catch::Matchers::StartsWith(":"));
}
