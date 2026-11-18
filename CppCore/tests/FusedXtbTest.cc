// MIT License
// Copyright 2023--present rgpot developers
//
// The fused cache in front of a real GFN2-xTB potential. A failed SCC is
// stored like a result: one libxtb call per geometry and setting, the same
// failure on every later request, the failure typed as an SCC that did not
// converge, and a changed setting as the only way to evaluate it again.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <catch2/catch_all.hpp>

#include "rgpot/HessianVector.hpp"
#include "rgpot/XTBPot/XTBPot.hpp"
#include "rgpot/potential.hpp"

using rgpot::ForceInput;
using rgpot::ForceOut;
using rgpot::FusedSettingsSpec;
using rgpot::PotentialHandle;
using rgpot::SccNotConverged;
using rgpot::XTBConfig;
using rgpot::XTBPot;

namespace {

using EvalFn = int (*)(void *, const DLManagedTensorVersioned *, double *);
using GradFn = int (*)(void *, const DLManagedTensorVersioned *,
                       DLManagedTensorVersioned *);

/// A real XTBPot behind forceImpl, with the libxtb calls counted. When
/// @c injectFailure is set the call is counted and fails before xtb runs.
struct CountingXtb {
  XTBPot pot;
  mutable int calls = 0;
  mutable bool injectFailure = false;

  explicit CountingXtb(const XTBConfig &config) : pot(config) {}

  void forceImpl(const ForceInput &in, ForceOut *out) const {
    ++calls;
    if (injectFailure) {
      throw std::runtime_error("injected engine failure");
    }
    pot.forceImpl(in, out);
  }

  FusedSettingsSpec fusedSettings() const { return pot.fusedSettings(); }
};

XTBConfig unconvergable() {
  XTBConfig cfg;
  cfg.max_iterations = 1;
  return cfg;
}

XTBConfig convergable() {
  XTBConfig cfg;
  cfg.max_iterations = 250;
  return cfg;
}

struct Water {
  std::vector<double> x{0.0,    0.0, 0.117,  0.0,  0.757,
                        -0.469, 0.0, -0.757, 0.469};
  std::vector<int> z{8, 1, 1};
  std::vector<double> box{20.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 20.0};

  ForceInput input() const {
    return {.nAtoms = z.size(),
            .pos = x.data(),
            .atmnrs = z.data(),
            .box = box.data()};
  }
};

/// One call of the base evaluation or gradient function of the eindir
/// objective of a handle: no wrapper that discards the status.
struct Probe {
  PotentialHandle &handle;
  std::size_t dim;

  struct Outcome {
    int status;
    std::string message;
    double value;
    std::vector<double> gradient;
  };

  static std::string lastError() {
    const char *msg = rgpot_last_error();
    return msg == nullptr ? std::string() : std::string(msg);
  }

  eindir_objective_t *objective() const {
    return reinterpret_cast<eindir_objective_t *>(handle.raw());
  }

  Outcome energy(std::vector<double> x) const {
    rgpot_set_last_error("stale message from elsewhere");
    DLManagedTensorVersioned *t =
        rgpot_tensor_cpu_f64_2d(x.data(), static_cast<int64_t>(dim / 3), 3);
    Outcome o{0, "", 0.0, {}};
    o.status = reinterpret_cast<EvalFn>(objective()->eval_fn)(
        objective()->user_data, t, &o.value);
    o.message = lastError();
    rgpot_tensor_free(t);
    return o;
  }

  Outcome gradient(std::vector<double> x) const {
    rgpot_set_last_error("stale message from elsewhere");
    DLManagedTensorVersioned *t =
        rgpot_tensor_cpu_f64_2d(x.data(), static_cast<int64_t>(dim / 3), 3);
    Outcome o{0, "", 0.0, std::vector<double>(dim, 0.0)};
    DLManagedTensorVersioned *g = rgpot_tensor_cpu_f64_2d(
        o.gradient.data(), static_cast<int64_t>(dim / 3), 3);
    o.status = reinterpret_cast<GradFn>(objective()->grad_fn)(
        objective()->user_data, t, g);
    o.message = lastError();
    rgpot_tensor_free(g);
    rgpot_tensor_free(t);
    return o;
  }
};

} // namespace

TEST_CASE("a failed xtb energy then gradient is one libxtb call, with the same "
          "failure",
          "[fused][xtb]") {
  Water w;
  CountingXtb xtb(unconvergable());
  auto pot =
      PotentialHandle::from_impl(xtb, w.z.size(), w.z.data(), w.box.data());
  Probe probe{pot, w.x.size()};

  const auto first = probe.energy(w.x);
  REQUIRE(first.status != 0);
  CHECK(xtb.calls == 1);
  CHECK(pot.fused_error_kind() == RGPOT_FUSED_SCC_NOT_CONVERGED);
  CHECK_FALSE(first.message.empty());
  CHECK(first.message != "stale message from elsewhere");

  const auto second = probe.gradient(w.x);
  CHECK(xtb.calls == 1);
  CHECK(second.status == first.status);
  CHECK(second.message == first.message);
  CHECK(pot.fused_error_kind() == RGPOT_FUSED_SCC_NOT_CONVERGED);

  const auto third = probe.energy(w.x);
  CHECK(xtb.calls == 1);
  CHECK(third.status == first.status);
  CHECK(third.message == first.message);
}

TEST_CASE("a converged xtb energy then gradient is one libxtb call",
          "[fused][xtb]") {
  Water w;
  CountingXtb xtb(convergable());
  auto pot =
      PotentialHandle::from_impl(xtb, w.z.size(), w.z.data(), w.box.data());
  Probe probe{pot, w.x.size()};

  const auto energy = probe.energy(w.x);
  REQUIRE(energy.status == 0);
  CHECK(std::isfinite(energy.value));
  CHECK(xtb.calls == 1);

  const auto gradient = probe.gradient(w.x);
  REQUIRE(gradient.status == 0);
  CHECK(xtb.calls == 1);
  CHECK(pot.fused_error_kind() == RGPOT_FUSED_OK);
  double norm = 0.0;
  for (double g : gradient.gradient) {
    REQUIRE(std::isfinite(g));
    norm += g * g;
  }
  CHECK(norm > 0.0);

  auto moved = w.x;
  moved[0] = std::nextafter(moved[0], 1.0);
  CHECK(probe.gradient(moved).status == 0);
  CHECK(xtb.calls == 2);
}

TEST_CASE("an xtb failure is retried only through a changed setting",
          "[fused][xtb]") {
  Water w;
  CountingXtb xtb(unconvergable());
  auto pot =
      PotentialHandle::from_impl(xtb, w.z.size(), w.z.data(), w.box.data());
  Probe probe{pot, w.x.size()};

  const auto failed = probe.energy(w.x);
  REQUIRE(failed.status != 0);
  REQUIRE(xtb.calls == 1);

  // The potential changed, the handle was not told: the failure stands.
  xtb.pot.setMaxIterations(250);
  CHECK(probe.energy(w.x).status == failed.status);
  CHECK(xtb.calls == 1);

  // Telling the handle makes the changed setting a new key.
  pot.sync_fused_settings(xtb);
  const auto retried = probe.energy(w.x);
  CHECK(retried.status == 0);
  CHECK(xtb.calls == 2);
  CHECK(pot.fused_error_kind() == RGPOT_FUSED_OK);
  CHECK(probe.gradient(w.x).status == 0);
  CHECK(xtb.calls == 2);
}

TEST_CASE("every fingerprint setting of xtb is a new key", "[fused][xtb]") {
  Water w;
  CountingXtb xtb(convergable());
  auto pot =
      PotentialHandle::from_impl(xtb, w.z.size(), w.z.data(), w.box.data());
  Probe probe{pot, w.x.size()};
  REQUIRE(probe.energy(w.x).status == 0);
  int expected = 1;
  REQUIRE(xtb.calls == expected);

  auto change = [&](auto &&mutate) {
    mutate();
    pot.sync_fused_settings(xtb);
    REQUIRE(probe.energy(w.x).status == 0);
    ++expected;
    CHECK(xtb.calls == expected);
    CHECK(probe.gradient(w.x).status == 0);
    CHECK(xtb.calls == expected);
  };
  change([&] { xtb.pot.setAccuracy(0.5); });
  change([&] { xtb.pot.setElectronicTemperature(500.0); });
  change([&] { xtb.pot.setMaxIterations(200); });
}

TEST_CASE("xtb separates an SCC that did not converge from another failure",
          "[fused][xtb]") {
  Water w;
  {
    CountingXtb xtb(unconvergable());
    auto pot =
        PotentialHandle::from_impl(xtb, w.z.size(), w.z.data(), w.box.data());
    Probe probe{pot, w.x.size()};
    REQUIRE(probe.energy(w.x).status != 0);
    CHECK(pot.fused_error_kind() == RGPOT_FUSED_SCC_NOT_CONVERGED);
  }
  {
    CountingXtb xtb(convergable());
    xtb.injectFailure = true;
    auto pot =
        PotentialHandle::from_impl(xtb, w.z.size(), w.z.data(), w.box.data());
    Probe probe{pot, w.x.size()};
    const auto first = probe.energy(w.x);
    REQUIRE(first.status != 0);
    CHECK(pot.fused_error_kind() == RGPOT_FUSED_ENGINE_FAILED);
    CHECK(first.message.find("injected engine failure") != std::string::npos);
    CHECK(probe.gradient(w.x).message == first.message);
    CHECK(xtb.calls == 1);
  }
}

TEST_CASE("the C++ cache stores an xtb SCC failure and throws it typed",
          "[fused][xtb]") {
  Water w;
  CountingXtb xtb(unconvergable());
  rgpot::HessianVectorCache cache;
  const ForceInput in = w.input();

  std::string first;
  try {
    cache.centre(xtb, in);
    FAIL("the centre should not converge");
  } catch (const SccNotConverged &e) {
    first = e.what();
  }
  CHECK(xtb.calls == 1);
  try {
    cache.centre(xtb, in);
    FAIL("the stored failure should be thrown again");
  } catch (const SccNotConverged &e) {
    CHECK(std::string(e.what()) == first);
  }
  CHECK(xtb.calls == 1);

  xtb.pot.setMaxIterations(250);
  const auto centre = cache.centre(xtb, in);
  CHECK(xtb.calls == 2);
  CHECK(std::isfinite(centre.energy));
  cache.centre(xtb, in);
  CHECK(xtb.calls == 2);
}
