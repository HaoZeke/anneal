// MIT License
// Copyright 2023--present rgpot developers

#include <catch2/catch_all.hpp>

#include "rgpot/simd/api.hpp"
#include "rgpot/simd/exp.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace {

std::uint64_t bits_of(double x) {
  std::uint64_t u = 0;
  std::memcpy(&u, &x, sizeof(u));
  return u;
}

void check_one(double a, double b, double c, const char *where) {
  const double got = rgpot::simd::exact_fma(a, b, c);
  const double via = rgpot::simd::mul_add(a, b, c);
  const double ref = std::fma(a, b, c);
  INFO(std::string(where) << " a=" << bits_of(a) << " b=" << bits_of(b)
                          << " c=" << bits_of(c));
  REQUIRE(bits_of(got) == bits_of(ref));
  REQUIRE(bits_of(via) == bits_of(ref));
}

} // namespace

TEST_CASE("exact fma matches hardware fma", "[fma]") {
  const double specials[] = {
      0.0,
      -0.0,
      1.0,
      -1.0,
      2.0,
      -2.0,
      0.5,
      -0.5,
      std::numeric_limits<double>::min(),
      std::numeric_limits<double>::max(),
      std::numeric_limits<double>::denorm_min(),
      std::numeric_limits<double>::infinity(),
      -std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::quiet_NaN(),
      0x1p-1022,
      0x1p-1074,
      0x1.fffffffffffffp1023,
      0x1p-500,
      0x1p500,
      3.141592653589793,
      -2.718281828459045,
      0x1.0000000000001p0,
      0x1.fffffffffffffp0,
  };
  const int ns = static_cast<int>(sizeof(specials) / sizeof(specials[0]));
  for (int i = 0; i < ns; ++i) {
    for (int j = 0; j < ns; ++j) {
      for (int k = 0; k < ns; ++k) {
        check_one(specials[i], specials[j], specials[k], "edge");
      }
    }
  }

  std::mt19937_64 rng(1);
  std::uniform_int_distribution<int> expd(-180, 180);
  std::uniform_int_distribution<std::uint64_t> mant(0, (1ULL << 52) - 1);
  std::uniform_int_distribution<int> sign(0, 1);
  auto rnd = [&]() {
    double x = std::ldexp(static_cast<double>(mant(rng) | (1ULL << 52)),
                          expd(rng) - 52);
    if (sign(rng) != 0) {
      x = -x;
    }
    return x;
  };
  for (int i = 0; i < 20000; ++i) {
    const double a = rnd();
    const double b = rnd();
    const double c = rnd();
    check_one(a, b, c, "rand");
    if (std::isfinite(a * b)) {
      check_one(a, b, -a * b, "cancel");
    }
    const double p = a * b;
    if (std::isfinite(p) && p != 0.0) {
      const double ulp = std::scalbn(1.0, std::ilogb(std::fabs(p)) - 52);
      check_one(a, b, 0.5 * ulp, "tie+");
      check_one(a, b, -0.5 * ulp, "tie-");
    }
  }
  for (int e = -60; e <= 40; e += 5) {
    const double a = std::ldexp(1.0, e);
    const double b = std::ldexp(0x1.0000000000001p0, -e - 30);
    check_one(a, b, -a * b, "under");
  }

  // The half-ulp tie just below a power of two. The repaired emulation
  // matches the hardware instruction. This is the case a truncating sum misses.
  const double tie_a = 0x1.0000000000001p0;
  const double tie_c = 0x1.921fb54442d18p1;
  check_one(tie_a, tie_a, tie_c, "classic-tie");
  {
    const double trunc = rgpot::simd::fma_trunc(tie_a, tie_a, tie_c);
    const double ref = std::fma(tie_a, tie_a, tie_c);
    REQUIRE(bits_of(rgpot::simd::fma_quick(tie_a, tie_a, tie_c)) == bits_of(ref));
    REQUIRE(bits_of(trunc) != bits_of(ref));
  }

  // The exp polynomial uses the truncating sum. It matches the hardware-fma
  // polynomial on the range Morse and EAM evaluate, and on the special bins.
  {
    const double edges[] = {
        0.0, -0.0, 1.0, -1.0, 0.5, -0.5,
        std::numeric_limits<double>::min(),
        std::numeric_limits<double>::denorm_min(),
        std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::quiet_NaN(),
        700.0, -800.0, 0x1p-40, -0x1p-40, 20.0, -20.0,
    };
    auto check_exp = [](double x) {
      const double soft = rgpot::simd::exp_fast<false>(x);
      const double hard = rgpot::simd::exp_fast<true>(x);
      INFO(bits_of(x));
      REQUIRE(bits_of(soft) == bits_of(hard));
    };
    for (double x : edges) {
      check_exp(x);
    }
    for (int i = 0; i < 100000; ++i) {
      const double x = (static_cast<double>(i) / 100000.0) * 40.0 - 20.0;
      check_exp(x);
    }
  }

  const int nrun = rgpot::simd::runnable_target_count();
  std::vector<double> as(64), bs(64), cs(64), out(64);
  for (int i = 0; i < 64; ++i) {
    as[static_cast<std::size_t>(i)] = rnd();
    bs[static_cast<std::size_t>(i)] = rnd();
    cs[static_cast<std::size_t>(i)] = specials[i % ns];
  }
  const std::size_t widths[] = {0, 1, 3, 7, 8, 17, 64};
  for (int t = 0; t < nrun; ++t) {
    const char *name = rgpot::simd::runnable_target_name(t);
    rgpot::simd::set_target_for_test(name);
    REQUIRE(std::strcmp(rgpot::simd::chosen_target_name(), name) == 0);
    for (std::size_t n : widths) {
      rgpot::simd::fma_array(as.data(), bs.data(), cs.data(), out.data(), n);
      for (std::size_t i = 0; i < n; ++i) {
        const double ref = std::fma(as[i], bs[i], cs[i]);
        INFO(name << " n=" << n << " i=" << i);
        REQUIRE(bits_of(out[i]) == bits_of(ref));
      }
    }
  }
}

// exp_hw_n is the body exp() runs on an FMA machine. exp_fast<true> called
// from this file lowers its fma to a libm call, so only this case reaches
// the target("fma") code.
TEST_CASE("exp_hw_n gives the bits of the software exp", "[fma][exp]") {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
  if (!rgpot::simd::detail::fma_hw_active()) {
    SKIP("no hardware FMA, exp_hw_n is not called");
  }
#endif
  std::vector<double> in = {
      0.0, -0.0, 1.0, -1.0, 0.5, -0.5,
      std::numeric_limits<double>::min(),
      std::numeric_limits<double>::denorm_min(),
      std::numeric_limits<double>::infinity(),
      -std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::quiet_NaN(),
      700.0, 709.8, 710.0, -745.2, -800.0, 0x1p-40, -0x1p-40, 20.0, -20.0,
  };
  for (int i = 0; i < 4001; ++i) {
    in.push_back((static_cast<double>(i) / 4000.0) * 40.0 - 20.0);
  }
  const std::size_t widths[] = {0, 1, 2, 3, in.size()};
  for (std::size_t n : widths) {
    std::vector<double> out(n, 0.0);
    rgpot::simd::exp_hw_n(in.data(), out.data(), n);
    for (std::size_t i = 0; i < n; ++i) {
      INFO("n=" << n << " x=" << bits_of(in[i]));
      REQUIRE(bits_of(out[i]) == bits_of(rgpot::simd::exp_fast<false>(in[i])));
    }
  }
}
