// MIT License
// Copyright 2023--present rgpot developers

// Generic MDI engine command: every concrete potential is constructed by the
// shared pot_spec factory and reaches MDI only through PotentialHandle.

#include "pot_spec.hpp"

#include "rgpot/potential.hpp"

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr const char *kUsage =
    R"(usage: rgpot-mdi --pot "Name [key value ...]" -mdi "OPTIONS"
                 [--mdi-library PATH]

Serve any potential accepted by rgpot's shared potential factory to an MDI
driver. OPTIONS normally contain:
  -role ENGINE -name RGPOT -method TCP -hostname HOST -port PORT

The MDI wire is atomic units; potential options remain Angstrom and eV.
)";

[[noreturn]] void fail(const std::string &message) {
  std::fprintf(stderr, "rgpot-mdi: %s\n%s", message.c_str(), kUsage);
  std::exit(2);
}

std::vector<std::string> words(const std::string &text) {
  std::istringstream input(text);
  std::vector<std::string> result;
  for (std::string word; input >> word;) {
    result.push_back(std::move(word));
  }
  return result;
}

rgpot::spec::Built build_potential(const std::string &text) {
  const auto tokens = words(text);
  if (tokens.empty()) {
    fail("--pot needs a potential name");
  }
  if (tokens.size() % 2 == 0) {
    fail("--pot needs a name followed by key/value pairs");
  }
  rgpot::spec::Options options;
  for (size_t i = 1; i < tokens.size(); i += 2) {
    options.emplace_back(tokens[i], tokens[i + 1]);
  }
  auto built = rgpot::spec::build(tokens[0], options, 0.0);
  if (!built.pot) {
    fail(built.error);
  }
  return built;
}

struct GenericPotential {
  rgpot::spec::Built built;

  void forceImpl(const rgpot::ForceInput &input, rgpot::ForceOut *output) {
    built.eval(input, output);
  }
};

} // namespace

int main(int argc, char **argv) {
  std::string potential;
  std::string mdi_options;
  std::string mdi_library;
  for (int i = 1; i < argc; ++i) {
    const std::string flag = argv[i];
    if (flag == "-h" || flag == "--help") {
      std::fputs(kUsage, stdout);
      return 0;
    }
    if (flag == "--list") {
      std::puts(rgpot::spec::names().c_str());
      return 0;
    }
    if (i + 1 >= argc) {
      fail(flag + " needs a value");
    }
    const std::string value = argv[++i];
    if (flag == "--pot") {
      potential = value;
    } else if (flag == "-mdi" || flag == "--mdi") {
      mdi_options = value;
    } else if (flag == "--mdi-library") {
      mdi_library = value;
    } else {
      fail("unknown option '" + flag + "'");
    }
  }
  if (potential.empty() || mdi_options.empty()) {
    fail("--pot and -mdi are required");
  }

  try {
    GenericPotential generic{build_potential(potential)};
    auto handle = rgpot::PotentialHandle::from_impl(generic);
    const char *library = mdi_library.empty() ? nullptr : mdi_library.c_str();
    const auto status =
        rgpot_mdi_server_start(handle.raw(), mdi_options.c_str(), library);
    if (status != RGPOT_SUCCESS) {
      const char *message = rgpot_last_error();
      std::fprintf(stderr, "rgpot-mdi: %s\n",
                   message == nullptr ? "MDI server failed" : message);
      return 1;
    }
  } catch (const std::exception &error) {
    std::fprintf(stderr, "rgpot-mdi: %s\n", error.what());
    return 1;
  }
  return 0;
}
