// MIT License
// Copyright 2023--present rgpot developers
//
// rgpot-ipi-driver connects to an i-PI-protocol server (i-PI, ASE's
// SocketIOCalculator, rgpot's IPIPot) and answers it with any potential
// engines/pot_spec.hpp builds. The flags follow i-pi-py_driver where they
// map (-u, -a, -p, -S, -m, -o, -v); see --help.

#include "pot_spec.hpp"

#include "rgpot/IPIPot/IPIDriver.hpp"

#include <cctype>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr const char *kUsage =
    R"(usage: rgpot-ipi-driver --pot "Name [key value ...]" (--species LIST | --species-from FILE)
                        [-u] [-a ADDRESS] [-p PORT] [-S PREFIX] [options]

Connects to an i-PI-protocol server and answers it with an rgpot potential.

potential:
  --pot SPEC             potential name and key value options, one string
                         ("LJ u0 0.0104 psi 3.4 cutoff 8.5"), as pair_coeff
  -m, --mode NAME        potential name (i-pi-py_driver -m); options from -o
  -o, --param LIST       key=value pairs separated by commas, for -m
  --list                 print the names this build accepts and exit

species (the protocol carries none; one entry per atom, server order):
  --species LIST         symbols or atomic numbers separated by commas;
                         Sym*N repeats (Cu*108,Ni*0 or 29,29,28)
  --species-from FILE    read the symbols from an .xyz/.extxyz or eOn .con file

socket (i-pi-py_driver conventions):
  -u, --unix             UNIX socket; -a is a name (PREFIX + name) or an
                         absolute path
  -a, --address ADDRESS  TCP host, or UNIX socket name (default localhost)
  -p, --port PORT        TCP port (default 12345; ignored with -u)
  -S, --sockets-prefix P prefix of a UNIX socket name (default /tmp/ipi_)

options:
  --cutoff A             host cutoff in Angstrom: the cutoff of a potential
                         whose spec names none, and the range used to tile
                         thin periodic cells and pad open boxes
  --preserve-open-cell   pass a singular cell to the potential unchanged
                         instead of padding it with a finite open box
  --stress-step H        central-difference virial with strain step H when
                         the potential reports no stress (0: zero virial)
  --timeout S            seconds one message may take (0: wait forever)
  --connect-timeout S    seconds to retry connecting (default 30)
  --no-extras            send an empty extra string instead of JSON
  -v, --verbose          report the connection, INIT and the call count
  -h, --help             print this help
)";

[[noreturn]] void usage_error(const std::string &message) {
  std::fprintf(stderr, "rgpot-ipi-driver: %s\n%s", message.c_str(), kUsage);
  std::exit(2);
}

std::string trim(const std::string &s) {
  size_t a = 0;
  size_t b = s.size();
  while (a < b && std::isspace(static_cast<unsigned char>(s[a])) != 0) {
    ++a;
  }
  while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1])) != 0) {
    --b;
  }
  return s.substr(a, b - a);
}

std::vector<std::string> split(const std::string &s, char sep) {
  std::vector<std::string> out;
  std::string cur;
  std::istringstream in(s);
  while (std::getline(in, cur, sep)) {
    out.push_back(trim(cur));
  }
  return out;
}

double number(const std::string &flag, const std::string &text) {
  size_t used = 0;
  double v = 0.0;
  try {
    v = std::stod(text, &used);
  } catch (const std::exception &) {
    used = 0;
  }
  if (used == 0 || used != text.size()) {
    usage_error(flag + " needs a number, not '" + text + "'");
  }
  return v;
}

int atomic_number(const std::string &token) {
  if (!token.empty() &&
      std::isdigit(static_cast<unsigned char>(token[0])) != 0) {
    size_t used = 0;
    int z = 0;
    try {
      z = std::stoi(token, &used);
    } catch (const std::exception &) {
      used = 0;
    }
    if (used != token.size() || z < 1 || z > 118) {
      throw std::invalid_argument("'" + token + "' is not an atomic number");
    }
    return z;
  }
  const int z = rgpot::spec::atomicNumber(token);
  if (z == 0) {
    throw std::invalid_argument("'" + token + "' is not an element symbol");
  }
  return z;
}

std::vector<int> parse_species(const std::string &list) {
  std::vector<int> out;
  for (const std::string &item : split(list, ',')) {
    if (item.empty()) {
      throw std::invalid_argument("empty entry in the species list");
    }
    const auto star = item.find('*');
    long count = 1;
    std::string name = item;
    if (star != std::string::npos) {
      name = trim(item.substr(0, star));
      const std::string reps = trim(item.substr(star + 1));
      size_t used = 0;
      try {
        count = std::stol(reps, &used);
      } catch (const std::exception &) {
        used = 0;
      }
      if (used == 0 || used != reps.size() || count < 0) {
        throw std::invalid_argument("'" + item + "' needs Sym*N with N >= 0");
      }
    }
    const int z = atomic_number(name);
    out.insert(out.end(), static_cast<size_t>(count), z);
  }
  return out;
}

bool ends_with(const std::string &s, const std::string &tail) {
  return s.size() >= tail.size() &&
         s.compare(s.size() - tail.size(), tail.size(), tail) == 0;
}

std::vector<int> species_from_xyz(std::istream &in, const std::string &path) {
  std::string line;
  if (!std::getline(in, line)) {
    throw std::invalid_argument(path + ": empty file");
  }
  long n = 0;
  try {
    n = std::stol(trim(line));
  } catch (const std::exception &) {
    n = -1;
  }
  if (n <= 0) {
    throw std::invalid_argument(path + ": the first line is not an atom count");
  }
  std::getline(in, line);
  std::vector<int> out;
  out.reserve(static_cast<size_t>(n));
  for (long i = 0; i < n; ++i) {
    if (!std::getline(in, line)) {
      throw std::invalid_argument(path + ": fewer atom lines than the count");
    }
    std::istringstream row(line);
    std::string symbol;
    row >> symbol;
    out.push_back(atomic_number(symbol));
  }
  return out;
}

// eOn .con: two comment lines, cell lengths, cell angles, two comment
// lines, the component count, the atoms per component, the masses, then
// per component its symbol, a "Coordinates of Component" line and its
// atom lines.
std::vector<int> species_from_con(std::istream &in, const std::string &path) {
  std::string line;
  for (int i = 0; i < 6; ++i) {
    if (!std::getline(in, line)) {
      throw std::invalid_argument(path + ": truncated .con header");
    }
  }
  long ncomp = 0;
  if (!(in >> ncomp) || ncomp <= 0) {
    throw std::invalid_argument(path + ": no component count");
  }
  std::vector<long> counts(static_cast<size_t>(ncomp));
  for (long &c : counts) {
    if (!(in >> c) || c < 0) {
      throw std::invalid_argument(path + ": bad atoms per component");
    }
  }
  std::getline(in, line);
  std::getline(in, line);
  std::vector<int> out;
  for (const long c : counts) {
    std::string symbol;
    if (!(in >> symbol)) {
      throw std::invalid_argument(path + ": missing component symbol");
    }
    const int z = atomic_number(symbol);
    std::getline(in, line);
    std::getline(in, line);
    for (long i = 0; i < c; ++i) {
      if (!std::getline(in, line)) {
        throw std::invalid_argument(path + ": fewer atom lines than counted");
      }
    }
    out.insert(out.end(), static_cast<size_t>(c), z);
  }
  return out;
}

std::vector<int> species_from_file(const std::string &path) {
  std::ifstream in(path);
  if (!in) {
    throw std::invalid_argument("cannot open " + path);
  }
  if (ends_with(path, ".con")) {
    return species_from_con(in, path);
  }
  if (ends_with(path, ".xyz") || ends_with(path, ".extxyz")) {
    return species_from_xyz(in, path);
  }
  throw std::invalid_argument(path + ": expected a .xyz, .extxyz or .con file");
}

} // namespace

int main(int argc, char **argv) {
  std::string pot;
  std::string mode;
  std::string params;
  std::string species_list;
  std::string species_file;
  bool unix_socket = false;
  std::string address = "localhost";
  std::string port = "12345";
  std::string prefix = "/tmp/ipi_";
  double cutoff = 0.0;
  bool verbose = false;
  rgpot::IPIDriverConfig cfg;

  std::vector<std::string> args(argv + 1, argv + argc);
  for (size_t k = 0; k < args.size(); ++k) {
    const std::string &a = args[k];
    auto value = [&]() -> std::string {
      if (k + 1 >= args.size()) {
        usage_error(a + " needs a value");
      }
      return args[++k];
    };
    if (a == "-h" || a == "--help") {
      std::fputs(kUsage, stdout);
      return 0;
    } else if (a == "--list") {
      std::printf("%s\n", rgpot::spec::names().c_str());
      return 0;
    } else if (a == "--pot") {
      pot = value();
    } else if (a == "-m" || a == "--mode") {
      mode = value();
    } else if (a == "-o" || a == "--param") {
      params = value();
    } else if (a == "--species") {
      species_list = value();
    } else if (a == "--species-from") {
      species_file = value();
    } else if (a == "-u" || a == "--unix") {
      unix_socket = true;
    } else if (a == "-a" || a == "--address") {
      address = value();
    } else if (a == "-p" || a == "--port") {
      port = value();
    } else if (a == "-S" || a == "--sockets-prefix") {
      prefix = value();
    } else if (a == "--cutoff") {
      cutoff = number(a, value());
    } else if (a == "--preserve-open-cell") {
      cfg.preserve_open_cell = true;
    } else if (a == "--stress-step") {
      cfg.stress_step = number(a, value());
    } else if (a == "--timeout") {
      cfg.timeout_s = number(a, value());
    } else if (a == "--connect-timeout") {
      cfg.connect_timeout_s = number(a, value());
    } else if (a == "--no-extras") {
      cfg.extras_json = false;
    } else if (a == "-v" || a == "--verbose") {
      verbose = true;
    } else {
      usage_error("unknown argument " + a);
    }
  }

  if (pot.empty() == mode.empty()) {
    usage_error("give exactly one of --pot and -m");
  }
  if (!params.empty() && mode.empty()) {
    usage_error("-o goes with -m; put options in the --pot string");
  }
  if (species_list.empty() == species_file.empty()) {
    usage_error("give exactly one of --species and --species-from");
  }
  if (!(cutoff >= 0.0)) {
    usage_error("--cutoff must be >= 0");
  }

  std::string name;
  rgpot::spec::Options options;
  if (!pot.empty()) {
    std::istringstream words(pot);
    std::vector<std::string> tokens;
    for (std::string w; words >> w;) {
      tokens.push_back(w);
    }
    if (tokens.empty() || tokens.size() % 2 != 1) {
      usage_error("--pot is a name followed by key value pairs");
    }
    name = tokens[0];
    for (size_t k = 1; k + 1 < tokens.size(); k += 2) {
      options.emplace_back(tokens[k], tokens[k + 1]);
    }
  } else {
    name = mode;
    if (!params.empty()) {
      for (const std::string &kv : split(params, ',')) {
        const auto eq = kv.find('=');
        if (eq == std::string::npos || eq == 0) {
          usage_error("-o takes key=value pairs, not '" + kv + "'");
        }
        options.emplace_back(trim(kv.substr(0, eq)), trim(kv.substr(eq + 1)));
      }
    }
  }

  try {
    cfg.species = species_list.empty() ? species_from_file(species_file)
                                       : parse_species(species_list);
  } catch (const std::exception &e) {
    usage_error(e.what());
  }

  if (unix_socket) {
    cfg.address = "unix:" + address;
    cfg.unix_prefix = prefix;
  } else {
    cfg.address = "tcp:" + address + ":" + port;
  }

  rgpot::spec::Built built = rgpot::spec::build(name, options, cutoff);
  if (!built.pot) {
    std::fprintf(stderr, "rgpot-ipi-driver: %s\n", built.error.c_str());
    return 2;
  }
  cfg.cutoff = built.cutoff > 0.0 ? built.cutoff : cutoff;

  // IPIDriver's socket writes already avoid SIGPIPE. This covers the other
  // writes (stderr piped to a launcher that has gone, a potential's own
  // sockets), which then fail with EPIPE instead of killing the run.
  std::signal(SIGPIPE, SIG_IGN);

  try {
    rgpot::IPIDriver driver(cfg, built.eval);
    if (verbose) {
      std::fprintf(stderr,
                   "rgpot-ipi-driver: %s for %zu atoms, connecting to %s\n",
                   name.c_str(), cfg.species.size(), driver.endpoint().c_str());
    }
    driver.connect();
    int32_t bead = -1;
    while (driver.step()) {
      if (verbose && driver.bead() != bead) {
        bead = driver.bead();
        std::fprintf(stderr, "rgpot-ipi-driver: INIT bead %d, '%s'\n",
                     static_cast<int>(bead), driver.initString().c_str());
      }
    }
    if (verbose) {
      std::fprintf(stderr, "rgpot-ipi-driver: EXIT after %llu evaluations\n",
                   static_cast<unsigned long long>(driver.evaluations()));
    }
  } catch (const std::invalid_argument &e) {
    std::fprintf(stderr, "rgpot-ipi-driver: %s\n", e.what());
    return 2;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "rgpot-ipi-driver: %s\n", e.what());
    return 1;
  }
  return 0;
}
