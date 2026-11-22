// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/MEAM/MEAMReader.hpp"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <istream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace rgpot {

namespace {

[[noreturn]] void bad(const std::string &what) {
  throw std::invalid_argument("MEAM: " + what);
}

std::string strip_comment(const std::string &line) {
  const std::size_t hash = line.find('#');
  return hash == std::string::npos ? line : line.substr(0, hash);
}

std::vector<std::string> split(const std::string &text, const char *separators) {
  std::vector<std::string> words;
  std::size_t pos = 0;
  while (pos < text.size()) {
    const std::size_t begin = text.find_first_not_of(separators, pos);
    if (begin == std::string::npos) {
      break;
    }
    std::size_t end = text.find_first_of(separators, begin);
    if (end == std::string::npos) {
      end = text.size();
    }
    words.push_back(text.substr(begin, end - begin));
    pos = end;
  }
  return words;
}

double to_double(const std::string &word, const std::string &where) {
  char *end = nullptr;
  const double v = std::strtod(word.c_str(), &end);
  if (end == word.c_str() || *end != '\0' || !std::isfinite(v)) {
    bad(where + ": '" + word + "' is not a number");
  }
  return v;
}

int to_int(const std::string &word, const std::string &where) {
  char *end = nullptr;
  const long v = std::strtol(word.c_str(), &end, 10);
  if (end == word.c_str() || *end != '\0' ||
      v < std::numeric_limits<int>::min() || v > std::numeric_limits<int>::max()) {
    bad(where + ": '" + word + "' is not an integer");
  }
  return static_cast<int>(v);
}

constexpr const char *kSymbols[118] = {
    "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne", "Na", "Mg",
    "Al", "Si", "P",  "S",  "Cl", "Ar", "K",  "Ca", "Sc", "Ti", "V",  "Cr",
    "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf",
    "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po",
    "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U",  "Np", "Pu", "Am", "Cm",
    "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs",
    "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"};

constexpr std::size_t kFieldsPerElement = 19;
constexpr const char *kWhite = " \t\n\r\f";

} // namespace

int meam_atomic_number(const std::string &name) {
  if (name.empty()) {
    return 0;
  }
  // Two letters first, so "Co" is cobalt and not carbon.
  if (name.size() >= 2) {
    const std::string two = name.substr(0, 2);
    for (int z = 0; z < 118; ++z) {
      if (two == kSymbols[z]) {
        return z + 1;
      }
    }
  }
  const std::string one = name.substr(0, 1);
  for (int z = 0; z < 118; ++z) {
    if (one == kSymbols[z]) {
      return z + 1;
    }
  }
  return 0;
}

std::vector<MEAMElement> read_meam_library(std::istream &in,
                                           const std::vector<std::string> &elements) {
  std::vector<MEAMElement> table(elements.size());
  std::vector<bool> found(elements.size(), false);
  const bool all = elements.empty();
  std::vector<MEAMElement> every;

  std::string line;
  std::string entry;
  std::size_t nwords = 0;
  std::size_t lineno = 0;
  std::size_t entry_line = 0;
  auto flush = [&]() {
    const std::string where = "library line " + std::to_string(entry_line);
    const std::vector<std::string> w = split(entry, "' \t\n\r\f");
    entry.clear();
    nwords = 0;
    const std::string &name = w[0];
    std::size_t index = elements.size();
    if (all) {
      for (const MEAMElement &e : every) {
        if (e.name == name) {
          return;
        }
      }
    } else {
      for (std::size_t k = 0; k < elements.size(); ++k) {
        if (elements[k] == name) {
          index = k;
          break;
        }
      }
      if (index == elements.size() || found[index]) {
        return;
      }
    }
    MEAMElement e;
    e.name = name;
    if (!meam_lattice_from_name(w[1], true, &e.lattice)) {
      bad(where + ": unrecognized lattice type " + w[1]);
    }
    const double z = to_double(w[2], where);
    e.z = to_int(w[3], where);
    e.mass = to_double(w[4], where);
    e.alpha = to_double(w[5], where);
    for (int l = 0; l < 4; ++l) {
      e.beta[l] = to_double(w[6 + static_cast<std::size_t>(l)], where);
    }
    e.alat = to_double(w[10], where);
    e.esub = to_double(w[11], where);
    e.asub = to_double(w[12], where);
    for (int l = 0; l < 4; ++l) {
      e.t[l] = to_double(w[13 + static_cast<std::size_t>(l)], where);
    }
    e.rho0 = to_double(w[17], where);
    e.ibar = to_int(w[18], where);
    if (std::fabs(e.t[0] - 1.0) >= 1e-20) {
      bad(where + ": unsupported parameter t0 != 1");
    }
    if (!(e.rho0 > 0.0)) {
      bad(where + ": rho0 must be > 0");
    }
    if (z != static_cast<double>(meam_coordination(e.lattice))) {
      bad(where + ": z does not match the lattice " + w[1]);
    }
    if (all) {
      every.push_back(e);
    } else {
      found[index] = true;
      table[index] = e;
    }
  };

  while (std::getline(in, line)) {
    ++lineno;
    const std::string text = strip_comment(line);
    const std::size_t words = split(text, kWhite).size();
    if (words == 0) {
      continue;
    }
    if (nwords == 0) {
      entry_line = lineno;
    }
    entry += text;
    entry += ' ';
    nwords += words;
    if (nwords >= kFieldsPerElement) {
      flush();
    }
  }
  if (nwords != 0) {
    bad("library line " + std::to_string(entry_line) + ": entry has " +
        std::to_string(nwords) + " fields, expected " + std::to_string(kFieldsPerElement));
  }
  if (all) {
    if (every.empty()) {
      bad("library holds no element");
    }
    return every;
  }
  std::string missing;
  for (std::size_t k = 0; k < elements.size(); ++k) {
    if (!found[k]) {
      missing += " " + elements[k];
    }
  }
  if (!missing.empty()) {
    bad("library lacks the elements:" + missing);
  }
  return table;
}

void read_meam_parameters(std::istream &in, MEAMParameters &parameters) {
  std::string line;
  std::size_t lineno = 0;
  while (std::getline(in, line)) {
    ++lineno;
    const std::string text = strip_comment(line);
    const std::vector<std::string> w = split(text, "=(), '\t\n\r\f");
    if (w.empty()) {
      continue;
    }
    const std::string where = "parameter line " + std::to_string(lineno);
    MEAMParameters::Key key;
    if (!MEAMParameters::key_from_keyword(w[0], &key)) {
      bad(where + ": keyword " + w[0] + " not recognized");
    }
    if (w.size() < 2) {
      bad(where + ": keyword " + w[0] + " has no value");
    }
    const std::size_t nindex = w.size() - 2;
    if (nindex > 3) {
      bad(where + ": too many indices");
    }
    std::vector<int> index;
    for (std::size_t k = 0; k < nindex; ++k) {
      index.push_back(to_int(w[1 + k], where) - 1);
    }
    double value;
    if (key == MEAMParameters::Key::lattce) {
      MEAMLattice lattice;
      if (!meam_lattice_from_name(w.back(), false, &lattice)) {
        bad(where + ": unrecognized lattice type " + w.back());
      }
      value = static_cast<double>(lattice);
    } else {
      value = to_double(w.back(), where);
    }
    try {
      parameters.set(key, value, index);
    } catch (const std::invalid_argument &e) {
      throw std::invalid_argument(std::string(e.what()) + " (" + where + ")");
    }
  }
}

MEAMModel load_meam_model(const std::string &library, const std::string &parameters,
                          const std::vector<std::string> &elements) {
  std::ifstream lib(library);
  if (!lib) {
    bad("cannot open library file " + library);
  }
  MEAMParameters params(read_meam_library(lib, elements));
  if (!parameters.empty() && parameters != "NULL") {
    std::ifstream par(parameters);
    if (!par) {
      bad("cannot open parameter file " + parameters);
    }
    read_meam_parameters(par, params);
  }
  return MEAMModel(std::move(params));
}

} // namespace rgpot
