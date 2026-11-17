// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/CP2KPot/CP2KDeck.hpp"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string_view>

namespace rgpot::cp2k {

namespace {

constexpr std::array<const char *, 118> kSymbols = {
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

// Fixed notation with 12 decimals: sub-femtometre resolution in Angstrom.
// libcp2k receives the exact positions through cp2k_set_positions before
// every evaluation, so the deck only names the starting geometry.
std::string fixed(double v) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.12f", v);
  return buf;
}

// Shortest text that reads back as the same double, e.g. 650 or 1e-09.
std::string general(double v) {
  char buf[64];
  for (int digits = 15; digits <= 17; ++digits) {
    std::snprintf(buf, sizeof(buf), "%.*g", digits, v);
    if (std::strtod(buf, nullptr) == v)
      break;
  }
  return buf;
}

std::string replaceOnce(std::string text, std::string_view token,
                        const std::string &with) {
  const auto at = text.find(token);
  if (at != std::string::npos)
    text.replace(at, token.size(), with);
  return text;
}

} // namespace

std::string elementSymbol(int z) {
  if (z < 1 || z > static_cast<int>(kSymbols.size()))
    throw std::invalid_argument("CP2K deck: no element for atomic number " +
                                std::to_string(z));
  return kSymbols[static_cast<std::size_t>(z - 1)];
}

std::string checkTemplate(const std::string &deck) {
  const bool coord = deck.find(kCoordToken) != std::string::npos;
  const bool cell = deck.find(kCellToken) != std::string::npos;
  if (coord && cell)
    return {};
  std::string missing;
  if (!coord)
    missing = kCoordToken;
  if (!cell)
    missing += missing.empty() ? kCellToken : std::string(" and ") + kCellToken;
  return "CP2K inputTemplate lacks " + missing;
}

std::string coordRows(std::size_t n_atoms, const double *pos_ang,
                      const int *atomic_numbers) {
  std::string out;
  for (std::size_t i = 0; i < n_atoms; ++i) {
    out += elementSymbol(atomic_numbers[i]);
    for (std::size_t k = 0; k < 3; ++k)
      out += " " + fixed(pos_ang[3 * i + k]);
    if (i + 1 < n_atoms)
      out += "\n";
  }
  return out;
}

std::string cellRows(const double *box_ang) {
  static constexpr const char *kNames[3] = {"A", "B", "C"};
  std::string out;
  for (std::size_t r = 0; r < 3; ++r) {
    out += kNames[r];
    for (std::size_t k = 0; k < 3; ++k)
      out += " " + fixed(box_ang[3 * r + k]);
    if (r < 2)
      out += "\n";
  }
  return out;
}

std::string defaultTemplate(const ::CP2KParams::Reader &params,
                            const std::vector<int> &atomic_numbers) {
  std::string d;
  d += "&GLOBAL\n";
  d += "  PROJECT rgpot\n";
  d += "  PRINT_LEVEL LOW\n";
  d += "&END GLOBAL\n";
  d += "&FORCE_EVAL\n";
  d += "  METHOD Quickstep\n";
  d += "  &DFT\n";
  d +=
      "    BASIS_SET_FILE_NAME " + std::string(params.getBasisSetFile()) + "\n";
  d += "    POTENTIAL_FILE_NAME " + std::string(params.getPotentialFile()) +
       "\n";
  d += "    CHARGE " + std::to_string(params.getCharge()) + "\n";
  d += "    MULTIPLICITY " + std::to_string(params.getMultiplicity()) + "\n";
  if (params.getUks())
    d += "    UKS .TRUE.\n";
  d += "    &MGRID\n";
  d += "      CUTOFF " + general(params.getCutoffRy()) + "\n";
  d += "      REL_CUTOFF " + general(params.getRelCutoffRy()) + "\n";
  d += "    &END MGRID\n";
  d += "    &SCF\n";
  d += "      SCF_GUESS ATOMIC\n";
  d += "      EPS_SCF " + general(params.getEpsScf()) + "\n";
  d += "      MAX_SCF " + std::to_string(params.getMaxScf()) + "\n";
  // CP2K aborts the whole MPI world on an unconverged SCF unless told
  // otherwise. The shim detects the failure and reports it instead.
  d += "      IGNORE_CONVERGENCE_FAILURE .TRUE.\n";
  if (params.getOt()) {
    d += "      &OT\n";
    d += "        MINIMIZER DIIS\n";
    d += "        PRECONDITIONER FULL_SINGLE_INVERSE\n";
    d += "      &END OT\n";
  }
  if (params.getOuterMaxScf() > 0) {
    d += "      &OUTER_SCF\n";
    d += "        EPS_SCF " + general(params.getEpsScf()) + "\n";
    d += "        MAX_SCF " + std::to_string(params.getOuterMaxScf()) + "\n";
    d += "      &END OUTER_SCF\n";
  }
  d += "      &PRINT\n";
  d += "        &RESTART OFF\n";
  d += "        &END RESTART\n";
  d += "      &END PRINT\n";
  d += "    &END SCF\n";
  d += "    &XC\n";
  d += "      &XC_FUNCTIONAL " + std::string(params.getFunctional()) + "\n";
  d += "      &END XC_FUNCTIONAL\n";
  d += "    &END XC\n";
  for (const auto section : params.getExtraDftSections())
    d += std::string(section) + "\n";
  d += "  &END DFT\n";
  d += "  &SUBSYS\n";
  d += "    &CELL\n";
  d += std::string(kCellToken) + "\n";
  d += "      PERIODIC " + std::string(params.getPeriodic()) + "\n";
  d += "    &END CELL\n";
  d += "    &COORD\n";
  d += std::string(kCoordToken) + "\n";
  d += "    &END COORD\n";
  std::vector<int> seen;
  for (const int z : atomic_numbers) {
    bool known = false;
    for (const int s : seen)
      known = known || s == z;
    if (known)
      continue;
    seen.push_back(z);
    d += "    &KIND " + elementSymbol(z) + "\n";
    d += "      BASIS_SET " + std::string(params.getBasisSet()) + "\n";
    d += "      POTENTIAL " + std::string(params.getPseudopotential()) + "\n";
    d += "    &END KIND\n";
  }
  d += "  &END SUBSYS\n";
  for (const auto section : params.getExtraForceEvalSections())
    d += std::string(section) + "\n";
  d += "&END FORCE_EVAL\n";
  return d;
}

std::string renderDeck(const ::CP2KParams::Reader &params, std::size_t n_atoms,
                       const double *pos_ang, const int *atomic_numbers,
                       const double *box_ang) {
  std::string deck;
  if (params.getInputTemplate().size() > 0) {
    deck = params.getInputTemplate();
    const std::string why = checkTemplate(deck);
    if (!why.empty())
      throw std::invalid_argument(why);
  } else {
    deck = defaultTemplate(
        params, std::vector<int>(atomic_numbers, atomic_numbers + n_atoms));
  }
  deck = replaceOnce(std::move(deck), kCoordToken,
                     coordRows(n_atoms, pos_ang, atomic_numbers));
  return replaceOnce(std::move(deck), kCellToken, cellRows(box_ang));
}

} // namespace rgpot::cp2k
