#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief CP2K input decks from CP2KParams and one geometry.
 *
 * Pure text functions without libcp2k, so the deck CP2KPot hands the shim
 * can be tested on its own.
 */

#include "rgpot/rpc/Potentials.capnp.h"

#include <cstddef>
#include <string>
#include <vector>

namespace rgpot::cp2k {

/// Token a template replaces with one `&COORD` row per atom (Angstrom).
inline constexpr const char *kCoordToken = "@RGPOT_COORD@";
/// Token a template replaces with the `&CELL` `A`, `B` and `C` rows
/// (Angstrom).
inline constexpr const char *kCellToken = "@RGPOT_CELL@";

/// Element symbol of atomic number @p z (1..118). Throws
/// std::invalid_argument outside that range.
std::string elementSymbol(int z);

/// Empty when @p deck contains both tokens, else the reason it does not.
std::string checkTemplate(const std::string &deck);

/// `Symbol x y z` rows, one per atom, positions in Angstrom.
std::string coordRows(std::size_t n_atoms, const double *pos_ang,
                      const int *atomic_numbers);

/// `A`, `B` and `C` rows from a row-major 3x3 cell in Angstrom (rows are
/// the cell vectors).
std::string cellRows(const double *box_ang);

/// The deck the typed fields describe, with both tokens in place of the
/// geometry. One `&KIND` block per element, in order of first appearance.
std::string defaultTemplate(const ::CP2KParams::Reader &params,
                            const std::vector<int> &atomic_numbers);

/// inputTemplate when it is set, else defaultTemplate, with the tokens
/// replaced by this geometry. Throws std::invalid_argument when the
/// template lacks a token or an atomic number has no symbol.
std::string renderDeck(const ::CP2KParams::Reader &params, std::size_t n_atoms,
                       const double *pos_ang, const int *atomic_numbers,
                       const double *box_ang);

} // namespace rgpot::cp2k
