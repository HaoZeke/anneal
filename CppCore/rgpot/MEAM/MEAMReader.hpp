#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @file MEAMReader.hpp
 * @brief Readers for the LAMMPS MEAM library and parameter files.
 *
 * The library file lists one element per entry of 19 whitespace separated
 * fields, which may span lines:
 *
 *   elt lat z ielement atwt alpha b0 b1 b2 b3 alat esub asub
 *   t0 t1 t2 t3 rozero ibar
 *
 * `#` starts a comment and quotes around the name and lattice are
 * dropped. `ielement` is the atomic number in some files and 1 for every
 * element in others, so MEAMPot takes atomic numbers from the element
 * names. `z` is checked against the coordination number of `lat` and
 * otherwise ignored. The parameter file holds `keyword(indices) = value`
 * lines (see MEAMParameters::Key), with one based element indices in the
 * order of the element list given to the library reader.
 */

#include <iosfwd>
#include <string>
#include <vector>

#include "rgpot/MEAM/MEAMModel.hpp"

namespace rgpot {

/**
 * @brief Read library entries.
 * @param elements Names to take, in the order that sets the element
 *        indices. Empty takes every entry of the file in file order. A
 *        name that occurs twice in the file keeps its first entry.
 * @throws std::invalid_argument for a missing name, a short or malformed
 *         entry, an unknown lattice, t0 other than 1, a non positive rho0
 *         or a coordination number that does not match the lattice.
 */
std::vector<MEAMElement> read_meam_library(std::istream &in,
                                           const std::vector<std::string> &elements);

/// Atomic number of the element a library name stands for, from the leading
/// chemical symbol: 26 for "Fe" and for "FeS". Returns 0 when the name does
/// not start with a symbol.
int meam_atomic_number(const std::string &name);

/// Apply a parameter file to `parameters`. Throws std::invalid_argument
/// with the line number for an unknown keyword, a malformed line, an index
/// outside the element table or a value the model does not allow.
void read_meam_parameters(std::istream &in, MEAMParameters &parameters);

/// Open and read a library and an optional parameter file ("" and "NULL"
/// mean none) and build the model. Throws std::invalid_argument when a file
/// cannot be opened.
MEAMModel load_meam_model(const std::string &library, const std::string &parameters,
                          const std::vector<std::string> &elements);

} // namespace rgpot
