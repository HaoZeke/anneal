/**
 * @brief Definitions for supported potential energy surface types.
 *
 * This file defines the central @c PotType enumeration used throughout  the
 * library to identify and instantiate specific potentials.
 *
 * @note The implementations here are intentionally limited for demonstration
 * only. [eOn](https://eondocs.org) has a server component with a much larger
 * set of supported potentials.
 */

#pragma once
// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/base_types.hpp"

namespace rgpot {

/**
 * @brief Supported potential energy surface types.
 *
 * This enumeration is used by the factory and registry systems to dispatch
 * calculation requests to the appropriate implementation.
 */
enum class PotType {
  UNKNOWN = 0, //!<  The type is not defined or is invalid.
  CuH2,        //!<  Copper-Hydrogen EAM potential.
  LJ,          //!<  Standard 12-6 Lennard-Jones pairwise potential.
  XTB,         //!<  GFN tight-binding via xtb (GFNFF/GFN0/GFN1/GFN2).
  TBLite,      //!<  GFN tight-binding via tblite (GFN1/GFN2/IPEA1).
  Metatomic,   //!<  ML atomistic models via metatomic/PyTorch.
  NWChem,      //!<  QM via runtime-loaded NWChem C ABI engine.
  CPMD,        //!<  PW-DFT via runtime-loaded CPMD C ABI engine.
  Morse,       //!<  Pairwise Morse potential (platinum parameters).
  LJCluster,   //!<  Free-boundary 12-6 Lennard-Jones for clusters.
  ZBL,         //!<  Screened nuclear repulsion (Ziegler-Biersack-Littmark).
  SWSi,        //!<  Stillinger-Weber silicon. eOn epsilon, 2.3149945 eV.
  EDIP,        //!<  Environment-dependent interatomic potential (silicon).
  LenoskySi,   //!<  Lenosky tight-binding-fit silicon.
  TersoffSi,   //!<  Tersoff silicon; 1989 Si-C when any Z is not 14.
  EAMAl,       //!<  Double-exponential EAM aluminium.
  FeHe,        //!<  Fe-He embedded atom with species-dependent terms.
  WaterH,      //!<  TIP4P water with an interacting hydrogen atom.
  D3,          //!<  Grimme DFT-D3 via s-dftd3 (in-process Potential summand).
  D4,          //!<  Grimme DFT-D4 via dftd4 (in-process Potential summand).
  Expr,        //!<  Lepton expression over named Potential children.
  Uma,         //!<  UMA / OMol via vesin + AOTInductor .pt2.
  Skala,       //!<  Skala XC on NWChem DFT (libnwchemc).
  MOPAC,       //!<  OpenMOPAC via runtime-loaded mopacc C ABI engine.
  EMT,         //!<  Effective medium theory (Asap). Copper unless rasmussen.
  EAMCell,     //!<  Cell-list embedded-atom potential.
  TIP4P,       //!<  Flexible TIP4P water. Built with -Dwith_water=true.
  SPCE,        //!<  Flexible SPC/E water. Built with -Dwith_water=true.
  TIP4PPt,     //!<  TIP4P water on platinum. Built with -Dwith_water=true.
  IPI,         //!<  i-PI socket server. Built with -Dwith_ipi (POSIX).
  SuttonChen,  //!<  Sutton-Chen 10-8. Reduced units by default.
  SWSi1985,    //!<  Stillinger-Weber silicon, 1985 parameters.
  EAMTable,    //!<  EAM / Finnis-Sinclair tables. -Dwith_eam_table=true.
  Lammps,      //!<  Shifted Lennard-Jones via the engine plugin.
  Gromacs,     //!<  Shifted Lennard-Jones; nm and kJ/mol stored once.
  OpenMM,      //!<  Shifted Lennard-Jones; nm and kJ/mol stored once.
  CP2K,        //!<  CP2K Quickstep via the libcp2k shim. -Dwith_cp2k.
  LrCoulomb,   //!<  Ewald, smooth PME, or damped shifted force.
  LrDispersion, //!<  D3(BJ), D4, MBD, or SO3LR QDO dispersion.
  LrNlh,       //!<  SO3LR NLH short-range repulsion.
  LrSum,       //!<  Weighted sum of child potentials (rgpot::SumPot).
  Eckart,
  MullerBrown,
  ModifiedMullerBrown,
  LepsHH2,
  LepsHCl2,
  LepsFH2,
  DoubleWell,
  CoupledProtonWell,
  CBE,          //!< External cbe-pes frontend.
  D2,           //!<  Grimme DFT-D2 lattice sum (rgpot::D2Pot).
  D2Cpmd        //!<  CPMD's minimum-image empirical D2 (rgpot::D2CpmdPot).
};

} // namespace rgpot
