// MIT License
// Copyright 2023--present rgpot developers

#include <algorithm>
#include <array>
#include <cstdint>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>

#include "rgpot/Tersoff/TersoffSiC.hpp"
#include "rgpot/fortran/FortranPots.hpp"
#include "rgpot/stress.hpp"

namespace rgpot {
namespace fortranpots {

namespace {

extern "C" {
void *rgpot_fortran_workspace_new();
void rgpot_fortran_workspace_free(void *workspace);
int rgpot_fortran_workspace_error(void *workspace, char *buffer,
                                  int buffer_len);

/// Kernels needing only geometry.
int rgpot_sw_force_ws(void *ws, int32_t natoms, const double *positions,
                      const double *cell, double *forces, double *energy,
                      double *strain);
int rgpot_edip_force_ws(void *ws, int32_t natoms, const double *positions,
                        const double *cell, double *forces, double *energy);
int rgpot_lenosky_force_ws(void *ws, int32_t natoms, const double *positions,
                           const double *cell, double *forces,
                           double *energy);
int rgpot_tersoff_force_ws(void *ws, int32_t natoms, const double *positions,
                           const double *cell, double *forces,
                           double *energy);
int rgpot_eam_al_force_ws(void *ws, int32_t natoms, const double *positions,
                          const double *cell, double *forces, double *energy);

/// Kernels dispatching on atomic number.
int rgpot_fehe_force_ws(void *ws, int32_t natoms, const double *positions,
                        const int32_t *atomic_numbers, const double *cell,
                        double *forces, double *energy);
int rgpot_cuh2_force_ws(void *ws, int32_t natoms, const double *positions,
                        const int32_t *atomic_numbers, const double *cell,
                        double *forces, double *energy);
int rgpot_water_h_force_ws(void *ws, int32_t natoms, const double *positions,
                           const int32_t *atomic_numbers, const double *cell,
                           double *forces, double *energy);

// F2: the saved parameter block in each *_capi module is written once
// under std::call_once and read-only afterwards. Python binds only the
// _ws entries (one workspace per potential instance).
void rgpot_sw_ensure_params();
void rgpot_edip_ensure_params();
void rgpot_lenosky_ensure_params();
void rgpot_tersoff_ensure_params();
void rgpot_eam_al_ensure_params();
void rgpot_fehe_ensure_params();
void rgpot_cuh2_ensure_params();
void rgpot_water_h_ensure_params();
}

/// F2: one initialisation for every kernel's saved parameter block.
void ensure_fortran_params() {
  static std::once_flag once;
  std::call_once(once, [] {
    rgpot_sw_ensure_params();
    rgpot_edip_ensure_params();
    rgpot_lenosky_ensure_params();
    rgpot_tersoff_ensure_params();
    rgpot_eam_al_ensure_params();
    rgpot_fehe_ensure_params();
    rgpot_cuh2_ensure_params();
    rgpot_water_h_ensure_params();
  });
}

/// Throw carrying the kernel's own message.
[[noreturn]] void raise(const char *pot, int status, void *ws) {
  std::array<char, 512> buffer{};
  const int written = rgpot_fortran_workspace_error(
      ws, buffer.data(), static_cast<int>(buffer.size()));

  std::string message = std::string(pot) + " potential failed (status " +
                        std::to_string(status) + ")";
  if (written > 0) {
    message += ": ";
    message.append(buffer.data(), static_cast<std::size_t>(written));
  }
  throw std::runtime_error(message);
}

/// True when the Fortran silicon gather is the right kernel. A null
/// species pointer is the historical silicon call.
bool siliconOnly(const ForceInput &in) {
  if (in.atmnrs == nullptr) {
    return true;
  }
  for (size_t i = 0; i < in.nAtoms; ++i) {
    if (in.atmnrs[i] != 14) {
      return false;
    }
  }
  return true;
}

} // namespace

FortranWorkspace::FortranWorkspace() : m_handle(rgpot_fortran_workspace_new()) {
  if (m_handle == nullptr) {
    throw std::bad_alloc();
  }
}

FortranWorkspace::~FortranWorkspace() {
  rgpot_fortran_workspace_free(m_handle);
}

FortranWorkspace::FortranWorkspace(const FortranWorkspace &)
    : FortranWorkspace() {}

FortranWorkspace &FortranWorkspace::operator=(const FortranWorkspace &) {
  return *this; // keep this instance's own table
}

#define RGPOT_FORTRAN_POT_IMPL(ClassName, Entry, Label)                        \
  void ClassName::forceImpl(const ForceInput &in, ForceOut *out) const {       \
    ensure_fortran_params();                                                   \
    const int status = Entry(m_ws.get(), static_cast<int32_t>(in.nAtoms),       \
                             in.pos, in.box, out->F, &out->energy);            \
    if (status != 0) {                                                         \
      raise(Label, status, m_ws.get());                                        \
    }                                                                          \
    out->variance = 0.0;                                                       \
  }

RGPOT_FORTRAN_POT_IMPL(EDIPPot, rgpot_edip_force_ws, "EDIP")
RGPOT_FORTRAN_POT_IMPL(LenoskyPot, rgpot_lenosky_force_ws, "Lenosky")
RGPOT_FORTRAN_POT_IMPL(EAMAlPot, rgpot_eam_al_force_ws, "EAM-Al")

#undef RGPOT_FORTRAN_POT_IMPL

void SWPot::forceImpl(const ForceInput &in, ForceOut *out) const {
  ensure_fortran_params();
  double strain[6] = {};
  const int status =
      rgpot_sw_force_ws(m_ws.get(), static_cast<int32_t>(in.nAtoms), in.pos,
                        in.box, out->F, &out->energy, strain);
  if (status != 0) {
    raise("SW", status, m_ws.get());
  }
  out->variance = 0.0;
  publishCauchyStress(out, strain, cellVolume(in.box));
}

void SWPot::forceBatchImpl(const ForceBatch &batch) const {
  ensure_fortran_params();
  for (size_t i = 0; i < batch.nSystems; ++i) {
    const ForceInput &in = batch.in[i];
    ForceOut &out = batch.out[i];
    const int status =
        rgpot_sw_force_ws(m_ws.get(), static_cast<int32_t>(in.nAtoms), in.pos,
                          in.box, out.F, &out.energy, nullptr);
    if (status != 0) {
      raise("SW", status, m_ws.get());
    }
    out.variance = 0.0;
  }
}

void TersoffPot::forceImpl(const ForceInput &in, ForceOut *out) const {
  // The C++ kernel searches with its internal skin unless setHeldSkin
  // was given one. On periodic cells of 512 and 4096 atoms a rebuild is
  // slower than the warm Fortran gather, so all-silicon stays
  // there. Any other Z needs the 1989 silicon-carbon parameters, which the
  // gather does not have.
  if (siliconOnly(in)) {
    ensure_fortran_params();
    const int status = rgpot_tersoff_force_ws(
        m_ws.get(), static_cast<int32_t>(in.nAtoms), in.pos, in.box, out->F,
        &out->energy);
    if (status != 0) {
      raise("Tersoff", status, m_ws.get());
    }
    out->variance = 0.0;
  } else {
    m_sic.forceImpl(in, out);
  }
  // The gather writes no stress and caps().stress is false, so no call
  // reports one, and a reused output keeps no earlier tensor.
  out->has_stress = 0;
  std::fill(out->stress, out->stress + 9, 0.0);
}

#define RGPOT_FORTRAN_SPECIES_POT_IMPL(ClassName, Entry, Label)                \
  void ClassName::forceImpl(const ForceInput &in, ForceOut *out) const {       \
    ensure_fortran_params();                                                   \
    const int status =                                                         \
        Entry(m_ws.get(), static_cast<int32_t>(in.nAtoms), in.pos, in.atmnrs,  \
              in.box, out->F, &out->energy);                                   \
    if (status != 0) {                                                         \
      raise(Label, status, m_ws.get());                                        \
    }                                                                          \
    out->variance = 0.0;                                                       \
  }

RGPOT_FORTRAN_SPECIES_POT_IMPL(FeHePot, rgpot_fehe_force_ws, "FeHe")
RGPOT_FORTRAN_SPECIES_POT_IMPL(CuH2Pot, rgpot_cuh2_force_ws, "CuH2")
RGPOT_FORTRAN_SPECIES_POT_IMPL(WaterHPot, rgpot_water_h_force_ws, "Water-H")

#undef RGPOT_FORTRAN_SPECIES_POT_IMPL

} // namespace fortranpots
} // namespace rgpot
