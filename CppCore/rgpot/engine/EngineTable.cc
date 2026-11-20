// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/engine_c_abi.h"

#include "rgpot/engine/EngineTable.hpp"
#include "rgpot/engine/Sha256.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <limits>
#include <mutex>
#include <string>
#include <vector>

#ifndef _WIN32
#include <dlfcn.h>
#endif

#ifndef RGPOT_ENGINE_PROJECT_VERSION
#define RGPOT_ENGINE_PROJECT_VERSION "unknown"
#endif

namespace {

// When this engine is called from a Python host (pyeonclient / nanobind) the
// GIL is often still held. Torch autograd refuses that. Soft-resolve CPython
// thread APIs so pure C++ hosts (eonclient binary) need no libpython link.
//
// Only SaveThread when PyGILState_Check() is true: nested release (Job.run
// already dropped the GIL) must not call SaveThread again.
struct SoftGilRelease {
#ifndef _WIN32
  using CheckFn = int (*)();
  using SaveFn = void *(*)();
  using RestoreFn = void (*)(void *);
  void *tstate = nullptr;
  RestoreFn restore = nullptr;
  SoftGilRelease() {
    auto *check =
        reinterpret_cast<CheckFn>(dlsym(RTLD_DEFAULT, "PyGILState_Check"));
    auto *save =
        reinterpret_cast<SaveFn>(dlsym(RTLD_DEFAULT, "PyEval_SaveThread"));
    restore = reinterpret_cast<RestoreFn>(
        dlsym(RTLD_DEFAULT, "PyEval_RestoreThread"));
    if (check && check() && save && restore) {
      tstate = save();
    }
  }
  ~SoftGilRelease() {
    if (tstate && restore) {
      restore(tstate);
    }
  }
#endif
  SoftGilRelease(const SoftGilRelease &) = delete;
  SoftGilRelease &operator=(const SoftGilRelease &) = delete;
};

/// One call at a time across every handle of this library (ProcessSerial).
std::mutex &process_mutex() {
  static std::mutex m;
  return m;
}

void copy_message(const std::string &message, char *buf, size_t len) {
  if (!buf || len == 0) {
    return;
  }
  const size_t n = message.size() < len - 1 ? message.size() : len - 1;
  std::memcpy(buf, message.data(), n);
  buf[n] = '\0';
}

} // namespace

struct RgpotEnginePot {
  std::unique_ptr<rgpot::engine::Backend> backend;
  rgpot::PotCaps caps;

  mutable std::mutex error_mutex;
  std::string error;

  /// Serializes calls on this handle when the backend is not shared.
  std::mutex call_mutex;

  mutable std::mutex digest_mutex;
  mutable bool digest_done = false;
  mutable std::string digest;

  void set_error(const std::string &message) {
    std::lock_guard<std::mutex> lock(error_mutex);
    error = message;
  }
  void clear_error() {
    std::lock_guard<std::mutex> lock(error_mutex);
    error.clear();
  }

  /// Holds the locks the backend's reentrancy asks for.
  struct CallGuard {
    std::unique_lock<std::mutex> handle;
    std::unique_lock<std::mutex> process;
    explicit CallGuard(RgpotEnginePot &p) {
      if (p.caps.reentrancy == rgpot::Reentrancy::ProcessSerial) {
        process = std::unique_lock<std::mutex>(process_mutex());
      }
      if (p.caps.reentrancy != rgpot::Reentrancy::SharedInstance) {
        handle = std::unique_lock<std::mutex>(p.call_mutex);
      }
    }
  };
};

namespace {

int fail(RgpotEnginePot *pot, int code, const std::string &message) {
  if (code == 2) {
    std::fprintf(stderr, "rgpot_engine(%s): %s\n",
                 rgpot::engine::backend_name(), message.c_str());
  }
  if (pot) {
    pot->set_error(message);
  }
  return code;
}

} // namespace

extern "C" {

int rgpot_engine_abi_version(void) { return RGPOT_ENGINE_ABI_VERSION; }

int rgpot_engine_available(void) { return 1; }

const char *rgpot_engine_name(void) { return rgpot::engine::backend_name(); }

const char *rgpot_engine_version(void) {
  static const std::string version = std::string(rgpot::engine::backend_name()) +
                                     "/" RGPOT_ENGINE_PROJECT_VERSION;
  return version.c_str();
}

RgpotEnginePot *rgpot_engine_create(const void *config, size_t config_len,
                                    char *errbuf, size_t errlen) {
  const std::string name = rgpot::engine::backend_name();
  if (!config || config_len == 0 || config_len % 8 != 0) {
    copy_message("rgpot_engine_create(" + name +
                     "): a word-aligned Cap'n Proto params message is required",
                 errbuf, errlen);
    return nullptr;
  }
  try {
    rgpot::engine::Created created = rgpot::engine::create(config, config_len);
    if (!created.backend || !created.backend->pot) {
      copy_message(created.error.empty()
                       ? "rgpot_engine_create(" + name + "): no potential"
                       : created.error,
                   errbuf, errlen);
      return nullptr;
    }
    auto pot = std::make_unique<RgpotEnginePot>();
    pot->caps = created.backend->pot->caps();
    pot->backend = std::move(created.backend);
    return pot.release();
  } catch (const std::exception &e) {
    copy_message(e.what(), errbuf, errlen);
    return nullptr;
  } catch (...) {
    copy_message("rgpot_engine_create(" + name + "): unknown exception", errbuf,
                 errlen);
    return nullptr;
  }
}

void rgpot_engine_destroy(RgpotEnginePot *pot) { delete pot; }

int rgpot_engine_force(RgpotEnginePot *pot, long nAtoms,
                       const double *positions, const int *atomicNrs,
                       double *forces, double *energy, double *variance,
                       const double *box,
                       rgpot_engine_coord_transform transform,
                       void *transform_user) {
  if (!pot || !pot->backend || !positions || !atomicNrs || !forces ||
      !energy || !box || nAtoms <= 0) {
    return fail(pot, 1, "invalid argument");
  }
  try {
    RgpotEnginePot::CallGuard guard(*pot);
    SoftGilRelease no_gil;
    // Scratch copies: a host transform must never mutate caller buffers.
    std::vector<double> R(positions, positions + 3 * nAtoms);
    std::vector<double> cell(box, box + 9);
    if (transform) {
      const int rc = transform(transform_user, nAtoms, R.data(), cell.data());
      if (rc != 0) {
        return fail(pot, rc,
                    "coordinate transform returned " + std::to_string(rc));
      }
    }
    std::vector<double> Fbuf(static_cast<size_t>(nAtoms) * 3);
    const rgpot::ForceInput in{static_cast<size_t>(nAtoms), R.data(), atomicNrs,
                               cell.data()};
    rgpot::ForceOut out{};
    out.F = Fbuf.data();
    const rgpot::ForceBatch batch{1, &in, &out};
    pot->backend->pot->forceBatch(batch);
    *energy = out.energy;
    if (variance) {
      *variance = out.variance;
    }
    std::memcpy(forces, Fbuf.data(), Fbuf.size() * sizeof(double));
    pot->clear_error();
    return 0;
  } catch (const std::exception &e) {
    return fail(pot, 2, e.what());
  } catch (...) {
    return fail(pot, 2, "unknown exception");
  }
}

int rgpot_engine_hvp(RgpotEnginePot *pot, long nAtoms,
                     const double *positions, const int *atomicNrs,
                     const double *direction, double *hv_out,
                     const double *box) {
  if (!pot || !pot->backend) {
    return RGPOT_ENGINE_ERR_INVALID;
  }
  if (!pot->caps.analyticHvp) {
    return fail(pot, RGPOT_ENGINE_ERR_UNSUPPORTED,
                "potential has no analytic Hessian-vector product");
  }
  if (!positions || !atomicNrs || !direction || !hv_out || !box ||
      nAtoms <= 0 ||
      static_cast<unsigned long>(nAtoms) >
          std::numeric_limits<size_t>::max() / (3 * sizeof(double))) {
    return fail(pot, RGPOT_ENGINE_ERR_INVALID, "invalid HVP argument");
  }
  const auto db = reinterpret_cast<std::uintptr_t>(direction);
  const auto hb = reinterpret_cast<std::uintptr_t>(hv_out);
  const std::uintptr_t bytes =
      static_cast<std::uintptr_t>(nAtoms) * 3 * sizeof(double);
  const auto overlaps = [hb, bytes](const void *input,
                                    std::uintptr_t inputBytes) {
    const auto begin = reinterpret_cast<std::uintptr_t>(input);
    return hb < begin + inputBytes && begin < hb + bytes;
  };
  if (overlaps(direction, bytes) || overlaps(positions, bytes) ||
      overlaps(atomicNrs, static_cast<std::uintptr_t>(nAtoms) * sizeof(int)) ||
      overlaps(box, 9 * sizeof(double))) {
    return fail(pot, RGPOT_ENGINE_ERR_INVALID,
                "HVP output overlaps an input buffer");
  }
  try {
    RgpotEnginePot::CallGuard guard(*pot);
    SoftGilRelease no_gil;
    const rgpot::ForceInput in{static_cast<size_t>(nAtoms),
                               positions, atomicNrs, box};
    pot->backend->pot->hessianVectorImpl(in, direction, hv_out);
    pot->clear_error();
    return RGPOT_ENGINE_OK;
  } catch (const std::exception &e) {
    return fail(pot, RGPOT_ENGINE_ERR_EVAL, e.what());
  } catch (...) {
    return fail(pot, RGPOT_ENGINE_ERR_EVAL, "unknown exception");
  }
}

int rgpot_engine_force_batch(RgpotEnginePot *pot, long nSystems, long nAtoms,
                             const double *positions, const int *atomicNrs,
                             double *forces, double *energies,
                             double *variances, const double *boxes,
                             rgpot_engine_coord_transform transform,
                             void *transform_user) {
  if (!pot || !pot->backend || !positions || !atomicNrs || !forces ||
      !energies || !boxes || nSystems <= 0 || nAtoms <= 0) {
    return fail(pot, 1, "invalid argument (batch)");
  }
  try {
    RgpotEnginePot::CallGuard guard(*pot);
    SoftGilRelease no_gil;
    const size_t B = static_cast<size_t>(nSystems);
    const size_t stride = static_cast<size_t>(nAtoms) * 3;
    std::vector<double> R(positions, positions + stride * B);
    std::vector<double> cells(boxes, boxes + 9 * B);
    if (transform) {
      for (size_t i = 0; i < B; ++i) {
        const int rc = transform(transform_user, nAtoms, R.data() + i * stride,
                                 cells.data() + i * 9);
        if (rc != 0) {
          return fail(pot, rc,
                      "coordinate transform returned " + std::to_string(rc));
        }
      }
    }
    std::vector<rgpot::ForceInput> in;
    std::vector<rgpot::ForceOut> out(B);
    in.reserve(B);
    for (size_t i = 0; i < B; ++i) {
      in.push_back(rgpot::ForceInput{static_cast<size_t>(nAtoms),
                                     R.data() + i * stride, atomicNrs,
                                     cells.data() + i * 9});
      out[i] = rgpot::ForceOut{};
      out[i].F = forces + i * stride;
    }
    const rgpot::ForceBatch batch{B, in.data(), out.data()};
    pot->backend->pot->forceBatch(batch);
    for (size_t i = 0; i < B; ++i) {
      energies[i] = out[i].energy;
      if (variances) {
        variances[i] = out[i].variance;
      }
    }
    pot->clear_error();
    return 0;
  } catch (const std::exception &e) {
    return fail(pot, 2, e.what());
  } catch (...) {
    return fail(pot, 2, "unknown exception");
  }
}

int rgpot_engine_abi_compatible(const RgpotEngineAbiStamp *stamp) {
  if (!stamp) {
    return 0;
  }
  return stamp->abi_major == RGPOT_ENGINE_ABI_VERSION &&
                 stamp->layout_revision == RGPOT_ENGINE_ABI_LAYOUT_REVISION &&
                 stamp->abi_minor <= RGPOT_ENGINE_ABI_MINOR
             ? 1
             : 0;
}

int rgpot_engine_caps(const RgpotEnginePot *pot, RgpotEngineCaps *out) {
  if (!pot || !pot->backend || !out) {
    return 1;
  }
  RgpotEngineCaps filled;
  std::memset(&filled, 0, sizeof(filled));
  filled.size = sizeof(RgpotEngineCaps);
  filled.reentrancy = static_cast<int>(pot->caps.reentrancy);
  filled.per_image_instances = pot->caps.perImageInstances ? 1 : 0;
  filled.batched = pot->caps.batched ? 1 : 0;
  filled.periodic = pot->caps.periodic ? 1 : 0;
  filled.stress = pot->caps.stress ? 1 : 0;
  filled.group_collective = pot->caps.groupCollective ? 1 : 0;
  filled.orbital_selection = pot->backend->select_orbitals ? 1 : 0;
  filled.orbital_copy = pot->backend->copy_orbitals ? 1 : 0;
  filled.analytic_hvp = pot->caps.analyticHvp ? 1 : 0;
  // The host's struct may be older (smaller) than this engine's: write only
  // the bytes it announced room for.
  const unsigned room = out->size < sizeof(filled) ? out->size : sizeof(filled);
  if (room < sizeof(unsigned)) {
    return 1;
  }
  std::memcpy(out, &filled, room);
  out->size = room;
  return 0;
}

size_t rgpot_engine_last_error(const RgpotEnginePot *pot, char *buf,
                               size_t len) {
  if (!pot) {
    return 0;
  }
  std::lock_guard<std::mutex> lock(pot->error_mutex);
  copy_message(pot->error, buf, len);
  return pot->error.size();
}

size_t rgpot_engine_model_digest(const RgpotEnginePot *pot, char *buf,
                                 size_t len) {
  if (!pot || !pot->backend || pot->backend->model_path.empty()) {
    if (buf && len > 0) {
      buf[0] = '\0';
    }
    return 0;
  }
  std::lock_guard<std::mutex> lock(pot->digest_mutex);
  if (!pot->digest_done) {
    pot->digest = rgpot::engine::Sha256::of_file(pot->backend->model_path);
    pot->digest_done = true;
  }
  copy_message(pot->digest, buf, len);
  return pot->digest.size();
}

int rgpot_engine_set_num_threads(RgpotEnginePot *pot, int intra_op,
                                 int inter_op) {
  if (!pot || !pot->backend) {
    return 1;
  }
  if (!pot->backend->set_num_threads) {
    return fail(pot, 3, "backend has no tensor runtime");
  }
  try {
    pot->backend->set_num_threads(intra_op, inter_op);
    pot->clear_error();
    return 0;
  } catch (const std::exception &e) {
    return fail(pot, 2, e.what());
  } catch (...) {
    return fail(pot, 2, "unknown exception");
  }
}

int rgpot_engine_set_charge_spin(RgpotEnginePot *pot, int charge, int spin) {
  if (!pot || !pot->backend || spin < 1) {
    return 1;
  }
  if (!pot->backend->set_charge_spin) {
    return fail(pot, 3, "backend has no charge or spin to change");
  }
  try {
    pot->backend->set_charge_spin(charge, spin);
    pot->clear_error();
    return 0;
  } catch (const rgpot::engine::Unsupported &e) {
    return fail(pot, 3, e.what());
  } catch (const std::exception &e) {
    return fail(pot, 2, e.what());
  } catch (...) {
    return fail(pot, 2, "unknown exception");
  }
}

int rgpot_engine_select_orbitals(RgpotEnginePot *pot, long long key) {
  if (!pot || !pot->backend) {
    return 1;
  }
  if (!pot->backend->select_orbitals) {
    return fail(pot, 3, "backend has no orbital selection");
  }
  try {
    pot->backend->select_orbitals(key);
    pot->clear_error();
    return 0;
  } catch (const rgpot::engine::Unsupported &e) {
    return fail(pot, 3, e.what());
  } catch (const std::exception &e) {
    return fail(pot, 2, e.what());
  } catch (...) {
    return fail(pot, 2, "unknown exception");
  }
}

int rgpot_engine_copy_orbitals(RgpotEnginePot *pot, long long from_key,
                               long long to_key) {
  if (!pot || !pot->backend) {
    return 1;
  }
  if (!pot->backend->copy_orbitals) {
    return fail(pot, 3, "backend has no orbital copy");
  }
  try {
    pot->backend->copy_orbitals(from_key, to_key);
    pot->clear_error();
    return 0;
  } catch (const rgpot::engine::Unsupported &e) {
    return fail(pot, 3, e.what());
  } catch (const std::exception &e) {
    return fail(pot, 2, e.what());
  } catch (...) {
    return fail(pot, 2, "unknown exception");
  }
}

} // extern "C"
