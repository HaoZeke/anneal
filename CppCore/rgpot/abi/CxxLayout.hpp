#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @file CxxLayout.hpp
 * @brief Layout stamp of the C++ types that cross the librgpot boundary.
 *
 * The header-only classes (@c Potential, the force structures, the pair
 * potentials) are compiled into the consumer, while their out-of-line members
 * live in the library. A library built from another revision of these
 * headers, or with another value of @c RGPOT_HAS_CACHE, runs with the wrong
 * member offsets or the wrong vtable and fails far from the cause.
 *
 * The stamp folds @c sizeof, @c alignof and @c offsetof of every such type
 * into one 64 bit hash. The hash is a @c constexpr of this header, so a
 * consumer holds what it compiled against; the library exports the same
 * computation through @c rgpot_cxx_layout_stamp so the consumer can read what
 * it loaded.
 *
 * @code
 * if (!rgpot::abi::layout_compatible()) {
 *   // refuse to run: the library differs from these headers
 * }
 * @endcode
 *
 * The covered types, the rule for @c kCxxLayoutRevision and the
 * virtual-function counters are documented in
 * @c docs/orgmode/explanation/abi-stamps.org.
 */

#include <array>
#include <cstddef>
#include <cstdint>

#include "rgpot/ForceStructs.hpp"
#include "rgpot/LennardJones/LJClusterPot.hpp"
#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/Morse/MorsePot.hpp"
#include "rgpot/CalculatorGroup.hpp"
#include "rgpot/Potential.hpp"
#include "rgpot/ZBL/ZBLPot.hpp"
#include "rgpot/pot_caps.hpp"
#include "rgpot/pot_types.hpp"

#ifdef RGPOT_HAS_CACHE
#include "rgpot/PotentialCache.hpp"
#endif

#if defined(_WIN32) && defined(RGPOT_CXX_LAYOUT_BUILDING)
#define RGPOT_CXX_LAYOUT_API __declspec(dllexport)
#else
#define RGPOT_CXX_LAYOUT_API
#endif

// Compile-time platform tag: os, arch, standard library, data model.
// kCxxSoversionTable is keyed by the string these pieces form.
#if defined(_WIN32)
#define RGPOT_CXX_OS_NAME "windows"
#define RGPOT_OS_WINDOWS 1
#elif defined(__APPLE__)
#define RGPOT_CXX_OS_NAME "macos"
#define RGPOT_OS_MACOS 1
#elif defined(__ANDROID__)
#define RGPOT_CXX_OS_NAME "android"
#define RGPOT_OS_ANDROID 1
#elif defined(__linux__) && defined(__GLIBC__)
#define RGPOT_CXX_OS_NAME "linux"
#define RGPOT_OS_LINUX 1
#elif defined(__linux__)
#define RGPOT_CXX_OS_NAME "linux-musl"
#define RGPOT_OS_LINUX_MUSL 1
#else
#define RGPOT_CXX_OS_NAME "unknown"
#define RGPOT_OS_UNKNOWN 1
#endif

#if defined(__x86_64__) || defined(_M_X64)
#define RGPOT_CXX_ARCH_NAME "x86_64"
#define RGPOT_ARCH_X86_64 1
#elif defined(__aarch64__) || defined(_M_ARM64)
#define RGPOT_CXX_ARCH_NAME "aarch64"
#define RGPOT_ARCH_AARCH64 1
#elif defined(__i386__) || defined(_M_IX86)
#define RGPOT_CXX_ARCH_NAME "i686"
#define RGPOT_ARCH_I686 1
#else
#define RGPOT_CXX_ARCH_NAME "unknown"
#define RGPOT_ARCH_UNKNOWN 1
#endif

#if defined(_LIBCPP_VERSION)
#define RGPOT_CXX_STDLIB_NAME "libc++"
#define RGPOT_STDLIB_LIBCPP 1
#elif defined(__GLIBCXX__)
#define RGPOT_CXX_STDLIB_NAME "libstdc++"
#define RGPOT_STDLIB_LIBSTDCXX 1
#elif defined(_MSC_VER)
#define RGPOT_CXX_STDLIB_NAME "msvc"
#define RGPOT_STDLIB_MSVC 1
#else
#define RGPOT_CXX_STDLIB_NAME "unknown"
#define RGPOT_STDLIB_UNKNOWN 1
#endif

#if defined(__LP64__) || defined(_LP64)
#define RGPOT_CXX_MODEL_NAME "lp64"
#define RGPOT_MODEL_LP64 1
#elif defined(_WIN64)
#define RGPOT_CXX_MODEL_NAME "llp64"
#define RGPOT_MODEL_LLP64 1
#else
#define RGPOT_CXX_MODEL_NAME "ilp32"
#define RGPOT_MODEL_ILP32 1
#endif

#define RGPOT_CXX_PLATFORM_TAG                                                 \
  RGPOT_CXX_OS_NAME "-" RGPOT_CXX_ARCH_NAME "-" RGPOT_CXX_STDLIB_NAME "-"      \
      RGPOT_CXX_MODEL_NAME

// Listed platforms. Each branch matches one kCxxSoversionTable row.
// The string is the row's tag. check_cxx_soversion.py requires the two
// sets to be equal. No branch means every platform is unlisted.
// aarch64, macOS and Windows stay unlisted until a no-cache CI stamp
// line for soversion 5 is copied in. Their soversion-4 hashes are the
// layouts from before LJPot and MorsePot grew their neighbour members.
#if defined(RGPOT_OS_LINUX) && defined(RGPOT_ARCH_X86_64) &&                    \
    defined(RGPOT_STDLIB_LIBSTDCXX) && defined(RGPOT_MODEL_LP64)
#define RGPOT_CXX_PLATFORM_LISTED 1
#define RGPOT_CXX_PLATFORM_LISTED_TAG "linux-x86_64-libstdc++-lp64"
#endif

/// Plain-C view of the stamp, returned by the exported function.
extern "C" {
typedef struct rgpot_cxx_layout_stamp_t {
  uint32_t revision; ///< kCxxLayoutRevision of the header that built it.
  uint32_t features; ///< Bit 0: RGPOT_HAS_CACHE.
  uint64_t hash;     ///< Fold of sizeof/alignof/offsetof of the covered types.
} rgpot_cxx_layout_stamp_t;

/// Stamp of the loaded library, computed from the headers it was built with.
RGPOT_CXX_LAYOUT_API rgpot_cxx_layout_stamp_t rgpot_cxx_layout_stamp(void);

/// Nonzero when @p caller equals the stamp of the loaded library.
RGPOT_CXX_LAYOUT_API int32_t
rgpot_cxx_layout_compatible(const rgpot_cxx_layout_stamp_t *caller);
}

namespace rgpot::abi {

/// Hand-maintained. Increment when a change the hash cannot see alters the
/// C++ ABI: reordering virtuals without changing their count, changing the
/// meaning of a field, changing a base class of a covered type. Any change to
/// size, alignment or a field offset moves the hash by itself.
inline constexpr std::uint32_t kCxxLayoutRevision = 1;

/// ELF soversion of librgpot (`librgpot.so.N`, `librgpot.N.dylib`).
/// Meson and CMake read this constant. The SONAME is not the layout hash.
/// The hash depends on the platform tag and includes the cache feature
/// bit, so builds of one revision do not share a hash. Bump this when a
/// recorded hash in `kCxxSoversionTable` changes, and replace that table
/// from the CI stamp lines in the same edit. `check_cxx_soversion.py`
/// fails when the fingerprint of the stamp inputs changes and this number
/// did not increase.
///
/// This is not `RGPOT_ABI_LAYOUT_REVISION`. The C ABI stamp in `rgpot.h`
/// is major 1, minor 4, layout revision 1. Minor 1 added
/// `rgpot_eindir_abi_compatible`. No exported C struct of that header
/// changed layout. The neighbour query is `RGPOT_NLIST_ABI` 4.
/// The C API is exported from librgpot, so it has no shared library of
/// its own and rides this SONAME. `librgpot_mpi` stays at soversion 3
/// (`RGPOT_MPI_ABI_VERSION` 1, unchanged).
///
/// 5 records LJPot and MorsePot gaining a Neighbours and a CachedPairs.
/// Soversion 4 hashed those classes at 120 and 136 bytes. They are 1016
/// and 1032 bytes with the members. The ledger of soversion 4 fingerprinted
/// the POD configs only, so the class members did not force a bump.
/// 6 drops the copied half-list from LJ and Morse, and gives ZBL and the
/// open Lennard-Jones cluster a neighbour member. The pair term runs on
/// the shared walk, so CachedPairs is no longer part of those objects.
inline constexpr std::uint32_t kCxxSoversion = 7;

/// One recorded no-cache layout hash. `tag` is `kCxxPlatformTag`.
struct CxxSoversionEntry {
  const char *tag;
  std::uint64_t hash;
};

/// No-cache `header_stamp().hash` for each platform ABI CI builds.
/// Rows are copied from `RGPOT_CXX_LAYOUT_STAMP` lines. An unlisted
/// platform still compiles. How to update the table:
/// docs/orgmode/contributing/developer/abi-bump.org.
inline constexpr std::array<CxxSoversionEntry, 1> kCxxSoversionTable{{
    {"linux-x86_64-libstdc++-lp64", 0xdf2b96303d5ac68full},
}};

/// This translation unit's platform tag (`os-arch-stdlib-datamodel`).
inline constexpr char kCxxPlatformTag[] = RGPOT_CXX_PLATFORM_TAG;

constexpr bool cxx_tag_eq(const char *a, const char *b) noexcept {
  while (*a != '\0' && *a == *b) {
    ++a;
    ++b;
  }
  return *a == *b;
}

/// Index of @p tag in @c kCxxSoversionTable, or -1 when the platform is absent.
/// A pointer into the table is not a constant expression under
/// @c -fsanitize=undefined, so callers compare this index with zero.
constexpr int find_soversion_index(const char *tag) noexcept {
  for (std::size_t i = 0; i < kCxxSoversionTable.size(); ++i) {
    if (cxx_tag_eq(kCxxSoversionTable[i].tag, tag)) {
      return static_cast<int>(i);
    }
  }
  return -1;
}

constexpr bool soversion_tags_unique() noexcept {
  for (std::size_t i = 0; i < kCxxSoversionTable.size(); ++i) {
    for (std::size_t j = i + 1; j < kCxxSoversionTable.size(); ++j) {
      if (cxx_tag_eq(kCxxSoversionTable[i].tag, kCxxSoversionTable[j].tag)) {
        return false;
      }
    }
  }
  return true;
}

static_assert(soversion_tags_unique(), "kCxxSoversionTable tags must be unique");

/// Virtual functions declared by @c PotentialBase, destructor included.
/// The cache hook @c setCache adds one when @c RGPOT_HAS_CACHE is defined.
/// A test counts the @c virtual declarations in Potential.hpp and fails when
/// this and the header disagree.
inline constexpr std::uint32_t kPotentialBaseVirtuals = 6;
inline constexpr std::uint32_t kPotentialBaseCacheVirtuals = 1;
/// Virtual functions @c Potential<Derived> declares itself (@c forceImpl and
/// @c forceBatchImpl).
inline constexpr std::uint32_t kPotentialVirtuals = 2;

#ifdef RGPOT_HAS_CACHE
inline constexpr std::uint32_t kFeatures = 1;
#else
inline constexpr std::uint32_t kFeatures = 0;
#endif

namespace detail {

/// A complete derived type: gives the CRTP template one instantiation whose
/// size and alignment the stamp covers. Complete because compilers that
/// instantiate the virtual members with the class need the base-to-derived
/// conversion to be valid.
struct LayoutProbe final : Potential<LayoutProbe> {
  LayoutProbe() : Potential(PotType::UNKNOWN) {}
  void forceImpl(const ForceInput &, ForceOut *) const override {}
};

struct Fold {
  std::uint64_t h = 0xcbf29ce484222325ull;
  constexpr void add(std::uint64_t v) {
    for (int i = 0; i < 8; ++i) {
      h ^= (v >> (8 * i)) & 0xffu;
      h *= 0x100000001b3ull;
    }
  }
  /// sizeof and alignof of a type.
  template <typename T> constexpr void type() {
    add(sizeof(T));
    add(alignof(T));
  }
};

constexpr std::uint64_t compute_hash() {
  Fold f;
  f.add(kCxxLayoutRevision);
  f.add(kFeatures);
  f.add(sizeof(void *));
  f.add(sizeof(std::size_t));

  // Plain structures with a fixed field order.
  f.type<ForceInput>();
  f.add(offsetof(ForceInput, nAtoms));
  f.add(offsetof(ForceInput, pos));
  f.add(offsetof(ForceInput, atmnrs));
  f.add(offsetof(ForceInput, box));
  f.type<ForceOut>();
  f.add(offsetof(ForceOut, F));
  f.add(offsetof(ForceOut, energy));
  f.add(offsetof(ForceOut, variance));
  f.add(offsetof(ForceOut, stress));
  f.add(offsetof(ForceOut, has_stress));
  f.type<ForceBatch>();
  f.add(offsetof(ForceBatch, nSystems));
  f.add(offsetof(ForceBatch, in));
  f.add(offsetof(ForceBatch, out));
  f.type<PotCaps>();
  f.add(offsetof(PotCaps, reentrancy));
  f.add(offsetof(PotCaps, perImageInstances));
  f.add(offsetof(PotCaps, batched));
  f.add(offsetof(PotCaps, periodic));
  f.add(offsetof(PotCaps, groupCollective));
  f.add(offsetof(PotCaps, stress));
  f.add(offsetof(PotCaps, analyticHvp));
  f.type<CalculatorGroup>();
  f.add(offsetof(CalculatorGroup, index));
  f.add(offsetof(CalculatorGroup, ranks));
  f.add(offsetof(CalculatorGroup, rank_in_group));
  f.add(offsetof(CalculatorGroup, world_size));
  f.type<LJConfig>();
  f.add(offsetof(LJConfig, u0));
  f.add(offsetof(LJConfig, cutoff));
  f.add(offsetof(LJConfig, psi));
  f.add(offsetof(LJConfig, switch_width));
  f.type<LJClusterConfig>();
  f.add(offsetof(LJClusterConfig, u0));
  f.add(offsetof(LJClusterConfig, cutoff));
  f.add(offsetof(LJClusterConfig, psi));
  f.add(offsetof(LJClusterConfig, switch_width));
  f.type<MorseConfig>();
  f.add(offsetof(MorseConfig, De));
  f.add(offsetof(MorseConfig, a));
  f.add(offsetof(MorseConfig, re));
  f.add(offsetof(MorseConfig, cutoff));
  f.add(offsetof(MorseConfig, switch_width));
  f.type<ZBLConfig>();
  f.add(offsetof(ZBLConfig, cut_inner));
  f.add(offsetof(ZBLConfig, cut_global));

  // Enumerations: underlying width and the values consumers switch on.
  f.type<Reentrancy>();
  f.add(static_cast<std::uint64_t>(Reentrancy::SharedInstance));
  f.add(static_cast<std::uint64_t>(Reentrancy::PerInstance));
  f.add(static_cast<std::uint64_t>(Reentrancy::ProcessSerial));
  f.type<PotType>();
  f.add(static_cast<std::uint64_t>(PotType::UNKNOWN));
  f.add(static_cast<std::uint64_t>(PotType::CuH2));
  f.add(static_cast<std::uint64_t>(PotType::LJ));
  f.add(static_cast<std::uint64_t>(PotType::Morse));
  f.add(static_cast<std::uint64_t>(PotType::LJCluster));
  f.add(static_cast<std::uint64_t>(PotType::ZBL));
  f.add(static_cast<std::uint64_t>(PotType::Eckart));
  f.add(static_cast<std::uint64_t>(PotType::MullerBrown));
  f.add(static_cast<std::uint64_t>(PotType::ModifiedMullerBrown));
  f.add(static_cast<std::uint64_t>(PotType::LepsHH2));
  f.add(static_cast<std::uint64_t>(PotType::LepsHCl2));
  f.add(static_cast<std::uint64_t>(PotType::LepsFH2));
  f.add(static_cast<std::uint64_t>(PotType::DoubleWell));
  f.add(static_cast<std::uint64_t>(PotType::CoupledProtonWell));
  f.add(static_cast<std::uint64_t>(PotType::CBE));

  // Polymorphic classes: size, alignment and the virtual-function counts.
  // offsetof is not defined for them.
  f.type<PotentialBase>();
  f.add(kPotentialBaseVirtuals +
        (kFeatures != 0 ? kPotentialBaseCacheVirtuals : 0u));
  f.type<Potential<LayoutProbe>>();
  f.add(kPotentialVirtuals);
  f.type<LJPot>();
  f.type<LJClusterPot>();
  f.type<MorsePot>();
  f.type<ZBLPot>();
#ifdef RGPOT_HAS_CACHE
  f.type<cache::EvalCounts>();
  f.add(offsetof(cache::EvalCounts, computed));
  f.add(offsetof(cache::EvalCounts, served));
  f.type<cache::PotentialCache>();
#endif
  return f.h;
}

} // namespace detail

/// What the consumer compiled against. Initialise a @c constexpr variable
/// with it; a call evaluated at run time is not tied to this translation
/// unit (see layout_compatible()).
inline constexpr rgpot_cxx_layout_stamp_t header_stamp() noexcept {
  return {kCxxLayoutRevision, kFeatures, detail::compute_hash()};
}

/// Recorded no-cache hash for this platform, or 0 when it has no row.
constexpr std::uint64_t recorded_layout_hash() noexcept {
  const int index = find_soversion_index(kCxxPlatformTag);
  if (index < 0) {
    return 0;
  }
  return kCxxSoversionTable[static_cast<std::size_t>(index)].hash;
}

/// True when this platform has no row, or the row matches `header_stamp`.
constexpr bool platform_soversion_ok() noexcept {
  const int index = find_soversion_index(kCxxPlatformTag);
  if (index < 0) {
    return true;
  }
  return header_stamp().hash ==
         kCxxSoversionTable[static_cast<std::size_t>(index)].hash;
}

/// What the loaded library was built with.
inline rgpot_cxx_layout_stamp_t library_stamp() noexcept {
  return rgpot_cxx_layout_stamp();
}

/// True when the loaded library was built from the same layouts as these
/// headers: equal revision, equal feature bits, equal hash.
inline bool layout_compatible() noexcept {
  // Evaluated at compile time: a runtime call to an inline function would
  // bind, across shared objects, to whichever definition the loader meets
  // first, and defeat the comparison.
  constexpr rgpot_cxx_layout_stamp_t mine = header_stamp();
  return rgpot_cxx_layout_compatible(&mine) != 0;
}

} // namespace rgpot::abi
