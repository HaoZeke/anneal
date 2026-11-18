cmake_minimum_required(VERSION 3.16)

if(POLICY CMP0110)
  cmake_policy(SET CMP0110 NEW)
endif()

project(
  rgpot
  # [[[cog import rgpot_version as v; cog.outl(f"  VERSION {v.VERSION}")]]]
  VERSION 3.7.0
  # [[[end]]]
  DESCRIPTION "Cpp core for potential energy surface functionality"
  HOMEPAGE_URL "https://github.com/OmniPotentRPC/rgpot"
  LANGUAGES C CXX)

# --- Top-level detection (CMake < 3.21 compat) ---
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(RGPOT_IS_TOP_LEVEL TRUE)
else()
  set(RGPOT_IS_TOP_LEVEL FALSE)
endif()

# --- Options ---
option(RGPOT_RPC_CLIENT_ONLY "Build ONLY the RPC client bridge (lightweight)"
       OFF)

if(RGPOT_RPC_CLIENT_ONLY)
  set(CMAKE_CXX_STANDARD 17) # C++17 is sufficient for the RPC bridge
  option(RGPOT_WITH_RPC "Build with Cap'n Proto RPC support" ON)

  # Force disable heavy dependencies
  set(RGPOT_WITH_XTENSOR OFF)
  set(RGPOT_WITH_EIGEN OFF)
  set(RGPOT_WITH_CACHE OFF)
  set(RGPOT_PURE_LIB ON)
  set(RGPOT_BUILD_EXAMPLES OFF)
else()
  set(CMAKE_CXX_STANDARD 20)
  option(RGPOT_WITH_XTENSOR "Build with xtensor support" OFF)
  option(RGPOT_WITH_EIGEN "Build with Eigen3 support" OFF)
  option(RGPOT_PURE_LIB "Build as a pure library (no fmt dependency)" ON)
  option(RGPOT_WITH_CACHE "Build with RocksDB caching support" ON)
  option(RGPOT_WITH_RPC "Build with Cap'n Proto RPC support" OFF)
  option(RGPOT_BUILD_EXAMPLES "Build examples" ${RGPOT_IS_TOP_LEVEL})
  option(RGPOT_WITH_HIGHWAY "Google Highway runtime ISA dispatch" ON)
  if(WIN32)
    set(_rgpot_ipi_default OFF)
  else()
    set(_rgpot_ipi_default ON)
  endif()
  option(RGPOT_WITH_IPI "i-PI socket server and driver client (POSIX sockets)"
         ${_rgpot_ipi_default})
  if(RGPOT_WITH_IPI AND WIN32)
    message(
      FATAL_ERROR
        "RGPOT_WITH_IPI requires POSIX sockets and cannot be enabled on Windows"
    )
  endif()
endif()

option(RGPOT_WITH_EAM_TABLE
       "EAM and Finnis-Sinclair potentials read from LAMMPS tables" OFF)
option(RGPOT_WITH_CBE "Linked external cbe-pes frontend" OFF)
option(RGPOT_BUILD_TESTS "Build tests" ${RGPOT_IS_TOP_LEVEL})
option(RGPOT_WITH_CP2K
       "CP2KPot, the potserv CP2K backend and libcp2kc_rgpot against libcp2k (pkg-config libcp2k)" OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# librgpot's ELF soversion is kCxxSoversion in CxxLayout.hpp, the same
# constant Meson reads. The C ABI macros in rgpot.h are a different stamp
# and did not change. They have no shared library of their own.
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/CppCore/rgpot/abi/CxxLayout.hpp"
     _rgpot_cxx_layout_hpp)
string(REGEX MATCH "kCxxSoversion = ([0-9]+)" _rgpot_cxx_so_match
             "${_rgpot_cxx_layout_hpp}")
if(NOT CMAKE_MATCH_1)
  message(FATAL_ERROR "kCxxSoversion missing from CxxLayout.hpp")
endif()
set(RGPOT_CXX_SOVERSION "${CMAKE_MATCH_1}")

# --- Dependencies ---
include(FetchContent)

# 1. RPC Dependency
if(RGPOT_WITH_RPC)
  find_package(CapnProto REQUIRED)
endif()

# 1. Server/Physics Dependencies
if(NOT RGPOT_RPC_CLIENT_ONLY)
  include(CheckCXXSourceCompiles)
  check_cxx_source_compiles(
    "#include <atomic>\n#include <cstdint>\nint main() { std::uint64_t w = 0; std::atomic_ref<std::uint64_t>(w).fetch_or(1u, std::memory_order_relaxed); return 0; }"
    RGPOT_HAS_STD_ATOMIC_REF)
  if(NOT RGPOT_HAS_STD_ATOMIC_REF)
    check_cxx_source_compiles(
      "#include <cstdint>\n#if defined(_MSC_VER)\n#include <intrin.h>\n#endif\nint main() {\n  std::uint64_t w = 0;\n#if defined(__GNUC__) || defined(__clang__)\n  __atomic_fetch_or(&w, 1u, __ATOMIC_RELAXED);\n#elif defined(_MSC_VER)\n  _InterlockedOr64(reinterpret_cast<volatile long long *>(&w), 1LL);\n#else\n#error \"no atomic_fetch_or fallback\"\n#endif\n  return 0;\n}"
      RGPOT_HAS_ATOMIC_FETCH_OR_FALLBACK)
    if(NOT RGPOT_HAS_ATOMIC_FETCH_OR_FALLBACK)
      message(
        FATAL_ERROR
          "std::atomic_ref requires libc++ 19 or newer (or a compiler with __atomic_fetch_or)"
      )
    endif()
    message(
      STATUS
        "std::atomic_ref unavailable, using compiler atomic fetch_or fallback")
  endif()
  # Core builds provide every public Fortran potential. The RPC-only
  # bridge has no numerical kernels and does not need a Fortran compiler.
  enable_language(Fortran)
  if(RGPOT_WITH_CACHE)
    # Conda macOS runners sometimes fail FindThreads before RocksDB; set the
    # pthread hint explicitly so RocksDBConfig's find_dependency(Threads) works.
    set(CMAKE_THREAD_LIBS_INIT "-lpthread")
    set(CMAKE_HAVE_THREADS_LIBRARY 1)
    set(CMAKE_USE_PTHREADS_INIT 1)
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)
    find_package(RocksDB REQUIRED)
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
    find_package(xxHash QUIET)
    list(POP_BACK CMAKE_MODULE_PATH)
    if(NOT xxHash_FOUND)
      message(
        STATUS "xxHash not found via find_package, fetching from source...")
      FetchContent_Declare(
        xxHash
        GIT_REPOSITORY https://github.com/Cyan4973/xxHash.git
        GIT_TAG v0.8.3)
      FetchContent_MakeAvailable(xxHash)
      if(NOT TARGET xxHash::xxhash)
        # The pinned release keeps its CMake project below the source root.
        set(XXHASH_BUILD_XXHSUM OFF)
        set(XXHASH_BUNDLED_MODE OFF)
        add_subdirectory("${xxhash_SOURCE_DIR}/cmake_unofficial"
                         "${xxhash_BINARY_DIR}")
      endif()
      set_target_properties(xxhash PROPERTIES POSITION_INDEPENDENT_CODE ON)
    endif()
  endif()

  if(NOT RGPOT_PURE_LIB)
    find_package(fmt REQUIRED)
  endif()

  if(RGPOT_WITH_XTENSOR)
    find_package(xtensor REQUIRED)
    find_package(xtensor-blas REQUIRED)
  endif()

  if(RGPOT_WITH_EIGEN)
    find_package(Eigen3 3.4.0 REQUIRED)
  endif()
endif()

# --- Build Targets ---

# The compiled schema is the vendored copy (the pinned potentials-schema
# file) with the rgpot-local series applied. capnp_generate_cpp needs the
# file at configure time, so the series is applied here and a change to it
# reconfigures. The neighbour tests read it without the RPC stack.
if(RGPOT_WITH_RPC OR RGPOT_BUILD_TESTS)
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  set(RGPOT_SCHEMA_SRC_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/schema")
  set(RGPOT_EFFECTIVE_SCHEMA
      "${RGPOT_SCHEMA_SRC_PREFIX}/CppCore/rgpot/rpc/Potentials.capnp")
  set(_rgpot_schema_dir "${CMAKE_CURRENT_SOURCE_DIR}/CppCore/rgpot/rpc")
  file(STRINGS "${_rgpot_schema_dir}/local/series" _rgpot_series_lines)
  set(_rgpot_schema_inputs "${_rgpot_schema_dir}/Potentials.capnp"
                           "${_rgpot_schema_dir}/local/series")
  foreach(_line IN LISTS _rgpot_series_lines)
    string(STRIP "${_line}" _line)
    if(_line AND NOT _line MATCHES "^#")
      list(APPEND _rgpot_schema_inputs "${_rgpot_schema_dir}/local/${_line}")
    endif()
  endforeach()
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
                                         ${_rgpot_schema_inputs})
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" "${_rgpot_schema_dir}/schema_ledger.py"
            apply --root "${CMAKE_CURRENT_SOURCE_DIR}"
            --out "${RGPOT_EFFECTIVE_SCHEMA}"
    RESULT_VARIABLE _rgpot_schema_result)
  if(NOT _rgpot_schema_result EQUAL 0)
    message(FATAL_ERROR "schema_ledger.py apply failed: ${_rgpot_schema_result}")
  endif()
endif()

if(RGPOT_WITH_RPC)
  set(CAPNP_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/CppCore")
  file(MAKE_DIRECTORY "${CAPNP_OUTPUT_DIR}/rgpot/rpc")

  set(CAPNPC_SRC_PREFIX "${RGPOT_SCHEMA_SRC_PREFIX}")
  set(CAPNPC_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}")
  capnp_generate_cpp(CAPNP_SOURCES CAPNP_HEADERS "${RGPOT_EFFECTIVE_SCHEMA}")

  # On Apple, static intermediate RPC libs avoid install_name / mixed
  # toolchain dylib issues when system clang links conda Cap'n Proto; the
  # full RPC/cache feature set is still exercised in executables/tests.
  if(APPLE)
    add_library(ptlrpc STATIC ${CAPNP_SOURCES})
  else()
    add_library(ptlrpc SHARED ${CAPNP_SOURCES})
  endif()
  target_link_libraries(ptlrpc PUBLIC CapnProto::capnp-rpc)
  # Always drive C++ links with the C++ linker.
  set_target_properties(ptlrpc PROPERTIES LINKER_LANGUAGE CXX)

  # Ensure the generated headers are in the include path
  target_include_directories(
    ptlrpc
    PUBLIC $<BUILD_INTERFACE:${CAPNP_OUTPUT_DIR}/rgpot/rpc>
           $<BUILD_INTERFACE:${CAPNP_OUTPUT_DIR}>
           $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
           $<INSTALL_INTERFACE:include/rgpot/rpc>
           $<INSTALL_INTERFACE:include>)

  # Client Bridge
  if(APPLE)
    add_library(rgpot_client_bridge STATIC CppCore/rgpot/rpc/pot_bridge.cc)
  else()
    add_library(rgpot_client_bridge SHARED CppCore/rgpot/rpc/pot_bridge.cc)
  endif()
  target_link_libraries(rgpot_client_bridge PUBLIC ptlrpc CapnProto::capnp-rpc)
  set_target_properties(rgpot_client_bridge PROPERTIES LINKER_LANGUAGE CXX)

  target_include_directories(
    rgpot_client_bridge
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/CppCore>
           $<INSTALL_INTERFACE:include>)
endif()

if(NOT RGPOT_RPC_CLIENT_ONLY)
  # Basic sources
  # The classical pots potserv can serve. Keep this in step with the arms in
  # rgpot/rpc/server.cpp: the server names them unconditionally, so a missing
  # source here surfaces as an undefined constructor at link time rather than
  # as a smaller build.
  set(RGPOT_SOURCES
      CppCore/rgpot/CalculatorGroup.cc
      CppCore/rgpot/PotHelpers.cc
      CppCore/rgpot/units.cc
      CppCore/rgpot/SumPot/SumPot.cc
      CppCore/rgpot/D2Pot/D2Pot.cc
      CppCore/rgpot/abi/CxxLayout.cc
      CppCore/rgpot/plugin/PluginRegistry.cc
      CppCore/rgpot/plugin/PluginBridge.cc
      CppCore/rgpot/plugin/builtins/lj_plugin.cpp
      CppCore/rgpot/LennardJones/LJPot.cc
      CppCore/rgpot/LennardJones/LJClusterPot.cc
      CppCore/rgpot/LennardJones/LJClusterKernel.cc
      CppCore/rgpot/Morse/MorsePot.cc
      CppCore/rgpot/Models/ModelPots.cc
      CppCore/rgpot/Models/ModelCapi.cc
      CppCore/rgpot/ZBL/ZBLPot.cc
      CppCore/rgpot/simd/dispatch.cc
      CppCore/rgpot/simd/scalar_kernels.cc
      CppCore/rgpot/CuH2/cuh2Utils.cc
      CppCore/rgpot/fortran/FortranPots.cc
      CppCore/rgpot/SuttonChen/SuttonChenPot.cc
      CppCore/rgpot/ReducedPair/ReducedPair.cc
      CppCore/rgpot/SuttonChen/SuttonChenScales.cc
      CppCore/rgpot/nlist/HalfShellWalk.cc
      CppCore/rgpot/nlist/HalfShell.cc
      CppCore/rgpot/StillingerWeber/SWSi.cc
      CppCore/rgpot/Tersoff/TersoffSiC.cc)
  # The rest of Meson's librgpot: pots potserv does not serve, and the
  # MOPAC, XTB and external-engine frontends, which load their engines at
  # run time.
  set(_rgpot_emt_sources
      CppCore/rgpot/EMT/Asap/NeighborList.cpp
      CppCore/rgpot/EMT/Asap/GhostAtoms.cpp
      CppCore/rgpot/EMT/Asap/CellList.cpp
      CppCore/rgpot/EMT/Asap/Vec.cpp
      CppCore/rgpot/EMT/Asap/Atoms.cpp
      CppCore/rgpot/EMT/Asap/Exception.cpp
      CppCore/rgpot/EMT/Asap/EMTRasmussenParameterProvider.cpp
      CppCore/rgpot/EMT/Asap/EMTDefaultParameterProvider.cpp
      CppCore/rgpot/EMT/Asap/EMT.cpp
      CppCore/rgpot/EMT/Asap/stacktrace.c
      CppCore/rgpot/EMT/Asap/SuperCell.cpp
      CppCore/rgpot/EMT/EMTPot.cc)
  list(APPEND RGPOT_SOURCES
       CppCore/rgpot/Morse/cambridge_morse.c
       CppCore/rgpot/Gupta/gupta.c
       CppCore/rgpot/EAM/EAM.cpp
       CppCore/rgpot/EAM/EAMPot.cc
       CppCore/rgpot/MOPACPot/MOPACPot.cc
       CppCore/rgpot/XTBPot/XTBDlopen.cc
       CppCore/rgpot/ExternalPot/ExternalPot.cc
       ${_rgpot_emt_sources})
  # Asap's short includes (Atoms.h, EMT.h) resolve in its own directory.
  set_source_files_properties(${_rgpot_emt_sources} PROPERTIES
    INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/CppCore/rgpot/EMT/Asap")
  set_source_files_properties(CppCore/rgpot/EAM/EAM.cpp CppCore/rgpot/EAM/EAMPot.cc
    PROPERTIES COMPILE_DEFINITIONS EAM_STANDALONE)

  if(RGPOT_WITH_HIGHWAY)
    list(APPEND RGPOT_SOURCES CppCore/rgpot/simd/hwy_kernels.cc)
  endif()

  # NWChemPot / CPMDPot are pure dlopen consumers of the split nwchemc / cpmdc
  # engines (parity with meson). Cap'n Proto params live in the generated schema,
  # so the frontends are only compiled when RGPOT_WITH_RPC enables ptlrpc.
  if(RGPOT_WITH_RPC)
    list(APPEND RGPOT_SOURCES CppCore/rgpot/NWChemPot/NWChemPot.cc
         CppCore/rgpot/CPMDPot/CPMDPot.cc)
  endif()
  # CP2KPot dlopens the libcp2k shim. It is built only with the shim, so a
  # build without libcp2k compiles no CP2K code.
  if(RGPOT_WITH_CP2K)
    if(NOT RGPOT_WITH_RPC)
      message(FATAL_ERROR "RGPOT_WITH_CP2K requires RGPOT_WITH_RPC")
    endif()
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(LIBCP2K REQUIRED IMPORTED_TARGET libcp2k)
    list(APPEND RGPOT_SOURCES CppCore/rgpot/CP2KPot/CP2KPot.cc
         CppCore/rgpot/CP2KPot/CP2KDeck.cc)
  endif()

  if(RGPOT_WITH_HIGHWAY)
    find_package(hwy 1.2 QUIET CONFIG)
    if(NOT TARGET hwy)
      set(HWY_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
      set(HWY_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
      set(HWY_ENABLE_CONTRIB OFF CACHE BOOL "" FORCE)
      set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
      FetchContent_Declare(
        highway
        URL https://github.com/google/highway/archive/refs/tags/1.2.0.tar.gz
        URL_HASH SHA256=7e0be78b8318e8bdbf6fa545d2ecb4c90f947df03f7aadc42c1967f019e63343)
      FetchContent_MakeAvailable(highway)
      if(TARGET hwy)
        set_target_properties(hwy PROPERTIES CXX_VISIBILITY_PRESET hidden)
        target_compile_definitions(
          hwy PRIVATE HWY_DISABLED_TARGETS=0xC0F800024041090)
      endif()
    endif()
  endif()

  if(RGPOT_WITH_IPI)
    list(APPEND RGPOT_SOURCES CppCore/rgpot/IPIPot/IPIWire.cc
         CppCore/rgpot/IPIPot/IPIPot.cc CppCore/rgpot/IPIPot/IPIDriver.cc)
  endif()

  if(RGPOT_WITH_EAM_TABLE)
    list(APPEND RGPOT_SOURCES CppCore/rgpot/EAMTable/EAMTablePot.cc
         CppCore/rgpot/EAMTable/EAMTableImages.cc)
  endif()

  if(RGPOT_BUILD_TESTS)
    enable_testing()
  endif()
  find_package(Threads REQUIRED)
  add_library(rgpot_nlist STATIC
      CppCore/rgpot/Slices.cc
      CppCore/rgpot/nlist/NeighbourList.cc
      CppCore/rgpot/nlist/Neighbours.cc
      CppCore/rgpot/nlist/MinimumImage.cpp
      CppCore/rgpot/simd/exact_fma_hw.cc
      CppCore/rgpot/simd/exp.cc)
  target_include_directories(rgpot_nlist PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/CppCore>)
  target_link_libraries(rgpot_nlist PUBLIC Threads::Threads)
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(rgpot_nlist PUBLIC OpenMP::OpenMP_CXX)
  endif()
  set_target_properties(rgpot_nlist PROPERTIES POSITION_INDEPENDENT_CODE ON)
  add_library(rgpot_vesin STATIC third_party/vesin/vesin-single-build.cpp)
  add_library(rgpot::vesin ALIAS rgpot_vesin)
  set_target_properties(rgpot_vesin PROPERTIES
    EXPORT_NAME vesin
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden)
  target_include_directories(rgpot_vesin
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party/vesin>
           $<INSTALL_INTERFACE:include/rgpot/vesin>)
  target_link_libraries(rgpot_vesin PRIVATE Threads::Threads)
  # The sources meson compiles without contraction: rgpot_core,
  # rgpot_minimage, rgpot_exact_fma, and the lennard_jones, morse, zbl,
  # gupta, sutton_chen, tersoff_sic, reduced_pair and eam_cell libraries.
  # Tests compare their bits with unfused references, and arm64 contracts
  # a*b+c by default, so a source missing here fails there and nowhere else.
  # tools/check_test_parity.sh checks this list against meson.
  set(_rgpot_unfused_sources
      CppCore/rgpot/CalculatorGroup.cc
      CppCore/rgpot/PotHelpers.cc
      CppCore/rgpot/PotentialCache.cc
      CppCore/rgpot/units.cc
      CppCore/rgpot/SumPot/SumPot.cc
      CppCore/rgpot/D2Pot/D2Pot.cc
      CppCore/rgpot/abi/CxxLayout.cc
      CppCore/rgpot/plugin/PluginRegistry.cc
      CppCore/rgpot/plugin/PluginBridge.cc
      CppCore/rgpot/plugin/builtins/lj_plugin.cpp
      CppCore/rgpot/nlist/MinimumImage.cpp
      CppCore/rgpot/simd/exact_fma_hw.cc
      CppCore/rgpot/simd/exp.cc
      CppCore/rgpot/simd/dispatch.cc
      CppCore/rgpot/simd/scalar_kernels.cc
      CppCore/rgpot/simd/hwy_kernels.cc
      CppCore/rgpot/LennardJones/LJPot.cc
      CppCore/rgpot/LennardJones/LJClusterPot.cc
      CppCore/rgpot/LennardJones/LJClusterKernel.cc
      CppCore/rgpot/Morse/MorsePot.cc
      CppCore/rgpot/Morse/cambridge_morse.c
      CppCore/rgpot/ZBL/ZBLPot.cc
      CppCore/rgpot/Gupta/gupta.c
      CppCore/rgpot/SuttonChen/SuttonChenPot.cc
      CppCore/rgpot/SuttonChen/SuttonChenScales.cc
      CppCore/rgpot/Tersoff/TersoffSiC.cc
      CppCore/rgpot/ReducedPair/ReducedPair.cc
      CppCore/rgpot/EAM/EAM.cpp
      CppCore/rgpot/EAM/EAMPot.cc)
  if(MSVC)
    set_source_files_properties(${_rgpot_unfused_sources}
      PROPERTIES COMPILE_OPTIONS "/fp:precise")
  else()
    set_source_files_properties(${_rgpot_unfused_sources}
      PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
    set_property(SOURCE CppCore/rgpot/ReducedPair/ReducedPair.cc
      APPEND PROPERTY COMPILE_OPTIONS -fno-fast-math)
    # Value-safe: lets the exp clamps if-convert and sqrt use sqrtpd, so
    # the exp and Morse row loops vectorise, also at -O2. Each flag is kept
    # only where the compiler accepts it (clang has no -fvect-cost-model),
    # as the Meson build does with get_supported_arguments.
    include(CheckCXXCompilerFlag)
    set(RGPOT_VECTOR_LOOP_FLAGS)
    foreach(flag -fno-trapping-math -fno-math-errno -ftree-vectorize
                 -fvect-cost-model=dynamic)
      string(MAKE_C_IDENTIFIER "RGPOT_HAS${flag}" flag_var)
      check_cxx_compiler_flag(${flag} ${flag_var})
      if(${flag_var})
        list(APPEND RGPOT_VECTOR_LOOP_FLAGS ${flag})
      endif()
    endforeach()
    set_property(SOURCE CppCore/rgpot/simd/exp.cc CppCore/rgpot/Morse/MorsePot.cc
      APPEND PROPERTY COMPILE_OPTIONS ${RGPOT_VECTOR_LOOP_FLAGS})
  endif()
  add_subdirectory(CppCore/rgpot/fortran)
  if(RGPOT_WITH_CBE)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(CBE REQUIRED cbe)
    find_path(CBE_HEADER_DIR NAMES cbe.h HINTS ${CBE_INCLUDE_DIRS}
              NO_DEFAULT_PATH REQUIRED)
    find_library(CBE_LIBRARY NAMES cbe HINTS ${CBE_LIBRARY_DIRS}
                 NO_DEFAULT_PATH REQUIRED)
    list(APPEND RGPOT_SOURCES CppCore/rgpot/CBEPot/CBEPot.cc)
  endif()
  add_library(rgpot ${RGPOT_SOURCES})
  target_compile_definitions(rgpot PRIVATE RGPOT_MODEL_BUILD)
  if(RGPOT_WITH_HIGHWAY)
    target_link_libraries(rgpot PRIVATE hwy)
    target_compile_definitions(rgpot PRIVATE RGPOT_WITH_HIGHWAY)
  endif()
  set_target_properties(rgpot PROPERTIES LINKER_LANGUAGE CXX
                                        POSITION_INDEPENDENT_CODE ON
                                        VERSION ${PROJECT_VERSION}
                                        SOVERSION ${RGPOT_CXX_SOVERSION})
  target_link_libraries(rgpot PRIVATE rgpot_fortran_kernels rgpot_nlist Threads::Threads)
  if(RGPOT_WITH_CBE)
    target_include_directories(rgpot PRIVATE ${CBE_HEADER_DIR})
    target_link_libraries(rgpot PRIVATE ${CBE_LIBRARY})
    target_compile_definitions(rgpot PUBLIC RGPOT_HAS_CBE)
  endif()
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_options(
      rgpot PRIVATE
      "LINKER:--exclude-libs,$<TARGET_FILE_NAME:rgpot_fortran_kernels>")
    get_target_property(_rgpot_library_type rgpot TYPE)
    if(_rgpot_library_type STREQUAL "STATIC_LIBRARY")
      # Static consumers perform the final link and must hide the kernel
      # archive's internal Fortran entries in their own shared objects.
      target_link_options(
        rgpot INTERFACE
        "$<BUILD_INTERFACE:LINKER:--exclude-libs,$<TARGET_FILE_NAME:rgpot_fortran_kernels>>"
        "$<INSTALL_INTERFACE:LINKER:--exclude-libs,$<TARGET_FILE_NAME:rgpot::rgpot_fortran_kernels>>")
    endif()
  endif()

  # librgpot never links MPI; it loads librgpot_mpi with dlopen when a host
  # asks for calculator groups (CppCore/rgpot/calculator_mpi_abi.h).
  target_link_libraries(rgpot PRIVATE ${CMAKE_DL_LIBS})
  find_package(MPI QUIET COMPONENTS CXX)
  set(RGPOT_HAVE_MPI OFF)
  if(MPI_FOUND)
    # A system MPI wrapper can be found without headers the active
    # compiler can include; only a configuration that compiles counts.
    include(CheckCXXSourceCompiles)
    set(CMAKE_REQUIRED_LIBRARIES MPI::MPI_CXX)
    check_cxx_source_compiles(
      "#include <mpi.h>
       int main(int argc, char **argv) { MPI_Init(&argc, &argv); return MPI_Finalize(); }"
      RGPOT_MPI_COMPILES)
    unset(CMAKE_REQUIRED_LIBRARIES)
    set(RGPOT_HAVE_MPI ${RGPOT_MPI_COMPILES})
  endif()
  if(RGPOT_HAVE_MPI)
    add_library(rgpot_mpi SHARED CppCore/rgpot/CalculatorGroupMpi.cc)
    target_include_directories(rgpot_mpi
                               PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/CppCore)
    target_link_libraries(rgpot_mpi PRIVATE MPI::MPI_CXX)
    # soversion 3 is librgpot_mpi's own SONAME. RGPOT_MPI_ABI_VERSION is
    # still 1, so this name does not move with librgpot's kCxxSoversion.
    set_target_properties(rgpot_mpi PROPERTIES CXX_VISIBILITY_PRESET hidden
                                               SOVERSION 3)
  endif()

  if(RGPOT_WITH_CACHE)
    target_sources(rgpot PRIVATE CppCore/rgpot/PotentialCache.cc)
    # A static RocksDB can bring jemalloc onto each executable's own link
    # line, where its malloc wins over libc's, and its free then recurses
    # through __tls_get_addr once a library with TLS (libmpi) is dlopen'd.
    # The shared library, which meson's pkg-config lookup links, keeps
    # jemalloc behind libc.
    if(TARGET RocksDB::rocksdb-shared)
      target_link_libraries(rgpot PUBLIC RocksDB::rocksdb-shared xxHash::xxhash)
    else()
      target_link_libraries(rgpot PUBLIC RocksDB::rocksdb xxHash::xxhash)
    endif()
    target_compile_definitions(rgpot PUBLIC RGPOT_HAS_CACHE)
  endif()

  if(RGPOT_WITH_IPI)
    target_compile_definitions(rgpot PUBLIC RGPOT_HAS_IPI)
  endif()

  if(RGPOT_WITH_EAM_TABLE)
    target_compile_definitions(rgpot PUBLIC RGPOT_HAS_EAM_TABLE)
  endif()

  if(RGPOT_WITH_RPC)
    target_link_libraries(rgpot PUBLIC ptlrpc)
    target_compile_definitions(rgpot PUBLIC RGPOT_HAS_NWCHEM_FRONTEND
                                            RGPOT_HAS_CPMD_FRONTEND)
    # DynLib / dlopen for libnwchemc.so / libcpmdc.so at runtime
    if(UNIX AND NOT APPLE)
      target_link_libraries(rgpot PUBLIC ${CMAKE_DL_LIBS})
    elseif(UNIX)
      target_link_libraries(rgpot PUBLIC dl)
    endif()
  endif()

  add_library(rgpot::rgpot ALIAS rgpot)

  # The libcp2k shim is the only target linking libcp2k and MPI; CP2KPot
  # dlopens it.
  if(RGPOT_WITH_CP2K)
    target_compile_definitions(rgpot PUBLIC RGPOT_HAS_CP2K)
    find_package(MPI REQUIRED COMPONENTS C)
    add_library(cp2kc_rgpot SHARED CppCore/rgpot/CP2KPot/cp2kc_shim.c)
    target_include_directories(cp2kc_rgpot
                               PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/CppCore/rgpot/CP2KPot)
    target_link_libraries(cp2kc_rgpot PRIVATE PkgConfig::LIBCP2K MPI::MPI_C)
    set_target_properties(cp2kc_rgpot PROPERTIES BUILD_RPATH "${LIBCP2K_LIBDIR}"
                                                 INSTALL_RPATH "${LIBCP2K_LIBDIR}")
    install(TARGETS cp2kc_rgpot LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
  endif()

  target_include_directories(
    rgpot PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/CppCore>
                 $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                 $<INSTALL_INTERFACE:include>)

  if(UNIX AND NOT APPLE)
    target_link_libraries(rgpot PUBLIC m)
  endif()

  if(NOT RGPOT_PURE_LIB)
    target_link_libraries(rgpot PUBLIC fmt)
    target_compile_definitions(rgpot PUBLIC NOT_PURE_LIB)
  endif()

  if(RGPOT_WITH_XTENSOR)
    target_link_libraries(rgpot PUBLIC xtensor xtensor-blas)
    target_compile_definitions(rgpot PUBLIC WITH_XTENSOR)
  endif()

  if(RGPOT_WITH_EIGEN)
    target_link_libraries(rgpot PUBLIC Eigen3::Eigen)
  endif()

  if(MSVC)
    target_compile_options(rgpot PRIVATE /W4)
  else()
    target_compile_options(rgpot PRIVATE -Wall -Wextra -Wpedantic)
  endif()
endif()

# The MPI hosts are explicit executables; the serial server and core library
# do not acquire an MPI load dependency.
if(RGPOT_WITH_RPC AND NOT RGPOT_RPC_CLIENT_ONLY)
  add_executable(potserv CppCore/rgpot/rpc/server.cpp)
  target_link_libraries(potserv PRIVATE rgpot::rgpot ptlrpc CapnProto::capnp-rpc)
  if(RGPOT_HAVE_MPI)
    add_executable(potserv_mpi CppCore/rgpot/rpc/server.cpp)
    target_compile_definitions(potserv_mpi PRIVATE RGPOT_POTSERV_MPI)
    target_link_libraries(potserv_mpi PRIVATE rgpot::rgpot ptlrpc
                                            CapnProto::capnp-rpc MPI::MPI_CXX)
    add_executable(nwchem_mpi_force_host CppCore/tools/nwchem_mpi_force_host.cc)
    target_link_libraries(nwchem_mpi_force_host PRIVATE rgpot::rgpot MPI::MPI_CXX)
  endif()
endif()

# --- Installation ---
include(GNUInstallDirs)

if(RGPOT_IS_TOP_LEVEL)
  if(TARGET potserv)
    install(TARGETS potserv RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
  endif()
  if(TARGET potserv_mpi)
    install(TARGETS potserv_mpi nwchem_mpi_force_host
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
  endif()
  if(RGPOT_WITH_RPC)
    install(
      TARGETS ptlrpc rgpot_client_bridge
      EXPORT rgpotTargets
      LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
      ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
      RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
    # Install the bridge header in the correct directory structure
    install(FILES CppCore/rgpot/rpc/pot_bridge.h
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rgpot/rpc)
    install(FILES ${CAPNP_HEADERS} "${RGPOT_EFFECTIVE_SCHEMA}"
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rgpot/rpc)
  endif()

  if(NOT RGPOT_RPC_CLIENT_ONLY)
    install(
      TARGETS rgpot rgpot_fortran_kernels rgpot_nlist rgpot_vesin
      EXPORT rgpotTargets
      LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
      ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
      RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
      INCLUDES
      DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
    install(FILES third_party/vesin/vesin.h
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rgpot/vesin)
    install(TARGETS rgpot_cuh2
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
    if(TARGET rgpot_mpi)
      install(TARGETS rgpot_mpi
              LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
              RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
    endif()
    install(FILES CppCore/rgpot/calculator_mpi_abi.h include/rgpot/plugin.h
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rgpot)
    install(
      DIRECTORY CppCore/rgpot
      DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
      FILES_MATCHING
      PATTERN "Morse/detail" EXCLUDE
      PATTERN "SuttonChen/detail" EXCLUDE
      PATTERN "*.hpp"
      PATTERN "*.h")
  endif()

  install(
    EXPORT rgpotTargets
    FILE rgpotTargets.cmake
    NAMESPACE rgpot::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rgpot)

  include(CMakePackageConfigHelpers)
  configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/rgpotConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/rgpotConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rgpot)
  write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/rgpotConfigVersion.cmake"
    COMPATIBILITY SameMajorVersion)
  install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindxxHash.cmake"
                "${CMAKE_CURRENT_BINARY_DIR}/rgpotConfig.cmake"
                "${CMAKE_CURRENT_BINARY_DIR}/rgpotConfigVersion.cmake"
          DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rgpot)
endif()

# --- Tests ---
# CppCore/tests/registry.txt lists them; tests.cmake registers them.
if(RGPOT_BUILD_TESTS)
  include("${CMAKE_CURRENT_SOURCE_DIR}/CppCore/tests/tests.cmake")
endif()
