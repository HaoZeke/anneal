# Every CMake test. CppCore/tests/registry.txt names them, gates them and
# holds their options, and meson reads the same file. CMakeLists.txt
# includes this file in its own directory, so the test executables sit
# beside librgpot_mpi, where CalculatorGroup.cc looks for it.
# A test name may hold spaces ("CuH2 Utils Test"). The functions below keep
# the policies set here, and the loops that read RGPOT_REGISTRY_ENTRIES in
# the caller's scope need CMP0110 so foreach(IN LISTS) does not split an
# entry on whitespace.
include("${CMAKE_CURRENT_LIST_DIR}/../../cmake/RgpotTestRegistry.cmake")
if(POLICY CMP0110)
  cmake_policy(SET CMP0110 NEW)
endif()

enable_testing()
find_package(Threads REQUIRED)
find_package(Catch2 3 REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
if(NOT TARGET CapnProto::capnp)
  find_package(CapnProto QUIET)
endif()
if(NOT OpenMP_CXX_FOUND)
  find_package(OpenMP)
endif()
find_program(RGPOT_CAPNP_TOOL capnp)
find_program(RGPOT_RUSTC rustc)
include(CheckCXXCompilerFlag)

if(MSVC)
  set(_rgpot_fp_contract_off /fp:precise)
else()
  set(_rgpot_fp_contract_off -ffp-contract=off)
endif()

# meson asks only whether the compiler accepts -fsanitize=thread, without
# linking: gcc-toolset-14 compiles with it but has no libtsan.
set(_rgpot_try_type "${CMAKE_TRY_COMPILE_TARGET_TYPE}")
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
check_cxx_compiler_flag(-fsanitize=thread RGPOT_CXX_ACCEPTS_TSAN)
set(CMAKE_TRY_COMPILE_TARGET_TYPE "${_rgpot_try_type}")
set(RGPOT_TEST_TSAN FALSE)
if(RGPOT_CXX_ACCEPTS_TSAN AND NOT CMAKE_C_FLAGS MATCHES "-fsanitize="
   AND NOT CMAKE_CXX_FLAGS MATCHES "-fsanitize=")
  set(RGPOT_TEST_TSAN TRUE)
endif()

# MinimumImage compiled once per instruction set, as CppCore/meson.build
# does: on x86_64 each set whose flags the compiler takes, elsewhere NEON.
set(RGPOT_ISA_SPECS)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
  set(_rgpot_isa_scalar -ffp-contract=off -fno-tree-vectorize -mno-avx -mno-fma)
  set(_rgpot_isa_sse2 -ffp-contract=off -msse2 -mno-avx -mno-fma)
  set(_rgpot_isa_avx2 -ffp-contract=off -mavx2 -mfma)
  set(_rgpot_isa_avx512 -ffp-contract=off -mavx512f)
  foreach(spec scalar sse2 avx2 avx512)
    set(_supported TRUE)
    foreach(flag IN LISTS _rgpot_isa_${spec})
      string(MAKE_C_IDENTIFIER "RGPOT_CXX_HAS${flag}" flag_var)
      check_cxx_compiler_flag(${flag} ${flag_var})
      if(NOT ${flag_var})
        set(_supported FALSE)
      endif()
    endforeach()
    if(_supported)
      list(APPEND RGPOT_ISA_SPECS ${spec})
    endif()
  endforeach()
else()
  check_cxx_compiler_flag(-ffp-contract=off RGPOT_CXX_HAS_ffp_contract_off)
  set(_rgpot_isa_neon)
  if(RGPOT_CXX_HAS_ffp_contract_off)
    set(_rgpot_isa_neon -ffp-contract=off)
  endif()
  list(APPEND RGPOT_ISA_SPECS neon)
endif()

# Core builds already ran FindMPI. meson's mpiexec is a launcher on the PATH
# in any build.
if(NOT MPIEXEC_EXECUTABLE)
  find_program(MPIEXEC_EXECUTABLE NAMES mpirun mpiexec)
endif()

macro(rgpot_test_set_feature feature)
  if(${ARGN})
    set(RGPOT_TEST_FEATURE_${feature} TRUE)
  else()
    set(RGPOT_TEST_FEATURE_${feature} FALSE)
  endif()
endmacro()
rgpot_test_set_feature(tests RGPOT_BUILD_TESTS)
rgpot_test_set_feature(examples RGPOT_BUILD_EXAMPLES)
rgpot_test_set_feature(core NOT RGPOT_RPC_CLIENT_ONLY)
rgpot_test_set_feature(client RGPOT_RPC_CLIENT_ONLY)
rgpot_test_set_feature(rpc RGPOT_WITH_RPC)
rgpot_test_set_feature(fortran NOT RGPOT_RPC_CLIENT_ONLY)
rgpot_test_set_feature(cache RGPOT_WITH_CACHE)
rgpot_test_set_feature(ipi RGPOT_WITH_IPI)
rgpot_test_set_feature(eam_table RGPOT_WITH_EAM_TABLE)
rgpot_test_set_feature(eigen RGPOT_WITH_EIGEN)
rgpot_test_set_feature(xtensor RGPOT_WITH_XTENSOR)
rgpot_test_set_feature(cp2k RGPOT_WITH_CP2K)
rgpot_test_set_feature(mpi RGPOT_HAVE_MPI)
rgpot_test_set_feature(mpiexec MPIEXEC_EXECUTABLE)
rgpot_test_set_feature(linux CMAKE_SYSTEM_NAME STREQUAL "Linux")
rgpot_test_set_feature(posix NOT WIN32)
rgpot_test_set_feature(python Python3_Interpreter_FOUND)
rgpot_test_set_feature(capnp TARGET CapnProto::capnp AND TARGET CapnProto::capnpc)
rgpot_test_set_feature(capnp_tool RGPOT_CAPNP_TOOL)
rgpot_test_set_feature(rustc RGPOT_RUSTC)
rgpot_test_set_feature(tsan RGPOT_TEST_TSAN)
rgpot_test_set_feature(isa_variants RGPOT_ISA_SPECS AND NOT RGPOT_RPC_CLIENT_ONLY)
rgpot_test_set_feature(toplevel RGPOT_IS_TOP_LEVEL)
foreach(feature water xtb tblite dftd3 dftd4 xckernel long_range expr
                metatomic skala rust_core probe)
  set(RGPOT_TEST_FEATURE_${feature} FALSE)
endforeach()

# What every test executable compiles with, as meson's test_args.
set(RGPOT_BUILD_REVISION "$ENV{RGPOT_SOURCE_REVISION}")
if(RGPOT_BUILD_REVISION STREQUAL "")
  find_package(Git QUIET)
  if(GIT_FOUND)
    execute_process(
      COMMAND "${GIT_EXECUTABLE}" -C "${RGPOT_TEST_ROOT}" rev-parse
              --short=12 HEAD
      OUTPUT_VARIABLE RGPOT_BUILD_REVISION
      OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
  endif()
endif()
set(_rgpot_build_identity RGPOT_BUILD_VERSION="${PROJECT_VERSION}"
                          RGPOT_BUILD_REVISION="${RGPOT_BUILD_REVISION}")
file(READ "${RGPOT_TEST_ROOT}/CppCore/rgpot/rpc/Potentials.capnp"
     _rgpot_schema_text)
string(FIND "${_rgpot_schema_text}" "buildVersion" _rgpot_schema_identity)
if(NOT _rgpot_schema_identity EQUAL -1)
  list(APPEND _rgpot_build_identity RGPOT_SCHEMA_HAS_BUILD_IDENTITY)
endif()
set(_rgpot_test_definitions RGPOTTEST ${_rgpot_build_identity})
if(RGPOT_HAVE_MPI)
  list(APPEND _rgpot_test_definitions RGPOT_HAS_MPI)
endif()
if(RGPOT_RPC_CLIENT_ONLY)
  add_library(units_for_tests STATIC CppCore/rgpot/units.cc)
  target_include_directories(units_for_tests PUBLIC CppCore)
  set(_rgpot_test_libs units_for_tests)
  if(RGPOT_WITH_RPC)
    list(APPEND _rgpot_test_libs rgpot_client_bridge ptlrpc CapnProto::capnp-rpc)
  endif()
else()
  list(APPEND _rgpot_test_definitions RGPOT_HAS_FORTRAN_POTS)
  set(_rgpot_test_libs rgpot::rgpot rgpot_vesin Threads::Threads ${CMAKE_DL_LIBS})
  # The library links OpenMP privately. Tests branch on _OPENMP to know how
  # many threads it can use.
  if(OpenMP_CXX_FOUND)
    list(APPEND _rgpot_test_libs OpenMP::OpenMP_CXX)
  endif()
  if(RGPOT_WITH_RPC)
    list(APPEND _rgpot_test_libs ptlrpc CapnProto::capnp-rpc)
  endif()
endif()

function(rgpot_test_executable target)
  add_executable(${target} ${ARGN})
  target_include_directories(${target} PRIVATE "${RGPOT_TEST_ROOT}/CppCore"
                                               "${RGPOT_TEST_ROOT}/include")
  target_compile_definitions(${target} PRIVATE ${_rgpot_test_definitions})
  target_link_libraries(${target} PRIVATE ${_rgpot_test_libs}
                                          Catch2::Catch2WithMain)
  set_target_properties(${target} PROPERTIES LINKER_LANGUAGE CXX)
endfunction()

# The neighbour archives meson builds, which the C ABI's callers link.
add_library(rgpot_exact_fma STATIC CppCore/rgpot/simd/exact_fma_hw.cc
                                   CppCore/rgpot/simd/exp.cc)
target_compile_options(rgpot_exact_fma PRIVATE ${_rgpot_fp_contract_off}
                                               ${RGPOT_VECTOR_LOOP_FLAGS})
add_library(rgpot_minimage STATIC CppCore/rgpot/nlist/MinimumImage.cpp)
target_compile_options(rgpot_minimage PRIVATE ${_rgpot_fp_contract_off})
# exp.cc's pair loops call MinimumImage::fold.
target_link_libraries(rgpot_exact_fma INTERFACE rgpot_minimage)
add_library(rgpot_halfshell STATIC
  CppCore/rgpot/Slices.cc
  CppCore/rgpot/nlist/HalfShell.cc
  CppCore/rgpot/nlist/HalfShellWalk.cc
  CppCore/rgpot/nlist/NeighbourList.cc
  CppCore/rgpot/nlist/Neighbours.cc)
target_link_libraries(rgpot_halfshell PUBLIC Threads::Threads)
if(OpenMP_CXX_FOUND)
  target_link_libraries(rgpot_halfshell PUBLIC OpenMP::OpenMP_CXX)
endif()
foreach(archive rgpot_exact_fma rgpot_minimage rgpot_halfshell)
  target_include_directories(${archive} PUBLIC CppCore)
  target_compile_features(${archive} PUBLIC cxx_std_20)
  set_target_properties(${archive} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endforeach()
set(_rgpot_nlist_link rgpot_halfshell rgpot_minimage rgpot_exact_fma)

# Header-only and role-independent.
add_executable(AotiPublicationTest CppCore/tests/AotiPublicationTest.cc)
add_executable(AotiPublicationStressTest
               CppCore/tests/AotiPublicationStressTest.cc)
add_executable(aoti_unpack_test CppCore/tests/AotiUnpackTest.cc)
target_link_libraries(aoti_unpack_test PRIVATE Catch2::Catch2WithMain)
# The RPC client bridge stays C++17. This test includes std::span.
add_executable(MolecularFrameTest CppCore/tests/MolecularFrameTest.cc)
foreach(target AotiPublicationTest AotiPublicationStressTest aoti_unpack_test
               MolecularFrameTest)
  target_include_directories(${target} PRIVATE CppCore)
  target_compile_features(${target} PRIVATE cxx_std_20)
  target_link_libraries(${target} PRIVATE Threads::Threads)
endforeach()
rgpot_add_test(AotiPublicationTest COMMAND AotiPublicationTest)
rgpot_add_test(AotiPublicationStressTest COMMAND AotiPublicationStressTest)
rgpot_add_test(MolecularFrameTest COMMAND MolecularFrameTest)
rgpot_add_test(AotiUnpackTest COMMAND aoti_unpack_test)

rgpot_registry_feature_args(_rgpot_features)
rgpot_add_test(test_list_parity
  COMMAND Python3::Interpreter
          "${RGPOT_TEST_ROOT}/CppCore/tests/check_test_registry.py" cmake
          "${PROJECT_BINARY_DIR}" --ctest "${CMAKE_CTEST_COMMAND}"
          ${_rgpot_features})

rgpot_registry_tests(unit _rgpot_units)
foreach(test IN LISTS _rgpot_units)
  rgpot_registry_field("${test}" args _args)
  list(GET _args 0 _exe)
  list(GET _args 1 _source)
  rgpot_test_executable(${_exe} CppCore/tests/${_source})
  rgpot_registry_field("${test}" unfused _unfused)
  if(_unfused)
    target_compile_options(${_exe} PRIVATE ${_rgpot_fp_contract_off})
  endif()
  if(test STREQUAL "CalculatorGroupTest" AND TARGET rgpot)
    get_target_property(_rgpot_lib_type rgpot TYPE)
    if(_rgpot_lib_type STREQUAL "SHARED_LIBRARY")
      target_compile_definitions(
        ${_exe} PRIVATE RGPOT_TEST_SHARED_LIBRARY
                        "RGPOT_TEST_SHARED_LIBRARY_PATH=\"$<TARGET_FILE:rgpot>\"")
    else()
      # No shared object to open: export the linked archive's symbols so
      # the test can find them with dlsym.
      set_target_properties(${_exe} PROPERTIES ENABLE_EXPORTS ON)
    endif()
  endif()
  rgpot_add_test("${test}" COMMAND ${_exe})
endforeach()
# FastNlistTest calls rgpot_water_h_force and rgpot_water_h_image directly, and
# a shared librgpot hides the Fortran kernel archive behind --exclude-libs.
if(TARGET fast_nlist_test AND TARGET rgpot_fortran_kernels)
  target_link_libraries(fast_nlist_test PRIVATE rgpot_fortran_kernels)
endif()

rgpot_registry_tests(rerun _rgpot_reruns)
foreach(test IN LISTS _rgpot_reruns)
  rgpot_registry_field("${test}" args _base)
  rgpot_registry_field("${_base}" args _args)
  list(GET _args 0 _exe)
  rgpot_add_test("${test}" COMMAND ${_exe})
endforeach()

rgpot_registry_tests(fortran _rgpot_fortran)
foreach(test IN LISTS _rgpot_fortran)
  rgpot_registry_field("${test}" args _args)
  list(GET _args 0 _exe)
  list(GET _args 1 _source)
  add_executable(${_exe} CppCore/rgpot/fortran/${_source})
  target_link_libraries(${_exe} PRIVATE rgpot_fortran_kernels)
  rgpot_add_test("${test}" COMMAND ${_exe})
endforeach()

rgpot_registry_tests(nlist _rgpot_nlist)
foreach(test IN LISTS _rgpot_nlist)
  rgpot_registry_field("${test}" args _args)
  list(GET _args 0 _exe)
  list(GET _args 1 _source)
  add_executable(${_exe} CppCore/examples/${_source})
  target_link_libraries(${_exe} PRIVATE ${_rgpot_nlist_link})
  set_target_properties(${_exe} PROPERTIES LINKER_LANGUAGE CXX)
  rgpot_add_test("${test}" COMMAND ${_exe})
endforeach()

rgpot_registry_tests(py _rgpot_py)
foreach(test IN LISTS _rgpot_py)
  rgpot_registry_field("${test}" args _args)
  list(POP_FRONT _args _script)
  list(TRANSFORM _args REPLACE "^@root@$" "${RGPOT_TEST_ROOT}")
  list(TRANSFORM _args REPLACE "^@capnp@$" "${CAPNP_EXECUTABLE}")
  rgpot_add_test("${test}"
    COMMAND Python3::Interpreter "${RGPOT_TEST_ROOT}/${_script}" ${_args})
endforeach()

# The neighbour C ABI over Cap'n Proto and from Rust. The neighbour
# structs exist only in the effective schema, which CMakeLists.txt writes.
rgpot_test_enabled(nlist_stream _on)
if(_on)
  add_executable(nlist_stream CppCore/examples/nlist_stream.cc)
  target_link_libraries(nlist_stream PRIVATE ${_rgpot_nlist_link}
                                             CapnProto::capnp CapnProto::capnpc)
  rgpot_add_test(nlist_stream COMMAND nlist_stream "${RGPOT_EFFECTIVE_SCHEMA}")
endif()
rgpot_test_enabled(nlist_capnp_roundtrip _on)
if(_on)
  rgpot_add_test(nlist_capnp_roundtrip
    COMMAND Python3::Interpreter
            "${RGPOT_TEST_ROOT}/CppCore/tests/test_nlist_capnp.py"
            "${RGPOT_TEST_ROOT}" "${RGPOT_CAPNP_TOOL}" "${RGPOT_EFFECTIVE_SCHEMA}")
endif()
rgpot_test_enabled(nlist_rust _on)
if(_on)
  # The script links the archives into a shared library outside CMake, so
  # rgpot_halfshell's OpenMP link interface does not reach it. AppleClang
  # has no -fopenmp: pass the libraries FindOpenMP chose, and the runtime's
  # directory as an rpath. SIP drops DYLD_* before sh runs, and conda's
  # libomp.dylib is an @rpath install name.
  set(_rgpot_nlist_rust_omp)
  if(OpenMP_CXX_FOUND)
    foreach(_rgpot_omp_name IN LISTS OpenMP_CXX_LIB_NAMES)
      set(_rgpot_omp_lib "${OpenMP_${_rgpot_omp_name}_LIBRARY}")
      list(APPEND _rgpot_nlist_rust_omp "${_rgpot_omp_lib}")
      if(_rgpot_omp_name MATCHES "omp" AND IS_ABSOLUTE "${_rgpot_omp_lib}")
        get_filename_component(_rgpot_omp_dir "${_rgpot_omp_lib}" DIRECTORY)
        list(APPEND _rgpot_nlist_rust_omp "-Wl,-rpath,${_rgpot_omp_dir}")
      endif()
    endforeach()
  endif()
  rgpot_add_test(nlist_rust
    COMMAND sh "${RGPOT_TEST_ROOT}/CppCore/examples/nlist_rust_test.sh"
            "${RGPOT_TEST_ROOT}" "${CMAKE_CURRENT_BINARY_DIR}"
            ${_rgpot_nlist_rust_omp}
    ENVIRONMENT "CXX=${CMAKE_CXX_COMPILER}")
endif()

if(RGPOT_RPC_CLIENT_ONLY)
  return()
endif()

# Calculator groups across ranks, run by the unit test's executable.
rgpot_test_enabled(CalculatorGroupShare _on)
if(_on)
  rgpot_registry_field(CalculatorGroupTest args _args)
  list(GET _args 0 _exe)
  foreach(case share:CalculatorGroupShare bad-owner:CalculatorGroupBadOwner)
    string(REPLACE ":" ";" case "${case}")
    list(GET case 0 _mode)
    list(GET case 1 _test)
    rgpot_add_test(${_test}
      COMMAND "${RGPOT_TEST_ROOT}/CppCore/tests/calculator_group_mpi.sh"
              $<TARGET_FILE:${_exe}> ${_mode})
  endforeach()
endif()

# The library's exported surface.
get_target_property(_rgpot_kind rgpot TYPE)
rgpot_test_enabled(FortranSymbolAudit _on)
if(_on)
  rgpot_add_test(FortranSymbolAudit
    COMMAND "${RGPOT_TEST_ROOT}/scripts/audit_fortran_symbols.sh"
            $<TARGET_FILE:rgpot>)
endif()
rgpot_test_enabled(CoreLinksNoMpi _on)
if(_on)
  rgpot_add_test(CoreLinksNoMpi
    COMMAND "${RGPOT_TEST_ROOT}/scripts/check_no_mpi_needed.sh"
            $<TARGET_FILE:rgpot>)
endif()
rgpot_test_enabled(ExprLeptonSymbols _on)
if(_on)
  rgpot_add_test(ExprLeptonSymbols
    COMMAND "${RGPOT_TEST_ROOT}/scripts/check_lepton_symbols.sh"
            $<TARGET_FILE:rgpot> absent)
endif()

add_library(rgpot_layout_fixture SHARED CppCore/tests/layout_fixture.cc)
target_include_directories(rgpot_layout_fixture PRIVATE CppCore)
target_compile_definitions(rgpot_layout_fixture PRIVATE ${_rgpot_test_definitions})
target_link_libraries(rgpot_layout_fixture PRIVATE rgpot::rgpot)
set_target_properties(rgpot_layout_fixture PROPERTIES PREFIX "lib")
rgpot_test_executable(cxx_layout_test CppCore/tests/CxxLayoutTest.cc)
rgpot_add_test(CxxLayoutTest COMMAND cxx_layout_test
  ENVIRONMENT "RGPOT_LAYOUT_FIXTURE=$<TARGET_FILE:rgpot_layout_fixture>")

# The stamp program does not link librgpot, but takes its feature macros.
add_executable(print_cxx_layout_stamp CppCore/tests/print_cxx_layout_stamp.cc)
target_include_directories(print_cxx_layout_stamp PRIVATE
  $<TARGET_PROPERTY:rgpot,INTERFACE_INCLUDE_DIRECTORIES>)
target_compile_definitions(print_cxx_layout_stamp PRIVATE
  $<TARGET_PROPERTY:rgpot,INTERFACE_COMPILE_DEFINITIONS>)
rgpot_add_test(CxxSoversion
  COMMAND Python3::Interpreter
          "${RGPOT_TEST_ROOT}/CppCore/tests/check_cxx_soversion.py"
          ${RGPOT_CXX_SOVERSION} $<TARGET_FILE:rgpot>
          $<TARGET_FILE:print_cxx_layout_stamp>)

rgpot_test_enabled(abi_symbols _on)
if(_on)
  # The script has no execute bit. Meson runs it from its #! line.
  set(_rgpot_abi_script sh "${RGPOT_TEST_ROOT}/scripts/abi_symbols.sh")
  # The neighbour C ABI is in rgpot_nlist, which a shared librgpot links
  # and a static one leaves beside it. CMake builds no fast list, so the
  # fast-list baseline does not apply.
  if(_rgpot_kind STREQUAL "STATIC_LIBRARY")
    set(_rgpot_abi_lib $<TARGET_FILE:rgpot_nlist>)
  else()
    set(_rgpot_abi_lib $<TARGET_FILE:rgpot>)
  endif()
  rgpot_add_test(abi_symbols
    COMMAND ${_rgpot_abi_script} ${_rgpot_abi_lib}
            "${RGPOT_TEST_ROOT}/scripts/abi_baseline.txt")
  add_library(rgpot_abi_fixture SHARED engines/abi_fixture.c)
  add_library(rgpot_abi_fixture_a STATIC engines/abi_fixture.c)
  add_library(rgpot_abi_fixture_lib STATIC engines/abi_fixture.c)
  set_target_properties(rgpot_abi_fixture_lib PROPERTIES PREFIX "" SUFFIX ".lib")
  set(_rgpot_abi_fixture_baseline
      "${RGPOT_TEST_ROOT}/engines/abi_fixture_baseline.txt")
  rgpot_add_test(abi_symbols_fixture_shared
    COMMAND ${_rgpot_abi_script} $<TARGET_FILE:rgpot_abi_fixture>
            "${_rgpot_abi_fixture_baseline}")
  rgpot_add_test(abi_symbols_fixture_archive_a
    COMMAND ${_rgpot_abi_script} $<TARGET_FILE:rgpot_abi_fixture_a>
            "${_rgpot_abi_fixture_baseline}")
  rgpot_add_test(abi_symbols_fixture_archive_lib
    COMMAND ${_rgpot_abi_script} $<TARGET_FILE:rgpot_abi_fixture_lib>
            "${_rgpot_abi_fixture_baseline}")
  rgpot_add_test(abi_symbols_fixture_shared_hidden
    COMMAND ${_rgpot_abi_script} $<TARGET_FILE:rgpot_abi_fixture>
            "${RGPOT_TEST_ROOT}/engines/abi_fixture_hidden.txt")
  rgpot_add_test(abi_symbols_fixture_archive_missing
    COMMAND ${_rgpot_abi_script} $<TARGET_FILE:rgpot_abi_fixture_a>
            "${RGPOT_TEST_ROOT}/scripts/abi_baseline.txt")
endif()

rgpot_test_enabled(capi_threads _on)
if(_on)
  add_executable(capi_threads engines/capi_threads.c)
  set(_rgpot_capi_targets capi_threads)
  rgpot_test_enabled(capi_threads_tsan _rgpot_tsan)
  if(_rgpot_tsan)
    add_executable(capi_threads_tsan engines/capi_threads.c)
    target_compile_options(capi_threads_tsan PRIVATE -fsanitize=thread)
    target_link_options(capi_threads_tsan PRIVATE -fsanitize=thread)
    list(APPEND _rgpot_capi_targets capi_threads_tsan)
  endif()
  foreach(target IN LISTS _rgpot_capi_targets)
    target_include_directories(${target} PRIVATE CppCore include)
    target_link_libraries(${target} PRIVATE rgpot::rgpot Threads::Threads)
    set_target_properties(${target} PROPERTIES LINKER_LANGUAGE CXX)
  endforeach()
  rgpot_add_test(capi_threads COMMAND capi_threads)
  if(_rgpot_tsan)
    # GCC's runtime (14 through 16) aborts with "unexpected memory mapping"
    # when the kernel randomises mmap with more than 28 bits. setarch -R
    # turns ASLR off for it, where the sandbox allows the call.
    set(_rgpot_tsan_command capi_threads_tsan)
    find_program(RGPOT_SETARCH setarch)
    if(RGPOT_SETARCH AND NOT CMAKE_CROSSCOMPILING)
      execute_process(COMMAND "${RGPOT_SETARCH}" -R true
                      RESULT_VARIABLE _rgpot_setarch_result
                      OUTPUT_QUIET ERROR_QUIET)
      if(_rgpot_setarch_result EQUAL 0)
        set(_rgpot_tsan_command "${RGPOT_SETARCH}" -R
                                $<TARGET_FILE:capi_threads_tsan>)
      endif()
    endif()
    rgpot_add_test(capi_threads_tsan COMMAND ${_rgpot_tsan_command})
  endif()
endif()

# The XTB dlopen frontend without an engine: the bare-soname checks skip.
rgpot_test_executable(xtb_dlopen_test CppCore/tests/XTBDlopenTest.cc)
rgpot_add_test(XTBDlopenTest COMMAND xtb_dlopen_test)

# Exact arithmetic and instruction-set dispatch.
rgpot_test_enabled(MinImageIsaVariants _on)
if(_on)
  set(_rgpot_isa_exes)
  foreach(spec IN LISTS RGPOT_ISA_SPECS)
    string(TOUPPER "${spec}" SPEC)
    add_library(rgpot_minimage_${spec} OBJECT
                CppCore/rgpot/nlist/MinimumImage.cpp)
    target_include_directories(rgpot_minimage_${spec} PRIVATE CppCore)
    target_compile_options(rgpot_minimage_${spec} PRIVATE ${_rgpot_isa_${spec}})
    add_executable(minimage_hash_${spec} CppCore/tests/minimage_isa_hash_main.cc
                                         $<TARGET_OBJECTS:rgpot_minimage_${spec}>)
    target_include_directories(minimage_hash_${spec} PRIVATE CppCore)
    target_compile_options(minimage_hash_${spec} PRIVATE ${_rgpot_fp_contract_off})
    target_compile_definitions(minimage_hash_${spec} PRIVATE
                               RGPOT_MINIMAGE_ISA_${SPEC})
    target_link_libraries(minimage_hash_${spec} PRIVATE rgpot_exact_fma)
    list(APPEND _rgpot_isa_exes $<TARGET_FILE:minimage_hash_${spec}>)
  endforeach()
  rgpot_add_test(MinImageIsaVariants
    COMMAND "${RGPOT_TEST_ROOT}/ci/gha/check_minimage_isa.sh"
            d6f5cbaa04ac49e6f02ee64c10eb83d44d951756556924dcc17966453dcd8b11
            ${_rgpot_isa_exes})
endif()
foreach(case exact_fma_eq:ExactFmaEq exp_fixed_eq:ExpFixedEq)
  string(REPLACE ":" ";" case "${case}")
  list(GET case 0 _exe)
  list(GET case 1 _test)
  add_executable(${_exe} CppCore/tests/${_exe}.cc)
  target_include_directories(${_exe} PRIVATE CppCore include)
  target_compile_options(${_exe} PRIVATE ${_rgpot_fp_contract_off})
  target_link_libraries(${_exe} PRIVATE rgpot_exact_fma rgpot_minimage)
  rgpot_add_test(${_test} COMMAND ${_exe})
endforeach()
rgpot_test_executable(fma_path_eq CppCore/tests/fma_path_eq.cc)
target_compile_options(fma_path_eq PRIVATE ${_rgpot_fp_contract_off})
rgpot_add_test(FmaPathEq COMMAND fma_path_eq)
add_executable(simd_env CppCore/tests/simd_env.cc)
target_include_directories(simd_env PRIVATE CppCore include)
target_link_libraries(simd_env PRIVATE rgpot::rgpot rgpot_vesin)
set_target_properties(simd_env PROPERTIES LINKER_LANGUAGE CXX)
rgpot_add_test(simd-env-scalar COMMAND simd_env known scalar)
rgpot_add_test(simd-env-unknown COMMAND simd_env unknown)
rgpot_add_test(simd-env-rvv COMMAND simd_env known rvv)
rgpot_add_test(simd-env-avx2 COMMAND simd_env known avx2)

# The generic engine table over a stand-in backend: no rpc stack, no torch.
rgpot_test_enabled(EngineAdapterTest _on)
if(_on)
  add_library(engine_stub SHARED CppCore/tests/engine_stub.cc
                                 CppCore/rgpot/engine/EngineTable.cc
                                 CppCore/rgpot/engine/EngineAbiStamp.cc)
  target_include_directories(engine_stub PRIVATE CppCore include)
  target_compile_definitions(engine_stub PRIVATE ${_rgpot_test_definitions}
    RGPOT_ENGINE_BUILD
    RGPOT_ENGINE_PROJECT_VERSION="${PROJECT_VERSION}")
  target_link_libraries(engine_stub PRIVATE rgpot::rgpot)
  rgpot_test_executable(engine_adapter_test CppCore/tests/EngineAdapterTest.cc)
  rgpot_add_test(EngineAdapterTest COMMAND engine_adapter_test
    ENVIRONMENT "RGPOT_ENGINE_STUB=$<TARGET_FILE:engine_stub>"
                "RGPOT_ENGINE_STUB_MODEL=${CMAKE_CURRENT_BINARY_DIR}/engine_stub_model.bin")
endif()

add_subdirectory(CppCore/tests/plugin)

# The MOPAC frontend over a fake engine. The empty-variable case cannot
# empty the variable itself: the MSVC runtime has no setenv.
add_library(mopacc_fake_engine SHARED CppCore/tests/mopacc_fake_engine.c)
target_include_directories(mopacc_fake_engine PRIVATE CppCore/rgpot/MOPACPot)
rgpot_test_executable(mopac_pot_test CppCore/tests/MOPACPotTest.cc)
set(_rgpot_mopacc_env "RGPOT_MOPACC_ENGINE=$<TARGET_FILE:mopacc_fake_engine>")
rgpot_add_test(MOPACPotTest COMMAND mopac_pot_test
               ENVIRONMENT ${_rgpot_mopacc_env})
rgpot_add_test(MOPACPotEmptyVariable COMMAND mopac_pot_test "[empty-variable]"
               ENVIRONMENT ${_rgpot_mopacc_env} "MOPACC_LIBRARY=")

# Each frontend's C ABI against its no-op stub.
set(_rgpot_abi_stubs mopac:MOPACPot)
if(RGPOT_WITH_RPC)
  list(APPEND _rgpot_abi_stubs nwchem:NWChemPot cpmd:CPMDPot)
endif()
foreach(stub IN LISTS _rgpot_abi_stubs)
  string(REPLACE ":" ";" stub "${stub}")
  list(GET stub 0 _abi)
  list(GET stub 1 _dir)
  add_library(${_abi}_abi_stub STATIC CppCore/rgpot/${_dir}/${_abi}_c_abi_stub.c)
  target_include_directories(${_abi}_abi_stub PUBLIC CppCore/rgpot/${_dir})
  add_executable(test_${_abi}_abi CppCore/tests/test_${_abi}_abi_main.c)
  target_link_libraries(test_${_abi}_abi PRIVATE ${_abi}_abi_stub)
  rgpot_add_test(test_${_abi}_abi COMMAND test_${_abi}_abi)
endforeach()

if(NOT RGPOT_WITH_RPC)
  return()
endif()

# The RPC, NWChem and CPMD frontends over fake engines. CMake names its
# generated schema sources .c++, so the check of the build's generated
# files against a regeneration is meson's alone.
rgpot_add_test(RPCSchemaSync
  COMMAND Python3::Interpreter
          "${RGPOT_TEST_ROOT}/CppCore/tests/test_rpc_schema_sync.py"
          "${RGPOT_TEST_ROOT}" "${CAPNP_EXECUTABLE}")

add_library(nwchemc_fake_engine SHARED CppCore/tests/nwchemc_fake_engine.c)
target_include_directories(nwchemc_fake_engine PRIVATE CppCore/rgpot/NWChemPot)
add_library(nwchemc_lifetime_engine SHARED
            CppCore/tests/nwchemc_lifetime_engine.cc)
target_include_directories(nwchemc_lifetime_engine PRIVATE CppCore)
rgpot_test_executable(nwchem_lifetime_test CppCore/tests/NWChemPotLifetimeTest.cc)
foreach(mode ordinary probe)
  rgpot_add_test(NWChemPotLifetime-${mode}
    COMMAND Python3::Interpreter
            "${RGPOT_TEST_ROOT}/CppCore/tests/test_nwchem_lifetime.py"
            $<TARGET_FILE:nwchem_lifetime_test>
            $<TARGET_FILE:nwchemc_lifetime_engine> ${mode})
endforeach()
set(_rgpot_nwchemc_env "RGPOT_NWCHEM_ENGINE=$<TARGET_FILE:nwchemc_fake_engine>")
rgpot_test_executable(nwchem_pot_test CppCore/tests/NWChemPotTest.cc)
rgpot_add_test(NWChemPotTest COMMAND nwchem_pot_test
               ENVIRONMENT ${_rgpot_nwchemc_env})
rgpot_test_executable(nwchem_pot_message_abi_test
                      CppCore/tests/NWChemPotMessageAbiTest.cc)
rgpot_add_test(NWChemPotMessageAbiTest COMMAND nwchem_pot_message_abi_test
               ENVIRONMENT ${_rgpot_nwchemc_env})

add_library(cpmdc_fake_engine SHARED CppCore/tests/cpmdc_fake_engine.cc)
target_compile_definitions(cpmdc_fake_engine PRIVATE ${_rgpot_build_identity})
target_link_libraries(cpmdc_fake_engine PRIVATE ptlrpc)
add_library(cpmdc_gradient_engine SHARED CppCore/tests/cpmdc_gradient_engine.cc)
foreach(engine cpmdc_fake_engine cpmdc_gradient_engine)
  target_include_directories(${engine} PRIVATE CppCore include CppCore/rgpot/CPMDPot)
endforeach()
set(_rgpot_cpmdc_env "RGPOT_CPMD_ENGINE=$<TARGET_FILE:cpmdc_fake_engine>")
rgpot_test_executable(cpmd_pot_gradient_stress_test
                      CppCore/tests/CPMDPotGradientStressTest.cc)
rgpot_add_test(CPMDPotGradientStressTest COMMAND cpmd_pot_gradient_stress_test
  ENVIRONMENT "RGPOT_CPMD_ENGINE=$<TARGET_FILE:cpmdc_gradient_engine>")
rgpot_test_executable(composite_cpmd_dispersion_test
                      CppCore/tests/CompositeCpmdDispersionTest.cc)
rgpot_add_test(CompositeCpmdDispersionTest COMMAND composite_cpmd_dispersion_test
  ENVIRONMENT "RGPOT_CPMD_ENGINE=$<TARGET_FILE:cpmdc_gradient_engine>")
rgpot_test_executable(cpmd_pot_message_abi_test
                      CppCore/tests/CPMDPotMessageAbiTest.cc)
rgpot_add_test(CPMDPotMessageAbiTest COMMAND cpmd_pot_message_abi_test
               ENVIRONMENT ${_rgpot_cpmdc_env})
rgpot_test_executable(profile_loader_test CppCore/tests/ProfileLoaderTest.cc)
rgpot_add_test(ProfileLoaderTest COMMAND profile_loader_test
               ENVIRONMENT ${_rgpot_cpmdc_env})

# CP2K over a fake engine, and over libcp2k through the shim.
if(RGPOT_WITH_CP2K)
  add_library(cp2kc_fake_engine SHARED CppCore/tests/cp2kc_fake_engine.cc)
  target_include_directories(cp2kc_fake_engine PRIVATE CppCore include)
  rgpot_test_executable(cp2k_pot_fake_engine_test
                        CppCore/tests/CP2KPotFakeEngineTest.cc)
  rgpot_add_test(CP2KPotFakeEngineTest COMMAND cp2k_pot_fake_engine_test
    ENVIRONMENT "RGPOT_CP2K_ENGINE=$<TARGET_FILE:cp2kc_fake_engine>"
                "CP2KC_LIBRARY=$<TARGET_FILE:cp2kc_fake_engine>")
  rgpot_test_executable(cp2k_deck_test CppCore/tests/CP2KDeckTest.cc)
  rgpot_add_test(CP2KDeckTest COMMAND cp2k_deck_test)
  rgpot_test_executable(cp2k_pot_smoke_test CppCore/tests/CP2KPotSmokeTest.cc)
  set(_rgpot_cp2k_smoke_env "RGPOT_CP2K_ENGINE=$<TARGET_FILE:cp2kc_rgpot>"
                            "OMP_NUM_THREADS=2")
  if(LIBCP2K_PREFIX)
    list(APPEND _rgpot_cp2k_smoke_env
         "CP2K_DATA_DIR=${LIBCP2K_PREFIX}/share/cp2k/data")
  endif()
  rgpot_add_test(CP2KPotSmokeTest COMMAND cp2k_pot_smoke_test
                 ENVIRONMENT ${_rgpot_cp2k_smoke_env})
  rgpot_test_enabled(CP2KPotSmokeMpi _on)
  if(_on)
    set(_rgpot_cp2k_mpi_args -n 2)
    execute_process(COMMAND "${MPIEXEC_EXECUTABLE}" --version
                    OUTPUT_VARIABLE _rgpot_mpiexec_version ERROR_QUIET)
    if(_rgpot_mpiexec_version MATCHES "Open MPI|OpenRTE")
      list(PREPEND _rgpot_cp2k_mpi_args --oversubscribe)
    endif()
    # Every rank runs the test cases in declaration order: their CP2K calls
    # are collectives.
    rgpot_add_test(CP2KPotSmokeMpi
      COMMAND "${MPIEXEC_EXECUTABLE}" ${_rgpot_cp2k_mpi_args}
              $<TARGET_FILE:cp2k_pot_smoke_test> --order decl
      ENVIRONMENT ${_rgpot_cp2k_smoke_env})
  endif()
endif()

# MPI hosts and calculator groups. Each test runs the launcher itself.
rgpot_test_enabled(MpiHost-owned _on)
if(_on)
  add_executable(MpiHostContractTest CppCore/tests/MpiHostContractTest.cc)
  target_include_directories(MpiHostContractTest PRIVATE CppCore)
  target_link_libraries(MpiHostContractTest PRIVATE MPI::MPI_CXX)
  add_library(collective_engine SHARED CppCore/tests/nwchemc_collective_engine.cc)
  target_link_libraries(collective_engine PRIVATE rgpot::rgpot MPI::MPI_CXX)
  add_executable(MpiRpcContractClient CppCore/tests/MpiRpcContractClient.cc)
  target_compile_definitions(MpiRpcContractClient PRIVATE ${_rgpot_test_definitions})
  target_link_libraries(MpiRpcContractClient PRIVATE rgpot::rgpot CapnProto::capnp-rpc)
  set(_rgpot_mpi_host_script "${RGPOT_TEST_ROOT}/CppCore/tests/test_mpi_hosts.py")
  foreach(mode owned borrowed)
    rgpot_add_test(MpiHost-${mode}
      COMMAND Python3::Interpreter "${_rgpot_mpi_host_script}" ${mode}
              "${MPIEXEC_EXECUTABLE}" $<TARGET_FILE:MpiHostContractTest>)
    set_tests_properties(MpiHost-${mode} PROPERTIES PROCESSORS 4)
  endforeach()
  rgpot_add_test(NWChemMpiForceHost
    COMMAND Python3::Interpreter "${_rgpot_mpi_host_script}" force
            "${MPIEXEC_EXECUTABLE}" $<TARGET_FILE:nwchem_mpi_force_host>
            $<TARGET_FILE:collective_engine>)
  set_tests_properties(NWChemMpiForceHost PROPERTIES PROCESSORS 2)
  set(_rgpot_backends NWChem CPMD)
  if(RGPOT_WITH_CP2K)
    list(APPEND _rgpot_backends CP2K)
  endif()
  foreach(backend IN LISTS _rgpot_backends)
    foreach(layout serial mpi)
      set(_rpc_server potserv)
      if(layout STREQUAL "mpi")
        set(_rpc_server potserv_mpi)
      endif()
      rgpot_add_test(CollectiveRPC-${backend}-${layout}
        COMMAND Python3::Interpreter "${_rgpot_mpi_host_script}" rpc
                "${MPIEXEC_EXECUTABLE}" $<TARGET_FILE:${_rpc_server}>
                $<TARGET_FILE:MpiRpcContractClient>
                $<TARGET_FILE:collective_engine> ${backend} ${layout})
      set_tests_properties(CollectiveRPC-${backend}-${layout}
                           PROPERTIES PROCESSORS 2)
    endforeach()
  endforeach()

  find_package(MPI REQUIRED COMPONENTS C CXX)
  set(_rgpot_tests "${RGPOT_TEST_ROOT}/CppCore/tests")
  rgpot_test_executable(rank_exception_abort CppCore/tests/RankExceptionAbort.cc)
  rgpot_test_executable(refused_split CppCore/tests/RefusedSplit.cc)
  rgpot_test_executable(adopt_comm CppCore/tests/AdoptComm.cc)
  foreach(target rank_exception_abort refused_split adopt_comm)
    target_link_libraries(${target} PRIVATE rgpot_mpi MPI::MPI_CXX)
  endforeach()
  foreach(case abort:RankExceptionAbort uneven:CalculatorUnevenBatch
               finalize-owner:RankExceptionFinalizeOwner
               finalize-guest:RankExceptionFinalizeGuest
               share-bad:ShareBroadcastMismatch share-ok:ShareBroadcastMatch)
    string(REPLACE ":" ";" case "${case}")
    list(GET case 0 _mode)
    list(GET case 1 _test)
    rgpot_add_test(${_test}
      COMMAND "${_rgpot_tests}/rank_exception_abort.sh"
              $<TARGET_FILE:rank_exception_abort> ${_mode}
      ENVIRONMENT ${_rgpot_cpmdc_env})
  endforeach()
  rgpot_add_test(RefusedSplit
    COMMAND "${_rgpot_tests}/refused_split.sh" $<TARGET_FILE:refused_split>)

  add_library(cpmdc_adopt_engine SHARED CppCore/tests/cpmdc_adopt_engine.cc)
  target_include_directories(cpmdc_adopt_engine PRIVATE CppCore include
                                                        CppCore/rgpot/CPMDPot)
  target_link_libraries(cpmdc_adopt_engine PRIVATE MPI::MPI_CXX)
  add_library(mpi_split_profile SHARED CppCore/tests/mpi_split_profile.c)
  target_link_libraries(mpi_split_profile PRIVATE MPI::MPI_C)
  set(_rgpot_adopt_cases
      missing:AdoptCommMissing:cpmdc_gradient_engine
      split:AdoptCommSplit:cpmdc_adopt_engine
      refused-hook:AdoptCommRefusedHook:cpmdc_adopt_engine
      probe:AdoptCommProbe:cpmdc_adopt_engine
      instance-first:AdoptCommInstanceFirst:cpmdc_adopt_engine
      bound-static:AdoptCommBoundStatic:cpmdc_adopt_engine
      bound-instance:AdoptCommBoundInstance:cpmdc_adopt_engine
      refused-force:AdoptCommRefusedForce:cpmdc_adopt_engine)
  if(RGPOT_WITH_CP2K)
    add_library(cp2kc_adopt_engine SHARED CppCore/tests/cp2kc_adopt_engine.cc)
    target_include_directories(cp2kc_adopt_engine PRIVATE CppCore include)
    target_link_libraries(cp2kc_adopt_engine PRIVATE MPI::MPI_CXX)
    list(APPEND _rgpot_adopt_cases cp2k-split:AdoptCommCP2K:cp2kc_adopt_engine)
  endif()
  foreach(case IN LISTS _rgpot_adopt_cases)
    string(REPLACE ":" ";" case "${case}")
    list(GET case 0 _mode)
    list(GET case 1 _test)
    list(GET case 2 _engine)
    rgpot_add_test(${_test}
      COMMAND "${_rgpot_tests}/adopt_comm.sh" ${_mode} $<TARGET_FILE:adopt_comm>
              $<TARGET_FILE:${_engine}> $<TARGET_FILE:mpi_split_profile>)
  endforeach()

  rgpot_test_enabled(CollectiveCache _on)
  if(_on)
    rgpot_test_executable(collective_cache_test
                          CppCore/tests/CollectiveCacheTest.cc)
    target_link_libraries(collective_cache_test PRIVATE rgpot_mpi MPI::MPI_CXX)
    rgpot_add_test(CollectiveCache
      COMMAND "${_rgpot_tests}/collective_cache.sh"
              $<TARGET_FILE:collective_cache_test>)
  endif()
endif()
