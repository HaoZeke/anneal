# CMake glue for anneal-core.
#
# Mirrors the meson script: drives the Rust crate build through Corrosion
# (so cargo handles every transitive crate dep), regenerates the public
# C header via cbindgen, and exposes an `anneal-core::anneal-core` target
# that downstream `find_package`/`add_subdirectory` consumers can link
# against. The cargo-c-installed pkg-config file (`anneal-core.pc`) gives
# system-wide consumers an alternate, install-time entry point.
cmake_minimum_required(VERSION 3.22)
project(anneal-core VERSION 0.2.0 LANGUAGES C CXX)

# Corrosion builds the Rust crate from Cargo.toml
include(FetchContent)
FetchContent_Declare(
    Corrosion
    GIT_REPOSITORY https://github.com/corrosion-rs/corrosion.git
    GIT_TAG v0.5
)
FetchContent_MakeAvailable(Corrosion)

corrosion_import_crate(MANIFEST_PATH ${CMAKE_CURRENT_SOURCE_DIR}/Cargo.toml
                       CRATES anneal-core)
# Corrosion >=0.5 mirrors cargo's post-1.79 dash-to-underscore rule for
# library target names, so the imported target is `anneal_core` even though
# the package name is `anneal-core`.
corrosion_set_env_vars(anneal_core "CARGO_CFG_TARGET_OS=$<LOWER_CASE:${CMAKE_SYSTEM_NAME}>")

# Generate the C header via cbindgen
find_program(CBINDGEN cbindgen REQUIRED)
set(ANNEAL_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/include/anneal-core.h)
add_custom_command(
    OUTPUT ${ANNEAL_HEADER}
    COMMAND ${CBINDGEN}
        -q
        --config ${CMAKE_CURRENT_SOURCE_DIR}/cbindgen.toml
        --crate anneal
        --output ${ANNEAL_HEADER}
        -- ${CMAKE_CURRENT_SOURCE_DIR}/src/lib.rs
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/cbindgen.toml
            ${CMAKE_CURRENT_SOURCE_DIR}/src/lib.rs
    COMMENT "Generating anneal-core.h via cbindgen"
)
add_custom_target(anneal-core-header DEPENDS ${ANNEAL_HEADER})
add_dependencies(anneal_core anneal-core-header)

# Exported target for consumers: anneal-core::anneal-core
add_library(anneal-core::anneal-core ALIAS anneal_core)

target_include_directories(anneal_core INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Install header alongside the library
install(FILES ${ANNEAL_HEADER} DESTINATION include)
