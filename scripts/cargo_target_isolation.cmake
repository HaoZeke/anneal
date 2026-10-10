# Native dependency configuration can launch Cargo while its caller holds
# the inherited target directory's lock. Child projects own their targets.
if(NOT "$ENV{ANNEAL_CMAKE_BASE_TOOLCHAIN}" STREQUAL "")
    include("$ENV{ANNEAL_CMAKE_BASE_TOOLCHAIN}")
endif()
unset(ENV{CARGO_TARGET_DIR})
