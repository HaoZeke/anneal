set(ENV{CARGO_TARGET_DIR} "parent-build-lock")
include("${CMAKE_CURRENT_LIST_DIR}/cargo_target_isolation.cmake")
if(DEFINED ENV{CARGO_TARGET_DIR})
    message(FATAL_ERROR "native configuration retains the parent Cargo target")
endif()
execute_process(
    COMMAND "${CMAKE_COMMAND}" -E environment
    OUTPUT_VARIABLE child_environment
    COMMAND_ERROR_IS_FATAL ANY
)
if(child_environment MATCHES "(^|\n)CARGO_TARGET_DIR=")
    message(FATAL_ERROR "nested commands inherit the parent Cargo build lock")
endif()
