#pragma once

/**
 * \file anneal-core.hpp
 * \brief C++ companion header for anneal-core.
 *
 * Hand-written companion to the cbindgen-generated `anneal-core.h`. Wraps
 * the C ABI exports in the `anneal` namespace for ergonomic C++ usage.
 */

#include "anneal-core.h"

namespace anneal {

/// Returns the anneal-core package version as a NUL-terminated C string.
inline const char* version() noexcept {
    return anneal_core_version();
}

}  // namespace anneal
