#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @file errors.hpp
 * @author rgpot Developers
 * @date 2026-02-15
 * @brief Maps C status codes to C++ exceptions.
 *
 * Provides @c rgpot::Error (an @c std::runtime_error subclass) and the
 * internal @c detail::check_status() helper used by all RAII wrappers.
 * On a non-success status code, @c check_status fetches the thread-local
 * error message from the Rust core via @c rgpot_last_error() and throws
 * an appropriately categorised exception.
 *
 * @ingroup rgpot_cpp
 */

#include <stdexcept>
#include <string>

#include "rgpot/c_api.hpp"

namespace rgpot {

/**
 * @class Error
 * @brief Exception type for errors originating from the rgpot Rust core.
 * @ingroup rgpot_cpp
 *
 * Thrown by the RAII wrappers when the underlying C API call returns a
 * non-success status code.  The @c what() string includes both the error
 * category and the human-readable message from @c rgpot_last_error().
 */
class Error : public std::runtime_error {
public:
  /**
   * @brief Constructs an Error with the given message.
   * @param msg Human-readable description of the failure.
   */
  explicit Error(const std::string &msg) : std::runtime_error(msg) {}
};

namespace detail {

/**
 * @brief Checks a status code and throws @c rgpot::Error on failure.
 *
 * If @a status is @c RGPOT_SUCCESS this function returns immediately.
 * Otherwise it calls @c rgpot_last_error() to retrieve the Rust-side
 * error message, prepends a category prefix, and throws @c rgpot::Error.
 *
 * @param status The status code returned by a C API function.
 * @return Void.
 * @throws rgpot::Error when @a status != @c RGPOT_SUCCESS.
 */
inline void check_status(rgpot_status_t status) {
  if (status == RGPOT_SUCCESS) {
    return;
  }
  const char *msg = rgpot_last_error();
  std::string error_msg = msg ? msg : "unknown error";

  switch (status) {
  case RGPOT_INVALID_PARAMETER:
    throw Error("invalid parameter: " + error_msg);
  case RGPOT_RPC_ERROR:
    throw Error("RPC error: " + error_msg);
  case RGPOT_BUFFER_SIZE_ERROR:
    throw Error("buffer size error: " + error_msg);
  case RGPOT_INTERNAL_ERROR:
  default:
    throw Error("internal error: " + error_msg);
  }
}

} // namespace detail
} // namespace rgpot
