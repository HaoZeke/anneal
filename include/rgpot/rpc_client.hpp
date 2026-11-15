#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @file rpc_client.hpp
 * @author rgpot Developers
 * @date 2026-02-15
 * @brief Move-only RAII wrapper around the rgpot Cap'n Proto RPC client.
 *
 * Provides @c rgpot::RpcClient, a thin C++ handle that connects to a
 * remote rgpot server and proxies @c calculate() calls over Cap'n Proto
 * RPC.  The class is only available when the Rust core is compiled with
 * the @c rpc feature. The build defines the @c RGPOT_HAS_RPC preprocessor
 * macro (Meson adds it with @c with_rust_core); @c rgpot.h only tests it.
 *
 * # Example
 * @code
 * #ifdef RGPOT_HAS_RPC
 * rgpot::RpcClient client("localhost", 12345);
 *
 * double pos[] = {0,0,0, 1,0,0};
 * int    atm[] = {1, 1};
 * double box[] = {10,0,0, 0,10,0, 0,0,10};
 * rgpot::InputSpec input(2, pos, atm, box);
 *
 * auto result = client.calculate(input);
 * double energy = result.energy();
 * #endif
 * @endcode
 *
 * @ingroup rgpot_cpp
 */

#include <string>

#include "rgpot.h"
#include "rgpot/errors.hpp"
#include "rgpot/types.hpp"

#ifdef RGPOT_HAS_RPC

namespace rgpot {

/**
 * @class RpcClient
 * @brief Move-only RAII wrapper around @c rgpot_rpc_client_t.
 * @ingroup rgpot_cpp
 *
 * Owns an opaque RPC client handle allocated by the Rust core.  The
 * connection is established during construction and torn down on
 * destruction via @c rgpot_rpc_client_free().  The class is non-copyable;
 * move semantics transfer ownership of the connection.
 */
class RpcClient final {
public:
  /**
   * @brief Connect to a remote rgpot server.
   *
   * Calls @c rgpot_rpc_client_new() and throws on failure.
   *
   * @param host Hostname or IP address of the server.
   * @param port TCP port the server listens on.
   * @throws rgpot::Error if the connection cannot be established.
   */
  RpcClient(const std::string &host, uint16_t port) {
    handle_ = rgpot_rpc_client_new(host.c_str(), port);
    if (!handle_) {
      const char *msg = rgpot_last_error();
      throw Error(msg ? msg : "failed to create RPC client");
    }
  }

  /**
   * @brief Destructor — disconnects and frees the Rust-side client.
   *
   * Safe to call on a moved-from handle (internal pointer is @c nullptr).
   */
  ~RpcClient() {
    if (handle_) {
      rgpot_rpc_client_free(handle_);
    }
  }

  /**
   * @brief Move constructor — transfers connection ownership.
   * @param other Client to move from; left in a null state.
   */
  RpcClient(RpcClient &&other) noexcept : handle_(other.handle_) {
    other.handle_ = nullptr;
  }

  /**
   * @brief Move assignment — releases current connection, transfers
   *        ownership.
   * @param other Client to move from; left in a null state.
   * @return Reference to @c *this.
   */
  RpcClient &operator=(RpcClient &&other) noexcept {
    if (this != &other) {
      if (handle_) {
        rgpot_rpc_client_free(handle_);
      }
      handle_ = other.handle_;
      other.handle_ = nullptr;
    }
    return *this;
  }

  RpcClient(const RpcClient &) = delete;
  RpcClient &operator=(const RpcClient &) = delete;

  /**
   * @brief Perform a remote force/energy calculation.
   *
   * Serialises the input via Cap'n Proto, sends the request to the
   * server, and deserialises the response into a @c CalcResult.
   *
   * @param input The atomic configuration to evaluate.
   * @return A @c CalcResult containing energy, variance, and forces.
   * @throws rgpot::Error on RPC transport or server-side failure.
   */
  CalcResult calculate(const InputSpec &input) {
    CalcResult result;
    auto status =
        rgpot_rpc_calculate(handle_, &input.c_struct(), &result.c_struct());
    detail::check_status(status);
    return result;
  }

private:
  rgpot_rpc_client_t *handle_ = nullptr; //!< Owned opaque RPC client handle.
};

} // namespace rgpot

#endif // RGPOT_HAS_RPC
