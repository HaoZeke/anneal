#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @file IPIWire.hpp
 * @brief Framing, socket and cell helpers shared by the i-PI server
 * (IPIPot) and the i-PI driver client (IPIDriver).
 *
 * i-PI socket protocol: Ceriotti, More and Manolopoulos, Comput. Phys.
 * Commun. 185, 1019 (2014), doi:10.1016/j.cpc.2013.10.027; i-PI 3.0,
 * J. Chem. Phys. 161, 062504 (2024), doi:10.1063/5.0215869.
 *
 * Headers are 12 bytes of space-padded ASCII. Integers are int32 and
 * floats float64 in host byte order, as in the reference C wrappers and
 * driver.py. Every error is thrown with the `who` prefix of the caller
 * ("IPIPot: ..." or "IPIDriver: ...").
 */

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace rgpot::ipi::detail {

inline constexpr int kHeaderLen = 12;
inline constexpr int32_t kMaxAtoms = 10'000'000;
inline constexpr int32_t kMaxString = 1 << 20;
inline constexpr const char *kUnixPrefix = "/tmp/ipi_";

using Header = std::array<char, kHeaderLen>;

/// Space-padded header; text longer than 12 bytes is cut.
Header header(std::string_view text);
bool sameHeader(const char *got, const Header &want);
/// Printable form of a received header, trailing spaces removed.
std::string headerText(const char *got);

/// Absolute volume of a row-major 3x3 cell. A null or singular cell is 0.
double cellVolume(const double *box);

using Clock = std::chrono::steady_clock;

/// Absolute end of one blocking exchange. never() has no end.
struct Deadline {
  Clock::time_point end{};
  bool unbounded = false;

  /// Throws std::invalid_argument for a negative or non-finite budget.
  static Deadline after(double seconds, const char *who);
  static Deadline never();

  [[nodiscard]] bool expired() const;
  /// Milliseconds left for poll(); -1 when unbounded.
  [[nodiscard]] int remainingMs() const;
};

/// One connected or listening socket, and the names used in errors.
struct Link {
  int fd = -1;
  const char *who = "";  ///< "IPIPot" or "IPIDriver"
  const char *peer = ""; ///< "i-PI driver" or "i-PI server"
};

[[noreturn]] void fail(const char *who, const std::string &message);
[[noreturn]] void failErrno(const char *who, const std::string &what);

/// Waits for `event` (POLLIN or POLLOUT). Throws "timed out <what>" or
/// "<peer> disconnected".
void waitFd(const Link &link, short event, const Deadline &deadline,
            const char *what);
void writeFull(const Link &link, const void *data, size_t n,
               const Deadline &deadline);
void readFull(const Link &link, void *data, size_t n, const Deadline &deadline);

void setCloexec(int fd);
void setNonblock(int fd);
void disableSigpipe(int fd);
void setNodelay(int fd);

/// Parsed form of `unix:/abs/path`, `unix:<name>`, `tcp:<host>:<port>` or
/// `tcp:<port>`. `unix:<name>` maps to `<unix_prefix><name>`.
struct Address {
  bool tcp = false;
  std::string host;
  int port = 0;
  std::string unixPath;
};

/// Throws std::invalid_argument, prefixed with `who`.
Address parseAddress(const std::string &address, const char *who,
                     const std::string &unix_prefix = kUnixPrefix);

/// rgpot cell (lattice vectors in rows, Angstrom) to the i-PI cell
/// (vectors in columns, bohr) and its inverse. Throws
/// std::invalid_argument for a singular cell.
void rowsToIpiCell(const double *box_angstrom, double h[9], double ih[9],
                   const char *who);

/// i-PI cell (vectors in columns, bohr) to rgpot rows in Angstrom.
void ipiCellToRows(const double h[9], double box_angstrom[9]);

} // namespace rgpot::ipi::detail
