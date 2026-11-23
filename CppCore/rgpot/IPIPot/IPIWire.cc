// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/IPIPot/IPIWire.hpp"

#include "rgpot/stress.hpp"
#include "rgpot/units.hpp"

#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace rgpot::ipi::detail {

Header header(std::string_view text) {
  Header out;
  out.fill(' ');
  const auto n = std::min(text.size(), static_cast<size_t>(kHeaderLen));
  std::copy_n(text.begin(), n, out.begin());
  return out;
}

bool sameHeader(const char *got, const Header &want) {
  return std::memcmp(got, want.data(), kHeaderLen) == 0;
}

std::string headerText(const char *got) {
  std::string text(got, got + kHeaderLen);
  for (char &ch : text) {
    if (ch < 32 || ch > 126) {
      ch = '?';
    }
  }
  while (!text.empty() && text.back() == ' ') {
    text.pop_back();
  }
  return text;
}

double cellVolume(const double *box) { return rgpot::cellVolume(box); }

Deadline Deadline::after(double seconds, const char *who) {
  if (!(seconds >= 0.0) || !std::isfinite(seconds)) {
    throw std::invalid_argument(std::string(who) +
                                ": timeout_s must be finite and >= 0");
  }
  const auto delta = std::chrono::duration_cast<Clock::duration>(
      std::chrono::duration<double>(seconds));
  return Deadline{Clock::now() + delta, false};
}

Deadline Deadline::never() { return Deadline{Clock::time_point{}, true}; }

bool Deadline::expired() const { return !unbounded && Clock::now() >= end; }

int Deadline::remainingMs() const {
  if (unbounded) {
    return -1;
  }
  const auto left = end - Clock::now();
  if (left <= Clock::duration::zero()) {
    return 0;
  }
  // Rounded up, so poll() does not return before the deadline.
  const auto ms = std::chrono::ceil<std::chrono::milliseconds>(left).count();
  if (ms > std::numeric_limits<int>::max()) {
    return std::numeric_limits<int>::max();
  }
  return static_cast<int>(ms);
}

void fail(const char *who, const std::string &message) {
  throw std::runtime_error(std::string(who) + ": " + message);
}

void failErrno(const char *who, const std::string &what) {
  fail(who, what + " (errno " + std::to_string(errno) + ")");
}

namespace {

// False when hangup_ok and the peer has gone; throws otherwise.
bool waitFdOrHangup(const Link &link, short event, const Deadline &deadline,
                    const char *what, bool hangup_ok) {
  while (true) {
    if (deadline.expired()) {
      fail(link.who, std::string("timed out ") + what);
    }
    pollfd pfd{};
    pfd.fd = link.fd;
    pfd.events = event;
    const int rc = ::poll(&pfd, 1, deadline.remainingMs());
    if (rc == 0) {
      fail(link.who, std::string("timed out ") + what);
    }
    if (rc < 0) {
      if (errno == EINTR) {
        continue;
      }
      failErrno(link.who, what);
    }
    if ((pfd.revents & POLLNVAL) != 0) {
      fail(link.who, what);
    }
    // Linux reports a reset peer as POLLERR. A hangup with no queued bytes
    // is the same disconnect; POLLIN still queued is handled by recv.
    if ((pfd.revents & POLLERR) != 0 ||
        ((pfd.revents & POLLHUP) != 0 && (pfd.revents & event) == 0)) {
      if (hangup_ok) {
        return false;
      }
      fail(link.who, std::string(link.peer) + " disconnected");
    }
    return true;
  }
}

} // namespace

void waitFd(const Link &link, short event, const Deadline &deadline,
            const char *what) {
  waitFdOrHangup(link, event, deadline, what, false);
}

void writeFull(const Link &link, const void *data, size_t n,
               const Deadline &deadline) {
  const std::string what = std::string("writing to the ") + link.peer;
  const auto *bytes = static_cast<const char *>(data);
  size_t sent = 0;
  while (sent < n) {
    waitFd(link, POLLOUT, deadline, what.c_str());
#ifdef MSG_NOSIGNAL
    constexpr int kSendFlags = MSG_NOSIGNAL;
#else
    constexpr int kSendFlags = 0;
#endif
    const ssize_t rc = ::send(link.fd, bytes + sent, n - sent, kSendFlags);
    if (rc < 0) {
      if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) {
        continue;
      }
      if (errno == EPIPE || errno == ECONNRESET) {
        fail(link.who, std::string(link.peer) + " disconnected");
      }
      failErrno(link.who, what);
    }
    if (rc == 0) {
      fail(link.who, std::string(link.peer) + " disconnected");
    }
    sent += static_cast<size_t>(rc);
  }
}

void readFull(const Link &link, void *data, size_t n,
              const Deadline &deadline) {
  const std::string what = std::string("reading from the ") + link.peer;
  auto *bytes = static_cast<char *>(data);
  size_t got = 0;
  while (got < n) {
    waitFd(link, POLLIN, deadline, what.c_str());
    const ssize_t rc = ::recv(link.fd, bytes + got, n - got, 0);
    if (rc < 0) {
      if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) {
        continue;
      }
      if (errno == ECONNRESET) {
        fail(link.who, std::string(link.peer) + " disconnected");
      }
      failErrno(link.who, what);
    }
    if (rc == 0) {
      fail(link.who, std::string(link.peer) + " disconnected");
    }
    got += static_cast<size_t>(rc);
  }
}

bool readFullOrEof(const Link &link, void *data, size_t n,
                   const Deadline &deadline) {
  const std::string what = std::string("reading from the ") + link.peer;
  auto *bytes = static_cast<char *>(data);
  ssize_t rc = 0;
  while (true) {
    if (!waitFdOrHangup(link, POLLIN, deadline, what.c_str(), true)) {
      return false;
    }
    rc = ::recv(link.fd, bytes, n, 0);
    if (rc > 0) {
      break;
    }
    if (rc == 0 || errno == ECONNRESET) {
      return false;
    }
    if (errno != EINTR && errno != EAGAIN && errno != EWOULDBLOCK) {
      failErrno(link.who, what);
    }
  }
  const auto got = static_cast<size_t>(rc);
  if (got < n) {
    readFull(link, bytes + got, n - got, deadline);
  }
  return true;
}

void setCloexec(int fd) {
  const int flags = ::fcntl(fd, F_GETFD);
  if (flags >= 0) {
    ::fcntl(fd, F_SETFD, flags | FD_CLOEXEC);
  }
}

void setNonblock(int fd) {
  const int flags = ::fcntl(fd, F_GETFL);
  if (flags >= 0) {
    ::fcntl(fd, F_SETFL, flags | O_NONBLOCK);
  }
}

void disableSigpipe(int fd) {
#ifdef SO_NOSIGPIPE
  const int one = 1;
  ::setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof(one));
#else
  (void)fd;
#endif
}

void setNodelay(int fd) {
  const int one = 1;
  ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
}

Address parseAddress(const std::string &address, const char *who,
                     const std::string &unix_prefix) {
  const std::string tag(who);
  Address out;
  if (address.rfind("unix:", 0) == 0) {
    const std::string rest = address.substr(5);
    if (rest.empty()) {
      throw std::invalid_argument(tag + ": UNIX address is empty");
    }
    if (rest.front() == '/') {
      out.unixPath = rest;
    } else {
      if (rest.find('/') != std::string::npos) {
        throw std::invalid_argument(
            tag + ": UNIX name must not contain '/'; use unix:/absolute/path");
      }
      out.unixPath = unix_prefix + rest;
    }
    return out;
  }
  if (address.rfind("tcp:", 0) != 0) {
    throw std::invalid_argument(tag +
                                ": address must start with unix: or tcp:");
  }
  out.tcp = true;
  const std::string rest = address.substr(4);
  const auto colon = rest.rfind(':');
  std::string port_text;
  if (colon == std::string::npos) {
    out.host = "127.0.0.1";
    port_text = rest;
  } else {
    out.host = rest.substr(0, colon);
    if (out.host.empty()) {
      out.host = "127.0.0.1";
    }
    port_text = rest.substr(colon + 1);
  }
  if (port_text.empty()) {
    throw std::invalid_argument(tag + ": TCP port is missing");
  }
  const std::string range = tag + ": TCP port must be an integer 0..65535";
  try {
    std::size_t used = 0;
    const int value = std::stoi(port_text, &used);
    if (used != port_text.size() || value < 0 || value > 65535) {
      throw std::invalid_argument(range);
    }
    out.port = value;
  } catch (const std::invalid_argument &) {
    throw std::invalid_argument(range);
  } catch (const std::out_of_range &) {
    throw std::invalid_argument(range);
  }
  return out;
}

void rowsToIpiCell(const double *box_angstrom, double h[9], double ih[9],
                   const char *who) {
  double box_bohr[9];
  for (int i = 0; i < 9; ++i) {
    box_bohr[i] = box_angstrom[i] * units::ANGSTROM_TO_BOHR;
  }
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      h[row * 3 + col] = box_bohr[col * 3 + row];
    }
  }
  const double a = h[0], b = h[1], c = h[2];
  const double d = h[3], e = h[4], f = h[5];
  const double g = h[6], hh = h[7], i = h[8];
  const double det =
      a * (e * i - f * hh) - b * (d * i - f * g) + c * (d * hh - e * g);
  if (!std::isfinite(det) || det == 0.0) {
    throw std::invalid_argument(std::string(who) + ": cell matrix is singular");
  }
  const double s = 1.0 / det;
  ih[0] = (e * i - f * hh) * s;
  ih[1] = (c * hh - b * i) * s;
  ih[2] = (b * f - c * e) * s;
  ih[3] = (f * g - d * i) * s;
  ih[4] = (a * i - c * g) * s;
  ih[5] = (c * d - a * f) * s;
  ih[6] = (d * hh - e * g) * s;
  ih[7] = (b * g - a * hh) * s;
  ih[8] = (a * e - b * d) * s;
}

void ipiCellToRows(const double h[9], double box_angstrom[9]) {
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      box_angstrom[row * 3 + col] = h[col * 3 + row] * units::BOHR_TO_ANGSTROM;
    }
  }
}

} // namespace rgpot::ipi::detail
