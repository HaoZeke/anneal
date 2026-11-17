// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/IPIPot/IPIPot.hpp"

#include "rgpot/IPIPot/IPIWire.hpp"
#include "rgpot/ParamHash.hpp"
#include "rgpot/units.hpp"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <poll.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace rgpot {
namespace {

using namespace ipi::detail;

constexpr const char *kWho = "IPIPot";
constexpr const char *kPeer = "i-PI driver";

[[noreturn]] void fail(const std::string &message) {
  ipi::detail::fail(kWho, message);
}

[[noreturn]] void fail_errno(const char *what) { failErrno(kWho, what); }

Link wire(int fd) { return Link{fd, kWho, kPeer}; }

const Header kStatus = header("STATUS");
const Header kPosdata = header("POSDATA");
const Header kGetforce = header("GETFORCE");
const Header kExit = header("EXIT");
const Header kInit = header("INIT");
const Header kReady = header("READY");
const Header kNeedinit = header("NEEDINIT");
const Header kForceready = header("FORCEREADY");

void append_bytes(std::vector<char> &buf, const void *data, size_t n) {
  const auto *bytes = static_cast<const char *>(data);
  buf.insert(buf.end(), bytes, bytes + n);
}

int listen_tcp(const std::string &host, int port, std::string *endpoint) {
  addrinfo hints{};
  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_STREAM;
  addrinfo *res = nullptr;
  const std::string port_text = std::to_string(port);
  const int gai = ::getaddrinfo(host.c_str(), port_text.c_str(), &hints, &res);
  if (gai != 0 || res == nullptr) {
    fail(std::string("cannot resolve ") + host + ": " + gai_strerror(gai));
  }
  int fd = -1;
  for (addrinfo *it = res; it != nullptr; it = it->ai_next) {
    fd = ::socket(it->ai_family, it->ai_socktype, it->ai_protocol);
    if (fd < 0) {
      continue;
    }
    setCloexec(fd);
    setNonblock(fd);
    disableSigpipe(fd);
    const int one = 1;
    ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    if (::bind(fd, it->ai_addr, it->ai_addrlen) == 0) {
      break;
    }
    ::close(fd);
    fd = -1;
  }
  ::freeaddrinfo(res);
  if (fd < 0) {
    fail_errno("binding the TCP socket");
  }
  if (::listen(fd, 1) != 0) {
    const int saved = errno;
    ::close(fd);
    errno = saved;
    fail_errno("listening on the TCP socket");
  }
  sockaddr_in bound{};
  socklen_t len = sizeof(bound);
  if (::getsockname(fd, reinterpret_cast<sockaddr *>(&bound), &len) != 0) {
    const int saved = errno;
    ::close(fd);
    errno = saved;
    fail_errno("reading the bound TCP port");
  }
  *endpoint = "tcp:" + host + ":" + std::to_string(ntohs(bound.sin_port));
  return fd;
}

int listen_unix(const std::string &path, std::string *endpoint) {
  if (path.size() >= sizeof(sockaddr_un::sun_path)) {
    throw std::invalid_argument("IPIPot: UNIX socket path is too long");
  }
  ::unlink(path.c_str());
  const int fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
  if (fd < 0) {
    fail_errno("opening a UNIX socket");
  }
  setCloexec(fd);
  setNonblock(fd);
  disableSigpipe(fd);
  sockaddr_un addr{};
  addr.sun_family = AF_UNIX;
  std::memcpy(addr.sun_path, path.c_str(), path.size());
  addr.sun_path[path.size()] = '\0';
  // Darwin reads sa_len. A zero length makes later accept/connect fault.
#ifdef __APPLE__
  addr.sun_len = sizeof(addr);
#endif
  if (::bind(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) != 0) {
    const int saved = errno;
    ::close(fd);
    errno = saved;
    fail_errno("binding the UNIX socket");
  }
  if (::listen(fd, 1) != 0) {
    const int saved = errno;
    ::close(fd);
    ::unlink(path.c_str());
    errno = saved;
    fail_errno("listening on the UNIX socket");
  }
  *endpoint = "unix:" + path;
  return fd;
}

} // namespace

IPIPot::IPIPot(const IPIConfig &config)
    : Potential(PotType::IPI), m_config(config) {
  if (!(config.timeout_s >= 0.0) || !std::isfinite(config.timeout_s)) {
    throw std::invalid_argument("IPIPot: timeout_s must be finite and >= 0");
  }
  bindSocket();
  Fnv1a fp;
  fp.u64(1);
  fp.str(m_config.address);
  fp.f64(m_config.timeout_s);
  m_paramsKey = fp.h;
}

IPIPot::~IPIPot() {
  std::lock_guard<std::mutex> lock(m_mu);
  if (m_conn >= 0) {
    try {
      const Deadline deadline = Deadline::after(1.0, kWho);
      writeFull(wire(m_conn), kExit.data(), kExit.size(), deadline);
    } catch (...) {
    }
  }
  closeConn();
  closeListen();
}

void IPIPot::bindSocket() {
  const Address address = parseAddress(m_config.address, kWho);
  m_tcp = address.tcp;
  if (address.tcp) {
    m_listen = listen_tcp(address.host, address.port, &m_endpoint);
  } else {
    m_unix_path = address.unixPath;
    m_listen = listen_unix(address.unixPath, &m_endpoint);
  }
}

void IPIPot::closeConn() const {
  if (m_conn >= 0) {
    ::close(m_conn);
    m_conn = -1;
  }
}

void IPIPot::closeListen() {
  if (m_listen >= 0) {
    ::close(m_listen);
    m_listen = -1;
  }
  if (!m_unix_path.empty()) {
    ::unlink(m_unix_path.c_str());
    m_unix_path.clear();
  }
}

bool IPIPot::hasStress() const {
  std::lock_guard<std::mutex> lock(m_mu);
  return m_has_stress != 0;
}

bool IPIPot::copyStress(double out[9]) const {
  std::lock_guard<std::mutex> lock(m_mu);
  if (m_has_stress == 0 || out == nullptr) {
    return false;
  }
  std::copy(m_stress, m_stress + 9, out);
  return true;
}

void IPIPot::evaluateLocked(const double h[9], const double ih[9],
                            const double *pos_bohr, size_t n_atoms,
                            double *energy, double *forces,
                            double virial[9]) const {
  const Deadline deadline = Deadline::after(m_config.timeout_s, kWho);
  if (m_conn < 0) {
    while (m_conn < 0) {
      waitFd(wire(m_listen), POLLIN, deadline, "waiting for an i-PI driver");
      const int fd = ::accept(m_listen, nullptr, nullptr);
      if (fd < 0) {
        if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) {
          continue;
        }
        fail_errno("accepting an i-PI driver");
      }
      setCloexec(fd);
      setNonblock(fd);
      disableSigpipe(fd);
      if (m_tcp) {
        setNodelay(fd);
      }
      m_conn = fd;
    }
  }
  bool ready = false;
  for (int attempt = 0; attempt < 8 && !ready; ++attempt) {
    writeFull(wire(m_conn), kStatus.data(), kStatus.size(), deadline);
    char reply[kHeaderLen];
    readFull(wire(m_conn), reply, kHeaderLen, deadline);
    if (sameHeader(reply, kReady)) {
      ready = true;
    } else if (sameHeader(reply, kNeedinit)) {
      std::vector<char> init;
      init.reserve(kHeaderLen + 8);
      append_bytes(init, kInit.data(), kInit.size());
      const int32_t rid = 0;
      const int32_t len = 0;
      append_bytes(init, &rid, sizeof(rid));
      append_bytes(init, &len, sizeof(len));
      writeFull(wire(m_conn), init.data(), init.size(), deadline);
    } else {
      fail("unexpected driver header '" + headerText(reply) + "'");
    }
  }
  if (!ready) {
    fail("driver did not become READY");
  }

  const auto n32 = static_cast<int32_t>(n_atoms);
  std::vector<char> payload;
  payload.reserve(kHeaderLen * 2 + 8 * (18 + static_cast<size_t>(n32) * 3) + 4);
  append_bytes(payload, kPosdata.data(), kPosdata.size());
  append_bytes(payload, h, 9 * sizeof(double));
  append_bytes(payload, ih, 9 * sizeof(double));
  append_bytes(payload, &n32, sizeof(n32));
  append_bytes(payload, pos_bohr,
               static_cast<size_t>(n32) * 3 * sizeof(double));
  append_bytes(payload, kGetforce.data(), kGetforce.size());
  writeFull(wire(m_conn), payload.data(), payload.size(), deadline);

  char reply[kHeaderLen];
  readFull(wire(m_conn), reply, kHeaderLen, deadline);
  if (!sameHeader(reply, kForceready)) {
    fail("unexpected driver header '" + headerText(reply) + "'");
  }
  readFull(wire(m_conn), energy, sizeof(double), deadline);
  int32_t nat = 0;
  readFull(wire(m_conn), &nat, sizeof(nat), deadline);
  if (nat != n32) {
    fail("driver atom count does not match the geometry");
  }
  readFull(wire(m_conn), forces, static_cast<size_t>(nat) * 3 * sizeof(double),
           deadline);
  readFull(wire(m_conn), virial, 9 * sizeof(double), deadline);
  int32_t extra = 0;
  readFull(wire(m_conn), &extra, sizeof(extra), deadline);
  if (extra < 0 || extra > kMaxString) {
    fail("driver extra string length is invalid");
  }
  if (extra > 0) {
    std::vector<char> sink(static_cast<size_t>(extra));
    readFull(wire(m_conn), sink.data(), sink.size(), deadline);
  }
}

void IPIPot::forceImpl(const ForceInput &in, ForceOut *out) const {
  if (out == nullptr || out->F == nullptr) {
    throw std::invalid_argument("IPIPot: force output is missing");
  }
  if (in.pos == nullptr || in.box == nullptr) {
    throw std::invalid_argument("IPIPot: positions and cell are required");
  }
  if (in.nAtoms == 0) {
    throw std::invalid_argument("IPIPot: at least one atom is required");
  }
  if (in.nAtoms > static_cast<size_t>(kMaxAtoms)) {
    throw std::invalid_argument(
        "IPIPot: atom count exceeds the protocol limit");
  }
  const size_t n3 = in.nAtoms * 3;
  for (size_t i = 0; i < n3; ++i) {
    if (!std::isfinite(in.pos[i])) {
      throw std::invalid_argument("IPIPot: positions must be finite");
    }
  }
  for (int i = 0; i < 9; ++i) {
    if (!std::isfinite(in.box[i])) {
      throw std::invalid_argument("IPIPot: cell must be finite");
    }
  }

  std::vector<double> pos_bohr(n3);
  for (size_t i = 0; i < n3; ++i) {
    pos_bohr[i] = in.pos[i] * units::ANGSTROM_TO_BOHR;
  }
  double h[9];
  double ih[9];
  rowsToIpiCell(in.box, h, ih, kWho);

  double energy_ha = 0.0;
  std::vector<double> forces_ha(n3);
  double virial_ha[9]{};
  {
    std::lock_guard<std::mutex> lock(m_mu);
    m_has_stress = 0;
    try {
      evaluateLocked(h, ih, pos_bohr.data(), in.nAtoms, &energy_ha,
                     forces_ha.data(), virial_ha);
    } catch (...) {
      closeConn();
      throw;
    }

    out->energy = energy_ha * units::HARTREE_TO_EV;
    out->variance = 0.0;
    for (size_t i = 0; i < n3; ++i) {
      out->F[i] = forces_ha[i] * units::HARTREE_BOHR_TO_EV_ANGSTROM;
    }
    const double volume = cellVolume(in.box);
    if (volume > 0.0) {
      const double scale = -units::HARTREE_TO_EV / volume;
      for (int i = 0; i < 9; ++i) {
        m_stress[i] = virial_ha[i] * scale;
      }
      m_has_stress = 1;
    }
  }
}

} // namespace rgpot
