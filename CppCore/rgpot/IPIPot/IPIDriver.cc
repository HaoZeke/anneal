// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/IPIPot/IPIDriver.hpp"

#include "rgpot/IPIPot/IPIWire.hpp"
#include "rgpot/units.hpp"

#include <netdb.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <utility>

namespace rgpot {
namespace {

using namespace ipi::detail;

constexpr const char *kWho = "IPIDriver";
constexpr const char *kPeer = "i-PI server";
// Same open-box rule as the PLUMED and GROMACS hosts: a cube wider than
// twice the cluster plus the cutoff keeps every real pair and no image.
constexpr double kOpenBoxFloor = 100.0;
constexpr double kOpenBoxMargin = 10.0;

const Header kStatus = header("STATUS");
const Header kInit = header("INIT");
const Header kPosdata = header("POSDATA");
const Header kGetforce = header("GETFORCE");
const Header kExit = header("EXIT");
const Header kReady = header("READY");
const Header kNeedinit = header("NEEDINIT");
const Header kHavedata = header("HAVEDATA");
const Header kForceready = header("FORCEREADY");

[[noreturn]] void fail(const std::string &message) {
  ipi::detail::fail(kWho, message);
}

void append_bytes(std::vector<char> &buf, const void *data, size_t n) {
  const auto *bytes = static_cast<const char *>(data);
  buf.insert(buf.end(), bytes, bytes + n);
}

bool retryable(int err) {
  return err == ECONNREFUSED || err == ENOENT || err == EAGAIN ||
         err == ECONNRESET || err == ETIMEDOUT || err == EHOSTUNREACH ||
         err == ENETUNREACH;
}

// Non-blocking connect bounded by the deadline. Returns the fd, or -1 with
// errno set when the attempt failed.
int try_connect(int family, int type, int proto, const sockaddr *addr,
                socklen_t len, const Deadline &deadline) {
  const int fd = ::socket(family, type, proto);
  if (fd < 0) {
    return -1;
  }
  setCloexec(fd);
  setNonblock(fd);
  disableSigpipe(fd);
  if (::connect(fd, addr, len) == 0) {
    return fd;
  }
  if (errno != EINPROGRESS) {
    const int saved = errno;
    ::close(fd);
    errno = saved;
    return -1;
  }
  pollfd pfd{};
  pfd.fd = fd;
  pfd.events = POLLOUT;
  int rc = 0;
  do {
    rc = ::poll(&pfd, 1, deadline.remainingMs());
  } while (rc < 0 && errno == EINTR);
  int err = ETIMEDOUT;
  if (rc > 0) {
    socklen_t elen = sizeof(err);
    if (::getsockopt(fd, SOL_SOCKET, SO_ERROR, &err, &elen) != 0) {
      err = errno;
    }
  } else if (rc < 0) {
    err = errno;
  }
  if (err == 0) {
    return fd;
  }
  ::close(fd);
  errno = err;
  return -1;
}

int connect_unix(const std::string &path, const Deadline &deadline) {
  sockaddr_un addr{};
  addr.sun_family = AF_UNIX;
  std::memcpy(addr.sun_path, path.c_str(), path.size());
  addr.sun_path[path.size()] = '\0';
#ifdef __APPLE__
  addr.sun_len = sizeof(addr);
#endif
  return try_connect(AF_UNIX, SOCK_STREAM, 0,
                     reinterpret_cast<const sockaddr *>(&addr), sizeof(addr),
                     deadline);
}

int connect_tcp(const std::string &host, int port, const Deadline &deadline) {
  addrinfo hints{};
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  addrinfo *res = nullptr;
  const std::string port_text = std::to_string(port);
  const int gai = ::getaddrinfo(host.c_str(), port_text.c_str(), &hints, &res);
  if (gai != 0 || res == nullptr) {
    throw std::runtime_error(std::string(kWho) + ": cannot resolve " + host +
                             ": " + gai_strerror(gai));
  }
  int fd = -1;
  int err = ECONNREFUSED;
  for (addrinfo *it = res; it != nullptr && fd < 0; it = it->ai_next) {
    fd = try_connect(it->ai_family, it->ai_socktype, it->ai_protocol,
                     it->ai_addr, it->ai_addrlen, deadline);
    if (fd < 0) {
      err = errno;
    }
  }
  ::freeaddrinfo(res);
  if (fd >= 0) {
    setNodelay(fd);
  } else {
    errno = err;
  }
  return fd;
}

void check_finite(const double *values, size_t n, const char *what) {
  for (size_t i = 0; i < n; ++i) {
    if (!std::isfinite(values[i])) {
      fail(std::string(what) + " from the i-PI server are not finite");
    }
  }
}

} // namespace

IPIDriver::IPIDriver(const IPIDriverConfig &config, IPIEvaluator evaluate)
    : m_config(config), m_evaluate(std::move(evaluate)) {
  const std::string tag(kWho);
  if (!m_evaluate) {
    throw std::invalid_argument(tag + ": the evaluator is empty");
  }
  if (m_config.species.empty()) {
    throw std::invalid_argument(tag + ": the species list is empty");
  }
  if (m_config.species.size() > static_cast<size_t>(kMaxAtoms)) {
    throw std::invalid_argument(tag +
                                ": atom count exceeds the protocol limit");
  }
  for (const int z : m_config.species) {
    if (z < 1 || z > 118) {
      throw std::invalid_argument(tag + ": atomic number " + std::to_string(z) +
                                  " is not 1..118");
    }
  }
  auto bad = [](double v) { return !(v >= 0.0) || !std::isfinite(v); };
  if (bad(m_config.connect_timeout_s) || bad(m_config.timeout_s)) {
    throw std::invalid_argument(tag + ": timeouts must be finite and >= 0");
  }
  if (bad(m_config.cutoff)) {
    throw std::invalid_argument(tag + ": cutoff must be finite and >= 0");
  }
  if (bad(m_config.stress_step) || m_config.stress_step >= 0.1) {
    throw std::invalid_argument(tag + ": stress_step must be in [0, 0.1)");
  }
  const Address address =
      parseAddress(m_config.address, kWho, m_config.unix_prefix);
  m_tcp = address.tcp;
  if (m_tcp) {
    if (address.port == 0) {
      throw std::invalid_argument(tag + ": TCP port 0 cannot be connected to");
    }
    m_host = address.host;
    m_port = address.port;
    m_endpoint = "tcp:" + m_host + ":" + std::to_string(m_port);
  } else {
    if (address.unixPath.size() >= sizeof(sockaddr_un::sun_path)) {
      throw std::invalid_argument(tag + ": UNIX socket path is too long");
    }
    m_unix_path = address.unixPath;
    m_endpoint = "unix:" + m_unix_path;
  }
  m_forces.assign(m_config.species.size() * 3, 0.0);
}

IPIDriver::~IPIDriver() { close(); }

void IPIDriver::close() noexcept {
  if (m_fd >= 0) {
    ::close(m_fd);
    m_fd = -1;
  }
  m_initialised = false;
  m_have_data = false;
}

void IPIDriver::connect() {
  if (m_fd >= 0) {
    return;
  }
  const Deadline deadline = Deadline::after(m_config.connect_timeout_s, kWho);
  while (true) {
    const int fd = m_tcp ? connect_tcp(m_host, m_port, deadline)
                         : connect_unix(m_unix_path, deadline);
    if (fd >= 0) {
      m_fd = fd;
      return;
    }
    const int err = errno;
    if (!retryable(err) || deadline.expired()) {
      ipi::detail::fail(kWho, "cannot connect to " + m_endpoint + " (" +
                                  std::strerror(err) + ")");
    }
    const int wait = std::min(50, std::max(deadline.remainingMs(), 1));
    std::this_thread::sleep_for(std::chrono::milliseconds(wait));
  }
}

void IPIDriver::run() {
  connect();
  while (step()) {
  }
}

bool IPIDriver::step() {
  if (m_fd < 0) {
    fail("not connected");
  }
  const Link link{m_fd, kWho, kPeer};
  const Deadline deadline = m_config.timeout_s > 0.0
                                ? Deadline::after(m_config.timeout_s, kWho)
                                : Deadline::never();
  try {
    char got[kHeaderLen];
    // ASE's SocketIOCalculator closes without EXIT; its own client takes a
    // close between messages as EXIT, and so does this one.
    if (!readFullOrEof(link, got, kHeaderLen, deadline)) {
      close();
      return false;
    }
    if (sameHeader(got, kStatus)) {
      const Header &reply = !m_initialised ? kNeedinit
                            : m_have_data  ? kHavedata
                                           : kReady;
      writeFull(link, reply.data(), reply.size(), deadline);
      return true;
    }
    if (sameHeader(got, kInit)) {
      int32_t bead = 0;
      int32_t len = 0;
      readFull(link, &bead, sizeof(bead), deadline);
      readFull(link, &len, sizeof(len), deadline);
      if (len < 0 || len > kMaxString) {
        fail("INIT string length is invalid");
      }
      std::string init(static_cast<size_t>(len), '\0');
      if (len > 0) {
        readFull(link, init.data(), init.size(), deadline);
      }
      m_bead = bead;
      m_init = std::move(init);
      m_initialised = true;
      return true;
    }
    if (sameHeader(got, kPosdata)) {
      double h[9];
      double ih[9];
      int32_t nat = 0;
      readFull(link, h, sizeof(h), deadline);
      readFull(link, ih, sizeof(ih), deadline);
      readFull(link, &nat, sizeof(nat), deadline);
      if (nat <= 0 || nat > kMaxAtoms) {
        fail("POSDATA atom count " + std::to_string(nat) + " is invalid");
      }
      if (static_cast<size_t>(nat) != m_config.species.size()) {
        fail("POSDATA atom count " + std::to_string(nat) +
             " does not match the " + std::to_string(m_config.species.size()) +
             " species the driver was given");
      }
      std::vector<double> pos(static_cast<size_t>(nat) * 3);
      readFull(link, pos.data(), pos.size() * sizeof(double), deadline);
      check_finite(h, 9, "cell values");
      check_finite(pos.data(), pos.size(), "positions");
      m_have_data = false;
      compute(h, pos.data());
      m_have_data = true;
      ++m_evals;
      return true;
    }
    if (sameHeader(got, kGetforce)) {
      if (!m_have_data) {
        fail("GETFORCE before POSDATA");
      }
      const auto nat = static_cast<int32_t>(m_config.species.size());
      const auto extra = static_cast<int32_t>(m_extra.size());
      std::vector<char> payload;
      payload.reserve(kHeaderLen + 8 + 4 + m_forces.size() * 8 + 72 + 4 +
                      m_extra.size());
      append_bytes(payload, kForceready.data(), kForceready.size());
      append_bytes(payload, &m_energy, sizeof(m_energy));
      append_bytes(payload, &nat, sizeof(nat));
      append_bytes(payload, m_forces.data(), m_forces.size() * sizeof(double));
      append_bytes(payload, m_virial, sizeof(m_virial));
      append_bytes(payload, &extra, sizeof(extra));
      append_bytes(payload, m_extra.data(), m_extra.size());
      writeFull(link, payload.data(), payload.size(), deadline);
      m_have_data = false;
      return true;
    }
    if (sameHeader(got, kExit)) {
      close();
      return false;
    }
    fail("unexpected server header '" + headerText(got) + "'");
  } catch (...) {
    close();
    throw;
  }
}

namespace {

struct Evaluated {
  double energy = 0.0;
  std::vector<double> forces;
  double virial[9]{}; ///< W = -sigma V of the caller's cell, eV.
  bool has_virial = false;
};

class Geometry {
public:
  Geometry(const std::vector<int> &species, double cutoff,
           const IPIEvaluator &evaluate)
      : m_species(species), m_cutoff(cutoff), m_evaluate(evaluate) {}

  /// rows in Angstrom (lattice vectors as rows), pos in Angstrom.
  Evaluated operator()(const double rows[9], const double *pos) const {
    const size_t n = m_species.size();
    const double volume = cellVolume(rows);
    std::vector<double> xyz(pos, pos + 3 * n);
    std::vector<int> z = m_species;
    double box[9];
    size_t copies = 1;
    if (!(volume > 0.0)) {
      openBox(&xyz, box);
    } else {
      std::copy(rows, rows + 9, box);
      copies = tile(&xyz, &z, box);
    }
    const size_t total = n * copies;
    std::vector<double> forces(3 * total, 0.0);
    ForceInput in{total, xyz.data(), z.data(), box};
    ForceOut out{forces.data(), 0.0, 0.0, {}, 0};
    m_evaluate(in, &out);
    Evaluated res;
    res.energy = out.energy / static_cast<double>(copies);
    forces.resize(3 * n);
    res.forces = std::move(forces);
    if (out.has_stress != 0) {
      // Stress is intensive. -sigma V of the kernel cell, over copies, is
      // W of the caller's cell; an open box has no images, so its W does
      // not depend on the padding.
      const double scale = -cellVolume(box) / static_cast<double>(copies);
      for (int k = 0; k < 9; ++k) {
        res.virial[k] = out.stress[k] * scale;
      }
      res.has_virial = true;
    }
    return res;
  }

private:
  void openBox(std::vector<double> *xyz, double box[9]) const {
    const size_t n = m_species.size();
    double lo[3] = {(*xyz)[0], (*xyz)[1], (*xyz)[2]};
    double hi[3] = {lo[0], lo[1], lo[2]};
    for (size_t i = 0; i < n; ++i) {
      for (int k = 0; k < 3; ++k) {
        lo[k] = std::min(lo[k], (*xyz)[3 * i + k]);
        hi[k] = std::max(hi[k], (*xyz)[3 * i + k]);
      }
    }
    double extent = 0.0;
    for (int k = 0; k < 3; ++k) {
      extent = std::max(extent, hi[k] - lo[k]);
    }
    const double side =
        std::max(kOpenBoxFloor, 2.0 * (extent + m_cutoff) + kOpenBoxMargin);
    for (size_t i = 0; i < n; ++i) {
      for (int k = 0; k < 3; ++k) {
        (*xyz)[3 * i + k] += 0.5 * side - 0.5 * (lo[k] + hi[k]);
      }
    }
    std::fill(box, box + 9, 0.0);
    box[0] = box[4] = box[8] = side;
  }

  // A minimum-image kernel sees each neighbour inside the cutoff once
  // when every perpendicular width of the cell exceeds twice the cutoff.
  size_t tile(std::vector<double> *xyz, std::vector<int> *z,
              double box[9]) const {
    if (!(m_cutoff > 0.0)) {
      return 1;
    }
    const double volume = cellVolume(box);
    size_t reps[3] = {1, 1, 1};
    for (int i = 0; i < 3; ++i) {
      const double *b = box + 3 * ((i + 1) % 3);
      const double *c = box + 3 * ((i + 2) % 3);
      const double cx = b[1] * c[2] - b[2] * c[1];
      const double cy = b[2] * c[0] - b[0] * c[2];
      const double cz = b[0] * c[1] - b[1] * c[0];
      const double width = volume / std::sqrt(cx * cx + cy * cy + cz * cz);
      if (width <= 2.0 * m_cutoff) {
        reps[i] = static_cast<size_t>(std::floor(2.0 * m_cutoff / width)) + 1;
      }
    }
    const size_t copies = reps[0] * reps[1] * reps[2];
    if (copies == 1) {
      return 1;
    }
    const size_t n = m_species.size();
    std::vector<double> tiled;
    tiled.reserve(3 * n * copies);
    z->clear();
    z->reserve(n * copies);
    for (size_t a = 0; a < reps[0]; ++a) {
      for (size_t b = 0; b < reps[1]; ++b) {
        for (size_t c = 0; c < reps[2]; ++c) {
          double shift[3];
          for (int k = 0; k < 3; ++k) {
            shift[k] = static_cast<double>(a) * box[k] +
                       static_cast<double>(b) * box[3 + k] +
                       static_cast<double>(c) * box[6 + k];
          }
          for (size_t i = 0; i < n; ++i) {
            for (int k = 0; k < 3; ++k) {
              tiled.push_back((*xyz)[3 * i + k] + shift[k]);
            }
          }
          z->insert(z->end(), m_species.begin(), m_species.end());
        }
      }
    }
    *xyz = std::move(tiled);
    for (int i = 0; i < 3; ++i) {
      for (int k = 0; k < 3; ++k) {
        box[3 * i + k] *= static_cast<double>(reps[i]);
      }
    }
    return copies;
  }

  const std::vector<int> &m_species;
  double m_cutoff;
  const IPIEvaluator &m_evaluate;
};

// x (I + strain) for every row of a 3-column array.
void deform(const double *in, size_t rows, const double strain[9],
            double *out) {
  for (size_t r = 0; r < rows; ++r) {
    const double *x = in + 3 * r;
    for (int j = 0; j < 3; ++j) {
      out[3 * r + j] =
          x[j] + x[0] * strain[j] + x[1] * strain[3 + j] + x[2] * strain[6 + j];
    }
  }
}

} // namespace

void IPIDriver::compute(const double h[9], const double *pos_bohr) {
  const size_t n = m_config.species.size();
  double rows[9];
  ipiCellToRows(h, rows);
  std::vector<double> pos(3 * n);
  for (size_t i = 0; i < 3 * n; ++i) {
    pos[i] = pos_bohr[i] * units::BOHR_TO_ANGSTROM;
  }
  const Geometry geometry(m_config.species, m_config.cutoff, m_evaluate);
  Evaluated res = geometry(rows, pos.data());

  const char *source = "kernel";
  if (!res.has_virial && m_config.stress_step > 0.0) {
    // A symmetric shear strain d puts d in both (i, j) and (j, i), so it
    // moves the energy by 2 dE/de_ij d, against dE/de_ii d on the diagonal.
    const double step = m_config.stress_step;
    std::vector<double> moved(3 * n);
    double moved_rows[9];
    for (int i = 0; i < 3; ++i) {
      for (int j = i; j < 3; ++j) {
        double e[2];
        for (int s = 0; s < 2; ++s) {
          double strain[9]{};
          strain[3 * i + j] = strain[3 * j + i] = s == 0 ? step : -step;
          deform(rows, 3, strain, moved_rows);
          deform(pos.data(), n, strain, moved.data());
          e[s] = geometry(moved_rows, moved.data()).energy;
        }
        const double dede = (e[0] - e[1]) / ((i == j ? 2.0 : 4.0) * step);
        res.virial[3 * i + j] = res.virial[3 * j + i] = -dede;
      }
    }
    res.has_virial = true;
    source = "finite_difference";
  } else if (!res.has_virial) {
    source = "none";
  }

  m_energy = res.energy * units::EV_TO_HARTREE;
  for (size_t i = 0; i < 3 * n; ++i) {
    m_forces[i] = res.forces[i] / units::HARTREE_BOHR_TO_EV_ANGSTROM;
  }
  for (int k = 0; k < 9; ++k) {
    m_virial[k] = res.has_virial ? res.virial[k] * units::EV_TO_HARTREE : 0.0;
  }
  if (m_config.extras_json) {
    m_extra = std::string("{\"virial\": \"") + source +
              "\", \"virial_units\": \"hartree\"}";
  } else {
    m_extra.clear();
  }
}

} // namespace rgpot
