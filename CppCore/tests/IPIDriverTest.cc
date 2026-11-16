// MIT License
// Copyright 2023--present rgpot developers
//
// IPIDriver against a fake in-process i-PI server: the STATUS / INIT /
// POSDATA / GETFORCE / EXIT state machine, units and the cell transpose,
// open and tiled cells, the finite-difference virial, malformed input,
// disconnects and timeouts. The last cases run IPIPot and IPIDriver
// against each other and compare with a direct evaluation.

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <catch2/catch_all.hpp>

#include "rgpot/IPIPot/IPIDriver.hpp"
#include "rgpot/IPIPot/IPIPot.hpp"
#include "rgpot/IPIPot/IPIWire.hpp"
#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/Morse/MorsePot.hpp"
#include "rgpot/SuttonChen/SuttonChenPot.hpp"
#include "rgpot/units.hpp"

using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
namespace wire = rgpot::ipi::detail;
namespace units = rgpot::units;

namespace {

constexpr int kHeader = 12;

std::string temp_socket(const std::string &tag) {
  static std::atomic<int> counter{0};
  const auto dir = std::filesystem::temp_directory_path();
  const std::string name = "rgpot-ipid-" + std::to_string(::getpid()) + "-" +
                           tag + "-" + std::to_string(counter++);
  return (dir / name).string();
}

// Minimal blocking server: one listening socket and one accepted peer.
class FakeServer {
public:
  static FakeServer unix_at(const std::string &path) {
    FakeServer s;
    s.m_path = path;
    ::unlink(path.c_str());
    s.m_listen = ::socket(AF_UNIX, SOCK_STREAM, 0);
    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    std::memcpy(addr.sun_path, path.c_str(), path.size() + 1);
#ifdef __APPLE__
    addr.sun_len = sizeof(addr);
#endif
    if (::bind(s.m_listen, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) !=
            0 ||
        ::listen(s.m_listen, 1) != 0) {
      throw std::runtime_error("fake server cannot listen on " + path);
    }
    s.m_address = "unix:" + path;
    return s;
  }

  static FakeServer tcp_any() {
    FakeServer s;
    s.m_listen = ::socket(AF_INET, SOCK_STREAM, 0);
    const int one = 1;
    ::setsockopt(s.m_listen, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = 0;
    ::inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
    socklen_t len = sizeof(addr);
    if (::bind(s.m_listen, reinterpret_cast<sockaddr *>(&addr), len) != 0 ||
        ::listen(s.m_listen, 1) != 0 ||
        ::getsockname(s.m_listen, reinterpret_cast<sockaddr *>(&addr), &len) !=
            0) {
      throw std::runtime_error("fake server cannot listen on tcp");
    }
    s.m_address = "tcp:127.0.0.1:" + std::to_string(ntohs(addr.sin_port));
    return s;
  }

  FakeServer() = default;
  FakeServer(FakeServer &&o) noexcept { *this = std::move(o); }
  FakeServer &operator=(FakeServer &&o) noexcept {
    std::swap(m_listen, o.m_listen);
    std::swap(m_conn, o.m_conn);
    std::swap(m_path, o.m_path);
    std::swap(m_address, o.m_address);
    return *this;
  }
  ~FakeServer() {
    hangup();
    if (m_listen >= 0) {
      ::close(m_listen);
    }
    if (!m_path.empty()) {
      ::unlink(m_path.c_str());
    }
  }

  [[nodiscard]] const std::string &address() const { return m_address; }

  void accept() {
    pollfd pfd{m_listen, POLLIN, 0};
    if (::poll(&pfd, 1, 5000) != 1) {
      throw std::runtime_error("fake server: no driver connected");
    }
    m_conn = ::accept(m_listen, nullptr, nullptr);
    if (m_conn < 0) {
      throw std::runtime_error("fake server: accept failed");
    }
  }

  void hangup() {
    if (m_conn >= 0) {
      ::close(m_conn);
      m_conn = -1;
    }
  }

  void send(const void *data, size_t n) {
    const auto *bytes = static_cast<const char *>(data);
    size_t sent = 0;
    while (sent < n) {
#ifdef MSG_NOSIGNAL
      const ssize_t rc = ::send(m_conn, bytes + sent, n - sent, MSG_NOSIGNAL);
#else
      const ssize_t rc = ::send(m_conn, bytes + sent, n - sent, 0);
#endif
      if (rc <= 0) {
        throw std::runtime_error("fake server: send failed");
      }
      sent += static_cast<size_t>(rc);
    }
  }

  void header(const char *text) {
    const auto h = wire::header(text);
    send(h.data(), h.size());
  }

  template <typename T> void value(T v) { send(&v, sizeof(v)); }

  /// False at end of stream.
  bool recv(void *data, size_t n) {
    auto *bytes = static_cast<char *>(data);
    size_t got = 0;
    while (got < n) {
      pollfd pfd{m_conn, POLLIN, 0};
      if (::poll(&pfd, 1, 5000) != 1) {
        throw std::runtime_error("fake server: read timed out");
      }
      const ssize_t rc = ::recv(m_conn, bytes + got, n - got, 0);
      if (rc == 0) {
        return false;
      }
      if (rc < 0) {
        if (errno == EINTR) {
          continue;
        }
        if (errno == ECONNRESET) {
          return false;
        }
        throw std::runtime_error("fake server: recv failed");
      }
      got += static_cast<size_t>(rc);
    }
    return true;
  }

  std::string reply() {
    char h[kHeader];
    if (!recv(h, kHeader)) {
      return "<eof>";
    }
    return wire::headerText(h);
  }

  std::string status() {
    header("STATUS");
    return reply();
  }

  void init(int32_t bead, const std::string &text) {
    header("INIT");
    value(bead);
    value(static_cast<int32_t>(text.size()));
    send(text.data(), text.size());
  }

  void posdata(const double h[9], const std::vector<double> &pos_bohr,
               int32_t nat) {
    double ih[9]{};
    header("POSDATA");
    send(h, 9 * sizeof(double));
    send(ih, 9 * sizeof(double));
    value(nat);
    send(pos_bohr.data(), pos_bohr.size() * sizeof(double));
  }

  struct Forces {
    double energy = 0.0;
    std::vector<double> forces;
    std::array<double, 9> virial{};
    std::string extra;
  };

  Forces getforce() {
    header("GETFORCE");
    REQUIRE(reply() == "FORCEREADY");
    Forces out;
    int32_t nat = 0;
    REQUIRE(recv(&out.energy, sizeof(double)));
    REQUIRE(recv(&nat, sizeof(nat)));
    out.forces.resize(static_cast<size_t>(nat) * 3);
    REQUIRE(recv(out.forces.data(), out.forces.size() * sizeof(double)));
    REQUIRE(recv(out.virial.data(), 9 * sizeof(double)));
    int32_t len = 0;
    REQUIRE(recv(&len, sizeof(len)));
    out.extra.resize(static_cast<size_t>(len));
    if (len > 0) {
      REQUIRE(recv(out.extra.data(), out.extra.size()));
    }
    return out;
  }

  /// True when the driver closed its end.
  bool peer_closed() {
    char byte = 0;
    return !recv(&byte, 1);
  }

private:
  int m_listen = -1;
  int m_conn = -1;
  std::string m_path;
  std::string m_address;
};

// Runs driver.run() on a thread and keeps its exception text.
class DriverThread {
public:
  explicit DriverThread(rgpot::IPIDriver &driver)
      : m_thread([this, &driver] {
          try {
            driver.run();
          } catch (const std::exception &e) {
            std::lock_guard<std::mutex> lock(m_mu);
            m_error = e.what();
          }
        }) {}
  ~DriverThread() {
    if (m_thread.joinable()) {
      m_thread.join();
    }
  }
  std::string join() {
    m_thread.join();
    std::lock_guard<std::mutex> lock(m_mu);
    return m_error;
  }

private:
  std::mutex m_mu;
  std::string m_error;
  std::thread m_thread;
};

// Evaluation with fixed results that records what it was given.
struct Recorder {
  std::mutex mu;
  std::vector<double> box;
  std::vector<double> pos;
  std::vector<int> z;
  size_t calls = 0;
  double energy = -1.5;
  bool stress = true;
  std::array<double, 9> sigma{
      {0.01, 0.002, -0.003, 0.002, 0.02, 0.004, -0.003, 0.004, 0.03}};

  rgpot::IPIEvaluator evaluator() {
    return [this](const rgpot::ForceInput &in, rgpot::ForceOut *out) {
      std::lock_guard<std::mutex> lock(mu);
      ++calls;
      box.assign(in.box, in.box + 9);
      pos.assign(in.pos, in.pos + 3 * in.nAtoms);
      z.assign(in.atmnrs, in.atmnrs + in.nAtoms);
      out->energy = energy;
      for (size_t i = 0; i < 3 * in.nAtoms; ++i) {
        out->F[i] = 0.1 * static_cast<double>(i + 1);
      }
      if (stress) {
        std::copy(sigma.begin(), sigma.end(), out->stress);
        out->has_stress = 1;
      }
    };
  }
};

// Lattice vectors as rows, Angstrom; a sheared, non-orthogonal cell.
constexpr double kRows[9] = {12.0, 0.0, 0.0, 1.5, 11.0, 0.0, -0.7, 2.1, 13.0};

void ipi_cell(const double rows[9], double h[9]) {
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      h[r * 3 + c] = rows[c * 3 + r] * units::ANGSTROM_TO_BOHR;
    }
  }
}

rgpot::IPIDriverConfig config_for(const std::string &address,
                                  std::vector<int> species) {
  rgpot::IPIDriverConfig cfg;
  cfg.address = address;
  cfg.species = std::move(species);
  cfg.connect_timeout_s = 5.0;
  cfg.timeout_s = 10.0;
  return cfg;
}

} // namespace

TEST_CASE("IPIDriver runs the i-PI state machine over a UNIX socket",
          "[ipi_driver]") {
  auto server = FakeServer::unix_at(temp_socket("seq"));
  Recorder rec;
  rgpot::IPIDriver driver(config_for(server.address(), {29, 29, 28}),
                          rec.evaluator());
  REQUIRE(driver.endpoint() == server.address());
  DriverThread thread(driver);
  server.accept();

  REQUIRE(server.status() == "NEEDINIT");
  server.init(3, "{\"bead\": 3}");
  REQUIRE(server.status() == "READY");

  double h[9];
  ipi_cell(kRows, h);
  const std::vector<double> pos_a{0.0, 0.0, 0.0, 1.8, 0.0, 0.0, 0.0, 1.8, 0.4};
  std::vector<double> pos_bohr(pos_a.size());
  for (size_t i = 0; i < pos_a.size(); ++i) {
    pos_bohr[i] = pos_a[i] * units::ANGSTROM_TO_BOHR;
  }
  server.posdata(h, pos_bohr, 3);
  REQUIRE(server.status() == "HAVEDATA");
  const auto res = server.getforce();
  REQUIRE(server.status() == "READY");

  {
    std::lock_guard<std::mutex> lock(rec.mu);
    REQUIRE(rec.calls == 1);
    REQUIRE(rec.z == std::vector<int>{29, 29, 28});
    for (int k = 0; k < 9; ++k) {
      CHECK_THAT(rec.box[k], WithinRel(kRows[k], 1e-14));
    }
    for (size_t i = 0; i < pos_a.size(); ++i) {
      CHECK_THAT(rec.pos[i], WithinAbs(pos_a[i], 1e-14));
    }
  }
  CHECK_THAT(res.energy, WithinRel(-1.5 * units::EV_TO_HARTREE, 1e-15));
  REQUIRE(res.forces.size() == 9);
  for (size_t i = 0; i < 9; ++i) {
    CHECK_THAT(res.forces[i], WithinRel(0.1 * static_cast<double>(i + 1) /
                                            units::HARTREE_BOHR_TO_EV_ANGSTROM,
                                        1e-15));
  }
  const double volume = wire::cellVolume(kRows);
  for (int k = 0; k < 9; ++k) {
    CHECK_THAT(res.virial[k],
               WithinRel(-rec.sigma[k] * volume * units::EV_TO_HARTREE, 1e-14));
  }
  CHECK(res.extra == "{\"virial\": \"kernel\", \"virial_units\": \"hartree\"}");

  server.header("EXIT");
  REQUIRE(thread.join().empty());
  CHECK(driver.bead() == 3);
  CHECK(driver.initString() == "{\"bead\": 3}");
  CHECK(driver.evaluations() == 1);
  CHECK_FALSE(driver.connected());
}

TEST_CASE("IPIDriver connects over TCP and serves repeated steps",
          "[ipi_driver]") {
  auto server = FakeServer::tcp_any();
  Recorder rec;
  rgpot::IPIDriver driver(config_for(server.address(), {18, 18}),
                          rec.evaluator());
  DriverThread thread(driver);
  server.accept();
  REQUIRE(server.status() == "NEEDINIT");
  server.init(0, "");
  double h[9];
  ipi_cell(kRows, h);
  for (int step = 0; step < 5; ++step) {
    REQUIRE(server.status() == "READY");
    server.posdata(h, {0.0, 0.0, 0.0, 3.0 + step, 0.0, 0.0}, 2);
    const auto res = server.getforce();
    CHECK(res.forces.size() == 6);
  }
  server.header("EXIT");
  REQUIRE(thread.join().empty());
  CHECK(driver.evaluations() == 5);
}

TEST_CASE("IPIDriver maps unix:<name> onto the i-PI prefix", "[ipi_driver]") {
  const std::string name = "rgpot-ipid-" + std::to_string(::getpid());
  auto server = FakeServer::unix_at(std::string(wire::kUnixPrefix) + name);
  Recorder rec;
  rgpot::IPIDriver driver(config_for("unix:" + name, {1}), rec.evaluator());
  REQUIRE(driver.endpoint() == server.address());
  DriverThread thread(driver);
  server.accept();
  REQUIRE(server.status() == "NEEDINIT");
  server.header("EXIT");
  REQUIRE(thread.join().empty());

  auto cfg = config_for("unix:abc", {1});
  cfg.unix_prefix = "/var/run/ipi_";
  CHECK(rgpot::IPIDriver(cfg, rec.evaluator()).endpoint() ==
        "unix:/var/run/ipi_abc");
}

TEST_CASE("IPIDriver rejects a POSDATA atom count that differs from the "
          "species",
          "[ipi_driver]") {
  auto server = FakeServer::unix_at(temp_socket("natoms"));
  Recorder rec;
  rgpot::IPIDriver driver(config_for(server.address(), {29, 29, 29}),
                          rec.evaluator());
  DriverThread thread(driver);
  server.accept();
  double h[9];
  ipi_cell(kRows, h);
  server.posdata(h, {0, 0, 0, 1, 1, 1}, 2);
  const std::string err = thread.join();
  CHECK_THAT(err, ContainsSubstring("atom count 2 does not match the 3"));
  CHECK(server.peer_closed());
  CHECK_FALSE(driver.connected());
  CHECK(rec.calls == 0);
}

TEST_CASE("IPIDriver reports malformed messages", "[ipi_driver]") {
  Recorder rec;
  SECTION("unknown header") {
    auto server = FakeServer::unix_at(temp_socket("bad"));
    rgpot::IPIDriver driver(config_for(server.address(), {1}), rec.evaluator());
    DriverThread thread(driver);
    server.accept();
    server.header("BOGUS");
    CHECK_THAT(thread.join(),
               ContainsSubstring("unexpected server header 'BOGUS'"));
    CHECK(server.peer_closed());
  }
  SECTION("non-printable header bytes") {
    auto server = FakeServer::unix_at(temp_socket("bin"));
    rgpot::IPIDriver driver(config_for(server.address(), {1}), rec.evaluator());
    DriverThread thread(driver);
    server.accept();
    char junk[kHeader];
    std::memset(junk, 1, sizeof(junk));
    server.send(junk, sizeof(junk));
    CHECK_THAT(thread.join(), ContainsSubstring("'????????????'"));
  }
  SECTION("negative INIT length") {
    auto server = FakeServer::unix_at(temp_socket("init"));
    rgpot::IPIDriver driver(config_for(server.address(), {1}), rec.evaluator());
    DriverThread thread(driver);
    server.accept();
    server.header("INIT");
    server.value<int32_t>(0);
    server.value<int32_t>(-4);
    CHECK_THAT(thread.join(), ContainsSubstring("INIT string length"));
  }
  SECTION("GETFORCE before POSDATA") {
    auto server = FakeServer::unix_at(temp_socket("early"));
    rgpot::IPIDriver driver(config_for(server.address(), {1}), rec.evaluator());
    DriverThread thread(driver);
    server.accept();
    server.header("GETFORCE");
    CHECK_THAT(thread.join(), ContainsSubstring("GETFORCE before POSDATA"));
  }
  SECTION("non-finite positions") {
    auto server = FakeServer::unix_at(temp_socket("nan"));
    rgpot::IPIDriver driver(config_for(server.address(), {1}), rec.evaluator());
    DriverThread thread(driver);
    server.accept();
    double h[9];
    ipi_cell(kRows, h);
    server.posdata(h, {0.0, std::nan(""), 0.0}, 1);
    CHECK_THAT(thread.join(), ContainsSubstring("not finite"));
  }
  CHECK(rec.calls == 0);
}

TEST_CASE("IPIDriver reports a server that disconnects", "[ipi_driver]") {
  Recorder rec;
  SECTION("between messages") {
    auto server = FakeServer::unix_at(temp_socket("gone"));
    rgpot::IPIDriver driver(config_for(server.address(), {1}), rec.evaluator());
    DriverThread thread(driver);
    server.accept();
    REQUIRE(server.status() == "NEEDINIT");
    server.hangup();
    CHECK_THAT(thread.join(), ContainsSubstring("i-PI server disconnected"));
  }
  SECTION("inside a header") {
    auto server = FakeServer::unix_at(temp_socket("half"));
    rgpot::IPIDriver driver(config_for(server.address(), {1}), rec.evaluator());
    DriverThread thread(driver);
    server.accept();
    server.send("STAT", 4);
    server.hangup();
    CHECK_THAT(thread.join(), ContainsSubstring("i-PI server disconnected"));
  }
  SECTION("inside POSDATA") {
    auto server = FakeServer::unix_at(temp_socket("cut"));
    rgpot::IPIDriver driver(config_for(server.address(), {1, 1}),
                            rec.evaluator());
    DriverThread thread(driver);
    server.accept();
    server.header("POSDATA");
    double h[9]{};
    server.send(h, sizeof(h));
    server.hangup();
    CHECK_THAT(thread.join(), ContainsSubstring("i-PI server disconnected"));
  }
}

TEST_CASE("IPIDriver times out on a silent server and waits forever at 0",
          "[ipi_driver]") {
  Recorder rec;
  auto server = FakeServer::unix_at(temp_socket("quiet"));
  auto cfg = config_for(server.address(), {1});
  cfg.timeout_s = 0.2;
  rgpot::IPIDriver driver(cfg, rec.evaluator());
  const auto t0 = std::chrono::steady_clock::now();
  DriverThread thread(driver);
  server.accept();
  CHECK_THAT(thread.join(),
             ContainsSubstring("timed out reading from the i-PI server"));
  const double waited =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
          .count();
  CHECK(waited >= 0.2);
  CHECK(waited < 5.0);

  auto server2 = FakeServer::unix_at(temp_socket("slow"));
  auto cfg2 = config_for(server2.address(), {1});
  cfg2.timeout_s = 0.0;
  rgpot::IPIDriver patient(cfg2, rec.evaluator());
  DriverThread thread2(patient);
  server2.accept();
  std::this_thread::sleep_for(std::chrono::milliseconds(300));
  REQUIRE(server2.status() == "NEEDINIT");
  server2.header("EXIT");
  CHECK(thread2.join().empty());
}

TEST_CASE("IPIDriver retries connecting until the server listens",
          "[ipi_driver]") {
  Recorder rec;
  const std::string path = temp_socket("late");
  auto cfg = config_for("unix:" + path, {1});
  cfg.connect_timeout_s = 5.0;
  rgpot::IPIDriver driver(cfg, rec.evaluator());
  DriverThread thread(driver);
  std::this_thread::sleep_for(std::chrono::milliseconds(250));
  auto server = FakeServer::unix_at(path);
  server.accept();
  REQUIRE(server.status() == "NEEDINIT");
  server.header("EXIT");
  CHECK(thread.join().empty());

  auto none = config_for("unix:" + temp_socket("none"), {1});
  none.connect_timeout_s = 0.1;
  rgpot::IPIDriver lonely(none, rec.evaluator());
  REQUIRE_THROWS_WITH(lonely.connect(), ContainsSubstring("cannot connect"));
  REQUIRE_THROWS_WITH(lonely.step(), ContainsSubstring("not connected"));
}

TEST_CASE("IPIDriver validates its configuration", "[ipi_driver]") {
  Recorder rec;
  auto ev = rec.evaluator();
  auto bad = [&](rgpot::IPIDriverConfig cfg) {
    REQUIRE_THROWS_AS(rgpot::IPIDriver(cfg, ev), std::invalid_argument);
  };
  bad(config_for("tcp:127.0.0.1:31415", {}));
  bad(config_for("tcp:127.0.0.1:31415", {0}));
  bad(config_for("tcp:127.0.0.1:31415", {119}));
  bad(config_for("tcp:127.0.0.1:0", {1}));
  bad(config_for("tcp:127.0.0.1:70000", {1}));
  bad(config_for("udp:1", {1}));
  bad(config_for("unix:", {1}));
  bad(config_for("unix:a/b", {1}));
  bad(config_for("unix:/" + std::string(200, 'x'), {1}));
  auto cfg = config_for("tcp:31415", {1});
  cfg.timeout_s = -1.0;
  bad(cfg);
  cfg = config_for("tcp:31415", {1});
  cfg.connect_timeout_s = std::nan("");
  bad(cfg);
  cfg = config_for("tcp:31415", {1});
  cfg.cutoff = -1.0;
  bad(cfg);
  cfg = config_for("tcp:31415", {1});
  cfg.stress_step = 0.5;
  bad(cfg);
  REQUIRE_THROWS_AS(rgpot::IPIDriver(config_for("tcp:31415", {1}), {}),
                    std::invalid_argument);
  CHECK(rgpot::IPIDriver(config_for("tcp:31415", {1}), ev).endpoint() ==
        "tcp:127.0.0.1:31415");
}

TEST_CASE("IPIDriver pads a singular cell into an open box", "[ipi_driver]") {
  auto server = FakeServer::unix_at(temp_socket("open"));
  Recorder rec;
  auto cfg = config_for(server.address(), {18, 18});
  cfg.cutoff = 8.5;
  rgpot::IPIDriver driver(cfg, rec.evaluator());
  DriverThread thread(driver);
  server.accept();
  const double h[9]{};
  server.posdata(h, {0.0, 0.0, 0.0, 7.0, 0.0, 0.0}, 2);
  const auto res = server.getforce();
  server.header("EXIT");
  REQUIRE(thread.join().empty());
  std::lock_guard<std::mutex> lock(rec.mu);
  const double side =
      std::max(100.0, 2.0 * (7.0 * units::BOHR_TO_ANGSTROM + 8.5) + 10.0);
  CHECK_THAT(rec.box[0], WithinRel(side, 1e-14));
  CHECK_THAT(rec.box[4], WithinRel(side, 1e-14));
  CHECK_THAT(rec.box[8], WithinRel(side, 1e-14));
  CHECK(rec.box[1] == 0.0);
  CHECK_THAT(rec.pos[3] - rec.pos[0],
             WithinRel(7.0 * units::BOHR_TO_ANGSTROM, 1e-14));
  CHECK_THAT(res.virial[0], WithinRel(-rec.sigma[0] * side * side * side *
                                          units::EV_TO_HARTREE,
                                      1e-13));
}

TEST_CASE("IPIDriver tiles a cell thinner than twice the cutoff",
          "[ipi_driver]") {
  auto server = FakeServer::unix_at(temp_socket("tile"));
  Recorder rec;
  rec.energy = -6.0;
  auto cfg = config_for(server.address(), {18});
  cfg.cutoff = 5.0;
  rgpot::IPIDriver driver(cfg, rec.evaluator());
  DriverThread thread(driver);
  server.accept();
  // Widths 4, 12 and 6 Angstrom against twice the cutoff, 10: 3, 1 and 2
  // copies, so the kernel sees six atoms in a 12 Angstrom cube.
  const double rows[9] = {4.0, 0.0, 0.0, 0.0, 12.0, 0.0, 0.0, 0.0, 6.0};
  double h[9];
  ipi_cell(rows, h);
  server.posdata(h, {0.5, 0.5, 0.5}, 1);
  const auto res = server.getforce();
  server.header("EXIT");
  REQUIRE(thread.join().empty());
  std::lock_guard<std::mutex> lock(rec.mu);
  REQUIRE(rec.z.size() == 6);
  CHECK_THAT(rec.box[0], WithinRel(12.0, 1e-14));
  CHECK_THAT(rec.box[4], WithinRel(12.0, 1e-14));
  CHECK_THAT(rec.box[8], WithinRel(12.0, 1e-14));
  CHECK_THAT(res.energy, WithinRel(-1.0 * units::EV_TO_HARTREE, 1e-14));
  REQUIRE(res.forces.size() == 3);
  const double volume = 4.0 * 12.0 * 6.0;
  CHECK_THAT(res.virial[4],
             WithinRel(-rec.sigma[4] * volume * units::EV_TO_HARTREE, 1e-14));
}

TEST_CASE("IPIDriver takes a central-difference virial when asked",
          "[ipi_driver]") {
  // E = c V is a pressure -c: dE/de_ij = c V delta_ij, so W = -c V I.
  const double c = 0.0025;
  rgpot::IPIEvaluator volume_energy = [c](const rgpot::ForceInput &in,
                                          rgpot::ForceOut *out) {
    out->energy = c * wire::cellVolume(in.box);
  };
  double h[9];
  ipi_cell(kRows, h);
  const double volume = wire::cellVolume(kRows);

  SECTION("finite difference") {
    auto server = FakeServer::unix_at(temp_socket("fd"));
    auto cfg = config_for(server.address(), {29});
    cfg.stress_step = 1e-5;
    rgpot::IPIDriver driver(cfg, volume_energy);
    DriverThread thread(driver);
    server.accept();
    server.posdata(h, {0.1, 0.2, 0.3}, 1);
    const auto res = server.getforce();
    server.header("EXIT");
    REQUIRE(thread.join().empty());
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        const double want = i == j ? -c * volume * units::EV_TO_HARTREE : 0.0;
        CHECK_THAT(res.virial[3 * i + j], WithinAbs(want, 1e-10));
      }
    }
    CHECK_THAT(res.extra, ContainsSubstring("finite_difference"));
  }
  SECTION("no stress and no step") {
    auto server = FakeServer::unix_at(temp_socket("none"));
    auto cfg = config_for(server.address(), {29});
    rgpot::IPIDriver driver(cfg, volume_energy);
    DriverThread thread(driver);
    server.accept();
    server.posdata(h, {0.1, 0.2, 0.3}, 1);
    const auto res = server.getforce();
    server.header("EXIT");
    REQUIRE(thread.join().empty());
    for (const double w : res.virial) {
      CHECK(w == 0.0);
    }
    CHECK_THAT(res.extra, ContainsSubstring("\"none\""));
  }
  SECTION("empty extra string") {
    auto server = FakeServer::unix_at(temp_socket("quiet"));
    auto cfg = config_for(server.address(), {29});
    cfg.extras_json = false;
    rgpot::IPIDriver driver(cfg, volume_energy);
    DriverThread thread(driver);
    server.accept();
    server.posdata(h, {0.1, 0.2, 0.3}, 1);
    CHECK(server.getforce().extra.empty());
    server.header("EXIT");
    REQUIRE(thread.join().empty());
  }
}

namespace {

struct Direct {
  double energy = 0.0;
  std::vector<double> forces;
  std::array<double, 9> stress{};
};

template <typename Pot>
Direct direct(const Pot &pot, const std::vector<double> &pos,
              const std::vector<int> &z, const double rows[9]) {
  Direct d;
  d.forces.assign(pos.size(), 0.0);
  rgpot::ForceInput in{z.size(), pos.data(), z.data(), rows};
  rgpot::ForceOut out{d.forces.data(), 0.0, 0.0, {}, 0};
  pot.forceImpl(in, &out);
  REQUIRE(out.has_stress == 1);
  d.energy = out.energy;
  std::copy(out.stress, out.stress + 9, d.stress.begin());
  return d;
}

// fcc block with a small deterministic displacement per atom, sheared by
// the cell rows so the cell is not orthogonal.
void fcc_block(double a, int reps, std::vector<double> *pos, double rows[9]) {
  const double basis[4][3] = {
      {0, 0, 0}, {0.5, 0.5, 0}, {0.5, 0, 0.5}, {0, 0.5, 0.5}};
  const double shear = 0.12;
  rows[0] = reps * a;
  rows[1] = 0.0;
  rows[2] = 0.0;
  rows[3] = shear * reps * a;
  rows[4] = reps * a;
  rows[5] = 0.0;
  rows[6] = 0.0;
  rows[7] = -shear * reps * a;
  rows[8] = reps * a;
  pos->clear();
  int k = 0;
  for (int i = 0; i < reps; ++i) {
    for (int j = 0; j < reps; ++j) {
      for (int l = 0; l < reps; ++l) {
        for (const auto &b : basis) {
          const double s[3] = {(i + b[0]) / reps, (j + b[1]) / reps,
                               (l + b[2]) / reps};
          for (int c = 0; c < 3; ++c) {
            const double x =
                s[0] * rows[c] + s[1] * rows[3 + c] + s[2] * rows[6 + c];
            pos->push_back(x + 0.03 * std::sin(1.7 * k + c));
          }
          ++k;
        }
      }
    }
  }
}

template <typename Pot>
void round_trip(const Pot &pot, const std::string &address, double a, int reps,
                int z) {
  std::vector<double> pos;
  double rows[9];
  fcc_block(a, reps, &pos, rows);
  const std::vector<int> species(pos.size() / 3, z);
  const Direct want = direct(pot, pos, species, rows);

  auto server = std::make_unique<rgpot::IPIPot>(
      rgpot::IPIConfig{.address = address, .timeout_s = 30.0});
  auto cfg = config_for(server->endpoint(), species);
  rgpot::IPIDriver driver(
      cfg, [&pot](const rgpot::ForceInput &in, rgpot::ForceOut *out) {
        pot.forceImpl(in, out);
      });
  DriverThread thread(driver);

  for (int call = 0; call < 2; ++call) {
    std::vector<double> forces(pos.size(), 0.0);
    rgpot::ForceInput in{species.size(), pos.data(), species.data(), rows};
    rgpot::ForceOut out{forces.data(), 0.0, 0.0, {}, 0};
    server->forceImpl(in, &out);
    double stress[9];
    REQUIRE(server->copyStress(stress));

    CHECK_THAT(out.energy, WithinRel(want.energy, 1e-12));
    double fmax = 0.0;
    for (const double f : want.forces) {
      fmax = std::max(fmax, std::abs(f));
    }
    for (size_t i = 0; i < forces.size(); ++i) {
      CHECK_THAT(forces[i], WithinRel(want.forces[i], 1e-12) ||
                                WithinAbs(want.forces[i], 1e-12 * fmax));
    }
    double smax = 0.0;
    for (const double s : want.stress) {
      smax = std::max(smax, std::abs(s));
    }
    for (int k = 0; k < 9; ++k) {
      CHECK_THAT(stress[k], WithinRel(want.stress[k], 1e-12) ||
                                WithinAbs(want.stress[k], 1e-12 * smax));
    }
  }
  // IPIPot's destructor sends EXIT, which ends the driver thread.
  server.reset();
  REQUIRE(thread.join().empty());
  CHECK(driver.evaluations() == 2);
}

} // namespace

TEST_CASE("IPIPot and IPIDriver round trip matches direct evaluation",
          "[ipi_driver][roundtrip]") {
  rgpot::LJConfig lj;
  lj.u0 = 0.0104;
  lj.psi = 3.40;
  lj.cutoff = 8.5;
  const rgpot::LJPot ljpot(lj);
  rgpot::MorseConfig morse;
  morse.cutoff = 7.0;
  const rgpot::MorsePot morsepot(morse);
  rgpot::SuttonChenConfig sc = rgpot::SuttonChenConfig{};
  sc.a = 3.92;
  sc.epsilon = 0.0197;
  sc.cutoff = 7.0;
  sc.periodic = true;
  sc.stress = true;
  const rgpot::SuttonChenPot scpot(sc);

  SECTION("LJ over a UNIX socket") {
    round_trip(ljpot, "unix:" + temp_socket("rt-lj"), 5.26, 4, 18);
  }
  SECTION("LJ over TCP") { round_trip(ljpot, "tcp:0", 5.26, 4, 18); }
  SECTION("Morse") {
    round_trip(morsepot, "unix:" + temp_socket("rt-morse"), 4.0, 4, 78);
  }
  SECTION("Sutton-Chen") {
    round_trip(scpot, "unix:" + temp_socket("rt-sc"), 3.92, 4, 78);
  }
}
