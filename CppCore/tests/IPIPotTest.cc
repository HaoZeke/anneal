// MIT License
// Copyright 2023--present rgpot developers
//
// IPIPot against a mock i-PI driver: message sequence, unit conversion,
// triclinic cells, errors and timeouts. One case runs i-pi-py_driver -m
// dummy when that program is on PATH.

#include <algorithm>
#include <array>
#include <cmath>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <poll.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>

#include <catch2/catch_all.hpp>

#include "rgpot/IPIPot/IPIPot.hpp"
#include "rgpot/types/AtomMatrix.hpp"
#include "rgpot/units.hpp"

using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::WithinAbs;
using rgpot::types::AtomMatrix;

namespace {

constexpr int kHeader = 12;

double cell_volume(const double *box) {
  const double m00 = box[0];
  const double m01 = box[1];
  const double m02 = box[2];
  const double m10 = box[3];
  const double m11 = box[4];
  const double m12 = box[5];
  const double m20 = box[6];
  const double m21 = box[7];
  const double m22 = box[8];
  return std::abs(m00 * (m11 * m22 - m12 * m21) -
                   m01 * (m10 * m22 - m12 * m20) +
                   m02 * (m10 * m21 - m11 * m20));
}

std::array<char, kHeader> header(std::string_view text) {
  std::array<char, kHeader> out;
  out.fill(' ');
  std::copy_n(text.begin(), std::min(text.size(), static_cast<size_t>(kHeader)),
              out.begin());
  return out;
}

std::string trim_header(const char *bytes) {
  std::string text(bytes, bytes + kHeader);
  while (!text.empty() && text.back() == ' ') {
    text.pop_back();
  }
  return text;
}

void read_full(int fd, void *data, size_t n) {
  auto *bytes = static_cast<char *>(data);
  size_t got = 0;
  while (got < n) {
    pollfd pfd{};
    pfd.fd = fd;
    pfd.events = POLLIN;
    const int prc = ::poll(&pfd, 1, 5000);
    if (prc == 0) {
      throw std::runtime_error("mock timed out reading");
    }
    const ssize_t rc = ::recv(fd, bytes + got, n - got, 0);
    if (rc < 0) {
      if (errno == EINTR) {
        continue;
      }
      throw std::runtime_error("mock read failed");
    }
    if (rc == 0) {
      throw std::runtime_error("mock eof");
    }
    got += static_cast<size_t>(rc);
  }
}

void write_full(int fd, const void *data, size_t n) {
  const auto *bytes = static_cast<const char *>(data);
  size_t sent = 0;
  while (sent < n) {
#ifdef MSG_NOSIGNAL
    constexpr int kSendFlags = MSG_NOSIGNAL;
#else
    constexpr int kSendFlags = 0;
#endif
    const ssize_t rc = ::send(fd, bytes + sent, n - sent, kSendFlags);
    if (rc < 0) {
      if (errno == EINTR) {
        continue;
      }
      throw std::runtime_error("mock write failed");
    }
    if (rc == 0) {
      throw std::runtime_error("mock write closed");
    }
    sent += static_cast<size_t>(rc);
  }
}

int connect_endpoint(const std::string &endpoint) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (std::chrono::steady_clock::now() < deadline) {
    int fd = -1;
    if (endpoint.rfind("unix:", 0) == 0) {
      const std::string path = endpoint.substr(5);
      fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
      sockaddr_un addr{};
      addr.sun_family = AF_UNIX;
      std::snprintf(addr.sun_path, sizeof(addr.sun_path), "%s", path.c_str());
#ifdef __APPLE__
      addr.sun_len = sizeof(addr);
#endif
      if (fd >= 0 && ::connect(fd, reinterpret_cast<sockaddr *>(&addr),
                               sizeof(addr)) == 0) {
        return fd;
      }
    } else if (endpoint.rfind("tcp:", 0) == 0) {
      const std::string rest = endpoint.substr(4);
      const auto colon = rest.rfind(':');
      const std::string host = rest.substr(0, colon);
      const int port = std::stoi(rest.substr(colon + 1));
      fd = ::socket(AF_INET, SOCK_STREAM, 0);
      sockaddr_in addr{};
      addr.sin_family = AF_INET;
      addr.sin_port = htons(static_cast<uint16_t>(port));
      ::inet_pton(AF_INET, host.c_str(), &addr.sin_addr);
      if (fd >= 0 && ::connect(fd, reinterpret_cast<sockaddr *>(&addr),
                               sizeof(addr)) == 0) {
        return fd;
      }
    }
    if (fd >= 0) {
      ::close(fd);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  throw std::runtime_error("mock could not connect to " + endpoint);
}

enum class Behavior { Ok, BadHeader, Disconnect, Silent };

struct Reply {
  double energy = 0.25;
  std::array<double, 6> force{{0.1, -0.2, 0.3, -0.1, 0.2, -0.3}};
  std::array<double, 9> virial{
      {1.0, 0.5, 0.0, 0.5, 2.0, -0.25, 0.0, -0.25, 3.0}};
  std::string extra = "ok";
  int nat_delta = 0;
};

struct Capture {
  // The mock thread writes this while the test reads it. The socket
  // exchange does not synchronize those C++ objects, and a torn vector
  // on macOS is SIGBUS or SIGSEGV.
  std::mutex mu;
  std::vector<std::string> headers;
  std::vector<double> cell;
  std::vector<double> icell;
  std::vector<double> pos;
  int32_t nat = -1;
  std::string error;
};

struct CapView {
  std::vector<std::string> headers;
  std::vector<double> cell;
  std::vector<double> icell;
  std::vector<double> pos;
  int32_t nat = -1;
  std::string error;
};

CapView view(Capture &cap) {
  std::lock_guard<std::mutex> lock(cap.mu);
  return CapView{cap.headers, cap.cell, cap.icell, cap.pos, cap.nat, cap.error};
}

void run_mock(const std::string &endpoint, Behavior behavior,
              const Reply &reply, Capture *cap) {
  try {
    const int fd = connect_endpoint(endpoint);
    if (behavior == Behavior::Silent) {
      // Stay connected and unread so the server budget expires on the
      // reply, rather than observing a peer close.
      std::this_thread::sleep_for(std::chrono::milliseconds(1000));
      ::close(fd);
      return;
    }
    bool inited = false;
    while (true) {
      char raw[kHeader];
      read_full(fd, raw, kHeader);
      const std::string hdr = trim_header(raw);
      {
        std::lock_guard<std::mutex> lock(cap->mu);
        cap->headers.push_back(hdr);
      }
      if (hdr == "STATUS") {
        if (behavior == Behavior::BadHeader) {
          const auto bad = header("BROKEN");
          write_full(fd, bad.data(), bad.size());
          break;
        }
        const auto msg = header(!inited ? "NEEDINIT" : "READY");
        write_full(fd, msg.data(), msg.size());
        if (behavior == Behavior::Disconnect && inited) {
          break;
        }
      } else if (hdr == "INIT") {
        int32_t rid = 0;
        int32_t len = 0;
        read_full(fd, &rid, sizeof(rid));
        read_full(fd, &len, sizeof(len));
        if (len < 0 || len > (1 << 20)) {
          throw std::runtime_error("mock init length");
        }
        if (len > 0) {
          std::vector<char> sink(static_cast<size_t>(len));
          read_full(fd, sink.data(), sink.size());
        }
        inited = true;
      } else if (hdr == "POSDATA") {
        std::vector<double> cell(9);
        std::vector<double> icell(9);
        int32_t nat = 0;
        read_full(fd, cell.data(), 9 * sizeof(double));
        read_full(fd, icell.data(), 9 * sizeof(double));
        read_full(fd, &nat, sizeof(nat));
        if (nat < 0 || nat > 1'000'000) {
          throw std::runtime_error("mock atom count");
        }
        std::vector<double> pos(static_cast<size_t>(nat) * 3);
        read_full(fd, pos.data(), pos.size() * sizeof(double));
        std::lock_guard<std::mutex> lock(cap->mu);
        cap->cell = std::move(cell);
        cap->icell = std::move(icell);
        cap->nat = nat;
        cap->pos = std::move(pos);
      } else if (hdr == "GETFORCE") {
        const auto msg = header("FORCEREADY");
        write_full(fd, msg.data(), msg.size());
        write_full(fd, &reply.energy, sizeof(reply.energy));
        int32_t seen_nat = 0;
        {
          std::lock_guard<std::mutex> lock(cap->mu);
          seen_nat = cap->nat;
        }
        const int32_t nat = seen_nat + reply.nat_delta;
        write_full(fd, &nat, sizeof(nat));
        if (reply.nat_delta == 0) {
          write_full(fd, reply.force.data(),
                     reply.force.size() * sizeof(double));
          write_full(fd, reply.virial.data(),
                     reply.virial.size() * sizeof(double));
          const int32_t extra = static_cast<int32_t>(reply.extra.size());
          write_full(fd, &extra, sizeof(extra));
          if (extra > 0) {
            write_full(fd, reply.extra.data(), reply.extra.size());
          }
        }
      } else if (hdr == "EXIT") {
        break;
      } else {
        throw std::runtime_error("mock unexpected header " + hdr);
      }
    }
    ::close(fd);
  } catch (const std::exception &ex) {
    if (behavior == Behavior::Ok) {
      std::lock_guard<std::mutex> lock(cap->mu);
      cap->error = ex.what();
    }
  }
}

struct Join {
  std::thread &thread;
  ~Join() {
    if (thread.joinable()) {
      thread.join();
    }
  }
};

std::string tag() {
  static std::atomic<int> n{0};
  return std::to_string(::getpid()) + "x" + std::to_string(n.fetch_add(1));
}

AtomMatrix sample_positions() {
  return AtomMatrix{{1.5, -2.0, 0.25}, {0.0, 0.5, -1.0}};
}

std::array<double, 9> triclinic_box() {
  return {10.0, 1.0, 0.5, 0.25, 8.0, -0.4, -0.3, 0.2, 12.0};
}

void expect_cell(const std::vector<double> &cell, const double *box) {
  REQUIRE(cell.size() == 9);
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      const double want = box[col * 3 + row] * rgpot::units::ANGSTROM_TO_BOHR;
      REQUIRE_THAT(cell[static_cast<size_t>(row * 3 + col)],
                   WithinAbs(want, 1e-12));
    }
  }
}

void expect_inverse(const std::vector<double> &h,
                    const std::vector<double> &ih) {
  REQUIRE(ih.size() == 9);
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      double acc = 0.0;
      for (int k = 0; k < 3; ++k) {
        acc += h[static_cast<size_t>(row * 3 + k)] *
               ih[static_cast<size_t>(k * 3 + col)];
      }
      REQUIRE_THAT(acc, WithinAbs(row == col ? 1.0 : 0.0, 1e-10));
    }
  }
}

std::string driver_program() {
  if (const char *env = std::getenv("RGPOT_IPI_DRIVER")) {
    if (env[0] != '\0') {
      return env;
    }
  }
  for (const char *name : {"i-pi-py_driver", "ipi-py_driver"}) {
    const std::string cmd =
        std::string("command -v ") + name + " >/dev/null 2>&1";
    if (std::system(cmd.c_str()) == 0) {
      return name;
    }
  }
  return {};
}

} // namespace

TEST_CASE("IPIPot unix driver sequence, units and triclinic cell", "[ipi]") {
  rgpot::IPIConfig cfg;
  cfg.address = "unix:rgp" + tag();
  cfg.timeout_s = 2.0;
  // Capture outlives the join. The driver writes it until the thread returns.
  Capture cap;
  std::thread worker;
  Join join{worker};
  const Reply reply;
  {
    rgpot::IPIPot pot(cfg);
    REQUIRE(pot.getType() == rgpot::PotType::IPI);
    REQUIRE(pot.caps().stress);
    REQUIRE(pot.caps().reentrancy == rgpot::Reentrancy::PerInstance);
    REQUIRE(pot.endpoint().rfind("unix:/tmp/ipi_", 0) == 0);
    const std::string endpoint = pot.endpoint();
    worker = std::thread(run_mock, endpoint, Behavior::Ok, reply, &cap);

    const AtomMatrix positions = sample_positions();
    const auto box = triclinic_box();
    const std::vector<int> types{1, 1};
    AtomMatrix forces(2, 3);
    rgpot::ForceOut out{forces.data(), 0.0, 0.0};
    pot.forceImpl(
        {positions.rows(), positions.data(), types.data(), box.data()}, &out);

    {
      const CapView snap = view(cap);
      REQUIRE(snap.error.empty());
      REQUIRE(snap.nat == 2);
      expect_cell(snap.cell, box.data());
      expect_inverse(snap.cell, snap.icell);
      REQUIRE(snap.pos.size() == 6);
      for (size_t i = 0; i < 6; ++i) {
        REQUIRE_THAT(snap.pos[i],
                     WithinAbs(positions.data()[i] *
                                   rgpot::units::ANGSTROM_TO_BOHR,
                               1e-12));
      }
    }
    REQUIRE_THAT(out.energy,
                 WithinAbs(reply.energy * rgpot::units::HARTREE_TO_EV, 1e-12));
    for (size_t i = 0; i < 6; ++i) {
      REQUIRE_THAT(
          out.F[i],
          WithinAbs(reply.force[i] * rgpot::units::HARTREE_BOHR_TO_EV_ANGSTROM,
                    1e-9));
    }
    const double volume = cell_volume(box.data());
    const double scale = -rgpot::units::HARTREE_TO_EV / volume;
    REQUIRE(pot.hasStress());
    double copied[9];
    REQUIRE(pot.copyStress(copied));
    for (int i = 0; i < 9; ++i) {
      REQUIRE_THAT(
          copied[i],
          WithinAbs(reply.virial[static_cast<size_t>(i)] * scale, 1e-9));
    }

    {
      std::lock_guard<std::mutex> lock(cap.mu);
      cap.cell.clear();
      cap.pos.clear();
    }
    pot.forceImpl(
        {positions.rows(), positions.data(), types.data(), box.data()}, &out);
    REQUIRE(view(cap).error.empty());
    REQUIRE_THAT(out.energy,
                 WithinAbs(reply.energy * rgpot::units::HARTREE_TO_EV, 1e-12));
  }
  if (worker.joinable()) {
    worker.join();
  }
  const CapView done = view(cap);
  REQUIRE(done.error.empty());
  const std::vector<std::string> want = {"STATUS",  "INIT",     "STATUS",
                                         "POSDATA", "GETFORCE", "STATUS",
                                         "POSDATA", "GETFORCE", "EXIT"};
  REQUIRE(done.headers == want);
}

TEST_CASE("IPIPot tcp transport", "[ipi]") {
  rgpot::IPIConfig cfg;
  cfg.address = "tcp:127.0.0.1:0";
  cfg.timeout_s = 2.0;
  Capture cap;
  std::thread worker;
  Join join{worker};
  {
    rgpot::IPIPot pot(cfg);
    REQUIRE(pot.endpoint().rfind("tcp:127.0.0.1:", 0) == 0);
    REQUIRE(pot.endpoint() != "tcp:127.0.0.1:0");
    worker = std::thread(run_mock, pot.endpoint(), Behavior::Ok, Reply{}, &cap);
    const AtomMatrix positions = sample_positions();
    const std::array<double, 9> box{5.0, 0, 0, 0, 6.0, 0, 0, 0, 7.0};
    AtomMatrix forces(2, 3);
    rgpot::ForceOut out{forces.data(), 0.0, 0.0};
    pot.forceImpl({2, positions.data(), nullptr, box.data()}, &out);
    {
      const CapView snap = view(cap);
      REQUIRE(snap.error.empty());
      expect_cell(snap.cell, box.data());
      expect_inverse(snap.cell, snap.icell);
    }
    REQUIRE(pot.hasStress());
  }
  if (worker.joinable()) {
    worker.join();
  }
  const CapView done = view(cap);
  REQUIRE(done.error.empty());
  REQUIRE(done.headers.back() == "EXIT");
}

TEST_CASE("IPIPot rejects a bad address before listening", "[ipi]") {
  rgpot::IPIConfig cfg;
  cfg.address = "file:/tmp/nope";
  REQUIRE_THROWS_AS(rgpot::IPIPot(cfg), std::invalid_argument);
  cfg.address = "unix:";
  REQUIRE_THROWS_AS(rgpot::IPIPot(cfg), std::invalid_argument);
  cfg.address = "tcp:127.0.0.1:not-a-port";
  REQUIRE_THROWS_AS(rgpot::IPIPot(cfg), std::invalid_argument);
  cfg.address = "tcp:127.0.0.1:65536";
  REQUIRE_THROWS_AS(rgpot::IPIPot(cfg), std::invalid_argument);
  cfg.address = "unix:/tmp/ipi_ok";
  cfg.timeout_s = -1.0;
  REQUIRE_THROWS_AS(rgpot::IPIPot(cfg), std::invalid_argument);
}

TEST_CASE("IPIPot singular cell fails before a driver connects", "[ipi]") {
  rgpot::IPIConfig cfg;
  cfg.address = "unix:rgp" + tag();
  cfg.timeout_s = 0.2;
  rgpot::IPIPot pot(cfg);
  const AtomMatrix positions = sample_positions();
  const std::array<double, 9> box{};
  AtomMatrix forces(2, 3);
  rgpot::ForceOut out{forces.data(), 0.0, 0.0};
  REQUIRE_THROWS_WITH(
      pot.forceImpl({2, positions.data(), nullptr, box.data()}, &out),
      ContainsSubstring("singular"));
}

TEST_CASE("IPIPot times out when no driver connects", "[ipi]") {
  rgpot::IPIConfig cfg;
  cfg.address = "unix:rgp" + tag();
  cfg.timeout_s = 0.2;
  rgpot::IPIPot pot(cfg);
  const AtomMatrix positions{{0.0, 0.0, 0.0}};
  const std::array<double, 9> box{4, 0, 0, 0, 4, 0, 0, 0, 4};
  AtomMatrix forces(1, 3);
  rgpot::ForceOut out{forces.data(), 0.0, 0.0};
  REQUIRE_THROWS_WITH(
      pot.forceImpl({1, positions.data(), nullptr, box.data()}, &out),
      ContainsSubstring("timed out waiting"));
}

TEST_CASE("IPIPot times out when the driver stays silent", "[ipi]") {
  rgpot::IPIConfig cfg;
  cfg.address = "unix:rgp" + tag();
  cfg.timeout_s = 0.2;
  Capture cap;
  std::thread worker;
  Join join{worker};
  {
    rgpot::IPIPot pot(cfg);
    worker =
        std::thread(run_mock, pot.endpoint(), Behavior::Silent, Reply{}, &cap);
    const AtomMatrix positions{{0.0, 0.0, 0.0}};
    const std::array<double, 9> box{4, 0, 0, 0, 4, 0, 0, 0, 4};
    AtomMatrix forces(1, 3);
    rgpot::ForceOut out{forces.data(), 0.0, 0.0};
    REQUIRE_THROWS_WITH(
        pot.forceImpl({1, positions.data(), nullptr, box.data()}, &out),
        ContainsSubstring("timed out"));
  }
}

TEST_CASE("IPIPot reports an unexpected header and a broken atom count",
          "[ipi]") {
  rgpot::IPIConfig cfg;
  cfg.address = "unix:rgp" + tag();
  cfg.timeout_s = 2.0;
  Capture cap;
  std::thread worker;
  Join join{worker};
  {
    rgpot::IPIPot pot(cfg);
    worker = std::thread(run_mock, pot.endpoint(), Behavior::BadHeader, Reply{},
                         &cap);
    const AtomMatrix positions{{0.0, 0.0, 0.0}};
    const std::array<double, 9> box{4, 0, 0, 0, 4, 0, 0, 0, 4};
    AtomMatrix forces(1, 3);
    rgpot::ForceOut out{forces.data(), 0.0, 0.0};
    REQUIRE_THROWS_WITH(
        pot.forceImpl({1, positions.data(), nullptr, box.data()}, &out),
        ContainsSubstring("unexpected"));
  }
  worker.join();
  worker = std::thread();

  cfg.address = "unix:rgp" + tag();
  Capture cap2;
  std::thread worker2;
  Join join2{worker2};
  Reply mismatch;
  mismatch.nat_delta = 3;
  {
    rgpot::IPIPot pot(cfg);
    worker2 =
        std::thread(run_mock, pot.endpoint(), Behavior::Ok, mismatch, &cap2);
    const AtomMatrix positions{{1.0, 0.0, 0.0}};
    const std::array<double, 9> box{4, 0, 0, 0, 4, 0, 0, 0, 4};
    AtomMatrix forces(1, 3);
    rgpot::ForceOut out{forces.data(), 0.0, 0.0};
    REQUIRE_THROWS_WITH(
        pot.forceImpl({1, positions.data(), nullptr, box.data()}, &out),
        ContainsSubstring("atom count"));
  }
}

TEST_CASE("IPIPot accepts a new driver after the first disconnects", "[ipi]") {
  rgpot::IPIConfig cfg;
  cfg.address = "unix:rgp" + tag();
  cfg.timeout_s = 2.0;
  Capture first;
  Capture second;
  std::thread worker;
  Join join{worker};
  std::thread again;
  Join join_again{again};
  {
    rgpot::IPIPot pot(cfg);
    worker = std::thread(run_mock, pot.endpoint(), Behavior::Disconnect,
                         Reply{}, &first);
    const AtomMatrix positions{{0.0, 0.0, 0.0}};
    const std::array<double, 9> box{4, 0, 0, 0, 4, 0, 0, 0, 4};
    AtomMatrix forces(1, 3);
    rgpot::ForceOut out{forces.data(), 0.0, 0.0};
    REQUIRE_THROWS_WITH(
        pot.forceImpl({1, positions.data(), nullptr, box.data()}, &out),
        ContainsSubstring("disconnected"));
    worker.join();

    again =
        std::thread(run_mock, pot.endpoint(), Behavior::Ok, Reply{}, &second);
    const AtomMatrix two = sample_positions();
    AtomMatrix forces2(2, 3);
    rgpot::ForceOut out2{forces2.data(), 0.0, 0.0};
    const auto triclinic = triclinic_box();
    pot.forceImpl({2, two.data(), nullptr, triclinic.data()}, &out2);
    REQUIRE(view(second).error.empty());
    REQUIRE_THAT(out2.energy,
                 WithinAbs(0.25 * rgpot::units::HARTREE_TO_EV, 1e-12));
    REQUIRE(pot.hasStress());
  }
}

TEST_CASE("IPIPot optional i-PI dummy driver", "[ipi]") {
  const std::string program = driver_program();
  if (program.empty()) {
    SKIP("i-PI dummy driver is not installed");
  }
  const std::string name = "rgpot" + tag();
  rgpot::IPIConfig cfg;
  cfg.address = "unix:" + name;
  cfg.timeout_s = 10.0;
  rgpot::IPIPot pot(cfg);

  const pid_t pid = ::fork();
  REQUIRE(pid >= 0);
  if (pid == 0) {
    const int devnull = ::open("/dev/null", O_WRONLY);
    if (devnull >= 0) {
      ::dup2(devnull, STDOUT_FILENO);
      ::dup2(devnull, STDERR_FILENO);
    }
    ::execlp(program.c_str(), program.c_str(), "-u", "-a", name.c_str(), "-m",
             "dummy", static_cast<char *>(nullptr));
    _exit(127);
  }
  struct Reap {
    pid_t pid;
    ~Reap() {
      if (pid <= 0) {
        return;
      }
      ::kill(pid, SIGTERM);
      for (int i = 0; i < 20; ++i) {
        int status = 0;
        if (::waitpid(pid, &status, WNOHANG) == pid) {
          return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
      }
      ::kill(pid, SIGKILL);
      ::waitpid(pid, nullptr, 0);
    }
  } reap{pid};

  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  int early = 0;
  if (::waitpid(pid, &early, WNOHANG) == pid) {
    reap.pid = -1;
    SKIP("i-PI dummy driver exited before connecting");
  }

  const AtomMatrix positions{{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}};
  const std::array<double, 9> box{8, 0, 0, 0, 8, 0, 0, 0, 8};
  AtomMatrix forces(2, 3);
  rgpot::ForceOut out{forces.data(), 0.0, 0.0};
  pot.forceImpl({2, positions.data(), nullptr, box.data()}, &out);
  REQUIRE_THAT(out.energy, WithinAbs(0.0, 1e-12));
  for (int i = 0; i < 6; ++i) {
    REQUIRE_THAT(out.F[i], WithinAbs(0.0, 1e-12));
  }
  REQUIRE(pot.hasStress());
  double copied[9];
  REQUIRE(pot.copyStress(copied));
  for (int i = 0; i < 9; ++i) {
    REQUIRE_THAT(copied[i], WithinAbs(0.0, 1e-12));
  }
}

TEST_CASE("IPIPot parameter keys follow the address", "[ipi]") {
  rgpot::IPIConfig a;
  a.address = "unix:rgp" + tag();
  rgpot::IPIConfig b = a;
  b.address += "b";
  rgpot::IPIPot first(a);
  rgpot::IPIPot second(b);
  REQUIRE(first.paramsKey() != 0);
  REQUIRE(first.paramsKey() != second.paramsKey());
}
