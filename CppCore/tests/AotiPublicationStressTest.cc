// SPDX-License-Identifier: MIT
#include "rgpot/UmaPot/aoti_execstack.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace {

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

// Eight writers each publish eight 4 MiB payloads while readers sample the
// destination. A holder keeps the file it opened before the round open until
// the writers finish, as a loader keeps a package: MoveFileExW refuses every
// replace of a destination with an open handle, so on Windows each writer
// depends on the POSIX-semantics replace. Each byte of a payload is
// 1 + writer * kIterations + iteration, so every sample names the publication
// it came from; the seed is 0. RGPOT_AOTI_STRESS_ROUNDS and
// RGPOT_AOTI_STRESS_READERS raise the rounds and the sampling readers.
constexpr int kWriters = 8;
constexpr int kIterations = 8;
constexpr std::size_t kSize = 4 * 1024 * 1024;
constexpr uint8_t kSeed = 0;
constexpr std::size_t kReportedFailures = 20;

int setting(const char *name, int fallback) {
  const char *text = std::getenv(name);
  const int value = text ? std::atoi(text) : 0;
  return value > 0 ? value : fallback;
}

std::string millis(Clock::duration d) {
  std::ostringstream out;
  out.setf(std::ios::fixed);
  out.precision(1);
  out << std::chrono::duration<double, std::milli>(d).count();
  return out.str();
}

bool known(uint8_t value) { return value <= kWriters * kIterations; }

std::string publication(uint8_t value) {
  if (value == kSeed)
    return "the seed";
  if (!known(value))
    return "unknown byte " + std::to_string(value);
  return "writer " + std::to_string((value - 1) / kIterations) +
         " iteration " + std::to_string((value - 1) % kIterations);
}

std::string tag(uint8_t value) {
  return "w" + std::to_string((value - 1) / kIterations) + "i" +
         std::to_string((value - 1) % kIterations);
}

// Empty for one complete publication, otherwise what is wrong with data.
std::string torn(const std::vector<uint8_t> &data) {
  if (data.size() != kSize)
    return std::to_string(data.size()) + " bytes, not " + std::to_string(kSize);
  const auto odd = std::find_if(data.begin(), data.end(), [&](uint8_t value) {
    return value != data.front();
  });
  if (odd != data.end())
    return "byte " + std::to_string(odd - data.begin()) + " is " +
           std::to_string(*odd) + " but byte 0 is " +
           std::to_string(data.front());
  if (!known(data.front()))
    return publication(data.front());
  return {};
}

// Opens the destination with the sharing read_all uses and keeps it open.
class Holder {
public:
  explicit Holder(const fs::path &path) {
#ifdef _WIN32
    m_handle =
        CreateFileW(path.c_str(), GENERIC_READ,
                    FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                    nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
#else
    m_in.open(path, std::ios::binary);
#endif
  }
  ~Holder() {
#ifdef _WIN32
    if (m_handle != INVALID_HANDLE_VALUE)
      CloseHandle(m_handle);
#endif
  }
  Holder(const Holder &) = delete;
  Holder &operator=(const Holder &) = delete;

  bool is_open() const {
#ifdef _WIN32
    return m_handle != INVALID_HANDLE_VALUE;
#else
    return m_in.is_open();
#endif
  }

  // The whole file the handle refers to, from its first byte.
  std::vector<uint8_t> read() {
#ifdef _WIN32
    LARGE_INTEGER size{};
    LARGE_INTEGER origin{};
    if (!GetFileSizeEx(m_handle, &size) ||
        !SetFilePointerEx(m_handle, origin, nullptr, FILE_BEGIN))
      return {};
    std::vector<uint8_t> data(static_cast<std::size_t>(size.QuadPart));
    std::size_t got = 0;
    while (got < data.size()) {
      DWORD n = 0;
      const auto chunk = static_cast<DWORD>(
          std::min<std::size_t>(data.size() - got, std::size_t{1} << 20));
      if (!ReadFile(m_handle, data.data() + got, chunk, &n, nullptr) || n == 0)
        break;
      got += n;
    }
    data.resize(got);
    return data;
#else
    m_in.clear();
    m_in.seekg(0, std::ios::end);
    std::vector<uint8_t> data(static_cast<std::size_t>(m_in.tellg()));
    m_in.seekg(0);
    m_in.read(reinterpret_cast<char *>(data.data()),
              static_cast<std::streamsize>(data.size()));
    data.resize(static_cast<std::size_t>(m_in.gcount()));
    return data;
#endif
  }

private:
#ifdef _WIN32
  HANDLE m_handle = INVALID_HANDLE_VALUE;
#else
  std::ifstream m_in;
#endif
};

class Failures {
public:
  explicit Failures(int round) : m_round(round) {}
  void add(const std::string &what) {
    std::lock_guard<std::mutex> lock(m_mu);
    m_lines.push_back("round " + std::to_string(m_round) + ": " + what);
  }
  std::size_t size() {
    std::lock_guard<std::mutex> lock(m_mu);
    return m_lines.size();
  }
  void report() {
    std::lock_guard<std::mutex> lock(m_mu);
    for (std::size_t i = 0; i < m_lines.size() && i < kReportedFailures; ++i)
      std::cerr << m_lines[i] << '\n';
    if (m_lines.size() > kReportedFailures)
      std::cerr << "round " << m_round << ": "
                << m_lines.size() - kReportedFailures << " more failures\n";
    std::cerr.flush();
  }

private:
  int m_round;
  std::mutex m_mu;
  std::vector<std::string> m_lines;
};

void run_round(const fs::path &target, int readers, Failures &failures,
               std::ostream &stats) {
  std::atomic<int> finished{0};
  std::barrier start(kWriters + readers);
  std::vector<int> published(kWriters, 0);
  std::vector<char> stopped(kWriters, 0);
  std::vector<Clock::duration> slowest_write(kWriters, Clock::duration{});
  std::vector<Clock::duration> slowest_read(readers, Clock::duration{});
  std::vector<int> reads(readers, 0);
  std::vector<std::thread> threads;
  threads.reserve(static_cast<std::size_t>(kWriters + readers));

  Holder holder(target);
  if (!holder.is_open())
    failures.add("cannot hold " + target.string() + " open");

  const auto begun = Clock::now();
  auto at = [&](Clock::time_point when) {
    return "at +" + millis(when - begun) + " ms";
  };

  for (int writer = 0; writer < kWriters; ++writer) {
    threads.emplace_back([&, writer] {
      std::vector<uint8_t> payload(kSize);
      start.arrive_and_wait();
      for (int iteration = 0; iteration < kIterations; ++iteration) {
        std::fill(payload.begin(), payload.end(),
                  static_cast<uint8_t>(1 + writer * kIterations + iteration));
        const auto t0 = Clock::now();
        try {
          rgpot::aoti_execstack::write_all(target.string(), payload);
        } catch (const std::exception &ex) {
          failures.add("writer " + std::to_string(writer) + " iteration " +
                       std::to_string(iteration) + " began " + at(t0) +
                       " and gave up after " + millis(Clock::now() - t0) +
                       " ms: " + ex.what());
          stopped[writer] = 1;
          break;
        }
        slowest_write[writer] =
            std::max(slowest_write[writer], Clock::now() - t0);
        ++published[writer];
      }
      ++finished;
    });
  }

  for (int reader = 0; reader < readers; ++reader) {
    threads.emplace_back([&, reader] {
      std::array<int, kWriters> newest;
      newest.fill(-1);
      bool saw_publication = false;
      start.arrive_and_wait();
      for (int attempt = 0; finished.load() < kWriters; ++attempt) {
        const auto t0 = Clock::now();
        const std::string who = "reader " + std::to_string(reader) +
                                " read " + std::to_string(attempt) + " " +
                                at(t0);
        std::vector<uint8_t> data;
        try {
          data = rgpot::aoti_execstack::read_all(target.string());
        } catch (const std::exception &ex) {
          failures.add(who + " gave up after " + millis(Clock::now() - t0) +
                       " ms: " + ex.what());
          continue;
        }
        ++reads[reader];
        slowest_read[reader] =
            std::max(slowest_read[reader], Clock::now() - t0);
        if (const auto what = torn(data); !what.empty()) {
          failures.add(who + " was torn: " + what);
          continue;
        }
        const uint8_t value = data.front();
        if (value == kSeed) {
          if (saw_publication)
            failures.add(who + " returned the seed after a publication");
          continue;
        }
        saw_publication = true;
        const int writer = (value - 1) / kIterations;
        const int iteration = (value - 1) % kIterations;
        if (iteration < newest[writer])
          failures.add(who + " returned " + publication(value) +
                       " after iteration " + std::to_string(newest[writer]) +
                       " of that writer");
        newest[writer] = std::max(newest[writer], iteration);
      }
    });
  }

  for (auto &thread : threads)
    thread.join();

  if (holder.is_open()) {
    const auto held = holder.read();
    if (const auto what = torn(held); !what.empty())
      failures.add("the held handle was torn: " + what);
    else if (held.front() != kSeed)
      failures.add("the held handle returned " + publication(held.front()) +
                   ", not the seed it opened");
  }

  int total_reads = 0;
  for (int count : reads)
    total_reads += count;
  if (total_reads == 0)
    failures.add("no reader completed a read");

  // The last replace is the last one of its writer.
  std::string final_tag = "-";
  if (std::none_of(stopped.begin(), stopped.end(), [](char s) { return s; })) {
    try {
      const auto data = rgpot::aoti_execstack::read_all(target.string());
      if (const auto what = torn(data); !what.empty())
        failures.add("the destination was torn after the round: " + what);
      else if (data.front() == kSeed ||
               (data.front() - 1) % kIterations != kIterations - 1)
        failures.add("the destination holds " + publication(data.front()) +
                     " after the round, not a last iteration");
      else
        final_tag = tag(data.front());
    } catch (const std::exception &ex) {
      failures.add(std::string("the read after the round gave up: ") +
                   ex.what());
    }
  }

  std::error_code ec;
  for (fs::directory_iterator it(target.parent_path(), ec), end;
       !ec && it != end; it.increment(ec)) {
    if (it->path() != target)
      failures.add("a publication left " + it->path().filename().string());
  }

  int publications = 0;
  for (int count : published)
    publications += count;
  const auto write_max =
      *std::max_element(slowest_write.begin(), slowest_write.end());
  const auto read_max =
      readers > 0 ? *std::max_element(slowest_read.begin(), slowest_read.end())
                  : Clock::duration{};
  stats << "writers=" << kWriters << " readers=" << readers
        << " publications=" << publications << '/' << kWriters * kIterations
        << " reads=" << total_reads << " max_write_ms=" << millis(write_max)
        << " max_read_ms=" << millis(read_max) << " final=" << final_tag
        << " round_ms=" << millis(Clock::now() - begun);
}

#ifdef _WIN32
// Delete-pending is the Windows state in which exists() throws
// "Access is denied" and an open without FILE_SHARE_DELETE fails.
bool read_while_delete_pending(const fs::path &target,
                               const std::vector<uint8_t> &payload,
                               std::string &error) {
  HANDLE handle = CreateFileW(
      target.c_str(), GENERIC_READ,
      FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
      OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_DELETE_ON_CLOSE,
      nullptr);
  if (handle == INVALID_HANDLE_VALUE) {
    error = "cannot hold a delete-pending publication";
    return false;
  }
  bool ok = true;
  try {
    const auto data = rgpot::aoti_execstack::read_all(target.string());
    if (data != payload) {
      error = "delete-pending read was not the published payload";
      ok = false;
    }
  } catch (const std::exception &ex) {
    error = ex.what();
    ok = false;
  }
  CloseHandle(handle);
  return ok;
}
#endif

} // namespace

int main() {
  const int rounds = setting("RGPOT_AOTI_STRESS_ROUNDS", 30);
  const int readers = setting("RGPOT_AOTI_STRESS_READERS", 1);
  const auto root =
      fs::temp_directory_path() /
      ("rgpot-publication-stress-" +
       std::to_string(Clock::now().time_since_epoch().count()));
  fs::create_directories(root);
  const auto target = root / "model.pt2";

  const std::vector<uint8_t> seed(kSize, kSeed);
  int failed_rounds = 0;
  for (int round = 1; round <= rounds; ++round) {
    Failures failures(round);
    std::ostringstream stats;
    bool seeded = true;
    try {
      rgpot::aoti_execstack::write_all(target.string(), seed);
#ifdef _WIN32
      std::string error;
      if (!read_while_delete_pending(target, seed, error))
        failures.add(error);
      // Closing the delete-on-close handle removes the file.
      rgpot::aoti_execstack::write_all(target.string(), seed);
#endif
    } catch (const std::exception &ex) {
      failures.add(std::string("publishing the seed gave up: ") + ex.what());
      seeded = false;
    }
    if (seeded)
      run_round(target, readers, failures, stats);
    else
      stats << "seed=unpublished";
    const std::size_t count = failures.size();
    if (count != 0) {
      ++failed_rounds;
      failures.report();
    }
    std::cout << "round=" << round << ' ' << stats.str()
              << " failures=" << count << '\n';
    std::cout.flush();
  }
  std::cout << "rounds=" << rounds << " failed_rounds=" << failed_rounds
            << '\n';
  std::error_code ignored;
  fs::remove_all(root, ignored);
  return failed_rounds == 0 ? 0 : 1;
}
