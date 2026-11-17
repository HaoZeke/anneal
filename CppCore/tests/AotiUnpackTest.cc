// MIT License
// aoti_unpack.hpp without torch: cache root resolution, the unpack layout,
// reuse, replacement of a broken unpack, concurrent unpacks from threads and
// processes, and that nothing lands in the system temporary directory. The
// packages are small stored ZIPs laid out as torch writes a .pt2.

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#ifndef _WIN32
#include <sys/wait.h>
#include <unistd.h>
#endif

#include <catch2/catch_all.hpp>

#include "rgpot/UmaPot/aoti_unpack.hpp"

namespace fs = std::filesystem;
namespace unpack = rgpot::aoti_unpack;
namespace execstack = rgpot::aoti_execstack;

namespace {

void set_env(const char *name, const std::string &value) {
#ifdef _WIN32
  _putenv_s(name, value.c_str());
#else
  setenv(name, value.c_str(), 1);
#endif
}

void unset_env(const char *name) {
#ifdef _WIN32
  _putenv_s(name, "");
#else
  unsetenv(name);
#endif
}

// Restores one environment variable when the scope ends.
class EnvGuard {
public:
  explicit EnvGuard(const char *name) : m_name(name) {
    if (const char *v = std::getenv(name)) {
      m_had = true;
      m_value = v;
    }
  }
  ~EnvGuard() {
    if (m_had)
      set_env(m_name, m_value);
    else
      unset_env(m_name);
  }
  EnvGuard(const EnvGuard &) = delete;
  EnvGuard &operator=(const EnvGuard &) = delete;

private:
  const char *m_name;
  bool m_had = false;
  std::string m_value;
};

// A scratch directory under the working directory (the meson build
// directory), never under the system temporary directory.
class Scratch {
public:
  Scratch() {
    path = fs::current_path() /
           ("aoti-unpack-test-" + unpack::detail::unique_token());
    fs::create_directories(path);
  }
  ~Scratch() {
    std::error_code ignored;
    fs::remove_all(path, ignored);
  }
  Scratch(const Scratch &) = delete;
  Scratch &operator=(const Scratch &) = delete;
  fs::path path;
};

std::vector<uint8_t> bytes(const std::string &s) {
  return {s.begin(), s.end()};
}

std::string sha256(const std::vector<uint8_t> &data) {
  rgpot::engine::Sha256 h;
  h.update(data.data(), data.size());
  return h.hex_digest();
}

std::vector<uint8_t> elf64_gnu_stack(uint32_t flags) {
  std::vector<uint8_t> elf(120, 0);
  elf[0] = 0x7f;
  elf[1] = 'E';
  elf[2] = 'L';
  elf[3] = 'F';
  elf[4] = 2;
  elf[5] = 1;
  elf[6] = 1;
  elf[16] = 3;
  elf[18] = 62;
  elf[20] = 1;
  elf[32] = 64;
  elf[52] = 64;
  elf[54] = 56;
  elf[56] = 1;
  execstack::wr32(elf.data() + 64, 0x6474e551u);
  execstack::wr32(elf.data() + 68, flags);
  return elf;
}

struct Member {
  std::string name;
  std::vector<uint8_t> data;
  uint16_t method = 0;
};

// Stored ZIP with one central directory record per member.
std::vector<uint8_t> zip(const std::vector<Member> &members) {
  std::vector<uint8_t> z;
  auto push16 = [&](std::vector<uint8_t> &v, uint16_t x) {
    v.push_back(uint8_t(x));
    v.push_back(uint8_t(x >> 8));
  };
  auto push32 = [&](std::vector<uint8_t> &v, uint32_t x) {
    for (int i = 0; i < 4; ++i)
      v.push_back(uint8_t(x >> (8 * i)));
  };
  std::vector<uint8_t> central;
  for (const auto &m : members) {
    const auto offset = static_cast<uint32_t>(z.size());
    const auto size = static_cast<uint32_t>(m.data.size());
    const auto namelen = static_cast<uint16_t>(m.name.size());
    push32(z, execstack::kZipLocal);
    push16(z, 20);
    push16(z, 0);
    push16(z, m.method);
    push16(z, 0);
    push16(z, 0);
    push32(z, 0);
    push32(z, size);
    push32(z, size);
    push16(z, namelen);
    push16(z, 0);
    z.insert(z.end(), m.name.begin(), m.name.end());
    z.insert(z.end(), m.data.begin(), m.data.end());
    push32(central, execstack::kZipCentral);
    push16(central, 20);
    push16(central, 20);
    push16(central, 0);
    push16(central, m.method);
    push16(central, 0);
    push16(central, 0);
    push32(central, 0);
    push32(central, size);
    push32(central, size);
    push16(central, namelen);
    push16(central, 0);
    push16(central, 0);
    push16(central, 0);
    push16(central, 0);
    push32(central, 0);
    push32(central, offset);
    central.insert(central.end(), m.name.begin(), m.name.end());
  }
  const auto central_off = static_cast<uint32_t>(z.size());
  z.insert(z.end(), central.begin(), central.end());
  push32(z, 0x06054b50u);
  push16(z, 0);
  push16(z, 0);
  push16(z, static_cast<uint16_t>(members.size()));
  push16(z, static_cast<uint16_t>(members.size()));
  push32(z, static_cast<uint32_t>(central.size()));
  push32(z, central_off);
  push16(z, 0);
  return z;
}

const char *kMetadata =
    R"({"AOTI_DEVICE_KEY": "cpu", "label": "fake \"line\" \u00e9", )"
    R"("counts": "{\"1\": 1}", "cutoff": "6.0"})";

std::vector<Member> fake_members(const std::string &weights) {
  const std::string m = "pkg/data/aotinductor/model/";
  return {
      {"pkg/archive_format", bytes("pt2")},
      {"pkg/version", bytes("0")},
      {m + "cabc.wrapper.cpp", bytes("// generated\n")},
      {m + "cabc.wrapper.so", elf64_gnu_stack(7)},
      {m + "cabc.wrapper_metadata.json", bytes(kMetadata)},
      {"pkg/data/constants/weights.bin", bytes(weights)},
      {"pkg/data/aotinductor/other/x.wrapper.so", elf64_gnu_stack(7)},
      {"pkg/extra/unrelated.txt", bytes("not unpacked")},
  };
}

fs::path write_package(const fs::path &dir, const std::string &name,
                       const std::vector<uint8_t> &data) {
  const auto path = dir / name;
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out.write(reinterpret_cast<const char *>(data.data()),
            static_cast<std::streamsize>(data.size()));
  return path;
}

std::vector<uint8_t> read_file(const fs::path &path) {
  std::ifstream in(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

std::set<std::string> listing(const fs::path &dir) {
  std::set<std::string> out;
  std::error_code ec;
  for (fs::directory_iterator it(dir, ec), end; !ec && it != end;
       it.increment(ec))
    out.insert(it->path().filename().string());
  return out;
}

// A temporary directory of the test's own. The system one is shared with
// every process of the user, and on macOS system agents add entries to
// $TMPDIR at any time. temp_directory_path reads TMPDIR, TMP, TEMP and
// TEMPDIR on POSIX; GetTempPath, which it calls on Windows, reads TMP and
// then TEMP.
class PrivateTemp {
public:
  explicit PrivateTemp(fs::path dir) : path(std::move(dir)) {
    fs::create_directories(path);
    for (const char *name : {"TMPDIR", "TMP", "TEMP", "TEMPDIR"})
      set_env(name, path.string());
  }
  PrivateTemp(const PrivateTemp &) = delete;
  PrivateTemp &operator=(const PrivateTemp &) = delete;
  const fs::path path;

private:
  EnvGuard m_tmpdir{"TMPDIR"};
  EnvGuard m_tmp{"TMP"};
  EnvGuard m_temp{"TEMP"};
  EnvGuard m_tempdir{"TEMPDIR"};
};

// Entries of /tmp missing from `before` that hold a file named like one
// under `unpacked`, or carry `sha` in their name: what an unpack leaves
// there, as torch's loader does in its mkdtemp directory. Other processes
// add entries to /tmp too, so a new name alone shows nothing.
std::vector<std::string> unpacks_in_tmp(const std::set<std::string> &before,
                                        const fs::path &unpacked,
                                        const std::string &sha) {
  std::set<std::string> files;
  std::error_code ec;
  for (fs::recursive_directory_iterator it(unpacked, ec), end; !ec && it != end;
       it.increment(ec)) {
    std::error_code kind;
    if (it->is_regular_file(kind))
      files.insert(it->path().filename().string());
  }
  const fs::path tmp("/tmp");
  std::vector<std::string> out;
  for (const auto &entry : listing(tmp)) {
    if (before.count(entry) != 0)
      continue;
    bool copy = files.count(entry) != 0 || entry.find(sha) != std::string::npos;
    std::error_code walk;
    for (fs::recursive_directory_iterator it(
             tmp / entry, fs::directory_options::skip_permission_denied, walk),
         end;
         !copy && !walk && it != end; it.increment(walk))
      copy = files.count(it->path().filename().string()) != 0;
    if (copy)
      out.push_back((tmp / entry).string());
  }
  return out;
}

} // namespace

TEST_CASE("the cache root follows config, RGPOT_CACHE_DIR, XDG, run directory",
          "[aoti][unpack]") {
  EnvGuard env(unpack::kCacheDirEnv);
  EnvGuard xdg("XDG_CACHE_HOME");
  Scratch s;
  const auto cfg = s.path / "from-config";
  const auto var = s.path / "from-env";
  const auto home = s.path / "xdg";
  set_env(unpack::kCacheDirEnv, var.string());
  set_env("XDG_CACHE_HOME", home.string());

  auto root = unpack::resolve_cache_root(cfg.string());
  REQUIRE(root.path == cfg);
  REQUIRE(root.source == "config");

  root = unpack::resolve_cache_root("");
  REQUIRE(root.path == var);
  REQUIRE(root.source == unpack::kCacheDirEnv);

  unset_env(unpack::kCacheDirEnv);
  root = unpack::resolve_cache_root("");
  REQUIRE(root.path == home / "rgpot");
  REQUIRE(root.source == "XDG_CACHE_HOME");

  // The XDG specification ignores a relative value.
  set_env("XDG_CACHE_HOME", "relative/cache");
  root = unpack::resolve_cache_root("");
  REQUIRE(root.path == fs::current_path() / unpack::kRunDirCache);
  REQUIRE(root.source == "run directory");

  unset_env("XDG_CACHE_HOME");
  root = unpack::resolve_cache_root("");
  REQUIRE(root.path == fs::current_path() / unpack::kRunDirCache);

  // A relative config value is made absolute against the working directory.
  root = unpack::resolve_cache_root("rel-cache");
  REQUIRE(root.path.is_absolute());
  REQUIRE(root.path == fs::current_path() / "rel-cache");
}

TEST_CASE("unpack lays out the model as torch does and reuses it",
          "[aoti][unpack]") {
  Scratch s;
  const auto data = zip(fake_members("w0"));
  const auto pkg = write_package(s.path, "fake.pt2", data);
  const auto root = s.path / "cache";

  const auto first = unpack::unpack(pkg, root);
  const auto sha = sha256(data);
  REQUIRE(first.sha256 == sha);
  REQUIRE_FALSE(first.reused);
  REQUIRE(first.dir == root / "aoti" / (sha + ".model"));
  const auto model = first.dir / "pkg/data/aotinductor/model";
  REQUIRE(first.model_dir == model);
  REQUIRE(first.so == model / "cabc.wrapper.so");
  REQUIRE(first.metadata_json == model / "cabc.wrapper_metadata.json");
  REQUIRE(first.blob.empty());
  REQUIRE(read_file(model / "weights.bin") == bytes("w0"));
  REQUIRE(read_file(model / "cabc.wrapper.cpp") == bytes("// generated\n"));
  REQUIRE_FALSE(fs::exists(first.dir / "pkg/data/aotinductor/other"));
  REQUIRE_FALSE(fs::exists(first.dir / "pkg/extra"));
  REQUIRE_FALSE(fs::exists(first.dir / "pkg/archive_format"));

  auto so = read_file(first.so);
  REQUIRE(so.size() == 120);
  REQUIRE_FALSE(execstack::elf_needs_gnu_stack_clear(so.data(), so.size()));
  REQUIRE(execstack::rd32(so.data() + 68) == 6u);

  const auto meta = unpack::read_metadata(first.metadata_json);
  REQUIRE(meta.at("AOTI_DEVICE_KEY") == "cpu");
  REQUIRE(meta.at("label") == "fake \"line\" \xc3\xa9");
  REQUIRE(meta.at("counts") == "{\"1\": 1}");
  REQUIRE(meta.at("cutoff") == "6.0");

  const auto when = fs::last_write_time(first.so);
  const auto again = unpack::unpack(pkg, root);
  REQUIRE(again.reused);
  REQUIRE(again.dir == first.dir);
  // libc++ file_time_type counts in __int128, which Catch2 cannot print.
  const bool untouched = fs::last_write_time(again.so) == when;
  REQUIRE(untouched);
  REQUIRE(listing(root / "aoti") == std::set<std::string>{sha + ".model"});

  // Other bytes are another model, in another directory.
  const auto other = write_package(s.path, "other.pt2", zip(fake_members("w1")));
  const auto second = unpack::unpack(other, root);
  REQUIRE(second.dir != first.dir);
  REQUIRE(read_file(second.model_dir / "weights.bin") == bytes("w1"));
  REQUIRE(listing(root / "aoti").size() == 2);
}

TEST_CASE("a broken unpack is replaced", "[aoti][unpack]") {
  Scratch s;
  const auto pkg = write_package(s.path, "fake.pt2", zip(fake_members("w0")));
  const auto root = s.path / "cache";
  const auto good = unpack::unpack(pkg, root);

  SECTION("truncated file") {
    std::ofstream(good.model_dir / "weights.bin",
                  std::ios::binary | std::ios::trunc);
  }
  SECTION("missing manifest") { fs::remove(good.dir / unpack::kManifest); }
  SECTION("partial copy without a manifest") {
    fs::remove_all(good.dir);
    fs::create_directories(good.dir / "pkg");
  }

  const auto fixed = unpack::unpack(pkg, root);
  REQUIRE_FALSE(fixed.reused);
  REQUIRE(fixed.dir == good.dir);
  REQUIRE(read_file(fixed.model_dir / "weights.bin") == bytes("w0"));
  REQUIRE(listing(root / "aoti").size() == 1);
}

TEST_CASE("an unpack keeps the ZIP64 and data descriptor sizes",
          "[aoti][unpack]") {
  Scratch s;
  const auto elf = elf64_gnu_stack(7);
  const std::string name = "pkg/data/aotinductor/model/m.wrapper.so";
  auto z = zip({{name, elf}, {"pkg/version", bytes("0")}});
  const std::size_t central = execstack::rd32(z.data() + z.size() - 6);

  SECTION("ZIP64 extra field in the central directory") {
    std::vector<uint8_t> extra(20, 0);
    extra[0] = 1;
    extra[2] = 16;
    execstack::wr32(extra.data() + 4, static_cast<uint32_t>(elf.size()));
    execstack::wr32(extra.data() + 12, static_cast<uint32_t>(elf.size()));
    execstack::wr32(z.data() + central + 20, 0xffffffffu);
    execstack::wr32(z.data() + central + 24, 0xffffffffu);
    z[central + 30] = 20;
    const std::size_t at = central + 46 + name.size();
    z.insert(z.begin() + static_cast<std::ptrdiff_t>(at), extra.begin(),
             extra.end());
    const auto size = execstack::rd32(z.data() + z.size() - 10);
    execstack::wr32(z.data() + z.size() - 10, size + 20);
  }
  SECTION("streamed record with zero local sizes") {
    z[6] = 8;
    z[central + 8] = 8;
    execstack::wr32(z.data() + 18, 0);
    execstack::wr32(z.data() + 22, 0);
  }
  const auto pkg = write_package(s.path, "edge.pt2", z);
  const auto u = unpack::unpack(pkg, s.path / "cache");
  const auto so = read_file(u.so);
  REQUIRE(so.size() == elf.size());
  REQUIRE(execstack::rd32(so.data() + 68) == 6u);
}

TEST_CASE("malformed packages are refused without a partial unpack",
          "[aoti][unpack]") {
  Scratch s;
  const auto root = s.path / "cache";
  const std::string m = "pkg/data/aotinductor/model/";

  SECTION("no compiled library") {
    const auto pkg = write_package(
        s.path, "noso.pt2",
        zip({{m + "a.wrapper.cpp", bytes("x")}, {"pkg/version", bytes("0")}}));
    REQUIRE_THROWS_WITH(unpack::unpack(pkg, root),
                        Catch::Matchers::ContainsSubstring("no compiled library"));
  }
  SECTION("compressed member") {
    const auto pkg = write_package(
        s.path, "deflate.pt2",
        zip({{m + "a.wrapper.so", elf64_gnu_stack(6), 8},
             {"pkg/version", bytes("0")}}));
    REQUIRE_THROWS_WITH(unpack::unpack(pkg, root),
                        Catch::Matchers::ContainsSubstring("compressed"));
  }
  SECTION("member escaping the unpack directory") {
    const auto pkg = write_package(
        s.path, "escape.pt2",
        zip({{m + "../../../../escape.so", elf64_gnu_stack(6)},
             {"pkg/version", bytes("0")}}));
    REQUIRE_THROWS_WITH(unpack::unpack(pkg, root),
                        Catch::Matchers::ContainsSubstring("unsafe member"));
    REQUIRE_FALSE(fs::exists(s.path / "escape.so"));
  }
  SECTION("not a ZIP") {
    const auto pkg = write_package(s.path, "junk.pt2", bytes("not a zip"));
    REQUIRE_THROWS(unpack::unpack(pkg, root));
  }
  SECTION("invalid model name") {
    const auto pkg = write_package(s.path, "fake.pt2", zip(fake_members("w")));
    REQUIRE_THROWS_WITH(unpack::unpack(pkg, root, "../model"),
                        Catch::Matchers::ContainsSubstring("model name"));
  }
  // Only the cache's aoti directory may exist, and it holds nothing.
  for (const auto &name : listing(root / "aoti"))
    FAIL("left behind: " << name);
}

TEST_CASE("concurrent unpacks publish one directory", "[aoti][unpack]") {
  Scratch s;
  std::string weights(1 << 20, 'w');
  const auto pkg = write_package(s.path, "big.pt2", zip(fake_members(weights)));
  const auto root = s.path / "cache";
  constexpr int threads = 8;
  std::vector<std::string> dirs(threads);
  std::vector<std::string> errors(threads);
  std::vector<std::thread> pool;

#ifndef _WIN32
  constexpr int processes = 4;
  std::vector<pid_t> children;
  for (int p = 0; p < processes; ++p) {
    const pid_t pid = fork();
    REQUIRE(pid >= 0);
    if (pid == 0) {
      int status = 0;
      try {
        const auto u = unpack::unpack(pkg, root);
        status = read_file(u.model_dir / "weights.bin").size() == weights.size()
                     ? 0
                     : 2;
      } catch (...) {
        status = 1;
      }
      _exit(status);
    }
    children.push_back(pid);
  }
#endif
  for (int t = 0; t < threads; ++t)
    pool.emplace_back([&, t] {
      try {
        dirs[t] = unpack::unpack(pkg, root).dir.string();
      } catch (const std::exception &e) {
        errors[t] = e.what();
      }
    });
  for (auto &th : pool)
    th.join();
#ifndef _WIN32
  for (const pid_t pid : children) {
    int status = 0;
    REQUIRE(waitpid(pid, &status, 0) == pid);
    REQUIRE(WIFEXITED(status));
    REQUIRE(WEXITSTATUS(status) == 0);
  }
#endif
  for (const auto &e : errors)
    REQUIRE(e.empty());
  REQUIRE(std::all_of(dirs.begin(), dirs.end(),
                      [&](const std::string &d) { return d == dirs.front(); }));
  const auto names = listing(root / "aoti");
  REQUIRE(names.size() == 1);
  REQUIRE(fs::path(dirs.front()).filename().string() == *names.begin());
  const auto u = unpack::unpack(pkg, root);
  REQUIRE(u.reused);
  REQUIRE(read_file(u.model_dir / "weights.bin").size() == weights.size());
}

TEST_CASE("stale build directories older than an hour are swept",
          "[aoti][unpack]") {
  Scratch s;
  const auto data = zip(fake_members("w0"));
  const auto pkg = write_package(s.path, "fake.pt2", data);
  const auto root = s.path / "cache";
  const auto sha = sha256(data);
  const auto aoti = root / "aoti";
  const auto old_tmp = aoti / ("." + sha + ".model.tmp-crashed");
  const auto new_tmp = aoti / ("." + sha + ".model.tmp-running");
  fs::create_directories(old_tmp / "pkg");
  fs::create_directories(new_tmp / "pkg");
  fs::last_write_time(old_tmp, fs::file_time_type::clock::now() -
                                   unpack::kLeftoverAge -
                                   std::chrono::minutes(5));
  unpack::unpack(pkg, root);
  REQUIRE_FALSE(fs::exists(old_tmp));
  REQUIRE(fs::exists(new_tmp));

  // A reused unpack sweeps too.
  fs::create_directories(old_tmp / "pkg");
  fs::last_write_time(old_tmp, fs::file_time_type::clock::now() -
                                   unpack::kLeftoverAge -
                                   std::chrono::minutes(5));
  REQUIRE(unpack::unpack(pkg, root).reused);
  REQUIRE_FALSE(fs::exists(old_tmp));
  REQUIRE(fs::exists(new_tmp));
}

TEST_CASE("RGPOT_CACHE_DIR unpacks without touching the temporary directory",
          "[aoti][unpack]") {
  EnvGuard env(unpack::kCacheDirEnv);
  EnvGuard xdg("XDG_CACHE_HOME");
  Scratch s;
  const PrivateTemp temp(s.path / "tmp");
  REQUIRE(fs::equivalent(fs::temp_directory_path(), temp.path));
  const auto tmp_before = listing("/tmp");
  const auto data = zip(fake_members("w0"));
  const auto pkg = write_package(s.path, "fake.pt2", data);
  set_env(unpack::kCacheDirEnv, (s.path / "env-cache").string());
  unset_env("XDG_CACHE_HOME");
  const auto root = unpack::resolve_cache_root("");
  const auto u = unpack::unpack(pkg, root.path);
  REQUIRE(u.dir.string().rfind((s.path / "env-cache").string(), 0) == 0);
  const auto again = unpack::unpack(pkg, unpack::resolve_cache_root("").path);
  REQUIRE(again.reused);
  for (const auto &name : listing(temp.path))
    FAIL("new entry in the temporary directory: " << name);
  for (const auto &path : unpacks_in_tmp(tmp_before, u.dir, sha256(data)))
    FAIL("unpacked files in " << path);
}
