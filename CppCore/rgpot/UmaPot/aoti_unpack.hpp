#pragma once
// Unpack an AOTInductor .pt2 into a persistent, content-addressed cache
// directory. torch's AOTIModelPackageLoader extracts into a mkdtemp
// directory under a hard-coded system temporary path and removes it only
// in its destructor, so a crashed or killed process leaks a full copy of
// the model there.
//
// Layout: <cache root>/aoti/<sha256 of the .pt2>.<model name>/ holds the
// archive's model directory as torch extracts it, plus kManifest. An
// unpack is built in a private sibling directory and renamed into place,
// so a reader sees either no directory or a complete one. Processes that
// race on one package each build a sibling; one rename wins and the
// others reuse its result.

#include "rgpot/UmaPot/aoti_execstack.hpp"
#include "rgpot/engine/Sha256.hpp"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <unordered_map>
#include <vector>

namespace rgpot {
namespace aoti_unpack {

namespace fs = std::filesystem;

/// Environment variable naming the cache root when the config has none.
inline constexpr const char *kCacheDirEnv = "RGPOT_CACHE_DIR";
/// Cache root below the run directory when nothing else names one.
inline constexpr const char *kRunDirCache = ".rgpot-cache";
/// Written last into an unpack; a directory without it is not an unpack.
inline constexpr const char *kManifest = "rgpot-unpack.manifest";
inline constexpr const char *kManifestMagic = "rgpot-aoti-unpack 1";
/// Leftover build or stale directories older than this are removed.
inline constexpr std::chrono::hours kLeftoverAge{1};

inline std::string sha256_file(const fs::path &path) {
  auto hex = engine::Sha256::of_file(path.string());
  if (hex.empty())
    throw std::runtime_error("UmaPot: cannot read AOTI package " +
                             path.string());
  return hex;
}

// -------------------------------------------------------------- cache root

struct CacheRoot {
  fs::path path;
  /// What chose the path: "config", kCacheDirEnv, "XDG_CACHE_HOME" or
  /// "run directory".
  std::string source;
};

/// The cache root, in order: the explicit config value, $RGPOT_CACHE_DIR,
/// $XDG_CACHE_HOME/rgpot (absolute values only, as the XDG base directory
/// specification requires), then .rgpot-cache in the working directory.
/// The result is absolute, so a later chdir does not move it.
inline CacheRoot resolve_cache_root(const std::string &configured) {
  auto absolute = [](const fs::path &p) {
    std::error_code ec;
    auto out = fs::absolute(p, ec);
    return ec ? p : out.lexically_normal();
  };
  if (!configured.empty())
    return {absolute(configured), "config"};
  if (const char *env = std::getenv(kCacheDirEnv); env && *env)
    return {absolute(env), kCacheDirEnv};
  if (const char *xdg = std::getenv("XDG_CACHE_HOME"); xdg && *xdg) {
    const fs::path base(xdg);
    if (base.is_absolute())
      return {absolute(base / "rgpot"), "XDG_CACHE_HOME"};
  }
  return {absolute(fs::current_path() / kRunDirCache), "run directory"};
}

// ------------------------------------------------------------- ZIP reading

struct ZipEntry {
  std::string name;
  uint16_t method{0};
  uint16_t flags{0};
  uint64_t csize{0};
  uint64_t usize{0};
  uint64_t local{0};
};

// The central directory owns sizes and offsets, including ZIP64 and
// streamed (data descriptor) records whose local header holds zeros.
class ZipReader {
public:
  explicit ZipReader(const fs::path &path) : m_path(path) {
    m_in.open(path, std::ios::binary);
    if (!m_in)
      throw std::runtime_error("UmaPot: cannot read AOTI package " +
                               path.string());
    m_in.seekg(0, std::ios::end);
    m_size = static_cast<uint64_t>(m_in.tellg());
    read_directory();
  }

  const std::vector<ZipEntry> &entries() const { return m_entries; }

  uint64_t data_offset(const ZipEntry &e) {
    const auto hdr = read(e.local, 30);
    if (aoti_execstack::rd32(hdr.data()) != aoti_execstack::kZipLocal)
      fail("invalid local header for " + e.name);
    const uint64_t off = e.local + 30 + aoti_execstack::rd16(hdr.data() + 26) +
                         aoti_execstack::rd16(hdr.data() + 28);
    require(off, e.csize);
    return off;
  }

  std::vector<uint8_t> read(uint64_t off, uint64_t len) {
    require(off, len);
    std::vector<uint8_t> buf(static_cast<std::size_t>(len));
    m_in.clear();
    m_in.seekg(static_cast<std::streamoff>(off));
    if (len && !m_in.read(reinterpret_cast<char *>(buf.data()),
                          static_cast<std::streamsize>(len)))
      fail("short read");
    return buf;
  }

  void copy(uint64_t off, uint64_t len, std::ostream &out) {
    require(off, len);
    m_in.clear();
    m_in.seekg(static_cast<std::streamoff>(off));
    std::vector<char> buf(1u << 20);
    while (len > 0) {
      const auto take =
          static_cast<std::size_t>(std::min<uint64_t>(len, buf.size()));
      if (!m_in.read(buf.data(), static_cast<std::streamsize>(take)))
        fail("short read");
      if (!out.write(buf.data(), static_cast<std::streamsize>(take)))
        throw std::runtime_error("UmaPot: short write while unpacking " +
                                 m_path.string());
      len -= take;
    }
  }

private:
  [[noreturn]] void fail(const std::string &what) const {
    throw std::runtime_error("UmaPot: AOTI package " + m_path.string() + ": " +
                             what);
  }

  void require(uint64_t off, uint64_t len) const {
    if (off > m_size || len > m_size - off)
      fail("truncated ZIP record");
  }

  void read_directory() {
    using aoti_execstack::rd16;
    using aoti_execstack::rd32;
    using aoti_execstack::rd64;
    if (m_size < 22)
      fail("no ZIP central directory");
    const uint64_t tail_len = std::min<uint64_t>(m_size, 65557);
    const uint64_t tail_off = m_size - tail_len;
    const auto tail = read(tail_off, tail_len);
    std::optional<uint64_t> eocd;
    for (uint64_t at = tail_len - 22;; --at) {
      if (rd32(tail.data() + at) == 0x06054b50u &&
          uint64_t(rd16(tail.data() + at + 20)) == tail_len - at - 22) {
        eocd = at;
        break;
      }
      if (at == 0)
        break;
    }
    if (!eocd)
      fail("no ZIP central directory");
    const uint8_t *e = tail.data() + *eocd;
    if (rd16(e + 4) != 0 || rd16(e + 6) != 0)
      fail("split ZIP archives are unsupported");
    uint64_t count = rd16(e + 10);
    uint64_t dir_size = rd32(e + 12);
    uint64_t dir_off = rd32(e + 16);
    if (count == 0xffffu || dir_size == 0xffffffffu ||
        dir_off == 0xffffffffu) {
      if (*eocd < 20 || rd32(e - 20) != 0x07064b50u)
        fail("missing ZIP64 locator");
      const auto z = read(rd64(e - 12), 56);
      if (rd32(z.data()) != 0x06064b50u || rd32(z.data() + 16) != 0 ||
          rd32(z.data() + 20) != 0)
        fail("invalid ZIP64 directory");
      count = rd64(z.data() + 32);
      dir_size = rd64(z.data() + 40);
      dir_off = rd64(z.data() + 48);
    }
    const auto dir = read(dir_off, dir_size);
    std::size_t pos = 0;
    m_entries.reserve(static_cast<std::size_t>(count));
    for (uint64_t i = 0; i < count; ++i) {
      if (dir.size() - pos < 46 ||
          rd32(dir.data() + pos) != aoti_execstack::kZipCentral)
        fail("invalid ZIP central entry");
      const uint8_t *c = dir.data() + pos;
      ZipEntry entry;
      entry.flags = rd16(c + 8);
      entry.method = rd16(c + 10);
      entry.csize = rd32(c + 20);
      entry.usize = rd32(c + 24);
      const uint16_t namelen = rd16(c + 28);
      const uint16_t extralen = rd16(c + 30);
      const uint16_t commentlen = rd16(c + 32);
      entry.local = rd32(c + 42);
      const std::size_t record = 46u + namelen + extralen + commentlen;
      if (record > dir.size() - pos)
        fail("truncated ZIP central entry");
      entry.name.assign(reinterpret_cast<const char *>(c + 46), namelen);
      if (entry.usize == 0xffffffffu || entry.csize == 0xffffffffu ||
          entry.local == 0xffffffffu) {
        const uint8_t *extra = c + 46 + namelen;
        const uint8_t *extra_end = extra + extralen;
        bool found = false;
        while (extra_end - extra >= 4) {
          const uint16_t kind = rd16(extra);
          const uint16_t len = rd16(extra + 2);
          extra += 4;
          if (len > extra_end - extra)
            fail("truncated ZIP extra field");
          if (kind == 1) {
            const uint8_t *field = extra;
            const uint8_t *field_end = extra + len;
            auto expand = [&](uint64_t &value) {
              if (value != 0xffffffffu)
                return;
              if (field_end - field < 8)
                fail("truncated ZIP64 size");
              value = rd64(field);
              field += 8;
            };
            expand(entry.usize);
            expand(entry.csize);
            expand(entry.local);
            found = true;
            break;
          }
          extra += len;
        }
        if (!found)
          fail("missing ZIP64 sizes");
      }
      m_entries.push_back(std::move(entry));
      pos += record;
    }
  }

  fs::path m_path;
  std::ifstream m_in;
  uint64_t m_size{0};
  std::vector<ZipEntry> m_entries;
};

// ----------------------------------------------------------------- unpack

struct Unpacked {
  fs::path dir;       ///< <root>/aoti/<sha256>.<model>
  fs::path model_dir; ///< torch's cubin / constants directory
  fs::path so;
  fs::path metadata_json; ///< empty when the package records none
  fs::path blob;          ///< empty when weights are inside the .so
  std::string sha256;
  bool reused{false}; ///< an earlier unpack was valid and kept
};

namespace detail {

inline bool starts_with(const std::string &s, const std::string &prefix) {
  return s.size() >= prefix.size() && s.compare(0, prefix.size(), prefix) == 0;
}

inline bool ends_with(const std::string &s, const std::string &suffix) {
  return s.size() >= suffix.size() &&
         s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// A relative path with no empty, "." or ".." component, so it cannot leave
// the unpack directory.
inline bool safe_relative(const std::string &rel) {
  if (rel.empty() || rel.front() == '/' || rel.find('\\') != std::string::npos ||
      rel.find(':') != std::string::npos)
    return false;
  std::size_t at = 0;
  while (at <= rel.size()) {
    const auto next = std::min(rel.find('/', at), rel.size());
    const auto part = rel.substr(at, next - at);
    if (part.empty() || part == "." || part == "..")
      return false;
    at = next + 1;
  }
  return true;
}

inline bool safe_model_name(const std::string &name) {
  if (name.empty() || name == "." || name == "..")
    return false;
  for (char c : name)
    if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '_' ||
          c == '-' || c == '.'))
      return false;
  return true;
}

// Hidden so the function and its local statics (counter, seed) are not
// emitted as STB_GNU_UNIQUE dynamic exports in librgpot.
#if (defined(__GNUC__) || defined(__clang__)) && !defined(_WIN32) &&            \
    !defined(__CYGWIN__)
__attribute__((visibility("hidden")))
#endif
inline std::string unique_token() {
  static std::atomic<uint64_t> counter{0};
  static const uint64_t seed = [] {
    std::random_device rd;
    return (uint64_t(rd()) << 32) ^ uint64_t(rd());
  }();
  const uint64_t mix =
      seed ^
      static_cast<uint64_t>(
          std::chrono::steady_clock::now().time_since_epoch().count()) ^
      (uint64_t(std::hash<std::thread::id>{}(std::this_thread::get_id()))
       << 1);
  std::ostringstream out;
  out << std::hex << mix << '-' << counter.fetch_add(1);
  return out.str();
}

struct Manifest {
  std::string sha256;
  std::string model;
  std::string so;
  std::string metadata;
  std::string blob;
  std::vector<std::pair<uint64_t, std::string>> files;
};

inline void write_manifest(const fs::path &path, const Manifest &m) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out << kManifestMagic << '\n'
      << "sha256 " << m.sha256 << '\n'
      << "model " << m.model << '\n'
      << "so " << m.so << '\n'
      << "metadata " << m.metadata << '\n'
      << "blob " << m.blob << '\n';
  for (const auto &[size, rel] : m.files)
    out << "file " << size << ' ' << rel << '\n';
  out.close();
  if (!out)
    throw std::runtime_error("UmaPot: cannot write " + path.string());
}

inline std::optional<Manifest> read_manifest(const fs::path &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in)
    return std::nullopt;
  std::string line;
  if (!std::getline(in, line) || line != kManifestMagic)
    return std::nullopt;
  Manifest m;
  auto field = [&](const char *key, std::string &out) {
    const std::string prefix = std::string(key) + ' ';
    if (!std::getline(in, line) || !starts_with(line, prefix))
      return false;
    out = line.substr(prefix.size());
    return true;
  };
  if (!field("sha256", m.sha256) || !field("model", m.model) ||
      !field("so", m.so) || !field("metadata", m.metadata) ||
      !field("blob", m.blob))
    return std::nullopt;
  while (std::getline(in, line)) {
    if (!starts_with(line, "file "))
      return std::nullopt;
    const auto space = line.find(' ', 5);
    if (space == std::string::npos)
      return std::nullopt;
    try {
      m.files.emplace_back(std::stoull(line.substr(5, space - 5)),
                           line.substr(space + 1));
    } catch (...) {
      return std::nullopt;
    }
  }
  return m;
}

// A complete unpack of this package and model, or nothing.
inline std::optional<Unpacked> load_valid(const fs::path &dir,
                                          const std::string &sha256,
                                          const std::string &model) {
  const auto m = read_manifest(dir / kManifest);
  if (!m || m->sha256 != sha256 || m->model != model || m->so.empty())
    return std::nullopt;
  for (const auto &[size, rel] : m->files) {
    if (!safe_relative(rel))
      return std::nullopt;
    std::error_code ec;
    const auto have = fs::file_size(dir / rel, ec);
    if (ec || have != size)
      return std::nullopt;
  }
  Unpacked u;
  u.dir = dir;
  u.so = dir / m->so;
  u.model_dir = u.so.parent_path();
  if (!m->metadata.empty())
    u.metadata_json = dir / m->metadata;
  if (!m->blob.empty())
    u.blob = dir / m->blob;
  u.sha256 = sha256;
  u.reused = true;
  return u;
}

// Extract the files torch's loader would extract for `model` into `out`
// and return the manifest. Shared libraries lose PT_GNU_STACK PF_X on the
// way, so dlopen does not need an executable stack.
inline Manifest extract(const fs::path &package, const fs::path &out,
                        const std::string &model, const std::string &sha256) {
  ZipReader zip(package);
  const auto &entries = zip.entries();
  if (entries.empty())
    throw std::runtime_error("UmaPot: AOTI package " + package.string() +
                             " has no files");
  // Archives from torch put every member under one top-level directory.
  std::string prefix;
  if (entries.size() >= 2) {
    const auto p0 = entries[0].name.substr(0, entries[0].name.find('/'));
    const auto p1 = entries[1].name.substr(0, entries[1].name.find('/'));
    if (!p0.empty() && p0 == p1 && p0 != entries[0].name)
      prefix = p0 + "/";
  }
  const std::string model_rel = prefix + "data/aotinductor/" + model;
  const std::string const_rel = prefix + "data/constants";

  Manifest m;
  m.sha256 = sha256;
  m.model = model;
  std::string cpp;
  std::vector<std::string> metadata_candidates;
  for (const auto &e : entries) {
    if (e.name.empty() || e.name.back() == '/')
      continue;
    std::string rel;
    if (starts_with(e.name, model_rel + "/")) {
      rel = e.name;
    } else if (starts_with(e.name, const_rel + "/")) {
      rel = model_rel + "/" + e.name.substr(e.name.find_last_of('/') + 1);
    } else {
      continue;
    }
    if (!safe_relative(rel))
      throw std::runtime_error("UmaPot: AOTI package " + package.string() +
                               " has an unsafe member name: " + e.name);
    if (e.method != 0 || (e.flags & 1u) || e.csize != e.usize)
      throw std::runtime_error(
          "UmaPot: AOTI package member " + e.name + " in " + package.string() +
          " is compressed or encrypted; torch writes stored members only");
    const fs::path dest = out / rel;
    fs::create_directories(dest.parent_path());
    const uint64_t data = zip.data_offset(e);
    {
      std::ofstream file(dest, std::ios::binary | std::ios::trunc);
      if (!file)
        throw std::runtime_error("UmaPot: cannot write " + dest.string());
      if (ends_with(rel, ".so")) {
        auto payload = zip.read(data, e.csize);
        aoti_execstack::clear_elf_gnu_stack(payload.data(), payload.size());
        file.write(reinterpret_cast<const char *>(payload.data()),
                   static_cast<std::streamsize>(payload.size()));
      } else {
        zip.copy(data, e.csize, file);
      }
      file.close();
      if (!file)
        throw std::runtime_error("UmaPot: short write of " + dest.string());
    }
    m.files.emplace_back(e.usize, rel);
    if (ends_with(rel, ".so"))
      m.so = rel;
    else if (ends_with(rel, ".cpp"))
      cpp = rel;
    else if (ends_with(rel, ".blob"))
      m.blob = rel;
    else if (ends_with(rel, "_metadata.json"))
      metadata_candidates.push_back(rel);
  }
  if (m.so.empty())
    throw std::runtime_error(
        "UmaPot: AOTI package " + package.string() +
        " has no compiled library for model '" + model +
        "'; export it with scripts/export_uma_aoti.py, which packages the .so");
  // torch reads <cpp stem>_metadata.json next to the generated source.
  for (const std::string &source : {cpp, m.so}) {
    if (source.empty())
      continue;
    const auto want = source.substr(0, source.find_last_of('.')) +
                      "_metadata.json";
    for (const auto &c : metadata_candidates)
      if (c == want)
        m.metadata = c;
    if (!m.metadata.empty())
      break;
  }
  if (m.metadata.empty() && metadata_candidates.size() == 1)
    m.metadata = metadata_candidates.front();
  write_manifest(out / kManifest, m);
  return m;
}

inline void remove_leftovers(const fs::path &aoti_dir,
                             const std::string &final_name) {
  const std::string mine = "." + final_name + ".";
  const auto now = fs::file_time_type::clock::now();
  std::error_code ec;
  for (fs::directory_iterator it(aoti_dir, ec), end; !ec && it != end;
       it.increment(ec)) {
    const auto name = it->path().filename().string();
    if (!starts_with(name, mine))
      continue;
    std::error_code tec;
    const auto when = fs::last_write_time(it->path(), tec);
    if (tec || now - when < kLeftoverAge)
      continue;
    std::error_code ignored;
    fs::remove_all(it->path(), ignored);
  }
}

} // namespace detail

/// Unpack `package` under `root` and return the unpacked paths. A valid
/// earlier unpack of the same bytes is reused; no file is ever written
/// outside `root`.
inline Unpacked unpack(const fs::path &package, const fs::path &root,
                       const std::string &model = "model") {
  if (!detail::safe_model_name(model))
    throw std::runtime_error("UmaPot: invalid AOTI model name '" + model + "'");
  const std::string sha = sha256_file(package);
  const fs::path aoti_dir = root / "aoti";
  const std::string final_name = sha + "." + model;
  const fs::path final_dir = aoti_dir / final_name;
  if (auto u = detail::load_valid(final_dir, sha, model)) {
    detail::remove_leftovers(aoti_dir, final_name);
    return *u;
  }

  std::error_code ec;
  fs::create_directories(aoti_dir, ec);
  if (ec && !fs::is_directory(aoti_dir))
    throw std::runtime_error("UmaPot: cannot create AOTI cache directory " +
                             aoti_dir.string() + ": " + ec.message());

  struct Build {
    fs::path path;
    ~Build() {
      if (path.empty())
        return;
      std::error_code ignored;
      fs::remove_all(path, ignored);
    }
  } build;
  for (int attempt = 0; build.path.empty(); ++attempt) {
    const fs::path candidate =
        aoti_dir / ("." + final_name + ".tmp-" + detail::unique_token());
    if (fs::create_directory(candidate, ec) && !ec) {
      build.path = candidate;
    } else if (attempt > 16) {
      throw std::runtime_error("UmaPot: cannot create an unpack directory in " +
                               aoti_dir.string() + ": " + ec.message());
    }
  }
  detail::extract(package, build.path, model, sha);

  auto delay = aoti_execstack::kPublicationInitialBackoff;
  const auto start = std::chrono::steady_clock::now();
  for (;;) {
    fs::rename(build.path, final_dir, ec);
    if (!ec) {
      build.path.clear();
      detail::remove_leftovers(aoti_dir, final_name);
      auto u = detail::load_valid(final_dir, sha, model);
      if (!u)
        throw std::runtime_error("UmaPot: unpack of " + package.string() +
                                 " in " + final_dir.string() +
                                 " vanished after publication");
      u->reused = false;
      return *u;
    }
    // Another process published first, or an invalid directory (an older
    // manifest format, a partial copy) holds the name.
    if (auto u = detail::load_valid(final_dir, sha, model))
      return *u;
    std::error_code exists_ec;
    if (fs::exists(final_dir, exists_ec)) {
      const fs::path stale =
          aoti_dir / ("." + final_name + ".stale-" + detail::unique_token());
      std::error_code moved;
      fs::rename(final_dir, stale, moved);
      if (!moved) {
        std::error_code ignored;
        fs::remove_all(stale, ignored);
      }
    }
    if (!aoti_execstack::publication_time_left(start))
      throw std::runtime_error("UmaPot: cannot publish unpack of " +
                               package.string() + " at " + final_dir.string() +
                               ": " + ec.message());
    aoti_execstack::publication_backoff(delay);
  }
}

// --------------------------------------------------------- metadata JSON

/// Parse the flat JSON object torch writes as <model>_metadata.json. String
/// values are unescaped; any other scalar is kept as its literal text.
inline std::unordered_map<std::string, std::string>
parse_flat_json(const std::string &text) {
  std::unordered_map<std::string, std::string> out;
  std::size_t i = 0;
  auto fail = [&](const char *what) -> void {
    throw std::runtime_error(std::string("UmaPot: AOTI metadata JSON: ") +
                             what + " at offset " + std::to_string(i));
  };
  auto ws = [&] {
    while (i < text.size() && std::isspace(static_cast<unsigned char>(text[i])))
      ++i;
  };
  auto utf8 = [](std::string &s, uint32_t cp) {
    if (cp < 0x80) {
      s.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
      s.push_back(static_cast<char>(0xc0 | (cp >> 6)));
      s.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
    } else if (cp < 0x10000) {
      s.push_back(static_cast<char>(0xe0 | (cp >> 12)));
      s.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3f)));
      s.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
    } else {
      s.push_back(static_cast<char>(0xf0 | (cp >> 18)));
      s.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3f)));
      s.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3f)));
      s.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
    }
  };
  auto hex4 = [&]() -> uint32_t {
    if (text.size() - i < 4)
      fail("short \\u escape");
    uint32_t v = 0;
    for (int k = 0; k < 4; ++k) {
      const char c = text[i++];
      v <<= 4;
      if (c >= '0' && c <= '9')
        v |= uint32_t(c - '0');
      else if (c >= 'a' && c <= 'f')
        v |= uint32_t(c - 'a' + 10);
      else if (c >= 'A' && c <= 'F')
        v |= uint32_t(c - 'A' + 10);
      else
        fail("bad \\u escape");
    }
    return v;
  };
  auto string = [&]() -> std::string {
    if (i >= text.size() || text[i] != '"')
      fail("expected a string");
    ++i;
    std::string s;
    while (i < text.size() && text[i] != '"') {
      char c = text[i++];
      if (c != '\\') {
        s.push_back(c);
        continue;
      }
      if (i >= text.size())
        fail("unterminated escape");
      c = text[i++];
      switch (c) {
      case '"': s.push_back('"'); break;
      case '\\': s.push_back('\\'); break;
      case '/': s.push_back('/'); break;
      case 'b': s.push_back('\b'); break;
      case 'f': s.push_back('\f'); break;
      case 'n': s.push_back('\n'); break;
      case 'r': s.push_back('\r'); break;
      case 't': s.push_back('\t'); break;
      case 'u': {
        uint32_t cp = hex4();
        if (cp >= 0xd800 && cp < 0xdc00 && text.compare(i, 2, "\\u") == 0) {
          i += 2;
          const uint32_t lo = hex4();
          cp = 0x10000 + ((cp - 0xd800) << 10) + (lo - 0xdc00);
        }
        utf8(s, cp);
        break;
      }
      default: fail("unknown escape");
      }
    }
    if (i >= text.size())
      fail("unterminated string");
    ++i;
    return s;
  };
  ws();
  if (i >= text.size() || text[i] != '{')
    fail("expected an object");
  ++i;
  ws();
  if (i < text.size() && text[i] == '}')
    return out;
  for (;;) {
    ws();
    std::string key = string();
    ws();
    if (i >= text.size() || text[i] != ':')
      fail("expected ':'");
    ++i;
    ws();
    std::string value;
    if (i < text.size() && text[i] == '"') {
      value = string();
    } else {
      const auto from = i;
      while (i < text.size() && text[i] != ',' && text[i] != '}' &&
             !std::isspace(static_cast<unsigned char>(text[i])))
        ++i;
      if (i == from)
        fail("expected a value");
      value = text.substr(from, i - from);
    }
    out[std::move(key)] = std::move(value);
    ws();
    if (i < text.size() && text[i] == ',') {
      ++i;
      continue;
    }
    if (i < text.size() && text[i] == '}')
      return out;
    fail("expected ',' or '}'");
  }
}

inline std::unordered_map<std::string, std::string>
read_metadata(const fs::path &path) {
  if (path.empty())
    return {};
  std::ifstream in(path, std::ios::binary);
  if (!in)
    throw std::runtime_error("UmaPot: cannot read AOTI metadata " +
                             path.string());
  std::ostringstream text;
  text << in.rdbuf();
  return parse_flat_json(text.str());
}

} // namespace aoti_unpack
} // namespace rgpot
