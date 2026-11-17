#pragma once
// Clear PT_GNU_STACK PF_X in AOTI shared libraries so dlopen of
// wrapper.so does not need an executable stack (Elja inductor output
// on a kernel that refuses mprotect(PROT_EXEC) on the stack).
// aoti_unpack.hpp applies it to every .so it unpacks.

#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <system_error>
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
#ifdef small
#undef small
#endif
#ifdef interface
#undef interface
#endif
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif
#endif

namespace rgpot {
namespace aoti_execstack {

inline constexpr uint32_t kPtGnuStack = 0x6474e551u;
inline constexpr uint32_t kPfX = 1u;
inline constexpr uint32_t kZipLocal = 0x04034b50u;
inline constexpr uint32_t kZipCentral = 0x02014b50u;

inline uint16_t rd16(const uint8_t *p) {
  return static_cast<uint16_t>(p[0] | (uint16_t(p[1]) << 8));
}

inline uint32_t rd32(const uint8_t *p) {
  return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) |
         (uint32_t(p[3]) << 24);
}

inline uint64_t rd64(const uint8_t *p) {
  return uint64_t(rd32(p)) | (uint64_t(rd32(p + 4)) << 32);
}

inline void wr32(uint8_t *p, uint32_t v) {
  p[0] = static_cast<uint8_t>(v);
  p[1] = static_cast<uint8_t>(v >> 8);
  p[2] = static_cast<uint8_t>(v >> 16);
  p[3] = static_cast<uint8_t>(v >> 24);
}

// Returns true if a GNU_STACK PF_X bit was cleared.
inline bool clear_elf_gnu_stack(uint8_t *data, size_t n) {
  if (n < 64 || data[0] != 0x7f || data[1] != 'E' || data[2] != 'L' ||
      data[3] != 'F')
    return false;
  if (data[4] != 2 || data[5] != 1)
    return false;
  const uint64_t phoff =
      uint64_t(rd32(data + 32)) | (uint64_t(rd32(data + 36)) << 32);
  const uint16_t phentsize = rd16(data + 54);
  const uint16_t phnum = rd16(data + 56);
  if (phentsize < 8 || phnum == 0)
    return false;
  bool changed = false;
  for (uint16_t i = 0; i < phnum; ++i) {
    const uint64_t off = phoff + uint64_t(i) * phentsize;
    if (off + 8 > n)
      break;
    if (rd32(data + off) != kPtGnuStack)
      continue;
    const uint32_t flags = rd32(data + off + 4);
    if (flags & kPfX) {
      wr32(data + off + 4, flags & ~kPfX);
      changed = true;
    }
  }
  return changed;
}

inline bool elf_needs_gnu_stack_clear(const uint8_t *data, size_t n) {
  if (n < 64 || data[0] != 0x7f || data[1] != 'E' || data[2] != 'L' ||
      data[3] != 'F')
    return false;
  if (data[4] != 2 || data[5] != 1)
    return false;
  const uint64_t phoff =
      uint64_t(rd32(data + 32)) | (uint64_t(rd32(data + 36)) << 32);
  const uint16_t phentsize = rd16(data + 54);
  const uint16_t phnum = rd16(data + 56);
  if (phentsize < 8 || phnum == 0)
    return false;
  for (uint16_t i = 0; i < phnum; ++i) {
    const uint64_t off = phoff + uint64_t(i) * phentsize;
    if (off + 8 > n)
      break;
    if (rd32(data + off) == kPtGnuStack)
      return (rd32(data + off + 4) & kPfX) != 0;
  }
  return false;
}

// Publication writes a private complete file, then replaces the destination.
// A reader therefore observes one full payload, whichever writer landed last.
// MoveFileExW refuses to replace a destination that has any open handle,
// whatever its sharing, with access denied (5), so a reader in a loop can
// keep a writer out past the deadline. That refusal is retried at once with
// POSIX semantics (FileRenameInfoEx, Windows 10 1607 and later, NTFS), which
// replace a destination held open with delete sharing, as read_all opens it;
// the holder keeps reading the file it opened. Access denied and sharing
// violation (32) that remain, from a holder without delete sharing or a
// filesystem without POSIX semantics, and those of exists and status while
// another replace is in flight, are retried with exponential backoff until
// the deadline. POSIX uses rename, and only permission_denied is transient.
inline constexpr std::chrono::milliseconds kPublicationInitialBackoff{1};
inline constexpr std::chrono::milliseconds kPublicationMaxBackoff{32};
inline constexpr std::chrono::milliseconds kPublicationDeadline{2000};

inline bool publication_time_left(std::chrono::steady_clock::time_point start) {
  return std::chrono::steady_clock::now() - start < kPublicationDeadline;
}

inline void publication_backoff(std::chrono::milliseconds &delay) {
  std::this_thread::sleep_for(delay);
  const auto doubled = delay * 2;
  delay = doubled < kPublicationMaxBackoff ? doubled : kPublicationMaxBackoff;
}

// The last error of a retry loop that gave up, with its code and the attempts
// the loop made before the deadline.
inline std::string
publication_failure(const std::error_code &ec, int attempts,
                    std::chrono::steady_clock::time_point start) {
  const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - start);
  return ec.message() + " (" + ec.category().name() + " error " +
         std::to_string(ec.value()) + ", " + std::to_string(attempts) +
         (attempts == 1 ? " attempt in " : " attempts in ") +
         std::to_string(elapsed.count()) + " ms)";
}

#ifdef _WIN32
inline std::error_code win_last_error() {
  return std::error_code(static_cast<int>(GetLastError()),
                         std::system_category());
}

inline std::vector<uint8_t> read_shared(const std::filesystem::path &path,
                                       std::error_code &ec) {
  ec.clear();
  HANDLE handle = CreateFileW(
      path.c_str(), GENERIC_READ,
      FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
      OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (handle == INVALID_HANDLE_VALUE) {
    ec = win_last_error();
    return {};
  }
  LARGE_INTEGER size{};
  if (!GetFileSizeEx(handle, &size) || size.QuadPart < 0) {
    ec = win_last_error();
    CloseHandle(handle);
    return {};
  }
  std::vector<uint8_t> buf(static_cast<std::size_t>(size.QuadPart));
  std::size_t got = 0;
  while (got < buf.size()) {
    DWORD n = 0;
    std::size_t chunk = buf.size() - got;
    if (chunk > (1u << 20))
      chunk = 1u << 20;
    if (!ReadFile(handle, buf.data() + got, static_cast<DWORD>(chunk), &n,
                  nullptr)) {
      ec = win_last_error();
      CloseHandle(handle);
      return {};
    }
    if (n == 0)
      break;
    got += static_cast<std::size_t>(n);
  }
  CloseHandle(handle);
  if (got != buf.size()) {
    ec = std::make_error_code(std::errc::io_error);
    return {};
  }
  return buf;
}
#endif

inline bool publication_retryable(const std::error_code &ec) {
  if (!ec)
    return false;
#ifdef _WIN32
  if (ec.category() == std::system_category())
    return ec.value() == ERROR_ACCESS_DENIED ||
           ec.value() == ERROR_SHARING_VIOLATION;
  return ec.default_error_condition() == std::errc::permission_denied;
#else
  return ec == std::errc::permission_denied;
#endif
}

// Creating or removing a private slot can report not-found while another
// thread is deleting that directory. That is transient on Windows only.
inline bool publication_slot_absent(const std::error_code &ec) {
#ifdef _WIN32
  if (!ec || ec.category() != std::system_category())
    return false;
  return ec.value() == ERROR_FILE_NOT_FOUND ||
         ec.value() == ERROR_PATH_NOT_FOUND;
#else
  (void)ec;
  return false;
#endif
}

inline bool publication_slot_taken(const std::error_code &ec) {
  if (!ec)
    return false;
  if (ec == std::errc::file_exists)
    return true;
#ifdef _WIN32
  if (ec.category() == std::system_category() &&
      (ec.value() == ERROR_ALREADY_EXISTS || ec.value() == ERROR_FILE_EXISTS))
    return true;
#endif
  return ec.default_error_condition() == std::errc::file_exists;
}

#ifdef _WIN32
// FILE_RENAME_INFO as FileRenameInfoEx reads it. Headers that target Windows
// before 10 1607 declare neither its Flags member nor the class.
struct PublicationRenameInfo {
  DWORD flags;
  HANDLE root_directory;
  DWORD file_name_length;
  WCHAR file_name[1];
};
static_assert(offsetof(PublicationRenameInfo, root_directory) ==
              offsetof(FILE_RENAME_INFO, RootDirectory));
static_assert(offsetof(PublicationRenameInfo, file_name_length) ==
              offsetof(FILE_RENAME_INFO, FileNameLength));
static_assert(offsetof(PublicationRenameInfo, file_name) ==
              offsetof(FILE_RENAME_INFO, FileName));
inline constexpr int kFileRenameInfoEx = 22;
inline constexpr DWORD kRenameReplaceIfExists = 0x1;
inline constexpr DWORD kRenamePosixSemantics = 0x2;

inline std::error_code publish_replace_posix(const std::filesystem::path &from,
                                             const std::filesystem::path &to) {
  std::error_code ec;
  const std::wstring target = std::filesystem::absolute(to, ec).native();
  if (ec)
    return ec;
  const std::size_t name_bytes = target.size() * sizeof(wchar_t);
  const std::size_t size =
      offsetof(PublicationRenameInfo, file_name) + name_bytes + sizeof(wchar_t);
  std::vector<std::max_align_t> storage(
      (size + sizeof(std::max_align_t) - 1) / sizeof(std::max_align_t));
  auto *bytes = reinterpret_cast<unsigned char *>(storage.data());
  auto *info = reinterpret_cast<PublicationRenameInfo *>(bytes);
  info->flags = kRenameReplaceIfExists | kRenamePosixSemantics;
  info->root_directory = nullptr;
  info->file_name_length = static_cast<DWORD>(name_bytes);
  std::memcpy(bytes + offsetof(PublicationRenameInfo, file_name),
              target.c_str(), name_bytes + sizeof(wchar_t));
  HANDLE handle = CreateFileW(
      from.c_str(), DELETE,
      FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
      OPEN_EXISTING, FILE_FLAG_OPEN_REPARSE_POINT, nullptr);
  if (handle == INVALID_HANDLE_VALUE)
    return win_last_error();
  if (!SetFileInformationByHandle(
          handle, static_cast<FILE_INFO_BY_HANDLE_CLASS>(kFileRenameInfoEx),
          info, static_cast<DWORD>(size)))
    ec = win_last_error();
  CloseHandle(handle);
  return ec;
}
#endif

inline std::error_code publish_replace(const std::filesystem::path &from,
                                       const std::filesystem::path &to) {
#ifdef _WIN32
  std::error_code ec;
  const auto dest = std::filesystem::symlink_status(to, ec);
  if (publication_retryable(ec))
    return ec;
  if (!ec && std::filesystem::is_directory(dest))
    return std::error_code(ERROR_DIRECTORY, std::system_category());
  if (MoveFileExW(from.c_str(), to.c_str(), MOVEFILE_REPLACE_EXISTING))
    return {};
  const std::error_code moved = win_last_error();
  if (moved.value() != ERROR_ACCESS_DENIED)
    return moved;
  const std::error_code posix = publish_replace_posix(from, to);
  if (!posix)
    return {};
  // Without POSIX semantics the class is rejected (invalid parameter, not
  // supported or invalid function), and the refusal stands until the
  // holder closes.
  return publication_retryable(posix) ? posix : moved;
#else
  std::error_code ec;
  std::filesystem::rename(from, to, ec);
  return ec;
#endif
}

inline std::vector<uint8_t> read_all(const std::string &path) {
  const auto start = std::chrono::steady_clock::now();
  auto delay = kPublicationInitialBackoff;
  const std::filesystem::path file(path);
  for (int attempts = 1;; ++attempts) {
#ifdef _WIN32
    std::error_code ec;
    auto buf = read_shared(file, ec);
    if (!ec)
      return buf;
    const bool again =
        publication_retryable(ec) || publication_slot_absent(ec);
    if (!again || !publication_time_left(start))
      throw std::runtime_error("UmaPot: cannot read AOTI package " + path +
                               ": " + publication_failure(ec, attempts, start));
    publication_backoff(delay);
#else
    errno = 0;
    std::ifstream in(path, std::ios::binary);
    if (in) {
      in.seekg(0, std::ios::end);
      const auto sz = static_cast<size_t>(in.tellg());
      in.seekg(0);
      std::vector<uint8_t> buf(sz);
      if (sz && !in.read(reinterpret_cast<char *>(buf.data()),
                         static_cast<std::streamsize>(sz)))
        throw std::runtime_error("UmaPot: short read of AOTI package " + path);
      return buf;
    }
    const std::error_code opened =
        errno ? std::error_code(errno, std::generic_category())
              : std::make_error_code(std::errc::io_error);
    std::error_code ec;
    const bool present = std::filesystem::exists(file, ec);
    if ((publication_retryable(ec) || present) &&
        publication_time_left(start)) {
      publication_backoff(delay);
      continue;
    }
    throw std::runtime_error("UmaPot: cannot read AOTI package " + path +
                             ": " +
                             publication_failure(opened, attempts, start));
#endif
  }
}

inline void write_all(const std::string &path,
                      const std::vector<uint8_t> &buf) {
  namespace fs = std::filesystem;
  const auto parent = fs::path(path).parent_path();
  if (!parent.empty()) {
    std::error_code created;
    fs::create_directories(parent, created);
    if (created) {
      std::error_code st;
      if (!fs::is_directory(parent, st))
        throw std::runtime_error("UmaPot: cannot create " + parent.string() +
                                 ": " + created.message());
    }
  }
  struct Publication {
    fs::path directory;
    fs::path payload;
    bool owned = false;
    ~Publication() {
      if (!owned)
        return;
      std::error_code ignored;
      fs::remove(payload, ignored);
      fs::remove(directory, ignored);
    }
  } publication;
  // Directory creation reserves a private publication slot across processes
  // and threads. Only a closed, complete file is renamed to the shared path.
  const auto reserve_start = std::chrono::steady_clock::now();
  auto delay = kPublicationInitialBackoff;
  int reserve_attempts = 0;
  for (std::size_t slot = 0; !publication.owned;) {
    publication.directory = path + ".tmp." + std::to_string(slot);
    publication.payload = publication.directory / "payload.pt2";
    std::error_code error;
    ++reserve_attempts;
    publication.owned = fs::create_directory(publication.directory, error);
    if (publication.owned)
      break;
    // libstdc++ reports an existing directory with a clear error_code.
    // MSVC reports ERROR_ALREADY_EXISTS. Either way the slot is taken.
    if (!error || publication_slot_taken(error)) {
      ++slot;
      continue;
    }
    // A slot another thread is deleting reports a transient error on Windows.
    if ((publication_retryable(error) || publication_slot_absent(error)) &&
        publication_time_left(reserve_start)) {
      publication_backoff(delay);
      continue;
    }
    throw std::runtime_error(
        "UmaPot: cannot reserve publication for " + path + ": " +
        publication_failure(error, reserve_attempts, reserve_start));
  }
  const std::string tmp = publication.payload.string();
  {
    std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
    if (!out)
      throw std::runtime_error("UmaPot: cannot write " + tmp);
    if (!buf.empty() && !out.write(reinterpret_cast<const char *>(buf.data()),
                                   static_cast<std::streamsize>(buf.size())))
      throw std::runtime_error("UmaPot: short write of " + tmp);
    out.close();
    if (!out)
      throw std::runtime_error("UmaPot: cannot close " + tmp);
  }
  const auto publish_start = std::chrono::steady_clock::now();
  delay = kPublicationInitialBackoff;
  int publish_attempts = 1;
  std::error_code ec = publish_replace(publication.payload, path);
  while (publication_retryable(ec) && publication_time_left(publish_start)) {
    publication_backoff(delay);
    if (!publication_time_left(publish_start))
      break;
    ++publish_attempts;
    ec = publish_replace(publication.payload, path);
  }
  if (ec)
    throw std::runtime_error(
        "UmaPot: cannot publish " + path + ": " +
        publication_failure(ec, publish_attempts, publish_start));
}

} // namespace aoti_execstack
} // namespace rgpot
