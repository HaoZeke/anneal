// MIT License
// Copyright 2023--present rgpot developers
//
// Two processes share one socket. The client sends NeighborRequest chunks
// (cursor, limit). The server answers with NeighborList. The client
// repeats the call while more is set, feeding next back as the cursor.
// The concatenated columns match a local build of the same geometry.

#include "rgpot/nlist/NeighbourList.hpp"

#if defined(__has_include)
#if !__has_include(<capnp/dynamic.h>)
#error "nlist_stream requires Cap'n Proto headers (<capnp/dynamic.h>); enable with_rpc"
#endif
#endif

#include <capnp/dynamic.h>
#include <capnp/message.h>
#include <capnp/schema-parser.h>
#include <capnp/serialize-packed.h>
#include <kj/filesystem.h>
#include <kj/io.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace {

constexpr uint32_t kLimit = 2;

void write_frame(int fd, capnp::MessageBuilder &message) {
  kj::VectorOutputStream buffer;
  capnp::writePackedMessage(buffer, message);
  const auto bytes = buffer.getArray();
  const auto len = static_cast<uint32_t>(bytes.size());
  if (write(fd, &len, sizeof len) != static_cast<ssize_t>(sizeof len) ||
      write(fd, bytes.begin(), bytes.size()) != static_cast<ssize_t>(bytes.size())) {
    throw std::runtime_error("short write");
  }
}

std::vector<kj::byte> read_frame(int fd) {
  uint32_t len = 0;
  if (read(fd, &len, sizeof len) != static_cast<ssize_t>(sizeof len)) {
    throw std::runtime_error("short read of frame length");
  }
  std::vector<kj::byte> packed(len);
  std::size_t got = 0;
  while (got < packed.size()) {
    const auto n = read(fd, packed.data() + got, packed.size() - got);
    if (n <= 0) {
      throw std::runtime_error("short read of frame");
    }
    got += static_cast<std::size_t>(n);
  }
  return packed;
}

template <typename Fn>
void read_struct(int fd, capnp::StructSchema schema, Fn &&fn) {
  auto packed = read_frame(fd);
  kj::ArrayInputStream input(kj::arrayPtr(packed.data(), packed.size()));
  capnp::PackedMessageReader reader(input);
  fn(reader.getRoot<capnp::DynamicStruct>(schema));
}

capnp::ParsedSchema load_schema(const std::string &path) {
  static capnp::SchemaParser parser;
  auto fs = kj::newDiskFilesystem();
  std::string relative = path;
  if (!relative.empty() && relative.front() == '/') {
    relative.erase(relative.begin());
  }
  return parser.parseFromDirectory(fs->getRoot(), kj::Path::parse(relative), nullptr);
}

const double kPos[] = {0.2, 0.2, 0.2, 1.4, 0.2, 0.4, 0.3, 1.6, 0.5, 2.2, 0.4, 1.1};
const double kBox[9] = {6, 0, 0, 0, 6, 0, 0, 0, 6};

rgpot::nlist::Query query() {
  rgpot::nlist::Query q;
  q.cutoff = 2.2;
  q.shape = rgpot::nlist::Pairs{rgpot::nlist::Shell::Half};
  return q;
}

void serve(int fd, capnp::StructSchema request_schema, capnp::StructSchema list_schema) {
  rgpot::nlist::Table table;
  rgpot::nlist::build(table, kPos, 4, kBox, query());
  bool stop = false;
  while (!stop) {
    read_struct(fd, request_schema, [&](capnp::DynamicStruct::Reader request) {
      if (request.get("mode").as<std::uint8_t>() == 0) {
        capnp::MallocMessageBuilder message;
        auto root = message.initRoot<capnp::DynamicStruct>(list_schema);
        root.set("status", 5);
        root.set("message", "mode 0 is not a neighbour mode");
        root.set("more", false);
        write_frame(fd, message);
        stop = true;
        return;
      }
      const auto cursor = request.get("cursor").as<std::uint64_t>();
      auto limit = request.get("limit").as<std::uint32_t>();
      if (limit == 0) {
        limit = 4096;
      }
      if (cursor > table.size()) {
        capnp::MallocMessageBuilder message;
        auto root = message.initRoot<capnp::DynamicStruct>(list_schema);
        root.set("status", 6);
        root.set("message", "cursor is past the end of the neighbour list");
        root.set("more", false);
        write_frame(fd, message);
        stop = true;
        return;
      }
      const auto count = std::min(static_cast<std::size_t>(limit),
                                   table.size() - static_cast<std::size_t>(cursor));
      capnp::MallocMessageBuilder message;
      auto root = message.initRoot<capnp::DynamicStruct>(list_schema);
      auto first = root.init("first", static_cast<unsigned>(count)).as<capnp::DynamicList>();
      auto second = root.init("second", static_cast<unsigned>(count)).as<capnp::DynamicList>();
      auto shift = root.init("shift", static_cast<unsigned>(count * 3)).as<capnp::DynamicList>();
      auto distance = root.init("distance", static_cast<unsigned>(count)).as<capnp::DynamicList>();
      for (std::size_t i = 0; i < count; ++i) {
        const auto src = static_cast<std::size_t>(cursor) + i;
        first.set(static_cast<unsigned>(i), table.first[src]);
        second.set(static_cast<unsigned>(i), table.second[src]);
        shift.set(static_cast<unsigned>(3 * i), table.shift[3 * src]);
        shift.set(static_cast<unsigned>(3 * i + 1), table.shift[3 * src + 1]);
        shift.set(static_cast<unsigned>(3 * i + 2), table.shift[3 * src + 2]);
        distance.set(static_cast<unsigned>(i), table.distance[src]);
      }
      const auto next = cursor + count;
      root.set("more", next < table.size());
      root.set("next", next);
      root.set("status", 0);
      write_frame(fd, message);
      stop = next >= table.size();
    });
  }
}

int client(int fd, capnp::StructSchema request_schema, capnp::StructSchema list_schema) {
  rgpot::nlist::Table table;
  rgpot::nlist::build(table, kPos, 4, kBox, query());
  std::uint64_t cursor = 0;
  std::size_t seen = 0;
  bool more = true;
  int failed = 0;
  while (more && failed == 0) {
    capnp::MallocMessageBuilder message;
    auto request = message.initRoot<capnp::DynamicStruct>(request_schema);
    request.set("cutoff", 2.2);
    request.set("format", std::uint8_t{0});
    request.set("mode", std::uint8_t{2});
    request.set("cursor", cursor);
    request.set("limit", kLimit);
    write_frame(fd, message);
    read_struct(fd, list_schema, [&](capnp::DynamicStruct::Reader chunk) {
      const auto status = chunk.get("status").as<std::int32_t>();
      if (status != 0) {
        std::fprintf(stderr, "server status %d\n", status);
        failed = 1;
        return;
      }
      auto first = chunk.get("first").as<capnp::DynamicList>();
      auto second = chunk.get("second").as<capnp::DynamicList>();
      if (first.size() > kLimit) {
        std::fprintf(stderr, "chunk longer than the limit\n");
        failed = 1;
        return;
      }
      for (unsigned i = 0; i < first.size(); ++i) {
        const auto src = static_cast<std::size_t>(cursor) + i;
        if (first[i].as<std::int32_t>() != table.first[src] ||
            second[i].as<std::int32_t>() != table.second[src]) {
          std::fprintf(stderr, "chunk pair %u does not match the local list\n", i);
          failed = 1;
          return;
        }
      }
      seen += first.size();
      const auto next = chunk.get("next").as<std::uint64_t>();
      if (next != cursor + first.size()) {
        std::fprintf(stderr, "next cursor skipped or rewound\n");
        failed = 1;
        return;
      }
      more = chunk.get("more").as<bool>();
      cursor = next;
    });
  }
  if (failed != 0) {
    return 1;
  }
  if (seen != table.size() || cursor != table.size()) {
    std::fprintf(stderr, "stream saw %zu of %zu\n", seen, table.size());
    return 1;
  }
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: nlist_stream path/to/Potentials.capnp\n");
    return 2;
  }
  try {
    auto parsed = load_schema(argv[1]);
    auto request_schema = parsed.getNested("NeighborRequest").asStruct();
    auto list_schema = parsed.getNested("NeighborList").asStruct();
    int socks[2];
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, socks) != 0) {
      std::perror("socketpair");
      return 1;
    }
    const pid_t pid = fork();
    if (pid < 0) {
      std::perror("fork");
      return 1;
    }
    if (pid == 0) {
      close(socks[0]);
      try {
        serve(socks[1], request_schema, list_schema);
      } catch (const std::exception &err) {
        std::fprintf(stderr, "server: %s\n", err.what());
        _exit(1);
      }
      _exit(0);
    }
    close(socks[1]);
    const int rc = client(socks[0], request_schema, list_schema);
    close(socks[0]);
    int status = 0;
    waitpid(pid, &status, 0);
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
      std::fprintf(stderr, "server process failed\n");
      return 1;
    }
    return rc;
  } catch (const std::exception &err) {
    std::fprintf(stderr, "%s\n", err.what());
    return 1;
  }
}
