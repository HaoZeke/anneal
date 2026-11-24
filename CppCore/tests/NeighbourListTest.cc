// MIT License
// Copyright 2023--present rgpot developers

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <map>
#include <random>
#include <set>
#include <span>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

#include <catch2/catch_all.hpp>

#include "TestFixtures.hpp"
#include "lsan_libomp.hpp"
#include "rgpot/nlist/NeighbourList.hpp"
#include "rgpot/nlist/rgpot_nlist.h"

namespace {

void set_omp_threads(const char *value) {
  rgpot::test_fixtures::setOmpThreads(value);
}

struct Key {
  int i, j, sa, sb, sc;
  bool operator<(const Key &o) const {
    return std::tie(i, j, sa, sb, sc) < std::tie(o.i, o.j, o.sa, o.sb, o.sc);
  }
  bool operator==(const Key &o) const {
    return std::tie(i, j, sa, sb, sc) == std::tie(o.i, o.j, o.sa, o.sb, o.sc);
  }
};

std::set<Key> brute(const std::vector<double> &pos, const double *box, double cutoff,
                    bool full, bool strict, const std::array<bool, 3> &periodic,
                    int span = 3) {
  const int n = static_cast<int>(pos.size() / 3);
  std::set<Key> out;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      for (int sa = -span; sa <= span; ++sa) {
        if (!periodic[0] && sa != 0) {
          continue;
        }
        for (int sb = -span; sb <= span; ++sb) {
          if (!periodic[1] && sb != 0) {
            continue;
          }
          for (int sc = -span; sc <= span; ++sc) {
            if (!periodic[2] && sc != 0) {
              continue;
            }
            double vx = pos[3 * j] - pos[3 * i];
            double vy = pos[3 * j + 1] - pos[3 * i + 1];
            double vz = pos[3 * j + 2] - pos[3 * i + 2];
            vx += sa * box[0] + sb * box[3] + sc * box[6];
            vy += sa * box[1] + sb * box[4] + sc * box[7];
            vz += sa * box[2] + sb * box[5] + sc * box[8];
            const double r2 = vx * vx + vy * vy + vz * vz;
            const double cut2 = cutoff * cutoff;
            const bool inside = strict ? r2 < cut2 : r2 <= cut2;
            if (!(r2 > 0.0) || !inside) {
              continue;
            }
            if (!full) {
              if (i > j) {
                continue;
              }
              if (i == j) {
                const int sum = sa + sb + sc;
                if (sum < 0 || (sum == 0 && (sc < 0 || (sc == 0 && sb < 0))) ||
                    (sa == 0 && sb == 0 && sc == 0)) {
                  continue;
                }
              }
            }
            out.insert(Key{i, j, sa, sb, sc});
          }
        }
      }
    }
  }
  return out;
}

std::set<Key> from_table(const rgpot::nlist::Table &table) {
  std::set<Key> out;
  for (std::size_t p = 0; p < table.size(); ++p) {
    const int mask = table.mask[p];
    const int origin = table.first[p];
    if (mask == 0xff) {
      out.insert(Key{origin, table.second[p], table.shift[3 * p], table.shift[3 * p + 1],
                     table.shift[3 * p + 2]});
    } else {
      for (int lane = 0; lane < 4; ++lane) {
        if ((mask & (1 << lane)) != 0) {
          out.insert(Key{origin + lane, table.second[p], table.shift[3 * p],
                         table.shift[3 * p + 1], table.shift[3 * p + 2]});
        }
      }
    }
  }
  return out;
}

std::vector<double> fractional_positions(const double box[9], std::size_t n) {
  std::mt19937_64 rng(731);
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  std::vector<double> pos;
  pos.reserve(3 * n);
  for (std::size_t i = 0; i < n; ++i) {
    const double f[3] = {unit(rng), unit(rng), unit(rng)};
    for (int c = 0; c < 3; ++c) {
      pos.push_back(f[0] * box[c] + f[1] * box[3 + c] + f[2] * box[6 + c]);
    }
  }
  // Pin a short pair so a missed near neighbour is not hidden by aggregate
  // pair counts.
  pos[3] = pos[0] + 0.3;
  pos[4] = pos[1];
  pos[5] = pos[2];
  return pos;
}

void require_image_sum(const std::vector<double> &pos, const double *box,
                       double cutoff, bool full,
                       const std::array<bool, 3> &periodic) {
  rgpot::nlist::Query query;
  query.cutoff = cutoff;
  query.shape = rgpot::nlist::Pairs{full ? rgpot::nlist::Shell::Full
                                         : rgpot::nlist::Shell::Half};
  query.cut = rgpot::nlist::Cut::Strict;
  query.periodic = periodic;
  rgpot::nlist::Table table;
  rgpot::nlist::build(table, pos.data(), pos.size() / 3, box, query);
  const auto ref = brute(pos, box, cutoff, full, true, periodic);
  REQUIRE(table.size() == ref.size());
  REQUIRE(from_table(table) == ref);
  for (std::size_t p = 0; p < table.size(); ++p) {
    const std::size_t i = static_cast<std::size_t>(table.first[p]);
    const std::size_t j = static_cast<std::size_t>(table.second[p]);
    const int sa = table.shift[3 * p];
    const int sb = table.shift[3 * p + 1];
    const int sc = table.shift[3 * p + 2];
    const double vx =
        pos[3 * j] - pos[3 * i] + sa * box[0] + sb * box[3] + sc * box[6];
    const double vy = pos[3 * j + 1] - pos[3 * i + 1] + sa * box[1] +
                      sb * box[4] + sc * box[7];
    const double vz = pos[3 * j + 2] - pos[3 * i + 2] + sa * box[2] +
                      sb * box[5] + sc * box[8];
    REQUIRE(table.distance[p] ==
            Catch::Approx(std::sqrt(vx * vx + vy * vy + vz * vz)).margin(1e-12));
  }
}

} // namespace

TEST_CASE("small periodic cells enumerate every image", "[nlist][api]") {
  const std::array<bool, 3> periodic{{true, true, true}};

  SECTION("one atom in a cubic cell") {
    const double box[9] = {2, 0, 0, 0, 2, 0, 0, 0, 2};
    const std::vector<double> pos{0.2, 0.3, 0.4};
    for (bool full : {false, true}) {
      require_image_sum(pos, box, 3.0, full, periodic);
      rgpot::nlist::Query query;
      query.cutoff = 3.0;
      query.shape = rgpot::nlist::Pairs{full ? rgpot::nlist::Shell::Full
                                             : rgpot::nlist::Shell::Half};
      rgpot::nlist::Table table;
      rgpot::nlist::build(table, pos.data(), 1, box, query);
      REQUIRE(table.size() == (full ? 18 : 9));
    }
  }

  SECTION("one atom in a triclinic cell") {
    const double box[9] = {2.0, 0, 0, 0.6, 1.9, 0, 0.2, 0.4, 2.1};
    const std::vector<double> pos{0.2, 0.3, 0.4};
    for (bool full : {false, true}) {
      require_image_sum(pos, box, 3.0, full, periodic);
    }
  }

  SECTION("two atoms across image-count thresholds") {
    const double box[9] = {4, 0, 0, 0, 4, 0, 0, 0, 4};
    const std::vector<double> pos{0.2, 0.3, 0.4, 1.3, 1.1, 0.9};
    for (double ratio : {0.49, 0.51, 0.99, 1.01, 1.49, 1.51}) {
      for (bool full : {false, true}) {
        CAPTURE(ratio, full);
        require_image_sum(pos, box, ratio * 4.0, full, periodic);
      }
    }
  }
}

TEST_CASE("neighbour query matches a triclinic brute force", "[nlist][api]") {
  const double box[9] = {4.0, 0.2, 0.1, 0.3, 4.5, -0.2, 0.05, 0.15, 4.2};
  const std::vector<double> pos = {0.2, 0.3, 0.4, 3.6, 0.5, 0.2, 1.0, 3.8, 0.7,
                                   2.2, 2.1, 3.9};
  const double cutoff = 3.2;
  const std::array<bool, 3> periodic{{true, true, false}};
  for (bool full : {false, true}) {
    for (bool strict : {true, false}) {
      rgpot::nlist::Query query;
      query.cutoff = cutoff;
      query.skin = 0;
      query.shape = rgpot::nlist::Pairs{full ? rgpot::nlist::Shell::Full
                                            : rgpot::nlist::Shell::Half};
      query.cut = strict ? rgpot::nlist::Cut::Strict : rgpot::nlist::Cut::Loose;
      query.periodic = periodic;
      rgpot::nlist::Table table;
      rgpot::nlist::build(table, pos.data(), pos.size() / 3, box, query);
      const auto got = from_table(table);
      const auto ref = brute(pos, box, cutoff, full, strict, periodic);
      REQUIRE(got == ref);
      for (std::size_t p = 0; p < table.size(); ++p) {
        const int i = table.mask[p] == 0xff ? table.first[p] : table.first[p];
        (void)i;
        if (table.mask[p] != 0xff) {
          continue;
        }
        const int j = table.second[p];
        const double vx = pos[3 * j] - pos[3 * table.first[p]] +
                          table.shift[3 * p] * box[0] + table.shift[3 * p + 1] * box[3] +
                          table.shift[3 * p + 2] * box[6];
        const double vy = pos[3 * j + 1] - pos[3 * table.first[p] + 1] +
                          table.shift[3 * p] * box[1] + table.shift[3 * p + 1] * box[4] +
                          table.shift[3 * p + 2] * box[7];
        const double vz = pos[3 * j + 2] - pos[3 * table.first[p] + 2] +
                          table.shift[3 * p] * box[2] + table.shift[3 * p + 1] * box[5] +
                          table.shift[3 * p + 2] * box[8];
        REQUIRE(std::abs(vx - table.vector[3 * p]) < 1e-12);
        REQUIRE(std::abs(vy - table.vector[3 * p + 1]) < 1e-12);
        REQUIRE(std::abs(vz - table.vector[3 * p + 2]) < 1e-12);
      }
    }
  }
}

TEST_CASE("a chunked walk returns the stored pairs", "[nlist][api]") {
  const double box[9] = {6, 0, 0, 0, 5, 0, 1, 0, 7};
  const std::vector<double> pos = {0.5, 0.5, 0.5, 2.0, 1.0, 1.0, 4.5, 3.0, 6.0};
  rgpot::nlist::Query query;
  query.cutoff = 4.0;
  query.shape = rgpot::nlist::Pairs{rgpot::nlist::Shell::Full};
  query.cut = rgpot::nlist::Cut::Strict;
  rgpot::nlist::Table table;
  rgpot::nlist::build(table, pos.data(), pos.size() / 3, box, query);
  rgpot::nlist::Walk walk(pos.data(), pos.size() / 3, box, query);
  std::vector<std::int32_t> first(2), second(2), shift(6);
  std::vector<double> vector(6), distance(2);
  std::vector<std::uint8_t> mask(2);
  std::set<Key> got;
  bool done = false;
  while (!done) {
    rgpot::nlist::Buffers dest;
    dest.first = first;
    dest.second = second;
    dest.shift = shift;
    dest.vector = vector;
    dest.distance = distance;
    dest.mask = mask;
    const std::size_t n = walk.pull(dest, done);
    for (std::size_t p = 0; p < n; ++p) {
      got.insert(Key{first[p], second[p], shift[3 * p], shift[3 * p + 1], shift[3 * p + 2]});
    }
  }
  REQUIRE(got == from_table(table));
}

TEST_CASE("the C ABI round-trips a stored list and a stream", "[nlist][api]") {
  REQUIRE(rgpot_nlist_abi_version() == RGPOT_NLIST_ABI);
  const double box[9] = {5, 0, 0, 0, 5, 0, 0.4, 0, 5};
  const double pos[] = {0.2, 0.2, 0.2, 1.4, 0.3, 0.5, 3.0, 3.0, 1.0};
  rgpot_nlist_request request;
  rgpot_nlist_request_init(&request);
  request.cutoff = 3.0;
  REQUIRE(request.periodic[0] == 1);
  REQUIRE(request.want_distances == 1);
  REQUIRE(request.shape == 0);
  REQUIRE(request.shell == 0);
  REQUIRE(request.bytes == sizeof(request));
  std::size_t count = 0;
  std::size_t needed = 0;
  REQUIRE(rgpot_nlist_build(pos, 3, box, &request, nullptr, &count, &needed) ==
          RGPOT_NLIST_OK);
  REQUIRE(needed > 0);
  std::vector<std::int32_t> first(needed), second(needed), shift(needed * 3);
  std::vector<double> vector(needed * 3), distance(needed);
  std::vector<std::uint8_t> mask(needed);
  rgpot_nlist_soa soa{};
  soa.bytes = sizeof(soa);
  soa.first = first.data();
  soa.second = second.data();
  soa.shift = shift.data();
  soa.vector = vector.data();
  soa.distance = distance.data();
  soa.mask = mask.data();
  soa.cap = needed;
  REQUIRE(rgpot_nlist_build(pos, 3, box, &request, &soa, &count, &needed) == RGPOT_NLIST_OK);
  REQUIRE(count == needed);
  soa.cap = 1;
  REQUIRE(rgpot_nlist_build(pos, 3, box, &request, &soa, &count, &needed) == RGPOT_NLIST_SHORT);
  REQUIRE(count == 1);
  REQUIRE(needed > 1);
  REQUIRE(std::string(rgpot_nlist_strerror(RGPOT_NLIST_SHORT)).find("needed") !=
          std::string::npos);
  rgpot_nlist_request zero{};
  REQUIRE(rgpot_nlist_build(pos, 3, box, &zero, nullptr, &count, &needed) == RGPOT_NLIST_ERR_ABI);
  request.shape = 9;
  REQUIRE(rgpot_nlist_build(pos, 3, box, &request, nullptr, &count, &needed) ==
          RGPOT_NLIST_SHAPE);
  request.shape = 1;
  request.shell = 0;
  REQUIRE(rgpot_nlist_build(pos, 3, box, &request, nullptr, &count, &needed) ==
          RGPOT_NLIST_SHAPE);
  request.shape = 0;
  request.shell = 0;
  request.cutoff = -1;
  REQUIRE(rgpot_nlist_build(pos, 3, box, &request, nullptr, &count, &needed) ==
          RGPOT_NLIST_CUTOFF);
  request.cutoff = 3.0;
  soa.cap = needed;
  REQUIRE(rgpot_nlist_build(pos, 3, box, &request, &soa, &count, &needed) == RGPOT_NLIST_OK);
  rgpot_nlist_walk *walk = nullptr;
  REQUIRE(rgpot_nlist_walk_open(pos, 3, box, &request, &walk) == RGPOT_NLIST_OK);
  REQUIRE(walk != nullptr);
  std::vector<std::int32_t> streamed_first, streamed_second, streamed_shift;
  int32_t done = 0;
  while (done == 0) {
    std::int32_t chunk_first[2], chunk_second[2], chunk_shift[6];
    double chunk_vector[6], chunk_distance[2];
    std::uint8_t chunk_mask[2];
    rgpot_nlist_soa chunk{};
    chunk.bytes = sizeof(chunk);
    chunk.first = chunk_first;
    chunk.second = chunk_second;
    chunk.shift = chunk_shift;
    chunk.vector = chunk_vector;
    chunk.distance = chunk_distance;
    chunk.mask = chunk_mask;
    chunk.cap = 2;
    std::size_t n = 0;
    REQUIRE(rgpot_nlist_walk_pull(walk, &chunk, &n, &done) == RGPOT_NLIST_OK);
    streamed_first.insert(streamed_first.end(), chunk_first, chunk_first + n);
    streamed_second.insert(streamed_second.end(), chunk_second, chunk_second + n);
    streamed_shift.insert(streamed_shift.end(), chunk_shift, chunk_shift + 3 * n);
  }
  rgpot_nlist_walk_close(walk);
  REQUIRE(streamed_first.size() == first.size());
  for (std::size_t p = 0; p < first.size(); ++p) {
    REQUIRE(streamed_first[p] == first[p]);
    REQUIRE(streamed_second[p] == second[p]);
    REQUIRE(streamed_shift[3 * p] == shift[3 * p]);
    REQUIRE(streamed_shift[3 * p + 1] == shift[3 * p + 1]);
    REQUIRE(streamed_shift[3 * p + 2] == shift[3 * p + 2]);
  }
}

TEST_CASE("a rejected geometry throws and keeps the previous table", "[nlist][api]") {
  const double box[9] = {4, 0, 0, 0, 4, 0, 0, 0, 4};
  const std::vector<double> pos = {0, 0, 0, 1, 0, 0};
  rgpot::nlist::Query query;
  query.cutoff = 1.5;
  rgpot::nlist::Builder builder;
  builder.update(pos.data(), pos.size() / 3, box, query);
  const auto n = builder.table().size();
  REQUIRE(n > 0);
  query.cutoff = -1;
  REQUIRE_THROWS_AS(builder.update(pos.data(), pos.size() / 3, box, query),
                    rgpot::nlist::Error);
  REQUIRE(builder.table().size() == n);
}

TEST_CASE("a cluster of four expands to the full list", "[nlist][api]") {
  const double box[9] = {4.0, 0.2, 0.1, 0.3, 4.5, -0.2, 0.05, 0.15, 4.2};
  const std::vector<double> pos = {0.2, 0.3, 0.4, 3.6, 0.5, 0.2, 1.0, 3.8, 0.7,
                                   2.2, 2.1, 3.9};
  rgpot::nlist::Query query;
  query.cutoff = 3.2;
  query.shape = rgpot::nlist::Cluster4{};
  query.periodic = {{true, true, false}};
  rgpot::nlist::Table clustered;
  rgpot::nlist::build(clustered, pos.data(), pos.size() / 3, box, query);
  query.shape = rgpot::nlist::Pairs{rgpot::nlist::Shell::Full};
  rgpot::nlist::Table full;
  rgpot::nlist::build(full, pos.data(), pos.size() / 3, box, query);
  REQUIRE(from_table(clustered) == from_table(full));
  for (std::uint8_t mask : clustered.mask) {
    REQUIRE(mask != 0);
    REQUIRE((mask & 0xf0) == 0);
  }
}

namespace {

// First, second, mask and the image shift of one stored row.
using Row = std::tuple<int, int, int, int, int, int>;

// The rows of `table`. The image comes from the shift column, or, when the
// table stores none, from the vector: v = r_j - r_i + S H gives
// S = (v - r_j + r_i) H^-1. A cluster row carries the vector of its lowest
// lane.
std::vector<Row> image_rows(const rgpot::nlist::Table &table, const std::vector<double> &pos,
                            const double *box) {
  const double det = box[0] * (box[4] * box[8] - box[5] * box[7]) -
                     box[1] * (box[3] * box[8] - box[5] * box[6]) +
                     box[2] * (box[3] * box[7] - box[4] * box[6]);
  const double inv[9] = {
      (box[4] * box[8] - box[5] * box[7]) / det, (box[2] * box[7] - box[1] * box[8]) / det,
      (box[1] * box[5] - box[2] * box[4]) / det, (box[5] * box[6] - box[3] * box[8]) / det,
      (box[0] * box[8] - box[2] * box[6]) / det, (box[2] * box[3] - box[0] * box[5]) / det,
      (box[3] * box[7] - box[4] * box[6]) / det, (box[1] * box[6] - box[0] * box[7]) / det,
      (box[0] * box[4] - box[1] * box[3]) / det,
  };
  std::vector<Row> out;
  for (std::size_t p = 0; p < table.size(); ++p) {
    const int mask = table.mask[p];
    int s[3] = {0, 0, 0};
    if (!table.shift.empty()) {
      for (int k = 0; k < 3; ++k) {
        s[k] = table.shift[3 * p + static_cast<std::size_t>(k)];
      }
    } else {
      int lane = 0;
      while (mask != 0xff && (mask & (1 << lane)) == 0) {
        ++lane;
      }
      const auto i = static_cast<std::size_t>(table.first[p] + (mask == 0xff ? 0 : lane));
      const auto j = static_cast<std::size_t>(table.second[p]);
      double d[3];
      for (int c = 0; c < 3; ++c) {
        const auto u = static_cast<std::size_t>(c);
        d[c] = table.vector[3 * p + u] - pos[3 * j + u] + pos[3 * i + u];
      }
      for (int k = 0; k < 3; ++k) {
        s[k] = static_cast<int>(
            std::lround(d[0] * inv[k] + d[1] * inv[3 + k] + d[2] * inv[6 + k]));
      }
    }
    out.emplace_back(table.first[p], table.second[p], mask, s[0], s[1], s[2]);
  }
  std::sort(out.begin(), out.end());
  return out;
}

// First, second and mask of every row, with the distance when stored.
std::vector<std::tuple<int, int, int, double>> plain_rows(const rgpot::nlist::Table &table) {
  std::vector<std::tuple<int, int, int, double>> out;
  for (std::size_t p = 0; p < table.size(); ++p) {
    out.emplace_back(table.first[p], table.second[p], table.mask[p],
                     table.distance.empty() ? 0.0 : table.distance[p]);
  }
  std::sort(out.begin(), out.end());
  return out;
}

// The pairs of a walk, in a table with only the requested columns.
rgpot::nlist::Table walk_table(const std::vector<double> &pos, const double *box,
                               const rgpot::nlist::Query &query) {
  rgpot::nlist::Walk walk(pos.data(), pos.size() / 3, box, query);
  constexpr std::size_t cap = 7;
  std::vector<std::int32_t> first(cap), second(cap), shift(3 * cap);
  std::vector<double> vector(3 * cap), distance(cap);
  std::vector<std::uint8_t> mask(cap);
  rgpot::nlist::Table out;
  bool done = false;
  while (!done) {
    rgpot::nlist::Buffers dest{first, second, shift, vector, distance, mask};
    const std::size_t n = walk.pull(dest, done);
    for (std::size_t p = 0; p < n; ++p) {
      out.first.push_back(first[p]);
      out.second.push_back(second[p]);
      out.mask.push_back(mask[p]);
      for (std::size_t c = 0; c < 3; ++c) {
        if (query.fields.shifts) {
          out.shift.push_back(shift[3 * p + c]);
        }
        if (query.fields.vectors) {
          out.vector.push_back(vector[3 * p + c]);
        }
      }
      if (query.fields.distances) {
        out.distance.push_back(distance[p]);
      }
    }
  }
  return out;
}

// Every subset of the output fields gives the rows of the full request,
// through build, a walk and a builder.
void check_fields_keep_rows(const std::vector<double> &pos, const double *box,
                            rgpot::nlist::Query query) {
  const std::size_t n = pos.size() / 3;
  query.fields = {true, true, true};
  rgpot::nlist::Table full;
  rgpot::nlist::build(full, pos.data(), n, box, query);
  const auto want = image_rows(full, pos, box);
  const auto want_plain = plain_rows(full);
  REQUIRE(!want.empty());
  rgpot::nlist::Table no_shifts = full;
  no_shifts.shift.clear();
  REQUIRE(image_rows(no_shifts, pos, box) == want);
  for (int bits = 0; bits < 8; ++bits) {
    query.fields = {(bits & 1) != 0, (bits & 2) != 0, (bits & 4) != 0};
    std::vector<rgpot::nlist::Table> tables(3);
    rgpot::nlist::build(tables[0], pos.data(), n, box, query);
    tables[1] = walk_table(pos, box, query);
    rgpot::nlist::Builder builder;
    builder.update(pos.data(), n, box, query);
    tables[2] = builder.table();
    for (const auto &table : tables) {
      INFO("fields " << bits << ", " << table.size() << " rows, want " << full.size());
      REQUIRE(table.size() == full.size());
      REQUIRE(table.shift.size() == (query.fields.shifts ? 3 * full.size() : 0));
      if (query.fields.shifts || query.fields.vectors) {
        REQUIRE(image_rows(table, pos, box) == want);
      }
      const auto got = plain_rows(table);
      REQUIRE(got.size() == want_plain.size());
      std::size_t differ = 0;
      for (std::size_t p = 0; p < got.size(); ++p) {
        const auto &[i, j, m, d] = got[p];
        const auto &[wi, wj, wm, wd] = want_plain[p];
        if (i != wi || j != wj || m != wm ||
            (query.fields.distances && !(std::fabs(d - wd) <= 1e-12))) {
          ++differ;
        }
      }
      REQUIRE(differ == 0);
    }
  }
}

} // namespace

TEST_CASE("the requested fields do not change the pairs", "[nlist][api]") {
  struct Case {
    const char *name;
    std::array<double, 9> box;
    std::array<bool, 3> periodic;
    std::vector<double> pos;
    double cutoff;
  };
  std::vector<Case> cases = {
      {"a cell smaller than the cutoff",
       {3, 0, 0, 0, 3, 0, 0, 0, 3},
       {{true, true, true}},
       {0.2, 0.3, 0.4, 1.7, 0.5, 0.2, 1.0, 2.6, 0.7, 2.2, 2.1, 2.9},
       4.0},
      {"a small sheared cell with a partial cluster",
       {4.0, 0.2, 0.1, 0.3, 4.5, -0.2, 0.05, 0.15, 4.2},
       {{true, true, true}},
       {0.2, 0.3, 0.4, 3.6, 0.5, 0.2, 1.0, 3.8, 0.7, 2.2, 2.1, 3.9, 1.1, 1.2, 1.3},
       5.0},
      {"a mixed-periodic sheared slab",
       {3.5, 0, 0, 0.6, 8.0, 0, 0.4, 0.3, 3.2},
       {{true, false, true}},
       {0.1, 0.2, 0.3, 1.9, 1.5, 0.4, 3.1, 4.0, 2.8, 0.7, 6.5, 1.6, 2.4, 2.2, 3.0, 1.3, 7.1, 0.2},
       4.0},
      {"coordinates whole cells outside a small cell",
       {3, 0, 0, 0, 3, 0, 0, 0, 3},
       {{true, true, true}},
       {6.2, 0.3, 0.4, 1.7, -8.5, 0.2, 1.0, 2.6, 12.7, -2.8, 2.1, -0.1},
       4.0},
  };
  {
    const double wide[9] = {9.0, 0, 0, 0.4, 9.5, 0, 0.3, -0.2, 9.2};
    cases.push_back({"600 atoms in a wide cell",
                     {wide[0], wide[1], wide[2], wide[3], wide[4], wide[5], wide[6], wide[7],
                      wide[8]},
                     {{true, true, true}},
                     fractional_positions(wide, 600),
                     2.5});
    const double thin[9] = {14.0, 0, 0, 0, 14.0, 0, 0, 0, 2.2};
    cases.push_back({"600 atoms in a cell thinner than the cutoff",
                     {thin[0], thin[1], thin[2], thin[3], thin[4], thin[5], thin[6], thin[7],
                      thin[8]},
                     {{true, true, true}},
                     fractional_positions(thin, 600),
                     2.6});
  }
  const std::vector<rgpot::nlist::Shape> shapes = {
      rgpot::nlist::Pairs{rgpot::nlist::Shell::Half},
      rgpot::nlist::Pairs{rgpot::nlist::Shell::Full}, rgpot::nlist::Cluster4{}};
  for (const auto &c : cases) {
    for (const auto &shape : shapes) {
      for (const double skin : {0.0, 0.4}) {
        INFO(c.name << ", cluster " << std::holds_alternative<rgpot::nlist::Cluster4>(shape)
                    << ", skin " << skin);
        rgpot::nlist::Query query;
        query.cutoff = c.cutoff;
        query.skin = skin;
        query.cut = skin > 0.0 ? rgpot::nlist::Cut::Loose : rgpot::nlist::Cut::Strict;
        query.shape = shape;
        query.periodic = c.periodic;
        check_fields_keep_rows(c.pos, c.box.data(), query);
      }
    }
  }
}

TEST_CASE("a wide orthorhombic cell matches the brute force", "[nlist][api]") {
  // The box is many cutoffs across, so the query takes the cell-shift path.
  const double box[9] = {40, 0, 0, 0, 36, 0, 0, 0, 44};
  std::vector<double> pos;
  for (int i = 0; i < 24; ++i) {
    pos.push_back(1.5 + (i % 5) * 7.0);
    pos.push_back(2.0 + (i % 7) * 4.5);
    pos.push_back(0.5 + (i % 4) * 9.0);
  }
  const std::array<bool, 3> periodic{{true, true, true}};
  for (bool full : {false, true}) {
    rgpot::nlist::Query query;
    query.cutoff = 6.0;
    query.skin = 0;
    query.shape = rgpot::nlist::Pairs{full ? rgpot::nlist::Shell::Full
                                          : rgpot::nlist::Shell::Half};
    query.cut = rgpot::nlist::Cut::Strict;
    query.periodic = periodic;
    rgpot::nlist::Table table;
    rgpot::nlist::build(table, pos.data(), pos.size() / 3, box, query);
    REQUIRE(from_table(table) == brute(pos, box, 6.0, full, true, periodic));
  }
}

TEST_CASE("large general cells match the brute force", "[nlist][api]") {
  constexpr std::size_t n = 512;
  constexpr double cutoff = 3.0;
  const std::array<bool, 3> periodic{{true, true, true}};
  const std::array<std::array<double, 9>, 3> boxes{{
      {{15.0, 0.5, 0.2, 0.4, 15.5, 0.4, 0.2, 0.3, 16.0}},
      {{16.0, 0.0, 0.0, 8.0, 13.856406460551018, 0.0, 0.0, 0.0, 16.0}},
      {{14.142135623730951, 14.142135623730950, 0.0, -14.142135623730950,
        14.142135623730951, 0.0, 0.0, 0.0, 20.0}},
  }};

  for (const auto &box : boxes) {
    const auto pos = fractional_positions(box.data(), n);
    rgpot::nlist::Query query;
    query.cutoff = cutoff;
    query.shape = rgpot::nlist::Pairs{rgpot::nlist::Shell::Half};
    query.cut = rgpot::nlist::Cut::Strict;
    query.periodic = periodic;
    rgpot::nlist::Table table;
    rgpot::nlist::build(table, pos.data(), n, box.data(), query);
    const auto got = from_table(table);
    const auto ref = brute(pos, box.data(), cutoff, false, true, periodic, 1);
    INFO("builder " << got.size() << " brute " << ref.size());
    CHECK(got == ref);
    CHECK(ref.count(Key{0, 1, 0, 0, 0}) == 1);
  }
}

TEST_CASE("rotated and sheared cells with a non-positive diagonal match the brute force",
          "[nlist][api]") {
  // n = 512 bins the atoms. Each cell has a periodic axis whose diagonal
  // entry is zero or negative, which is not a period of that axis.
  struct Case {
    const char *name;
    std::array<double, 9> box;
    std::array<bool, 3> periodic;
  };
  const double c120 = -0.5;
  const double s120 = 0.86602540378443865;
  const std::array<Case, 5> cases{{
      {"cube rotated by 90 degrees", {{0, 15, 0, -15, 0, 0, 0, 0, 15}}, {{true, true, true}}},
      {"cube rotated by 120 degrees",
       {{15 * c120, 15 * s120, 0, -15 * s120, 15 * c120, 0, 0, 0, 15}},
       {{true, true, true}}},
      {"cube rotated by 180 degrees", {{-15, 0, 0, 0, -15, 0, 0, 0, 15}}, {{true, true, true}}},
      {"slab rotated by 90 degrees", {{0, 20, 0, -20, 0, 0, 0, 0, 18}}, {{true, true, false}}},
      {"sheared cell", {{-2, 15, 0, -15, -1, 0, 1, 2, 15}}, {{true, true, true}}},
  }};
  constexpr std::size_t n = 512;
  constexpr double cutoff = 3.0;
  for (const Case &c : cases) {
    INFO(c.name);
    const auto pos = fractional_positions(c.box.data(), n);
    const auto ref = brute(pos, c.box.data(), cutoff, false, true, c.periodic, 1);
    REQUIRE(ref.size() > 1000);
    rgpot::nlist::Query query;
    query.cutoff = cutoff;
    query.shape = rgpot::nlist::Pairs{rgpot::nlist::Shell::Half};
    query.cut = rgpot::nlist::Cut::Strict;
    query.periodic = c.periodic;
    rgpot::nlist::Table table;
    rgpot::nlist::build(table, pos.data(), n, c.box.data(), query);
    INFO("builder " << table.size() << " brute " << ref.size());
    CHECK(from_table(table) == ref);
    rgpot::nlist::Table by_atom;
    rgpot::nlist::buildByAtom(by_atom, pos.data(), n, c.box.data(), query);
    CHECK(from_table(by_atom) == ref);

    rgpot::nlist::Spec spec;
    spec.cutoff = cutoff;
    spec.periodic = c.periodic;
    spec.image = rgpot::nlist::Image::Round;
    spec.mode = rgpot::nlist::Mode::Stored;
    rgpot::nlist::Neighbours cells;
    cells.update(pos.data(), n, c.box.data(), spec);
    std::size_t within = 0;
    cells.forHalf(0, static_cast<std::int64_t>(n),
                  [&](std::int32_t, std::int32_t, double, double, double, double r2) {
                    within += r2 < cutoff * cutoff ? 1 : 0;
                  });
    CHECK(within == ref.size());
  }
}

TEST_CASE("a sheared mixed-periodic slab uses a real cell grid", "[nlist][api]") {
  constexpr std::size_t n = 600;
  constexpr double cutoff = 2.5;
  const double box[9] = {30, 0, 0, 7.5, 30, 0, 0, 0, 18};
  const std::array<bool, 3> periodic{{true, true, false}};
  const auto pos = fractional_positions(box, n);

  rgpot::nlist::Spec spec;
  spec.cutoff = cutoff;
  spec.periodic = periodic;
  spec.image = rgpot::nlist::Image::General;
  spec.mode = rgpot::nlist::Mode::Stream;
  rgpot::nlist::Neighbours cells;
  cells.update(pos.data(), n, box, spec);
  REQUIRE(cells.cellCount() > 1);

  rgpot::nlist::Query query;
  query.cutoff = cutoff;
  query.periodic = periodic;
  rgpot::nlist::Table table;
  rgpot::nlist::build(table, pos.data(), n, box, query);
  const auto expected = brute(pos, box, cutoff, false, true, periodic, 1);
  REQUIRE_FALSE(expected.empty());
  REQUIRE(from_table(table) == expected);
}

namespace {

// Moves atom a by offsets[3a + k] lattice vectors k.
std::vector<double> unwrap(const std::vector<double> &pos, const double *box,
                           const std::vector<int> &offsets) {
  std::vector<double> out = pos;
  for (std::size_t a = 0; a < pos.size() / 3; ++a) {
    for (int k = 0; k < 3; ++k) {
      for (int c = 0; c < 3; ++c) {
        out[3 * a + static_cast<std::size_t>(c)] +=
            offsets[3 * a + static_cast<std::size_t>(k)] * box[3 * k + c];
      }
    }
  }
  return out;
}

std::vector<int> random_offsets(std::size_t n, const std::array<bool, 3> &periodic,
                                std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<int> cells(-3, 3);
  std::vector<int> out(3 * n, 0);
  for (std::size_t a = 0; a < n; ++a) {
    for (std::size_t k = 0; k < 3; ++k) {
      out[3 * a + k] = periodic[k] ? cells(rng) : 0;
    }
  }
  return out;
}

void require_vectors(const rgpot::nlist::Table &table, const std::vector<double> &pos,
                     const double *box) {
  for (std::size_t p = 0; p < table.size(); ++p) {
    const auto i = static_cast<std::size_t>(table.first[p]);
    const auto j = static_cast<std::size_t>(table.second[p]);
    for (int c = 0; c < 3; ++c) {
      const double v = pos[3 * j + static_cast<std::size_t>(c)] -
                       pos[3 * i + static_cast<std::size_t>(c)] +
                       table.shift[3 * p] * box[c] + table.shift[3 * p + 1] * box[3 + c] +
                       table.shift[3 * p + 2] * box[6 + c];
      REQUIRE(std::abs(v - table.vector[3 * p + static_cast<std::size_t>(c)]) < 1e-12);
    }
  }
}

// The keys of `keys` after atom a moved by offsets[3a + k] lattice vectors
// k: a pair keeps its image when its shift gains W_i - W_j.
std::set<Key> shifted(const std::set<Key> &keys, const std::vector<int> &offsets) {
  std::set<Key> out;
  for (const Key &key : keys) {
    const auto i = static_cast<std::size_t>(3 * key.i);
    const auto j = static_cast<std::size_t>(3 * key.j);
    out.insert(Key{key.i, key.j, key.sa + offsets[i] - offsets[j],
                   key.sb + offsets[i + 1] - offsets[j + 1],
                   key.sc + offsets[i + 2] - offsets[j + 2]});
  }
  return out;
}

} // namespace

TEST_CASE("an atom several cells out keeps its pair", "[nlist][api]") {
  // 18.5 is 1.5 from the image of the origin two cells back. A stencil of
  // one image around zero sees 8.5 and drops the pair. The shift is the one
  // for the coordinates as given: 18.5 - 2 * 10 - 0 = -1.5.
  const double box[9] = {10, 0, 0, 0, 10, 0, 0, 0, 10};
  const std::vector<double> pos = {0, 0, 0, 18.5, 0, 0};
  rgpot::nlist::Query query;
  query.cutoff = 2.0;
  query.shape = rgpot::nlist::Pairs{rgpot::nlist::Shell::Half};
  query.cut = rgpot::nlist::Cut::Strict;
  rgpot::nlist::Table table;
  rgpot::nlist::build(table, pos.data(), 2, box, query);
  REQUIRE(table.size() == 1);
  REQUIRE(from_table(table) == std::set<Key>{Key{0, 1, -2, 0, 0}});
  REQUIRE(std::abs(table.vector[0] + 1.5) < 1e-12);
  REQUIRE(std::abs(table.distance[0] - 1.5) < 1e-12);

  rgpot::nlist::Walk walk(pos.data(), 2, box, query);
  std::vector<std::int32_t> first(4), second(4), shift(12);
  std::vector<double> vector(12), distance(4);
  std::vector<std::uint8_t> mask(4);
  bool done = false;
  rgpot::nlist::Buffers dest{first, second, shift, vector, distance, mask};
  REQUIRE(walk.pull(dest, done) == 1);
  REQUIRE(shift[0] == -2);

  query.skin = 0.5;
  query.cut = rgpot::nlist::Cut::Loose;
  rgpot::nlist::Builder builder;
  builder.update(pos.data(), 2, box, query);
  REQUIRE(from_table(builder.table()) == std::set<Key>{Key{0, 1, -2, 0, 0}});
}

TEST_CASE("unwrapped coordinates match the brute force", "[nlist][api]") {
  struct Case {
    std::array<double, 9> box;
    double cutoff;
    std::array<bool, 3> periodic;
  };
  // A wide cell with one image, a cell narrower than the cutoff, a sheared
  // cell, and a slab.
  const std::array<Case, 4> cases{{
      {{{12, 0, 0, 0, 11, 0, 0, 0, 13}}, 3.0, {{true, true, true}}},
      {{{4, 0, 0, 0, 4.5, 0, 0, 0, 5}}, 3.2, {{true, true, true}}},
      {{{6.0, 0.4, 0.2, 0.8, 6.5, -0.3, 0.5, 0.6, 5.5}}, 3.4, {{true, true, true}}},
      {{{7, 0.5, 0, 0.3, 6, 0, 0, 0, 9}}, 3.0, {{true, true, false}}},
  }};
  constexpr std::size_t n = 24;
  for (const Case &c : cases) {
    const auto wrapped = fractional_positions(c.box.data(), n);
    const auto offsets = random_offsets(n, c.periodic, 17);
    const auto pos = unwrap(wrapped, c.box.data(), offsets);
    for (bool full : {false, true}) {
      rgpot::nlist::Query query;
      query.cutoff = c.cutoff;
      query.shape =
          rgpot::nlist::Pairs{full ? rgpot::nlist::Shell::Full : rgpot::nlist::Shell::Half};
      query.cut = rgpot::nlist::Cut::Strict;
      query.periodic = c.periodic;
      rgpot::nlist::Table table;
      rgpot::nlist::build(table, pos.data(), n, c.box.data(), query);
      const auto ref = brute(pos, c.box.data(), c.cutoff, full, true, c.periodic, 10);
      REQUIRE_FALSE(ref.empty());
      REQUIRE(from_table(table) == ref);
      require_vectors(table, pos, c.box.data());
    }
  }
}

TEST_CASE("large unwrapped systems match the wrapped list", "[nlist][api]") {
  struct Case {
    std::array<double, 9> box;
    double cutoff;
  };
  // Past the direct loop, so the cells bin the atoms. The wide cell takes
  // the one-image walk; the narrow and sheared ones try several shifts.
  const std::array<Case, 3> cases{{
      {{{20, 0, 0, 0, 21, 0, 0, 0, 22}}, 3.0},
      {{{9, 0, 0, 0, 9.5, 0, 0, 0, 10}}, 5.0},
      {{{16.0, 0.5, 0.2, 0.4, 15.5, 0.4, 0.2, 0.3, 16.0}}, 3.0},
  }};
  constexpr std::size_t n = 700;
  const std::array<bool, 3> periodic{{true, true, true}};
  for (const Case &c : cases) {
    const auto wrapped = fractional_positions(c.box.data(), n);
    const auto offsets = random_offsets(n, periodic, 29);
    const auto pos = unwrap(wrapped, c.box.data(), offsets);
    for (bool full : {false, true}) {
      rgpot::nlist::Query query;
      query.cutoff = c.cutoff;
      query.shape =
          rgpot::nlist::Pairs{full ? rgpot::nlist::Shell::Full : rgpot::nlist::Shell::Half};
      query.cut = rgpot::nlist::Cut::Strict;
      query.periodic = periodic;
      rgpot::nlist::Table in_cell;
      rgpot::nlist::build(in_cell, wrapped.data(), n, c.box.data(), query);
      const auto expect = shifted(from_table(in_cell), offsets);
      REQUIRE(expect.size() > n);

      rgpot::nlist::Table table;
      rgpot::nlist::build(table, pos.data(), n, c.box.data(), query);
      REQUIRE(from_table(table) == expect);
      require_vectors(table, pos, c.box.data());

      rgpot::nlist::Table by_atom;
      rgpot::nlist::buildByAtom(by_atom, pos.data(), n, c.box.data(), query);
      REQUIRE(from_table(by_atom) == expect);
    }
    rgpot::nlist::Query query;
    query.cutoff = c.cutoff;
    query.shape = rgpot::nlist::Cluster4{};
    query.cut = rgpot::nlist::Cut::Strict;
    query.periodic = periodic;
    rgpot::nlist::Table clustered;
    rgpot::nlist::build(clustered, pos.data(), n, c.box.data(), query);
    query.shape = rgpot::nlist::Pairs{rgpot::nlist::Shell::Full};
    rgpot::nlist::Table in_cell;
    rgpot::nlist::build(in_cell, wrapped.data(), n, c.box.data(), query);
    REQUIRE(from_table(clustered) == shifted(from_table(in_cell), offsets));
  }
}

TEST_CASE("whole-cell drift keeps the pairs and moves their shifts", "[nlist][api]") {
  const std::vector<double> wrapped = {
      0.4, 0.5, 0.6, 1.2, 2.0, 1.1, 2.5, 5.7, 6.8,
  };
  const std::array<bool, 3> periodic{{true, true, true}};
  const std::array<std::array<double, 9>, 2> boxes{{
      {{5, 0, 0, 0, 6, 0, 0, 0, 7}},
      {{5, 0, 0, 0.7, 6, 0, -0.4, 0.3, 7}},
  }};
  rgpot::nlist::Query query;
  query.cutoff = 6.0;
  query.cut = rgpot::nlist::Cut::Loose;
  query.periodic = periodic;
  for (std::size_t cell = 0; cell < boxes.size(); ++cell) {
    const auto &box = boxes[cell];
    rgpot::nlist::Table reference;
    rgpot::nlist::build(reference, wrapped.data(), wrapped.size() / 3, box.data(), query);
    const auto expected = from_table(reference);
    REQUIRE_FALSE(expected.empty());
    if (cell == 0) {
      REQUIRE(expected.size() == 16);
    }
    for (int drift : {0, 1, 2, 3, -3}) {
      std::vector<int> offsets(wrapped.size(), 0);
      offsets[0] = drift;
      offsets[1] = drift;
      offsets[2] = drift;
      const auto pos = unwrap(wrapped, box.data(), offsets);
      rgpot::nlist::Table table;
      rgpot::nlist::build(table, pos.data(), pos.size() / 3, box.data(), query);
      INFO("cell " << cell << " drift " << drift);
      REQUIRE(from_table(table) == shifted(expected, offsets));
    }

    query.skin = 0.5;
    rgpot::nlist::Builder builder;
    builder.update(wrapped.data(), wrapped.size() / 3, box.data(), query);
    const auto buffered = from_table(builder.table());
    std::vector<int> offsets(wrapped.size(), 0);
    offsets[0] = 3;
    offsets[1] = 3;
    offsets[2] = 3;
    const auto drifted = unwrap(wrapped, box.data(), offsets);
    builder.update(drifted.data(), drifted.size() / 3, box.data(), query);
    REQUIRE(builder.reused());
    REQUIRE(builder.builds() == 1);
    REQUIRE(from_table(builder.table()) == shifted(buffered, offsets));
    query.skin = 0.0;
  }
}

TEST_CASE("a coordinate too many cells out is rejected", "[nlist][api][edge]") {
  const double box[9] = {10, 0, 0, 0, 10, 0, 0, 0, 10};
  const std::vector<double> pos = {0, 0, 0, 1.0e12, 0, 0};
  rgpot::nlist::Query query;
  query.cutoff = 2.0;
  rgpot::nlist::Table table;
  try {
    rgpot::nlist::build(table, pos.data(), 2, box, query);
    FAIL("a coordinate 1e11 cells out was accepted");
  } catch (const rgpot::nlist::Error &err) {
    REQUIRE(err.code() == rgpot::nlist::Code::Box);
    REQUIRE(std::string(err.what()).find("2^29 cells") != std::string::npos);
  }
}

TEST_CASE("a coordinate too many cells out keeps the previous table", "[nlist][api][edge]") {
  const double box[9] = {12, 0, 0, 0, 12, 0, 0, 0, 12};
  std::vector<double> pos = fractional_positions(box, 600);
  rgpot::nlist::Query query;
  query.cutoff = 2.0;
  for (const bool by_atom : {false, true}) {
    INFO("buildByAtom " << by_atom);
    rgpot::nlist::Table table;
    rgpot::nlist::build(table, pos.data(), pos.size() / 3, box, query);
    const auto rows = from_table(table);
    REQUIRE(!rows.empty());
    std::vector<double> far = pos;
    far[3 * 599] = 1.0e12;
    if (by_atom) {
      REQUIRE_THROWS_AS(rgpot::nlist::buildByAtom(table, far.data(), far.size() / 3, box, query),
                        rgpot::nlist::Error);
    } else {
      REQUIRE_THROWS_AS(rgpot::nlist::build(table, far.data(), far.size() / 3, box, query),
                        rgpot::nlist::Error);
    }
    REQUIRE(table.stored);
    REQUIRE(from_table(table) == rows);
  }
  rgpot::nlist::Builder builder;
  builder.update(pos.data(), pos.size() / 3, box, query);
  const auto rows = from_table(builder.table());
  pos[3 * 599] = 1.0e12;
  REQUIRE_THROWS_AS(builder.update(pos.data(), pos.size() / 3, box, query), rgpot::nlist::Error);
  REQUIRE(from_table(builder.table()) == rows);
}

TEST_CASE("a large orthorhombic list is not an all-pairs scan", "[nlist][api][perf]") {
  // 30^3 FCC cells, the 108000-atom copper block. An all-pairs scan does
  // not finish in tens of seconds. A cell list does. macOS CI measured
  // 5.4s, so the bound stays well under an all-pairs run, and each atom
  // has the twelve, six and twenty-four shells inside 5.
  constexpr int ncell = 30;
  constexpr double lattice = 3.615;
  const double basis[4][3] = {{0, 0, 0}, {0, 0.5, 0.5}, {0.5, 0, 0.5}, {0.5, 0.5, 0}};
  std::vector<double> pos;
  pos.reserve(static_cast<std::size_t>(ncell) * ncell * ncell * 4 * 3);
  for (int ix = 0; ix < ncell; ++ix) {
    for (int iy = 0; iy < ncell; ++iy) {
      for (int iz = 0; iz < ncell; ++iz) {
        for (const auto &b : basis) {
          pos.push_back((ix + b[0]) * lattice);
          pos.push_back((iy + b[1]) * lattice);
          pos.push_back((iz + b[2]) * lattice);
        }
      }
    }
  }
  const double edge = ncell * lattice;
  const double box[9] = {edge, 0, 0, 0, edge, 0, 0, 0, edge};
  rgpot::nlist::Query query;
  query.cutoff = 5.0;
  query.skin = 0;
  query.shape = rgpot::nlist::Pairs{rgpot::nlist::Shell::Half};
  query.cut = rgpot::nlist::Cut::Strict;
  rgpot::nlist::Table table;
  const auto t0 = std::chrono::steady_clock::now();
  rgpot::nlist::build(table, pos.data(), pos.size() / 3, box, query);
  const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  REQUIRE(pos.size() / 3 == 108000);
  REQUIRE(table.size() == 108000 * 21);
  REQUIRE(seconds < 30.0);
  for (std::size_t p = 0; p < 64; ++p) {
    const int i = table.first[p];
    const int j = table.second[p];
    const double vx = pos[3 * j] - pos[3 * i] + table.shift[3 * p] * box[0];
    const double vy = pos[3 * j + 1] - pos[3 * i + 1] + table.shift[3 * p + 1] * box[4];
    const double vz = pos[3 * j + 2] - pos[3 * i + 2] + table.shift[3 * p + 2] * box[8];
    REQUIRE(std::abs(vx - table.vector[3 * p]) < 1e-9);
    REQUIRE(std::abs(vy - table.vector[3 * p + 1]) < 1e-9);
    REQUIRE(std::abs(vz - table.vector[3 * p + 2]) < 1e-9);
    REQUIRE(std::abs(std::sqrt(vx * vx + vy * vy + vz * vz) - table.distance[p]) < 1e-9);
    REQUIRE(table.distance[p] < 5.0);
    REQUIRE(table.distance[p] > 0.0);
  }
}

TEST_CASE("a skin rebuild keeps the same pairs", "[nlist][api]") {
  const double box[9] = {8, 0, 0, 0, 8, 0, 0, 0, 8};
  std::vector<double> pos = {1, 1, 1, 2, 1, 1, 1, 3, 1};
  rgpot::nlist::Query query;
  query.cutoff = 2.5;
  query.skin = 0.4;
  query.shape = rgpot::nlist::Pairs{rgpot::nlist::Shell::Half};
  query.cut = rgpot::nlist::Cut::Loose;
  rgpot::nlist::Builder builder;
  builder.update(pos.data(), pos.size() / 3, box, query);
  const auto first = builder.builds();
  pos[0] += 0.05;
  builder.update(pos.data(), pos.size() / 3, box, query);
  REQUIRE(builder.reused());
  REQUIRE(builder.builds() == first);
}

TEST_CASE("a builder folds the positions once per update", "[nlist][api]") {
  double box[9] = {10, 0, 0, 0, 10, 0, 0, 0, 10};
  std::vector<double> pos = fractional_positions(box, 600);
  rgpot::nlist::Query query;
  query.cutoff = 2.5;
  query.skin = 0.5;
  query.cut = rgpot::nlist::Cut::Loose;
  for (const bool cluster : {false, true}) {
    INFO("cluster " << cluster);
    query.shape = cluster ? rgpot::nlist::Shape{rgpot::nlist::Cluster4{}}
                          : rgpot::nlist::Shape{rgpot::nlist::Pairs{}};
    rgpot::nlist::Builder builder;
    REQUIRE(builder.frames() == 0);
    std::uint64_t updates = 0;
    auto update = [&]() {
      builder.update(pos.data(), pos.size() / 3, box, query);
      ++updates;
      REQUIRE(builder.frames() == updates);
    };
    update();
    REQUIRE(builder.builds() == 1);
    update();
    REQUIRE(builder.unchanged());
    pos[0] += 0.1;
    update();
    REQUIRE(builder.reused());
    pos[0] += 1.0;
    update();
    REQUIRE(builder.builds() == 2);
    box[0] = 10.001;
    update();
    REQUIRE(builder.reused());
    box[0] = 12.0;
    update();
    REQUIRE(builder.builds() == 3);
    for (int c = 0; c < 3; ++c) {
      pos[3 * 7 + static_cast<std::size_t>(c)] += 3.0 * box[3 * c + c];
    }
    update();
    REQUIRE(builder.builds() == (cluster ? 4 : 3));
    query.cutoff = 2.0;
    update();
    REQUIRE(builder.builds() == (cluster ? 5 : 4));
    query.cutoff = -1.0;
    REQUIRE_THROWS_AS(builder.update(pos.data(), pos.size() / 3, box, query),
                      rgpot::nlist::Error);
    REQUIRE(builder.frames() == updates);
    query.cutoff = 2.5;
    const rgpot::nlist::Builder copy = builder;
    REQUIRE(copy.frames() == updates);
    rgpot::nlist::Builder moved = std::move(builder);
    REQUIRE(moved.frames() == updates);
    pos[0] -= 1.1;
    box[0] = 10.0;
  }
}

TEST_CASE("the skin rebuild follows half the skin", "[nlist][api]") {
  const double box[9] = {10, 0, 0, 0, 10, 0, 0, 0, 10};
  // The first atom sits at the origin so a displacement of one ulp past
  // half the skin is representable. Adding that ulp to a coordinate near
  // 1 rounds back to the exact half-skin position.
  const std::vector<double> origin = {0, 0, 0, 2.2, 1, 1, 1, 3.4, 1, 4, 4, 1};
  const double skin = 0.5;
  const double half = 0.5 * skin;
  rgpot::nlist::Query query;
  query.cutoff = 3.0;
  query.skin = skin;
  query.shape = rgpot::nlist::Pairs{rgpot::nlist::Shell::Full};
  query.cut = rgpot::nlist::Cut::Loose;
  query.fields = {true, true, true};

  rgpot::nlist::Builder edge;
  auto pos = origin;
  edge.update(pos.data(), pos.size() / 3, box, query);
  REQUIRE(edge.builds() == 1);
  pos[0] = half;
  edge.update(pos.data(), pos.size() / 3, box, query);
  REQUIRE(edge.reused());
  REQUIRE(edge.builds() == 1);
  pos[0] = std::nextafter(half, 1.0);
  edge.update(pos.data(), pos.size() / 3, box, query);
  REQUIRE_FALSE(edge.reused());
  REQUIRE(edge.builds() == 2);

  rgpot::nlist::Builder builder;
  pos = origin;
  builder.update(pos.data(), pos.size() / 3, box, query);
  pos[0] = 0.5 * half;
  builder.update(pos.data(), pos.size() / 3, box, query);
  REQUIRE(builder.reused());
  const auto &kept = builder.table();
  rgpot::nlist::Table fresh;
  rgpot::nlist::build(fresh, pos.data(), pos.size() / 3, box, query);
  REQUIRE(kept.size() == fresh.size());

  auto key_of = [](const rgpot::nlist::Table &table, std::size_t p) {
    return std::tuple{table.first[p], table.second[p], table.shift[3 * p],
                      table.shift[3 * p + 1], table.shift[3 * p + 2]};
  };
  std::map<std::tuple<int, int, int, int, int>, std::array<double, 4>> got;
  for (std::size_t p = 0; p < kept.size(); ++p) {
    const int i = kept.first[p];
    const int j = kept.second[p];
    const int sa = kept.shift[3 * p];
    const int sb = kept.shift[3 * p + 1];
    const int sc = kept.shift[3 * p + 2];
    const double vx = pos[3 * static_cast<std::size_t>(j)] - pos[3 * static_cast<std::size_t>(i)] +
                      sa * box[0] + sb * box[3] + sc * box[6];
    const double vy = pos[3 * static_cast<std::size_t>(j) + 1] -
                      pos[3 * static_cast<std::size_t>(i) + 1] + sa * box[1] + sb * box[4] +
                      sc * box[7];
    const double vz = pos[3 * static_cast<std::size_t>(j) + 2] -
                      pos[3 * static_cast<std::size_t>(i) + 2] + sa * box[2] + sb * box[5] +
                      sc * box[8];
    got[{i, j, sa, sb, sc}] = {vx, vy, vz, std::sqrt(vx * vx + vy * vy + vz * vz)};
  }
  for (std::size_t p = 0; p < fresh.size(); ++p) {
    const auto key = key_of(fresh, p);
    REQUIRE(got.count(key) == 1);
    REQUIRE(std::abs(got[key][0] - fresh.vector[3 * p]) < 1e-12);
    REQUIRE(std::abs(got[key][1] - fresh.vector[3 * p + 1]) < 1e-12);
    REQUIRE(std::abs(got[key][2] - fresh.vector[3 * p + 2]) < 1e-12);
    REQUIRE(std::abs(got[key][3] - fresh.distance[p]) < 1e-12);
  }
}

TEST_CASE("a small cell change reuses the builder and a large change rebuilds",
          "[nlist][api]") {
  const double box[9] = {8.0, 0.2, 0.0, 0.1, 8.5, 0.15, 0.0, -0.1, 7.8};
  std::vector<double> pos = {
      0.4, 0.5, 0.6, 2.1, 0.8, 0.7, 7.5, 0.6, 0.5,
      1.2, 3.0, 1.1, 3.3, 3.1, 1.4, 0.8, 7.9, 7.2,
  };
  const std::size_t n = pos.size() / 3;
  const double cutoff = 3.2;
  const double skin = 0.5;
  const std::array<bool, 3> periodic{{true, true, true}};

  rgpot::nlist::Query query;
  query.cutoff = cutoff;
  query.skin = skin;
  query.shape = rgpot::nlist::Pairs{rgpot::nlist::Shell::Half};
  query.cut = rgpot::nlist::Cut::Loose;
  query.periodic = periodic;
  query.fields = {true, false, false};

  auto filter_strict = [&](const rgpot::nlist::Table &table, const std::vector<double> &p,
                           const double *b) {
    std::set<Key> out;
    const double cut2 = cutoff * cutoff;
    for (std::size_t k = 0; k < table.size(); ++k) {
      const int i = table.first[k];
      const int j = table.second[k];
      const int sa = table.shift[3 * k];
      const int sb = table.shift[3 * k + 1];
      const int sc = table.shift[3 * k + 2];
      const double vx = p[3 * static_cast<std::size_t>(j)] -
                        p[3 * static_cast<std::size_t>(i)] +
                        sa * b[0] + sb * b[3] + sc * b[6];
      const double vy = p[3 * static_cast<std::size_t>(j) + 1] -
                        p[3 * static_cast<std::size_t>(i) + 1] +
                        sa * b[1] + sb * b[4] + sc * b[7];
      const double vz = p[3 * static_cast<std::size_t>(j) + 2] -
                        p[3 * static_cast<std::size_t>(i) + 2] +
                        sa * b[2] + sb * b[5] + sc * b[8];
      const double r2 = vx * vx + vy * vy + vz * vz;
      if (r2 > 0.0 && r2 < cut2) {
        out.insert(Key{i, j, sa, sb, sc});
      }
    }
    return out;
  };

  rgpot::nlist::Builder builder;
  builder.update(pos.data(), n, box, query);
  REQUIRE_FALSE(builder.reused());
  REQUIRE_FALSE(builder.unchanged());
  REQUIRE(builder.builds() == 1);
  REQUIRE(filter_strict(builder.table(), pos, box) ==
          brute(pos, box, cutoff, false, true, periodic));

  builder.update(pos.data(), n, box, query);
  REQUIRE(builder.reused());
  REQUIRE(builder.unchanged());
  REQUIRE(builder.builds() == 1);

  const double small_box[9] = {8.03, 0.21, 0.0, 0.1, 8.48, 0.15, 0.0, -0.09, 7.82};
  for (double &x : pos) {
    x += 0.02;
  }
  const auto t0 = std::chrono::steady_clock::now();
  builder.update(pos.data(), n, small_box, query);
  const auto elapsed =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  REQUIRE(elapsed < 1.0);
  REQUIRE(builder.reused());
  REQUIRE_FALSE(builder.unchanged());
  REQUIRE(builder.builds() == 1);
  REQUIRE(filter_strict(builder.table(), pos, small_box) ==
          brute(pos, small_box, cutoff, false, true, periodic));

  const double large_box[9] = {8.6, 0.2, 0.0, 0.1, 9.1, 0.15, 0.0, -0.1, 8.4};
  builder.update(pos.data(), n, large_box, query);
  REQUIRE_FALSE(builder.reused());
  REQUIRE_FALSE(builder.unchanged());
  REQUIRE(builder.builds() == 2);
  REQUIRE(filter_strict(builder.table(), pos, large_box) ==
          brute(pos, large_box, cutoff, false, true, periodic));
}

TEST_CASE("mixed-sign cell deformation keeps the same pairs as a fresh build",
          "[nlist][api][edge]") {
  const double ref_box[9] = {10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0};
  const double deformed_box[9] = {9.4, 0.0, 0.0, 0.6, 10.0, 0.0, 0.0, 0.0, 10.0};
  const std::vector<double> pos = {9.332, 0.0, 0.0, 3.5, 10.0, 0.0};
  rgpot::nlist::Query query;
  query.cutoff = 3.0;
  query.skin = 1.0;
  query.shape = rgpot::nlist::Pairs{rgpot::nlist::Shell::Half};
  query.cut = rgpot::nlist::Cut::Loose;
  query.periodic = {{true, true, true}};
  query.fields = {true, true, true};

  rgpot::nlist::Builder builder;
  builder.update(pos.data(), pos.size() / 3, ref_box, query);
  builder.update(pos.data(), pos.size() / 3, deformed_box, query);
  const auto reused_view = builder.table().view();

  rgpot::nlist::Table fresh;
  rgpot::nlist::build(fresh, pos.data(), pos.size() / 3, deformed_box, query);
  const auto fresh_view = fresh.view();
  REQUIRE(from_table(builder.table()) == from_table(fresh));
  REQUIRE(reused_view.first.size() == fresh_view.first.size());

  bool found = false;
  for (std::size_t p = 0; p < reused_view.first.size(); ++p) {
    if (reused_view.first[p] == 0 && reused_view.second[p] == 1 &&
        reused_view.shift[3 * p] == 1 && reused_view.shift[3 * p + 1] == -1 &&
        reused_view.shift[3 * p + 2] == 0) {
      found = true;
      const double x = pos[3] - pos[0] +
                       static_cast<double>(reused_view.shift[3 * p]) * deformed_box[0] +
                       static_cast<double>(reused_view.shift[3 * p + 1]) * deformed_box[3];
      const double y = pos[4] - pos[1] +
                       static_cast<double>(reused_view.shift[3 * p]) * deformed_box[1] +
                       static_cast<double>(reused_view.shift[3 * p + 1]) * deformed_box[4];
      const double z = pos[5] - pos[2] +
                       static_cast<double>(reused_view.shift[3 * p]) * deformed_box[2] +
                       static_cast<double>(reused_view.shift[3 * p + 1]) * deformed_box[5];
      REQUIRE(std::sqrt(x * x + y * y + z * z) == Catch::Approx(2.968).margin(1e-12));
      break;
    }
  }
  REQUIRE(found);
}

TEST_CASE("opposite-sign cell changes match a brute force pair set", "[nlist][api]") {
  // Orthorhombic 5 x 6 x 7. a_y grows by 0.5 and b_y shrinks by 0.5.
  // The image (2, -2, 0) of pair (0, 1) then moves by 2, past skin 1,
  // while every positive corner of the shift box moves by at most 1.
  // cutoff + skin is 9.5 and each height is at least 5, so the stored
  // shift span is 2 on every axis.
  const double box[9] = {5, 0, 0, 0, 6, 0, 0, 0, 7};
  const double sheared[9] = {5, 0.5, 0, 0, 5.5, 0, 0, 0, 7};
  const double mild[9] = {5, 0.02, 0, 0, 5.98, 0, 0, 0, 7};
  const std::vector<double> pos = {4.0, 0.2, 0.2, 0.2, 5.0, 0.2};
  const std::size_t n = pos.size() / 3;
  const double cutoff = 8.5;
  const double skin = 1.0;
  const double span[3] = {2, 2, 2};
  const std::array<bool, 3> periodic{{true, true, true}};

  rgpot::nlist::Query query;
  query.cutoff = cutoff;
  query.skin = skin;
  query.shape = rgpot::nlist::Pairs{rgpot::nlist::Shell::Half};
  query.cut = rgpot::nlist::Cut::Loose;
  query.periodic = periodic;

  auto filter_strict = [&](const rgpot::nlist::Table &table, const double *b) {
    std::set<Key> out;
    const double cut2 = cutoff * cutoff;
    for (std::size_t k = 0; k < table.size(); ++k) {
      const int i = table.first[k];
      const int j = table.second[k];
      const int sa = table.shift[3 * k];
      const int sb = table.shift[3 * k + 1];
      const int sc = table.shift[3 * k + 2];
      const double vx = pos[3 * static_cast<std::size_t>(j)] -
                        pos[3 * static_cast<std::size_t>(i)] + sa * b[0] + sb * b[3] +
                        sc * b[6];
      const double vy = pos[3 * static_cast<std::size_t>(j) + 1] -
                        pos[3 * static_cast<std::size_t>(i) + 1] + sa * b[1] + sb * b[4] +
                        sc * b[7];
      const double vz = pos[3 * static_cast<std::size_t>(j) + 2] -
                        pos[3 * static_cast<std::size_t>(i) + 2] + sa * b[2] + sb * b[5] +
                        sc * b[8];
      const double r2 = vx * vx + vy * vy + vz * vz;
      if (r2 > 0.0 && r2 < cut2) {
        out.insert(Key{i, j, sa, sb, sc});
      }
    }
    return out;
  };

  rgpot::nlist::Builder builder;
  builder.update(pos.data(), n, box, query);
  REQUIRE_FALSE(builder.reused());
  REQUIRE(filter_strict(builder.table(), box) ==
          brute(pos, box, cutoff, false, true, periodic));

  builder.update(pos.data(), n, mild, query);
  REQUIRE(builder.reused());
  REQUIRE_FALSE(builder.unchanged());
  REQUIRE(filter_strict(builder.table(), mild) ==
          brute(pos, mild, cutoff, false, true, periodic));

  builder.update(pos.data(), n, sheared, query);
  const auto got = filter_strict(builder.table(), sheared);
  const auto ref =
      brute(pos, sheared, cutoff, false, true, periodic);
  INFO("reused " << builder.reused() << " builder " << got.size() << " brute " << ref.size());
  REQUIRE(got == ref);
  REQUIRE(ref.size() == 23);
  REQUIRE_FALSE(builder.reused());
  REQUIRE_FALSE(rgpot::nlist::withinSkin(pos.data(), pos.data(), n, box, sheared, span, skin));
  REQUIRE(rgpot::nlist::withinSkin(pos.data(), pos.data(), n, box, mild, span, skin));
}

TEST_CASE("random cell perturbations match a brute force pair set", "[nlist][api]") {
  const double box0[9] = {5, 0, 0, 0, 6, 0, 0, 0, 7};
  const std::vector<double> pos = {0.4, 0.5, 0.6, 1.2, 2.0, 1.1, 3.3, 4.1, 2.4, 4.6, 0.8, 5.5};
  const std::size_t n = pos.size() / 3;
  const double cutoff = 4.0;
  const double skin = 0.8;
  const std::array<bool, 3> periodic{{true, true, true}};

  rgpot::nlist::Query query;
  query.cutoff = cutoff;
  query.skin = skin;
  query.shape = rgpot::nlist::Pairs{rgpot::nlist::Shell::Half};
  query.cut = rgpot::nlist::Cut::Loose;
  query.periodic = periodic;

  auto filter_strict = [&](const rgpot::nlist::Table &table, const double *b) {
    std::set<Key> out;
    const double cut2 = cutoff * cutoff;
    for (std::size_t k = 0; k < table.size(); ++k) {
      const int i = table.first[k];
      const int j = table.second[k];
      const int sa = table.shift[3 * k];
      const int sb = table.shift[3 * k + 1];
      const int sc = table.shift[3 * k + 2];
      const double vx = pos[3 * static_cast<std::size_t>(j)] -
                        pos[3 * static_cast<std::size_t>(i)] + sa * b[0] + sb * b[3] +
                        sc * b[6];
      const double vy = pos[3 * static_cast<std::size_t>(j) + 1] -
                        pos[3 * static_cast<std::size_t>(i) + 1] + sa * b[1] + sb * b[4] +
                        sc * b[7];
      const double vz = pos[3 * static_cast<std::size_t>(j) + 2] -
                        pos[3 * static_cast<std::size_t>(i) + 2] + sa * b[2] + sb * b[5] +
                        sc * b[8];
      const double r2 = vx * vx + vy * vy + vz * vz;
      if (r2 > 0.0 && r2 < cut2) {
        out.insert(Key{i, j, sa, sb, sc});
      }
    }
    return out;
  };

  auto volume = [](const double *b) {
    const double bx = b[4] * b[8] - b[5] * b[7];
    const double by = b[5] * b[6] - b[3] * b[8];
    const double bz = b[3] * b[7] - b[4] * b[6];
    return std::fabs(b[0] * bx + b[1] * by + b[2] * bz);
  };

  std::mt19937 rng(1);
  std::uniform_real_distribution<double> delta(-0.25, 0.25);
  int done = 0;
  for (int trial = 0; trial < 32; ++trial) {
    double box[9];
    for (int k = 0; k < 9; ++k) {
      box[k] = box0[k];
    }
    if (trial % 4 == 0) {
      const double step = 0.05 + 0.03 * static_cast<double>(trial % 5);
      box[1] += step;
      box[4] -= step;
    } else {
      for (int k = 0; k < 9; ++k) {
        box[k] += delta(rng);
      }
    }
    if (!(volume(box) > 1.0)) {
      continue;
    }
    rgpot::nlist::Builder builder;
    builder.update(pos.data(), n, box0, query);
    builder.update(pos.data(), n, box, query);
    const auto got = filter_strict(builder.table(), box);
    const auto ref =
        brute(pos, box, cutoff, false, true, periodic, 4);
    INFO("trial " << trial << " builder " << got.size() << " brute " << ref.size()
                  << " reused " << builder.reused());
    REQUIRE(got == ref);
    ++done;
  }
  REQUIRE(done >= 24);
}

TEST_CASE("a builder on unwrapped coordinates matches a brute force pair set",
          "[nlist][api]") {
  const double box0[9] = {5, 0.3, 0, 0.2, 6, 0, 0, -0.4, 7};
  const std::vector<double> start = {0.4, 0.5, 0.6, 1.2, 2.0, 1.1, 3.3, 4.1, 2.4, 4.6, 0.8, 5.5,
                                     2.5, 5.7, 6.8, 4.9, 3.0, 0.1};
  const std::size_t n = start.size() / 3;
  const std::array<bool, 3> periodic{{true, true, true}};
  const std::vector<int> offsets = {2, 0, -1, -3, 1, 0, 0, 0, 0, 1, -2, 3, 0, 4, 0, -1, -1, -1};
  std::vector<double> pos = unwrap(start, box0, offsets);
  const double cutoff = 3.5;
  rgpot::nlist::Query query;
  query.cutoff = cutoff;
  query.skin = 0.6;
  query.shape = rgpot::nlist::Pairs{rgpot::nlist::Shell::Half};
  query.cut = rgpot::nlist::Cut::Loose;
  query.periodic = periodic;
  query.fields = {true, false, false};

  auto strict_keys = [&](const rgpot::nlist::Table &table, const double *b) {
    std::set<Key> out;
    for (std::size_t k = 0; k < table.size(); ++k) {
      const auto i = static_cast<std::size_t>(table.first[k]);
      const auto j = static_cast<std::size_t>(table.second[k]);
      const int s[3] = {table.shift[3 * k], table.shift[3 * k + 1], table.shift[3 * k + 2]};
      double r2 = 0.0;
      for (int c = 0; c < 3; ++c) {
        const double v = pos[3 * j + static_cast<std::size_t>(c)] -
                         pos[3 * i + static_cast<std::size_t>(c)] + s[0] * b[c] +
                         s[1] * b[3 + c] + s[2] * b[6 + c];
        r2 += v * v;
      }
      if (r2 > 0.0 && r2 < cutoff * cutoff) {
        out.insert(Key{table.first[k], table.second[k], s[0], s[1], s[2]});
      }
    }
    return out;
  };

  // Small steps that carry atoms across faces, and a cell that drifts.
  std::mt19937 rng(5);
  std::uniform_real_distribution<double> step(-0.12, 0.12);
  rgpot::nlist::Builder builder;
  double box[9];
  std::copy(box0, box0 + 9, box);
  int reused = 0;
  for (int t = 0; t < 40; ++t) {
    builder.update(pos.data(), n, box, query);
    reused += builder.reused() ? 1 : 0;
    INFO("step " << t << " reused " << builder.reused());
    REQUIRE(strict_keys(builder.table(), box) ==
            brute(pos, box, cutoff, false, true, periodic, 12));
    for (double &x : pos) {
      x += step(rng);
    }
    pos[0] += 0.1;
    for (int k : {0, 4, 8}) {
      box[k] += 0.25 * step(rng);
    }
    box[1] += 0.1 * step(rng);
  }
  REQUIRE(reused > 0);
}

TEST_CASE("a step across a cell face keeps the list", "[nlist][api]") {
  // Atom 0 steps from just inside the +x face to just outside it. A caller
  // that wraps puts it back near x = 0, one that does not leaves it past
  // the face. Either way the atom has moved by 0.1, well inside the skin.
  const double box[9] = {10, 0, 0, 1.5, 10, 0, 0, 0, 10};
  const std::array<bool, 3> periodic{{true, true, true}};
  constexpr double cutoff = 3.0;
  auto pos = fractional_positions(box, 64);
  pos[0] = 9.95;
  pos[1] = 0.0;
  pos[2] = 5.0;
  auto stepped = pos;
  stepped[0] = 10.05;
  auto wrapped = stepped;
  wrapped[0] -= 10.0;

  auto strict_keys = [&](const rgpot::nlist::Table &table, const std::vector<double> &at) {
    std::set<Key> out;
    for (std::size_t k = 0; k < table.size(); ++k) {
      const auto i = static_cast<std::size_t>(table.first[k]);
      const auto j = static_cast<std::size_t>(table.second[k]);
      const int sh[3] = {table.shift[3 * k], table.shift[3 * k + 1], table.shift[3 * k + 2]};
      double r2 = 0.0;
      for (int c = 0; c < 3; ++c) {
        const double v = at[3 * j + static_cast<std::size_t>(c)] -
                         at[3 * i + static_cast<std::size_t>(c)] + sh[0] * box[c] +
                         sh[1] * box[3 + c] + sh[2] * box[6 + c];
        r2 += v * v;
      }
      if (r2 < cutoff * cutoff) {
        out.insert(Key{table.first[k], table.second[k], sh[0], sh[1], sh[2]});
      }
    }
    return out;
  };

  for (bool cluster : {false, true}) {
    for (const auto *next : {&stepped, &wrapped}) {
      INFO("cluster " << cluster << " wrapped " << (next == &wrapped));
      rgpot::nlist::Query query;
      query.cutoff = cutoff;
      query.skin = 1.0;
      query.cut = rgpot::nlist::Cut::Loose;
      query.periodic = periodic;
      if (cluster) {
        query.shape = rgpot::nlist::Cluster4{};
      }
      rgpot::nlist::Builder builder;
      builder.update(pos.data(), 64, box, query);
      builder.update(next->data(), 64, box, query);
      // A cluster row shares one shift among four atoms, so a wrapped atom
      // changes the shift of its lane alone and the cluster list rebuilds.
      const bool kept = !cluster || next == &stepped;
      CHECK(builder.reused() == kept);
      CHECK(builder.builds() == (kept ? 1u : 2u));
      if (!cluster) {
        CHECK(strict_keys(builder.table(), *next) ==
              brute(*next, box, cutoff, false, true, periodic, 2));
      } else {
        auto inside = [&](const std::set<Key> &keys) {
          std::set<Key> out;
          for (const Key &key : keys) {
            const auto i = static_cast<std::size_t>(key.i);
            const auto j = static_cast<std::size_t>(key.j);
            double r2 = 0.0;
            for (int c = 0; c < 3; ++c) {
              const double v = (*next)[3 * j + static_cast<std::size_t>(c)] -
                               (*next)[3 * i + static_cast<std::size_t>(c)] +
                               key.sa * box[c] + key.sb * box[3 + c] + key.sc * box[6 + c];
              r2 += v * v;
            }
            if (r2 < cutoff * cutoff) {
              out.insert(key);
            }
          }
          return out;
        };
        rgpot::nlist::Table fresh;
        rgpot::nlist::build(fresh, next->data(), 64, box, query);
        CHECK(inside(from_table(builder.table())) == inside(from_table(fresh)));
      }
    }
  }
}

TEST_CASE("a negative skin is not inside the skin", "[nlist][api]") {
  const double pos[3] = {1.0, 2.0, 3.0};
  const double box[9] = {5, 0, 0, 0, 6, 0, 0, 0, 7};
  const double sheared[9] = {5, 0.5, 0, 0, 5.5, 0, 0, 0, 7};
  const double span[3] = {2, 2, 2};
  REQUIRE_FALSE(rgpot::nlist::withinSkin(pos, pos, 1, -0.1));
  REQUIRE_FALSE(rgpot::nlist::withinSkin(pos, pos, 1, box, box, span, -0.1));
  REQUIRE_FALSE(rgpot::nlist::withinSkin(pos, pos, 1, box, sheared, span, -0.1));
  REQUIRE(rgpot::nlist::withinSkin(pos, pos, 1, 0.0));
  REQUIRE(rgpot::nlist::withinSkin(pos, pos, 1, box, box, span, 0.0));
}

TEST_CASE("a non-finite displacement is not inside the skin", "[nlist][api]") {
  const double pos[3] = {1.0, 2.0, 3.0};
  const double current[3] = {std::numeric_limits<double>::quiet_NaN(), 2.0,
                             3.0};
  const double box[9] = {5, 0, 0, 0, 6, 0, 0, 0, 7};
  double changed[9] = {5, 0, 0, 0, 6, 0, 0, 0, 7};
  changed[1] = std::numeric_limits<double>::quiet_NaN();
  const double span[3] = {1, 1, 1};
  REQUIRE_FALSE(rgpot::nlist::withinSkin(pos, current, 1, 1.0));
  REQUIRE_FALSE(rgpot::nlist::withinSkin(pos, current, 1, box, box, span, 1.0));
  REQUIRE_FALSE(rgpot::nlist::withinSkin(pos, pos, 1, box, changed, span, 1.0));
}

TEST_CASE("an infinite or NaN skin is not inside the skin", "[nlist][api]") {
  const double pos[3] = {1.0, 2.0, 3.0};
  const double moved[3] = {101.0, 2.0, 3.0};
  const double box[9] = {5, 0, 0, 0, 6, 0, 0, 0, 7};
  const double larger[9] = {50, 0, 0, 0, 6, 0, 0, 0, 7};
  const double span[3] = {1, 1, 1};
  for (double skin : {std::numeric_limits<double>::infinity(),
                      -std::numeric_limits<double>::infinity(),
                      std::numeric_limits<double>::quiet_NaN()}) {
    INFO("skin " << skin);
    REQUIRE_FALSE(rgpot::nlist::withinSkin(pos, pos, 1, skin));
    REQUIRE_FALSE(rgpot::nlist::withinSkin(pos, moved, 1, skin));
    REQUIRE_FALSE(rgpot::nlist::withinSkin(pos, pos, 1, box, box, span, skin));
    REQUIRE_FALSE(rgpot::nlist::withinSkin(pos, moved, 1, box, larger, span, skin));
  }
}

TEST_CASE("withinSkin rejects non-finite corner shifts", "[nlist][api][edge]") {
  const std::vector<double> pos = {0.0, 0.0, 0.0, 0.1, 0.1, 0.1};
  const double ref_box[9] = {10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0};
  const double inf_box[9] = {std::numeric_limits<double>::infinity(), 0.0, 0.0,
                             0.0,                                 10.0, 0.0,
                             0.0,                                 0.0,  10.0};
  const double span[3] = {1.0, 1.0, 0.0};
  REQUIRE_FALSE(rgpot::nlist::withinSkin(pos.data(), pos.data(), pos.size() / 3, ref_box,
                                         inf_box, span, 0.5));
}

TEST_CASE("a positive skin with a strict cut is rejected", "[nlist][api]") {
  const double box[9] = {10, 0, 0, 0, 10, 0, 0, 0, 10};
  const double pos[] = {0, 0, 0, 1, 0, 0};
  rgpot::nlist::Query query;
  query.cutoff = 2.0;
  query.skin = 0.2;
  query.cut = rgpot::nlist::Cut::Strict;
  rgpot::nlist::Table table;
  REQUIRE_THROWS_AS(rgpot::nlist::build(table, pos, 2, box, query), rgpot::nlist::Error);
  try {
    rgpot::nlist::build(table, pos, 2, box, query);
  } catch (const rgpot::nlist::Error &err) {
    REQUIRE(err.code() == rgpot::nlist::Code::Skin);
  }
}

TEST_CASE("a null positions pointer is rejected when atoms are present",
          "[nlist][api][edge]") {
  // check_geometry's Code::Null branch for positions: no existing test
  // calls build() with a null positions pointer and a nonzero atom count.
  const double box[9] = {10, 0, 0, 0, 10, 0, 0, 0, 10};
  rgpot::nlist::Query query;
  query.cutoff = 2.0;
  rgpot::nlist::Table table;
  REQUIRE_THROWS_AS(rgpot::nlist::build(table, nullptr, 2, box, query),
                    rgpot::nlist::Error);
  try {
    rgpot::nlist::build(table, nullptr, 2, box, query);
  } catch (const rgpot::nlist::Error &err) {
    REQUIRE(err.code() == rgpot::nlist::Code::Null);
  }
  // A null positions pointer with zero atoms is not an error: there is
  // nothing to read.
  REQUIRE_NOTHROW(rgpot::nlist::build(table, nullptr, 0, box, query));
  REQUIRE(table.size() == 0);
}

TEST_CASE("a null box pointer is rejected", "[nlist][api][edge]") {
  const double pos[] = {0, 0, 0, 1, 0, 0};
  rgpot::nlist::Query query;
  query.cutoff = 2.0;
  rgpot::nlist::Table table;
  REQUIRE_THROWS_AS(rgpot::nlist::build(table, pos, 2, nullptr, query),
                    rgpot::nlist::Error);
  try {
    rgpot::nlist::build(table, pos, 2, nullptr, query);
  } catch (const rgpot::nlist::Error &err) {
    REQUIRE(err.code() == rgpot::nlist::Code::Null);
  }
}

TEST_CASE("a periodic box with zero volume is rejected", "[nlist][api][edge]") {
  // check_geometry's Code::Box branch: a degenerate (singular) periodic
  // cell, which the minimum image search cannot invert.
  const double flat[9] = {10, 0, 0, 0, 10, 0, 0, 0, 0};
  const double pos[] = {0, 0, 0, 1, 0, 0};
  rgpot::nlist::Query query;
  query.cutoff = 2.0;
  query.periodic = {{true, true, true}};
  rgpot::nlist::Table table;
  REQUIRE_THROWS_AS(rgpot::nlist::build(table, pos, 2, flat, query),
                    rgpot::nlist::Error);
  try {
    rgpot::nlist::build(table, pos, 2, flat, query);
  } catch (const rgpot::nlist::Error &err) {
    REQUIRE(err.code() == rgpot::nlist::Code::Box);
  }
  // The same cell is fine once no direction is periodic: zero volume
  // never reaches the minimum image search.
  query.periodic = {{false, false, false}};
  REQUIRE_NOTHROW(rgpot::nlist::build(table, pos, 2, flat, query));
}

TEST_CASE("a negative or non-finite skin is rejected", "[nlist][api][edge]") {
  // Distinct from "a positive skin with a strict cut is rejected": that
  // test exercises query.skin > 0 with Cut::Strict (Code::Skin). This one
  // exercises the skin value itself being out of range (Code::Cutoff),
  // independent of the cut mode.
  const double box[9] = {10, 0, 0, 0, 10, 0, 0, 0, 10};
  const double pos[] = {0, 0, 0, 1, 0, 0};
  rgpot::nlist::Table table;

  rgpot::nlist::Query negative;
  negative.cutoff = 2.0;
  negative.skin = -0.1;
  REQUIRE_THROWS_AS(rgpot::nlist::build(table, pos, 2, box, negative),
                    rgpot::nlist::Error);
  try {
    rgpot::nlist::build(table, pos, 2, box, negative);
  } catch (const rgpot::nlist::Error &err) {
    REQUIRE(err.code() == rgpot::nlist::Code::Cutoff);
  }

  rgpot::nlist::Query non_finite;
  non_finite.cutoff = 2.0;
  non_finite.cut = rgpot::nlist::Cut::Loose;
  for (double skin : {std::numeric_limits<double>::quiet_NaN(),
                      std::numeric_limits<double>::infinity(),
                      -std::numeric_limits<double>::infinity()}) {
    non_finite.skin = skin;
    try {
      rgpot::nlist::build(table, pos, 2, box, non_finite);
      FAIL("build accepted the skin " << skin);
    } catch (const rgpot::nlist::Error &err) {
      REQUIRE(err.code() == rgpot::nlist::Code::Cutoff);
      REQUIRE(err.what() == std::string("skin must be finite and >= 0"));
    }
  }
}

TEST_CASE("an out-of-range shell or cut mode is rejected", "[nlist][api][edge]") {
  // The validation checks `pairs->shell != Shell::Half && != Shell::Full`
  // and `cut != Strict && != Loose`: defensive guards against a Query
  // built with a value outside the enum's named set, which the type
  // system does not forbid for an `enum class`.
  const double box[9] = {10, 0, 0, 0, 10, 0, 0, 0, 10};
  const double pos[] = {0, 0, 0, 1, 0, 0};
  rgpot::nlist::Table table;

  rgpot::nlist::Query badShell;
  badShell.cutoff = 2.0;
  badShell.shape = rgpot::nlist::Pairs{static_cast<rgpot::nlist::Shell>(7)};
  REQUIRE_THROWS_AS(rgpot::nlist::build(table, pos, 2, box, badShell),
                    rgpot::nlist::Error);
  try {
    rgpot::nlist::build(table, pos, 2, box, badShell);
  } catch (const rgpot::nlist::Error &err) {
    REQUIRE(err.code() == rgpot::nlist::Code::Shape);
  }

  rgpot::nlist::Query badCut;
  badCut.cutoff = 2.0;
  badCut.cut = static_cast<rgpot::nlist::Cut>(7);
  REQUIRE_THROWS_AS(rgpot::nlist::build(table, pos, 2, box, badCut),
                    rgpot::nlist::Error);
  try {
    rgpot::nlist::build(table, pos, 2, box, badCut);
  } catch (const rgpot::nlist::Error &err) {
    REQUIRE(err.code() == rgpot::nlist::Code::Shape);
  }
}

TEST_CASE("a non-finite box element is rejected", "[nlist][api][edge]") {
  const double nan_box[9] = {10, 0, 0, 0, 10, 0, 0, 0,
                             std::numeric_limits<double>::quiet_NaN()};
  const double pos[] = {0, 0, 0, 1, 0, 0};
  rgpot::nlist::Query query;
  query.cutoff = 2.0;
  query.periodic = {{false, false, false}};
  rgpot::nlist::Table table;
  REQUIRE_THROWS_AS(rgpot::nlist::build(table, pos, 2, nan_box, query),
                    rgpot::nlist::Error);
  try {
    rgpot::nlist::build(table, pos, 2, nan_box, query);
  } catch (const rgpot::nlist::Error &err) {
    REQUIRE(err.code() == rgpot::nlist::Code::Box);
  }
}

TEST_CASE("a streaming walk rejects buffers too short for the pair cap",
          "[nlist][api][edge]") {
  // Walk::pull checks each buffer against the pair cap in a fixed order and
  // every check throws Code::Short with its own message. Each sub-block gives
  // every enabled buffer a valid size except the one under test, so only that
  // check can fire, and the message says which one did. The cap is one more
  // than the number of pairs, so a short buffer is still long enough for
  // every pair a pull would write and a skipped check cannot overrun it.
  const double box[9] = {6, 0, 0, 0, 5, 0, 1, 0, 7};
  const std::vector<double> pos = {0.5, 0.5, 0.5, 2.0, 1.0, 1.0, 4.5, 3.0, 6.0};
  const std::size_t natoms = pos.size() / 3;
  rgpot::nlist::Query query;
  query.cutoff = 4.0;
  query.shape = rgpot::nlist::Pairs{rgpot::nlist::Shell::Full};
  query.cut = rgpot::nlist::Cut::Strict;
  REQUIRE(query.fields.shifts);
  REQUIRE(query.fields.vectors);
  REQUIRE(query.fields.distances);

  rgpot::nlist::Query cluster_query = query;
  cluster_query.shape = rgpot::nlist::Cluster4{};

  auto pair_count = [&](const rgpot::nlist::Query &q) {
    rgpot::nlist::Table table;
    rgpot::nlist::build(table, pos.data(), natoms, box, q);
    return table.size();
  };
  const std::size_t pairs = pair_count(query);
  const std::size_t cluster_pairs = pair_count(cluster_query);
  REQUIRE(pairs >= 2);
  REQUIRE(cluster_pairs >= 1);

  struct Storage {
    explicit Storage(std::size_t cap)
        : first(cap), second(cap), shift(3 * cap), vector(3 * cap), distance(cap), mask(cap) {}
    std::vector<std::int32_t> first, second, shift;
    std::vector<double> vector, distance;
    std::vector<std::uint8_t> mask;
    rgpot::nlist::Buffers valid() {
      return {first, second, shift, vector, distance, mask};
    }
  };

  auto expect_short = [&](const rgpot::nlist::Query &q, std::size_t cap,
                          auto &&shorten, const char *what) {
    Storage store(cap);
    rgpot::nlist::Buffers dest = store.valid();
    shorten(dest);
    rgpot::nlist::Walk walk(pos.data(), natoms, box, q);
    bool done = false;
    try {
      (void)walk.pull(dest, done);
      FAIL("pull accepted a short buffer: " << what);
    } catch (const rgpot::nlist::Error &err) {
      REQUIRE(err.code() == rgpot::nlist::Code::Short);
      REQUIRE(err.what() == std::string(what));
    }
  };

  const std::size_t cap = pairs + 1;
  expect_short(
      query, cap, [&](rgpot::nlist::Buffers &d) { d.second = d.second.first(cap - 1); },
      "first and second buffers have different lengths");
  expect_short(
      query, cap, [&](rgpot::nlist::Buffers &d) { d.shift = d.shift.first(3 * cap - 1); },
      "shift buffer needs 3 entries per pair");
  expect_short(
      query, cap, [&](rgpot::nlist::Buffers &d) { d.vector = d.vector.first(3 * cap - 1); },
      "vector buffer needs 3 entries per pair");
  expect_short(
      query, cap, [&](rgpot::nlist::Buffers &d) { d.distance = d.distance.first(cap - 1); },
      "distance buffer is shorter than the pair buffer");
  const std::size_t cluster_cap = cluster_pairs + 1;
  expect_short(
      cluster_query, cluster_cap,
      [&](rgpot::nlist::Buffers &d) { d.mask = d.mask.first(cluster_cap - 1); },
      "a cluster list needs one mask byte per pair");

  {
    Storage store(cap);
    rgpot::nlist::Buffers dest = store.valid();
    rgpot::nlist::Walk walk(pos.data(), natoms, box, query);
    bool done = false;
    REQUIRE(walk.pull(dest, done) == pairs);
    REQUIRE(done);
  }
}

TEST_CASE("a moved-from builder and walk stay usable", "[nlist][api]") {
  const double box[9] = {10, 0, 0, 0, 10, 0, 0, 0, 10};
  const double pos[] = {1, 1, 1, 2, 1, 1, 8, 8, 8};
  rgpot::nlist::Query query;
  query.cutoff = 2.0;
  query.skin = 0.5;
  query.cut = rgpot::nlist::Cut::Loose;
  query.shape = rgpot::nlist::Pairs{rgpot::nlist::Shell::Half};

  rgpot::nlist::Builder first;
  first.update(pos, 3, box, query);
  REQUIRE(first.table().size() == 1);
  rgpot::nlist::Builder copy(first);
  rgpot::nlist::Builder moved(std::move(first));
  REQUIRE(moved.table().size() == 1);
  REQUIRE(first.table().size() == 0);
  REQUIRE(first.builds() == 0);
  REQUIRE_FALSE(first.reused());
  first.update(pos, 3, box, query);
  REQUIRE(first.table().size() == 1);
  REQUIRE(first.builds() == 1);
  copy.update(pos, 3, box, query);
  REQUIRE(copy.reused());
  REQUIRE(copy.builds() == 1);
  first = copy;
  REQUIRE(first.builds() == 1);

  rgpot::nlist::Walk walk(pos, 3, box, query);
  rgpot::nlist::Walk taken(std::move(walk));
  std::vector<std::int32_t> a(4);
  std::vector<std::int32_t> b(4);
  std::vector<std::int32_t> s(12);
  std::vector<double> v(12);
  std::vector<double> d(4);
  bool done = false;
  rgpot::nlist::Buffers buf{a, b, s, v, d, {}};
  REQUIRE(taken.pull(buf, done) == 1);
  try {
    (void)walk.pull(buf, done);
    FAIL("a moved-from walk pulled pairs");
  } catch (const rgpot::nlist::Error &err) {
    REQUIRE(err.code() == rgpot::nlist::Code::Null);
  }
}

TEST_CASE("a batch keeps the systems apart", "[nlist][api]") {
  const double positions[] = {0, 0, 0, 1.2, 0, 0, 0, 0, 0, 1.2, 0, 0};
  const std::int32_t count[] = {2, 2};
  const double box[] = {8, 0, 0, 0, 8, 0, 0, 0, 8, 9, 0, 0, 0, 9, 0, 0, 0, 9};
  rgpot::nlist::Query query;
  query.cutoff = 2.0;
  rgpot::nlist::Systems systems{positions, count, 2, box};
  rgpot::nlist::Batch batch;
  rgpot::nlist::build(batch, systems, query);
  REQUIRE(batch.pairs.size() == 2);
  REQUIRE(batch.system[0] == 0);
  REQUIRE(batch.system[1] == 1);
  REQUIRE(batch.pairs.first[1] >= 2);
  rgpot::nlist::Table one;
  rgpot::nlist::build(one, positions, 2, box, query);
  REQUIRE(one.size() == 1);
  REQUIRE(batch.pairs.distance[0] == one.distance[0]);
}

TEST_CASE("one atom per thread matches the stored list", "[nlist][api][device]") {
  constexpr int ncell = 8;
  constexpr double lattice = 3.615;
  const double basis[4][3] = {{0, 0, 0}, {0, 0.5, 0.5}, {0.5, 0, 0.5}, {0.5, 0.5, 0}};
  std::vector<double> pos;
  for (int ix = 0; ix < ncell; ++ix) {
    for (int iy = 0; iy < ncell; ++iy) {
      for (int iz = 0; iz < ncell; ++iz) {
        for (const auto &b : basis) {
          pos.push_back((ix + b[0]) * lattice);
          pos.push_back((iy + b[1]) * lattice);
          pos.push_back((iz + b[2]) * lattice);
        }
      }
    }
  }
  const double edge = ncell * lattice;
  const double box[9] = {edge, 0, 0, 0, edge, 0, 0, 0, edge};
  rgpot::nlist::Query query;
  query.cutoff = 5.0;
  query.shape = rgpot::nlist::Pairs{rgpot::nlist::Shell::Half};
  rgpot::nlist::Table bulk;
  rgpot::nlist::Table by_atom;
  rgpot::nlist::build(bulk, pos.data(), pos.size() / 3, box, query);
  rgpot::nlist::buildByAtom(by_atom, pos.data(), pos.size() / 3, box, query);
  REQUIRE(pos.size() / 3 >= 512);
  REQUIRE(from_table(bulk) == from_table(by_atom));
}

TEST_CASE("the fortran bridge reuses the builder and refreshes vectors", "[nlist][api]") {
  double pos[] = {0, 0, 0, 1.2, 0, 0};
  const double box[9] = {8, 0, 0, 0, 8, 0, 0, 0, 8};
  const int32_t periodic[3] = {1, 1, 1};
  auto *list = rgpot_nlist_fortran_open();
  REQUIRE(list != nullptr);
  size_t needed = 0;
  int32_t reused = 1;
  REQUIRE(rgpot_nlist_fortran_build(list, pos, 2, box, 2.0, 0.4, periodic, nullptr, nullptr,
                                    nullptr, nullptr, 0, &needed, &reused) == RGPOT_NLIST_OK);
  REQUIRE(needed == 2);
  REQUIRE(reused == 0);
  std::vector<int32_t> first(needed), second(needed);
  std::vector<double> vector(needed * 3), distance(needed);
  REQUIRE(rgpot_nlist_fortran_build(list, pos, 2, box, 2.0, 0.4, periodic, first.data(),
                                    second.data(), vector.data(), distance.data(), needed, &needed,
                                    &reused) == RGPOT_NLIST_OK);
  REQUIRE(distance[0] == Catch::Approx(1.2));
  pos[0] = 0.05;
  REQUIRE(rgpot_nlist_fortran_build(list, pos, 2, box, 2.0, 0.4, periodic, first.data(),
                                    second.data(), vector.data(), distance.data(), needed, &needed,
                                    &reused) == RGPOT_NLIST_OK);
  REQUIRE(reused == 1);
  REQUIRE(distance[0] == Catch::Approx(1.15));
  const auto view_n = needed;
  REQUIRE(view_n == 2);
  rgpot_nlist_fortran_close(list);
}

TEST_CASE("the fortran bridge keeps pairs of unwrapped coordinates", "[nlist][api]") {
  // Atom 0 lies one cell out. Its pair with atom 1 is 1.5 apart.
  const double pos[] = {10.5, 0.5, 0.5, 2.0, 0.5, 0.5};
  const double box[9] = {10, 0, 0, 0, 10, 0, 0, 0, 10};
  const int32_t periodic[3] = {1, 1, 1};
  auto *list = rgpot_nlist_fortran_open();
  REQUIRE(list != nullptr);
  std::vector<int32_t> first(4), second(4);
  std::vector<double> vector(12), distance(4);
  size_t count = 0;
  int32_t reused = 1;
  REQUIRE(rgpot_nlist_fortran_build(list, pos, 2, box, 3.0, 0.5, periodic, first.data(),
                                    second.data(), vector.data(), distance.data(), 4, &count,
                                    &reused) == RGPOT_NLIST_OK);
  REQUIRE(count == 2);
  for (std::size_t p = 0; p < count; ++p) {
    REQUIRE(distance[p] == Catch::Approx(1.5));
    const double sign = first[p] == 0 ? 1.0 : -1.0;
    REQUIRE(vector[3 * p] == Catch::Approx(sign * 1.5));
  }
  rgpot_nlist_fortran_close(list);
}

TEST_CASE("C shifts are relative to the positions as given", "[nlist][api]") {
  // r_j - r_i + S H is the vector for the caller's coordinates, also when
  // they lie whole cells outside the box and when a builder keeps its list.
  const double box[9] = {6.0, 0.4, 0.2, 0.8, 6.5, -0.3, 0.5, 0.6, 5.5};
  const std::array<bool, 3> periodic{{true, true, true}};
  constexpr std::size_t n = 24;
  constexpr double cutoff = 3.4;
  const auto wrapped = fractional_positions(box, n);
  auto offsets = random_offsets(n, periodic, 41);
  auto pos = unwrap(wrapped, box, offsets);

  rgpot_nlist_request request;
  rgpot_nlist_request_init(&request);
  request.cutoff = cutoff;
  request.shell = 1;
  constexpr std::size_t cap = 4096;
  std::vector<int32_t> first(cap), second(cap), shift(3 * cap);
  std::vector<double> vector(3 * cap), distance(cap);
  std::vector<uint8_t> mask(cap);
  rgpot_nlist_soa out{};
  out.bytes = sizeof(out);
  out.first = first.data();
  out.second = second.data();
  out.shift = shift.data();
  out.vector = vector.data();
  out.distance = distance.data();
  out.mask = mask.data();
  out.cap = cap;

  auto keys = [&](std::size_t count, const std::vector<double> &at, bool check_vectors) {
    std::set<Key> got;
    for (std::size_t p = 0; p < count; ++p) {
      for (int lane = 0; lane < 4; ++lane) {
        if (mask[p] != 0xff && (mask[p] & (1 << lane)) == 0) {
          continue;
        }
        const int i = first[p] + (mask[p] == 0xff ? 0 : lane);
        const int j = second[p];
        const int *s = &shift[3 * p];
        double r2 = 0.0;
        for (int c = 0; c < 3; ++c) {
          const double v = at[3 * static_cast<std::size_t>(j) + static_cast<std::size_t>(c)] -
                           at[3 * static_cast<std::size_t>(i) + static_cast<std::size_t>(c)] +
                           s[0] * box[c] + s[1] * box[3 + c] + s[2] * box[6 + c];
          if (check_vectors && (mask[p] == 0xff || lane == 0)) {
            REQUIRE(std::abs(v - vector[3 * p + static_cast<std::size_t>(c)]) < 1e-12);
          }
          r2 += v * v;
        }
        REQUIRE(r2 < cutoff * cutoff);
        got.insert(Key{i, j, s[0], s[1], s[2]});
        if (mask[p] == 0xff) {
          break;
        }
      }
    }
    return got;
  };

  const auto ref = brute(pos, box, cutoff, true, true, periodic, 10);
  REQUIRE(ref.size() > n);
  size_t count = 0;
  size_t needed = 0;
  REQUIRE(rgpot_nlist_build(pos.data(), n, box, &request, &out, &count, &needed) ==
          RGPOT_NLIST_OK);
  REQUIRE(keys(count, pos, true) == ref);

  rgpot_nlist_walk *walk = nullptr;
  REQUIRE(rgpot_nlist_walk_open(pos.data(), n, box, &request, &walk) == RGPOT_NLIST_OK);
  int32_t done = 0;
  REQUIRE(rgpot_nlist_walk_pull(walk, &out, &count, &done) == RGPOT_NLIST_OK);
  REQUIRE(done == 1);
  rgpot_nlist_walk_close(walk);
  REQUIRE(keys(count, pos, true) == ref);

  request.shape = 1;
  REQUIRE(rgpot_nlist_build(pos.data(), n, box, &request, &out, &count, &needed) ==
          RGPOT_NLIST_OK);
  REQUIRE(keys(count, pos, false) == ref);
  request.shape = 0;

  request.skin = 0.4;
  request.cut = 1;
  rgpot_nlist_builder *builder = nullptr;
  REQUIRE(rgpot_nlist_builder_open(&builder) == RGPOT_NLIST_OK);
  int32_t reused = 1;
  REQUIRE(rgpot_nlist_builder_update(builder, pos.data(), n, box, &request, &out, &count,
                                     &needed, &reused) == RGPOT_NLIST_OK);
  REQUIRE(reused == 0);
  // Move three atoms by whole cells. The list is kept and the shifts follow.
  for (std::size_t a : {std::size_t{0}, std::size_t{5}, std::size_t{11}}) {
    offsets[3 * a] += 2;
    offsets[3 * a + 2] -= 1;
  }
  pos = unwrap(wrapped, box, offsets);
  REQUIRE(rgpot_nlist_builder_update(builder, pos.data(), n, box, &request, &out, &count,
                                     &needed, &reused) == RGPOT_NLIST_OK);
  REQUIRE(reused == 1);
  std::set<Key> strict;
  for (std::size_t p = 0; p < count; ++p) {
    double r2 = 0.0;
    for (int c = 0; c < 3; ++c) {
      const double v =
          pos[3 * static_cast<std::size_t>(second[p]) + static_cast<std::size_t>(c)] -
          pos[3 * static_cast<std::size_t>(first[p]) + static_cast<std::size_t>(c)] +
          shift[3 * p] * box[c] + shift[3 * p + 1] * box[3 + c] + shift[3 * p + 2] * box[6 + c];
      REQUIRE(std::abs(v - vector[3 * p + static_cast<std::size_t>(c)]) < 1e-12);
      r2 += v * v;
    }
    if (r2 < cutoff * cutoff) {
      strict.insert(Key{first[p], second[p], shift[3 * p], shift[3 * p + 1], shift[3 * p + 2]});
    }
  }
  REQUIRE(strict == brute(pos, box, cutoff, true, true, periodic, 10));
  rgpot_nlist_builder_close(builder);
}

TEST_CASE("eight threads match the serial list", "[nlist][api]") {
  constexpr int ncell = 14;
  constexpr double lattice = 3.615;
  const double basis[4][3] = {{0, 0, 0}, {0, 0.5, 0.5}, {0.5, 0, 0.5}, {0.5, 0.5, 0}};
  std::vector<double> pos;
  for (int ix = 0; ix < ncell; ++ix) {
    for (int iy = 0; iy < ncell; ++iy) {
      for (int iz = 0; iz < ncell; ++iz) {
        for (const auto &b : basis) {
          pos.push_back((ix + b[0]) * lattice);
          pos.push_back((iy + b[1]) * lattice);
          pos.push_back((iz + b[2]) * lattice);
        }
      }
    }
  }
  const double edge = ncell * lattice;
  const double box[9] = {edge, 0, 0, 0, edge, 0, 0, 0, edge};
  rgpot::nlist::Query query;
  query.cutoff = 5.0;
  query.shape = rgpot::nlist::Pairs{rgpot::nlist::Shell::Half};
  set_omp_threads("1");
  rgpot::nlist::Table serial;
  rgpot::nlist::build(serial, pos.data(), pos.size() / 3, box, query);
  set_omp_threads("8");
  rgpot::nlist::Table threaded;
  rgpot::nlist::build(threaded, pos.data(), pos.size() / 3, box, query);
  set_omp_threads("1");
  REQUIRE(pos.size() / 3 >= 8192);
  REQUIRE(serial.size() == threaded.size());
  REQUIRE(from_table(serial) == from_table(threaded));
}

TEST_CASE("threads match the serial list when a thin or sheared box needs several images",
          "[nlist][api]") {
  constexpr std::size_t n = 9000;
  const auto box = GENERATE(std::array<double, 9>{240.0, 0, 0, 0, 240.0, 0, 0, 0, 4.0},
                            std::array<double, 9>{120.0, 0, 0, 3.0, 120.0, 0, 2.0, 1.5, 5.0});
  std::vector<double> pos(3 * n);
  std::uint64_t state = 12345;
  auto uniform = [&state]() {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<double>(state >> 11) * 0x1.0p-53;
  };
  for (std::size_t i = 0; i < n; ++i) {
    const double a = uniform();
    const double b = uniform();
    const double c = uniform();
    for (int k = 0; k < 3; ++k) {
      pos[3 * i + k] = a * box[k] + b * box[3 + k] + c * box[6 + k];
    }
  }
  const auto shell = GENERATE(rgpot::nlist::Shell::Full, rgpot::nlist::Shell::Half);
  rgpot::nlist::Query query;
  query.cutoff = 3.0;
  query.shape = rgpot::nlist::Pairs{shell};
  set_omp_threads("1");
  rgpot::nlist::Table serial;
  rgpot::nlist::build(serial, pos.data(), n, box.data(), query);
  set_omp_threads("8");
  rgpot::nlist::Table threaded;
  rgpot::nlist::build(threaded, pos.data(), n, box.data(), query);
  set_omp_threads("1");
  const auto pairs = from_table(serial);
  std::size_t repeated = 0;
  const Key *last = nullptr;
  for (const auto &key : pairs) {
    repeated += last != nullptr && last->i == key.i && last->j == key.j;
    last = &key;
  }
  REQUIRE(repeated > 0);
  REQUIRE(serial.size() == threaded.size());
  REQUIRE(pairs == from_table(threaded));
}
