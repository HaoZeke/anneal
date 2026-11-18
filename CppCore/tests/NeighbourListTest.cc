// MIT License
// Copyright 2023--present rgpot developers

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <map>
#include <set>
#include <span>
#include <string>
#include <tuple>
#include <vector>

#include <catch2/catch_all.hpp>

#include "lsan_libomp.hpp"
#include "rgpot/nlist/NeighbourList.hpp"
#include "rgpot/nlist/rgpot_nlist.h"

namespace {

void set_omp_threads(const char *value) {
#if defined(_MSC_VER)
  _putenv_s("OMP_NUM_THREADS", value);
#else
  setenv("OMP_NUM_THREADS", value, 1);
#endif
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
                    bool full, bool strict, const std::array<bool, 3> &periodic) {
  const int n = static_cast<int>(pos.size() / 3);
  const int span = 3;
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

} // namespace

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
  non_finite.skin = std::numeric_limits<double>::infinity();
  REQUIRE_THROWS_AS(rgpot::nlist::build(table, pos, 2, box, non_finite),
                    rgpot::nlist::Error);
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
  // The chunked-walk test above ("a chunked walk returns the stored
  // pairs") only ever hands pull() buffers that are long enough. Walk::pull
  // validates each optional buffer independently (Code::Short); none of
  // those branches run anywhere else.
  const double box[9] = {6, 0, 0, 0, 5, 0, 1, 0, 7};
  const std::vector<double> pos = {0.5, 0.5, 0.5, 2.0, 1.0, 1.0, 4.5, 3.0, 6.0};
  rgpot::nlist::Query query;
  query.cutoff = 4.0;
  query.shape = rgpot::nlist::Pairs{rgpot::nlist::Shell::Full};
  query.cut = rgpot::nlist::Cut::Strict;

  auto freshWalk = [&] {
    return rgpot::nlist::Walk(pos.data(), pos.size() / 3, box, query);
  };

  std::vector<std::int32_t> first(2), second(2), shift(6), bad_shift(2);
  std::vector<double> vector(6), bad_vector(2), distance(2), bad_distance(1);
  std::vector<std::uint8_t> mask(2), bad_mask(1);

  {
    auto walk = freshWalk();
    rgpot::nlist::Buffers dest;
    dest.first = first;
    dest.second = std::span<std::int32_t>(second.data(), 1); // mismatched
    bool done = false;
    REQUIRE_THROWS_AS(walk.pull(dest, done), rgpot::nlist::Error);
  }
  {
    auto walk = freshWalk();
    rgpot::nlist::Buffers dest;
    dest.first = first;
    dest.second = second;
    dest.shift = bad_shift; // too short: needs 3 * cap
    bool done = false;
    REQUIRE_THROWS_AS(walk.pull(dest, done), rgpot::nlist::Error);
  }
  {
    auto walk = freshWalk();
    rgpot::nlist::Buffers dest;
    dest.first = first;
    dest.second = second;
    dest.vector = bad_vector; // too short: needs 3 * cap
    bool done = false;
    REQUIRE_THROWS_AS(walk.pull(dest, done), rgpot::nlist::Error);
  }
  {
    auto walk = freshWalk();
    rgpot::nlist::Buffers dest;
    dest.first = first;
    dest.second = second;
    dest.distance = bad_distance; // too short: needs cap
    bool done = false;
    REQUIRE_THROWS_AS(walk.pull(dest, done), rgpot::nlist::Error);
  }
  {
    // A cluster-of-four query needs a mask byte per pair.
    rgpot::nlist::Query clusterQuery = query;
    clusterQuery.shape = rgpot::nlist::Cluster4{};
    rgpot::nlist::Walk walk(pos.data(), pos.size() / 3, box, clusterQuery);
    rgpot::nlist::Buffers dest;
    dest.first = first;
    dest.second = second;
    dest.mask = bad_mask; // too short: needs cap
    bool done = false;
    REQUIRE_THROWS_AS(walk.pull(dest, done), rgpot::nlist::Error);
  }
  // A well-formed call still succeeds after each of the rejections above.
  {
    auto walk = freshWalk();
    rgpot::nlist::Buffers dest;
    dest.first = first;
    dest.second = second;
    dest.shift = shift;
    dest.vector = vector;
    dest.distance = distance;
    dest.mask = mask;
    bool done = false;
    REQUIRE_NOTHROW(walk.pull(dest, done));
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
