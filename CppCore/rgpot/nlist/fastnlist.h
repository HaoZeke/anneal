#pragma once
// MIT License
// Copyright 2023--present rgpot developers
//
// C ABI of the linkcell / minimage static library. rgpot_fast_pairs and
// rgpot_fast_wrap_axes need an orthorhombic cell; rgpot_fast_wrap takes a
// general cell. Callers do not keep a scalar scan beside these entry points.

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Unique unordered pairs `(i, j)` with `i < j`, sorted. Every pair with
/// an image inside `list_cutoff` is present. On a cell only a few cutoffs
/// wide the buffer is every unordered pair, `(0, 1), (0, 2), ...,
/// (n - 2, n - 1)`, and the caller drops the rest. That count is
/// `n * (n - 1) / 2`, and the call answers a short buffer from it without
/// building the list. Self images are never included.
///
/// The cell must be orthorhombic; any other cell returns a negative code.
/// `cell9` is nine doubles, lattice vectors a, b, c packed as
/// ax, ay, az, bx, by, bz, cx, cy, cz. `periodic` is three ints, nonzero
/// when that axis folds. An open axis is padded inside the call so a
/// periodic image of it cannot fall inside the cutoff.
///
/// `out_ij` holds `cap` pairs (two ints each). A short buffer returns 1,
/// writes the needed pair count to `*out_count`, and leaves `out_ij`
/// untouched; `out_ij` may be NULL for that probe. Success returns 0. A
/// hard failure returns a negative code; `rgpot_fast_last_error` then
/// points at a thread-local message. Both 0 and 1 clear that message.
int rgpot_fast_pairs(const double *xyz, size_t n, const double *cell9,
                     const int32_t *periodic, double list_cutoff,
                     int32_t *out_ij, size_t cap, size_t *out_count);

/// Minimum-image wrap of `n` packed difference vectors. `cell9` is the
/// same layout as `rgpot_fast_pairs`. Every axis folds. A singular cell
/// returns -1. On a sheared cell this is the fractional wrap, which is the
/// minimum image only for vectors shorter than half the smallest face
/// width.
int rgpot_fast_wrap(const double *cell9, const double *diffs, size_t n,
                    double *out);

/// Same wrap, but an axis with `periodic[k] == 0` is left unchanged.
/// The cell must be orthorhombic. Periodic widths must be positive.
int rgpot_fast_wrap_axes(const double *cell9, const int32_t *periodic,
                         const double *diffs, size_t n, double *out);

/// Thread-local message for the most recent negative return on this
/// thread, or NULL when the last call returned 0 or 1. Do not free it.
const char *rgpot_fast_last_error(void);

#ifdef __cplusplus
}
#endif
