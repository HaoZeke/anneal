#!/usr/bin/env bash
# Build a double-precision, thread-MPI GROMACS whose CP2K QM/MM interface
# links librgpot_gromacs instead of libcp2k, and install it under PREFIX.
# BUILD is an rgpot meson build directory configured with
# -Dwith_gromacs_dropin=enabled; GROMACS finds the library through
# BUILD/engines/gromacs/pkgconfig/libcp2k.pc.
#
# Usage: build_gromacs.sh VERSION PREFIX BUILD
#   VERSION is 2024.6, 2025.5 or 2026.4 (the pinned tarballs).
set -euo pipefail

if [ $# -ne 3 ]; then
  echo "usage: $0 VERSION PREFIX BUILD" >&2
  exit 2
fi
version=$1
prefix=$2
build=$(cd "$3" && pwd)

case $version in
  2024.6) sha256=7cbad81f51c71a144d646515a7249aa74940b3f68071f51410e3a9473f05b339 ;;
  2025.5) sha256=7d0c7791675d0171af71dbfb35bfbb90bc4deb08266e62fbdcdac996f1f60174 ;;
  2026.4) sha256=58eda60979b124fcf9dff8b1b63cedfbc72fdb36d99e921dc1bdc310571098b6 ;;
  *) echo "no pinned tarball for GROMACS $version" >&2; exit 2 ;;
esac

if [ ! -f "$build/engines/gromacs/pkgconfig/libcp2k.pc" ]; then
  echo "$build has no engines/gromacs/pkgconfig/libcp2k.pc; configure rgpot with -Dwith_gromacs_dropin=enabled" >&2
  exit 2
fi

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
tarball="$work/gromacs-$version.tar.gz"
curl -fsSL --retry 3 -o "$tarball" "https://ftp.gromacs.org/gromacs/gromacs-$version.tar.gz"
echo "$sha256  $tarball" | sha256sum -c -
tar xzf "$tarball" -C "$work"

cmake -S "$work/gromacs-$version" -B "$work/build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$prefix" \
  -DGMX_DOUBLE=ON \
  -DGMX_MPI=OFF \
  -DGMX_GPU=OFF \
  -DGMX_FFT_LIBRARY=fftpack \
  -DGMX_EXTERNAL_BLAS=OFF \
  -DGMX_EXTERNAL_LAPACK=OFF \
  -DGMX_CP2K=ON \
  -DCP2K_DIR="$build/engines/gromacs" \
  -DGMX_USE_PLUMED=OFF \
  -DGMX_NNPOT=OFF \
  -DGMX_HWLOC=OFF \
  -DGMX_BUILD_HELP=OFF \
  -DGMX_DEFAULT_SUFFIX=ON \
  -DBUILD_TESTING=OFF
cmake --build "$work/build" --target install -j "$(nproc 2>/dev/null || echo 2)"
"$prefix/bin/gmx_d" --version | grep -E 'GROMACS version|Precision'
ldd "$prefix"/lib*/libgromacs_d.so | grep -q librgpot_gromacs ||
  { echo "libgromacs_d.so does not link librgpot_gromacs" >&2; exit 1; }
