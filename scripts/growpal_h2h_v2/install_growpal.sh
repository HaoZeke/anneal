#!/usr/bin/env bash
# Build GrowPAL and aegon from their published source and install them.
#
# GrowPAL 0.1.6 and aegon 1.3.15 are built into wheels from their PyPI
# sdists, checked by sha256. The LJ10 seeds come from GrowPAL commit 73952fa.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
WORK=/work/growpal-h2h-v2
if [[ ! -d /work ]]; then
  sudo -n mkdir -p /work
  sudo -n chown "$(id -u):$(id -g)" /work
fi
SRC=$WORK/src
VENV=$WORK/venv
WHEELS=$WORK/wheels
GROWPAL_VERSION=0.1.6
GROWPAL_SHA256=1625495ce899eddd7f6c8f449a4272f463efc5d7c8c8b1d9c2d40114ab195077
AEGON_VERSION=1.3.15
AEGON_SHA256=9d26a388d97ef7c75c3b613c6dc38660a3fdccda4bf714481ea6886d86ac881f
GIT_URL=https://github.com/isaacgcampos/GrowPAL.git
GIT_COMMIT=73952fa3a33ca9a1b1618d79bd154fd1480df518

mkdir -p "$SRC" "$WHEELS"
if [[ ! -d $SRC/GrowPAL/.git ]]; then
  git clone --quiet "$GIT_URL" "$SRC/GrowPAL"
fi
git -C "$SRC/GrowPAL" checkout --quiet "$GIT_COMMIT"
test "$(git -C "$SRC/GrowPAL" rev-parse HEAD)" = "$GIT_COMMIT"

if [[ ! -x $VENV/bin/python ]]; then
  python3 -m venv "$VENV"
fi
"$VENV/bin/pip" install --quiet --upgrade pip setuptools wheel
cd "$SRC"
for spec in "growpal==${GROWPAL_VERSION}" "aegon==${AEGON_VERSION}"; do
  if [[ ! -f ${spec%%==*}-${spec##*==}.tar.gz ]]; then
    "$VENV/bin/pip" download --quiet --no-deps --no-binary :all: "$spec" -d "$SRC"
  fi
done
sha256sum -c - << EOF
$GROWPAL_SHA256  growpal-${GROWPAL_VERSION}.tar.gz
$AEGON_SHA256  aegon-${AEGON_VERSION}.tar.gz
EOF

growpal_wheel=$WHEELS/growpal-${GROWPAL_VERSION}-py3-none-any.whl
aegon_wheel=$WHEELS/aegon-${AEGON_VERSION}-py3-none-any.whl
if [[ ! -f $growpal_wheel || ! -f $aegon_wheel ]]; then
  rm -f "$WHEELS"/growpal-*.whl "$WHEELS"/aegon-*.whl
  "$VENV/bin/pip" wheel --quiet --no-deps --no-build-isolation -w "$WHEELS" \
    "aegon-${AEGON_VERSION}.tar.gz" "growpal-${GROWPAL_VERSION}.tar.gz"
  "$VENV/bin/pip" install --quiet --force-reinstall --no-deps "$aegon_wheel" "$growpal_wheel"
fi
"$VENV/bin/pip" install --quiet -c "$HERE/constraints.txt" "$aegon_wheel" "$growpal_wheel"
"$VENV/bin/pip" check

ase_version=$("$VENV/bin/python" - << PY
import importlib.metadata as meta
for name, want in (("growpal", "${GROWPAL_VERSION}"), ("aegon", "${AEGON_VERSION}")):
    have = meta.version(name)
    if have != want:
        raise SystemExit(f"{name} {have} != {want}")
from growpal.libgrowpal import growpal_parallel
from growpal.libselect_clustering import select_by_clustering
from aegon.optimization.potentials.lj import _opt_positions
from aegon.optimization.potentials.ase_generic import _opt_ASE_single
from ase.calculators.morse import MorsePotential
print(meta.version("ase"))
PY
)

cat > "$WORK/growpal-source.txt" << EOF
growpal_version=$GROWPAL_VERSION
growpal_sdist=https://pypi.org/project/growpal/${GROWPAL_VERSION}/
growpal_sdist_sha256=$GROWPAL_SHA256
growpal_wheel_sha256=$(sha256sum "$growpal_wheel" | cut -d' ' -f1)
growpal_url=$GIT_URL
growpal_commit=$GIT_COMMIT
aegon_version=$AEGON_VERSION
aegon_sdist=https://pypi.org/project/aegon/${AEGON_VERSION}/
aegon_sdist_sha256=$AEGON_SHA256
aegon_wheel_sha256=$(sha256sum "$aegon_wheel" | cut -d' ' -f1)
ase_version=$ase_version
python=$("$VENV/bin/python" -c 'import platform; print(platform.python_version())')
seeds=$SRC/GrowPAL/examples/LJ/LJ010.xyz
seeds_sha256=$(sha256sum "$SRC/GrowPAL/examples/LJ/LJ010.xyz" | cut -d' ' -f1)
EOF
"$VENV/bin/pip" freeze > "$WORK/growpal-freeze.txt"
cat "$WORK/growpal-source.txt"
