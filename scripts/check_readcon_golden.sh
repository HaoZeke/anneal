#!/usr/bin/env bash
# Install the eonviz readcon pin and compare atoms_to_con against the golden.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python3}"
REQ="$("$PY" - <<'PY'
import tomllib
spec = tomllib.load(open("pixi.toml", "rb"))["feature"]["eonviz"]["pypi-dependencies"]["readcon"]
if isinstance(spec, dict) and "rev" in spec:
    print("readcon @ git+https://github.com/lode-org/readcon-core.git@" + spec["rev"])
elif isinstance(spec, str) and spec.strip():
    text = spec.strip()
    print(text if text.startswith("readcon") else "readcon" + text)
else:
    raise SystemExit("readcon pin is not a version or a git rev")
PY
)"
tmp_root="${RUNNER_TEMP:-${TMPDIR:-/tmp}}"
if [[ ! -d "$tmp_root" || ! -w "$tmp_root" ]]; then
  echo "readcon golden: ${tmp_root} is not a writable directory" >&2
  exit 1
fi
VENV=$(mktemp -d "${tmp_root}/readcon-golden.XXXXXX")
trap 'rm -rf "$VENV"' EXIT
uv venv "$VENV"
uv pip install --python "$VENV/bin/python" \
  pytest numpy ase \
  "$REQ"
"$VENV/bin/python" -m pytest -q tests/test_readcon_restart.py --tb=short
