# Copyright (c) anneal developers
"""Write the TIP4P Cambridge energy table from structure files.

Each row is one cluster size. The energy is the Energy line of that
size's xyz file, not a typed constant. A size with no file in the
repository is downloaded from the Cambridge Cluster Database into
``build/tip4p-ccd/``.
"""

from __future__ import annotations

import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build" / "tip4p-ccd"
OUT = ROOT / "data" / "tip4p_ccd.csv"
SIZES = range(2, 22)
PAGE = "https://www-wales.ch.cam.ac.uk/~wales/CCD/TIP4P/xyz/TIP4P-{n}.xyz"
# sha256 of the Cambridge xyz files that are not already in the repository.
PINNED = {
    2: "e7fc82f98676af378a9eafaf64cd8349360b570715defb3b419d96b093fff503",
    3: "8057429008ee1d1a167b69b0f2e7bd7a5b723ca066df2362c60814ca20af710e",
    5: "e02b093112c0ca7a5e0d12f747414db79380c7bd9dca3032467a1d28ed5d4384",
}
ENERGY = re.compile(r"Energy\s*=\s*(-?\d+\.\d+)", re.IGNORECASE)
NAME = re.compile(r"(?i)tip4p-(\d+)\.xyz$")


def sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def energy_token(path: Path) -> str:
    for line in path.read_text().splitlines()[:4]:
        match = ENERGY.search(line)
        if match:
            return match.group(1)
    raise SystemExit(f"{path} has no Energy line")


def local_files() -> dict[int, list[Path]]:
    found: dict[int, list[Path]] = {}
    for path in ROOT.rglob("*.xyz"):
        if "build" in path.parts or "tip4p-ccd" in path.parts:
            continue
        match = NAME.search(path.name)
        if not match:
            continue
        found.setdefault(int(match.group(1)), []).append(path)
    return found


def download(n: int) -> Path:
    BUILD.mkdir(parents=True, exist_ok=True)
    dest = BUILD / f"TIP4P-{n}.xyz"
    url = PAGE.format(n=n)
    request = urllib.request.Request(url, headers={"User-Agent": "anneal"})
    dest.write_bytes(urllib.request.urlopen(request, timeout=60).read())
    digest = sha256(dest)
    if digest != PINNED[n]:
        raise SystemExit(f"TIP4P-{n}.xyz sha256 is {digest}, pinned {PINNED[n]}")
    return dest


def choose(found: dict[int, list[Path]], n: int) -> tuple[str, str, str]:
    if n in PINNED:
        dest = BUILD / f"TIP4P-{n}.xyz"
        if not dest.is_file() or sha256(dest) != PINNED[n]:
            dest = download(n)
        return energy_token(dest), PAGE.format(n=n), PINNED[n]
    paths = found.get(n, [])
    tokens = {energy_token(path): path for path in paths}
    if len(tokens) > 1:
        raise SystemExit(f"size {n} has disagreeing Energy lines: {set(tokens)}")
    if len(tokens) != 1:
        raise SystemExit(f"size {n} has no structure file in the repository")
    token, path = next(iter(tokens.items()))
    return token, str(path.relative_to(ROOT)), sha256(path)


def main() -> None:
    found = local_files()
    rows = [choose(found, n) for n in SIZES]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    lines = ["n,energy,source,sha256"]
    lines.extend(f"{n},{energy},{source},{digest}" for n, (energy, source, digest) in zip(SIZES, rows))
    text = "\n".join(lines) + "\n"
    OUT.write_text(text)
    copy = BUILD / "tip4p_ccd.csv"
    BUILD.mkdir(parents=True, exist_ok=True)
    copy.write_text(text)
    sys.stdout.write(text)


if __name__ == "__main__":
    main()
