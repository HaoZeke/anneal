# Copyright (c) anneal developers
"""Per-size Lennard-Jones GMIN data files from one radius rule.

The keyword lines of ``examples/h2h_gmin_lj.data`` are copied unchanged.
``RADIUS`` is the container from ``bench/gmin-panel/lj-datafile-rule.md``.
"""

from __future__ import annotations

from pathlib import Path

# The literal in source/io1.f90, not a recomputed 4*pi*sqrt(2).
DENSITY = 17.77153175
SIZES = (
    ("lj38", 38),
    ("lj55", 55),
    ("lj75", 75),
    ("lj98", 98),
    ("lj102", 102),
    ("lj103", 103),
    ("lj104", 104),
    ("lj150", 150),
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def base_path(root: Path | None = None) -> Path:
    return (root or repo_root()) / "examples" / "h2h_gmin_lj.data"


def data_dir(root: Path | None = None) -> Path:
    return (root or repo_root()) / "bench" / "gmin-panel" / "lj-data"


def radius(n: int) -> float:
    """Default Lennard-Jones container in sigma, for ``n`` atoms."""
    return (2.0 + (3.0 * n / DENSITY) ** (1.0 / 3.0)) * 2.0 ** (1.0 / 6.0)


def radius_text(n: int) -> str:
    return f"{radius(n):.10f}"


def keyword_body(base_text: str) -> str:
    """Keyword lines of the LJ38 file, comments removed, order kept."""
    lines: list[str] = []
    started = False
    for line in base_text.splitlines():
        if line.startswith("!"):
            continue
        if not started:
            if not line.strip():
                continue
            started = True
        lines.append(line)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines) + "\n"


def render(n: int, base_text: str) -> str:
    header = (
        "! Per-size Lennard-Jones input.\n"
        "! Keyword lines match examples/h2h_gmin_lj.data.\n"
        "! RADIUS follows bench/gmin-panel/lj-datafile-rule.md.\n"
        "! The driver appends one RANSEED line per run.\n"
        "\n"
    )
    return header + keyword_body(base_text) + f"RADIUS {radius_text(n)}\n"


def atoms(system: str) -> int:
    for name, n in SIZES:
        if name == system:
            return n
    raise KeyError(system)


def path_for(system: str, root: Path | None = None) -> Path:
    atoms(system)
    return data_dir(root) / f"{system}.data"


def check(system: str, text: str, root: Path | None = None) -> None:
    """Refuse a file whose keywords or container are not the rule for ``system``."""
    n = atoms(system)
    base = keyword_body(base_path(root).read_text())
    keywords = [line for line in text.splitlines() if line.strip() and not line.startswith("!")]
    radii = [line for line in keywords if line.startswith("RADIUS ")]
    rest = [line for line in keywords if not line.startswith("RADIUS ")]
    base_keywords = [line for line in base.splitlines() if line.strip()]
    if rest != base_keywords:
        raise RuntimeError(f"{system} keywords differ from the LJ38 file")
    want = f"RADIUS {radius_text(n)}"
    if radii != [want]:
        raise RuntimeError(f"{system} radius is {radii}, expected {want}")


def load(system: str, root: Path | None = None) -> str:
    path = path_for(system, root)
    text = path.read_text()
    if not text.endswith("\n"):
        text += "\n"
    check(system, text, root)
    return text


def write_all(root: Path | None = None) -> None:
    root = root or repo_root()
    base = base_path(root).read_text()
    out = data_dir(root)
    out.mkdir(parents=True, exist_ok=True)
    for system, n in SIZES:
        (out / f"{system}.data").write_text(render(n, base))


def main() -> None:
    write_all()


if __name__ == "__main__":
    main()
