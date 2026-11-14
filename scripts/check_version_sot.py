#!/usr/bin/env python3
"""Fail when a package version string disagrees with the workspace manifest.

The canonical value is ``[package].version`` in the workspace ``Cargo.toml``.
Generated copies are checked with cogapp (https://nedbatchelder.com/code/cog/).
Towncrier reads the same field (https://towncrier.readthedocs.io/en/stable/).
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import anneal_version

SEMVER = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")


def fail(errors: list[str], message: str) -> None:
    errors.append(message)
    print(message, file=sys.stderr)


def package_version() -> str:
    version = anneal_version.package_version()
    if not isinstance(version, str) or not SEMVER.fullmatch(version):
        raise SystemExit(f"canonical version is not a plain triple: {version!r}")
    return version


def one_match(text: str, pattern: str, label: str, errors: list[str]) -> str | None:
    found = re.findall(pattern, text, flags=re.M)
    if len(found) != 1:
        fail(errors, f"{label}: expected one match, found {found!r}")
        return None
    return found[0]


def main() -> int:
    errors: list[str] = []
    version = package_version()
    print(f"canonical {version}")

    pyproject = anneal_version.load_toml(ROOT / "pyproject.toml")
    if pyproject["project"]["version"] != version:
        fail(errors, f"pyproject.toml has {pyproject['project']['version']}")

    pixi = anneal_version.load_toml(ROOT / "pixi.toml")
    if pixi["workspace"]["version"] != version:
        fail(errors, f"pixi.toml has {pixi['workspace']['version']}")

    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    cited = one_match(citation, r"^version:\s*(\S+)\s*$", "CITATION.cff", errors)
    if cited is not None and cited.strip('"') != version:
        fail(errors, f"CITATION.cff version is {cited}")

    meson = (ROOT / "meson.build").read_text(encoding="utf-8")
    meson_version = one_match(
        meson, r"^\s*version:\s*'([^']+)'", "meson.build", errors
    )
    if meson_version is not None and meson_version != version:
        fail(errors, f"meson.build version is {meson_version}")

    header = (ROOT / "include" / "anneal-version.h").read_text(encoding="utf-8")
    macro = one_match(
        header,
        r'^#define ANNEAL_CORE_VERSION "([^"]+)"',
        "include/anneal-version.h",
        errors,
    )
    if macro is not None and macro != version:
        fail(errors, f"ANNEAL_CORE_VERSION is {macro}")

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    badge = one_match(
        readme,
        r"badge/version-([0-9]+\.[0-9]+\.[0-9]+)-",
        "README.md badge",
        errors,
    )
    if badge is not None and badge != version:
        fail(errors, f"README badge is {badge}")
    if f"/releases/tag/v{version}" not in readme:
        fail(errors, "README release link does not use the canonical version")

    version_rs = (ROOT / "src" / "version.rs").read_text(encoding="utf-8")
    if 'env!("CARGO_PKG_VERSION")' not in version_rs:
        fail(errors, "src/version.rs does not read CARGO_PKG_VERSION")
    if re.search(r'"[0-9]+\.[0-9]+\.[0-9]+"', version_rs):
        fail(errors, "src/version.rs still contains a handwritten version")

    cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    if "Cargo.toml" not in cmake or "VERSION ${ANNEAL_VERSION}" not in cmake:
        fail(errors, "CMakeLists.txt does not read the workspace manifest")
    if re.search(r"project\(anneal-core VERSION [0-9]", cmake):
        fail(errors, "CMakeLists.txt still hardcodes the project version")

    conf = (ROOT / "docs" / "source" / "conf.py").read_text(encoding="utf-8")
    if "anneal_version" not in conf or "package_version" not in conf:
        fail(errors, "docs/source/conf.py does not read the workspace manifest")
    if re.search(r'(?m)^(version|release)\s*=\s*["\'][0-9]', conf):
        fail(errors, "docs/source/conf.py hardcodes a Sphinx version")

    towncrier = (ROOT / "towncrier.toml").read_text(encoding="utf-8")
    if any(line.strip().startswith("version") for line in towncrier.splitlines()):
        fail(errors, "towncrier.toml still has its own version")
    if 'package = "anneal_version"' not in towncrier:
        fail(errors, "towncrier.toml does not import anneal_version")
    if 'package_dir = "scripts"' not in towncrier:
        fail(errors, "towncrier.toml does not point package_dir at scripts")
    if "/tree/v{version}" not in towncrier:
        fail(errors, "towncrier title_format lost the version link")

    cargo_text = (ROOT / "Cargo.toml").read_text(encoding="utf-8")
    first_version = next(
        (
            line.split('"', 2)[1]
            for line in cargo_text.splitlines()
            if line.startswith('version = "')
        ),
        None,
    )
    # The first column-0 `version = "` line is [package].version.
    if first_version != version:
        fail(errors, f"first Cargo.toml version line is {first_version}")

    lock = anneal_version.load_toml(ROOT / "Cargo.lock")
    locked = [
        pkg.get("version")
        for pkg in lock.get("package", [])
        if pkg.get("name") == "anneal-core"
    ]
    if locked != [version]:
        fail(errors, f"Cargo.lock anneal-core version is {locked}")

    if anneal_version.__version__ != version:
        fail(errors, f"anneal_version.__version__ is {anneal_version.__version__}")

    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    heading = f"## [{version}](https://github.com/HaoZeke/anneal/tree/v{version})"
    if heading not in changelog:
        fail(errors, "CHANGELOG.md heading link does not match the canonical version")

    zenodo_path = ROOT / ".zenodo.json"
    zenodo = json.loads(zenodo_path.read_text(encoding="utf-8"))
    if zenodo.get("version") != version:
        fail(errors, f".zenodo.json version is {zenodo.get('version')}")
    if zenodo.get("creators") != anneal_version.citation_creators():
        fail(errors, ".zenodo.json creators do not match CITATION.cff")

    with tempfile.TemporaryDirectory() as tmp:
        fresh = Path(tmp) / "zenodo.json"
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "version_cog.py"),
                "-d",
                "-o",
                str(fresh),
                "packaging/zenodo.cog",
            ],
            cwd=ROOT,
            check=True,
        )
        generated = fresh.read_text(encoding="utf-8")
    committed = zenodo_path.read_text(encoding="utf-8")
    if generated != committed:
        fail(errors, ".zenodo.json does not match cog output from packaging/zenodo.cog")

    carriers = [
        ROOT / "pyproject.toml",
        ROOT / "pixi.toml",
        ROOT / "meson.build",
        ROOT / "CITATION.cff",
        ROOT / "include" / "anneal-version.h",
        ROOT / "README.md",
        ROOT / ".zenodo.json",
        ROOT / "src" / "version.rs",
        ROOT / "CMakeLists.txt",
        ROOT / "docs" / "source" / "conf.py",
        ROOT / "towncrier.toml",
    ]
    assignment = re.compile(
        r"(?m)^[ \t]*(?:version\s*=\s*|version:\s*|#define ANNEAL_CORE_VERSION\s+)"
        r"['\"]?([0-9]+\.[0-9]+\.[0-9]+)['\"]?"
    )
    for path in carriers:
        text = path.read_text(encoding="utf-8")
        for found in assignment.findall(text):
            if found != version:
                rel = path.relative_to(ROOT)
                fail(errors, f"{rel} assignment {found} != {version}")

    if errors:
        print(f"{len(errors)} version drift error(s)", file=sys.stderr)
        return 1
    print("version strings match")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
