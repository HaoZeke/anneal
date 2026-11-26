# Copyright (c) anneal developers
"""Read the workspace package version for towncrier.

Towncrier imports this module when ``towncrier.toml`` names it.
https://towncrier.readthedocs.io/en/stable/
"""

from pathlib import Path


def _toml_module():
    """tomllib on 3.11 and newer, otherwise tomli when that module imports."""
    try:
        import tomllib
    except ModuleNotFoundError:
        tomllib = None
    if tomllib is not None:
        return tomllib
    try:
        import tomli
    except ModuleNotFoundError:
        return None
    return tomli


def _uncomment(line):
    """Drop a TOML comment that is not inside a basic string."""
    out = []
    quote = ""
    for ch in line:
        if quote:
            out.append(ch)
            if ch == quote:
                quote = ""
            continue
        if ch in "\"'":
            quote = ch
            out.append(ch)
            continue
        if ch == "#":
            break
        out.append(ch)
    return "".join(out).strip()


def _table_header(line):
    """Return a standard table name, or None for any other line."""
    if len(line) < 2 or not line.startswith("[") or not line.endswith("]"):
        return None
    if line.startswith("[["):
        return None
    parts = [part.strip() for part in line[1:-1].split(".")]
    if not parts or any(not part or not part.replace("_", "a").isalnum() for part in parts):
        return None
    return ".".join(parts)


def _basic_string(value):
    if len(value) < 2 or value[0] not in "\"'" or value[-1] != value[0]:
        return None
    body = value[1:-1]
    if "\\" in body or value[0] in body:
        return None
    return body


def _table_string(text, table, key):
    """Read one basic string from a TOML table without a TOML library.

    This is only the fallback for the workspace version and the package
    name. It does not parse documents. A value it cannot read is None,
    and the caller fails instead of returning some other version.
    """
    current = None
    for raw in text.splitlines():
        line = _uncomment(raw)
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            current = _table_header(line)
            continue
        if current != table or not line.startswith(key):
            continue
        rest = line[len(key) :].lstrip()
        if not rest.startswith("="):
            continue
        found = _basic_string(rest[1:].strip())
        if found is not None:
            return found
    return None


def load_toml(path):
    """Load a TOML document.

    Python 3.11 and newer use tomllib. Python 3.10 uses tomli when it
    is installed. A whole document still needs one of those libraries.
    """
    module = _toml_module()
    if module is None:
        raise SystemExit(
            "reading this TOML file needs tomllib or tomli "
            "(Python 3.10: the dev dependency group)"
        )
    return module.loads(Path(path).read_text(encoding="utf-8"))


def _read(path):
    text = Path(path).read_text(encoding="utf-8")
    module = _toml_module()
    if module is None:
        return text, None
    return text, module.loads(text)


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in [here.parent, *here.parents]:
        cargo = candidate / "Cargo.toml"
        if not cargo.is_file():
            continue
        text, loaded = _read(cargo)
        if loaded is None:
            name = _table_string(text, "package", "name")
        else:
            name = (loaded.get("package") or {}).get("name")
        if name == "anneal-core":
            return candidate
    raise SystemExit("workspace Cargo.toml for anneal-core was not found")


def package_version() -> str:
    """The ``[workspace.package]`` version.

    tomllib or tomli is used when it imports. When neither does, the
    version string is read from that table directly. The release
    workflow and its contract test call this, including on the Python
    3.10 CI cell, where cargo test runs before tomli is installed.
    """
    text, loaded = _read(repo_root() / "Cargo.toml")
    if loaded is None:
        version = _table_string(text, "workspace.package", "version")
    else:
        shared = (loaded.get("workspace") or {}).get("package") or {}
        version = shared.get("version")
    if not isinstance(version, str) or version == "":
        raise SystemExit("workspace package version is missing")
    return version


def citation_creators(path=None):
    """Zenodo creators taken from the CITATION.cff authors list."""
    text = Path(path or (repo_root() / "CITATION.cff")).read_text(encoding="utf-8")
    creators = []
    current = None
    in_authors = False
    for line in text.splitlines():
        if not in_authors:
            if line == "authors:":
                in_authors = True
            continue
        if line and not line.startswith((" ", "\t")):
            break
        if line.startswith("  - "):
            if current:
                creators.append(_creator(current))
            current = {}
            _store(current, line[4:])
        elif current is not None and line.startswith("    "):
            _store(current, line.strip())
    if current:
        creators.append(_creator(current))
    return creators


def _store(current, assignment):
    if ":" not in assignment:
        return
    key, value = assignment.split(":", 1)
    current[key.strip()] = value.strip().strip('"')


def _creator(fields):
    item = {"name": "%s, %s" % (fields.get("family-names", ""), fields.get("given-names", ""))}
    if fields.get("affiliation"):
        item["affiliation"] = fields["affiliation"]
    orcid = fields.get("orcid", "")
    if orcid:
        prefix = "https://orcid.org/"
        if orcid.startswith(prefix):
            orcid = orcid[len(prefix) :]
        item["orcid"] = orcid
    return item


__version__ = package_version()
