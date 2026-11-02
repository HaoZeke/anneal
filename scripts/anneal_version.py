"""Read the workspace package version for towncrier.

Towncrier imports this module when ``towncrier.toml`` names it.
https://towncrier.readthedocs.io/en/stable/
"""

from pathlib import Path


def load_toml(path):
    """Load a TOML file. Python 3.10 uses tomli; 3.11 and newer use tomllib."""
    try:
        import tomllib
    except ModuleNotFoundError:
        try:
            import tomli as tomllib
        except ModuleNotFoundError as exc:
            raise SystemExit(
                "Python 3.10 needs tomli from the dev dependency group"
            ) from exc
    return tomllib.loads(Path(path).read_text(encoding="utf-8"))


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in [here.parent, *here.parents]:
        cargo = candidate / "Cargo.toml"
        if not cargo.is_file():
            continue
        package = load_toml(cargo).get("package") or {}
        if package.get("name") == "anneal-core":
            return candidate
    raise SystemExit("workspace Cargo.toml for anneal-core was not found")


def package_version() -> str:
    version = load_toml(repo_root() / "Cargo.toml")["package"]["version"]
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
