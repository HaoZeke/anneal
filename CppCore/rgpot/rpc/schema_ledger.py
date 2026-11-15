#!/usr/bin/env python3
"""Append-only ordinal ledger for Potentials.capnp.

Cap'n Proto identifies a field, union arm, enumerant or method on the wire
by its ordinal. An ordinal may never be renumbered, reused or given another
type, and peers (gpridge, eOn, other potentials-schema consumers) only agree
when every copy keeps upstream's numbering. Potentials.ordinals records each
ordinal once as

    <scope> @<ordinal> <kind> <name> <type> <origin>

kind is field, arm (union member), group, enumerant or method. origin is
upstream (in the pinned potentials-schema commit) or local (added by a diff
in the local series and shipped by rgpot).

The two vendored copies are the pinned upstream file, byte for byte, which
is what a parent project such as gpridge compares them with. The local
series (CppCore/rgpot/rpc/local/series, mirrored in rgpot-core/schema/local
for the crate) lists add-only diffs that apply on top of the vendored copy;
the result is the effective schema that rgpot compiles, installs and serves.

check fails when

* a ledger line changes or disappears relative to --base (append only),
* the unpatched pinned upstream schema drops, renumbers or retypes an
  upstream entry, declares one the ledger lacks, or collides with a local
  entry,
* a vendored copy is not byte for byte the pinned upstream schema, or
  declares anything but the upstream ledger entries,
* the effective schema differs from the ledger in any entry,
* a series diff removes or rewrites a line carrying an @N ordinal, or adds
  one that is not a local ledger entry,
* the series mirror differs from the series, or the wrap applies
  diff_files to the pinned checkout.

append adds the entries the effective schema declares and the ledger lacks,
at the end of the ledger. It never rewrites an existing line.

apply writes the effective schema: the vendored copy plus the series.
"""

from __future__ import annotations

import argparse
import configparser
import dataclasses
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

LEDGER = Path("CppCore/rgpot/rpc/Potentials.ordinals")
VENDORED = (
    Path("CppCore/rgpot/rpc/Potentials.capnp"),
    Path("rgpot-core/schema/Potentials.capnp"),
)
WRAP = Path("subprojects/potentials-schema.wrap")
SERIES_DIRS = (
    Path("CppCore/rgpot/rpc/local"),
    Path("rgpot-core/schema/local"),
)
SERIES = "series"
SCHEMA_NAME = "Potentials.capnp"
DIVERGES = "vendored Potentials.capnp diverges from pinned potentials-schema release"

ORDINAL = re.compile(r"@\d+\b")


@dataclasses.dataclass(frozen=True)
class Entry:
    scope: str
    ordinal: int
    kind: str
    name: str
    type: str
    origin: str = ""

    @property
    def key(self) -> tuple[str, int]:
        return (self.scope, self.ordinal)

    def same(self, other: "Entry") -> bool:
        return (self.scope, self.ordinal, self.kind, self.name, self.type) == (
            other.scope,
            other.ordinal,
            other.kind,
            other.name,
            other.type,
        )

    def describe(self) -> str:
        return f"{self.scope} @{self.ordinal} {self.kind} {self.name} {self.type}"

    def line(self) -> str:
        return f"{self.describe()} {self.origin}"


class LedgerError(Exception):
    pass


# ---------------------------------------------------------------- parsing


def _strip_comment(line: str) -> str:
    quoted = False
    escaped = False
    for i, ch in enumerate(line):
        if escaped:
            escaped = False
        elif ch == "\\" and quoted:
            escaped = True
        elif ch == '"':
            quoted = not quoted
        elif ch == "#" and not quoted:
            return line[:i]
    return line


def _strip_annotations(text: str) -> str:
    """Drop $annotation and $annotation(...) applications."""
    out = []
    i = 0
    while i < len(text):
        if text[i] != "$":
            out.append(text[i])
            i += 1
            continue
        i += 1
        while i < len(text) and (text[i].isalnum() or text[i] in "._"):
            i += 1
        if i < len(text) and text[i] == "(":
            depth = 0
            while i < len(text):
                depth += text[i] == "("
                depth -= text[i] == ")"
                i += 1
                if depth == 0:
                    break
    return "".join(out)


def _statements(text: str):
    """Yield (token, statement, line) with token one of ';', '{', '}'."""
    buf: list[str] = []
    start = 0
    depth = 0
    quoted = False
    for lineno, raw in enumerate(text.splitlines(), 1):
        for ch in _strip_comment(raw) + "\n":
            if quoted:
                buf.append(ch)
                if ch == '"' and (len(buf) < 2 or buf[-2] != "\\"):
                    quoted = False
                continue
            if ch == '"':
                quoted = True
            elif ch in "([":
                depth += 1
            elif ch in ")]":
                depth -= 1
            elif depth == 0 and ch in ";{}":
                yield ch, " ".join("".join(buf).split()), start
                buf = []
                start = 0
                continue
            if not start and not ch.isspace():
                start = lineno
            buf.append(ch)


_DECL = re.compile(r"^(struct|enum|interface)\s+(\w+)")
_GROUP = re.compile(r"^(\w+)\s*(?:@(\d+)\s*)?:\s*(union|group)$")
_FIELD = re.compile(r"^(\w+)\s*@(\d+)\s*:\s*(.+)$")
_ENUMERANT = re.compile(r"^(\w+)\s*@(\d+)$")
_METHOD = re.compile(r"^(\w+)\s*@(\d+)\s*(\(.*)$")


def _field_type(rest: str) -> str:
    depth = 0
    for i, ch in enumerate(rest):
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        elif ch == "=" and depth == 0:
            rest = rest[:i]
            break
    return ":" + "".join(rest.split())


def parse_schema(text: str) -> list[tuple[Entry, int]]:
    """Every ordinal a Cap'n Proto schema declares, with its source line."""
    # frame: kind, ordinal owner scope, name prefix, decl path
    stack: list[tuple[str, str, str, str]] = [("file", "", "", "")]
    found: list[tuple[Entry, int]] = []
    for token, stmt, line in _statements(text):
        kind, owner, prefix, path = stack[-1]
        stmt = " ".join(_strip_annotations(stmt).split())
        if token == "}":
            if stmt:
                raise LedgerError(f"line {line}: unterminated statement {stmt!r}")
            if len(stack) == 1:
                raise LedgerError(f"line {line}: unbalanced closing brace")
            stack.pop()
            continue
        if token == "{":
            decl = _DECL.match(stmt)
            group = _GROUP.match(stmt)
            if decl:
                name = f"{path}.{decl.group(2)}" if path else decl.group(2)
                stack.append((decl.group(1), name, "", name))
            elif stmt == "union" and kind in ("struct", "group", "union"):
                stack.append(("union", owner, prefix, path))
            elif group and kind in ("struct", "group", "union"):
                gname, gord, gkind = group.groups()
                if gord is not None:
                    container = "arm" if kind == "union" else "group"
                    found.append((Entry(owner, int(gord), container, prefix + gname, gkind), line))
                stack.append(("union" if gkind == "union" else "group", owner, f"{prefix}{gname}.", path))
            else:
                stack.append(("other", owner, prefix, path))
            continue
        if not stmt or kind in ("file", "other"):
            continue
        if stmt.split()[0] in ("using", "const", "annotation"):
            continue
        if kind in ("struct", "group", "union"):
            m = _FIELD.match(stmt)
            if m:
                container = "arm" if kind == "union" else "field"
                found.append(
                    (
                        Entry(
                            owner, int(m.group(2)), container, prefix + m.group(1), _field_type(m.group(3))
                        ),
                        line,
                    )
                )
            elif ORDINAL.search(stmt):
                raise LedgerError(f"line {line}: cannot read {stmt!r}")
        elif kind == "enum":
            m = _ENUMERANT.match(stmt)
            if not m:
                raise LedgerError(f"line {line}: cannot read enumerant {stmt!r}")
            found.append((Entry(owner, int(m.group(2)), "enumerant", m.group(1), "-"), line))
        elif kind == "interface":
            m = _METHOD.match(stmt)
            if not m:
                raise LedgerError(f"line {line}: cannot read method {stmt!r}")
            found.append(
                (Entry(owner, int(m.group(2)), "method", m.group(1), "".join(m.group(3).split())), line)
            )
    if len(stack) != 1:
        raise LedgerError(f"unclosed {stack[-1][0]} {stack[-1][3]}")
    return found


def schema_entries(text: str, label: str) -> list[Entry]:
    try:
        found = [entry for entry, _ in parse_schema(text)]
    except LedgerError as exc:
        raise LedgerError(f"{label}: {exc}") from exc
    _unique(found, label)
    return found


def _unique(entries: list[Entry], label: str) -> None:
    keys: dict[tuple[str, int], Entry] = {}
    names: dict[tuple[str, str], Entry] = {}
    for entry in entries:
        if entry.key in keys:
            raise LedgerError(
                f"{label}: {entry.describe()} reuses the ordinal of {keys[entry.key].describe()}"
            )
        twin = names.get((entry.scope, entry.name))
        if twin is not None:
            raise LedgerError(f"{label}: {entry.describe()} reuses the name of {twin.describe()}")
        keys[entry.key] = entry
        names[(entry.scope, entry.name)] = entry


# ----------------------------------------------------------------- ledger

_LEDGER_LINE = re.compile(r"^(\S+) @(\d+) (field|arm|group|enumerant|method) (\S+) (\S+) (upstream|local)$")


def parse_ledger(text: str, label: str) -> list[Entry]:
    entries = []
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _LEDGER_LINE.match(line)
        if not m:
            raise LedgerError(f"{label}:{lineno}: malformed ledger line {raw!r}")
        scope, ordinal, kind, name, kind_type, origin = m.groups()
        entries.append(Entry(scope, int(ordinal), kind, name, kind_type, origin))
    _unique(entries, label)
    return entries


LEDGER_HEADER = """\
# Append-only ordinal ledger for Potentials.capnp (see schema_ledger.py).
# <scope> @<ordinal> <kind> <name> <type> <origin>
# upstream: in the pinned potentials-schema commit.
# local: added by a diff in the local series. Never edit or delete a line;
# append only.
"""


# ------------------------------------------------------------- unified diff


@dataclasses.dataclass
class Hunk:
    old_start: int
    lines: list[str]


def parse_diff(text: str, label: str) -> dict[str, list[Hunk]]:
    files: dict[str, list[Hunk]] = {}
    current: list[Hunk] | None = None
    hunk: Hunk | None = None
    for raw in text.splitlines():
        if raw.startswith("+++ "):
            target = raw[4:].split("\t")[0].strip()
            target = target[2:] if target.startswith("b/") else target
            current = files.setdefault(target, [])
            hunk = None
        elif raw.startswith("@@ "):
            m = re.match(r"^@@ -(\d+)(?:,\d+)? \+\d+(?:,\d+)? @@", raw)
            if not m or current is None:
                raise LedgerError(f"{label}: bad hunk header {raw!r}")
            hunk = Hunk(int(m.group(1)), [])
            current.append(hunk)
        elif hunk is not None and raw[:1] in (" ", "+", "-"):
            hunk.lines.append(raw)
        elif hunk is not None and raw == "":
            hunk.lines.append(" ")
        elif raw.startswith("\\"):
            continue
        else:
            hunk = None
    return files


def apply_hunks(text: str, hunks: list[Hunk], label: str) -> tuple[str, list[int]]:
    """Apply hunks like patch -p1 (exact context, any offset).

    Returns the new text and the 1-based new line number of every added line.
    """
    lines = text.splitlines()
    out: list[str] = []
    added: list[int] = []
    pos = 0
    for number, hunk in enumerate(hunks, 1):
        old = [ln[1:] for ln in hunk.lines if ln[0] in (" ", "-")]
        new_kinds = [(ln[0], ln[1:]) for ln in hunk.lines if ln[0] in (" ", "+")]
        want = hunk.old_start - 1 if old else hunk.old_start
        candidates = sorted(range(pos, len(lines) - len(old) + 1), key=lambda i: abs(i - want))
        at = next((i for i in candidates if lines[i : i + len(old)] == old), None)
        if at is None:
            raise LedgerError(f"{label}: hunk {number} does not apply")
        out.extend(lines[pos:at])
        for kind, body in new_kinds:
            out.append(body)
            if kind == "+":
                added.append(len(out))
        pos = at + len(old)
    out.extend(lines[pos:])
    return "\n".join(out) + ("\n" if text.endswith("\n") or not text else ""), added


# ------------------------------------------------------------------- inputs


@dataclasses.dataclass
class Wrap:
    url: str
    revision: str
    diff_files: list[str]


def read_wrap(root: Path) -> Wrap:
    parser = configparser.ConfigParser()
    parser.read(root / WRAP, encoding="utf-8")
    section = parser["wrap-git"]
    names = [n.strip() for n in section.get("diff_files", "").split(",") if n.strip()]
    return Wrap(section["url"], section["revision"], names)


def read_series(directory: Path) -> list[Path]:
    """The diffs a series file lists, in order. No series file means none."""
    path = directory / SERIES
    if not path.is_file():
        return []
    names = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        name = raw.strip()
        if name and not name.startswith("#"):
            names.append(directory / name)
    return names


def apply_series(text: str, diffs: list[Path]) -> str:
    for diff in diffs:
        text, _ = apply_hunks(text, schema_hunks(diff), diff.name)
    return text


def schema_hunks(diff: Path) -> list[Hunk]:
    files = parse_diff(diff.read_text(encoding="utf-8"), diff.name)
    other = sorted(set(files) - {SCHEMA_NAME})
    if other:
        raise LedgerError(
            f"{diff.name} patches {', '.join(other)}; schema diffs may only touch {SCHEMA_NAME}"
        )
    return files.get(SCHEMA_NAME, [])


def _git(*args: str, cwd: Path | None = None) -> str:
    # The schema is UTF-8; the locale's code page (cp1252 on Windows) would
    # turn its non-ASCII comments into different text.
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
    ).stdout


def fetch_upstream(wrap: Wrap) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        _git("init", "-q", cwd=Path(tmp))
        _git("fetch", "-q", "--depth", "1", wrap.url, wrap.revision, cwd=Path(tmp))
        head = _git("rev-parse", "FETCH_HEAD", cwd=Path(tmp)).strip()
        if head != wrap.revision:
            raise LedgerError(f"fetched {head}, the wrap pins {wrap.revision}")
        return _git("show", f"{wrap.revision}:{SCHEMA_NAME}", cwd=Path(tmp))


def upstream_from_subproject(schema: Path, wrap: Wrap) -> tuple[str, str]:
    """Upstream text from a meson checkout of the wrap.

    The git object at the pin is preferred, so a parent project that patches
    its own checkout still yields the pinned file. Otherwise the file is read
    as it is, and the canonical check compares the vendored copies with it.
    """
    directory = schema.resolve().parent
    try:
        top = Path(_git("rev-parse", "--show-toplevel", cwd=directory).strip()).resolve()
        if top == directory and _git("rev-parse", "HEAD", cwd=directory).strip() == wrap.revision:
            return _git("show", f"{wrap.revision}:{SCHEMA_NAME}", cwd=directory), "git object"
    except (OSError, subprocess.CalledProcessError):
        pass
    return schema.read_text(encoding="utf-8"), str(schema)


def base_ledger(root: Path, base: str) -> str | None:
    try:
        _git("rev-parse", "--verify", "--quiet", f"{base}^{{commit}}", cwd=root)
    except subprocess.CalledProcessError as exc:
        raise LedgerError(f"base revision {base} is not a commit here") from exc
    try:
        return _git("show", f"{base}:{LEDGER.as_posix()}", cwd=root)
    except subprocess.CalledProcessError:
        return None


# ------------------------------------------------------------------- checks


def check_append_only(base: list[Entry], current: list[Entry]) -> list[str]:
    errors = []
    position = {e.key: i for i, e in enumerate(current)}
    old_keys = {e.key for e in base}
    last = -1
    for old in base:
        at = position.get(old.key)
        if at is None:
            errors.append(f"ledger entry disappeared: {old.line()}")
        elif current[at] != old:
            errors.append(f"ledger entry changed: {old.line()} is now {current[at].line()}")
        elif at < last:
            errors.append(f"ledger entry moved: {old.line()}")
        else:
            last = at
    for entry in current[: last + 1]:
        if entry.key not in old_keys:
            errors.append(f"ledger entry inserted before existing ones: {entry.line()} (append at the end)")
    return errors


def check_upstream(ledger: list[Entry], upstream: list[Entry]) -> list[str]:
    errors = []
    up_key = {e.key: e for e in upstream}
    up_name = {(e.scope, e.name): e for e in upstream}
    led_key = {e.key: e for e in ledger}
    for entry in ledger:
        theirs = up_key.get(entry.key)
        if entry.origin == "upstream":
            if theirs is None:
                errors.append(f"pinned upstream no longer declares {entry.line()}")
            elif not theirs.same(entry):
                errors.append(f"pinned upstream has {theirs.describe()}, ledger has {entry.line()}")
        else:
            if theirs is not None and not theirs.same(entry):
                errors.append(f"local {entry.describe()} collides with upstream {theirs.describe()}")
            other = up_name.get((entry.scope, entry.name))
            if other is not None and other.ordinal != entry.ordinal:
                errors.append(f"local {entry.describe()} collides with upstream {other.describe()}")
    for entry in upstream:
        if entry.key not in led_key:
            errors.append(f"pinned upstream declares {entry.describe()}, not in the ledger (run append)")
    return errors


def check_copy(ledger: list[Entry], copy: list[Entry], label: str) -> list[str]:
    errors = []
    mine = {e.key: e for e in copy}
    led = {e.key: e for e in ledger}
    for entry in ledger:
        have = mine.get(entry.key)
        if have is None:
            errors.append(f"{label} lacks {entry.line()}")
        elif not have.same(entry):
            errors.append(f"{label} has {have.describe()}, ledger has {entry.line()}")
    for entry in copy:
        if entry.key not in led:
            errors.append(f"{label} declares {entry.describe()}, not in the ledger (run append)")
    return errors


def check_vendored(ledger: list[Entry], copy: list[Entry], label: str) -> list[str]:
    """A vendored copy holds the upstream entries and none of the local ones."""
    errors = []
    local = {e.key: e for e in ledger if e.origin == "local"}
    rest = []
    for entry in copy:
        mine = local.get(entry.key)
        if mine is not None and mine.same(entry):
            errors.append(f"{label} carries local {mine.describe()}; it belongs in the local series")
        else:
            rest.append(entry)
    return errors + check_copy([e for e in ledger if e.origin == "upstream"], rest, label)


def check_series_mirror(root: Path) -> list[str]:
    errors = []
    main, mirror = (root / d for d in SERIES_DIRS)
    names = {p.name for p in main.iterdir() if p.is_file()} if main.is_dir() else set()
    copies = {p.name for p in mirror.iterdir() if p.is_file()} if mirror.is_dir() else set()
    for name in sorted(names | copies):
        if name not in copies:
            errors.append(f"{SERIES_DIRS[1] / name} is missing; copy {SERIES_DIRS[0] / name}")
        elif name not in names:
            errors.append(f"{SERIES_DIRS[1] / name} has no counterpart in {SERIES_DIRS[0]}")
        elif (main / name).read_bytes().replace(b"\r\n", b"\n") != (mirror / name).read_bytes().replace(
            b"\r\n", b"\n"
        ):
            errors.append(f"{SERIES_DIRS[1] / name} differs from {SERIES_DIRS[0] / name}")
    return errors


def check_diffs(base_text: str, diffs: list[Path], ledger: list[Entry]) -> tuple[list[str], str]:
    errors = []
    local = {e.key: e for e in ledger if e.origin == "local"}
    by_key = {e.key: e for e in ledger}
    text = base_text
    for diff in diffs:
        if not diff.is_file():
            errors.append(f"series diff {diff.name} is missing")
            continue
        hunks = schema_hunks(diff)
        for hunk in hunks:
            for line in hunk.lines:
                if line[0] == "-" and ORDINAL.search(line):
                    errors.append(f"{diff.name} removes or rewrites an ordinal line: {line[1:].strip()}")
        try:
            text, added = apply_hunks(text, hunks, diff.name)
        except LedgerError as exc:
            errors.append(str(exc))
            continue
        try:
            at_line = {line: entry for entry, line in parse_schema(text)}
        except LedgerError as exc:
            errors.append(f"{diff.name}: {exc}")
            continue
        lines = text.splitlines()
        for number in added:
            body = lines[number - 1]
            if not ORDINAL.search(body):
                continue
            entry = at_line.get(number)
            if entry is None:
                errors.append(f"{diff.name} adds an ordinal outside a declaration: {body.strip()}")
                continue
            mine = local.get(entry.key)
            if mine is None or not mine.same(entry):
                have = by_key.get(entry.key)
                if have is None:
                    where = ", not a local ledger entry"
                elif have.same(entry):
                    where = ", which the ledger records as upstream; a wrap diff may only add local entries"
                else:
                    where = f", ledger has {have.line()}"
                errors.append(f"{diff.name} adds {entry.describe()}{where}")
    return errors, text


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8").replace("\r\n", "\n")


def run_check(args: argparse.Namespace) -> int:
    root = args.root.resolve()
    wrap = read_wrap(root)
    ledger_path = args.ledger or root / LEDGER
    ledger = parse_ledger(ledger_path.read_text(encoding="utf-8"), str(ledger_path))
    errors: list[str] = []
    notes: list[str] = []

    if args.base:
        old = base_ledger(root, args.base)
        if old is None:
            notes.append(f"{LEDGER} does not exist at {args.base}; append-only check skipped")
        else:
            errors += check_append_only(parse_ledger(old, f"{args.base}:{LEDGER}"), ledger)

    if wrap.diff_files:
        errors.append(
            f"{WRAP} applies diff_files ({', '.join(wrap.diff_files)}); a parent project resolves "
            f"the pin from its own wrap without them. List rgpot-local diffs in "
            f"{(SERIES_DIRS[0] / SERIES).as_posix()} instead"
        )
    errors += check_series_mirror(root)

    texts = {rel: _read(root / rel) for rel in VENDORED}
    for rel, text in texts.items():
        errors += check_vendored(ledger, schema_entries(text, str(rel)), str(rel))
    first = texts[VENDORED[0]]
    for rel in VENDORED[1:]:
        if texts[rel] != first:
            errors.append(f"{rel} differs from {VENDORED[0]}")

    upstream_text = None
    if args.upstream:
        upstream_text, source = _read(args.upstream), str(args.upstream)
    elif args.subproject_schema:
        upstream_text, source = upstream_from_subproject(args.subproject_schema, wrap)
    elif args.fetch:
        upstream_text, source = fetch_upstream(wrap), f"{wrap.url} at {wrap.revision}"
    if upstream_text is None:
        notes.append("no pinned upstream schema given; upstream and canonical checks skipped")
    else:
        upstream_text = upstream_text.replace("\r\n", "\n")
        notes.append(f"pinned upstream: {source}")
        errors += check_upstream(ledger, schema_entries(upstream_text, "pinned upstream"))
        for rel, text in texts.items():
            if text != upstream_text:
                errors.append(f"{rel}: {DIVERGES} {wrap.revision[:8]}")

    diffs = read_series(root / SERIES_DIRS[0])
    diff_errors, effective = check_diffs(first if upstream_text is None else upstream_text, diffs, ledger)
    errors += diff_errors
    if not diff_errors:
        errors += check_copy(ledger, schema_entries(effective, "effective schema"), "effective schema")

    for note in notes:
        print(f"note: {note}", flush=True)
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        print(f"{len(errors)} ordinal ledger violation(s)", file=sys.stderr)
        return 1
    print(f"ordinal ledger OK: {len(ledger)} entries")
    return 0


def run_append(args: argparse.Namespace) -> int:
    root = args.root.resolve()
    wrap = read_wrap(root)
    ledger_path = args.ledger or root / LEDGER
    text = ledger_path.read_text(encoding="utf-8") if ledger_path.exists() else LEDGER_HEADER
    ledger = parse_ledger(text, str(ledger_path))
    upstream_text = args.upstream.read_text(encoding="utf-8") if args.upstream else fetch_upstream(wrap)
    upstream = {e.key: e for e in schema_entries(upstream_text, "pinned upstream")}
    known = {e.key: e for e in ledger}
    effective = apply_series(_read(root / VENDORED[0]), read_series(root / SERIES_DIRS[0]))
    copy = schema_entries(effective, "effective schema")
    fresh = []
    for entry in copy:
        old = known.get(entry.key)
        if old is not None:
            if not old.same(entry):
                raise LedgerError(
                    f"effective schema has {entry.describe()}, ledger has {old.line()}; "
                    "append will not rewrite it"
                )
            continue
        theirs = upstream.get(entry.key)
        origin = "upstream" if theirs is not None and theirs.same(entry) else "local"
        fresh.append(dataclasses.replace(entry, origin=origin))
    if fresh:
        if not text.endswith("\n"):
            text += "\n"
        text += "".join(e.line() + "\n" for e in fresh)
        ledger_path.write_text(text, encoding="utf-8", newline="\n")
    print(f"appended {len(fresh)} entries to {ledger_path}")
    return 0


def run_apply(args: argparse.Namespace) -> int:
    root = args.root.resolve()
    schema = args.schema or root / VENDORED[0]
    text = apply_series(_read(schema), read_series(args.series or root / SERIES_DIRS[0]))
    # Rewriting identical bytes would make every capnp consumer rebuild.
    if not args.out.is_file() or args.out.read_text(encoding="utf-8") != text:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8", newline="\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "append", "apply"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    for name in ("check", "append"):
        cmd = sub.choices[name]
        cmd.add_argument("--ledger", type=Path)
        cmd.add_argument("--upstream", type=Path, help="unpatched pinned Potentials.capnp")
    apply = sub.choices["apply"]
    apply.add_argument("--schema", type=Path, help="vendored Potentials.capnp (default: the C++ copy)")
    apply.add_argument("--series", type=Path, help="directory holding the series file")
    apply.add_argument("--out", type=Path, required=True, help="where to write the effective schema")
    check = sub.choices["check"]
    source = check.add_mutually_exclusive_group()
    source.add_argument(
        "--subproject-schema", type=Path, help="Potentials.capnp from the meson checkout of the wrap"
    )
    source.add_argument("--fetch", action="store_true", help="fetch the pinned revision from the wrap url")
    check.add_argument("--base", help="git revision whose ledger must be a prefix of this one")
    args = parser.parse_args(argv)
    if args.command == "check" and args.upstream and (args.subproject_schema or args.fetch):
        parser.error("give one of --upstream, --subproject-schema and --fetch")
    try:
        return {"check": run_check, "append": run_append, "apply": run_apply}[args.command](args)
    except (LedgerError, OSError, subprocess.CalledProcessError, KeyError) as exc:
        detail = getattr(exc, "stderr", None)
        print(f"error: {exc}{': ' + detail.strip() if detail else ''}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    os.environ.setdefault("GIT_TERMINAL_PROMPT", "0")
    raise SystemExit(main())
