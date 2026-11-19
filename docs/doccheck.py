#!/usr/bin/env python3
"""Check the documentation against the source tree.

Three checks, each a subcommand:

``blocks``
    Every code block in ``readme_src.org``, ``CONTRIBUTING.md``,
    ``docs/*.md``, ``docs/*.org`` and ``docs/orgmode/**/*.org`` carries a
    classification.  In org the classification is an ``#+attr_rgpot:``
    line above the block; in Markdown it is an ``<!-- rgpot-doc: ... -->``
    comment above the fence.  ``README.md`` must carry the same code blocks
    as ``readme_src.org``.  The static part of each class runs here; with
    ``--build-dir`` the ``run`` and ``compile`` classes execute.

``reference``
    Every public name has a reference entry under ``docs/orgmode/reference``
    with non-empty Inputs, Units, Outputs and Errors fields, and every
    signature written in an entry matches the source.  The entry of a
    ``*Config`` struct names each of its fields, and a reference page names
    each field the rgpot-local schema diffs add.

``facts``
    Statements that can be read off the source: meson and CMake options and
    their defaults, ``pixi`` tasks and environments, repository paths, and
    ``RGPOT_*`` identifiers, and the ``&RGPOT`` keys of the GROMACS drop-in.

``coverage``
    Every org page under ``docs/orgmode`` is reachable from
    ``docs/orgmode/index.org``, every design note named there exists, and
    the pinned citations and reference values are still on their pages.

The module uses the standard library only, so meson can run it without
pytest.  ``tests/docs`` wraps the same functions for pytest.
"""

from __future__ import annotations

import ast
import functools
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import textwrap
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------- doc files

PAGE_GLOBS = ("docs/*.md", "docs/*.org", "docs/orgmode/**/*.org")
PAGE_FILES = ("readme_src.org", "CONTRIBUTING.md")
# The changelog page only includes CHANGELOG.md.
PAGE_EXCLUDE = {"docs/orgmode/changelog.org"}
REFERENCE_DIR = "docs/orgmode/reference"
# Pages owned by other work: their blocks need no classification, but the
# static checks still run on them.
UNOWNED_PAGES = {
    "docs/orgmode/howto/ase.org",
    "docs/orgmode/howto/engine-plumed.org",
}


def doc_pages(root: Path = ROOT) -> list[Path]:
    pages = [root / name for name in PAGE_FILES if (root / name).is_file()]
    for pattern in PAGE_GLOBS:
        pages.extend(sorted(root.glob(pattern)))
    seen: set[Path] = set()
    out = []
    for page in pages:
        rel = page.relative_to(root).as_posix()
        if rel in PAGE_EXCLUDE or page in seen:
            continue
        seen.add(page)
        out.append(page)
    return out


def rel(path: Path, root: Path = ROOT) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


# ------------------------------------------------------------- code blocks

KINDS = ("run", "compile", "syntax", "fragment", "signature", "output", "skip")
# Kinds whose text after the kind is a required reason.
REASON_KINDS = ("output", "skip")

LANG_ALIASES = {
    "sh": "bash",
    "shell": "bash",
    "console": "bash",
    "c++": "cpp",
    "cxx": "cpp",
    "py": "python",
    "python3": "python",
    "f90": "fortran",
}


@dataclass
class Block:
    path: Path
    line: int
    lang: str
    body: str
    kind: str | None
    options: dict[str, str] = field(default_factory=dict)
    reason: str = ""
    raw_class: str | None = None

    def where(self) -> str:
        return f"{rel(self.path)}:{self.line}"


def parse_classification(text: str | None):
    """Split ``kind key=value ... reason`` into its parts."""
    if text is None:
        return None, {}, "", "missing classification"
    words = text.split()
    if not words:
        return None, {}, "", "empty classification"
    kind = words[0]
    if kind not in KINDS:
        return None, {}, "", f"unknown kind {kind!r}"
    options: dict[str, str] = {}
    rest = words[1:]
    while rest and re.fullmatch(r"[a-z_]+=\S+", rest[0]):
        key, value = rest.pop(0).split("=", 1)
        options[key] = value
    reason = " ".join(rest)
    if kind in REASON_KINDS and not reason:
        return None, {}, "", f"{kind} needs a reason"
    return kind, options, reason, None


_ORG_BEGIN = re.compile(r"^(\s*)#\+begin_(src|example)\b[ \t]*(\S*)(.*)$", re.I)
_ORG_ATTR = re.compile(r"^\s*#\+attr_rgpot:\s*(.*?)\s*$", re.I)
_ORG_AFFIL = re.compile(r"^\s*#\+(attr_\w+|caption|name|results|header)\b", re.I)
_MD_FENCE = re.compile(r"^(\s*)(`{3,}|~{3,})\s*([^\s`]*)(.*)$")
_MD_CLASS = re.compile(r"^\s*<!--\s*rgpot-doc:\s*(.*?)\s*-->\s*$")


def _unescape_org(line: str) -> str:
    return re.sub(r"^(\s*),(\*|#\+)", r"\1\2", line)


def org_blocks(path: Path, text: str | None = None) -> list[Block]:
    lines = (text if text is not None else path.read_text(encoding="utf-8")).splitlines()
    blocks = []
    i = 0
    while i < len(lines):
        m = _ORG_BEGIN.match(lines[i])
        if not m:
            i += 1
            continue
        kind_word = m.group(2).lower()
        lang = m.group(3).lower() if kind_word == "src" else "example"
        lang = LANG_ALIASES.get(lang, lang)
        attr = None
        j = i - 1
        while j >= 0 and _ORG_AFFIL.match(lines[j]):
            am = _ORG_ATTR.match(lines[j])
            if am:
                attr = am.group(1)
            j -= 1
        end = re.compile(rf"^\s*#\+end_{kind_word}\b", re.I)
        k = i + 1
        body = []
        while k < len(lines) and not end.match(lines[k]):
            body.append(_unescape_org(lines[k]))
            k += 1
        blocks.append(_make_block(path, i + 1, lang, textwrap.dedent("\n".join(body)), attr))
        i = k + 1
    return blocks


def md_blocks(path: Path, text: str | None = None) -> list[Block]:
    lines = (text if text is not None else path.read_text(encoding="utf-8")).splitlines()
    blocks = []
    i = 0
    while i < len(lines):
        m = _MD_FENCE.match(lines[i])
        if not m:
            i += 1
            continue
        fence = m.group(2)
        lang = LANG_ALIASES.get(m.group(3).lower(), m.group(3).lower()) or "text"
        attr = None
        j = i - 1
        while j >= 0 and not lines[j].strip():
            j -= 1
        if j >= 0:
            cm = _MD_CLASS.match(lines[j])
            if cm:
                attr = cm.group(1)
        k = i + 1
        body = []
        while k < len(lines) and not lines[k].strip().startswith(fence[0] * len(fence)):
            body.append(lines[k])
            k += 1
        blocks.append(_make_block(path, i + 1, lang, textwrap.dedent("\n".join(body)), attr))
        i = k + 1
    return blocks


def _make_block(path, line, lang, body, attr) -> Block:
    kind, options, reason, _ = parse_classification(attr)
    return Block(path, line, lang, body, kind, options, reason, attr)


def page_blocks(path: Path, text: str | None = None) -> list[Block]:
    if path.suffix == ".org":
        return org_blocks(path, text)
    return md_blocks(path, text)


def all_blocks(root: Path = ROOT) -> list[Block]:
    out = []
    for page in doc_pages(root):
        out.extend(page_blocks(page))
    return out


def classification_errors(blocks: list[Block]) -> list[str]:
    errors = []
    for block in blocks:
        _, _, _, problem = parse_classification(block.raw_class)
        if problem == "missing classification" and rel(block.path) in UNOWNED_PAGES:
            continue
        if problem:
            errors.append(f"{block.where()}: {block.lang} block: {problem}")
            continue
        if block.kind in ("run", "compile") and block.lang not in RUNNERS[block.kind]:
            errors.append(
                f"{block.where()}: {block.kind} is not available for {block.lang}"
            )
        if block.kind == "syntax" and block.lang not in SYNTAX_LANGS:
            errors.append(f"{block.where()}: syntax is not available for {block.lang}")
        if block.kind == "signature" and REFERENCE_DIR not in rel(block.path):
            errors.append(f"{block.where()}: signature blocks belong to reference pages")
    return errors


def readme_errors(root: Path = ROOT) -> list[str]:
    """README.md is the export of readme_src.org: same code, same headings."""
    src = root / "readme_src.org"
    out = root / "README.md"
    if not src.is_file() or not out.is_file():
        return []
    errors = []
    a = [b.body.strip() for b in org_blocks(src)]
    b = [blk.body.strip() for blk in md_blocks(out)]
    if a != b:
        errors.append(
            "README.md code blocks differ from readme_src.org; re-export the README"
        )
    org_heads = [
        re.sub(r"[=~]", "", m.group(2)).strip()
        for m in re.finditer(r"(?m)^(\*+)\s+(.*)$", src.read_text(encoding="utf-8"))
    ]
    md_heads = [
        re.sub(r"`", "", m.group(2)).strip()
        for m in re.finditer(r"(?m)^(#+)\s+(.*)$", "\n".join(
            line for line in _md_outside_fences(out.read_text(encoding="utf-8"))
        ))
    ]
    if org_heads != md_heads:
        errors.append(
            f"README.md headings {md_heads} differ from readme_src.org {org_heads}"
        )
    return errors


def _md_outside_fences(text: str):
    inside = False
    for line in text.splitlines():
        if re.match(r"^\s*(```|~~~)", line):
            inside = not inside
            continue
        if not inside:
            yield line


# ----------------------------------------------------------- source facts


def tracked_files(root: Path = ROOT) -> list[str]:
    try:
        out = subprocess.run(
            ["git", "ls-files"], cwd=root, capture_output=True, text=True, check=True
        ).stdout
        return out.splitlines()
    except (OSError, subprocess.CalledProcessError):
        return [
            p.relative_to(root).as_posix()
            for p in root.rglob("*")
            if p.is_file() and ".git" not in p.parts and ".pixi" not in p.parts
        ]


def meson_options(root: Path = ROOT) -> dict[str, dict]:
    """Name -> {type, value, choices} from meson_options.txt."""
    text = (root / "meson_options.txt").read_text(encoding="utf-8")
    text = re.sub(r"(?m)#.*$", "", text)
    out = {}
    for m in re.finditer(r"option\s*\((.*?)\)\s*(?=option\s*\(|\Z)", text, re.S):
        args = m.group(1)
        name = re.match(r"\s*'([^']+)'", args).group(1)
        typ = re.search(r"type\s*:\s*'(\w+)'", args).group(1)
        vm = re.search(r"value\s*:\s*('[^']*'|true|false|-?\d+)", args)
        value = vm.group(1).strip("'") if vm else None
        cm = re.search(r"choices\s*:\s*\[([^\]]*)\]", args)
        choices = re.findall(r"'([^']*)'", cm.group(1)) if cm else None
        if value is None:
            value = {"boolean": "true", "feature": "auto", "string": ""}.get(typ)
        dm = re.search(r"description\s*:\s*'((?:[^'\\]|\\.)*)'", args)
        out[name] = {
            "type": typ,
            "value": value,
            "choices": choices,
            "description": dm.group(1) if dm else "",
        }
    return out


def meson_value_ok(opt: dict, value: str) -> bool:
    typ = opt["type"]
    if typ == "boolean":
        return value in ("true", "false")
    if typ == "feature":
        return value in ("enabled", "disabled", "auto")
    if typ == "combo":
        return value in (opt["choices"] or [])
    if typ == "integer":
        return re.fullmatch(r"-?\d+", value) is not None
    return True


# Meson's own options, accepted after -D.
MESON_BUILTIN = {
    "buildtype", "prefix", "libdir", "includedir", "bindir", "default_library",
    "b_sanitize", "b_lundef", "b_coverage", "b_ndebug", "b_lto", "b_pie",
    "cpp_std", "c_std", "warning_level", "werror", "optimization", "debug",
    "wrap_mode", "python.install_env", "pkg_config_path", "cmake_prefix_path",
    "backend", "strip", "unity", "b_vscrt", "cpp_args", "c_args", "fortran_args",
    "cpp_link_args", "c_link_args", "python.platlibdir", "python.purelibdir",
}


def _cmake_bool(value: str) -> str:
    return {"TRUE": "ON", "FALSE": "OFF", "True": "ON", "False": "OFF"}.get(value, value)


def cmake_options(root: Path = ROOT) -> dict[str, list[str]]:
    """RGPOT_* option() names in the CMake files, with every default they can take.

    An option declared in two branches, or defaulting to a variable that is
    set in two branches, has more than one default.
    """
    out: dict[str, list[str]] = {}
    files = [root / "CMakeLists.txt", *sorted((root / "cmake").glob("*.cmake"))]
    for path in files:
        text = path.read_text(encoding="utf-8")
        sets: dict[str, list[str]] = {}
        for m in re.finditer(r"set\s*\(\s*(\w+)\s+(\w+)\s*\)", text):
            sets.setdefault(m.group(1), []).append(_cmake_bool(m.group(2)))
        for m in re.finditer(r"option\s*\(\s*(RGPOT_\w+)\s+\"[^\"]*\"\s*([^)\s]*)", text):
            raw = m.group(2) or "OFF"
            var = re.fullmatch(r"\$\{(\w+)\}", raw)
            values = sets.get(var.group(1), [raw]) if var else [_cmake_bool(raw)]
            have = out.setdefault(m.group(1), [])
            have.extend(v for v in values if v not in have)
        for m in re.finditer(r"set\s*\(\s*(RGPOT_\w+)\s[^)]*CACHE", text):
            out.setdefault(m.group(1), [])
    return out


def cmake_variables(root: Path = ROOT) -> set[str]:
    names: set[str] = set()
    for path in [root / "CMakeLists.txt", *root.glob("cmake/*"), *root.glob("**/CMakeLists.txt")]:
        if ".pixi" in path.parts or not path.is_file():
            continue
        names.update(re.findall(r"\bRGPOT_\w+", path.read_text(encoding="utf-8", errors="replace")))
    return names


def pixi_manifest(root: Path = ROOT) -> tuple[set[str], set[str]]:
    data = tomllib.loads((root / "pixi.toml").read_text(encoding="utf-8"))
    envs = set(data.get("environments", {})) | {"default"}
    tasks = set(data.get("tasks", {}))
    for feature in data.get("feature", {}).values():
        tasks |= set(feature.get("tasks", {}))
    return envs, tasks


# ------------------------------------------------------------- the C ABI

C_HEADERS = (
    "rgpot-core/include/rgpot.h",
    "CppCore/rgpot/nlist/rgpot_nlist.h",
    "CppCore/rgpot/nlist/fastnlist.h",
    "CppCore/rgpot/Gupta/gupta.h",
    "CppCore/rgpot/Morse/cambridge_morse.h",
    "CppCore/rgpot/ReducedPair/ReducedPair.hpp",
    "CppCore/rgpot/SuttonChen/SuttonChenPot.hpp",
    "CppCore/rgpot/SuttonChen/SuttonChenScales.hpp",
    "include/rgpot/plugin.h",
    "CppCore/rgpot/engine_c_abi.h",
    "CppCore/rgpot/ExternalPot/external_potentials.h",
    "CppCore/rgpot/calculator_mpi_abi.h",
    "CppCore/rgpot/Models/ModelCapi.h",
    "CppCore/rgpot/MetatomicPot/metatomic_c_abi.h",
    "CppCore/rgpot/XTBPot/xtb_c_abi.h",
    "engines/gromacs/libcp2k.h",
)
# Function prefix of a header when it is not rgpot_: librgpot_gromacs serves
# the CP2K library's API under CP2K's names.
C_PREFIXES = {"engines/gromacs/libcp2k.h": "cp2k_"}
# DLPack and eindir declare these; rgpot only forwards them.
C_FOREIGN = {"DLManagedTensorVersioned", "DLDevice", "eindir_abi_stamp_t", "eindir_objective_t"}


def strip_c_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    return re.sub(r"//[^\n]*", " ", text)


def _c_statements(text: str) -> list[str]:
    """Top-level statements of a C header, comments and directives removed."""
    text = strip_c_comments(text)
    text = re.sub(r'(?m)^\s*#.*?(?<!\\)$', ";", text)
    text = text.replace('extern "C" {', ";")
    out = []
    depth = 0
    cur = []
    for ch in text:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth < 0:
                depth = 0
                cur = []
                continue
        if ch == ";" and depth == 0:
            stmt = " ".join("".join(cur).split())
            if stmt:
                out.append(stmt)
            cur = []
            continue
        cur.append(ch)
    return out


def normalize_c(decl: str) -> str:
    decl = strip_c_comments(decl)
    decl = re.sub(r"\bRGPOT_\w+_API\b", " ", decl)
    decl = re.sub(r"\b(struct|enum)\s+(?=\w+\s*[\*\w])", "", decl)
    decl = " ".join(decl.replace(";", " ").split())
    decl = re.sub(r"\s*([\*\(\),\[\]&])\s*", r"\1", decl)
    return decl.strip()


@dataclass
class CName:
    name: str
    kind: str  # function | type
    decl: str
    header: str


def c_public_names(root: Path = ROOT) -> dict[str, CName]:
    out: dict[str, CName] = {}
    for header in C_HEADERS:
        path = root / header
        prefix = C_PREFIXES.get(header, "rgpot_")
        for stmt in _c_statements(path.read_text(encoding="utf-8")):
            if stmt.startswith("typedef"):
                m = None if "{" in stmt else re.search(r"\(\s*\*\s*(\w+)\s*\)\s*\(", stmt)
                if m:
                    name = m.group(1)
                else:
                    m = re.search(r"(\w+)\s*$", stmt)
                    if not m:
                        continue
                    name = m.group(1)
                if name not in C_FOREIGN:
                    out.setdefault(name, CName(name, "type", stmt, header))
                continue
            m = re.match(r"^(?:struct|enum)\s+(\w+)\s*\{", stmt)
            if m:
                out.setdefault(m.group(1), CName(m.group(1), "type", stmt, header))
                continue
            m = re.search(rf"\b({prefix}\w+)\s*\(", stmt)
            if m and "=" not in stmt.split("(")[0]:
                name = m.group(1)
                out[name] = CName(name, "function", stmt, header)
    return out


# --------------------------------------------------------- the C++ API

CPP_API_HEADERS = (
    "include/rgpot/errors.hpp",
    "include/rgpot/potential.hpp",
    "include/rgpot/types.hpp",
    "include/rgpot/rpc_client.hpp",
)
# Namespaces that are not part of the public surface by convention.
CPP_PRIVATE_NS = {"detail", "impl"}
CPP_SKIP_DIRS = ("CppCore/rgpot/rpc/",)
# Members every potential inherits; rgpot::PotentialBase and rgpot::Potential
# document them once.
CPP_BASE_MEMBERS = {
    "forceImpl", "forceBatchImpl", "forceBatch", "caps", "hessianVectorImpl", "setCache",
}


@dataclass
class CppName:
    name: str  # qualified, e.g. rgpot::PotentialHandle::calculate
    kind: str  # class | function | method
    decls: list[str]
    header: str


def _match_brace(text: str, start: int) -> int:
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i
    return len(text) - 1


def normalize_cpp(decl: str) -> str:
    decl = strip_c_comments(decl)
    decl = re.sub(r"template\s*<[^{;]*?>\s*(?=(static|inline|explicit|constexpr|\w))", "", decl)
    decl = re.sub(r"\[\[\w+\]\]", " ", decl)
    decl = re.sub(r"\b(inline|static|explicit|virtual|constexpr|override|final|noexcept)\b", " ", decl)
    decl = re.sub(r"\s+", " ", decl).strip()
    decl = re.sub(r"\s*([\*\(\),<>&=:\[\]])\s*", r"\1", decl)
    return decl.rstrip(";").strip()


def _decl_head(stmt: str) -> str | None:
    """Declaration up to the closing parenthesis and trailing const."""
    i = stmt.find("(")
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(stmt)):
        if stmt[j] == "(":
            depth += 1
        elif stmt[j] == ")":
            depth -= 1
            if depth == 0:
                tail = stmt[j + 1 :]
                m = re.match(r"\s*(const)?", tail)
                return stmt[: j + 1] + (" const" if m and m.group(1) else "")
    return None


def _class_members(cls: str, body: str, is_struct: bool):
    """Yield (name, head, access) for member declarations at depth 0."""
    access = "public" if is_struct else "private"
    body = re.sub(r"(?m)^[ \t]*#.*$", " ", body)
    i = 0
    n = len(body)
    stmt_start = 0
    parens = 0
    while i < n:
        if body[i] == "(":
            parens += 1
        elif body[i] == ")":
            parens -= 1
        if parens > 0:
            i += 1
            continue
        m = re.match(r"\s*(public|private|protected)\s*:(?!:)", body[i:])
        if m and body[stmt_start:i].strip() == "":
            access = m.group(1)
            i += m.end()
            stmt_start = i
            continue
        ch = body[i]
        if ch == "{":
            end = _match_brace(body, i)
            stmt = body[stmt_start:i]
            yield from _member(cls, stmt, access)
            i = end + 1
            if i < n and body[i] == ";":
                i += 1
            stmt_start = i
            continue
        if ch == ";":
            yield from _member(cls, body[stmt_start:i], access)
            i += 1
            stmt_start = i
            continue
        i += 1


def _member(cls: str, stmt: str, access: str):
    stmt = " ".join(stmt.split())
    if not stmt or "(" not in stmt or stmt.startswith(("using ", "friend ", "typedef ")):
        return
    if re.search(r"=\s*(delete|default)\s*$", stmt):
        return
    head = _decl_head(stmt)
    if head is None:
        return
    before = head.split("(")[0]
    names = re.findall(r"[~\w]+", before)
    if not names:
        return
    name = names[-1]
    if "operator" in before:
        return
    if re.search(r"\b(class|struct|enum|union)\b", before):
        return
    yield name, head, access


def _namespaces_at(text: str) -> list[tuple[int, int, str]]:
    """(start, end, qualified) spans of each namespace block."""
    spans = []
    for m in re.finditer(r"\bnamespace\s+([\w:]+)\s*\{", text):
        end = _match_brace(text, m.end() - 1)
        spans.append((m.start(), end, m.group(1)))
    return spans


def _qualifier(spans, pos: int) -> str:
    parts = [ns for (s, e, ns) in spans if s < pos < e]
    return "::".join(parts)


def cpp_public_names(root: Path = ROOT) -> dict[str, CppName]:
    out: dict[str, CppName] = {}
    for header in CPP_API_HEADERS:
        text = strip_c_comments((root / header).read_text(encoding="utf-8"))
        spans = _namespaces_at(text)
        for m in re.finditer(r"\b(class|struct)\s+(\w+)\s*(final)?\s*(:[^{;]*)?\{", text):
            qual = _qualifier(spans, m.start())
            if not qual.startswith("rgpot") or set(qual.split("::")) & CPP_PRIVATE_NS:
                continue
            cls = m.group(2)
            end = _match_brace(text, m.end() - 1)
            body = text[m.end() : end]
            full = f"{qual}::{cls}"
            ctors = []
            for name, head, access in _class_members(cls, body, m.group(1) == "struct"):
                if access != "public":
                    continue
                if name == cls:
                    if re.search(rf"\(\s*(const\s+)?{cls}\s*&", head):
                        continue
                    ctors.append(head)
                    continue
                if name.startswith("~"):
                    continue
                key = f"{full}::{name}"
                entry = out.setdefault(key, CppName(key, "method", [], header))
                entry.decls.append(head)
            out[full] = CppName(full, "class", ctors, header)
    out.update(cpp_potential_classes(root))
    out.update(cpp_support_names(root))
    out.update(cpp_member_classes(root))
    return out


# Value types and free functions that the potentials' constructors and the
# calculator groups use.
CPP_TYPE_HEADERS = (
    ("CppCore/rgpot/ForceStructs.hpp", ("ForceInput", "ForceOut", "ForceBatch")),
    ("CppCore/rgpot/pot_caps.hpp", ("PotCaps", "Reentrancy")),
    ("CppCore/rgpot/pot_types.hpp", ("PotType",)),
    ("CppCore/rgpot/CalculatorGroup.hpp", ("CalculatorGroup",)),
)
CPP_FUNCTION_HEADERS = (
    "CppCore/rgpot/ExternalPot/ExternalPot.hpp",
    "CppCore/rgpot/CalculatorGroup.hpp",
)
# Classes outside the potential hierarchy whose public members are part of
# the surface: the frontends that open the Metatomic and XTB engine tables,
# the checkpoint provenance a UmaPot reports, and the i-PI driver client.
CPP_MEMBER_CLASSES = (
    ("CppCore/rgpot/MetatomicPot/MetatomicDlopen.hpp", "MetatomicDlopen"),
    ("CppCore/rgpot/XTBPot/XTBDlopen.hpp", "XTBDlopen"),
    ("CppCore/rgpot/UmaPot/UmaProvenance.hpp", "UmaCheckpointProvenance"),
    ("CppCore/rgpot/IPIPot/IPIDriver.hpp", "IPIDriver"),
)


def _blank_bodies(text: str) -> str:
    """Replace class, struct, enum and function bodies with spaces."""
    out = list(text)
    for m in re.finditer(r"\b(class|struct|enum|union)\b[^;{]*\{|\)\s*(const\s*)?(noexcept\s*)?\{", text):
        start = m.end() - 1
        if out[start] != "{":
            continue
        end = _match_brace(text, start)
        for i in range(start, end + 1):
            out[i] = " "
    return "".join(out)


def cpp_support_names(root: Path = ROOT) -> dict[str, CppName]:
    out: dict[str, CppName] = {}
    for header, names in CPP_TYPE_HEADERS:
        path = root / header
        if not path.is_file():
            continue
        text = strip_c_comments(path.read_text(encoding="utf-8"))
        for name in names:
            if re.search(rf"\b{name}\b", text):
                out[f"rgpot::{name}"] = CppName(f"rgpot::{name}", "class", [], header)
    for path in sorted((root / "CppCore/rgpot").rglob("*.hpp")):
        header = path.relative_to(root).as_posix()
        if header.startswith(CPP_SKIP_DIRS) or "/simd/" in header:
            continue
        text = strip_c_comments(path.read_text(encoding="utf-8"))
        spans = _namespaces_at(text)
        for m in re.finditer(r"\bstruct\s+(\w+Config)\s*\{", text):
            qual = _qualifier(spans, m.start())
            if not qual.startswith("rgpot") or set(qual.split("::")) & CPP_PRIVATE_NS:
                continue
            full = f"{qual}::{m.group(1)}"
            out[full] = CppName(full, "class", [], header)
    for header in CPP_FUNCTION_HEADERS:
        path = root / header
        if not path.is_file():
            continue
        raw = strip_c_comments(path.read_text(encoding="utf-8"))
        spans = _namespaces_at(raw)
        text = _blank_bodies(raw)
        for m in re.finditer(r"[^;{}]*?\b(\w+)\s*\([^;{}]*\)\s*;", text):
            stmt = " ".join(m.group(0).split())
            if stmt.startswith(("using ", "typedef ", "return ", "#")) or "=" in stmt.split("(")[0]:
                continue
            qual = _qualifier(spans, m.start(1))
            if not qual.startswith("rgpot") or set(qual.split("::")) & CPP_PRIVATE_NS:
                continue
            full = f"{qual}::{m.group(1)}"
            head = _decl_head(stmt) or stmt
            entry = out.setdefault(full, CppName(full, "function", [], header))
            entry.decls.append(head)
    return out


def cpp_member_classes(root: Path = ROOT) -> dict[str, CppName]:
    out: dict[str, CppName] = {}
    for header, cls in CPP_MEMBER_CLASSES:
        path = root / header
        if not path.is_file():
            continue
        text = strip_c_comments(path.read_text(encoding="utf-8"))
        m = re.search(rf"\b(class|struct)\s+{cls}\b[^;{{]*\{{", text)
        if not m:
            continue
        qual = _qualifier(_namespaces_at(text), m.start())
        end = _match_brace(text, m.end() - 1)
        _add_class(out, f"{qual}::{cls}", cls, text[m.end() : end], m.group(1) == "struct", header)
    return out


def _struct_fields(body: str) -> list[str]:
    """Non-static data members declared at the top level of a struct body."""
    body = re.sub(r"(?m)^[ \t]*#.*$", " ", body)
    chars = list(body)
    heads = r"\)\s*(?:const\s*)?(?:noexcept\s*)?(?:override\s*)?\{|\b(?:class|struct|enum|union)\b[^;{()]*\{"
    for m in re.finditer(heads, body):
        start = m.end() - 1
        if chars[start] != "{":
            continue
        end = _match_brace(body, start)
        chars[start : end + 1] = " " * (end - start) + ";"
    fields = []
    depth = 0
    cur: list[str] = []
    for ch in "".join(chars):
        if ch in "{(":
            depth += 1
        elif ch in "})":
            depth -= 1
        if ch == ";" and depth == 0:
            stmt = re.sub(r"^(?:\s*(?:public|private|protected)\s*:(?!:))+", "", "".join(cur))
            stmt = " ".join(stmt.split())
            cur = []
            if not stmt or stmt.startswith(
                ("using ", "static ", "friend ", "typedef ", "template", "enum ", "struct ", "class ", "union ")
            ):
                continue
            decl = re.split(r"[={]", stmt, maxsplit=1)[0]
            if "(" in decl:
                continue
            names = re.findall(r"(\w+)\s*(?:\[[^\]]*\])*\s*$", decl)
            if names:
                fields.append(names[0])
            continue
        cur.append(ch)
    return fields


def cpp_config_fields(root: Path = ROOT) -> dict[str, tuple[str, list[str]]]:
    """Qualified name -> (header, data members) of each public *Config struct."""
    out: dict[str, tuple[str, list[str]]] = {}
    for path in sorted((root / "CppCore/rgpot").rglob("*.hpp")):
        header = path.relative_to(root).as_posix()
        if header.startswith(CPP_SKIP_DIRS) or "/simd/" in header:
            continue
        text = strip_c_comments(path.read_text(encoding="utf-8"))
        spans = _namespaces_at(text)
        for m in re.finditer(r"\bstruct\s+(\w+Config)\s*\{", text):
            qual = _qualifier(spans, m.start())
            if not qual.startswith("rgpot") or set(qual.split("::")) & CPP_PRIVATE_NS:
                continue
            end = _match_brace(text, m.end() - 1)
            out[f"{qual}::{m.group(1)}"] = (header, _struct_fields(text[m.end() : end]))
    return out


def _expand_class_macros(text: str) -> str:
    """Replace RGPOT_FORTRAN_POT_CLASS(Name, Type) with the class it defines."""
    m = re.search(r"#define\s+RGPOT_FORTRAN_POT_CLASS\(ClassName,\s*PotTypeValue\)((?:.*\\\n)+.*)", text)
    if not m:
        return text
    body = m.group(1).replace("\\\n", "\n")
    text = text.replace(m.group(0), "")

    def expand(call):
        return body.replace("ClassName", call.group(1)).replace("PotTypeValue", call.group(2))

    return re.sub(r"RGPOT_FORTRAN_POT_CLASS\((\w+),\s*(\w+)\)", expand, text)


def _add_class(out, full, cls, body, is_struct, header, skip_members=()):
    ctors = []
    for name, head, access in _class_members(cls, body, is_struct):
        if access != "public" or name.startswith("~"):
            continue
        if name == cls:
            if not re.search(rf"\(\s*(const\s+)?{cls}\s*&", head):
                ctors.append(head)
            continue
        if name in skip_members:
            continue
        key = f"{full}::{name}"
        entry = out.setdefault(key, CppName(key, "method", [], header))
        if head not in entry.decls:
            entry.decls.append(head)
    out[full] = CppName(full, "class", ctors, header)


def cpp_potential_classes(root: Path = ROOT) -> dict[str, CppName]:
    """Concrete Potential<Derived> classes, PotentialBase and Potential."""
    out: dict[str, CppName] = {}
    pattern = re.compile(
        r"\bclass\s+(\w+)\s*(?:final\s*)?:\s*public\s+(?:rgpot::)?Potential\s*<\s*(\w+)\s*>"
    )
    for path in sorted((root / "CppCore/rgpot").rglob("*.hpp")):
        r = rel(path, root)
        if any(r.startswith(d) for d in CPP_SKIP_DIRS):
            continue
        raw = path.read_text(encoding="utf-8", errors="replace")
        text = strip_c_comments(_expand_class_macros(raw))
        spans = _namespaces_at(text)
        for m in pattern.finditer(text):
            if m.group(1) != m.group(2):
                continue
            qual = _qualifier(spans, m.start())
            if not qual.startswith("rgpot") or set(qual.split("::")) & CPP_PRIVATE_NS:
                continue
            brace = text.find("{", m.end())
            end = _match_brace(text, brace)
            cls = m.group(1)
            _add_class(out, f"{qual}::{cls}", cls, text[brace + 1 : end], False, r, CPP_BASE_MEMBERS)
    base = "CppCore/rgpot/Potential.hpp"
    text = strip_c_comments((root / base).read_text(encoding="utf-8"))
    for cls in ("PotentialBase", "Potential"):
        m = re.search(rf"\bclass\s+{cls}\b[^;{{]*\{{", text)
        end = _match_brace(text, m.end() - 1)
        _add_class(out, f"rgpot::{cls}", cls, text[m.end() : end], False, base)
    return out


# ------------------------------------------------------------ Python API

PY_INIT = "python/rgpot/__init__.py"
PY_STUB = "python/rgpot/_core.pyi"
# rgpot.ase is documented with the ASE calculator; it is not in this check.
PY_EXCLUDE = {"ase"}


@dataclass
class PyParam:
    name: str
    default: object
    has_default: bool
    kind: str  # positional | keyword


@dataclass
class PyName:
    name: str  # rgpot.X or rgpot.X.member
    kind: str  # class | function | variable | method | property
    params: list[PyParam] | None


_NO_DEFAULT = object()


def _params(fn: ast.FunctionDef | ast.AsyncFunctionDef, drop_self: bool) -> list[PyParam]:
    args = fn.args
    positional = list(args.posonlyargs) + list(args.args)
    defaults = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
    out = []
    for arg, default in zip(positional, defaults):
        out.append(_param(arg.arg, default, "positional"))
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        out.append(_param(arg.arg, default, "keyword"))
    if drop_self and out and out[0].name in ("self", "cls"):
        out = out[1:]
    return out


def _param(name, default, kind) -> PyParam:
    if default is None:
        return PyParam(name, _NO_DEFAULT, False, kind)
    try:
        value = ast.literal_eval(default)
    except ValueError:
        value = ast.unparse(default)
    return PyParam(name, value, True, kind)


def _stub_classes(root: Path):
    tree = ast.parse((root / PY_STUB).read_text(encoding="utf-8"))
    classes = {}
    functions = {}
    variables = set()
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            members = {}
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    decos = {ast.unparse(d) for d in item.decorator_list}
                    kind = "property" if "property" in decos else "method"
                    members[item.name] = (kind, _params(item, "staticmethod" not in decos))
            classes[node.name] = members
        elif isinstance(node, ast.FunctionDef):
            functions[node.name] = _params(node, False)
        elif isinstance(node, (ast.AnnAssign, ast.Assign)):
            target = node.target if isinstance(node, ast.AnnAssign) else node.targets[0]
            if isinstance(target, ast.Name):
                variables.add(target.id)
    return classes, functions, variables


def py_public_names(root: Path = ROOT) -> dict[str, PyName]:
    classes, functions, variables = _stub_classes(root)
    tree = ast.parse((root / PY_INIT).read_text(encoding="utf-8"))
    local_defs = {}
    exported: dict[str, str | None] = {}

    def core_name(value) -> str | None:
        if isinstance(value, ast.Attribute) and ast.unparse(value.value) == "_core":
            return value.attr
        if isinstance(value, ast.Call) and ast.unparse(value.func) == "getattr":
            if ast.unparse(value.args[0]) == "_core":
                return ast.literal_eval(value.args[1])
        return None

    def visit(nodes):
        for node in nodes:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        exported.setdefault(target.id, core_name(node.value))
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                exported.setdefault(node.target.id, core_name(node.value) if node.value else None)
            elif isinstance(node, ast.ImportFrom) and node.module == "rgpot._core":
                for alias in node.names:
                    exported[alias.asname or alias.name] = alias.name
            elif isinstance(node, ast.FunctionDef):
                local_defs[node.name] = node
                exported[node.name] = None
            elif isinstance(node, ast.Try):
                visit(node.body)

    visit(tree.body)
    out: dict[str, PyName] = {}
    for name, core in exported.items():
        if name.startswith("_") and name != "__version__":
            continue
        if name in PY_EXCLUDE or name in ("__all__",):
            continue
        full = f"rgpot.{name}"
        if name in local_defs:
            out[full] = PyName(full, "function", _params(local_defs[name], False))
        elif core in classes:
            init = classes[core].get("__init__")
            out[full] = PyName(full, "class", init[1] if init else None)
            for member, (kind, params) in classes[core].items():
                if member.startswith("_") and member != "__call__":
                    continue
                out[f"{full}.{member}"] = PyName(
                    f"{full}.{member}", kind, params if kind == "method" else None
                )
        elif core in functions:
            out[full] = PyName(full, "function", functions[core])
        else:
            out[full] = PyName(full, "variable", None)
    return out


# --------------------------------------------------------- reference pages

FIELDS = ("Inputs", "Units", "Outputs", "Errors")
_HEADING = re.compile(r"^(\*+)\s+(.*?)\s*$")
_ENTRY = re.compile(r"^=([^=\s]+)=$")
_FIELD = re.compile(r"^-\s+([A-Z][\w ]*?)\s+::\s*(.*)$")


@dataclass
class Entry:
    name: str
    path: Path
    line: int
    fields: dict[str, str]
    signatures: list[str]

    def where(self) -> str:
        return f"{rel(self.path)}:{self.line}"


def reference_entries(root: Path = ROOT, pages: list[Path] | None = None) -> list[Entry]:
    pages = pages if pages is not None else sorted((root / REFERENCE_DIR).rglob("*.org"))
    entries = []
    for page in pages:
        text = page.read_text(encoding="utf-8")
        entries.extend(_page_entries(page, text))
    return entries


def _page_entries(page: Path, text: str) -> list[Entry]:
    lines = text.splitlines()
    blocks = {b.line: b for b in org_blocks(page, text)}
    entries = []
    current = None
    level = 0
    field_name = None
    for idx, line in enumerate(lines, start=1):
        hm = _HEADING.match(line)
        if hm:
            depth = len(hm.group(1))
            em = _ENTRY.match(hm.group(2))
            if em:
                current = Entry(em.group(1), page, idx, {}, [])
                entries.append(current)
                level = depth
                field_name = None
                continue
            if current is not None and depth <= level:
                current = None
            field_name = None
            continue
        if current is None:
            continue
        block = blocks.get(idx)
        if block is not None and block.kind == "signature":
            current.signatures.extend(_signature_lines(block.body))
            field_name = None
            continue
        fm = _FIELD.match(line)
        if fm:
            field_name = fm.group(1)
            current.fields[field_name] = fm.group(2).strip()
            continue
        if field_name and line.startswith("  ") and line.strip():
            current.fields[field_name] = (current.fields[field_name] + " " + line.strip()).strip()
        elif not line.strip() or line.lstrip().startswith("#+"):
            continue
        else:
            field_name = None
    return entries


def _signature_lines(body: str) -> list[str]:
    """Split a signature block into declarations."""
    out = []
    cur = []
    depth = 0
    for line in body.splitlines():
        if not line.strip():
            continue
        cur.append(line.strip())
        depth += line.count("(") - line.count(")") + line.count("{") - line.count("}")
        if depth <= 0 and (line.rstrip().endswith((";", ")", "}", "const")) or "->" in line):
            out.append(" ".join(cur))
            cur = []
            depth = 0
    if cur:
        out.append(" ".join(cur))
    return out


def _entry_ok(entry: Entry) -> list[str]:
    problems = []
    for name in FIELDS:
        value = entry.fields.get(name, "").strip()
        if not value:
            problems.append(f"{entry.where()}: {entry.name}: missing {name}")
        elif re.search(r"\b(TODO|TBD|FIXME)\b", value):
            problems.append(f"{entry.where()}: {entry.name}: {name} is a placeholder")
    return problems


def _applies_to(entry: Entry) -> list[str]:
    value = entry.fields.get("Applies to", "")
    return [w.strip(" .=~") for w in re.split(r"[,\s]+", value) if w.strip(" .=~")]


def _py_signature(sig: str):
    m = re.match(r"^([\w.]+)\((.*)\)\s*(->.*)?$", sig.strip())
    if not m:
        return None, None
    try:
        fn = ast.parse(f"def _f({m.group(2)}): pass").body[0]
    except SyntaxError:
        return m.group(1), None
    return m.group(1), _params(fn, False)


def _py_params_differ(doc: list[PyParam], src: list[PyParam]) -> str | None:
    dn = [(p.name, p.has_default, p.default if p.has_default else None, p.kind) for p in doc]
    sn = [(p.name, p.has_default, p.default if p.has_default else None, p.kind) for p in src]
    if dn != sn:
        return f"documented {_fmt_params(doc)} but the source has {_fmt_params(src)}"
    return None


def _fmt_params(params: list[PyParam]) -> str:
    parts = []
    star = False
    for p in params:
        if p.kind == "keyword" and not star:
            parts.append("*")
            star = True
        parts.append(f"{p.name}={p.default!r}" if p.has_default else p.name)
    return "(" + ", ".join(parts) + ")"


def reference_errors(root: Path = ROOT, pages: list[Path] | None = None) -> list[str]:
    entries = reference_entries(root, pages)
    errors = []
    by_name: dict[str, list[Entry]] = {}
    for entry in entries:
        by_name.setdefault(entry.name, []).append(entry)
        errors.extend(_entry_ok(entry))
    for name, found in by_name.items():
        if len(found) > 1 and not name.startswith("."):
            errors.append(f"{found[1].where()}: second entry for {name}")

    c_names = c_public_names(root)
    cpp_names = cpp_public_names(root)
    py_names = py_public_names(root)

    # Member entries (".name") cover the member on every class they list.
    covered: set[str] = set()
    for entry in entries:
        if not entry.name.startswith("."):
            continue
        targets = _applies_to(entry)
        if not targets:
            errors.append(f"{entry.where()}: {entry.name}: missing Applies to")
        for target in targets:
            sep = "::" if "::" in target else "."
            member = f"{target}{sep}{entry.name[1:]}"
            if member not in py_names and member not in cpp_names:
                errors.append(
                    f"{entry.where()}: {entry.name}: {target} has no member {entry.name[1:]}"
                )
                continue
            covered.add(member)
            errors.extend(_check_signatures(entry, member, c_names, cpp_names, py_names))

    for entry in entries:
        if entry.name.startswith("."):
            continue
        name = entry.name
        if name not in c_names and name not in cpp_names and name not in py_names:
            errors.append(f"{entry.where()}: {name} is not a public name in the source")
            continue
        covered.add(name)
        errors.extend(_check_signatures(entry, name, c_names, cpp_names, py_names))

    for name in sorted(set(c_names) | set(cpp_names) | set(py_names)):
        if name not in covered:
            errors.append(f"no reference entry for {name}")
    errors.extend(config_field_errors(root, entries))
    if pages is None:
        errors.extend(local_schema_errors(root))
    return errors


def config_field_errors(root: Path, entries: list[Entry]) -> list[str]:
    """The entry of each *Config struct names every field of the struct."""
    by_name = {e.name: e for e in entries if not e.name.startswith(".")}
    errors = []
    for name, (header, fields) in sorted(cpp_config_fields(root).items()):
        entry = by_name.get(name)
        if entry is None:
            continue
        text = " ".join(entry.fields.values())
        for f in fields:
            if f"={f}=" not in text:
                errors.append(f"{entry.where()}: {name}: field {f} of {header} is not described")
    return errors


LOCAL_SCHEMA_SERIES = "CppCore/rgpot/rpc/local/series"
_CAPNP_SCOPE = re.compile(r"^[ +]\s*(?:struct|interface)\s+(\w+)")
_CAPNP_MEMBER = re.compile(r"^\+\s*(\w+)\s+@\d+\s*[:(]")


def local_schema_fields(root: Path = ROOT) -> list[tuple[str, str]]:
    """(Struct.member, diff) for each field or method an rgpot-local diff adds."""
    series = root / LOCAL_SCHEMA_SERIES
    if not series.is_file():
        return []
    out = []
    for name in series.read_text(encoding="utf-8").splitlines():
        name = name.strip()
        if not name or name.startswith("#"):
            continue
        scope = None
        for line in (series.parent / name).read_text(encoding="utf-8").splitlines():
            if line.startswith(("+++", "---")):
                continue
            hunk = re.match(r"^@@[^@]*@@\s*(?:struct|interface)\s+(\w+)", line)
            if line.startswith("@@"):
                scope = hunk.group(1) if hunk else None
                continue
            sm = _CAPNP_SCOPE.match(line)
            if sm:
                scope = sm.group(1)
                continue
            if re.match(r"^[ +]\}", line):
                scope = None
                continue
            mm = _CAPNP_MEMBER.match(line)
            if mm and scope:
                out.append((f"{scope}.{mm.group(1)}", name))
    return out


def local_schema_errors(root: Path = ROOT, pages: list[Path] | None = None) -> list[str]:
    """A reference page names each field the rgpot-local schema diffs add."""
    pages = pages if pages is not None else sorted((root / REFERENCE_DIR).rglob("*.org"))
    text = "\n".join(p.read_text(encoding="utf-8") for p in pages)
    return [
        f"{LOCAL_SCHEMA_SERIES}: {diff} adds {member}, which no reference page names"
        for member, diff in local_schema_fields(root)
        if not re.search(rf"(?<![\w.]){re.escape(member)}(?!\w)", text)
    ]


def _check_signatures(entry, name, c_names, cpp_names, py_names) -> list[str]:
    errors = []
    sigs = entry.signatures
    if name in c_names:
        cn = c_names[name]
        if cn.kind == "function" and not sigs:
            errors.append(f"{entry.where()}: {name}: missing signature")
        for sig in sigs:
            if normalize_c(sig) != normalize_c(cn.decl):
                errors.append(
                    f"{entry.where()}: {name}: signature {normalize_c(sig)!r} "
                    f"differs from {rel(ROOT / cn.header)}: {normalize_c(cn.decl)!r}"
                )
    elif name in cpp_names:
        cn = cpp_names[name]
        if cn.kind == "class" and not sigs:
            return errors
        have = {normalize_cpp(d) for d in cn.decls}
        doc = {normalize_cpp(_decl_head(s) or s) for s in sigs}
        for sig in sorted(doc - have):
            errors.append(f"{entry.where()}: {name}: signature {sig!r} is not in {cn.header}")
        for sig in sorted(have - doc):
            errors.append(f"{entry.where()}: {name}: undocumented overload {sig!r}")
    elif name in py_names:
        pn = py_names[name]
        if pn.params is None:
            return errors
        if not sigs:
            errors.append(f"{entry.where()}: {name}: missing signature")
            return errors
        short = name.rsplit(".", 1)[-1]
        for sig in sigs:
            called, params = _py_signature(sig)
            if params is None:
                errors.append(f"{entry.where()}: {name}: cannot parse signature {sig!r}")
                continue
            if called.rsplit(".", 1)[-1] != short:
                errors.append(f"{entry.where()}: {name}: signature names {called}")
            problem = _py_params_differ(params, pn.params)
            if problem:
                errors.append(f"{entry.where()}: {name}: {problem}")
    return errors


# ------------------------------------------------------------------- facts

_PATH_TOKEN = re.compile(
    r"(?<![\w/.$-])((?:CppCore|rgpot-core|scripts|tests|docs|engines|tools|python|ci|"
    r"include|bind|benchmarks|cmake|subprojects|third_party|potctl|rgpot-pair|lean|"
    r"branding|benchflow)/[\w./+-]*[\w+-])"
)
_RGPOT_TOKEN = re.compile(r"\bRGPOT_[A-Z0-9_]*[A-Z0-9]\b")
_MESON_D = re.compile(r"(?<![\w-])-D([a-z_][\w.]*)(?:=([\w.+-]*))?")
_CMAKE_D = re.compile(r"(?<![\w-])-D(RGPOT_\w+)")
_PIXI_CMD = re.compile(r"\bpixi\s+(r|run|shell|install)\b([^\n`=~]*)")


def _doc_text(page: Path) -> str:
    return page.read_text(encoding="utf-8")


def _strip_urls(text: str) -> str:
    return re.sub(r"https?://\S+", " ", text)


def _path_exists(token: str, root: Path, tracked: set[str], dirs: set[str]) -> bool:
    token = token.rstrip(".")
    if token in tracked or token in dirs:
        return True
    if any(ch in token for ch in "*<>{}"):
        return True
    return (root / token).exists() or _git_ignored(str(root), token)


def _git_ignored(root: str, token: str) -> bool:
    """True for a path .gitignore covers: an output of the build, such as docs/build.

    The top-level .gitignore is read here rather than through git, which has
    no repository to ask in a tree exported with git archive.  As in git,
    nothing below an excluded directory comes back.
    """
    rules = _ignore_rules(root)
    parts = token.strip("/").split("/")
    for depth in range(1, len(parts) + 1):
        path = "/".join(parts[:depth])
        is_dir = depth < len(parts) or (Path(root) / path).is_dir()
        ignored = False
        for pattern, negated, dir_only in rules:
            if (is_dir or not dir_only) and pattern.match(path):
                ignored = not negated
        if ignored:
            return True
    return False


@functools.lru_cache(maxsize=None)
def _ignore_rules(root: str) -> tuple[tuple[re.Pattern, bool, bool], ...]:
    """(pattern, negated, directories only) for each line of the top-level .gitignore."""
    try:
        lines = (Path(root) / ".gitignore").read_text(encoding="utf-8").splitlines()
    except OSError:
        return ()
    rules = []
    for line in lines:
        line = re.sub(r"(?<!\\)\s+$", "", line)
        if not line or line.startswith("#"):
            continue
        negated = line.startswith("!")
        if negated:
            line = line[1:]
        dir_only = line.endswith("/")
        line = line.rstrip("/")
        # A slash before the end ties the pattern to the top; without one it
        # matches a name at any depth.
        prefix = "" if "/" in line else "(?:.*/)?"
        regex = prefix + _gitignore_glob(line.lstrip("/")) + r"\Z"
        rules.append((re.compile(regex), negated, dir_only))
    return tuple(rules)


def _gitignore_glob(glob: str) -> str:
    """Regex for one .gitignore glob, where * and ? stop at a slash and ** does not."""
    out = []
    i = 0
    while i < len(glob):
        at_start = i == 0 or glob[i - 1] == "/"
        if at_start and glob.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif at_start and glob.startswith("**", i) and i + 2 == len(glob):
            out.append(".*")
            i += 2
        elif glob[i] == "*":
            out.append("[^/]*")
            i += 1
        elif glob[i] == "?":
            out.append("[^/]")
            i += 1
        elif glob[i] == "[" and (end := glob.find("]", i + 2)) != -1:
            body = glob[i + 1 : end].replace("\\", "\\\\")
            if body[0] in "!^":
                body = "^" + body[1:]
            out.append(f"[{body}]")
            i = end + 1
        elif glob[i] == "\\" and i + 1 < len(glob):
            out.append(re.escape(glob[i + 1]))
            i += 2
        else:
            out.append(re.escape(glob[i]))
            i += 1
    return "".join(out)


_FOREIGN = re.compile(r"^(?:#\+rgpot_foreign:|<!--\s*rgpot-foreign:)(.*?)(?:-->)?\s*$", re.M)


def foreign_tokens(text: str) -> set[str]:
    """Paths and -D options a page declares as belonging to another project.

    Org pages use ``#+rgpot_foreign: TOKEN...``; Markdown pages use
    ``<!-- rgpot-foreign: TOKEN... -->``.
    """
    return {tok for m in _FOREIGN.finditer(text) for tok in m.group(1).split()}


def facts_errors(root: Path = ROOT, pages: list[Path] | None = None) -> list[str]:
    pages = pages if pages is not None else doc_pages(root)
    tracked = set(tracked_files(root))
    dirs = {str(Path(p).parent.as_posix()) for p in tracked}
    for d in list(dirs):
        parts = d.split("/")
        for i in range(1, len(parts)):
            dirs.add("/".join(parts[:i]))
    options = meson_options(root)
    cmake = cmake_options(root)
    cmake_vars = cmake_variables(root)
    envs, tasks = pixi_manifest(root)
    source_text = _source_identifiers(root, tracked)
    errors = []
    for page in pages:
        text = _strip_urls(_doc_text(page))
        foreign = foreign_tokens(text)
        where = rel(page, root)
        for lineno, line in enumerate(text.splitlines(), start=1):
            loc = f"{where}:{lineno}"
            if _FOREIGN.match(line):
                continue
            for m in _PATH_TOKEN.finditer(line):
                token = m.group(1)
                if token.rstrip(".") in foreign:
                    continue
                if not _path_exists(token, root, tracked, dirs):
                    errors.append(f"{loc}: path {token} does not exist")
            for m in _RGPOT_TOKEN.finditer(line):
                if m.group(0) not in source_text:
                    errors.append(f"{loc}: {m.group(0)} does not occur in the source")
            for m in _MESON_D.finditer(line):
                name, value = m.group(1), (m.group(2) or "").rstrip(".") or None
                if name.startswith("RGPOT") or name.isupper() or name in MESON_BUILTIN:
                    continue
                if "." in name and name.split(".")[0] not in options:
                    continue
                if f"-D{name}" in foreign:
                    continue
                if name not in options:
                    errors.append(f"{loc}: -D{name} is not a meson option")
                elif value is not None and value and not meson_value_ok(options[name], value):
                    errors.append(f"{loc}: -D{name}={value} is not a valid value")
            for m in _CMAKE_D.finditer(line):
                if m.group(1) not in cmake and m.group(1) not in cmake_vars:
                    errors.append(f"{loc}: -D{m.group(1)} is not a CMake option")
            for m in _PIXI_CMD.finditer(line):
                errors.extend(_pixi_errors(loc, m.group(1), m.group(2), envs, tasks))
    errors.extend(build_option_table_errors(root, options, cmake))
    errors.extend(gromacs_key_errors(root))
    errors.extend(engine_table_export_errors(root))
    return errors


ENGINE_ABI_HEADER = "CppCore/rgpot/engine_c_abi.h"
ENGINE_TABLE_SOURCES = (
    "CppCore/rgpot/engine/EngineTable.cc",
    "CppCore/rgpot/engine/EngineAbiStamp.cc",
)
ENGINE_ABI_PAGE = "docs/orgmode/reference/engine_abi.org"


def engine_table_export_errors(root: Path = ROOT, page: Path | None = None) -> list[str]:
    """The library export list matches the engine implementation and C ABI header."""
    hdr = root / ENGINE_ABI_HEADER
    sources = [root / source for source in ENGINE_TABLE_SOURCES]
    if not hdr.is_file() or any(not source.is_file() for source in sources):
        return []
    page = page if page is not None else root / ENGINE_ABI_PAGE
    where = rel(page, root)
    if not page.is_file():
        return [f"{where} is missing"]
    hdr_text = hdr.read_text(encoding="utf-8")
    marker = "Optional symbols:"
    if marker not in hdr_text:
        return [f"{ENGINE_ABI_HEADER}: {marker!r} marker was not found"]
    optional = set()
    for stmt in _c_statements(hdr_text.split(marker, 1)[1]):
        if stmt.startswith("typedef"):
            continue
        m = re.search(r"\b(rgpot_engine_\w+)\s*\(", stmt)
        if m:
            optional.add(m.group(1))
    table_code = "\n".join(
        strip_c_comments(source.read_text(encoding="utf-8")) for source in sources
    )
    exported = {name for name in optional if re.search(rf"\b{name}\s*\(", table_code)}
    omitted = optional - exported
    doc_text = page.read_text(encoding="utf-8")
    m = re.search(
        r"Every library above exports\s+(.*?)\s+None of them exports\s+(.*?)(?:\n\s*\n|\Z)",
        doc_text,
        re.S,
    )
    if not m:
        return [f"{where}: the library export list was not found"]
    doc_exported = set(re.findall(r"=(rgpot_engine_\w+)=", m.group(1)))
    doc_omitted = set(re.findall(r"=(rgpot_engine_\w+)=", m.group(2)))
    errors = [
        f"{where}: engine source export {s} is not in the library export list"
        for s in sorted(exported - doc_exported)
    ]
    errors += [
        f"{where}: {s} is not an optional symbol exported by the engine sources"
        for s in sorted(doc_exported - exported)
    ]
    errors += [
        f"{where}: unexported optional symbol {s} is not listed"
        for s in sorted(omitted - doc_omitted)
    ]
    errors += [
        f"{where}: {s} is not an unexported optional symbol of {ENGINE_ABI_HEADER}"
        for s in sorted(doc_omitted - omitted)
    ]
    return errors


GROMACS_DROPIN_SOURCE = "engines/gromacs/rgpot_gromacs.cpp"
GROMACS_DROPIN_PAGE = "docs/orgmode/reference/gromacs_dropin.org"


def gromacs_key_errors(root: Path = ROOT, page: Path | None = None) -> list[str]:
    """The &RGPOT key table of the drop-in reference lists the keys the parser reads."""
    src = root / GROMACS_DROPIN_SOURCE
    if not src.is_file():
        return []
    page = page if page is not None else root / GROMACS_DROPIN_PAGE
    where = rel(page, root)
    if not page.is_file():
        return [f"{where} is missing"]
    code = strip_c_comments(src.read_text(encoding="utf-8"))
    m = re.search(r'if \(key == "&END"\)(.*?)if \(!saw_rgpot\)', code, re.S)
    if not m:
        return [f"{GROMACS_DROPIN_SOURCE}: the &RGPOT parser was not found"]
    parsed = set(re.findall(r'\bkey\s*[!=]=\s*"([A-Z_]+)"', m.group(1)))
    listed: dict[str, int] = {}
    section = False
    for lineno, line in enumerate(page.read_text(encoding="utf-8").splitlines(), start=1):
        hm = _HEADING.match(line)
        if hm:
            section = "&RGPOT" in hm.group(2)
            continue
        km = re.match(r"^\|\s*=([A-Z_]+)=", line)
        if section and km:
            listed[km.group(1)] = lineno
    errors = [f"{where}: &RGPOT key {k} is not in the table" for k in sorted(parsed - set(listed))]
    errors += [
        f"{where}:{listed[k]}: {k} is not a key {GROMACS_DROPIN_SOURCE} reads"
        for k in sorted(set(listed) - parsed)
    ]
    return errors


def _pixi_errors(loc, verb, rest, envs, tasks) -> list[str]:
    errors = []
    words = rest.split()
    env = None
    i = 0
    while i < len(words):
        if words[i] in ("-e", "--environment") and i + 1 < len(words):
            env = words[i + 1]
            i += 2
            continue
        if words[i].startswith("-"):
            i += 1
            continue
        break
    if env is not None and env.strip("'\"") not in envs and not env.startswith("$"):
        errors.append(f"{loc}: pixi environment {env} does not exist")
    if verb in ("r", "run") and i < len(words):
        task = words[i].strip("'\"")
        if task.startswith("<"):
            return errors
        if task not in tasks and shutil.which(task) is None and task not in KNOWN_PROGRAMS:
            errors.append(f"{loc}: pixi task {task} does not exist")
    return errors


# Programs a pixi environment provides that need not be on the checker's PATH.
KNOWN_PROGRAMS = {
    "python", "python3", "meson", "cmake", "ctest", "ninja", "cargo", "bash", "pytest",
    "uvx", "uv", "emacs", "sphinx-build", "doxygen", "mpirun", "lmp", "gmx", "plumed",
    "maturin", "pip", "cog", "potctl", "towncrier", "nickel", "prek", "lychee", "snapper",
}


def _source_identifiers(root: Path, tracked: set[str]) -> set[str]:
    names: set[str] = set()
    doc_like = (".org", ".md", ".rst", ".txt")
    for path in tracked:
        if path.startswith(("docs/", "CHANGELOG")) or path.endswith(doc_like[:3]):
            continue
        full = root / path
        try:
            if full.stat().st_size > 2_000_000:
                continue
            text = full.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        names.update(_RGPOT_TOKEN.findall(text))
    return names


BUILD_OPTIONS_PAGE = "docs/orgmode/reference/build_options.org"


def build_option_table_errors(root: Path, options=None, cmake=None) -> list[str]:
    """The build options page lists every option with its default."""
    page = root / BUILD_OPTIONS_PAGE
    if not page.is_file():
        return [f"{BUILD_OPTIONS_PAGE} is missing"]
    options = options if options is not None else meson_options(root)
    cmake = cmake if cmake is not None else cmake_options(root)
    rows = {}
    for lineno, line in enumerate(page.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.startswith("|") or set(line) <= set("|-+ "):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 3 or cells[0].lower() in ("option", "name"):
            continue
        rows[cells[0].strip("=~")] = (lineno, cells[1].strip("=~"), cells[2].strip("=~"))
    errors = []
    for name, opt in options.items():
        if name not in rows:
            errors.append(f"{BUILD_OPTIONS_PAGE}: meson option {name} is not listed")
            continue
        lineno, typ, default = rows[name]
        if typ != opt["type"]:
            errors.append(f"{BUILD_OPTIONS_PAGE}:{lineno}: {name} is {opt['type']}, not {typ}")
        want = opt["value"] if opt["value"] != "" else "(empty)"
        if default != want:
            errors.append(f"{BUILD_OPTIONS_PAGE}:{lineno}: {name} defaults to {want}, not {default}")
    for name, defaults in cmake.items():
        if name not in rows:
            errors.append(f"{BUILD_OPTIONS_PAGE}: CMake option {name} is not listed")
            continue
        lineno, typ, doc_default = rows[name]
        words = re.findall(r"\w+", doc_default)
        if typ != "BOOL":
            errors.append(f"{BUILD_OPTIONS_PAGE}:{lineno}: {name} is BOOL, not {typ}")
        if not defaults or words[:1] not in [[d] for d in defaults] or set(defaults) - set(words):
            want = " or ".join(defaults)
            errors.append(f"{BUILD_OPTIONS_PAGE}:{lineno}: {name} defaults to {want}, not {doc_default}")
    for name in rows:
        if name not in options and name not in cmake:
            errors.append(f"{BUILD_OPTIONS_PAGE}: {name} is not a build option")
    return errors


# ------------------------------------------------------- static block checks

SYNTAX_LANGS = {"python", "bash", "toml", "json", "yaml"}


def python_surface(root: Path = ROOT) -> tuple[set[str], dict[str, set[str]]]:
    """Top-level public names and the members of each class."""
    names = py_public_names(root)
    top = {n.split(".", 1)[1] for n in names if n.count(".") == 1}
    members: dict[str, set[str]] = {}
    for n, info in names.items():
        parts = n.split(".")
        if len(parts) == 3:
            members.setdefault(parts[1], set()).add(parts[2])
    # Excluded names still resolve; their checks live elsewhere.
    top |= PY_EXCLUDE | {"__version__"}
    classes = {n.split(".")[1] for n, i in names.items() if i.kind == "class"}
    for cls in classes:
        members.setdefault(cls, set())
    return top, members


def python_name_errors(code: str, top: set[str], members: dict[str, set[str]]) -> list[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"syntax error: {exc.msg} (line {exc.lineno})"]
    aliases = set()
    imported = {}
    errors = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "rgpot":
                    aliases.add(alias.asname or "rgpot")
        elif isinstance(node, ast.ImportFrom) and node.module == "rgpot":
            for alias in node.names:
                if alias.name not in top:
                    errors.append(f"rgpot has no public name {alias.name}")
                imported[alias.asname or alias.name] = alias.name
    instances: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id in aliases and node.attr not in top:
                errors.append(f"rgpot has no public name {node.attr}")
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            cls = _called_class(node.value.func, aliases, imported)
            if cls and cls in members:
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        instances[target.id] = cls
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            cls = instances.get(node.value.id)
            if cls and node.attr not in members[cls]:
                errors.append(f"rgpot.{cls} has no member {node.attr}")
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Attribute):
            inner = node.value
            if isinstance(inner.value, ast.Name) and inner.value.id in aliases:
                cls = inner.attr
                if cls in members and node.attr not in members[cls]:
                    errors.append(f"rgpot.{cls} has no member {node.attr}")
    return sorted(set(errors))


def _called_class(func, aliases, imported):
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        if func.value.id in aliases:
            return func.attr
    if isinstance(func, ast.Name) and func.id in imported:
        return imported[func.id]
    return None


_C_IDENT = re.compile(r"\b(rgpot_[a-z0-9_]+)\b")
_CPP_QUAL = re.compile(r"\brgpot::((?:\w+::)*\w+)")
_INCLUDE = re.compile(r'#\s*include\s*[<"]([^>"]+)[>"]')
INCLUDE_ROOTS = ("include", "CppCore", "rgpot-core/include", "third_party", "CppCore/rgpot")
# Headers the build generates or third parties install.
GENERATED_INCLUDES = {"Potentials.capnp.h", "rgpot/rpc/Potentials.capnp.h"}


def c_name_errors(code: str, root: Path, corpus: str) -> list[str]:
    errors = []
    for m in _C_IDENT.finditer(code):
        name = m.group(1)
        if not re.search(rf"\b{re.escape(name)}\b", corpus):
            errors.append(f"{name} is not declared in the source")
    for m in _CPP_QUAL.finditer(code):
        last = m.group(1).split("::")[-1]
        if not re.search(rf"\b{re.escape(last)}\b", corpus):
            errors.append(f"rgpot::{m.group(1)} is not declared in the source")
    for m in _INCLUDE.finditer(code):
        inc = m.group(1)
        if not (inc.startswith(("rgpot", "Potentials")) or inc.endswith(".hpp") and "/" in inc):
            continue
        if inc in GENERATED_INCLUDES or inc.endswith(".capnp.h"):
            continue
        if not any((root / base / inc).is_file() for base in INCLUDE_ROOTS):
            errors.append(f"#include {inc} does not exist")
    return sorted(set(errors))


def _code_corpus(root: Path) -> str:
    parts = []
    for path in tracked_files(root):
        if path.endswith((".h", ".hpp", ".cc", ".cpp", ".c", ".rs", ".capnp", ".f90", ".F90")):
            try:
                parts.append((root / path).read_text(encoding="utf-8", errors="ignore"))
            except OSError:
                pass
    return "\n".join(parts)


def bash_errors(code: str, root: Path, ctx) -> list[str]:
    errors = []
    # The path from PATH: for a bare name Windows searches System32, where
    # WSL puts its own bash.exe, before PATH.  Bytes, because a text pipe
    # there writes CRLF, and bash then reads "then\r" for "then".
    bash = shutil.which("bash")
    if bash:
        proc = subprocess.run([bash, "-n"], input=code.encode(), capture_output=True)
        if proc.returncode != 0:
            errors.append(f"bash -n: {proc.stderr.decode(errors='replace').strip()}")
    errors.extend(e.split(": ", 1)[1] for e in facts_line_errors(code, root, ctx))
    return errors


def facts_line_errors(text: str, root: Path, ctx) -> list[str]:
    out = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for m in _MESON_D.finditer(line):
            name, value = m.group(1), m.group(2)
            if name.startswith("RGPOT") or name.isupper() or name in MESON_BUILTIN:
                continue
            if name not in ctx["options"]:
                out.append(f"{lineno}: -D{name} is not a meson option")
            elif value and not meson_value_ok(ctx["options"][name], value):
                out.append(f"{lineno}: -D{name}={value} is not a valid value")
        for m in _PIXI_CMD.finditer(line):
            out.extend(_pixi_errors(str(lineno), m.group(1), m.group(2), ctx["envs"], ctx["tasks"]))
        for m in _PATH_TOKEN.finditer(line):
            if not _path_exists(m.group(1), root, ctx["tracked"], ctx["dirs"]):
                out.append(f"{lineno}: path {m.group(1)} does not exist")
    return out


def static_context(root: Path = ROOT) -> dict:
    tracked = set(tracked_files(root))
    dirs = {str(Path(p).parent.as_posix()) for p in tracked}
    for d in list(dirs):
        parts = d.split("/")
        for i in range(1, len(parts)):
            dirs.add("/".join(parts[:i]))
    envs, tasks = pixi_manifest(root)
    top, members = python_surface(root)
    return {
        "tracked": tracked,
        "dirs": dirs,
        "options": meson_options(root),
        "envs": envs,
        "tasks": tasks,
        "py_top": top,
        "py_members": members,
        "corpus": _code_corpus(root),
    }


def static_block_errors(block: Block, root: Path, ctx: dict) -> list[str]:
    """The checks every class gets without a build, and unclassified blocks."""
    if block.kind in ("skip", "output", "signature"):
        return []
    lang = block.lang
    if lang == "python":
        return python_name_errors(block.body, ctx["py_top"], ctx["py_members"])
    if lang == "bash":
        return bash_errors(block.body, root, ctx)
    if lang in ("c", "cpp"):
        return c_name_errors(block.body, root, ctx["corpus"])
    if lang == "toml":
        try:
            tomllib.loads(block.body)
        except tomllib.TOMLDecodeError as exc:
            return [f"toml: {exc}"]
    if lang == "json":
        try:
            json.loads(block.body)
        except json.JSONDecodeError as exc:
            return [f"json: {exc}"]
    if lang in ("meson", "cmake", "fortran", "rust", "julia", "capnp"):
        return []
    return []


def check_blocks_static(root: Path = ROOT, blocks: list[Block] | None = None) -> list[str]:
    whole = blocks is None
    blocks = blocks if blocks is not None else all_blocks(root)
    errors = classification_errors(blocks)
    if whole:
        errors.extend(readme_errors(root))
    ctx = static_context(root)
    for block in blocks:
        for problem in static_block_errors(block, root, ctx):
            errors.append(f"{block.where()}: {problem}")
    return errors


# ------------------------------------------------------------ execution

RUNNERS = {
    "run": {"python", "bash", "c", "cpp"},
    "compile": {"c", "cpp", "fortran", "python"},
}


@dataclass
class BuildInfo:
    build_dir: Path
    cpp_flags: list[str]
    c_flags: list[str]
    lib_dirs: list[str]
    python_ok: bool
    features: dict[str, bool]
    targets: list[dict] = field(default_factory=list)
    compilers: dict[str, str] = field(default_factory=dict)


def _flags_from_compile_commands(build_dir: Path, suffix: str) -> list[str]:
    path = build_dir / "compile_commands.json"
    if not path.is_file():
        return []
    for cmd in json.loads(path.read_text()):
        if not cmd["file"].endswith(suffix):
            continue
        args = cmd.get("arguments") or shlex.split(cmd["command"])
        flags = []
        it = iter(args[1:])
        for arg in it:
            if arg in ("-I", "-isystem", "-D", "-std"):
                value = next(it)
                arg = arg + value if arg != "-isystem" else f"-isystem{value}"
            if arg.startswith(("-I", "-D", "-std=", "-isystem")):
                if arg.startswith("-I") and not os.path.isabs(arg[2:]):
                    arg = "-I" + str((Path(cmd["directory"]) / arg[2:]).resolve())
                flags.append(arg)
        return flags
    return []


def build_info(build_dir: Path | None, root: Path = ROOT) -> BuildInfo | None:
    if build_dir is None:
        return None
    build_dir = build_dir.resolve()
    if not build_dir.is_dir():
        return None
    # A test has the RGPOT_HAS_* macros a consumer of librgpot gets; some
    # libraries, such as librgpot_mpi, have none, and RGPOT_HAS_CACHE changes
    # the C++ layout stamp.
    cpp = (
        _flags_from_compile_commands(build_dir, "CppCore/tests/ExprEindirTest.cc")
        or _flags_from_compile_commands(build_dir, "CppCore/tests/CxxLayoutTest.cc")
        or _flags_from_compile_commands(build_dir, ".cc")
    )
    c = _flags_from_compile_commands(build_dir, ".c")
    libs = sorted(
        {str(p.parent) for glob in ("*.so*", "*.dylib") for p in build_dir.rglob(glob) if p.is_file()}
    )
    python_ok = False
    features = {}
    _stage_python(build_dir, root)
    env = _python_env(build_dir, root)
    proc = subprocess.run(
        [sys.executable, "-c", "import json, rgpot; print(json.dumps({k: bool(getattr(rgpot, k)) for k in dir(rgpot) if k.startswith('has_')}))"],
        capture_output=True, text=True, env=env, cwd=root,
    )
    if proc.returncode == 0:
        python_ok = True
        features = json.loads(proc.stdout.strip().splitlines()[-1])
    try:
        targets = json.loads((build_dir / "meson-info" / "intro-targets.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        targets = []
    compilers = _build_compilers(build_dir)
    return BuildInfo(build_dir, cpp, c, libs, python_ok, features, targets, compilers)


def _build_compilers(build_dir: Path) -> dict[str, str]:
    path = build_dir / "meson-info" / "intro-compilers.json"
    try:
        host = json.loads(path.read_text(encoding="utf-8")).get("host", {})
    except (OSError, ValueError, AttributeError):
        return {}
    out: dict[str, str] = {}
    extra_dirs = [str(Path(sys.executable).resolve().parent)]
    for lang, meta in host.items():
        exelist = meta.get("exelist") if isinstance(meta, dict) else None
        if not exelist:
            continue
        dirs = list(extra_dirs)
        for token in exelist:
            if os.path.isabs(token):
                dirs.insert(0, str(Path(token).parent))
        search_path = os.pathsep.join([os.environ.get("PATH", ""), *dirs])
        cand = exelist[-1]
        resolved = cand if (os.path.isabs(cand) and os.access(cand, os.X_OK)) else shutil.which(cand, path=search_path)
        if resolved:
            out[lang] = resolved
    return out


def _stage_python(build_dir: Path, root: Path) -> Path | None:
    """Assemble ``rgpot`` from the sources and the extension built in build_dir."""
    built = sorted((build_dir / "python").glob("_core*.so")) + sorted(
        (build_dir / "python").glob("_core*.pyd")
    )
    if not built:
        return None
    stage = build_dir / "doccheck-python"
    pkg = stage / "rgpot"
    want = {
        item.name: item.resolve()
        for item in (root / "python" / "rgpot").iterdir()
        if item.name != "__pycache__"
        and not (item.name.startswith("_core.") and item.suffix != ".pyi")
    }
    want.update({ext.name: ext.resolve() for ext in built})
    if pkg.is_dir():
        have = {p.name: p.resolve() for p in pkg.iterdir() if p.name != "__pycache__"}
        if have == want:
            return stage
    # Build beside the old tree and swap, so a concurrent run keeps a whole package.
    fresh = stage / f"rgpot.{os.getpid()}"
    fresh.mkdir(parents=True)
    for name, target in want.items():
        (fresh / name).symlink_to(target)
    old = stage / f"rgpot.old.{os.getpid()}"
    if pkg.exists():
        pkg.rename(old)
    fresh.rename(pkg)
    shutil.rmtree(old, ignore_errors=True)
    return stage


def _python_env(build_dir: Path, root: Path) -> dict:
    env = dict(os.environ)
    stage = build_dir / "doccheck-python"
    paths = [str(stage)] if (stage / "rgpot").is_dir() else []
    env["PYTHONPATH"] = os.pathsep.join(
        paths + [str(root / "python"), str(root / "tests"), env.get("PYTHONPATH", "")]
    )
    lib = str(build_dir / "CppCore")
    env["LD_LIBRARY_PATH"] = os.pathsep.join([lib, env.get("LD_LIBRARY_PATH", "")])
    env["RGPOT_BUILD_DIR"] = str(build_dir)
    return env


def _flag_features(flags: list[str]) -> dict[str, bool]:
    """``has_<name>`` for each ``-DRGPOT_HAS_<NAME>``, the names ``rgpot`` exports."""
    out = {}
    for flag in flags:
        match = re.match(r"-DRGPOT_HAS_(\w+?)(?:=.*)?$", flag)
        if match:
            out["has_" + match.group(1).lower()] = True
    return out


def _option_features(build_dir: Path) -> dict[str, bool]:
    """``has_rpc`` when the build has RPC on, as ``rpc_enabled`` in meson.build.

    Only a ``with_rust_core`` build also defines ``RGPOT_HAS_RPC``.
    """
    path = build_dir / "meson-info" / "intro-buildoptions.json"
    try:
        options = {o["name"]: o["value"] for o in json.loads(path.read_text(encoding="utf-8"))}
    except (OSError, ValueError, KeyError, TypeError):
        return {}
    return {"has_rpc": options.get("with_rpc") is True or options.get("with_rpc_client_only") is True}


def run_block(block: Block, info: BuildInfo, root: Path = ROOT) -> tuple[str, str]:
    """Return (status, detail) with status one of ok, fail, skip."""
    requires = [r for r in block.options.get("requires", "").split(",") if r]
    if block.lang == "python":
        if not info.python_ok:
            return "skip", "rgpot is not importable from the build"
        missing = [r for r in requires if r.startswith("has_") and not info.features.get(r)]
        if missing:
            return "skip", f"build lacks {', '.join(missing)}"
        if block.kind == "compile":
            try:
                compile(block.body, block.where(), "exec")
            except SyntaxError as exc:
                return "fail", str(exc)
            return "ok", ""
        return _run([sys.executable, "-"], block.body, _python_env(info.build_dir, root), root)
    if block.lang == "bash":
        env = _python_env(info.build_dir, root)
        return _run(["bash", "-euo", "pipefail", "-s"], block.body, env, root)
    if block.lang in ("c", "cpp"):
        have = _flag_features(info.cpp_flags) | _option_features(info.build_dir)
        missing = [r for r in requires if r.startswith("has_") and not have.get(r)]
        if missing:
            return "skip", f"build lacks {', '.join(missing)}"
        return _compile_c(block, info, root)
    if block.lang == "fortran":
        fc_env = os.environ.get("FC")
        fc = (shutil.which(fc_env) if fc_env else info.compilers.get("fortran")) or shutil.which("gfortran")
        if not fc:
            return "skip", "no Fortran compiler"
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "block.f90"
            src.write_text(block.body)
            return _run([fc, "-fsyntax-only", "-ffree-line-length-none", str(src)], "", _tool_env(fc), Path(tmp))
    return "skip", f"no runner for {block.lang}"


def _tool_env(*tools: str) -> dict[str, str] | None:
    dirs: list[str] = []
    for tool in tools:
        if tool and os.path.isabs(tool):
            parent = str(Path(tool).parent)
            if parent not in dirs:
                dirs.append(parent)
    if not dirs:
        return None
    env = dict(os.environ)
    cur = env.get("PATH", "")
    parts = [p for p in cur.split(os.pathsep) if p]
    for d in dirs:
        if d not in parts:
            parts.append(d)
    env["PATH"] = os.pathsep.join(parts)
    return env


def _compile_c(block: Block, info: BuildInfo, root: Path) -> tuple[str, str]:
    cpp = block.lang == "cpp"
    env_key = "CXX" if cpp else "CC"
    lang_key = "cpp" if cpp else "c"
    default_key = "c++" if cpp else "cc"
    env_val = os.environ.get(env_key)
    compiler = (shutil.which(env_val) if env_val else info.compilers.get(lang_key)) or shutil.which(default_key)
    if not compiler:
        return "skip", "no compiler"
    compile_env = _tool_env(compiler)
    flags = info.cpp_flags if cpp else (info.c_flags or [f for f in info.cpp_flags if not f.startswith("-std")])
    flags = flags + [f"-I{root / d}" for d in ("CppCore", "include", "rgpot-core/include")]
    if cpp and not any(f.startswith("-std") for f in flags):
        flags = flags + ["-std=c++20"]
    extra = shlex.split(block.options.get("flags", "").replace(",", " "))
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / ("block.cc" if cpp else "block.c")
        src.write_text(block.body + "\n")
        if block.kind == "compile":
            return _run([compiler, "-fsyntax-only", *flags, *extra, str(src)], "", compile_env, root)
        obj, exe = Path(tmp) / "block.o", Path(tmp) / "block"
        status, detail = _run([compiler, "-c", *flags, *extra, str(src), "-o", str(obj)], "", compile_env, root)
        if status != "ok":
            return status, detail
        names = [name for name in block.options.get("link", "").split(",") if name]
        linker, args = _meson_link(info, names, block.lang) or (
            [compiler], [f"-L{d}" for d in info.lib_dirs] + [f"-l{name}" for name in names]
        )
        search_path = (compile_env or os.environ).get("PATH", "")
        resolved_linker = linker[0] if (os.path.isabs(linker[0]) and os.access(linker[0], os.X_OK)) else shutil.which(linker[0], path=search_path)
        if not resolved_linker:
            return "skip", f"no linker {linker[0]}"
        linker = [resolved_linker, *linker[1:]]
        rpath = [f"-Wl,-rpath,{d}" for d in info.lib_dirs]
        # Meson names its own outputs relative to the build directory.
        status, detail = _run([*linker, str(obj), "-o", str(exe), *args, *rpath], "", _tool_env(compiler, resolved_linker), info.build_dir)
        if status != "ok":
            return status, detail
        return _run([str(exe)], "", None, root)


def _meson_link(info: BuildInfo, names: list[str], lang: str) -> tuple[list[str], list[str]] | None:
    """Linker and arguments of the plainest ``lang`` program meson links to ``names``.

    A static library carries none of its dependencies: a program that links
    librgpot.a names rocksdb, OpenMP and the Fortran runtime itself, and a C
    program links with the C++ driver, which is what brings in the C++
    runtime on Darwin.  Meson records that line for every executable it links.
    """
    libs = [t for t in info.targets if t["type"].endswith("library")]
    outputs = [{Path(f) for t in libs if t["name"] == name for f in t["filename"]} for name in names]
    if not outputs or not all(outputs):
        return None
    lines = []
    for target in info.targets:
        if target["type"] != "executable":
            continue
        langs = {s["language"] for s in target["target_sources"] if "language" in s}
        for link in (s for s in target["target_sources"] if "linker" in s):
            named = {info.build_dir / p for p in link["parameters"]}
            if langs == {lang} and all(out & named for out in outputs):
                lines.append(link)
    if not lines:
        return None
    # The fewest arguments: capi_threads, not its -fsanitize=thread twin.
    best = min(lines, key=lambda link: len(link["parameters"]))
    return best["linker"], best["parameters"]


def _run(cmd, stdin, env, cwd, timeout=300) -> tuple[str, str]:
    try:
        proc = subprocess.run(
            cmd, input=stdin, capture_output=True, text=True, env=env, cwd=cwd, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        return "fail", f"timed out after {timeout} s"
    if proc.returncode != 0:
        return "fail", (proc.stdout + proc.stderr).strip()[-2000:]
    return "ok", proc.stdout


def execute_blocks(build_dir: Path | None, root: Path = ROOT, blocks=None):
    """Yield (block, status, detail) for every run or compile block."""
    info = build_info(build_dir, root)
    blocks = blocks if blocks is not None else all_blocks(root)
    for block in blocks:
        if block.kind not in ("run", "compile"):
            continue
        if info is None:
            yield block, "skip", "no build directory"
            continue
        status, detail = run_block(block, info, root)
        yield block, status, detail


# ---------------------------------------------------------------- command


# (page, text that must remain). Each token is a DOI or a reference value
# checked while writing the page.
COVERAGE_MARKERS = (
    ("docs/orgmode/reference/gupta.org", "10.1103/PhysRevB.23.6265"),
    ("docs/orgmode/reference/gupta.org", "10.1103/PhysRevB.48.22"),
    ("docs/orgmode/reference/gupta.org", "-3.1678"),
    ("docs/orgmode/reference/gupta.org", "10.1021/jp970984n"),
    ("docs/orgmode/reference/gupta.org", "573.011875"),
    ("docs/orgmode/reference/cpp_potentials.org", "10.1098/rspa.1924.0082"),
    ("docs/orgmode/reference/cpp_potentials.org", "-0.999999648834"),
    ("docs/orgmode/reference/cpp_potentials.org", "10.1103/PhysRev.34.57"),
    ("docs/orgmode/reference/cpp_potentials.org", "10.1063/1.2841941"),
    ("docs/orgmode/reference/cpp_potentials.org", "-0.710164462003"),
    ("docs/orgmode/reference/cpp_potentials.org", "10.1103/PhysRevB.29.6443"),
    ("docs/orgmode/reference/cpp_potentials.org", "3.63262"),
    ("docs/orgmode/reference/cpp_potentials.org", "10.1103/PhysRevB.35.7423"),
    ("docs/orgmode/reference/cpp_potentials.org", "5.129167"),
    ("docs/orgmode/reference/cpp_potentials.org", "10.1063/1.445869"),
    ("docs/orgmode/reference/cpp_potentials.org", "10.1021/j100308a038"),
    ("docs/orgmode/reference/cpp_potentials.org", "10.1016/j.chemphys.2004.04.024"),
    ("docs/orgmode/reference/cpp_potentials.org", "10.1063/1.467012"),
    ("docs/orgmode/reference/cpp_potentials.org", "0.141344038570"),
    ("docs/orgmode/reference/cpp_potentials.org", "10.1063/1.3382344"),
    ("docs/orgmode/reference/cpp_potentials.org", "10.1063/1.5090222"),
    ("docs/orgmode/reference/sw_si.org", "10.1103/PhysRevB.31.5262"),
    ("docs/orgmode/reference/tersoff_sic.org", "10.1103/PhysRevB.39.5566"),
    ("docs/orgmode/reference/cpp_backends.org", "10.1103/PhysRevLett.55.2471"),
    ("docs/orgmode/reference/cpp_backends.org", "10.1021/acs.jctc.8b01176"),
    ("docs/orgmode/reference/cpp_backends.org", "10.1063/5.0304911"),
    ("docs/orgmode/reference/cpp_backends.org", "98374.87753058573"),
    ("docs/orgmode/reference/cpp_backends.org", "10.1016/j.cpc.2004.12.014"),
    ("docs/orgmode/reference/cpp_backends.org", "50.613487749571874"),
    ("docs/orgmode/reference/cpp_backends.org", "10.1021/ja00299a024"),
    ("docs/orgmode/reference/cpp_backends.org", "10.1007/s00894-007-0233-4"),
    ("docs/orgmode/reference/cpp_backends.org", "6.802846561497"),
    ("docs/orgmode/reference/cpp_backends.org", "10.1016/j.cpc.2010.04.018"),
    ("docs/orgmode/reference/cpp_backends.org", "10.1063/5.0004997"),
    ("docs/orgmode/reference/cpp_backends.org", "-2070.0"),
    ("docs/orgmode/reference/cpp_backends.org", "10.48550/arXiv.2506.14665"),
    ("docs/orgmode/reference/cpp_backends.org", "2.8 kcal/mol"),
    ("docs/orgmode/reference/cpp_backends.org", "10.1002/wcms.1493"),
    ("docs/orgmode/reference/cpp_backends.org", "10.1021/acs.jctc.7b00118"),
    ("docs/orgmode/reference/cpp_backends.org", "10.1039/c7sc00601b"),
    ("docs/orgmode/reference/cpp_backends.org", "-137.9"),
    ("docs/orgmode/reference/cpp_backends.org", "10.26434/chemrxiv.8326202.v1"),
    ("docs/orgmode/reference/cpp_backends.org", "10.1002/anie.202004239"),
    ("docs/orgmode/reference/cpp_backends.org", "10.52202/085713-4310"),
    ("docs/orgmode/reference/cpp_backends.org", "-2542.4200323230975"),
    ("docs/orgmode/reference/cpp_backends.org", "10.1016/j.cpc.2013.10.027"),
    ("docs/orgmode/reference/cpp_backends.org", "10.1016/j.cpc.2018.09.020"),
    ("docs/orgmode/reference/cpp_backends.org", "10.1016/j.cpc.2021.108171"),
    ("docs/orgmode/reference/cpp_backends.org", "10.1016/j.softx.2015.06.001"),
    ("docs/orgmode/reference/cpp_backends.org", "10.1371/journal.pcbi.1005659"),
    ("docs/orgmode/reference/cpp_potentials.org", "-2.7114096242662238"),
    ("docs/orgmode/reference/cpp_potentials.org", "-5.217864006173226"),
    ("docs/orgmode/reference/cpp_potentials.org", "10.1103/PhysRevB.56.8542"),
    ("docs/orgmode/reference/cpp_potentials.org", "-18.838135"),
    ("docs/orgmode/reference/cpp_potentials.org", "10.1088/0953-8984/16/27/003"),
    ("docs/orgmode/reference/cpp_potentials.org", "10.1080/00268978700101941"),
    ("docs/orgmode/reference/cpp_potentials.org", "-43.959774"),
    ("docs/orgmode/reference/cpp_potentials.org", "10.1088/0965-0393/8/6/305"),
    ("docs/orgmode/reference/cpp_potentials.org", "-17.284558"),
    ("docs/orgmode/reference/cpp_potentials.org", "10.1103/PhysRevB.31.5262"),
    ("docs/orgmode/reference/cpp_potentials.org", "-16.204955"),
    ("docs/orgmode/reference/cpp_potentials.org", "10.1103/PhysRevB.39.5566"),
    ("docs/orgmode/reference/cpp_potentials.org", "-17.440266"),
    ("docs/orgmode/reference/cpp_potentials.org", "10.1063/1.2162901"),
    ("docs/xckernel.md", "2608.26440"),
    ("docs/xckernel.md", "10.1063/5.0358101"),
    ("docs/orgmode/explanation/global-optimization.org", "10.3389/fchem.2026.1807063"),
    ("docs/orgmode/explanation/global-optimization.org", "10.1063/5.0283639"),
)

DESIGN_NOTES = (
    "eam_cell.md",
    "neighbour_layer.md",
    "xckernel.md",
    "xtb_backends.md",
    "neighbour_api_design.md",
    "eon_nlist.md",
    "eon_pkgconfig.md",
)


def _toctree_targets(text: str) -> list[str]:
    """Sphinx toctree entries in an org page, in source order."""
    targets = []
    active = False
    for line in text.splitlines():
        if ".. toctree::" in line:
            active = True
            continue
        if not active:
            continue
        if not line.strip():
            continue
        if line[:1].strip():
            active = False
            if ".. toctree::" in line:
                active = True
            continue
        token = line.strip()
        if token.startswith(":"):
            continue
        targets.append(token.split("#", 1)[0])
    return targets


def org_index_errors(root: Path = ROOT) -> list[str]:
    """Org pages not reachable by toctree from the manual index."""
    orgmode = root / "docs/orgmode"
    pages = {path.resolve() for path in orgmode.rglob("*.org")}
    start = (orgmode / "index.org").resolve()
    seen: set[Path] = set()
    stack = [start]
    while stack:
        page = stack.pop()
        if page in seen or not page.is_file():
            continue
        seen.add(page)
        for token in _toctree_targets(page.read_text(encoding="utf-8")):
            target = (page.parent / token).resolve()
            org = target if target.suffix == ".org" else Path(str(target) + ".org")
            if org.is_file():
                stack.append(org)
    return [
        f"{path.relative_to(root).as_posix()}: not reachable from docs/orgmode/index.org"
        for path in sorted(pages - seen)
    ]


def coverage_errors(root: Path = ROOT) -> list[str]:
    """Index reachability and the pinned citations and reference values."""
    errors = org_index_errors(root)
    index = (root / "docs/orgmode/index.org").read_text(encoding="utf-8")
    for name in DESIGN_NOTES:
        stem = name[: -len(".md")]
        if f"notes/{stem}" not in index:
            errors.append(f"docs/orgmode/index.org: design note {name} is not in the index")
        if not (root / "docs" / name).is_file():
            errors.append(f"docs/{name}: design note listed in the index is missing")
    for page, marker in COVERAGE_MARKERS:
        path = root / page
        if not path.is_file() or marker not in path.read_text(encoding="utf-8"):
            errors.append(f"{page}: missing pinned text {marker}")
    return errors


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("check", choices=("blocks", "reference", "facts", "coverage", "all"))
    parser.add_argument("--build-dir", type=Path, default=None)
    parser.add_argument("--root", type=lambda p: Path(p).resolve(), default=ROOT)
    parser.add_argument(
        "--page", action="append", default=[],
        help="report only problems located in this page (repeatable)",
    )
    args = parser.parse_args(argv)
    errors: list[str] = []
    pages = [p.rstrip("/") for p in args.page]

    def wanted(where: str) -> bool:
        return not pages or any(where.startswith(p) for p in pages)

    if args.check in ("blocks", "all"):
        errors.extend(check_blocks_static(args.root))
        if args.build_dir is not None:
            blocks = [b for b in all_blocks(args.root) if wanted(rel(b.path, args.root))]
            for block, status, detail in execute_blocks(args.build_dir, args.root, blocks):
                if status == "fail":
                    errors.append(f"{block.where()}: {block.kind} failed:\n{detail}")
                elif status == "skip":
                    print(f"skip {block.where()}: {detail}")
    if args.check in ("reference", "all"):
        errors.extend(reference_errors(args.root))
    if args.check in ("facts", "all"):
        errors.extend(facts_errors(args.root))
    if args.check in ("coverage", "all"):
        errors.extend(coverage_errors(args.root))
    if pages:
        errors = [e for e in errors if wanted(e)]
    for error in errors:
        print(error, file=sys.stderr)
    if errors:
        print(f"{len(errors)} documentation problem(s)", file=sys.stderr)
        return 1
    print(f"doccheck {args.check}: clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
