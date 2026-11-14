#!/usr/bin/env python3
"""Contract check for rgpot RPC schema copies."""

from __future__ import annotations

import difflib
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def _struct_body(lines: list[str], name: str, rel: Path) -> list[str]:
    needle = f"struct {name} {{"
    try:
        start = lines.index(needle)
    except ValueError as exc:
        raise AssertionError(f"{rel} must expose {name}") from exc
    depth = 0
    body: list[str] = []
    for line in lines[start:]:
        body.append(line)
        depth += line.count("{") - line.count("}")
        if depth == 0:
            return body
    raise AssertionError(f"{rel} has unclosed struct {name}")


def _assert_expr_potspec(schema_path: Path, lines: list[str]) -> None:
    rel = schema_path
    expr_term = "\n".join(_struct_body(lines, "ExprTerm", rel))
    if not re.search(r"name\s+@0\s*:\s*Text", expr_term):
        raise AssertionError(f"{rel} ExprTerm.name must be Text")
    if not re.search(r"pot\s+@1\s*:\s*PotSpec", expr_term):
        raise AssertionError(f"{rel} ExprTerm.pot must be PotSpec")

    expr_params = "\n".join(_struct_body(lines, "ExprParams", rel))
    if not re.search(r"expression\s+@0\s*:\s*Text", expr_params):
        raise AssertionError(f"{rel} ExprParams.expression must be Text")
    if not re.search(r"terms\s+@1\s*:\s*List\(ExprTerm\)", expr_params):
        raise AssertionError(f"{rel} ExprParams.terms must be List(ExprTerm)")

    pot_spec = "\n".join(_struct_body(lines, "PotSpec", rel))
    if not re.search(r"expr\s+@0\s*:\s*ExprParams", pot_spec):
        raise AssertionError(f"{rel} PotSpec.expr must be ExprParams")
    if not re.search(r"none\s+@1\s*:\s*Void", pot_spec):
        raise AssertionError(f"{rel} PotSpec.none must be Void (capnp unions need two arms)")
    if re.search(r"^\s+(lj|d3|d4|nwchem)\s+@", pot_spec, re.MULTILINE):
        raise AssertionError(f"{rel} PotSpec must not invent dummy leaf arms")

    cfg = "\n".join(_struct_body(lines, "PotentialConfig", rel))
    if not re.search(r"^\s+expr\s+@16\s*:\s*ExprParams;", cfg, re.MULTILINE):
        raise AssertionError(f"{rel} PotentialConfig.expr must be ExprParams @16")
    if re.search(r"^\s+(sum|list)\s+@", cfg, re.MULTILINE):
        raise AssertionError(f"{rel} PotentialConfig must not grow sum/list arms")


_CP2K_FIELDS = (
    ("inputTemplate", 0, "Text"),
    ("functional", 1, "Text"),
    ("basisSet", 2, "Text"),
    ("pseudopotential", 3, "Text"),
    ("basisSetFile", 4, "Text"),
    ("potentialFile", 5, "Text"),
    ("cutoffRy", 6, "Float64"),
    ("relCutoffRy", 7, "Float64"),
    ("charge", 8, "Int32"),
    ("multiplicity", 9, "Int32"),
    ("uks", 10, "Bool"),
    ("periodic", 11, "Text"),
    ("epsScf", 12, "Float64"),
    ("maxScf", 13, "Int32"),
    ("outerMaxScf", 14, "Int32"),
    ("ot", 15, "Bool"),
    ("dataDir", 16, "Text"),
    ("workDir", 17, "Text"),
    ("keepWorkDir", 18, "Bool"),
    ("enginePath", 19, "Text"),
    ("extraDftSections", 20, r"List\(Text\)"),
    ("extraForceEvalSections", 21, r"List\(Text\)"),
)


def _assert_cp2k_arm(schema_path: Path, lines: list[str]) -> None:
    """CP2KParams keeps its ordinals and both unions carry the cp2k arm."""
    rel = schema_path
    params = "\n".join(_struct_body(lines, "CP2KParams", rel))
    for name, ordinal, kind in _CP2K_FIELDS:
        if not re.search(rf"^\s+{name}\s+@{ordinal}\s*:\s*{kind}\s*[;=]", params, re.MULTILINE):
            raise AssertionError(f"{rel} CP2KParams.{name} must be @{ordinal} :{kind}")
    if re.search(rf"@{len(_CP2K_FIELDS)}\b", params):
        raise AssertionError(f"{rel} CP2KParams has a field this check does not know")

    spec = "\n".join(_struct_body(lines, "PotentialSpec", rel))
    if not re.search(r"^\s+cp2k\s+@6\s*:\s*CP2KParams;", spec, re.MULTILINE):
        raise AssertionError(f"{rel} PotentialSpec.cp2k must be CP2KParams @6")
    cfg = "\n".join(_struct_body(lines, "PotentialConfig", rel))
    if not re.search(r"^\s+cp2k\s+@17\s*:\s*CP2KParams;", cfg, re.MULTILINE):
        raise AssertionError(f"{rel} PotentialConfig.cp2k must be CP2KParams @17")


def _regenerate(root: Path, capnp: str, schema: Path, out: Path) -> dict[str, bytes]:
    """Run the repository's own generator wrapper and return the artifacts."""
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            str(root / "CppCore" / "rgpot" / "rpc" / "capnp_compile.py"),
            capnp,
            str(out),
            str(schema.parent),
            str(schema),
        ],
        check=True,
    )
    return {p.name: p.read_bytes() for p in sorted(out.iterdir())}


def _request(capnp: str, schema: Path) -> bytes:
    """CodeGeneratorRequest bytes: the one input every capnp backend sees."""
    return subprocess.run(
        [capnp, "compile", "-o-", f"--src-prefix={schema.parent}", str(schema)],
        check=True,
        stdout=subprocess.PIPE,
    ).stdout


def _assert_generated_in_sync(root: Path, capnp: str, build_dir: Path | None) -> None:
    cpp_schema = root / "CppCore" / "rgpot" / "rpc" / "Potentials.capnp"
    rust_schema = root / "rgpot-core" / "schema" / "Potentials.capnp"
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        # Each copy sits alone in its own directory so the generated names and
        # the embedded source paths cannot differ for incidental reasons.
        cpp_copy = tmp_path / "cpp" / "Potentials.capnp"
        rust_copy = tmp_path / "rust" / "Potentials.capnp"
        for src, dst in ((cpp_schema, cpp_copy), (rust_schema, rust_copy)):
            dst.parent.mkdir(parents=True)
            shutil.copyfile(src, dst)

        from_cpp = _regenerate(root, capnp, cpp_copy, tmp_path / "gen_cpp")
        from_rust = _regenerate(root, capnp, rust_copy, tmp_path / "gen_rust")
        assert set(from_cpp) == {"Potentials.capnp.cpp", "Potentials.capnp.h"}, from_cpp.keys()
        for name in from_cpp:
            if from_cpp[name] != from_rust[name]:
                raise AssertionError(f"regenerated {name} differs between the C++ and Rust schema copies")
        if _request(capnp, cpp_copy) != _request(capnp, rust_copy):
            raise AssertionError("CodeGeneratorRequest differs between the schema copies")

        if build_dir is not None:
            built = build_dir / "CppCore" / "rgpot" / "rpc"
            for name, fresh in from_cpp.items():
                artifact = built / name
                if not artifact.is_file():
                    raise AssertionError(f"{artifact} was not generated by the build")
                if artifact.read_bytes() != fresh:
                    raise AssertionError(
                        f"{artifact} is stale: regenerating the vendored schema gives different bytes"
                    )


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    capnp = sys.argv[2] if len(sys.argv) > 2 else shutil.which("capnp")
    build_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    cpp_schema = root / "CppCore" / "rgpot" / "rpc" / "Potentials.capnp"
    rust_schema = root / "rgpot-core" / "schema" / "Potentials.capnp"

    cpp_lines = cpp_schema.read_text(encoding="utf-8").splitlines()
    rust_lines = rust_schema.read_text(encoding="utf-8").splitlines()
    for schema_path, lines in ((cpp_schema, cpp_lines), (rust_schema, rust_lines)):
        if "enum CPMDSectionKind {" not in lines:
            rel = schema_path.relative_to(root)
            raise AssertionError(f"{rel} must expose CPMDSectionKind")
        _assert_expr_potspec(schema_path.relative_to(root), lines)
        _assert_cp2k_arm(schema_path.relative_to(root), lines)

    if cpp_lines == rust_lines:
        if capnp is None:
            raise AssertionError("capnp is required to regenerate and diff the artifacts")
        _assert_generated_in_sync(root, capnp, build_dir)
        return 0

    diff = difflib.unified_diff(
        rust_lines,
        cpp_lines,
        fromfile=str(rust_schema.relative_to(root)),
        tofile=str(cpp_schema.relative_to(root)),
        lineterm="",
        n=3,
    )
    print("\n".join(list(diff)[:120]))
    raise AssertionError("rgpot-core bundled schema must match C++ RPC schema")


if __name__ == "__main__":
    raise SystemExit(main())
