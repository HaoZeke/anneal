#!/usr/bin/env python3
"""Self-test for CppCore/rgpot/rpc/schema_ledger.py on small schemas."""

from __future__ import annotations

import configparser
import contextlib
import io
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "CppCore" / "rgpot" / "rpc"))

import schema_ledger as L  # noqa: E402

UPSTREAM = textwrap.dedent(
    """\
    @0xbd1f89fa17369103;

    struct ForceInput {
      pos @0 :List(Float64);
      lengthUnit @1 :Text = "angstrom";  # @brief ignored @7 in a comment
    }

    struct XTBParams {
      method @0 :Text = "gfn2";
    }

    struct MorseParams {
      de @0 :Float64 = 0.7102;
      re @1 :Float64 = 2.897;
    }

    struct PotentialConfig {
      union {
        none  @0 :Void;
        xtb   @2 :XTBParams;
        morse @3 :MorseParams;
      }
      common @1 :Text;
      extra :group {
        flag @4 :Bool;
      }
    }

    enum Kind {
      first @0;
      second @1;
    }

    interface Potential {
      calculate @0 (fip :ForceInput) -> (energy :Float64);
    }
    """
)

LOCAL_DIFF = textwrap.dedent(
    """\
    diff --git a/Potentials.capnp b/Potentials.capnp
    --- a/Potentials.capnp
    +++ b/Potentials.capnp
    @@ -3,6 +3,7 @@
     struct ForceInput {
       pos @0 :List(Float64);
       lengthUnit @1 :Text = "angstrom";  # @brief ignored @7 in a comment
    +  tag @2 :Text;
     }

     struct XTBParams {
    """
)


def _git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "commit.gpgsign=false", "-c", "user.name=t", "-c", "user.email=t@t", *args],
        cwd=root,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


class LedgerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.upstream = self.root / "upstream.capnp"
        self.write_tree(UPSTREAM, LOCAL_DIFF)
        self.assertEqual(self.run_tool("append", "--upstream", str(self.upstream)), 0)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def write_tree(self, upstream: str, *diffs: str) -> None:
        self.upstream.write_text(upstream, encoding="utf-8")
        names = [f"d{i}.diff" for i in range(len(diffs))]
        for directory in L.SERIES_DIRS:
            (self.root / directory).mkdir(parents=True, exist_ok=True)
            for old in (self.root / directory).iterdir():
                old.unlink()
            for name, diff in zip(names, diffs):
                (self.root / directory / name).write_text(diff, encoding="utf-8")
            (self.root / directory / L.SERIES).write_text(
                "# local series\n" + "".join(f"{n}\n" for n in names), encoding="utf-8"
            )
        wrap = self.root / L.WRAP
        wrap.parent.mkdir(parents=True, exist_ok=True)
        wrap.write_text(
            f"[wrap-git]\nurl = https://example.invalid/schema.git\nrevision = {'0' * 40}\n",
            encoding="utf-8",
        )
        self.set_vendored(upstream)

    def effective(self) -> str:
        return L.apply_series(self.vendored(), L.read_series(self.root / L.SERIES_DIRS[0]))

    def set_vendored(self, text: str) -> None:
        for rel in L.VENDORED:
            (self.root / rel).parent.mkdir(parents=True, exist_ok=True)
            (self.root / rel).write_text(text, encoding="utf-8")

    def vendored(self) -> str:
        return (self.root / L.VENDORED[0]).read_text(encoding="utf-8")

    def ledger(self) -> Path:
        return self.root / L.LEDGER

    def run_tool(self, *args: str) -> int:
        self.err = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(self.err):
            return L.main([args[0], "--root", str(self.root), *args[1:]])

    def check(self, *extra: str) -> int:
        return self.run_tool("check", "--upstream", str(self.upstream), *extra)

    def assertRejected(self, needle: str, *extra: str) -> None:
        self.assertEqual(self.check(*extra), 1, self.err.getvalue())
        self.assertIn(needle, self.err.getvalue())

    def test_seeded_ledger_passes(self) -> None:
        self.assertEqual(self.check(), 0, self.err.getvalue())
        lines = self.ledger().read_text(encoding="utf-8")
        self.assertIn("PotentialConfig @2 arm xtb :XTBParams upstream\n", lines)
        self.assertIn("PotentialConfig @4 field extra.flag :Bool upstream\n", lines)
        self.assertIn("Kind @1 enumerant second - upstream\n", lines)
        self.assertIn("Potential @0 method calculate (fip:ForceInput)->(energy:Float64) upstream\n", lines)
        self.assertIn("ForceInput @2 field tag :Text local\n", lines)
        self.assertNotIn("@7", lines)

    def test_renumbered_arm_in_vendored_copy(self) -> None:
        self.set_vendored(
            self.vendored()
            .replace("xtb   @2 :XTBParams", "xtb   @5 :XTBParams")
            .replace("    morse @3", "    morse @2")
            .replace("    xtb   @5", "    xtb   @3")
        )
        self.assertRejected(
            "has PotentialConfig @2 arm morse :MorseParams, ledger has PotentialConfig @2 arm xtb"
        )

    def test_retyped_field(self) -> None:
        self.set_vendored(self.vendored().replace("de @0 :Float64", "de @0 :Float32"))
        self.assertRejected("has MorseParams @0 field de :Float32")

    def test_diff_that_adds_an_upstream_slot(self) -> None:
        # 68df95cd's shape: the pin lacks the arm, a diff adds it at a slot the
        # ledger gives to another arm, and the vendored copies follow the diff.
        older = UPSTREAM.replace("    xtb   @2 :XTBParams;\n", "    # xtb @2 :XTBParams;\n").replace(
            "    morse @3 :MorseParams;\n", ""
        )
        morse_at_2 = textwrap.dedent(
            """\
            diff --git a/Potentials.capnp b/Potentials.capnp
            --- a/Potentials.capnp
            +++ b/Potentials.capnp
            @@ -20,6 +20,7 @@
               union {
                 none  @0 :Void;
                 # xtb @2 :XTBParams;
            +    morse @2 :MorseParams;
               }
               common @1 :Text;
               extra :group {
            """
        )
        self.write_tree(older, LOCAL_DIFF, morse_at_2)
        self.assertRejected(
            "d1.diff adds PotentialConfig @2 arm morse :MorseParams, ledger has PotentialConfig @2 arm xtb"
        )
        err = self.err.getvalue()
        self.assertIn("pinned upstream no longer declares PotentialConfig @3 arm morse", err)
        self.assertIn("lacks PotentialConfig @2 arm xtb :XTBParams upstream", err)

    def test_diff_that_removes_an_ordinal_line(self) -> None:
        rewrite = textwrap.dedent(
            """\
            diff --git a/Potentials.capnp b/Potentials.capnp
            --- a/Potentials.capnp
            +++ b/Potentials.capnp
            @@ -8,3 +8,3 @@
             struct XTBParams {
            -  method @0 :Text = "gfn2";
            +  method @0 :Text = "gfn1";
             }
            """
        )
        self.write_tree(UPSTREAM, LOCAL_DIFF, rewrite)
        self.assertRejected("d1.diff removes or rewrites an ordinal line: method @0")

    def test_diff_that_adds_an_unledgered_ordinal(self) -> None:
        extra = LOCAL_DIFF.replace("+  tag @2 :Text;\n", "+  tag @2 :Text;\n+  more @3 :Bool;\n").replace(
            "@@ -3,6 +3,7 @@", "@@ -3,6 +3,8 @@"
        )
        self.write_tree(UPSTREAM, extra)
        self.assertRejected("d0.diff adds ForceInput @3 field more :Bool, not a local ledger entry")

    def test_diff_that_adds_an_ordinal_in_a_comment(self) -> None:
        reserve = LOCAL_DIFF.replace(
            "+  tag @2 :Text;\n", "+  tag @2 :Text;\n+  # spare @3 :Bool;\n"
        ).replace("@@ -3,6 +3,7 @@", "@@ -3,6 +3,8 @@")
        self.write_tree(UPSTREAM, reserve)
        self.assertRejected("d0.diff adds an ordinal outside a declaration: # spare @3 :Bool;")

    def test_vendored_copy_diverges_from_the_pin(self) -> None:
        self.set_vendored(self.vendored().replace('"gfn2"', '"gfn1"'))
        self.assertRejected(L.DIVERGES)
        self.assertEqual(self.run_tool("check"), 0, self.err.getvalue())

    def test_vendored_copy_carrying_a_local_field(self) -> None:
        # eb357630's shape: the local field sits in the vendored copies, so a
        # parent comparing them with its own checkout of the pin sees drift.
        self.set_vendored(self.effective())
        self.assertRejected(L.DIVERGES)
        self.assertIn(
            "carries local ForceInput @2 field tag :Text; it belongs in the local series", self.err.getvalue()
        )
        self.assertEqual(self.run_tool("check"), 1)
        self.assertIn("carries local ForceInput @2 field tag", self.err.getvalue())

    def test_vendored_copies_differ(self) -> None:
        (self.root / L.VENDORED[1]).write_text(self.vendored() + "# drift\n", encoding="utf-8")
        self.assertEqual(self.run_tool("check"), 1)
        self.assertIn(f"{L.VENDORED[1]} differs from {L.VENDORED[0]}", self.err.getvalue())

    def test_wrap_diff_files_are_refused(self) -> None:
        wrap = self.root / L.WRAP
        wrap.write_text(wrap.read_text(encoding="utf-8") + "diff_files = d0.diff\n", encoding="utf-8")
        self.assertRejected("applies diff_files (d0.diff)")

    def test_series_mirror_drift(self) -> None:
        mirror = self.root / L.SERIES_DIRS[1] / "d0.diff"
        mirror.write_text(mirror.read_text(encoding="utf-8").replace("tag", "tags"), encoding="utf-8")
        self.assertRejected(f"{L.SERIES_DIRS[1] / 'd0.diff'} differs from {L.SERIES_DIRS[0] / 'd0.diff'}")
        mirror.unlink()
        self.assertRejected(f"{L.SERIES_DIRS[1] / 'd0.diff'} is missing")

    def test_apply_writes_the_effective_schema(self) -> None:
        out = self.root / "build" / L.SCHEMA_NAME
        self.assertEqual(self.run_tool("apply", "--out", str(out)), 0, self.err.getvalue())
        self.assertEqual(out.read_text(encoding="utf-8"), self.effective())
        self.assertIn("  tag @2 :Text;\n", out.read_text(encoding="utf-8"))
        self.assertNotIn("tag @2", self.vendored())
        before = out.stat().st_mtime_ns
        self.assertEqual(self.run_tool("apply", "--out", str(out)), 0)
        self.assertEqual(out.stat().st_mtime_ns, before)

    def test_local_field_lands_through_a_second_diff(self) -> None:
        # The feat/uma-cache-dir shape: a new local field is a new diff at the
        # end of the series plus one appended local ledger line; the vendored
        # copies stay the pin.
        cache = textwrap.dedent(
            """\
            diff --git a/Potentials.capnp b/Potentials.capnp
            --- a/Potentials.capnp
            +++ b/Potentials.capnp
            @@ -11,4 +11,5 @@
             struct MorseParams {
               de @0 :Float64 = 0.7102;
               re @1 :Float64 = 2.897;
            +  cacheDir @2 :Text;
             }
            """
        )
        self.write_tree(UPSTREAM, LOCAL_DIFF, cache)
        self.assertRejected("d1.diff adds MorseParams @2 field cacheDir :Text, not a local ledger entry")
        self.assertEqual(self.run_tool("append", "--upstream", str(self.upstream)), 0)
        self.assertTrue(
            self.ledger().read_text(encoding="utf-8").endswith("MorseParams @2 field cacheDir :Text local\n")
        )
        self.assertEqual(self.check(), 0, self.err.getvalue())
        self.assertEqual(self.vendored(), UPSTREAM)

    def test_new_upstream_entry_needs_append(self) -> None:
        grown = UPSTREAM.replace(
            "  re @1 :Float64 = 2.897;\n", "  re @1 :Float64 = 2.897;\n  a @2 :Float64;\n"
        )
        self.write_tree(grown, LOCAL_DIFF)
        self.assertRejected("pinned upstream declares MorseParams @2 field a :Float64, not in the ledger")
        self.assertEqual(self.run_tool("append", "--upstream", str(self.upstream)), 0)
        self.assertEqual(self.check(), 0, self.err.getvalue())
        self.assertTrue(
            self.ledger().read_text(encoding="utf-8").endswith("MorseParams @2 field a :Float64 upstream\n")
        )

    def test_upstream_collides_with_a_local_entry(self) -> None:
        taken = UPSTREAM.replace(
            '  lengthUnit @1 :Text = "angstrom";  # @brief ignored @7 in a comment\n',
            '  lengthUnit @1 :Text = "angstrom";  # @brief ignored @7 in a comment\n  charge @2 :Int32;\n',
        )
        self.upstream.write_text(taken, encoding="utf-8")
        self.assertRejected(
            "local ForceInput @2 field tag :Text collides with upstream ForceInput @2 field charge :Int32"
        )

    def test_crlf_checkout_passes(self) -> None:
        # git with core.autocrlf=true checks every text file out with CRLF.
        files = [
            *(self.root / rel for rel in L.VENDORED),
            self.ledger(),
            self.root / L.WRAP,
            self.upstream,
            *(p for d in L.SERIES_DIRS for p in sorted((self.root / d).iterdir())),
        ]
        for path in files:
            path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
        self.assertIn(b"\r\n", (self.root / L.VENDORED[0]).read_bytes())
        self.assertEqual(self.check(), 0, self.err.getvalue())

        patched = self.root / "checkout" / L.SCHEMA_NAME
        patched.parent.mkdir()
        patched.write_bytes((self.root / L.VENDORED[0]).read_bytes())
        self.assertEqual(
            self.run_tool("check", "--subproject-schema", str(patched)), 0, self.err.getvalue()
        )

    @unittest.skipIf(shutil.which("git") is None, "git is not installed")
    def test_subproject_at_the_pin_is_read_as_utf8(self) -> None:
        # meson's wrap-git checkout sits at the pinned revision, so the
        # upstream text comes from git. A locale code page that is not UTF-8
        # (cp1252 on Windows) must not change a non-ASCII comment in it.
        upstream = UPSTREAM + "# Pinned upstream \u2014 edit it there.\n"
        self.write_tree(upstream, LOCAL_DIFF)
        sub = self.root / "subprojects" / "potentials-schema"
        sub.mkdir(parents=True)
        (sub / L.SCHEMA_NAME).write_text(upstream, encoding="utf-8")
        _git(sub, "init", "-q")
        _git(sub, "add", "-A")
        _git(sub, "commit", "-q", "-m", "pin")
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=sub, check=True, stdout=subprocess.PIPE, encoding="utf-8"
        ).stdout.strip()
        wrap = self.root / L.WRAP
        wrap.write_text(wrap.read_text(encoding="utf-8").replace("0" * 40, head), encoding="utf-8")
        # A parent may patch its own checkout; the pinned git object still wins.
        (sub / L.SCHEMA_NAME).write_text(self.effective(), encoding="utf-8")

        env = {**os.environ, "PYTHONUTF8": "0", "PYTHONCOERCECLOCALE": "0", "LC_ALL": "C"}
        done = subprocess.run(
            [
                sys.executable,
                str(ROOT / "CppCore" / "rgpot" / "rpc" / "schema_ledger.py"),
                "check",
                "--root",
                str(self.root),
                "--subproject-schema",
                str(sub / L.SCHEMA_NAME),
            ],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            errors="replace",
        )
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("pinned upstream: git object", done.stdout)

    def test_append_refuses_to_rewrite(self) -> None:
        self.set_vendored(self.vendored().replace("de @0 :Float64", "de @0 :Float32"))
        self.assertEqual(self.run_tool("append", "--upstream", str(self.upstream)), 1)
        self.assertIn("append will not rewrite it", self.err.getvalue())

    @unittest.skipIf(shutil.which("git") is None, "git is not installed")
    def test_ledger_is_append_only_against_base(self) -> None:
        _git(self.root, "init", "-q")
        _git(self.root, "add", "-A")
        _git(self.root, "commit", "-q", "-m", "base")
        original = self.ledger().read_text(encoding="utf-8")

        self.ledger().write_text(original + "MorseParams @9 field spare :Bool local\n", encoding="utf-8")
        self.assertEqual(self.run_tool("check", "--base", "HEAD"), 1)
        self.assertNotIn("ledger entry", self.err.getvalue())

        changed = original.replace(
            "PotentialConfig @2 arm xtb :XTBParams upstream",
            "PotentialConfig @2 arm tblite :XTBParams upstream",
        )
        self.ledger().write_text(changed, encoding="utf-8")
        self.assertEqual(self.run_tool("check", "--base", "HEAD"), 1)
        self.assertIn("ledger entry changed: PotentialConfig @2 arm xtb", self.err.getvalue())

        dropped = original.replace("Kind @1 enumerant second - upstream\n", "")
        self.ledger().write_text(dropped, encoding="utf-8")
        self.assertEqual(self.run_tool("check", "--base", "HEAD"), 1)
        self.assertIn("ledger entry disappeared: Kind @1 enumerant second - upstream", self.err.getvalue())

        lines = original.splitlines(keepends=True)
        body = [ln for ln in lines if not ln.startswith("#")]
        moved = [ln for ln in lines if ln.startswith("#")] + body[1:] + body[:1]
        self.ledger().write_text("".join(moved), encoding="utf-8")
        self.assertEqual(self.run_tool("check", "--base", "HEAD"), 1)
        self.assertIn("ledger entry moved", self.err.getvalue())

        inserted = (
            [ln for ln in lines if ln.startswith("#")] + ["Kind @2 enumerant third - upstream\n"] + body
        )
        self.ledger().write_text("".join(inserted), encoding="utf-8")
        self.assertEqual(self.run_tool("check", "--base", "HEAD"), 1)
        self.assertIn(
            "ledger entry inserted before existing ones: Kind @2 enumerant third", self.err.getvalue()
        )
        self.assertNotIn("ledger entry moved", self.err.getvalue())

        self.ledger().write_text(original, encoding="utf-8")
        self.assertEqual(
            self.run_tool("check", "--base", "HEAD", "--upstream", str(self.upstream)), 0, self.err.getvalue()
        )


def _pinned_upstream(tmp: Path) -> Path:
    """The pinned Potentials.capnp: meson's checkout of the wrap, else a fetch."""
    wrap = L.read_wrap(ROOT)
    parser = configparser.ConfigParser()
    parser.read(ROOT / L.WRAP, encoding="utf-8")
    checkout = ROOT / "subprojects" / parser["wrap-git"]["directory"] / L.SCHEMA_NAME
    if checkout.is_file():
        text, _ = L.upstream_from_subproject(checkout, wrap)
    else:
        try:
            text = L.fetch_upstream(wrap)
        except (OSError, subprocess.CalledProcessError) as exc:
            if os.environ.get("CI"):
                raise
            raise unittest.SkipTest(f"cannot fetch {wrap.url} at {wrap.revision}: {exc}") from exc
    path = tmp / L.SCHEMA_NAME
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


class RepositoryTest(unittest.TestCase):
    def test_vendored_copies_are_the_pinned_release(self) -> None:
        # The comparison a parent project (gpridge) runs as
        # schema-sync-canonical-rpc and schema-sync-canonical-core.
        sync = ROOT / "CppCore" / "rgpot" / "rpc" / "schema_sync_check.py"
        with tempfile.TemporaryDirectory() as tmp:
            pinned = _pinned_upstream(Path(tmp))
            for rel in L.VENDORED:
                done = subprocess.run(
                    [sys.executable, str(sync), str(ROOT / rel), str(pinned)],
                    capture_output=True,
                    check=False,
                    encoding="utf-8",
                )
                self.assertEqual(done.returncode, 0, f"{rel}: {done.stderr.strip()}")

    def test_vendored_copies_match_the_ledger(self) -> None:
        err = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
            code = L.main(["check", "--root", str(ROOT)])
        self.assertEqual(code, 0, err.getvalue())

    def test_upstream_owns_potential_config_six_and_ten(self) -> None:
        ledger = L.parse_ledger((ROOT / L.LEDGER).read_text(encoding="utf-8"), "ledger")
        arms = {e.ordinal: (e.name, e.type, e.origin) for e in ledger if e.scope == "PotentialConfig"}
        self.assertEqual(arms[6], ("xtb", ":XTBParams", "upstream"))
        self.assertEqual(arms[10], ("morse", ":MorseParams", "upstream"))

    def test_parser_reads_every_ordinal_line(self) -> None:
        # Each line that declares an ordinal outside a comment must be an entry.
        for rel in L.VENDORED:
            text = (ROOT / rel).read_text(encoding="utf-8")
            entries = {line for _, line in L.parse_schema(text)}
            for number, raw in enumerate(text.splitlines(), 1):
                code = L._strip_comment(raw)
                if L.ORDINAL.search(code) and not code.lstrip().startswith("@0x"):
                    self.assertIn(number, entries, f"{rel}:{number}: {raw.strip()}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
