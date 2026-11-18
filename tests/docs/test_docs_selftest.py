"""The documentation checks fail on planted mistakes.

Each test writes a small page with one wrong statement or one broken example
and asserts that doccheck reports it.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "docs"))

import doccheck  # noqa: E402


def _page(text: str, name: str = "page.org") -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="doccheck-"))
    page = tmp / name
    page.write_text(textwrap.dedent(text).lstrip(), encoding="utf-8")
    return page


def _entry(name: str, signature: str, lang: str, extra: str = "") -> str:
    return f"""
    * Page

    *** ={name}=

    #+attr_rgpot: signature
    #+begin_src {lang}
    {signature}
    #+end_src

    {extra}- Inputs :: x.
    - Units :: none.
    - Outputs :: x.
    - Errors :: none.
    """


def _about(errors: list[str], name: str) -> list[str]:
    return [e for e in errors if name in e and not e.startswith("no reference entry")]


class PlantedReferenceMistakes(unittest.TestCase):
    def test_correct_c_entry_passes(self):
        page = _page(_entry("rgpot_abi_stamp", "rgpot_abi_stamp_t rgpot_abi_stamp(void);", "c"))
        self.assertEqual(_about(doccheck.reference_errors(ROOT, [page]), "rgpot_abi_stamp"), [])

    def test_wrong_c_return_type(self):
        page = _page(_entry("rgpot_abi_stamp", "int32_t rgpot_abi_stamp(void);", "c"))
        errors = _about(doccheck.reference_errors(ROOT, [page]), "rgpot_abi_stamp")
        self.assertTrue(any("differs from" in e for e in errors), errors)

    def test_wrong_python_default(self):
        name = next(
            n for n, v in doccheck.py_public_names(ROOT).items()
            if n.endswith(".hvp") and v.params
        )
        params = doccheck.py_public_names(ROOT)[name].params
        wrong = ", ".join(p.name for p in params) + ", bogus=1"
        page = _page(_entry(name, f"hvp({wrong})", "python"))
        errors = _about(doccheck.reference_errors(ROOT, [page]), name)
        self.assertTrue(any("but the source has" in e for e in errors), errors)

    def test_member_claimed_on_a_class_without_it(self):
        names = doccheck.py_public_names(ROOT)
        classes = {n.split(".")[1] for n, v in names.items() if v.kind == "class"}
        lacking = sorted(c for c in classes if f"rgpot.{c}.hvp" not in names)
        self.assertTrue(lacking, "every class has hvp; pick another member")
        page = _page(
            f"""
            * Page

            *** =.hvp=

            - Applies to :: rgpot.{lacking[0]}
            - Inputs :: x.
            - Units :: none.
            - Outputs :: x.
            - Errors :: none.
            """
        )
        errors = doccheck.reference_errors(ROOT, [page])
        self.assertIn(f"rgpot.{lacking[0]} has no member hvp", "\n".join(errors))

    def test_missing_field(self):
        page = _page(
            """
            * Page

            *** =rgpot_abi_stamp_t=

            - Inputs :: none.
            - Outputs :: the stamp.
            - Errors :: TODO.
            """
        )
        errors = _about(doccheck.reference_errors(ROOT, [page]), "rgpot_abi_stamp_t")
        text = "\n".join(errors)
        self.assertIn("missing Units", text)
        self.assertIn("Errors is a placeholder", text)

    def test_name_that_is_not_public(self):
        page = _page(_entry("rgpot_no_such_function", "void rgpot_no_such_function(void);", "c"))
        errors = doccheck.reference_errors(ROOT, [page])
        self.assertIn("rgpot_no_such_function is not a public name", "\n".join(errors))

    def test_wrong_libcp2k_signature(self):
        page = _page(_entry("cp2k_get_natom", "void cp2k_get_natom(force_env_t force_env, long *natom);", "c"))
        errors = _about(doccheck.reference_errors(ROOT, [page]), "cp2k_get_natom")
        self.assertTrue(any("differs from" in e for e in errors), errors)

    def test_config_entry_without_a_field(self):
        header, fields = doccheck.cpp_config_fields(ROOT)["rgpot::XTBConfig"]
        self.assertIn("warm_min_gap", fields, header)
        listed = ", ".join(f"={f}=" for f in fields if f != "warm_min_gap")
        page = _page(
            f"""
            * Page

            *** =rgpot::XTBConfig=

            - Inputs :: {listed}.
            - Units :: none.
            - Outputs :: a plain struct.
            - Errors :: none.
            """
        )
        errors = _about(doccheck.reference_errors(ROOT, [page]), "rgpot::XTBConfig")
        self.assertEqual(
            errors,
            [f"{page}:3: rgpot::XTBConfig: field warm_min_gap of {header} is not described"],
        )

    def test_local_schema_field_that_no_page_names(self):
        members = [m for m, _ in doccheck.local_schema_fields(ROOT)]
        self.assertIn("UmaParams.cacheDir", members)
        named = "\n".join(f"| ={m}= | x |" for m in members if m != "UmaParams.cacheDir")
        errors = doccheck.local_schema_errors(ROOT, [_page(f"* Page\n\n{named}\n")])
        self.assertEqual(len(errors), 1, errors)
        self.assertIn("adds UmaParams.cacheDir, which no reference page names", errors[0])


class PlantedFactMistakes(unittest.TestCase):
    def _facts(self, text: str) -> str:
        return "\n".join(doccheck.facts_errors(ROOT, [_page(text)]))

    def test_unknown_meson_option(self):
        self.assertIn("-Dwith_nonsense is not a meson option", self._facts("Pass =-Dwith_nonsense=true=.\n"))

    def test_bad_meson_value(self):
        self.assertIn("-Dwith_rpc=maybe is not a valid value", self._facts("Pass =-Dwith_rpc=maybe=.\n"))

    def test_unknown_cmake_option(self):
        self.assertIn("-DRGPOT_WITH_NONSENSE is not a CMake option", self._facts("Pass =-DRGPOT_WITH_NONSENSE=ON=.\n"))

    def test_missing_path(self):
        self.assertIn("path CppCore/rgpot/NoSuch.hpp does not exist", self._facts("See =CppCore/rgpot/NoSuch.hpp=.\n"))

    def test_unknown_pixi_task(self):
        self.assertIn("pixi task no-such-task does not exist", self._facts("Run =pixi r no-such-task=.\n"))

    def test_foreign_tokens_are_declared_per_page(self):
        text = "#+rgpot_foreign: -Dwith_rgpot\nIn eOn, pass =-Dwith_rgpot=true=.\n"
        self.assertEqual(self._facts(text), "")

    def test_build_outputs_count_as_paths(self):
        self.assertEqual(self._facts("Open =docs/build/index.html=.\n"), "")

    def test_build_outputs_count_as_paths_without_a_repository(self):
        # A tree from git archive has the .gitignore and no .git.
        with tempfile.TemporaryDirectory(prefix="doccheck-tree-") as tmp:
            shutil.copy(ROOT / ".gitignore", tmp)
            for token in ("docs/build/index.html", "docs/source/crates/rgpot_core",
                          "subprojects/xxHash-0.8.3/xxhash.h"):
                self.assertTrue(doccheck._git_ignored(tmp, token), token)
            for token in ("docs/source/conf.py", "subprojects/packagefiles/xxhash.diff",
                          "CppCore/rgpot/Potential.hpp"):
                self.assertFalse(doccheck._git_ignored(tmp, token), token)

    def test_gitignore_rules_as_git_reads_them(self):
        rules = "build/\n/out\n*.log\n!keep.log\nsub/*\n!sub/kept\ndocs/**/gen\n"
        expected = {
            "build/x": True, "a/build/x": True, "build": False,
            "out": True, "a/out": False,
            "x.log": True, "a/x.log": True, "keep.log": False, "a/keep.log": False,
            "sub/a": True, "sub/kept": False, "sub/kept/y": False,
            "logs.log/y": True, "build/keep.log": True,
            "docs/gen": True, "docs/a/b/gen": True, "a/docs/gen": False,
        }
        with tempfile.TemporaryDirectory(prefix="doccheck-tree-") as tmp:
            (Path(tmp) / ".gitignore").write_text(rules, encoding="utf-8")
            got = {token: doccheck._git_ignored(tmp, token) for token in expected}
        self.assertEqual(got, expected)

    def test_rgpot_keys_against_the_drop_in_parser(self):
        page = (ROOT / doccheck.GROMACS_DROPIN_PAGE).read_text(encoding="utf-8")
        self.assertEqual(doccheck.gromacs_key_errors(ROOT), [])
        planted = "\n".join(
            "| =THREADS= n | threads |" if line.startswith("| =EMBEDDING=") else line
            for line in page.splitlines()
        )
        self.assertNotEqual(page, planted)
        errors = "\n".join(doccheck.gromacs_key_errors(ROOT, _page(planted)))
        self.assertIn("&RGPOT key EMBEDDING is not in the table", errors)
        self.assertIn("THREADS is not a key", errors)

    def test_engine_table_exports_against_the_adapter(self):
        page = (ROOT / doccheck.ENGINE_ABI_PAGE).read_text(encoding="utf-8")
        self.assertEqual(doccheck.engine_table_export_errors(ROOT), [])
        planted = page.replace("=rgpot_engine_hvp=, ", "", 1)
        self.assertNotEqual(page, planted)
        errors = "\n".join(doccheck.engine_table_export_errors(ROOT, _page(planted)))
        self.assertIn("export rgpot_engine_hvp is not in the library export list", errors)

    def test_wrong_default_in_build_options_page(self):
        options = doccheck.meson_options(ROOT)
        cmake = doccheck.cmake_options(ROOT)
        page = (ROOT / doccheck.BUILD_OPTIONS_PAGE).read_text(encoding="utf-8")
        planted = page.replace("| =with_rpc= | boolean | =false= |", "| =with_rpc= | boolean | =true= |")
        self.assertNotEqual(page, planted)
        tmp = Path(tempfile.mkdtemp(prefix="doccheck-"))
        target = tmp / doccheck.BUILD_OPTIONS_PAGE
        target.parent.mkdir(parents=True)
        target.write_text(planted, encoding="utf-8")
        errors = doccheck.build_option_table_errors(tmp, options, cmake)
        self.assertIn("with_rpc defaults to false, not true", "\n".join(errors))


class PlantedBrokenExamples(unittest.TestCase):
    def _blocks(self, text: str, name: str = "page.org"):
        page = _page(text, name)
        return doccheck.page_blocks(page)

    def _static(self, text: str, name: str = "page.org") -> str:
        return "\n".join(doccheck.check_blocks_static(ROOT, self._blocks(text, name)))

    def test_unclassified_block(self):
        text = "#+begin_src python\nimport rgpot\n#+end_src\n"
        self.assertIn("missing classification", self._static(text))

    def test_unowned_page_block_still_gets_the_name_check(self):
        page = _page("#+begin_src python\nimport rgpot\nrgpot.NoSuchPot()\n#+end_src\n")
        with mock.patch.object(doccheck, "UNOWNED_PAGES", {doccheck.rel(page)}):
            report = "\n".join(doccheck.check_blocks_static(ROOT, doccheck.page_blocks(page)))
        self.assertNotIn("missing classification", report)
        self.assertIn("rgpot has no public name NoSuchPot", report)

    def test_python_name_that_does_not_exist(self):
        text = "#+attr_rgpot: fragment\n#+begin_src python\nimport rgpot\nrgpot.NoSuchPot()\n#+end_src\n"
        self.assertIn("NoSuchPot", self._static(text))

    def test_c_function_that_does_not_exist(self):
        text = "#+attr_rgpot: fragment\n#+begin_src c\nrgpot_no_such_call(0);\n#+end_src\n"
        self.assertIn("rgpot_no_such_call", self._static(text))

    def test_bash_syntax_error(self):
        text = "#+attr_rgpot: syntax\n#+begin_src bash\nif then fi (\n#+end_src\n"
        self.assertNotEqual(self._static(text), "")

    def test_markdown_classification(self):
        text = "<!-- rgpot-doc: fragment -->\n```python\nimport rgpot\nrgpot.NoSuchPot()\n```\n"
        self.assertIn("NoSuchPot", self._static(text, "page.md"))

    def test_skip_needs_a_reason(self):
        text = "#+attr_rgpot: skip\n#+begin_src text\nx\n#+end_src\n"
        self.assertIn("skip needs a reason", self._static(text))

    def test_cpp_block_skips_without_its_feature(self):
        info = doccheck.BuildInfo(ROOT, ["-DRGPOT_HAS_CACHE=TRUE"], [], [], False, {})
        text = "#+attr_rgpot: run requires=has_expr link=rgpot\n#+begin_src cpp\nint main() { return 1; }\n#+end_src\n"
        [block] = self._blocks(text)
        self.assertEqual(doccheck.run_block(block, info, ROOT), ("skip", "build lacks has_expr"))

    def test_cpp_flags_come_from_a_test(self):
        with tempfile.TemporaryDirectory(prefix="doccheck-build-") as tmp:
            build = Path(tmp)
            commands = [
                {"directory": str(build), "file": "../CppCore/rgpot/CalculatorGroupMpi.cc",
                 "arguments": ["c++", "-DNDEBUG", "-c", "../CppCore/rgpot/CalculatorGroupMpi.cc"]},
                {"directory": str(build), "file": "../CppCore/tests/CxxLayoutTest.cc",
                 "arguments": ["c++", "-DRGPOT_HAS_CACHE=TRUE", "-c", "../CppCore/tests/CxxLayoutTest.cc"]},
            ]
            (build / "compile_commands.json").write_text(json.dumps(commands), encoding="utf-8")
            info = doccheck.build_info(build, ROOT)
        self.assertIn("-DRGPOT_HAS_CACHE=TRUE", info.cpp_flags)

    def test_link_dirs_from_elf_and_mach_o_libraries(self):
        with tempfile.TemporaryDirectory(prefix="doccheck-build-") as tmp:
            build = Path(tmp).resolve()
            for lib in ("elf/librgpot.so.0", "macho/librgpot.dylib"):
                (build / lib).parent.mkdir()
                (build / lib).touch()
            info = doccheck.build_info(build, ROOT)
        self.assertEqual(info.lib_dirs, [str(build / "elf"), str(build / "macho")])

    def test_link_line_of_the_plainest_program_in_the_block_language(self):
        def exe(name, lang, args):
            return {"name": name, "type": "executable", "filename": [f"/build/{name}"],
                    "target_sources": [{"language": lang}, {"linker": ["c++"], "parameters": args}]}

        targets = [
            {"name": "rgpot", "type": "static library", "filename": ["/build/CppCore/librgpot.a"],
             "target_sources": []},
            exe("capi_threads_tsan", "c", ["-fsanitize=thread", "CppCore/librgpot.a", "-ldl"]),
            exe("capi_threads", "c", ["CppCore/librgpot.a", "-ldl"]),
            exe("rgpot-eval", "cpp", ["CppCore/librgpot.a"]),
            exe("plain_c", "c", ["-ldl"]),
        ]
        info = doccheck.BuildInfo(Path("/build"), [], [], [], False, {}, targets)
        self.assertEqual(doccheck._meson_link(info, ["rgpot"], "c"), (["c++"], ["CppCore/librgpot.a", "-ldl"]))
        self.assertEqual(doccheck._meson_link(info, ["rgpot"], "cpp"), (["c++"], ["CppCore/librgpot.a"]))
        self.assertIsNone(doccheck._meson_link(info, ["m"], "c"))
        self.assertIsNone(doccheck._meson_link(info, [], "c"))

    @unittest.skipUnless(
        all(shutil.which(os.environ.get(var, tool)) for var, tool in (("CC", "cc"), ("CXX", "c++"), ("AR", "ar"))),
        "needs cc, c++ and ar",
    )
    def test_c_block_links_a_static_cxx_library_as_meson_does(self):
        cxx = shutil.which(os.environ.get("CXX", "c++"))
        ar = shutil.which(os.environ.get("AR", "ar"))
        text = (
            "#+attr_rgpot: run link=cxxlib\n#+begin_src c\nint cxxlib_answer(void);\n"
            "int main(void) { return cxxlib_answer() == 42 ? 0 : 1; }\n#+end_src\n"
        )
        [block] = self._blocks(text)
        with tempfile.TemporaryDirectory(prefix="doccheck-build-") as tmp:
            build = Path(tmp).resolve()
            # libdep.a stands in for rocksdb, and new and throw for the C++
            # runtime: the block names neither, so only meson's line links them.
            (build / "dep.cc").write_text('extern "C" int dep_base() { return 40; }\n', encoding="utf-8")
            (build / "cxxlib.cc").write_text(textwrap.dedent("""\
                extern "C" int dep_base();
                extern "C" int cxxlib_answer() {
                    int *base = new int(dep_base());
                    try {
                        throw *base + 2;
                    } catch (int answer) {
                        delete base;
                        return answer;
                    }
                }
            """), encoding="utf-8")
            for name in ("dep", "cxxlib"):
                subprocess.run([cxx, "-c", f"{name}.cc", "-o", f"{name}.o"], cwd=build, check=True)
                subprocess.run([ar, "rcs", f"lib{name}.a", f"{name}.o"], cwd=build, check=True)
            targets = [
                {"name": "cxxlib", "type": "static library", "filename": [str(build / "libcxxlib.a")],
                 "target_sources": [{"language": "cpp"}]},
                {"name": "consumer", "type": "executable", "filename": [str(build / "consumer")],
                 "target_sources": [{"language": "c"},
                                    {"linker": [cxx], "parameters": ["libcxxlib.a", "libdep.a"]}]},
            ]
            (build / "meson-info").mkdir()
            (build / "meson-info" / "intro-targets.json").write_text(json.dumps(targets), encoding="utf-8")
            info = doccheck.build_info(build, ROOT)
            self.assertEqual(doccheck.run_block(block, info, ROOT), ("ok", ""))

    def test_cpp_block_needs_an_rpc_build(self):
        text = "#+attr_rgpot: compile requires=has_rpc\n#+begin_src cpp\nint main() { return 1; }\n#+end_src\n"
        [block] = self._blocks(text)
        with tempfile.TemporaryDirectory(prefix="doccheck-build-") as tmp:
            build = Path(tmp)
            (build / "meson-info").mkdir()
            options = build / "meson-info" / "intro-buildoptions.json"
            info = doccheck.BuildInfo(build, [], [], [], False, {})
            for rpc, client_only, lacks in ((False, False, True), (False, True, False), (True, False, False)):
                with self.subTest(with_rpc=rpc, with_rpc_client_only=client_only):
                    options.write_text(json.dumps([
                        {"name": "with_rpc", "value": rpc},
                        {"name": "with_rpc_client_only", "value": client_only},
                    ]), encoding="utf-8")
                    result = doccheck.run_block(block, info, ROOT)
                    if lacks:
                        self.assertEqual(result, ("skip", "build lacks has_rpc"))
                    else:
                        self.assertIn(result, [("ok", ""), ("skip", "no compiler")])

    @unittest.skipUnless(
        os.environ.get("RGPOT_DOC_BUILD_DIR"), "RGPOT_DOC_BUILD_DIR is not set"
    )
    def test_failing_run_block(self):
        build = Path(os.environ["RGPOT_DOC_BUILD_DIR"])
        text = "#+attr_rgpot: run\n#+begin_src python\nimport rgpot\nassert rgpot.LJPot is None\n#+end_src\n"
        results = list(doccheck.execute_blocks(build, ROOT, self._blocks(text)))
        self.assertEqual([status for _, status, _ in results], ["fail"])


class Coverage(unittest.TestCase):
    def test_tree_is_covered(self):
        self.assertEqual(doccheck.coverage_errors(ROOT), [])

    def test_missing_marker_is_reported(self):
        planted = list(doccheck.COVERAGE_MARKERS) + [
            ("docs/orgmode/reference/gupta.org", "not-a-real-marker")
        ]
        with mock.patch.object(doccheck, "COVERAGE_MARKERS", planted):
            errors = "\n".join(doccheck.coverage_errors(ROOT))
        self.assertIn("not-a-real-marker", errors)


if __name__ == "__main__":
    unittest.main()
