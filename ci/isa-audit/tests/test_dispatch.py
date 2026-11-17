"""Dispatch proofs against objects built by the installed compiler."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from elftools.elf.elffile import ELFFile  # noqa: E402
from isa_audit import AuditStats, _elf_funcs, _finish_ranges, audit_path, load_allowlist  # noqa: E402

try:
    import dispatch
    from plant import plant
except ImportError:  # aarch64 runners have no iced-x86 wheel
    dispatch = None  # type: ignore[assignment]

# libgcc's __builtin_cpu_supports spelling of each CPU_MODEL_FEATURES index.
BUILTIN_NAMES = {
    0: "cmov",
    1: "mmx",
    2: "popcnt",
    3: "sse",
    4: "sse2",
    5: "sse3",
    6: "ssse3",
    7: "sse4.1",
    8: "sse4.2",
    9: "avx",
    10: "avx2",
    11: "sse4a",
    12: "fma4",
    13: "xop",
    14: "fma",
    15: "avx512f",
    16: "bmi",
    17: "bmi2",
    18: "aes",
    19: "pclmul",
    20: "avx512vl",
    21: "avx512bw",
    22: "avx512dq",
    23: "avx512cd",
    24: "avx512er",
    25: "avx512pf",
    26: "avx512vbmi",
    27: "avx512ifma",
    28: "avx5124vnniw",
    29: "avx5124fmaps",
    30: "avx512vpopcntdq",
    31: "avx512vbmi2",
    32: "gfni",
    33: "vpclmulqdq",
    34: "avx512vnni",
    35: "avx512bitalg",
    36: "avx512bf16",
    49: "f16c",
    57: "lzcnt",
    58: "movbe",
}

# GCC -m option and predefined macro for each iced name in IMPLIES.
GCC_OPTIONS = {
    "sse3": ("sse3", "SSE3"),
    "ssse3": ("ssse3", "SSSE3"),
    "sse4_1": ("sse4.1", "SSE4_1"),
    "sse4_2": ("sse4.2", "SSE4_2"),
    "sse4a": ("sse4a", "SSE4A"),
    "popcnt": ("popcnt", "POPCNT"),
    "xsave": ("xsave", "XSAVE"),
    "avx": ("avx", "AVX"),
    "avx2": ("avx2", "AVX2"),
    "fma": ("fma", "FMA"),
    "f16c": ("f16c", "F16C"),
    "fma4": ("fma4", "FMA4"),
    "xop": ("xop", "XOP"),
    "avx512f": ("avx512f", "AVX512F"),
    "avx512vl": ("avx512vl", "AVX512VL"),
    "avx512bw": ("avx512bw", "AVX512BW"),
    "avx512dq": ("avx512dq", "AVX512DQ"),
    "avx512cd": ("avx512cd", "AVX512CD"),
    "avx512er": ("avx512er", "AVX512ER"),
    "avx512pf": ("avx512pf", "AVX512PF"),
    "avx512_ifma": ("avx512ifma", "AVX512IFMA"),
    "avx512_vpopcntdq": ("avx512vpopcntdq", "AVX512VPOPCNTDQ"),
    "avx512_vbmi": ("avx512vbmi", "AVX512VBMI"),
    "avx512_vbmi2": ("avx512vbmi2", "AVX512VBMI2"),
    "avx512_vnni": ("avx512vnni", "AVX512VNNI"),
    "avx512_bitalg": ("avx512bitalg", "AVX512BITALG"),
    "avx512_bf16": ("avx512bf16", "AVX512BF16"),
    "avx512_4vnniw": ("avx5124vnniw", "AVX5124VNNIW"),
    "avx512_4fmaps": ("avx5124fmaps", "AVX5124FMAPS"),
}


def _gcc() -> str | None:
    cc = os.environ.get("CC_X86", "gcc")
    path = shutil.which(cc)
    if path is None:
        return None
    try:
        version = subprocess.run([path, "--version"], capture_output=True, text=True, check=True).stdout
        machine = subprocess.run([path, "-dumpmachine"], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    if "Free Software Foundation" not in version or not machine.startswith("x86_64"):
        return None
    return path


GCC = _gcc()


def _proofs(path: Path) -> dict[str, frozenset[str]]:
    """Proven features of every function in a linked object, by raw name."""
    with path.open("rb") as handle:
        elf = ELFFile(handle)
        sections = []
        for index, section in enumerate(elf.iter_sections()):
            data = section.data() if section["sh_flags"] & 0x4 else b""
            if data:
                base = section["sh_addr"]
                funcs = _finish_ranges(_elf_funcs(elf, index, base, False), base + len(data))
                sections.append((base, data, funcs))
        wanted = {func.start for _base, _data, funcs in sections for func in funcs}
        return {proof.name: proof.features for proof in dispatch.prove(elf, sections, wanted).values()}


def _audit(path: Path, allow: str = "allow-dispatch.toml") -> AuditStats:
    stats = AuditStats()
    audit_path(path, stats, load_allowlist(HERE / allow), "x86-64", True)
    return stats


@unittest.skipIf(dispatch is None, "iced-x86 is not installed")
@unittest.skipIf(GCC is None, "no x86_64 GCC")
class DispatchTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="isa-dispatch-"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _build(self, source: Path | str, name: str, *flags: str) -> Path | None:
        if isinstance(source, str):
            src = self.tmp / f"{name}.c"
            src.write_text(source)
            source = src
        out = self.tmp / f"{name}.so"
        cmd = [GCC, "-shared", "-fPIC", "-O2", "-march=x86-64", *flags, str(source), "-o", str(out)]
        done = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if done.returncode != 0:
            return None
        return out

    def _fixture(self, source: str, name: str, *flags: str) -> Path | None:
        # libgcc.a keeps its toolchain's -march, so the fixtures import the CPU model.
        return self._build(HERE / source, name, "-nodefaultlibs", *flags)

    def test_bits_match_libgcc(self) -> None:
        checked = 0
        for index, builtin in BUILTIN_NAMES.items():
            source = (
                "__attribute__((noinline)) static void probe(void) { __asm__ volatile(\"vzeroupper\"); }\n"
                f"void entry(void) {{ if (__builtin_cpu_supports(\"{builtin}\")) probe(); }}\n"
            )
            lib = self._build(source, f"bit{index}")
            if lib is None:
                continue
            expected = dispatch.CPU_MODEL_FEATURES[index]
            with self.subTest(builtin=builtin):
                self.assertEqual(_proofs(lib).get("probe"), dispatch.closure({expected}))
            checked += 1
        self.assertGreater(checked, 30)

    def test_implications_match_gcc(self) -> None:
        macros = {macro: name for name, (_option, macro) in GCC_OPTIONS.items()}
        for name in sorted(dispatch.IMPLIES):
            option, _macro = GCC_OPTIONS[name]
            done = subprocess.run(
                [GCC, "-march=x86-64", f"-m{option}", "-dM", "-E", "-x", "c", "/dev/null"],
                capture_output=True,
                text=True,
                check=False,
            )
            if done.returncode != 0:
                continue
            defined = set()
            for line in done.stdout.splitlines():
                parts = line.split()
                if len(parts) >= 2 and parts[1].startswith("__") and parts[1].endswith("__"):
                    macro = parts[1][2:-2]
                    if macro in macros:
                        defined.add(macros[macro])
            implied = set()
            todo = [name]
            while todo:
                for item in dispatch.IMPLIES.get(todo.pop(), ()):
                    if item not in implied:
                        implied.add(item)
                        todo.append(item)
            with self.subTest(option=option):
                self.assertEqual(implied | {name}, defined)

    def test_fixture_clones_are_proven(self) -> None:
        # At -O0 have_avx2_fma is a call and the resolver branches with jle.
        for level in ("-O2", "-O0"):
            lib = self._fixture("dispatch_ok.c", f"ok{level}", level)
            self.assertIsNotNone(lib)
            stats = _audit(lib)
            names = {item["name"]: item["via"] for item in stats.dispatched}
            with self.subTest(level=level):
                self.assertEqual(stats.violations, [])
                self.assertEqual(names.get("clone_avx2"), ["dispatch_entry"])
                self.assertEqual(names.get("clone_tail"), ["clone_avx2"])
                self.assertEqual(names.get("probed_avx2"), ["probed_entry"])
                self.assertIn("tc_scale.avx2", names)
                self.assertIn("tc_scale.avx512f", names)

    def test_lookalikes_are_violations(self) -> None:
        for level in ("-O2", "-O0"):
            lib = self._fixture("dispatch_bad.c", f"bad{level}", level, "-fno-ipa-icf")
            self.assertIsNotNone(lib)
            stats = _audit(lib)
            flagged = {item.symbol for item in stats.violations}
            with self.subTest(level=level):
                self.assertEqual(stats.dispatched, [])
                for name in (
                    "unguarded_avx2",
                    "wrong_guard_avx2",
                    "inverted_guard_avx2",
                    "escaped_avx2",
                    "partial_guard_avx2",
                    "exported_avx2",
                    "lying_probe_avx2",
                    "weak_probe_avx2",
                ):
                    self.assertIn(name, flagged)

    def test_planted_avx_outside_a_clone_fails(self) -> None:
        lib = self._fixture("dispatch_ok.c", "plant")
        self.assertIsNotNone(lib)
        blob = lib.read_bytes()
        for symbol, dispatched in (
            ("clone_scalar", False),
            ("dispatch_entry", False),
            ("tc_scale.default", False),
            ("clone_avx2", True),
        ):
            patched, address = plant(blob, symbol)
            out = self.tmp / f"planted-{symbol}.so"
            out.write_bytes(patched)
            stats = _audit(out)
            hits = [
                item
                for item in stats.violations
                if item.symbol == symbol and item.address == f"0x{address:x}" and item.mnemonic == "vzeroupper"
            ]
            with self.subTest(symbol=symbol):
                if dispatched:
                    self.assertEqual(stats.violations, [])
                else:
                    self.assertEqual(len(hits), 1, stats.violations)
                    self.assertEqual(hits[0].feature, "avx")

    def test_stripped_library_proves_nothing(self) -> None:
        lib = self._fixture("dispatch_ok.c", "stripped")
        self.assertIsNotNone(lib)
        subprocess.run(["strip", "--strip-all", str(lib)], check=True)
        stats = _audit(lib)
        self.assertEqual(stats.dispatched, [])
        self.assertTrue(stats.violations)


@unittest.skipIf(dispatch is None, "iced-x86 is not installed")
class ClosureTest(unittest.TestCase):
    def test_avx2_closure(self) -> None:
        self.assertEqual(
            dispatch.closure({"avx2"}),
            {"avx2", "avx", "sse4_2", "sse4_1", "ssse3", "sse3", "popcnt", "xsave"},
        )

    def test_avx512f_has_the_vex_forms(self) -> None:
        self.assertTrue({"fma", "f16c", "avx2"} <= dispatch.closure({"avx512f"}))


if __name__ == "__main__":
    unittest.main()
