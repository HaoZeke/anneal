"""Static ISA audit for ELF and Mach-O binaries.

Decode every executable section. An instruction outside the baseline
(x86-64 or armv8-a) is a violation unless its demangled function matches
an allow rule whose feature list contains that feature, or the file path
matches a vendored rule that lists the feature, or dispatch.py proves the
function is a dispatched clone that is entered only after a CPU check for
that feature. A file with no function symbols fails closed: release jobs
must audit unstripped artefacts.

x86_64 is decoded with iced-x86. aarch64 and Mach-O are decoded with
capstone (pinned in requirements.txt). Mach-O that this parser cannot
read is reported as not audited and fails the run, so a silent skip
cannot hide an extension.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import io
import json
import os
import re
import struct
import sys
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from capstone import CS_ARCH_ARM64, CS_MODE_LITTLE_ENDIAN, Cs
from elftools.elf.elffile import ELFFile

from demangle import demangle
from dispatch import prove

try:
    from iced_x86 import CpuidFeature, Decoder, Formatter, FormatterSyntax
except ImportError:  # aarch64 runners have no iced-x86 wheel
    CpuidFeature = None  # type: ignore[misc, assignment]
    Decoder = None  # type: ignore[misc, assignment]
    Formatter = None  # type: ignore[misc, assignment]
    FormatterSyntax = None  # type: ignore[misc, assignment]

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]


_ALIASES = {
    "sse4.1": "sse4_1",
    "sse4.2": "sse4_2",
    "cx16": "cmpxchg16b",
    "lahf": "lahf_sahf",
    "lahf-sahf": "lahf_sahf",
    "avx512bf16": "avx512_bf16",
    "avx512vnni": "avx512_vnni",
    "avx512vpopcntdq": "avx512_vpopcntdq",
    "avx512ifma": "avx512_ifma",
    "avx512vbmi": "avx512_vbmi",
    "avx512vbmi2": "avx512_vbmi2",
    "avx512bitalg": "avx512_bitalg",
    "avx512fp16": "avx512_fp16",
    "avx512vp2intersect": "avx512_vp2intersect",
}

# x86-64 v1, using iced-x86 CpuidFeature names in lower case.
# CX16 is CMPXCHG16B. LAHF/SAHF decode as the 8086 class.
# ENDBR is CET_IBT and is a NOP on CPUs without CET.
# Legacy TZCNT (F3 0F BC, REX form F3 48 0F BC) is BMI1 in the opcode
# map and BSF when the F3 prefix is ignored. LLVM emits it for ctz on
# baseline x86-64 (D130956, D132520). It is filtered in the decoder.
# ANDN is a different BMI1 encoding and stays above the baseline.
_X86_64_V1 = {
    "intel8086",
    "intel8086_only",
    "intel186",
    "intel286",
    "intel286_only",
    "intel386",
    "intel386_only",
    "intel386_a0_only",
    "intel486",
    "intel486_a_only",
    "umov",
    "x64",
    "fpu",
    "fpu287",
    "fpu287xl_only",
    "fpu387",
    "fpu387sl_only",
    "cmov",
    "cx8",
    "mmx",
    "sse",
    "sse2",
    "pause",
    "multibytenop",
    "syscall",
    "tsc",
    "rdtscp",
    "cpuid",
    "cmpxchg16b",
    "fxsr",
    "clfsh",
    "cet_ibt",
    "sep",
    "msr",
}
_X86_64_V2 = _X86_64_V1 | {"sse3", "ssse3", "sse4_1", "sse4_2", "popcnt"}

_UNWINDER = re.compile(
    r"(?i)(?:_unwind_|unwind_|uw_|__gcc_personality|__gxx_personality|"
    r"rust_eh_personality|__libunwind|libunwind|eh_personality)"
)

# Hint-space encodings. They execute as NOP on a CPU that lacks the
# extension, the same reason ENDBR is baseline.
_ARM_HINTS = {
    "nop",
    "hint",
    "bti",
    "paciasp",
    "pacibsp",
    "autiasp",
    "autibsp",
    "paciaz",
    "pacibz",
    "autiaz",
    "autibz",
    "xpaclri",
    "pacia1716",
    "pacib1716",
    "autia1716",
    "autib1716",
}

_SVE_MNEMONICS = {
    "cntb",
    "cnth",
    "cntw",
    "cntd",
    "incb",
    "inch",
    "incw",
    "incd",
    "decb",
    "dech",
    "decw",
    "decd",
    "sqincb",
    "sqinch",
    "sqincw",
    "sqincd",
    "uqincb",
    "uqinch",
    "uqincw",
    "uqincd",
    "sqdecb",
    "sqdech",
    "sqdecw",
    "sqdecd",
    "uqdecb",
    "uqdech",
    "uqdecw",
    "uqdecd",
    "whilelo",
    "whilels",
    "whilelt",
    "whilele",
    "whilehi",
    "whilehs",
    "ptrue",
    "pfalse",
    "pnext",
    "pfirst",
    "rdffr",
    "wrffr",
    "setffr",
    "movprfx",
}

# SVE2-only mnemonics. An unlisted SVE2 mnemonic that uses z or p
# registers is reported as sve, which is still outside armv8-a.
_SVE2_MNEMONICS = {
    "adclb",
    "adclt",
    "addhnb",
    "addhnt",
    "bdep",
    "bext",
    "bgrp",
    "bsl1n",
    "bsl2n",
    "eorbt",
    "eortb",
    "histcnt",
    "histseg",
    "match",
    "nmatch",
    "pmullb",
    "pmullt",
    "raddhnb",
    "raddhnt",
    "rsubhnb",
    "rsubhnt",
    "sabalb",
    "sabalt",
    "sabdlb",
    "sabdlt",
    "saddlb",
    "saddlt",
    "saddwb",
    "saddwt",
    "smullb",
    "smullt",
    "splice",
    "sqcadd",
    "sshllb",
    "sshllt",
    "ssublb",
    "ssublt",
    "ssubwb",
    "ssubwt",
    "tbx",
    "uabalb",
    "uabalt",
    "uabdlb",
    "uabdlt",
    "uaddlb",
    "uaddlt",
    "uaddwb",
    "uaddwt",
    "umullb",
    "umullt",
    "ushllb",
    "ushllt",
    "usublb",
    "usublt",
    "usubwb",
    "usubwt",
    "whilege",
    "whilegt",
    "whilerw",
    "whilewr",
    "smlalb",
    "smlalt",
    "smlslb",
    "smlslt",
    "umlalb",
    "umlalt",
    "umlslb",
    "umlslt",
    "sqdmullb",
    "sqdmullt",
    "sqrdcmlah",
}

# Ordering precedes the size suffix. Loads take a, l, or al. Stores take l.
_LSE_OP = r"add|clr|eor|set|smax|smin|umax|umin"
_LSE = re.compile(
    rf"^(?:cas(?:p)?(?:a|l|al)?(?:b|h)?|"
    rf"ld(?:{_LSE_OP})(?:a|l|al)?(?:b|h)?|"
    rf"swp(?:a|l|al)?(?:b|h)?|"
    rf"st(?:{_LSE_OP})l?(?:b|h)?)$"
)
# libgcc and compiler-builtins hide LSE behind these names and a runtime check.
_OUTLINE_ATOMICS = re.compile(
    r"^__aarch64_(?:cas|ldadd|swp|ldclr|ldeor|ldset)"
)
_ZREG = re.compile(r"(?:^|[^A-Za-z0-9_])z\d+")
_PREG = re.compile(r"(?:^|[^A-Za-z0-9_])p\d+")
_ZAREG = re.compile(r"(?:^|[^A-Za-z0-9_])za\d*")
_HREG = re.compile(r"(?:^|[^A-Za-z0-9_])h\d+\b")
_HARR = re.compile(r"\.\d*h\b")
_FP16_MNEMONICS = {
    "fabs",
    "fadd",
    "fccmp",
    "fccmpe",
    "fcmp",
    "fcmpe",
    "fcsel",
    "fcvtas",
    "fcvtau",
    "fcvtms",
    "fcvtmu",
    "fcvtns",
    "fcvtnu",
    "fcvtps",
    "fcvtpu",
    "fcvtzs",
    "fcvtzu",
    "fdiv",
    "fmadd",
    "fmax",
    "fmaxnm",
    "fmin",
    "fminnm",
    "fmla",
    "fmls",
    "fmov",
    "fmsub",
    "fmul",
    "fneg",
    "fnmadd",
    "fnmsub",
    "fnmul",
    "frinta",
    "frinti",
    "frintm",
    "frintn",
    "frintp",
    "frintx",
    "frintz",
    "fsqrt",
    "fsub",
    "scvtf",
    "ucvtf",
}

_ELF_MAGIC = b"\x7fELF"
_MACHO_MAGICS = {
    0xFEEDFACF,
    0xCFFAEDFE,
    0xFEEDFACE,
    0xCEFAEDFE,
    0xCAFEBABE,
    0xBEBAFECA,
    0xCAFEBABF,
    0xBFBAFECA,
}
_CPU_ARM64 = 0x0100000C
_CPU_X86_64 = 0x01000007
_LC_SEGMENT_64 = 0x19
_LC_SYMTAB = 0x02
_LC_VERSION_MIN_MACOSX = 0x24
_LC_BUILD_VERSION = 0x32
_PLATFORM_MACOS = 1
_S_ATTR_PURE_INSTRUCTIONS = 0x80000000
_S_ATTR_SOME_INSTRUCTIONS = 0x00000400
# Apple Silicon is the only CPU that can map a macOS arm64 slice, and it
# implements both. These are the compiler's atomics, not a dispatched kernel.
_MACOS_ARM64_BASELINE = frozenset({"lse", "rcpc"})


def canonical_feature(name: str) -> str:
    """Normalise an allowlist or decoder feature name."""
    text = name.strip().lower().replace("-", "_")
    return _ALIASES.get(text, text)


@dataclass
class Func:
    name: str
    demangled: str
    start: int
    end: int
    local: bool = False


@dataclass
class Rule:
    pattern: re.Pattern[str]
    features: set[str]
    reason: str


@dataclass
class Allowlist:
    baseline: str | None
    allows: list[Rule] = field(default_factory=list)
    vendored: list[Rule] = field(default_factory=list)


@dataclass
class Violation:
    file: str
    symbol: str
    address: str
    mnemonic: str
    feature: str
    func_start: int | None = field(default=None, compare=False, repr=False)

    def line(self) -> str:
        return (
            f"VIOLATION {self.file} {self.symbol} {self.address} "
            f"{self.mnemonic} {self.feature}"
        )


@dataclass
class AuditStats:
    files_audited: int = 0
    instructions_decoded: int = 0
    undecoded: int = 0
    allowed_feature_counts: dict[str, int] = field(default_factory=dict)
    violations: list[Violation] = field(default_factory=list)
    not_audited: list[dict[str, str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    duplicates: int = 0
    dispatched: list[dict[str, object]] = field(default_factory=list)

    def allow(self, feature: str) -> None:
        self.allowed_feature_counts[feature] = (
            self.allowed_feature_counts.get(feature, 0) + 1
        )


def load_allowlist(path: Path) -> Allowlist:
    """Read an isa-allow.toml file."""
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    baseline = data.get("baseline")
    if baseline is not None and baseline not in ("x86-64", "x86-64-v2", "armv8-a"):
        raise SystemExit(f"unknown baseline {baseline!r} in {path}")
    allow = Allowlist(baseline=baseline)
    for key, dest in (("allow", allow.allows), ("vendored", allow.vendored)):
        for entry in data.get(key, []):
            raw = entry.get("symbol") if key == "allow" else entry.get("path")
            if not raw:
                raise SystemExit(f"{key} entry in {path} is missing its pattern")
            try:
                pattern = re.compile(raw)
            except re.error as exc:
                raise SystemExit(f"bad regex {raw!r} in {path}: {exc}") from exc
            features = {canonical_feature(item) for item in entry.get("features", [])}
            dest.append(
                Rule(
                    pattern=pattern,
                    features=features,
                    reason=str(entry.get("reason", "")),
                )
            )
    return allow


def _baseline_features(name: str) -> set[str]:
    if name == "x86-64":
        return _X86_64_V1
    if name == "x86-64-v2":
        return _X86_64_V2
    if name == "armv8-a":
        return set()
    raise SystemExit(f"unknown baseline {name!r}")


def _file_baseline(arch: str, override: str | None, allow: Allowlist) -> str:
    if arch == "x86_64":
        chosen = override or allow.baseline or "x86-64"
        if chosen not in ("x86-64", "x86-64-v2"):
            raise SystemExit(f"baseline {chosen} does not apply to an x86_64 file")
        return chosen
    if arch == "aarch64":
        chosen = override or "armv8-a"
        if chosen != "armv8-a":
            raise SystemExit(f"baseline {chosen} does not apply to an aarch64 file")
        return chosen
    raise SystemExit(f"unsupported architecture {arch}")


def _rules_allow(rules: list[Rule], text: str, feature: str) -> bool:
    for rule in rules:
        if feature in rule.features and rule.pattern.search(text):
            return True
    return False


def _function_allows(allow: Allowlist, func: Func | None, feature: str) -> bool:
    if func is None:
        return False
    return _rules_allow(allow.allows, func.demangled, feature) or _rules_allow(
        allow.allows, func.name, feature
    )


def _is_unwinder(func: Func | None) -> bool:
    if func is None:
        return False
    return bool(_UNWINDER.search(func.demangled) or _UNWINDER.search(func.name))


class FuncIndex:
    """Function starts for one section, sorted once and then bisected."""

    __slots__ = ("funcs", "starts")

    def __init__(self, funcs: list[Func]) -> None:
        self.funcs = funcs
        self.starts = [item.start for item in funcs]

    def lookup(self, ip: int) -> Func | None:
        if not self.starts:
            return None
        index = bisect.bisect_right(self.starts, ip) - 1
        if index < 0:
            return None
        func = self.funcs[index]
        if func.start <= ip < func.end:
            return func
        return None


def _finish_ranges(funcs: list[Func], section_end: int) -> list[Func]:
    ordered = sorted(funcs, key=lambda item: (item.start, item.end))
    if not ordered:
        return []
    for index, func in enumerate(ordered):
        if func.end <= func.start:
            nxt = ordered[index + 1].start if index + 1 < len(ordered) else section_end
            func.end = max(nxt, func.start + 1)
    return ordered


def _outline_atomic(func: Func | None, feature: str) -> bool:
    if feature != "lse" or func is None:
        return False
    return bool(
        _OUTLINE_ATOMICS.match(func.name) or _OUTLINE_ATOMICS.match(func.demangled)
    )


def _record(
    stats: AuditStats,
    allow: Allowlist,
    path: str,
    func: Func | None,
    address: int,
    mnemonic: str,
    features: list[str],
    has_symbols: bool,
) -> None:
    for feature in features:
        feature = canonical_feature(feature)
        if feature == "cet_ss" and _is_unwinder(func):
            continue
        if _outline_atomic(func, feature):
            stats.allow(feature)
            continue
        if _rules_allow(allow.vendored, path, feature):
            stats.allow(feature)
            continue
        if _function_allows(allow, func, feature):
            stats.allow(feature)
            continue
        if not has_symbols:
            symbol = "no symbols"
        elif func is not None:
            symbol = func.demangled
        else:
            symbol = "unknown"
        stats.violations.append(
            Violation(
                file=path,
                symbol=symbol,
                address=f"0x{address:x}",
                mnemonic=mnemonic,
                feature=feature,
                func_start=func.start if func is not None and func.local else None,
            )
        )


def _x86_feature_table() -> dict[int, str]:
    if CpuidFeature is None:
        raise SystemExit("iced-x86 is required to decode x86_64")
    table: dict[int, str] = {}
    for name in dir(CpuidFeature):
        if name.startswith("_") or name in {"CpuidFeature", "typing"}:
            continue
        value = getattr(CpuidFeature, name)
        if isinstance(value, int):
            table[value] = name.lower()
    return table


def _decode_x86(
    data: bytes,
    base: int,
    stats: AuditStats,
    allow: Allowlist,
    path: str,
    funcs: list[Func],
    has_symbols: bool,
    baseline: set[str],
) -> None:
    if Decoder is None or Formatter is None or FormatterSyntax is None:
        raise SystemExit("iced-x86 is required to decode x86_64")
    names = _x86_feature_table()
    index = FuncIndex(funcs)
    formatter = Formatter(FormatterSyntax.INTEL)
    decoder = Decoder(64, data, ip=base)
    for ins in decoder:
        if ins.is_invalid:
            stats.undecoded += 1
            continue
        stats.instructions_decoded += 1
        mnemonic = formatter.format_mnemonic(ins)
        above = []
        for raw in ins.cpuid_features():
            feature = names.get(int(raw), f"feature_{int(raw)}")
            # Legacy TZCNT is REP BSF (F3 0F BC). See the baseline comment.
            if feature == "bmi1" and mnemonic == "tzcnt":
                continue
            if feature not in baseline:
                above.append(feature)
        if not above:
            continue
        func = index.lookup(ins.ip)
        _record(
            stats,
            allow,
            path,
            func,
            ins.ip,
            mnemonic,
            above,
            has_symbols,
        )


def classify_aarch64(mnemonic: str, op_str: str) -> list[str]:
    """Return above-baseline features for one aarch64 instruction."""
    mnemonic = mnemonic.lower()
    if mnemonic in _ARM_HINTS:
        return []
    if mnemonic in {"smstart", "smstop"} or _ZAREG.search(op_str):
        return ["sme"]
    if mnemonic in _SVE2_MNEMONICS and (_ZREG.search(op_str) or _PREG.search(op_str)):
        return ["sve2"]
    if _ZREG.search(op_str) or _PREG.search(op_str) or mnemonic in _SVE_MNEMONICS:
        return ["sve"]
    if _LSE.match(mnemonic):
        return ["lse"]
    if mnemonic in {"fcadd", "fcmla"}:
        return ["fcma"]
    if mnemonic in {"usdot", "sudot", "smmla", "ummla", "usmmla"}:
        return ["i8mm"]
    if mnemonic in {"sdot", "udot"}:
        return ["dotprod"]
    if mnemonic in {"bfdot", "bfmlalb", "bfmlalt", "bfcvt", "bfcvtn", "bfcvtn2", "bfmmla"}:
        return ["bf16"]
    if mnemonic.startswith("crc32"):
        return ["crc32"]
    if mnemonic in {"aese", "aesd", "aesmc", "aesimc"}:
        return ["aes"]
    if mnemonic.startswith("sha1"):
        return ["sha1"]
    if mnemonic.startswith("sha256"):
        return ["sha2"]
    if mnemonic.startswith("sha512"):
        return ["sha512"]
    if mnemonic in {"sha3", "rax1", "xar", "bcax", "eor3"}:
        return ["sha3"]
    if mnemonic.startswith("sm3"):
        return ["sm3"]
    if mnemonic.startswith("sm4"):
        return ["sm4"]
    if mnemonic in {"pmull", "pmull2"}:
        return ["pmull"]
    if mnemonic in {"sqrdmlah", "sqrdmlsh"}:
        return ["rdm"]
    if mnemonic in {"fmlal", "fmlsl", "fmlal2", "fmlsl2"}:
        return ["fp16fml"]
    if mnemonic in {
        "irg",
        "addg",
        "subg",
        "subp",
        "subps",
        "gmi",
        "ldg",
        "stg",
        "st2g",
        "stzg",
        "stz2g",
        "stgp",
    }:
        return ["mte"]
    if mnemonic == "fjcvtzs":
        return ["jscvt"]
    if mnemonic in {"frint32z", "frint32x", "frint64z", "frint64x"}:
        return ["frintts"]
    if mnemonic in {"ldapr", "ldaprb", "ldaprh"}:
        return ["rcpc"]
    if mnemonic in {"cfinv", "rmif", "setf8", "setf16", "axflag", "xaflag"}:
        return ["flagm"]
    if mnemonic in {"retaa", "retab", "pacia", "pacib", "autia", "autib", "xpaci", "xpacd"}:
        return ["pauth"]
    if mnemonic in _FP16_MNEMONICS and (_HREG.search(op_str) or _HARR.search(op_str)):
        return ["fp16"]
    return []


def _decode_aarch64(
    data: bytes,
    base: int,
    stats: AuditStats,
    allow: Allowlist,
    path: str,
    funcs: list[Func],
    has_symbols: bool,
    platform_baseline: frozenset[str] = frozenset(),
) -> None:
    decoder = Cs(CS_ARCH_ARM64, CS_MODE_LITTLE_ENDIAN)
    index = FuncIndex(funcs)
    # Capstone stops at the first word it cannot decode. OpenBLAS puts a
    # constant pool in the text segment and SVE kernels after it, so a
    # stop must resume at the next word.
    # A bytes slice is copied on every resume. One writable buffer is not.
    buf = bytearray(data)
    view = memoryview(buf)
    offset = 0
    total = len(buf)
    while offset < total:
        start = offset
        for ins in decoder.disasm(view[offset:], base + offset):
            stats.instructions_decoded += 1
            features = [
                feature
                for feature in classify_aarch64(ins.mnemonic, ins.op_str)
                if feature not in platform_baseline
            ]
            if features:
                func = index.lookup(ins.address)
                _record(
                    stats,
                    allow,
                    path,
                    func,
                    ins.address,
                    ins.mnemonic,
                    features,
                    has_symbols,
                )
            offset = ins.address - base + ins.size
        if offset == start:
            stats.undecoded += 1
            offset += 4


def _elf_file_has_functions(elf: ELFFile) -> bool:
    for name in (".symtab", ".dynsym"):
        section = elf.get_section_by_name(name)
        if section is None:
            continue
        for sym in section.iter_symbols():
            if sym["st_info"]["type"] in {"STT_FUNC", "STT_GNU_IFUNC"} and sym.name:
                return True
    return False


def _elf_funcs(elf: ELFFile, section_index: int, base: int, reloc: bool) -> list[Func]:
    chosen = None
    for name in (".symtab", ".dynsym"):
        section = elf.get_section_by_name(name)
        if section is None:
            continue
        found: list[Func] = []
        for sym in section.iter_symbols():
            info = sym["st_info"]["type"]
            if info not in {"STT_FUNC", "STT_GNU_IFUNC"}:
                continue
            shndx = sym["st_shndx"]
            if shndx != section_index:
                continue
            start = sym["st_value"]
            if reloc:
                start = base + start
            size = sym["st_size"]
            raw = sym.name or ""
            if not raw:
                continue
            found.append(
                Func(
                    name=raw,
                    demangled=demangle(raw),
                    start=start,
                    end=start + size if size else start,
                    local=sym["st_info"]["bind"] == "STB_LOCAL",
                )
            )
        if found:
            chosen = found
            break
    return chosen or []


def _audit_elf(
    blob: bytes,
    path: str,
    stats: AuditStats,
    allow: Allowlist,
    override: str | None,
) -> None:
    elf = ELFFile(io.BytesIO(blob))
    machine = elf.header["e_machine"]
    if machine == "EM_X86_64":
        arch = "x86_64"
    elif machine == "EM_AARCH64":
        arch = "aarch64"
    else:
        stats.not_audited.append({"file": path, "reason": f"ELF machine {machine} is not audited"})
        return
    baseline_name = _file_baseline(arch, override, allow)
    baseline = _baseline_features(baseline_name)
    elf_type = elf.header["e_type"]
    reloc = elf_type == "ET_REL"
    file_has_symbols = _elf_file_has_functions(elf)
    any_code = False
    first_violation = len(stats.violations)
    sections: list[tuple[int, bytes, list[Func]]] = []
    for index, section in enumerate(elf.iter_sections()):
        flags = section["sh_flags"]
        if not flags & 0x4:
            continue
        data = section.data()
        if not data:
            continue
        any_code = True
        base = section["sh_addr"]
        funcs = _finish_ranges(
            _elf_funcs(elf, index, base, reloc),
            base + len(data),
        )
        sections.append((base, data, funcs))
        if arch == "x86_64":
            _decode_x86(data, base, stats, allow, path, funcs, file_has_symbols, baseline)
        else:
            _decode_aarch64(data, base, stats, allow, path, funcs, file_has_symbols)
    if not any_code:
        stats.not_audited.append({"file": path, "reason": "ELF has no executable section"})
        return
    stats.files_audited += 1
    if arch == "x86_64":
        _allow_dispatched(elf, sections, stats, path, first_violation)


def _allow_dispatched(
    elf: ELFFile,
    sections: list[tuple[int, bytes, list[Func]]],
    stats: AuditStats,
    path: str,
    first: int,
) -> None:
    """Drop this file's violations that sit in a proven dispatched clone."""
    pending = stats.violations[first:]
    wanted = {item.func_start for item in pending if item.func_start is not None}
    if not wanted:
        return
    proofs = prove(elf, sections, wanted)
    kept: list[Violation] = []
    used: dict[int, set[str]] = {}
    symbols: dict[int, str] = {}
    for item in pending:
        proof = proofs.get(item.func_start) if item.func_start is not None else None
        if proof is not None and item.feature in proof.features:
            stats.allow(item.feature)
            used.setdefault(item.func_start, set()).add(item.feature)
            symbols[item.func_start] = item.symbol
            continue
        kept.append(item)
    stats.violations[first:] = kept
    for start in sorted(used):
        stats.dispatched.append(
            {
                "file": path,
                "symbol": symbols[start],
                "name": proofs[start].name,
                "features": sorted(used[start]),
                "proven": sorted(proofs[start].features),
                "via": list(proofs[start].via),
            }
        )


def _macho_slices(blob: bytes) -> list[tuple[str, bytes]]:
    if len(blob) < 8:
        return []
    magic = struct.unpack_from("<I", blob, 0)[0]
    if magic in (0xFEEDFACF, 0xCFFAEDFE):
        return [("thin", blob)]
    if magic in (0xCAFEBABE, 0xBEBAFECA):
        count = struct.unpack_from(">I", blob, 4)[0]
        slices = []
        offset = 8
        for _ in range(count):
            _cputype, _subtype, file_off, size, _align = struct.unpack_from(
                ">IIIII", blob, offset
            )
            slices.append(("fat", blob[file_off : file_off + size]))
            offset += 20
        return slices
    if magic in (0xCAFEBABF, 0xBFBAFECA):
        count = struct.unpack_from(">I", blob, 4)[0]
        slices = []
        offset = 8
        for _ in range(count):
            _cputype, _subtype, _reserved, file_off, size, _align = struct.unpack_from(
                ">IIQQII", blob, offset
            )
            slices.append(("fat64", blob[file_off : file_off + size]))
            offset += 32
        return slices
    return []


def _parse_macho(
    blob: bytes,
) -> tuple[str, list[tuple[int, bytes, list[Func]]], frozenset[str]] | None:
    if len(blob) < 32:
        return None
    magic, cputype, _subtype, _filetype, ncmds, sizeofcmds, _flags, _reserved = (
        struct.unpack_from("<IIIIIIII", blob, 0)
    )
    if magic != 0xFEEDFACF:
        return None
    if cputype == _CPU_ARM64:
        arch = "aarch64"
    elif cputype == _CPU_X86_64:
        arch = "x86_64"
    else:
        return None
    offset = 32
    end = 32 + sizeofcmds
    sections: list[tuple[int, int, int, int, int]] = []
    sym_off = sym_count = str_off = str_size = 0
    platform_baseline: set[str] = set()
    while offset + 8 <= min(end, len(blob)):
        cmd, cmdsize = struct.unpack_from("<II", blob, offset)
        if cmdsize < 8 or offset + cmdsize > len(blob):
            return None
        if cmd == _LC_BUILD_VERSION and cmdsize >= 24 and arch == "aarch64":
            platform, _minos, _sdk, _ntools = struct.unpack_from("<IIII", blob, offset + 8)
            if platform == _PLATFORM_MACOS:
                platform_baseline.update(_MACOS_ARM64_BASELINE)
        elif cmd == _LC_VERSION_MIN_MACOSX and cmdsize >= 16 and arch == "aarch64":
            platform_baseline.update(_MACOS_ARM64_BASELINE)
        elif cmd == _LC_SEGMENT_64 and cmdsize >= 72:
            (
                _segname,
                _vmaddr,
                _vmsize,
                _fileoff,
                _filesize,
                _maxprot,
                _initprot,
                nsects,
                _segflags,
            ) = struct.unpack_from("<16sQQQQIIIi", blob, offset + 8)
            sec_off = offset + 72
            for _ in range(nsects):
                if sec_off + 80 > len(blob):
                    return None
                _sect, _seg, addr, size, fileoff, _align, _reloff, _nreloc, flags, *_rest = (
                    struct.unpack_from("<16s16sQQIIIIIIII", blob, sec_off)
                )
                sections.append((addr, size, fileoff, flags, len(sections) + 1))
                sec_off += 80
        elif cmd == _LC_SYMTAB and cmdsize >= 24:
            sym_off, sym_count, str_off, str_size = struct.unpack_from("<IIII", blob, offset + 8)
        offset += cmdsize
    code_sections = []
    raw_syms: list[tuple[int, int, str]] = []
    if sym_count and str_off + str_size <= len(blob):
        string_table = blob[str_off : str_off + str_size]
        for index in range(sym_count):
            entry = sym_off + index * 16
            if entry + 16 > len(blob):
                break
            n_strx, n_type, n_sect, _n_desc, n_value = struct.unpack_from("<IBBHQ", blob, entry)
            if n_type & 0xE0:
                continue
            if (n_type & 0x0E) != 0x0E or n_sect == 0:
                continue
            if n_strx >= len(string_table):
                continue
            end = string_table.find(b"\x00", n_strx)
            name = string_table[n_strx:end].decode("utf-8", "replace")
            if not name or name.startswith("_$") or name in {"$x", "$d"}:
                continue
            raw_syms.append((n_sect, n_value, name))
    for addr, size, fileoff, flags, sect_index in sections:
        executable = bool(
            flags & (_S_ATTR_PURE_INSTRUCTIONS | _S_ATTR_SOME_INSTRUCTIONS)
        )
        if not executable:
            continue
        if fileoff + size > len(blob):
            return None
        funcs = []
        for sect, value, name in raw_syms:
            if sect != sect_index:
                continue
            plain = name[1:] if name.startswith("_") else name
            funcs.append(
                Func(name=name, demangled=demangle(plain), start=value, end=value)
            )
        funcs = _finish_ranges(funcs, addr + size)
        code_sections.append((addr, blob[fileoff : fileoff + size], funcs))
    return arch, code_sections, frozenset(platform_baseline)


def _audit_macho(
    blob: bytes,
    path: str,
    stats: AuditStats,
    allow: Allowlist,
    override: str | None,
) -> None:
    slices = _macho_slices(blob)
    if not slices:
        stats.not_audited.append({"file": path, "reason": "Mach-O not audited: unrecognised container"})
        return
    audited = False
    for _kind, piece in slices:
        parsed = _parse_macho(piece)
        if parsed is None:
            stats.not_audited.append(
                {"file": path, "reason": "Mach-O not audited: slice is not a 64-bit arm64 or x86_64 image"}
            )
            continue
        arch, sections, platform_baseline = parsed
        baseline_name = _file_baseline(arch, override, allow)
        baseline = _baseline_features(baseline_name)
        has_symbols = any(funcs for _addr, _data, funcs in sections)
        if not sections:
            stats.not_audited.append({"file": path, "reason": "Mach-O has no executable section"})
            continue
        if platform_baseline:
            note = (
                "platform baseline macOS arm64: "
                + ", ".join(sorted(platform_baseline))
                + " (architectural on every CPU that can load this binary)"
            )
            if note not in stats.notes:
                stats.notes.append(note)
        for addr, data, funcs in sections:
            if arch == "x86_64":
                _decode_x86(data, addr, stats, allow, path, funcs, has_symbols, baseline)
            else:
                _decode_aarch64(
                    data,
                    addr,
                    stats,
                    allow,
                    path,
                    funcs,
                    has_symbols,
                    platform_baseline,
                )
        audited = True
    if audited:
        stats.files_audited += 1


def _sniff(blob: bytes) -> str | None:
    if blob.startswith(_ELF_MAGIC):
        return "elf"
    if len(blob) >= 4:
        magic = struct.unpack_from("<I", blob, 0)[0]
        if magic in _MACHO_MAGICS:
            return "macho"
    return None


def audit_bytes(
    blob: bytes,
    path: str,
    stats: AuditStats,
    allow: Allowlist,
    override: str | None,
) -> None:
    kind = _sniff(blob)
    if kind == "elf":
        _audit_elf(blob, path, stats, allow, override)
    elif kind == "macho":
        _audit_macho(blob, path, stats, allow, override)
    else:
        stats.not_audited.append({"file": path, "reason": "not an ELF or Mach-O"})


def _audit_whl(
    path: Path,
    display: str,
    stats: AuditStats,
    allow: Allowlist,
    override: str | None,
    seen: set[str] | None = None,
) -> None:
    if seen is None:
        seen = set()
    with tempfile.TemporaryDirectory(prefix="isa-audit-") as tmp:
        root = Path(tmp)
        with zipfile.ZipFile(path) as archive:
            archive.extractall(root)
        found = False
        for dirpath, _dirnames, filenames in os.walk(root):
            for name in filenames:
                file_path = Path(dirpath) / name
                blob = file_path.read_bytes()
                if _sniff(blob) is None:
                    continue
                found = True
                digest = hashlib.sha256(blob).hexdigest()
                if digest in seen:
                    stats.duplicates += 1
                    continue
                seen.add(digest)
                rel = file_path.relative_to(root).as_posix()
                audit_bytes(blob, f"{display}:{rel}", stats, allow, override)
        if not found:
            stats.not_audited.append({"file": display, "reason": "wheel contains no ELF or Mach-O"})


def audit_path(
    path: Path,
    stats: AuditStats,
    allow: Allowlist,
    override: str | None,
    explicit: bool,
    seen: set[str] | None = None,
) -> None:
    if seen is None:
        seen = set()
    if path.is_dir():
        for dirpath, _dirnames, filenames in os.walk(path):
            for name in filenames:
                audit_path(Path(dirpath) / name, stats, allow, override, False, seen)
        return
    if not path.is_file():
        if explicit:
            stats.not_audited.append({"file": str(path), "reason": "not a file"})
        return
    if path.suffix == ".whl":
        _audit_whl(path, str(path), stats, allow, override, seen)
        return
    blob = path.read_bytes()
    if _sniff(blob) is None:
        if explicit:
            stats.not_audited.append({"file": str(path), "reason": "not an ELF, Mach-O, or wheel"})
        return
    digest = hashlib.sha256(blob).hexdigest()
    if digest in seen:
        stats.duplicates += 1
        return
    seen.add(digest)
    audit_bytes(blob, str(path), stats, allow, override)


def render_text(stats: AuditStats) -> str:
    lines = [
        f"files audited: {stats.files_audited}",
        f"instructions decoded: {stats.instructions_decoded}",
        f"undecoded instructions: {stats.undecoded}",
        "allowed feature counts:",
    ]
    if stats.allowed_feature_counts:
        for name in sorted(stats.allowed_feature_counts):
            lines.append(f"  {name}: {stats.allowed_feature_counts[name]}")
    else:
        lines.append("  (none)")
    if stats.duplicates:
        lines.append(f"duplicate files skipped: {stats.duplicates}")
    if stats.notes:
        lines.append("notes:")
        for note in stats.notes:
            lines.append(f"  {note}")
    if stats.dispatched:
        lines.append(f"dispatched clones: {len(stats.dispatched)}")
        for item in stats.dispatched:
            lines.append(
                f"  DISPATCHED {item['file']} {item['name']} "
                f"[{', '.join(item['features'])}] via {', '.join(item['via'])}"
            )
    lines.append(f"violations: {len(stats.violations)}")
    for item in stats.violations:
        lines.append(item.line())
    if stats.not_audited:
        lines.append("not audited:")
        for item in stats.not_audited:
            lines.append(f"  {item['file']}: {item['reason']}")
    return "\n".join(lines) + "\n"


def render_json(stats: AuditStats) -> dict[str, object]:
    return {
        "files_audited": stats.files_audited,
        "instructions_decoded": stats.instructions_decoded,
        "undecoded_instructions": stats.undecoded,
        "allowed_feature_counts": stats.allowed_feature_counts,
        "violations": [
            {
                "file": item.file,
                "symbol": item.symbol,
                "address": item.address,
                "mnemonic": item.mnemonic,
                "feature": item.feature,
            }
            for item in stats.violations
        ],
        "not_audited": stats.not_audited,
        "dispatched": stats.dispatched,
        "notes": stats.notes,
        "duplicate_files_skipped": stats.duplicates,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit binaries for instructions outside the ISA baseline.")
    parser.add_argument("paths", nargs="+", help="ELF, Mach-O, directories, or .whl files")
    parser.add_argument("--allow", required=True, help="isa-allow.toml")
    parser.add_argument(
        "--baseline",
        choices=("x86-64", "x86-64-v2", "armv8-a"),
        default=None,
        help="Override the per-file default (x86-64 on x86_64, armv8-a on aarch64)",
    )
    parser.add_argument("--json", dest="json_path", help="Write a JSON report to this path")
    args = parser.parse_args(argv)
    allow_path = Path(args.allow)
    if not allow_path.is_file():
        print(f"allow file not found: {allow_path}", file=sys.stderr)
        return 2
    allow = load_allowlist(allow_path)
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    stats = AuditStats()
    for raw in args.paths:
        audit_path(Path(raw), stats, allow, args.baseline, True)
    text = render_text(stats)
    sys.stdout.write(text)
    if args.json_path:
        Path(args.json_path).write_text(json.dumps(render_json(stats), indent=2) + "\n")
    if stats.files_audited == 0:
        print("no ELF or Mach-O files were audited", file=sys.stderr)
        return 2
    if stats.not_audited and any("not audited" in item["reason"] for item in stats.not_audited):
        # A slice we refused to decode is a failed audit, not a pass.
        if any(item["reason"].startswith("Mach-O not audited") for item in stats.not_audited):
            return 1
    return 1 if stats.violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
