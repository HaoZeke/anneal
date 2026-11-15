"""Prove from a linked x86_64 ELF which local functions run only after a CPU check.

A function is a dispatched clone when nothing outside the file can enter
it and every way in from inside the file passes a branch on libgcc's CPU
model. That covers the three forms the compilers emit:

- an ifunc resolver from ``target_clones``, which calls
  ``__cpu_indicator_init``, tests ``__cpu_model`` and returns the
  address of one clone;
- ``if (__builtin_cpu_supports("avx2")) clone();``, a load and mask of
  ``__cpu_model`` (or ``__cpu_features2``) ahead of a conditional branch,
  with the clone reached by a call or tail jump on the taken side;
- the same test moved into a probe function, ``if (have_avx2()) clone();``,
  when the probe is not inlined (``-O0``, ``noinline``). A probe called
  directly is summarised by the features in force wherever it returns a
  value other than a constant zero, following ``setcc``, ``movzx`` and
  per-path constants into ``eax``;
- a chain of local helpers that are reached only from such a clone.

"Nothing outside can enter" means ``STB_LOCAL`` with no global, weak or
dynamic alias anywhere in its range, and no relocation whose target lies
in that range (a function pointer in data, ``.init_array``, a vtable).
Every code reference is a direct branch or a RIP-relative operand,
because the file is position independent. A reference from code outside
any function symbol, from a function with a computed jump, or from an
unguarded path proves nothing.

The features a clone may use are the bits the guard proved, closed under
the implications GCC applies to ``-m`` options (``-mavx2`` also enables
AVX, SSE4.2, POPCNT and XSAVE). ``test_dispatch.py`` checks the bit table
and the closure against the installed compiler.

Only ET_DYN files are analysed: in a non-PIC executable an absolute
address may sit in data with no relocation.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field, replace

from elftools.elf.elffile import ELFFile

try:
    from iced_x86 import (
        ConditionCode,
        Decoder,
        FlowControl,
        InstructionInfoFactory,
        MemorySizeExt,
        Mnemonic,
        OpAccess,
        OpKind,
        Register,
        RegisterExt,
    )
except ImportError:  # aarch64 runners have no iced-x86 wheel
    Decoder = None  # type: ignore[misc, assignment]

# libgcc's enum processor_features. Bits 0-31 are __cpu_model.__cpu_features[0];
# 32 onwards are __cpu_features2, 32 per word. Values are iced-x86 names.
# Only indices checked against GCC are listed; an unlisted bit proves nothing.
CPU_MODEL_FEATURES: dict[int, str] = {
    0: "cmov",
    1: "mmx",
    2: "popcnt",
    3: "sse",
    4: "sse2",
    5: "sse3",
    6: "ssse3",
    7: "sse4_1",
    8: "sse4_2",
    9: "avx",
    10: "avx2",
    11: "sse4a",
    12: "fma4",
    13: "xop",
    14: "fma",
    15: "avx512f",
    16: "bmi1",
    17: "bmi2",
    18: "aes",
    19: "pclmulqdq",
    20: "avx512vl",
    21: "avx512bw",
    22: "avx512dq",
    23: "avx512cd",
    24: "avx512er",
    25: "avx512pf",
    26: "avx512_vbmi",
    27: "avx512_ifma",
    28: "avx512_4vnniw",
    29: "avx512_4fmaps",
    30: "avx512_vpopcntdq",
    31: "avx512_vbmi2",
    32: "gfni",
    33: "vpclmulqdq",
    34: "avx512_vnni",
    35: "avx512_bitalg",
    36: "avx512_bf16",
    49: "f16c",
    57: "lzcnt",
    58: "movbe",
}

# Direct implications of GCC's -m options (gcc -m<isa> -dM -E), as iced names.
IMPLIES: dict[str, frozenset[str]] = {
    "ssse3": frozenset({"sse3"}),
    "sse4_1": frozenset({"ssse3"}),
    "sse4_2": frozenset({"sse4_1", "popcnt"}),
    "sse4a": frozenset({"sse3"}),
    "avx": frozenset({"sse4_2", "xsave"}),
    "avx2": frozenset({"avx"}),
    "fma": frozenset({"avx"}),
    "f16c": frozenset({"avx"}),
    "fma4": frozenset({"avx", "sse4a"}),
    "xop": frozenset({"fma4"}),
    "avx512f": frozenset({"avx2"}),
    "avx512vl": frozenset({"avx512f"}),
    "avx512bw": frozenset({"avx512f"}),
    "avx512dq": frozenset({"avx512f"}),
    "avx512cd": frozenset({"avx512f"}),
    "avx512er": frozenset({"avx512f"}),
    "avx512pf": frozenset({"avx512f"}),
    "avx512_ifma": frozenset({"avx512f"}),
    "avx512_vpopcntdq": frozenset({"avx512f"}),
    "avx512_vbmi2": frozenset({"avx512f"}),
    "avx512_vnni": frozenset({"avx512f"}),
    "avx512_bitalg": frozenset({"avx512f"}),
    "avx512_4vnniw": frozenset({"avx512f"}),
    "avx512_4fmaps": frozenset({"avx512f"}),
    "avx512_vbmi": frozenset({"avx512bw"}),
    "avx512_bf16": frozenset({"avx512bw"}),
}

_FEATURE_SYMBOLS = {"__cpu_model": (12, 4, 0), "__cpu_features2": (0, None, 32)}
# pyelftools names STT_GNU_IFUNC by its generic value, STT_LOOS.
_IFUNC = {"STT_GNU_IFUNC", "STT_LOOS"}
_R_X86_64_64 = 1
_R_X86_64_GLOB_DAT = 6
_R_X86_64_RELATIVE = 8
_R_X86_64_IRELATIVE = 37
# Pseudo addresses for __cpu_model and __cpu_features2 when they are
# imported from libgcc_s and only reachable through a GOT slot.
_IMPORTED_BASE = 1 << 63
# Bounds on probe summaries: a probe is a few loads and tests, so larger or
# deeper callees are not summarised and their result proves nothing.
_PROBE_DEPTH = 2
_PROBE_BYTES = 512


# GNU as encodes an EVEX instruction as VEX when its operands allow it, so
# an AVX-512F body holds VEX FMA and F16C encodings although -mavx512f
# does not enable those options.
ENCODER_IMPLIES: dict[str, frozenset[str]] = {
    "avx512f": frozenset({"fma", "f16c"}),
}


def closure(features: set[str] | frozenset[str]) -> frozenset[str]:
    """Close a feature set under IMPLIES and ENCODER_IMPLIES."""
    out = set(features)
    todo = list(out)
    while todo:
        name = todo.pop()
        for implied in IMPLIES.get(name, frozenset()) | ENCODER_IMPLIES.get(name, frozenset()):
            if implied not in out:
                out.add(implied)
                todo.append(implied)
    return frozenset(out)


@dataclass(frozen=True)
class Proof:
    name: str
    features: frozenset[str]
    via: tuple[str, ...]


@dataclass
class _Fn:
    name: str
    start: int
    end: int
    local: bool
    escaped: bool = False
    refs: list[tuple[int, int]] = field(default_factory=list)


@dataclass(frozen=True)
class _Ptr:
    addr: int


@dataclass(frozen=True)
class _Word:
    """Bits shift onwards of the feature word at addr; width counts the bits left."""

    addr: int
    width: int
    inverted: bool = False
    mask: int | None = None
    shift: int = 0


@dataclass(frozen=True)
class _Const:
    value: int


@dataclass(frozen=True)
class _Ret:
    """A probe result: nonzero in its low byte only under byte, in the whole register only under word."""

    byte: frozenset[str]
    word: frozenset[str]


@dataclass(frozen=True)
class _Flags:
    # "z": result of and/test; "all": cmp of a masked word with its mask;
    # "c": bt; "nz": zero test of a probe result, ZF clear under proven
    kind: str
    word: _Word | None
    mask: int
    sign: int = -1
    proven: frozenset[str] = frozenset()


_TOP = None


class _Image:
    """Feature words, GOT slots and relocation targets of one ELF."""

    def __init__(self, elf: ELFFile) -> None:
        self.elf = elf
        self.byte_base: dict[int, int] = {}
        self.ranges: list[tuple[int, int]] = []
        self.slots: dict[int, int] = {}
        self.reloc_targets: list[int] = []
        symbols = self._symbols()
        imported: dict[str, int] = {}
        for index, (name, (offset, size, first)) in enumerate(_FEATURE_SYMBOLS.items()):
            defined = [sym for sym in symbols if sym.name == name and sym["st_shndx"] != "SHN_UNDEF"]
            if defined:
                base = defined[0]["st_value"]
                sym_size = defined[0]["st_size"]
            else:
                base = _IMPORTED_BASE + index * 0x1000
                sym_size = 0
                imported[name] = base
            words = size if size is not None else (sym_size or 12)
            start = base + offset
            self.ranges.append((base, base + max(sym_size, offset + words)))
            for k in range(words):
                self.byte_base[start + k] = first + 8 * k
        self._relocations(symbols, imported)

    def _symbols(self) -> list:
        out = []
        for name in (".symtab", ".dynsym"):
            section = self.elf.get_section_by_name(name)
            if section is not None:
                out.extend(section.iter_symbols())
        return out

    def in_feature_range(self, addr: int) -> bool:
        return any(lo <= addr < hi for lo, hi in self.ranges)

    def read_word(self, addr: int) -> int | None:
        for section in self.elf.iter_sections():
            lo = section["sh_addr"]
            if section["sh_type"] != "SHT_NOBITS" and lo <= addr and addr + 8 <= lo + section["sh_size"]:
                data = section.data()
                return int.from_bytes(data[addr - lo : addr - lo + 8], "little")
        return None

    def _relocations(self, symbols: list, imported: dict[str, int]) -> None:
        for section in self.elf.iter_sections():
            kind = section["sh_type"]
            if kind not in ("SHT_RELA", "SHT_REL", "SHT_RELR"):
                continue
            if kind == "SHT_RELR":
                for rel in section.iter_relocations():
                    value = self.read_word(rel["r_offset"])
                    if value is not None:
                        self._note(rel["r_offset"], value)
                continue
            symtab = None
            if section["sh_link"]:
                symtab = self.elf.get_section(section["sh_link"])
            for rel in section.iter_relocations():
                offset = rel["r_offset"]
                if rel.is_RELA():
                    addend = rel["r_addend"]
                else:
                    addend = self.read_word(offset) or 0
                sym_index = rel["r_info_sym"]
                rtype = rel["r_info_type"]
                if sym_index == 0 or rtype in (_R_X86_64_RELATIVE, _R_X86_64_IRELATIVE):
                    self._note(offset, addend)
                    continue
                if symtab is None or not hasattr(symtab, "get_symbol"):
                    continue
                sym = symtab.get_symbol(sym_index)
                if sym.name in imported and rtype in (_R_X86_64_GLOB_DAT, _R_X86_64_64):
                    self.slots[offset] = imported[sym.name] + addend
                    continue
                if sym["st_shndx"] != "SHN_UNDEF":
                    self._note(offset, sym["st_value"] + addend)

    def _note(self, slot: int, target: int) -> None:
        self.reloc_targets.append(target)
        if self.in_feature_range(target):
            self.slots[slot] = target


def _signed(value: int) -> int:
    return value - (1 << 64) if value >= 1 << 63 else value


class _Machine:
    """Symbolic registers and flags for one function's feature tests."""

    def __init__(self, image: _Image, summary=None) -> None:
        self.image = image
        self.summary = summary
        self.info = InstructionInfoFactory()
        self.high = {Register.AH: Register.RAX, Register.BH: Register.RBX, Register.CH: Register.RCX, Register.DH: Register.RDX}

    def _address(self, ins, regs: dict) -> int | None:
        if ins.is_ip_rel_memory_operand:
            return ins.ip_rel_memory_address
        if ins.memory_index != Register.NONE or ins.memory_base == Register.NONE:
            return None
        held = regs.get(RegisterExt.full_register(ins.memory_base))
        if isinstance(held, _Ptr):
            return held.addr + _signed(ins.memory_displacement)
        return None

    def _reg_view(self, reg) -> tuple[int, int, int]:
        """Full register, bit shift and width in bits of a register operand."""
        if reg in self.high:
            return self.high[reg], 8, 8
        return RegisterExt.full_register(reg), 0, RegisterExt.size(reg) * 8

    def _mask_in_word(self, word: _Word, imm: int, shift: int, width: int) -> int | None:
        mask = (imm & ((1 << width) - 1)) << shift
        if mask == 0 or mask >> word.width:
            return None
        return mask

    def step(self, ins, regs: dict, flags: _Flags | None) -> _Flags | None:
        """Apply one instruction. Mutates regs and returns the new flags."""
        mn = ins.mnemonic
        result_reg = None
        result_val = None
        new_flags: _Flags | None | bool = False  # False: derive from rflags_modified
        op0 = ins.op0_kind
        op1 = ins.op1_kind if ins.op_count > 1 else None

        if mn == Mnemonic.LEA and op0 == OpKind.REGISTER and ins.is_ip_rel_memory_operand:
            addr = ins.ip_rel_memory_address
            if self.image.in_feature_range(addr):
                result_reg, result_val = RegisterExt.full_register(ins.op0_register), _Ptr(addr)
        elif mn in (Mnemonic.MOV, Mnemonic.MOVZX) and op0 == OpKind.REGISTER and op1 == OpKind.MEMORY:
            dest = ins.op0_register
            dest_bytes = RegisterExt.size(dest)
            addr = self._address(ins, regs)
            mem_bytes = MemorySizeExt.size(ins.memory_size)
            if addr is not None and dest_bytes >= 4 and mem_bytes in (1, 2, 4, 8):
                full = RegisterExt.full_register(dest)
                if mem_bytes == 8 and addr in self.image.slots:
                    result_reg, result_val = full, _Ptr(self.image.slots[addr])
                elif addr in self.image.byte_base:
                    result_reg, result_val = full, _Word(addr, mem_bytes * 8)
        elif mn == Mnemonic.MOV and op0 == OpKind.REGISTER and op1 == OpKind.REGISTER:
            dest, src = ins.op0_register, ins.op1_register
            size = RegisterExt.size(dest)
            if size >= 4 and RegisterExt.size(src) == size:
                held = regs.get(RegisterExt.full_register(src))
                if (
                    (isinstance(held, _Word) and held.width <= size * 8)
                    or (isinstance(held, _Ptr) and size == 8)
                    or isinstance(held, (_Const, _Ret))
                ):
                    result_reg, result_val = RegisterExt.full_register(dest), held
        elif mn == Mnemonic.MOV and op0 == OpKind.REGISTER and op1 in _IMMEDIATES:
            if RegisterExt.size(ins.op0_register) >= 4:
                result_reg = RegisterExt.full_register(ins.op0_register)
                result_val = _Const(ins.immediate(1) & ((1 << 8 * RegisterExt.size(ins.op0_register)) - 1))
        elif (
            mn == Mnemonic.XOR
            and op0 == OpKind.REGISTER
            and op1 == OpKind.REGISTER
            and ins.op0_register == ins.op1_register
            and RegisterExt.size(ins.op0_register) >= 4
        ):
            result_reg, result_val = RegisterExt.full_register(ins.op0_register), _Const(0)
        elif mn == Mnemonic.MOVZX and op0 == OpKind.REGISTER and op1 == OpKind.REGISTER:
            full, shift, width = self._reg_view(ins.op1_register)
            held = regs.get(full)
            if shift == 0 and width == 8 and RegisterExt.size(ins.op0_register) >= 4:
                if isinstance(held, _Ret):
                    result_reg, result_val = RegisterExt.full_register(ins.op0_register), _Ret(held.byte, held.byte)
                elif isinstance(held, _Const):
                    result_reg, result_val = RegisterExt.full_register(ins.op0_register), _Const(held.value & 0xFF)
        elif mn in _SETS and op0 == OpKind.REGISTER:
            full, shift, width = self._reg_view(ins.op0_register)
            if shift == 0 and flags is not None:
                held = regs.get(full)
                features = frozenset(_edge_features(self.image, flags, ins.condition_code, True))
                clean = isinstance(held, _Const) and held.value < 0x100
                result_reg, result_val = full, _Ret(features, features if clean else frozenset())
        elif mn == Mnemonic.SHR and op0 == OpKind.REGISTER and op1 in _IMMEDIATES:
            full, shift, width = self._reg_view(ins.op0_register)
            held = regs.get(full)
            count = ins.immediate(1) & 63
            if isinstance(held, _Word) and shift == 0 and width >= 32 and held.mask is None and 0 < count < held.width:
                result_reg = full
                result_val = replace(held, width=held.width - count, shift=held.shift + count)
        elif mn == Mnemonic.NOT and op0 == OpKind.REGISTER:
            full, shift, width = self._reg_view(ins.op0_register)
            held = regs.get(full)
            if isinstance(held, _Word) and shift == 0 and width >= 32 and held.width <= width and held.mask is None:
                result_reg, result_val = full, replace(held, inverted=not held.inverted)
        elif mn in (Mnemonic.AND, Mnemonic.TEST) and op0 == OpKind.REGISTER and op1 in _IMMEDIATES:
            full, shift, width = self._reg_view(ins.op0_register)
            held = regs.get(full)
            new_flags = None
            if isinstance(held, _Word) and held.mask is None:
                mask = self._mask_in_word(held, ins.immediate(1), shift, width)
                if mask is not None:
                    new_flags = _Flags("z", held, mask, shift + width - 1)
                    if mn == Mnemonic.AND and shift == 0 and width >= 32:
                        result_reg = full
                        result_val = replace(held, mask=mask)
        elif mn == Mnemonic.TEST and op0 == OpKind.REGISTER and op1 == OpKind.REGISTER and ins.op0_register == ins.op1_register:
            full, shift, width = self._reg_view(ins.op0_register)
            held = regs.get(full)
            new_flags = None
            if isinstance(held, _Word):
                mask = held.mask if held.mask is not None else (1 << held.width) - 1
                mask &= ((1 << width) - 1) << shift
                if mask and not mask >> held.width:
                    new_flags = _Flags("z", held, mask, shift + width - 1)
            elif isinstance(held, _Ret) and shift == 0:
                new_flags = _Flags("nz", None, 0, proven=held.byte if width == 8 else held.word)
        elif mn == Mnemonic.TEST and op0 == OpKind.MEMORY and op1 in _IMMEDIATES:
            addr = self._address(ins, regs)
            new_flags = None
            if addr is not None and addr in self.image.byte_base:
                width = MemorySizeExt.size(ins.memory_size) * 8
                word = _Word(addr, width)
                mask = self._mask_in_word(word, ins.immediate(1), 0, width)
                if mask is not None:
                    new_flags = _Flags("z", word, mask, width - 1)
        elif mn == Mnemonic.CMP and op0 == OpKind.REGISTER and op1 in _IMMEDIATES:
            full, shift, width = self._reg_view(ins.op0_register)
            held = regs.get(full)
            new_flags = None
            if isinstance(held, _Word) and held.mask is not None and shift == 0:
                if (ins.immediate(1) & ((1 << width) - 1)) == held.mask:
                    new_flags = _Flags("all", held, held.mask)
            elif isinstance(held, _Ret) and shift == 0 and ins.immediate(1) == 0:
                new_flags = _Flags("nz", None, 0, proven=held.byte if width == 8 else held.word)
        elif mn == Mnemonic.BT and op1 in _IMMEDIATES:
            new_flags = None
            word = None
            shift = 0
            if op0 == OpKind.REGISTER:
                full, shift, width = self._reg_view(ins.op0_register)
                held = regs.get(full)
                if isinstance(held, _Word) and held.mask is None:
                    word = held
            elif op0 == OpKind.MEMORY:
                addr = self._address(ins, regs)
                if addr is not None and addr in self.image.byte_base:
                    word = _Word(addr, 32)
            if word is not None:
                bit = 1 << ((ins.immediate(1) & 31) + shift)
                if not bit >> word.width:
                    new_flags = _Flags("c", word, bit)

        if ins.flow_control in (FlowControl.CALL, FlowControl.INDIRECT_CALL):
            regs.clear()
            target = ins.near_branch_target if ins.flow_control == FlowControl.CALL else 0
            probe = self.summary(target) if target and self.summary is not None else None
            if probe is not None:
                regs[Register.RAX] = _Ret(*probe)
        else:
            for used in self.info.info(ins).used_registers():
                if used.access in _WRITES:
                    regs.pop(RegisterExt.full_register(used.register), None)
        if result_reg is not None:
            regs[result_reg] = result_val
        if new_flags is False:
            return None if ins.rflags_modified else flags
        return new_flags


if Decoder is not None:
    _IMMEDIATES = {OpKind.IMMEDIATE8, OpKind.IMMEDIATE16, OpKind.IMMEDIATE32, OpKind.IMMEDIATE8TO16,
                   OpKind.IMMEDIATE8TO32, OpKind.IMMEDIATE8TO64, OpKind.IMMEDIATE32TO64}
    _WRITES = {OpAccess.WRITE, OpAccess.COND_WRITE, OpAccess.READ_WRITE, OpAccess.READ_COND_WRITE}
    _READS = {OpAccess.READ, OpAccess.COND_READ, OpAccess.READ_WRITE, OpAccess.READ_COND_WRITE}
    _CMOVS = {getattr(Mnemonic, name) for name in dir(Mnemonic) if name.startswith("CMOV")}
    _SETS = {getattr(Mnemonic, name) for name in dir(Mnemonic) if name.startswith("SET") and len(name) <= 6}


def _edge_features(image: _Image, flags: _Flags | None, cc, taken: bool) -> set[str]:
    """Features known on one side of a conditional branch."""
    if flags is not None and flags.kind == "nz":
        if not taken:
            cc = _NEGATE.get(cc)
        return set(flags.proven) if cc in _NONZERO else set()
    return _features_of(image, _edge_bits(flags, cc, taken))


def _edge_bits(flags: _Flags | None, cc, taken: bool) -> list[tuple[_Word, int]]:
    """Word bits known set on one side of a conditional branch."""
    if flags is None:
        return []
    if not taken:
        cc = _NEGATE.get(cc)
        if cc is None:
            return []
    word, mask = flags.word, flags.mask
    inv = word.inverted
    single = mask & (mask - 1) == 0
    if flags.kind == "z":
        if cc in _NONZERO and not inv and single:
            return [(word, mask)]
        if cc == ConditionCode.E and inv:
            return [(word, mask)]
        sign = 1 << flags.sign if flags.sign >= 0 else 0
        if sign & mask and ((cc == ConditionCode.S and not inv) or (cc == ConditionCode.NS and inv)):
            return [(word, sign)]
        return []
    if flags.kind == "all":
        if cc == ConditionCode.E and not inv:
            return [(word, mask)]
        return []
    if flags.kind == "c" and ((cc == ConditionCode.B and not inv) or (cc == ConditionCode.AE and inv)):
        return [(word, mask)]
    return []


if Decoder is not None:
    _NEGATE = {
        ConditionCode.E: ConditionCode.NE,
        ConditionCode.NE: ConditionCode.E,
        ConditionCode.B: ConditionCode.AE,
        ConditionCode.AE: ConditionCode.B,
        ConditionCode.S: ConditionCode.NS,
        ConditionCode.NS: ConditionCode.S,
        ConditionCode.G: ConditionCode.LE,
        ConditionCode.LE: ConditionCode.G,
        ConditionCode.A: ConditionCode.BE,
        ConditionCode.BE: ConditionCode.A,
    }
    # and, test and cmp with zero leave OF and CF clear, so after them "greater" and
    # "above" mean ZF clear.
    _NONZERO = {ConditionCode.NE, ConditionCode.G, ConditionCode.A}


def _features_of(image: _Image, bits: list[tuple[_Word, int]]) -> set[str]:
    out = set()
    for word, mask in bits:
        k = 0
        while mask >> k:
            if mask >> k & 1:
                bit = k + word.shift
                base = image.byte_base.get(word.addr + bit // 8)
                if base is not None:
                    name = CPU_MODEL_FEATURES.get(base + bit % 8)
                    if name is not None:
                        out.add(name)
            k += 1
    return out


def _is_nop(ins) -> bool:
    if ins.mnemonic == Mnemonic.NOP:
        return True
    return (
        ins.mnemonic == Mnemonic.XCHG
        and ins.op_count == 2
        and ins.op0_kind == OpKind.REGISTER
        and ins.op1_kind == OpKind.REGISTER
        and ins.op0_register == ins.op1_register
    )


def _meet(a, b):
    """Meet of two block states (features, regs, flags). None is top."""
    if a is _TOP:
        return b
    if b is _TOP:
        return a
    regs = {key: val for key, val in a[1].items() if b[1].get(key) == val}
    flags = a[2] if a[2] == b[2] else None
    return (a[0] & b[0], regs, flags)


class _Flow:
    """Basic blocks of one function and the must-analysis over them."""

    def __init__(self, machine: _Machine, code: bytes, base: int, start: int, end: int) -> None:
        self.machine = machine
        self.ok = False
        data = code[start - base : end - base]
        instrs = list(Decoder(64, data, ip=start))
        if not instrs or any(ins.is_invalid for ins in instrs):
            return
        leaders = {start}
        for ins in instrs:
            fc = ins.flow_control
            target = ins.near_branch_target
            if target and start <= target < end:
                leaders.add(target)
            if fc not in (FlowControl.NEXT, FlowControl.CALL, FlowControl.INDIRECT_CALL):
                leaders.add(ins.next_ip)
            if fc == FlowControl.INDIRECT_BRANCH and not ins.is_ip_rel_memory_operand:
                return
        blocks: list[list] = []
        for ins in instrs:
            if ins.ip in leaders or not blocks:
                blocks.append([])
            blocks[-1].append(ins)
        by_start = {block[0].ip: index for index, block in enumerate(blocks)}
        succs: list[list[tuple[int, object, bool]]] = []
        for block in blocks:
            last = block[-1]
            fc = last.flow_control
            out = []
            target = last.near_branch_target
            if fc == FlowControl.CONDITIONAL_BRANCH:
                if target in by_start:
                    out.append((by_start[target], last.condition_code, True))
                if last.next_ip in by_start:
                    out.append((by_start[last.next_ip], last.condition_code, False))
            elif fc == FlowControl.UNCONDITIONAL_BRANCH:
                if target in by_start:
                    out.append((by_start[target], None, True))
            elif fc in (FlowControl.NEXT, FlowControl.CALL, FlowControl.INDIRECT_CALL) and last.next_ip in by_start:
                out.append((by_start[last.next_ip], None, False))
            succs.append(out)
        preds: list[list[tuple[int, object, bool]]] = [[] for _ in blocks]
        for index, out in enumerate(succs):
            for dest, cc, taken in out:
                preds[dest].append((index, cc, taken))
        self.blocks, self.succs, self.preds = blocks, succs, preds
        self.ok = True
        self._solve()

    def _solve(self) -> None:
        blocks, succs, preds = self.blocks, self.succs, self.preds
        state_in: list = [_TOP] * len(blocks)
        # A block no direct edge reaches may be a landing pad or a computed
        # target, so it starts with nothing proven. Alignment padding after a
        # jump is not an entry.
        self.entries = set()
        for index in range(len(blocks)):
            if index == 0 or (not preds[index] and not all(_is_nop(ins) for ins in blocks[index])):
                state_in[index] = (frozenset(), {}, None)
                self.entries.add(index)
        exits: list = [None] * len(blocks)
        work = sorted(self.entries)
        while work:
            index = work.pop()
            features, regs, flags = state_in[index]
            regs = dict(regs)
            for ins in blocks[index]:
                flags = self.machine.step(ins, regs, flags)
            exits[index] = (regs, flags)
            for dest, cc, taken in succs[index]:
                incoming = (self._gained(features, flags, cc, taken), regs, flags)
                merged = _meet(state_in[dest], incoming)
                if dest not in self.entries and merged != state_in[dest]:
                    state_in[dest] = merged
                    work.append(dest)
        self.state_in, self.exits = state_in, exits

    def _gained(self, features, flags, cc, taken) -> frozenset[str]:
        if cc is None:
            return features
        return frozenset(features | _edge_features(self.machine.image, flags, cc, taken))

    def incoming(self, index: int) -> list:
        """The state on each edge into a block, or its merged state for an entry."""
        if index in self.entries:
            return [self.state_in[index]]
        out = []
        for pred, cc, taken in self.preds[index]:
            if self.exits[pred] is None:
                continue
            regs, flags = self.exits[pred]
            out.append((self._gained(self.state_in[pred][0], flags, cc, taken), regs, flags))
        return out


def guards(
    image: _Image,
    code: bytes,
    base: int,
    start: int,
    end: int,
    targets: set[int],
    resolver: bool = False,
    summary=None,
) -> dict[int, frozenset[str]]:
    """Features proven at each instruction of [start, end) whose address is in targets.

    resolver marks an ifunc resolver, whose only result is rax. summary maps
    a direct call target to its probe summary, see _summary.
    """
    machine = _Machine(image, summary)
    flow = _Flow(machine, code, base, start, end)
    if not flow.ok:
        return {ip: frozenset() for ip in targets}
    blocks, succs, state_in, exits = flow.blocks, flow.succs, flow.state_in, flow.exits
    proven: dict[int, frozenset[str]] = {}
    for index, block in enumerate(blocks):
        if state_in[index] is _TOP:
            for ins in block:
                if ins.ip in targets:
                    proven[ins.ip] = frozenset()
            continue
        features = state_in[index][0]
        for pos, ins in enumerate(block):
            if ins.ip not in targets:
                continue
            got = set(features)
            if ins is block[-1] and ins.flow_control == FlowControl.CONDITIONAL_BRANCH:
                flags = exits[index][1] if exits[index] else None
                got |= _edge_features(image, flags, ins.condition_code, True)
            elif ins.mnemonic == Mnemonic.LEA and ins.op0_kind == OpKind.REGISTER:
                got = set(_follow_address(image, machine, blocks, succs, state_in[index], index, pos, resolver))
            proven[ins.ip] = closure(got)
    for ip in targets:
        proven.setdefault(ip, frozenset())
    return proven


def _single_bit(image: _Image, word: _Word, mask: int) -> frozenset[str]:
    """Features known when word & mask is nonzero."""
    if mask == 0 or mask & (mask - 1):
        return frozenset()
    return frozenset(_features_of(image, [(word, mask)]))


def _summary(flow: _Flow) -> tuple[frozenset[str], frozenset[str]] | None:
    """Features in force whenever a function returns nonzero in al, and in eax.

    Each return is judged on every edge into its block, so a shared epilogue
    after "mov eax, 1" on one path and "xor eax, eax" on another still counts
    only the first path.
    """
    if not flow.ok:
        return None
    byte = word = _TOP
    for index, block in enumerate(flow.blocks):
        if block[-1].flow_control != FlowControl.RETURN or flow.state_in[index] is _TOP:
            continue
        for features, regs, flags in flow.incoming(index):
            regs = dict(regs)
            for ins in block:
                flags = flow.machine.step(ins, regs, flags)
            held = regs.get(Register.RAX)
            if isinstance(held, _Const) and held.value == 0:
                continue
            low, full = features, features
            if isinstance(held, _Ret):
                low, full = features | held.byte, features | held.word
            elif isinstance(held, _Word) and held.mask is not None and not held.inverted:
                low = features | _single_bit(flow.machine.image, held, held.mask & 0xFF)
                full = features | _single_bit(flow.machine.image, held, held.mask)
            byte = low if byte is _TOP else byte & low
            word = full if word is _TOP else word & full
    if byte is _TOP:
        return None
    return frozenset(byte), frozenset(word)


def _follow_address(image, machine, blocks, succs, state, index, pos, resolver) -> frozenset[str]:
    """Features in force wherever the address that a LEA loads is used.

    The address may move through register copies and CMOVcc, which adds the
    condition of the move, along unconditional control flow. Any other read
    of a register that holds it, a call, a conditional branch or an exit
    from the function counts as a use with the features of that register.
    ret uses rax, and rdx too unless the function is an ifunc resolver.
    """
    features, regs, flags = state
    regs = dict(regs)
    block = blocks[index]
    for ins in block[:pos]:
        flags = machine.step(ins, regs, flags)
    lea = block[pos]
    flags = machine.step(lea, regs, flags)
    live: dict[int, frozenset[str]] = {RegisterExt.full_register(lea.op0_register): frozenset(features)}
    result: frozenset[str] | None = _TOP

    def use(names) -> None:
        nonlocal result
        for reg in names:
            if reg in live:
                result = live[reg] if result is _TOP else result & live[reg]

    returns = {Register.RAX} if resolver else {Register.RAX, Register.RDX}
    pos += 1
    for _step in range(256):
        if pos >= len(block):
            nexts = [dest for dest, cc, _taken in succs[index] if cc is None]
            if len(succs[index]) != 1 or len(nexts) != 1:
                use(list(live))
                break
            index, pos = nexts[0], 0
            block = blocks[index]
            continue
        ins = block[pos]
        fc = ins.flow_control
        reads = set()
        writes = set()
        for used in machine.info.info(ins).used_registers():
            full = RegisterExt.full_register(used.register)
            if used.access in _READS:
                reads.add(full)
            if used.access in _WRITES:
                writes.add(full)
        moved: dict[int, frozenset[str]] = {}
        if ins.op_count == 2 and ins.op0_kind == OpKind.REGISTER and ins.op1_kind == OpKind.REGISTER:
            dest = RegisterExt.full_register(ins.op0_register)
            src = RegisterExt.full_register(ins.op1_register)
            if src in live and RegisterExt.size(ins.op1_register) == 8:
                if ins.mnemonic == Mnemonic.MOV:
                    moved[dest] = live[src]
                    reads.discard(src)
                elif ins.mnemonic in _CMOVS:
                    moved[dest] = live[src] | frozenset(
                        _edge_features(image, flags, ins.condition_code, True)
                    )
                    reads.discard(src)
                    reads.discard(dest)
        if fc == FlowControl.UNCONDITIONAL_BRANCH and len(succs[index]) == 1:
            index, pos = succs[index][0][0], 0
            block = blocks[index]
            continue
        if fc == FlowControl.RETURN:
            use(returns)
            break
        if fc != FlowControl.NEXT:
            use(list(live))
            break
        use(reads)
        for reg in writes:
            live.pop(reg, None)
        live.update(moved)
        flags = machine.step(ins, regs, flags)
        if not live:
            break
        pos += 1
    else:
        use(list(live))
    return frozenset() if result is _TOP else result


def prove(elf: ELFFile, sections: list[tuple[int, bytes, list]], wanted: set[int]) -> dict[int, Proof]:
    """Proofs for the functions starting at the addresses in wanted.

    sections holds (base, bytes, funcs) for each executable section, where
    each func has name, start, end and local. Names in a Proof are raw
    symbol names: demangling drops the clone suffix.
    """
    if Decoder is None or elf.header["e_type"] != "ET_DYN" or elf.header["e_machine"] != "EM_X86_64":
        return {}
    image = _Image(elf)
    fns: list[_Fn] = []
    for _base, _data, funcs in sections:
        for func in funcs:
            fns.append(_Fn(func.name, func.start, func.end, func.local))
    fns.sort(key=lambda item: item.start)
    starts = [item.start for item in fns]

    def lookup(addr: int) -> _Fn | None:
        index = bisect.bisect_right(starts, addr) - 1
        if index >= 0 and fns[index].start <= addr < fns[index].end:
            return fns[index]
        return None

    exported: list[int] = []
    resolvers: set[int] = set()
    for name in (".symtab", ".dynsym"):
        section = elf.get_section_by_name(name)
        if section is None:
            continue
        for sym in section.iter_symbols():
            if sym["st_shndx"] == "SHN_UNDEF":
                continue
            ifunc = sym["st_info"]["type"] in _IFUNC
            if ifunc:
                resolvers.add(sym["st_value"])
            if name == ".dynsym" or sym["st_info"]["bind"] != "STB_LOCAL" or ifunc:
                exported.append(sym["st_value"])
    for addr in exported + image.reloc_targets:
        fn = lookup(addr)
        if fn is not None:
            fn.escaped = True

    code_at: list[tuple[int, bytes]] = [(base, data) for base, data, _funcs in sections]
    for base, data in code_at:
        for ins in Decoder(64, data, ip=base):
            target = ins.near_branch_target
            if not target and ins.is_ip_rel_memory_operand:
                target = ins.ip_rel_memory_address
            if not target:
                continue
            callee = lookup(target)
            if callee is None or not callee.local or callee.escaped:
                continue
            caller = lookup(ins.ip)
            if caller is callee:
                continue
            if caller is None:
                callee.escaped = True
                continue
            callee.refs.append((caller.start, ins.ip))

    by_start = {item.start: item for item in fns}
    needed: set[int] = set()
    todo = [addr for addr in wanted if addr in by_start]
    while todo:
        addr = todo.pop()
        if addr in needed:
            continue
        needed.add(addr)
        fn = by_start[addr]
        if not fn.local or fn.escaped:
            continue
        for caller, _ip in fn.refs:
            todo.append(caller)

    def code_of(fn: _Fn) -> tuple[bytes, int, int] | None:
        for base, data in code_at:
            if base <= fn.start < base + len(data):
                return data, base, min(fn.end, base + len(data))
        return None

    # A probe is a small function whose callers test its return value. A
    # direct call binds to the definition in this file whatever its binding.
    summaries: dict[int, tuple[frozenset[str], frozenset[str]] | None] = {}

    def summary_at(depth: int):
        def summary(addr: int):
            if addr in summaries:
                return summaries[addr]
            fn = by_start.get(addr)
            located = code_of(fn) if fn is not None else None
            if located is None or depth >= _PROBE_DEPTH or fn.end - fn.start > _PROBE_BYTES:
                return None
            summaries[addr] = None
            data, base, end = located
            flow = _Flow(_Machine(image, summary_at(depth + 1)), data, base, fn.start, end)
            summaries[addr] = _summary(flow)
            return summaries[addr]

        return summary

    edge_guard: dict[int, frozenset[str]] = {}
    sites: dict[int, set[int]] = {}
    for addr in needed:
        fn = by_start[addr]
        if fn.local and not fn.escaped:
            for caller, ip in fn.refs:
                sites.setdefault(caller, set()).add(ip)
    for caller, ips in sites.items():
        fn = by_start[caller]
        located = code_of(fn)
        if located is None:
            edge_guard.update({ip: frozenset() for ip in ips})
            continue
        data, base, end = located
        edge_guard.update(guards(image, data, base, fn.start, end, ips, fn.start in resolvers, summary_at(0)))

    def eligible(fn: _Fn) -> bool:
        return fn.local and not fn.escaped and bool(fn.refs)

    proven: dict[int, frozenset[str] | None] = {}
    for addr in needed:
        proven[addr] = _TOP if eligible(by_start[addr]) else frozenset()
    changed = True
    while changed:
        changed = False
        for addr in needed:
            fn = by_start[addr]
            if not eligible(fn):
                continue
            acc: frozenset[str] | None = _TOP
            for caller, ip in fn.refs:
                upstream = proven.get(caller, frozenset())
                here = _TOP if upstream is _TOP else upstream | edge_guard.get(ip, frozenset())
                if here is _TOP:
                    continue
                acc = here if acc is _TOP else acc & here
            if acc != proven[addr]:
                proven[addr] = acc
                changed = True
    out: dict[int, Proof] = {}
    for addr in wanted:
        features = proven.get(addr) or frozenset()
        if not features:
            continue
        fn = by_start[addr]
        via = tuple(sorted({by_start[caller].name for caller, _ip in fn.refs}))
        out[addr] = Proof(name=fn.name, features=features, via=via)
    return out
