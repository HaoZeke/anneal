"""Overwrite one instruction of a named x86_64 ELF function with an AVX one.

    python plant.py IN OUT SYMBOL

One instruction of SYMBOL becomes vzeroupper (C5 F8 77) padded with NOPs,
so the rest of the function still decodes: ENDBR64 at the entry when there
is one, else the first instruction of three bytes or more that neither
branches, nor touches memory, nor writes flags, so a CPU check in SYMBOL
is left intact. Failing both, the first instruction long enough is used.
The audit of OUT must then report SYMBOL unless SYMBOL is a dispatched
clone.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

from elftools.elf.elffile import ELFFile
from iced_x86 import Code, Decoder, FlowControl, OpKind

VZEROUPPER = bytes.fromhex("c5f877")


def _inert(ins) -> bool:
    if ins.flow_control != FlowControl.NEXT or ins.rflags_modified:
        return False
    return all(ins.op_kind(i) != OpKind.MEMORY for i in range(ins.op_count))


def plant(blob: bytes, symbol: str, code: bytes = VZEROUPPER) -> tuple[bytes, int]:
    """Return the patched bytes and the virtual address that was patched."""
    elf = ELFFile(io.BytesIO(blob))
    symtab = elf.get_section_by_name(".symtab")
    if symtab is None:
        raise SystemExit("no .symtab")
    matches = [sym for sym in symtab.iter_symbols() if sym.name == symbol and sym["st_size"]]
    if not matches:
        raise SystemExit(f"symbol {symbol} not found")
    sym = matches[0]
    section = elf.get_section(sym["st_shndx"])
    offset = sym["st_value"] - section["sh_addr"] + section["sh_offset"]
    body = blob[offset : offset + sym["st_size"]]
    instrs = list(Decoder(64, body, ip=sym["st_value"]))
    if instrs and instrs[0].code == Code.ENDBR64:
        chosen = instrs[0]
    else:
        long_enough = [ins for ins in instrs if ins.len >= len(code)]
        chosen = next((ins for ins in long_enough if _inert(ins)), long_enough[0] if long_enough else None)
    if chosen is not None:
        at = offset + chosen.ip - sym["st_value"]
        patch = code + b"\x90" * (chosen.len - len(code))
        return blob[:at] + patch + blob[at + chosen.len :], chosen.ip
    raise SystemExit(f"{symbol} has no instruction of {len(code)} bytes or more")


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    src, dst, symbol = argv
    patched, address = plant(Path(src).read_bytes(), symbol)
    Path(dst).write_bytes(patched)
    print(f"planted vzeroupper in {symbol} at 0x{address:x}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
