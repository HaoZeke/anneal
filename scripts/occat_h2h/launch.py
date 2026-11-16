#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Run the released OCCat binary with a fixed seed and an evaluation cap.

The release is a race-enabled Go executable. Its text cannot be made
writable from inside the process, so this launcher pokes a small timing
stub into a fresh executable page and detaches before the search. The
stub charges one energy evaluation per call of the Lennard-Jones or
Gupta energy, times that call and the matching gradient, and writes the
totals to the inherited account file.
"""

from __future__ import annotations

import ctypes
import os
import struct
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUB_S = HERE / "stub.S"
PTRACE_TRACEME = 0
PTRACE_PEEKTEXT = 1
PTRACE_POKETEXT = 4
PTRACE_CONT = 7
PTRACE_GETREGS = 12
PTRACE_SETREGS = 13
PTRACE_DETACH = 17
PTRACE_SYSCALL = 24
SYS_MMAP = 9
SYS_MEMFD = 319
PROT_RWX = 7
MAP_SHARED_32 = 0x01 | 0x40

libc = ctypes.CDLL("libc.so.6", use_errno=True)
libc.syscall.restype = ctypes.c_long


class Regs(ctypes.Structure):
    _fields_ = [
        ("r15", ctypes.c_ulonglong),
        ("r14", ctypes.c_ulonglong),
        ("r13", ctypes.c_ulonglong),
        ("r12", ctypes.c_ulonglong),
        ("rbp", ctypes.c_ulonglong),
        ("rbx", ctypes.c_ulonglong),
        ("r11", ctypes.c_ulonglong),
        ("r10", ctypes.c_ulonglong),
        ("r9", ctypes.c_ulonglong),
        ("r8", ctypes.c_ulonglong),
        ("rax", ctypes.c_ulonglong),
        ("rcx", ctypes.c_ulonglong),
        ("rdx", ctypes.c_ulonglong),
        ("rsi", ctypes.c_ulonglong),
        ("rdi", ctypes.c_ulonglong),
        ("orig_rax", ctypes.c_ulonglong),
        ("rip", ctypes.c_ulonglong),
        ("cs", ctypes.c_ulonglong),
        ("eflags", ctypes.c_ulonglong),
        ("rsp", ctypes.c_ulonglong),
        ("ss", ctypes.c_ulonglong),
        ("fs_base", ctypes.c_ulonglong),
        ("gs_base", ctypes.c_ulonglong),
        ("ds", ctypes.c_ulonglong),
        ("es", ctypes.c_ulonglong),
        ("fs", ctypes.c_ulonglong),
        ("gs", ctypes.c_ulonglong),
    ]


def _ptrace(req: int, pid: int, addr: int = 0, data: int = 0) -> int:
    libc.ptrace.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
    libc.ptrace.restype = ctypes.c_ulong
    ctypes.set_errno(0)
    value = libc.ptrace(req, pid, ctypes.c_void_p(addr), ctypes.c_void_p(data))
    err = ctypes.get_errno()
    if value == 2**64 - 1 and err != 0:
        raise OSError(err, f"ptrace {req}")
    return value


def getregs(pid: int) -> Regs:
    regs = Regs()
    libc.ptrace.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(Regs)]
    libc.ptrace.restype = ctypes.c_long
    if libc.ptrace(PTRACE_GETREGS, pid, None, ctypes.byref(regs)) != 0:
        raise OSError(ctypes.get_errno(), "GETREGS")
    return regs


def setregs(pid: int, regs: Regs) -> None:
    libc.ptrace.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(Regs)]
    libc.ptrace.restype = ctypes.c_long
    if libc.ptrace(PTRACE_SETREGS, pid, None, ctypes.byref(regs)) != 0:
        raise OSError(ctypes.get_errno(), "SETREGS")


def peek(pid: int, addr: int) -> int:
    libc.ptrace.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
    libc.ptrace.restype = ctypes.c_ulong
    ctypes.set_errno(0)
    value = libc.ptrace(PTRACE_PEEKTEXT, pid, ctypes.c_void_p(addr & ~7), None)
    if value == 2**64 - 1 and ctypes.get_errno() != 0:
        raise OSError(ctypes.get_errno(), f"peek {addr:#x}")
    return value


def poke_word(pid: int, addr: int, value: int) -> None:
    aligned = addr & ~7
    if aligned != addr:
        raise ValueError(f"unaligned poke {addr:#x}")
    _ptrace(PTRACE_POKETEXT, pid, aligned, value & 0xFFFFFFFFFFFFFFFF)


def poke_bytes(pid: int, addr: int, data: bytes) -> None:
    offset = 0
    while offset < len(data):
        base = (addr + offset) & ~7
        shift = (addr + offset) - base
        old = peek(pid, base).to_bytes(8, "little")
        take = min(8 - shift, len(data) - offset)
        new = old[:shift] + data[offset : offset + take] + old[shift + take :]
        poke_word(pid, base, int.from_bytes(new, "little"))
        offset += take


def wait_stop(pid: int) -> int:
    status = os.waitpid(pid, 0)[1]
    if os.WIFSTOPPED(status):
        return os.WSTOPSIG(status)
    raise RuntimeError(f"OCCat pid {pid} status {status}")


def vdso_syscall(pid: int) -> int:
    maps = Path(f"/proc/{pid}/maps").read_text().splitlines()
    span = None
    for line in maps:
        if "[vdso]" in line:
            a, b = line.split()[0].split("-")
            span = (int(a, 16), int(b, 16))
            break
    if span is None:
        raise RuntimeError("no vdso")
    mem = os.open(f"/proc/{pid}/mem", os.O_RDONLY)
    try:
        os.lseek(mem, span[0], os.SEEK_SET)
        blob = os.read(mem, span[1] - span[0])
    finally:
        os.close(mem)
    at = blob.find(b"\x0f\x05")
    if at < 0:
        raise RuntimeError("no syscall instruction in the vdso")
    return span[0] + at


def inject_mmap(pid: int, syscall_at: int, fd: int, size: int) -> int:
    saved = getregs(pid)
    regs = getregs(pid)
    regs.rax = SYS_MMAP
    regs.rdi = 0
    regs.rsi = size
    regs.rdx = PROT_RWX
    regs.r10 = MAP_SHARED_32
    regs.r8 = fd
    regs.r9 = 0
    regs.rip = syscall_at
    setregs(pid, regs)
    _ptrace(PTRACE_SYSCALL, pid, 0, 0)
    wait_stop(pid)
    _ptrace(PTRACE_SYSCALL, pid, 0, 0)
    wait_stop(pid)
    result = getregs(pid).rax
    setregs(pid, saved)
    if result > 2**63:
        errno = (2**64 - result) & 0xFFFFFFFF
        raise OSError(errno, "injected mmap")
    return result


def assemble() -> tuple[bytes, dict[str, int]]:
    work = Path("/tmp/occat-stub")
    work.mkdir(exist_ok=True)
    subprocess.run(
        ["gcc", "-c", "-o", str(work / "stub.o"), str(STUB_S)],
        check=True,
    )
    subprocess.run(
        ["ld", "-nostdlib", "-Ttext=0", "-o", str(work / "stub.elf"), str(work / "stub.o")],
        check=True,
    )
    subprocess.run(
        ["objcopy", "-O", "binary", "-j", ".text", str(work / "stub.elf"), str(work / "stub.bin")],
        check=True,
    )
    symbols = {}
    text = subprocess.check_output(["nm", "-n", str(work / "stub.elf")], text=True)
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[1] in {"T", "t"}:
            symbols[parts[2]] = int(parts[0], 16)
    return (work / "stub.bin").read_bytes(), symbols


def patch_rel(pid: int, site: int, dest: int, opcode: int, width: int) -> None:
    rel = dest - (site + 5)
    if rel != ctypes.c_int32(rel).value:
        raise RuntimeError(f"rel32 {site:#x} -> {dest:#x} does not fit")
    raw = bytes([opcode]) + struct.pack("<i", rel) + b"\x90" * (width - 5)
    original = b"".join(
        peek(pid, site + i).to_bytes(8, "little")[(site + i) & 7 : ((site + i) & 7) + 1]
        for i in range(width)
    )
    poke_bytes(pid, site, raw)
    # The original bytes are checked by the caller when it wants to.


def ns_per_tsc() -> float:
    libc.clock_gettime.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_long * 2)]
    a = (ctypes.c_long * 2)()
    b = (ctypes.c_long * 2)()
    libc.clock_gettime(1, a)

    def rdtsc() -> int:
        return int.from_bytes(subprocess.check_output(["/tmp/occat-rdtsc"]), "little")

    # Fallback below if the helper is absent. Build it once.
    helper = Path("/tmp/occat-rdtsc.c")
    if not Path("/tmp/occat-rdtsc").exists():
        helper.write_text(
            "int main(void){unsigned a,d; __asm__ volatile(\"lfence;rdtsc\":\"=a\"(a),\"=d\"(d));"
            "unsigned long long t=((unsigned long long)d<<32)|a; "
            "return (int)(t & 255);} /* unused */\n"
        )
    return 0.3  # replaced


def calibrate() -> float:
    src = Path("/tmp/occat-cal.c")
    src.write_text(
        r"""
        #include <stdio.h>
        #include <time.h>
        static unsigned long long rdtsc(void) {
            unsigned a, d;
            __asm__ volatile("lfence; rdtsc" : "=a"(a), "=d"(d));
            return ((unsigned long long)d << 32) | a;
        }
        int main(void) {
            struct timespec a, b;
            clock_gettime(CLOCK_MONOTONIC, &a);
            unsigned long long t0 = rdtsc();
            do { clock_gettime(CLOCK_MONOTONIC, &b); }
            while ((b.tv_sec - a.tv_sec) * 1000000000L + (b.tv_nsec - a.tv_nsec) < 20000000L);
            unsigned long long t1 = rdtsc();
            double ns = (b.tv_sec - a.tv_sec) * 1e9 + (b.tv_nsec - a.tv_nsec);
            printf("%.10f\n", ns / (double)(t1 - t0));
            return 0;
        }
        """
    )
    bin_path = Path("/tmp/occat-cal")
    if not bin_path.exists():
        subprocess.run(["gcc", "-O2", "-o", str(bin_path), str(src)], check=True)
    return float(subprocess.check_output([str(bin_path)], text=True).strip())


def run(
    config: Path,
    account: Path,
    seed: int,
    budget: int,
    reference: float,
    tol: float,
) -> dict[str, float | int]:
    blob, symbols = assemble()
    memfd = libc.syscall(SYS_MEMFD, ctypes.c_char_p(b"occat-acct"), 0)
    if memfd < 0:
        raise OSError(ctypes.get_errno(), "memfd_create")
    os.ftruncate(memfd, 8192)
    os.lseek(memfd, 0, os.SEEK_SET)
    os.write(memfd, blob.ljust(8192, b"\0"))
    pid = os.fork()
    if pid == 0:
        os.chdir(config.parent)
        _ptrace(PTRACE_TRACEME, 0, 0, 0)
        env = os.environ.copy()
        env["GOMAXPROCS"] = "1"
        env["OMP_NUM_THREADS"] = "1"
        os.execve(
            "/tmp/OCCat_release/bin/occat",
            ["occat", "--config", str(config)],
            env,
        )
        os._exit(127)
    try:
        wait_stop(pid)
        syscall_at = vdso_syscall(pid)
        cave = inject_mmap(pid, syscall_at, memfd, 8192)
        image = bytearray(blob)
        def put_u64(name: str, value: int) -> None:
            at = symbols[name]
            image[at : at + 8] = value.to_bytes(8, "little")

        def put_f64(name: str, value: float) -> None:
            at = symbols[name]
            image[at : at + 8] = struct.pack("<d", value)

        limit = reference + tol if reference < 0.0 else 1.0e300
        put_f64("best", 1.0e300)
        put_f64("reference", reference)
        put_f64("tol", tol)
        put_f64("limit", limit)
        put_u64("budget", budget)
        put_u64("seedv", seed if seed else 1)
        put_u64("acct_fd", 0)
        poke_bytes(pid, cave, bytes(image))
        patch_rel(pid, 0x5AEFC0, cave + symbols["energy_lj"], 0xE8, 5)
        patch_rel(pid, 0x5ACA60, cave + symbols["energy_gupta"], 0xE8, 5)
        patch_rel(pid, 0x4A8DD9, cave + symbols["seed_stub"], 0xE8, 5)
        patch_rel(pid, 0x5AE540, cave + symbols["grad_enter_lj"], 0xE9, 8)
        patch_rel(pid, 0x5AE7D6, cave + symbols["grad_exit_lj"], 0xE9, 8)
        patch_rel(pid, 0x5AABC0, cave + symbols["grad_enter_gupta"], 0xE9, 8)
        patch_rel(pid, 0x5AB6C6, cave + symbols["grad_exit_gupta"], 0xE9, 8)
        started = time.perf_counter()
        _ptrace(PTRACE_DETACH, pid, 0, 0)
        _, status = os.waitpid(pid, 0)
        wall = time.perf_counter() - started
    except Exception:
        try:
            os.kill(pid, 9)
        except OSError:
            pass
        raise
    os.lseek(memfd, symbols["acct_energy"], os.SEEK_SET)
    raw = os.read(memfd, 96)
    account.write_bytes(raw)
    os.close(memfd)
    energy, grad, pot = struct.unpack_from("<QQQ", raw, 0)
    best, ref, tol_v, limit_v = struct.unpack_from("<dddd", raw, 24)
    budget_v, seed_v = struct.unpack_from("<QQ", raw, 56)
    hit, stop = struct.unpack_from("<II", raw, 72)
    scale = calibrate()
    code = os.waitstatus_to_exitcode(status) if hasattr(os, "waitstatus_to_exitcode") else (
        os.WEXITSTATUS(status) if os.WIFEXITED(status) else -os.WTERMSIG(status)
    )
    return {
        "exit": code,
        "wall_s": wall,
        "energy_evals": energy,
        "grad_evals": grad,
        "pot_s": pot * scale / 1e9,
        "best": best,
        "hit": hit,
        "stop": stop,
        "budget": budget_v,
        "seed": seed_v,
    }


if __name__ == "__main__":
    config = Path(sys.argv[1])
    account = Path(sys.argv[2])
    seed = int(sys.argv[3])
    budget = int(sys.argv[4])
    reference = float(sys.argv[5])
    row = run(config, account, seed, budget, reference, 0.001)
    print(
        "occat_row "
        + " ".join(f"{key}={value}" for key, value in row.items()),
        flush=True,
    )
