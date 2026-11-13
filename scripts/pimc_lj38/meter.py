# Copyright (c) anneal developers
"""Shared counter for RPQA force calls.

The driver and the i-PI process map the same file. A completed local
optimisation reads the count after the driver has returned the last force
of that relax. Writes are native little-endian integers, one cache line.
"""

from __future__ import annotations

import mmap
import os
import struct

SIZE = 40
# calls u64, pot_ns u64, n_pins u64, hit_calls u64, best f64


class Meter:
    def __init__(self, path: str, create: bool = False):
        flags = os.O_RDWR | (os.O_CREAT if create else 0)
        self.fd = os.open(path, flags, 0o644)
        if create:
            os.ftruncate(self.fd, SIZE)
        self.mm = mmap.mmap(self.fd, SIZE)
        if create:
            self.mm[:] = b"\x00" * SIZE
            struct.pack_into("<d", self.mm, 32, float("inf"))

    def close(self) -> None:
        self.mm.close()
        os.close(self.fd)

    def calls(self) -> int:
        return struct.unpack_from("<Q", self.mm, 0)[0]

    def pot_ns(self) -> int:
        return struct.unpack_from("<Q", self.mm, 8)[0]

    def n_pins(self) -> int:
        return struct.unpack_from("<Q", self.mm, 16)[0]

    def hit_calls(self) -> int:
        return struct.unpack_from("<Q", self.mm, 24)[0]

    def best(self) -> float:
        return struct.unpack_from("<d", self.mm, 32)[0]

    def charge(self, dt_ns: int, budget: int) -> bool:
        """Count one finished evaluation. False means the budget is now full."""
        calls, pot = struct.unpack_from("<QQ", self.mm, 0)
        if calls >= budget:
            return False
        calls += 1
        struct.pack_into("<QQ", self.mm, 0, calls, pot + dt_ns)
        return calls < budget

    def at_budget(self, budget: int) -> bool:
        return self.calls() >= budget

    def note_pin(self, energy_ha: float) -> None:
        n_pins = self.n_pins() + 1
        struct.pack_into("<Q", self.mm, 16, n_pins)
        if energy_ha < self.best():
            struct.pack_into("<d", self.mm, 32, energy_ha)

    def mark_hit(self) -> None:
        struct.pack_into("<Q", self.mm, 24, self.calls())
