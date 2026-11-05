#!/usr/bin/env python3
"""Record a hardware name for airspeed velocity.

The name is not the host name. The CPU string drops the registered marks.
"""

from __future__ import annotations

from asv.machine import Machine, MachineCollection


def main() -> None:
    info = Machine.get_defaults()
    info["machine"] = "scaling-host"
    cpu = info["cpu"].replace("(R)", "").replace("(TM)", "")
    info["cpu"] = " ".join(cpu.split())
    MachineCollection.save("scaling-host", info)


if __name__ == "__main__":
    main()
