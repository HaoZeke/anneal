#!/usr/bin/env python3
# Copyright (c) anneal developers
"""One i-PI process running the published RPQA LJ input.

After every completed ``rpqa`` relaxation the best bead energy is recorded.
A bead at or below the Cambridge minimum plus 0.001, in the driver's
epsilon, ends the process. The dynamics and the optimiser are the ones in
afiorent/i-pi ``4e86f5fd``.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np

from meter import Meter

for _candidate in (Path(__file__).resolve().parent, Path(__file__).resolve().parents[1]):
    if (_candidate / "target_tol.py").is_file() and str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))
        break
from target_tol import energy_hits  # noqa: E402


def main() -> None:
    ipi_root = os.environ["IPI_ROOT"]
    sys.path.insert(0, ipi_root)
    level_ha = float(os.environ["RPQA_LEVEL_HA"])
    meter = Meter(os.environ["RPQA_METER"])

    import ipi.engine.smotion.rpqa as rpqa_mod
    from ipi.utils.depend import dstrip

    original = rpqa_mod.RPQA._relax

    def relax(self, isys):
        nrelax = original(self, isys)
        pots = dstrip(self.syslist[isys].forces.pots)
        best = float(np.min(pots))
        meter.note_pin(best)
        if energy_hits(best, level_ha, 0.0):
            meter.mark_hit()
            os._exit(0)
        return nrelax

    rpqa_mod.RPQA._relax = relax

    from ipi.engine.simulation import Simulation

    simulation = Simulation.load_from_xml(
        open(sys.argv[1]),
        request_banner=False,
        custom_verbosity="low",
        sockets_prefix="/tmp/ipi_",
    )
    simulation.run()


if __name__ == "__main__":
    main()
