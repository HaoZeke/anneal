#!/usr/bin/env python3
"""Launch the historical pyar-tabu script with a campaign seed.

Chakraborty and Anoop call ``pyar-tabu`` and do not pass a seed. That script
draws a fresh Latin hypercube on every composite. Seeding the sampler from
``LJ_BONOBO_SEED``, and advancing it once per construction, keeps those draws
repeatable without changing the tabu filter or the contact placement.
"""

from __future__ import annotations

import os
import runpy
import sys


def main() -> None:
    script = os.environ.get("PYAR_TABU_SCRIPT")
    if not script:
        sys.exit("PYAR_TABU_SCRIPT is not set")
    raw = os.environ.get("LJ_BONOBO_SEED")
    if raw is not None:
        import numpy as np
        from scipy.stats import qmc

        campaign = int(raw)
        np.random.seed(campaign)
        original = qmc.LatinHypercube
        drawn = {"n": 0}

        def latin_hypercube(d, **kwargs):
            kwargs.setdefault("seed", campaign * 1_000_003 + drawn["n"])
            drawn["n"] += 1
            return original(d, **kwargs)

        qmc.LatinHypercube = latin_hypercube
    sys.argv = [script, *sys.argv[1:]]
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
