# Copyright (c) anneal developers
"""One matched GFN2 water seed against CREST.

The clock is the serial arm in ``sudo unshare --uts`` with hostname
bench. Rows go to a scratch directory, so a timing does not append to
``bench/water-gfn2-scc2``.
"""

from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[3]
ENTER = ROOT / "scripts" / "water_gfn2_scc2" / "enter_bench.sh"


class WaterGfn2Scc2:
    params = ([6, 8, 12],)
    param_names = ["molecules"]
    timeout = 86400

    def setup(self, molecules):
        if not ENTER.is_file():
            raise RuntimeError(f"missing {ENTER}")
        self.scratch = Path(tempfile.mkdtemp(prefix=f"water-gfn2-scc2-{molecules}-"))

    def time_seed(self, molecules):
        subprocess.run(
            [
                "bash",
                str(ENTER),
                "--crest-only",
                "--size",
                str(molecules),
                "--seed",
                "0",
                "--runs",
                str(self.scratch / "runs"),
                "--out",
                str(self.scratch / "out"),
            ],
            check=True,
        )
