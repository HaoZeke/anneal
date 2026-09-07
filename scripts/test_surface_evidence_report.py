"""Small fixture checks for evaluation-matched surface comparisons."""

import tempfile
import unittest
from pathlib import Path

from surface_evidence_report import summarize


class SurfaceEvidenceReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        (self.root / "manifest.txt").write_text(
            "source=5223a998\nfeatures=bank-rpc,ira,featomic\nn=13\n"
            "budget=1000\nreplicas=2\nensembles=1\nblock=5\n"
            "options=catalog,surfaces,noclimb\nchannels=surface-evidence-only\n"
        )
        for mode in ("private", "shared"):
            for replica in range(2):
                peer = 0 if mode == "private" else 2
                self.worker(mode, replica).write_text(
                    "LJ13, budget 1000 charged evaluations, 1 seeds, reference -44.326801\n"
                    "  catalog channels: surface evidence only; geometry policy disabled\n"
                    "  policy: leaves 0 other 0 walk 0 hole 0 refused 0\n"
                    '{"kind":"local_work","aggregate_charged":1000}\n'
                    f"SURFACE_EVIDENCE seed {replica} local_blocks 2 peer_blocks {peer} "
                    "local_draws [1, 1] local_means [0.0, -0.1]\n"
                    f"  seed {replica}: best {-40 - replica:.6f}  hops 10  charged 1000\n"
                )

    def worker(self, mode, replica):
        return self.root / f"{mode}-0-{replica}.log"

    def replace(self, mode, replica, old, new):
        path = self.worker(mode, replica)
        path.write_text(path.read_text().replace(old, new))

    def test_reports_independent_ensemble_minima_at_aggregate_work(self):
        result = summarize(self.root)
        self.assertEqual(result["ensemble_budget"], 2000)
        self.assertEqual(
            result["ensemble_best"], {"private": [-41.0], "shared": [-41.0]}
        )
        self.assertEqual(result["shared_peer_blocks"], 4)

    def test_rejects_incomplete_worker(self):
        self.replace("shared", 1, "  seed 1: best", "  interrupted 1: best")
        with self.assertRaises(ValueError):
            summarize(self.root)

    def test_rejects_uncharged_terminal_work(self):
        self.replace("shared", 0, '"aggregate_charged":1000', '"aggregate_charged":999')
        with self.assertRaises(ValueError):
            summarize(self.root)

    def test_catalog_snapshots_use_the_ensemble_budget(self):
        self.replace(
            "shared", 0,
            '{"kind":"local_work","aggregate_charged":1000}',
            '{"kind":"population_pending","aggregate_charged":2000}\n'
            '{"kind":"local_work","aggregate_charged":1000}',
        )
        self.assertEqual(summarize(self.root)["ensemble_budget"], 2000)

    def test_rejects_catalog_work_above_the_ensemble_budget(self):
        self.replace(
            "shared", 0,
            '{"kind":"local_work","aggregate_charged":1000}',
            '{"kind":"population_pending","aggregate_charged":2001}\n'
            '{"kind":"local_work","aggregate_charged":1000}',
        )
        with self.assertRaises(ValueError):
            summarize(self.root)

    def test_rejects_geometry_policy_actions(self):
        self.replace("shared", 0, "leaves 0", "leaves 1")
        with self.assertRaises(ValueError):
            summarize(self.root)

    def test_rejects_a_shared_arm_without_peer_evidence(self):
        for replica in range(2):
            self.replace("shared", replica, "peer_blocks 2", "peer_blocks 0")
        with self.assertRaises(ValueError):
            summarize(self.root)

    def test_rejects_mismatched_seeds(self):
        self.replace("shared", 1, "seed 1", "seed 7")
        with self.assertRaises(ValueError):
            summarize(self.root)

    def test_rejects_nonfinite_surface_rewards(self):
        self.replace("shared", 0, "[0.0, -0.1]", "[NaN, -0.1]")
        with self.assertRaises(ValueError):
            summarize(self.root)


if __name__ == "__main__":
    unittest.main()
