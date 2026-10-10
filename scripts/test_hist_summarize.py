"""A missing or failed task is not a search miss in paired comparisons."""

import contextlib
import io
from pathlib import Path
import tempfile
import unittest

from hist_summarize import compare, summarise


def result(seed, *, solved=True, finished=True):
    marker = " SOLVED" if solved else ""
    footer = "gap to reference 0.0\n" if finished else ""
    return (
        f"  seed {seed} ensemble: best -397.492331 aggregate charged 1234 "
        f"first_target_calls 1234 wall 2.0s{marker}\n{footer}"
    )


class CampaignCompletionTests(unittest.TestCase):
    def test_truncated_target_record_is_not_a_finished_search(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "control_0.out").write_text(result(0, finished=False))
            record = summarise(directory)["control"]
        self.assertEqual(record["tasks"], 1)
        self.assertEqual(record["done"], 0)
        self.assertEqual(record["solved"], [])
        self.assertEqual(record["wall"], [])

    def test_nonzero_exit_is_not_accepted_even_with_a_footer(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "control_0.out").write_text(result(0))
            Path(directory, "control_0.exitcode").write_text("137\n")
            record = summarise(directory)["control"]
        self.assertEqual(record["tasks"], 1)
        self.assertEqual(record["done"], 0)
        self.assertEqual(record["solved"], [])

    def test_paired_comparison_uses_jointly_finished_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            for arm, seed, solved in [
                ("control", 0, True),
                ("control", 1, False),
                ("control", 2, True),
                ("treatment", 1, True),
                ("treatment", 2, False),
                ("treatment", 3, True),
            ]:
                Path(directory, f"{arm}_{seed}.out").write_text(
                    result(seed, solved=solved)
                )
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                compare(summarise(directory), "control", "treatment")
        self.assertIn("paired seeds 2", output.getvalue())
        self.assertIn("1 vs 1 solved", output.getvalue())
        self.assertIn("gained 1 [1]; lost 1 [2]", output.getvalue())
        self.assertIn("sign test p=1.000", output.getvalue())

    def test_zero_exit_and_footer_preserve_a_valid_result(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "control_3.out").write_text(result(3))
            Path(directory, "control_3.exitcode").write_text("0\n")
            record = summarise(directory)["control"]
        self.assertEqual(record["done"], 1)
        self.assertEqual(record["solved"], [3])
        self.assertEqual(record["first"], [1234.0])
        self.assertEqual(record["wall"], [2.0])

    def test_managed_task_requires_its_terminal_exit_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "control_0.out").write_text(result(0))
            Path(directory, "control_0.meta").write_text("task=0 slurm_job=123\n")
            record = summarise(directory)["control"]
        self.assertEqual(record["tasks"], 1)
        self.assertEqual(record["done"], 0)
        self.assertEqual(record["solved"], [])

    def test_footer_without_a_matching_seed_result_is_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "control_0.out").write_text(result(7))
            record = summarise(directory)["control"]
        self.assertEqual(record["tasks"], 1)
        self.assertEqual(record["done"], 0)
        self.assertEqual(record["solved"], [])


if __name__ == "__main__":
    unittest.main()
