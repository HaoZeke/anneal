"""The documentation agrees with the source.

Runs under pytest and under ``python -m unittest``. Set ``RGPOT_DOC_BUILD_DIR``
to a configured and compiled build directory to also run the examples.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "docs"))

import doccheck  # noqa: E402


def _report(errors: list[str]) -> str:
    return "\n" + "\n".join(errors)


class DocsContract(unittest.TestCase):
    def test_every_public_name_has_a_reference_entry(self):
        errors = doccheck.reference_errors(ROOT)
        self.assertEqual(errors, [], _report(errors))

    def test_statements_read_off_the_source(self):
        errors = doccheck.facts_errors(ROOT)
        self.assertEqual(errors, [], _report(errors))

    def test_every_code_block_is_classified_and_checked(self):
        errors = doccheck.check_blocks_static(ROOT)
        self.assertEqual(errors, [], _report(errors))

    @unittest.skipUnless(
        os.environ.get("RGPOT_DOC_BUILD_DIR"), "RGPOT_DOC_BUILD_DIR is not set"
    )
    def test_examples_run_against_the_build(self):
        build = Path(os.environ["RGPOT_DOC_BUILD_DIR"])
        failures = [
            f"{block.where()}: {detail}"
            for block, status, detail in doccheck.execute_blocks(build, ROOT)
            if status == "fail"
        ]
        self.assertEqual(failures, [], _report(failures))


if __name__ == "__main__":
    unittest.main()
