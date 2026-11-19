"""Tests of the manifest of the frame tables kept outside git.

Run from this directory with: python -m unittest test_external
"""
import contextlib
import csv
import hashlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import analyze_controls
import external

RUN_FIELDS = ["start", "source", "refs", "code", "run", "hit", "frames", "references_loaded", "gate_A",
              "frames_inside_gate", "frames_unknown", "frames_one_cluster", "first_frame_components",
              "min_rmsd_A", "crest_wall_s"]
FRAME_FIELDS = ["start", "source", "refs", "run", "code", "frame", "isomer", "rmsd_A"]


def write_rows(path, fields, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def toy_tree(root):
    """A results directory with one control that has runs and frames."""
    directory = root / "controls" / "toy"
    directory.mkdir(parents=True)
    runs, frames = [], []
    for code in ("A", "B"):
        runs.append({"start": "toy", "source": "driver", "refs": "used", "code": code, "run": 0, "hit": 1,
                     "frames": 2, "references_loaded": 3, "gate_A": "0.598496", "frames_inside_gate": 1,
                     "frames_unknown": 0, "frames_one_cluster": 2, "first_frame_components": 1,
                     "min_rmsd_A": "0.4", "crest_wall_s": "12.5"})
        for frame, (isomer, rmsd) in enumerate((("prism", "0.4"), ("-", "1.7"))):
            frames.append({"start": "toy", "source": "driver", "refs": "used", "run": 0, "code": code,
                           "frame": frame, "isomer": isomer, "rmsd_A": rmsd})
    write_rows(directory / "controls_runs.csv", RUN_FIELDS, runs)
    write_rows(directory / "controls_frames.csv", FRAME_FIELDS, frames)
    return directory


def run_controls(arguments):
    out = io.StringIO()
    with mock.patch.object(sys, "argv", ["analyze_controls.py", *arguments]), contextlib.redirect_stdout(out):
        status = analyze_controls.main()
    return status, out.getvalue()


class ManifestTest(unittest.TestCase):
    def setUp(self):
        holder = tempfile.TemporaryDirectory()
        self.addCleanup(holder.cleanup)
        self.root = Path(holder.name).resolve()
        self.data = self.root / "data"
        self.control = toy_tree(self.data)
        self.manifest_path = self.data / "external_manifest.csv"
        external.write_manifest(self.manifest_path, external.entries_for(self.data, external.find_frame_files(self.data)))
        self.store = self.root / "store"
        args = mock.Mock(manifest=self.manifest_path, data_dir=self.data, out_dir=self.store)
        with contextlib.redirect_stdout(io.StringIO()):
            external.command_stage(args)
        self.frames_path = self.control / "controls_frames.csv"
        self.name = "controls/toy/controls_frames.csv"

    def test_manifest_describes_the_file(self):
        manifest = external.read_manifest(self.manifest_path)
        self.assertEqual(list(manifest), [self.name])
        entry = manifest[self.name]
        data = self.frames_path.read_bytes()
        self.assertEqual(entry["sha256"], hashlib.sha256(data).hexdigest())
        self.assertEqual(int(entry["bytes"]), len(data))
        self.assertEqual(int(entry["rows"]), 4)
        self.assertTrue(entry["description"])
        self.assertNotIn("\r", self.manifest_path.read_text())

    def test_store_holds_files_under_their_hash(self):
        entry = external.read_manifest(self.manifest_path)[self.name]
        self.assertEqual(external.stored_name(entry), f"{entry['sha256']}-controls_frames.csv")
        self.assertTrue((self.store / external.stored_name(entry)).is_file())
        sums = (self.store / "SHA256SUMS").read_text().splitlines()
        self.assertEqual(sums, [f"{entry['sha256']}  {external.stored_name(entry)}"])

    def test_rows_are_loaded_after_the_check(self):
        manifest = external.read_manifest(self.manifest_path)
        rows = external.load_rows(manifest, self.name, self.store)
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[0]["isomer"], "prism")

    def test_changed_file_is_refused(self):
        manifest = external.read_manifest(self.manifest_path)
        entry = manifest[self.name]
        stored = self.store / external.stored_name(entry)
        data = stored.read_bytes()
        stored.write_bytes(data.replace(b"prism", b"cages"))
        with self.assertRaisesRegex(external.ExternalError, "sha256"):
            external.load_rows(manifest, self.name, self.store)
        stored.write_bytes(data + b"\n")
        with self.assertRaisesRegex(external.ExternalError, "bytes"):
            external.load_rows(manifest, self.name, self.store)

    def test_missing_and_unlisted_files_are_refused(self):
        manifest = external.read_manifest(self.manifest_path)
        with self.assertRaisesRegex(external.ExternalError, "no file named"):
            external.load_rows(manifest, self.name, self.root / "empty")
        with self.assertRaisesRegex(external.ExternalError, "not listed"):
            external.load_rows(manifest, "controls/other/controls_frames.csv", self.store)

    def test_wrong_row_count_in_the_manifest_is_refused(self):
        manifest = external.read_manifest(self.manifest_path)
        manifest[self.name]["rows"] = "5"
        with self.assertRaisesRegex(external.ExternalError, "rows"):
            external.load_rows(manifest, self.name, self.store)

    def test_file_in_its_relative_place_is_accepted(self):
        manifest = external.read_manifest(self.manifest_path)
        self.assertEqual(len(external.load_rows(manifest, self.name, self.data)), 4)

    def test_tables_regenerate_from_the_store(self):
        status, in_place = run_controls([str(self.control)])
        self.assertEqual(status, 0)
        self.frames_path.unlink()
        status, through_manifest = run_controls(
            ["--manifest", str(self.manifest_path), "--external-dir", str(self.store), str(self.control)])
        self.assertEqual(status, 0)
        self.assertEqual(through_manifest, in_place)
        self.assertIn("Build A against build B on the same frames", through_manifest)

    def test_tables_are_not_written_from_a_changed_file(self):
        entry = external.read_manifest(self.manifest_path)[self.name]
        stored = self.store / external.stored_name(entry)
        stored.write_bytes(stored.read_bytes().replace(b"prism", b"cages"))
        self.frames_path.unlink()
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            status, output = run_controls(
                ["--manifest", str(self.manifest_path), "--external-dir", str(self.store), str(self.control)])
        self.assertEqual(status, 1)
        self.assertEqual(output, "")
        self.assertIn("sha256", err.getvalue())

    def test_verify_checks_every_file(self):
        args = mock.Mock(manifest=self.manifest_path, directory=self.store)
        with contextlib.redirect_stdout(io.StringIO()) as out:
            external.command_verify(args)
        self.assertIn("1 files verified", out.getvalue())


if __name__ == "__main__":
    unittest.main()
