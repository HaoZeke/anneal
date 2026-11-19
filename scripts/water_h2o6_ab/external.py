#!/usr/bin/env python3
"""Frame-level CSV files of the water hexamer A/B that are kept outside git.

The per-frame tables are large. They are published under the name
<sha256>-<basename> and listed in external_manifest.csv with their relative
path, sha256, size in bytes, number of data rows and a one-line description.
The analysis scripts read them through the manifest: a file is looked up in a
local directory, its sha256 and size are checked against the manifest, and it
is refused when they differ.

Subcommands:

  build   write a manifest for the frame files found under a results directory
  stage   copy the files listed in a manifest from a results directory into a
          directory under their published names, with a SHA256SUMS file
  verify  check every file of a manifest in a local directory
"""
import argparse
import csv
import hashlib
import io
import sys
from pathlib import Path

COLUMNS = ("path", "sha256", "bytes", "rows", "description")
FRAME_FILES = ("crest_frames.csv", "controls_frames.csv")


class ExternalError(Exception):
    pass


def digest(data):
    return hashlib.sha256(data).hexdigest()


def parse(data):
    return list(csv.DictReader(io.StringIO(data.decode("utf-8"), newline="")))


def read_manifest(path):
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or tuple(rows[0].keys()) != COLUMNS:
        raise ExternalError(f"{path}: expected the columns {','.join(COLUMNS)}")
    return {row["path"]: row for row in rows}


def stored_name(entry):
    return f"{entry['sha256']}-{Path(entry['path']).name}"


def describe(relative):
    path = Path(relative)
    if path.name == "crest_frames.csv":
        return ("one row per CREST frame of the crest and crest_fair arms, with the verdicts of both builds "
                "and the RMSD against each reference")
    if path.name == "controls_frames.csv":
        return (f"one row per CREST frame scored in the control {path.parent.name}, with the verdicts of both "
                "builds for every reference ordering")
    return "per-frame table"


def locate(entry, directory):
    directory = Path(directory)
    for candidate in (directory / stored_name(entry), directory / entry["path"]):
        if candidate.is_file():
            return candidate
    raise ExternalError(f"{entry['path']}: no file named {stored_name(entry)} in {directory}")


def read_verified(entry, directory):
    """Bytes of the file of a manifest entry in a directory, checked against the manifest."""
    path = locate(entry, directory)
    data = path.read_bytes()
    if len(data) != int(entry["bytes"]):
        raise ExternalError(f"{entry['path']}: {len(data)} bytes, the manifest has {entry['bytes']}")
    if digest(data) != entry["sha256"]:
        raise ExternalError(f"{entry['path']}: sha256 differs from the manifest")
    return data


def load_rows(manifest, relative, directory):
    """Rows of a listed frame file, read from a local directory after the sha256 check."""
    if relative not in manifest:
        raise ExternalError(f"{relative} is not listed in the manifest")
    entry = manifest[relative]
    rows = parse(read_verified(entry, directory))
    if len(rows) != int(entry["rows"]):
        raise ExternalError(f"{relative}: {len(rows)} rows, the manifest has {entry['rows']}")
    return rows


def find_frame_files(data_dir):
    data_dir = Path(data_dir)
    found = [p for p in data_dir.rglob("*.csv") if p.name in FRAME_FILES]
    return sorted(p.relative_to(data_dir).as_posix() for p in found)


def entries_for(data_dir, relatives):
    entries = []
    for relative in relatives:
        data = (Path(data_dir) / relative).read_bytes()
        entries.append({
            "path": relative, "sha256": digest(data), "bytes": len(data),
            "rows": len(parse(data)), "description": describe(relative),
        })
    return entries


def write_manifest(path, entries):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(entries)


def command_build(args):
    relatives = find_frame_files(args.data_dir)
    if not relatives:
        raise ExternalError(f"no frame files under {args.data_dir}")
    write_manifest(args.manifest, entries_for(args.data_dir, relatives))
    print(f"{len(relatives)} files listed")


def command_stage(args):
    manifest = read_manifest(args.manifest)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sums = set()
    for relative, entry in sorted(manifest.items()):
        data = read_verified(entry, args.data_dir)
        (out / stored_name(entry)).write_bytes(data)
        sums.add(f"{entry['sha256']}  {stored_name(entry)}\n")
    (out / "SHA256SUMS").write_text("".join(sorted(sums)))
    print(f"{len(sums)} files staged")


def command_verify(args):
    manifest = read_manifest(args.manifest)
    for relative in sorted(manifest):
        load_rows(manifest, relative, args.directory)
    print(f"{len(manifest)} files verified")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--data-dir", required=True)
    build.add_argument("--manifest", required=True)
    build.set_defaults(run=command_build)
    stage = sub.add_parser("stage")
    stage.add_argument("--manifest", required=True)
    stage.add_argument("--data-dir", required=True)
    stage.add_argument("--out-dir", required=True)
    stage.set_defaults(run=command_stage)
    verify = sub.add_parser("verify")
    verify.add_argument("--manifest", required=True)
    verify.add_argument("--directory", required=True)
    verify.set_defaults(run=command_verify)
    args = parser.parse_args()
    try:
        args.run(args)
    except ExternalError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
