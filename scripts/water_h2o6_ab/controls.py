#!/usr/bin/env python3
"""Controls for the CREST arm of the water hexamer A/B.

For every run index CREST is run twice from the same start geometry: once
through the call server on the engine of the A/B with the full call budget,
and once on its built-in GFN2 without a budget, stopped as soon as more than
180 frames exist. The first 180 frames of each run are kept. Every set of
frames is scored by both builds against the references in three atom
orderings: as used in the A/B, as found in the original reference files, and
shuffled at random. The single-reference RMSD of every frame is recorded for
the references as used.

The controls do not time anything and run outside the A/B session.
"""
import argparse
import os
import random
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from argparse import Namespace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_ab  # noqa: E402

FRAMES_KEPT = 180
SOURCES = ("driver", "native")
FIELDS = [
    "start", "run", "source", "refs", "code", "frame", "epot_Eh", "intact", "rmsd_A", "isomer", "reason",
    "rmsd_prism_A", "rmsd_cage_A", "rmsd_book_A", "oo_components", "oo_gap_A", "oo_rg_A",
]
SUMMARY_FIELDS = [
    "start", "run", "source", "frames", "refs", "code", "gate_A", "references_loaded", "frames_intact",
    "frames_unknown", "frames_inside_gate", "hit", "min_rmsd_A", "frames_one_cluster",
    "first_frame_components", "crest_status", "crest_wall_s",
]
CONTACT_A = 3.5


def descriptors(frame):
    """Connected components of the oxygens at an O-O contact of 3.5 A, the
    smallest O-O distance between two components, and the radius of gyration
    of the oxygens."""
    points = []
    for line in frame["atoms"]:
        fields = line.split()
        if fields and fields[0].upper() == "O":
            points.append([float(value) for value in fields[1:4]])
    if not points:
        return None, None, None
    points = np.array(points)
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    label = list(range(len(points)))

    def find(i):
        while label[i] != i:
            label[i] = label[label[i]]
            i = label[i]
        return i

    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if distances[i, j] < CONTACT_A:
                label[find(i)] = find(j)
    roots = [find(i) for i in range(len(points))]
    components = len(set(roots))
    gap = 0.0
    if components > 1:
        gap = min(distances[i, j] for i in range(len(points)) for j in range(len(points)) if roots[i] != roots[j])
    radius = float(np.sqrt(((points - points.mean(axis=0)) ** 2).sum(axis=1).mean()))
    return components, gap, radius


def read_atoms(path):
    lines = Path(path).read_text().splitlines()
    count = int(lines[0].split()[0])
    return lines[1], [line.split() for line in lines[2 : 2 + count]]


def write_shuffled(source_dir, target_dir, seed):
    rng = random.Random(seed)
    target_dir.mkdir(parents=True, exist_ok=True)
    for name in run_ab.REFERENCES:
        comment, atoms = read_atoms(source_dir / f"{name}.xyz")
        molecules = [atoms[i : i + 3] for i in range(0, len(atoms), 3)]
        rng.shuffle(molecules)
        out = [str(len(atoms)), comment]
        for molecule in molecules:
            hydrogens = molecule[1:]
            rng.shuffle(hydrogens)
            for atom in [molecule[0], *hydrogens]:
                out.append(" ".join(atom))
        (target_dir / f"{name}.xyz").write_text("\n".join(out) + "\n")


def displaced_start(path, work, sigma, seed):
    comment, atoms = read_atoms(path)
    rng = random.Random(seed)
    lines = [str(len(atoms)), f"{comment} displaced by {sigma} A"]
    for fields in atoms:
        x, y, z = (float(value) + rng.gauss(0.0, sigma) for value in fields[1:4])
        lines.append(f"{fields[0]} {x:.10f} {y:.10f} {z:.10f}")
    target = work / "start-displaced.xyz"
    target.write_text("\n".join(lines) + "\n")
    return target


def random_rotation(rng):
    w, x, y, z = rng.normal(size=4)
    norm = (w * w + x * x + y * y + z * z) ** 0.5
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def packed_start(path, work, radius, seed):
    """Waters of the geometry of the first molecule of the start file, in
    random orientations with their centres of mass inside a sphere of the given
    radius. No two oxygens are closer than 2.4 A, no hydrogen is closer than
    1.5 A to the oxygen of another molecule and no two hydrogens of different
    molecules are closer than 1.2 A."""
    comment, atoms = read_atoms(path)
    count = len(atoms) // 3
    template = np.array([[float(value) for value in fields[1:4]] for fields in atoms[:3]])
    template -= template.mean(axis=0)
    rng = np.random.default_rng(seed)
    placed = []
    for _ in range(200000):
        if len(placed) == count:
            break
        direction = rng.normal(size=3)
        centre = direction / np.linalg.norm(direction) * radius * rng.random() ** (1.0 / 3.0)
        molecule = template @ random_rotation(rng).T + centre
        clash = False
        for other in placed:
            if np.linalg.norm(molecule[0] - other[0]) < 2.4:
                clash = True
            for i in (1, 2):
                if np.linalg.norm(molecule[i] - other[0]) < 1.5 or np.linalg.norm(molecule[0] - other[i]) < 1.5:
                    clash = True
                for j in (1, 2):
                    if np.linalg.norm(molecule[i] - other[j]) < 1.2:
                        clash = True
        if not clash:
            placed.append(molecule)
    if len(placed) != count:
        raise RuntimeError("the waters could not be packed")
    lines = [str(len(atoms)), f"{count} waters packed inside a sphere of radius {radius} A, seed {seed}"]
    for index, molecule in enumerate(placed):
        for offset, point in enumerate(molecule):
            lines.append(f"{atoms[3 * index + offset][0]} {point[0]:.10f} {point[1]:.10f} {point[2]:.10f}")
    target = work / "start-packed.xyz"
    target.write_text("\n".join(lines) + "\n")
    return target


def classify_with_notes(ctx, code, frames_dir, references, tag, work):
    stderr_path = work / f"classify-{tag}.err"
    rows, gate = run_ab.classify(ctx, code, frames_dir, references, tag, work)
    loaded = len(run_ab.REFERENCES)
    if stderr_path.exists():
        loaded -= sum(1 for line in stderr_path.read_text().splitlines() if "failed the monomer check" in line)
    return rows, gate, loaded


def native_frames(ctx, work, timeout):
    directory = work / "native"
    directory.mkdir()
    shutil.copy(ctx.water, directory / "start.xyz")
    paths = {"trj": directory / "crest_dynamics.trj", "mdfiles": directory / "MDFILES"}
    with open(work / "native.out", "wb") as out, open(work / "native.err", "wb") as err:
        begin = time.monotonic()
        proc = subprocess.Popen(
            [str(ctx.crest), "start.xyz", "--gfn2", "-T", "1"], env=run_ab.base_environment(),
            cwd=directory, stdout=out, stderr=err, start_new_session=True,
            preexec_fn=run_ab.limit_file_size,
        )
        state = "running"
        while state == "running":
            try:
                proc.wait(timeout=1.0)
                state = str(proc.returncode)
            except subprocess.TimeoutExpired:
                if len(run_ab.collect_frames(paths)) > FRAMES_KEPT:
                    state = "frames"
                elif time.monotonic() - begin > timeout:
                    state = "timeout"
        wall = time.monotonic() - begin
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    proc.wait()
    return run_ab.collect_frames(paths)[:FRAMES_KEPT], state, wall


def driver_frames(ctx, work):
    result = run_ab.crest_run(ctx, ctx.args.budget, work)
    frames = run_ab.collect_frames(result)[:FRAMES_KEPT]
    return frames, ("timeout" if result["timed_out"] else str(result["status"])), result["wall"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("binary-a", "binary-b", "engine", "shim", "driver", "crest", "water-xyz",
                 "references-dir", "original-references-dir", "out-dir", "scratch-dir"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--core", type=int, required=True)
    parser.add_argument("--start-label", required=True, help="name of the start geometry in the output")
    parser.add_argument("--runs", required=True, help="run indices, for example 0-23")
    parser.add_argument("--budget", type=int, default=4000)
    parser.add_argument("--shuffle-seed", type=int, default=7)
    parser.add_argument("--sources", default=",".join(SOURCES), help="driver, native or both")
    parser.add_argument("--reference-variants", default="used,original,shuffled")
    parser.add_argument("--native-timeout", type=float, default=1500)
    parser.add_argument("--start-noise", type=float, default=0.0,
                        help="standard deviation in angstrom of a random displacement of every start coordinate")
    parser.add_argument("--pack-radius", type=float, default=0.0,
                        help="radius in angstrom of a sphere for random packings of the waters; replaces the start file")
    parser.add_argument("--noise-seed", type=int, default=1000)
    args = parser.parse_args()

    ctx = run_ab.Context(Namespace(
        binary_a=args.binary_a, binary_b=args.binary_b, engine=args.engine, shim=args.shim,
        driver=args.driver, crest=args.crest, water_xyz=args.water_xyz,
        references_dir=args.references_dir, out_dir=args.out_dir, scratch_dir=args.scratch_dir,
        budget=args.budget, crest_timeout=900, classify_timeout=300,
    ))
    run_ab.os.sched_setaffinity(0, {args.core})
    ctx.out.mkdir(parents=True, exist_ok=True)
    ctx.scratch.mkdir(parents=True, exist_ok=True)

    available = {"used": ctx.references, "original": Path(args.original_references_dir).resolve()}
    shuffled = ctx.scratch / "references-shuffled"
    write_shuffled(ctx.references, shuffled, args.shuffle_seed)
    available["shuffled"] = shuffled
    variants = {name: available[name] for name in args.reference_variants.split(",") if name}
    sources = [name for name in args.sources.split(",") if name]
    single = {}
    for name in run_ab.REFERENCES:
        directory = ctx.scratch / f"single-{name}"
        directory.mkdir(exist_ok=True)
        shutil.copy(ctx.references / f"{name}.xyz", directory / f"{name}.xyz")
        single[name] = directory

    frames_handle, frames_writer = run_ab.open_writer(ctx.out / "controls_frames.csv", FIELDS)
    summary_handle, summary_writer = run_ab.open_writer(ctx.out / "controls_runs.csv", SUMMARY_FIELDS)
    base_start = ctx.water
    started = time.time()
    for run in run_ab.parse_seed_list(args.runs):
        for source in sources:
            work = Path(tempfile.mkdtemp(dir=ctx.scratch, prefix="control-"))
            try:
                if args.start_noise > 0:
                    ctx.water = displaced_start(base_start, work, args.start_noise, args.noise_seed + run)
                elif args.pack_radius > 0:
                    ctx.water = packed_start(base_start, work, args.pack_radius, args.noise_seed + run)
                if source == "driver":
                    frames, status, wall = driver_frames(ctx, work)
                else:
                    frames, status, wall = native_frames(ctx, work, args.native_timeout)
                frames_dir = work / "frames"
                run_ab.write_frames(frames, frames_dir)
                shapes = [descriptors(frame) for frame in frames]
                per_reference = {
                    name: run_ab.classify(ctx, "A", frames_dir, single[name], f"single-{name}", work)[0]
                    for name in run_ab.REFERENCES
                }
                for label, references in variants.items():
                    for code in run_ab.CODES:
                        try:
                            rows, gate, loaded = classify_with_notes(
                                ctx, code, frames_dir, references, f"{label}-{code}", work)
                        except RuntimeError:
                            summary_writer.writerow({
                                "start": args.start_label,
                                "run": run, "source": source, "frames": len(frames), "refs": label,
                                "code": code, "references_loaded": "classification failed",
                                "crest_status": status, "crest_wall_s": round(wall, 3),
                            })
                            continue
                        inside = 0
                        rmsds = []
                        for index, frame in enumerate(frames):
                            name = f"frame-{index:04d}.xyz"
                            row = rows.get(name)
                            if row is None:
                                continue
                            hit = row["isomer"] != "unassigned"
                            inside += int(hit)
                            if row["rmsd"] is not None:
                                rmsds.append(row["rmsd"])
                            record = {
                                "start": args.start_label,
                                "run": run, "source": source, "refs": label, "code": code, "frame": index,
                                "epot_Eh": "" if frame["epot"] is None else f"{frame['epot']:.9f}",
                                "intact": row["intact"],
                                "rmsd_A": "" if row["rmsd"] is None else f"{row['rmsd']:.5f}",
                                "isomer": "-" if not hit else row["isomer"],
                                "reason": row["reason"],
                                "oo_components": "" if shapes[index][0] is None else shapes[index][0],
                                "oo_gap_A": "" if shapes[index][1] is None else f"{shapes[index][1]:.3f}",
                                "oo_rg_A": "" if shapes[index][2] is None else f"{shapes[index][2]:.3f}",
                            }
                            for ref in run_ab.REFERENCES:
                                other = per_reference[ref].get(name)
                                record[f"rmsd_{ref}_A"] = (
                                    "" if label != "used" or code != "A" or other is None or other["rmsd"] is None
                                    else f"{other['rmsd']:.5f}"
                                )
                            frames_writer.writerow(record)
                        summary_writer.writerow({
                            "start": args.start_label,
                            "run": run, "source": source, "frames": len(frames), "refs": label, "code": code,
                            "gate_A": "" if gate is None else gate, "references_loaded": loaded,
                            "frames_intact": sum(1 for r in rows.values() if r["intact"] == "true"),
                            "frames_unknown": sum(1 for r in rows.values() if r["intact"] == "unknown"),
                            "frames_inside_gate": inside, "hit": int(inside > 0),
                            "min_rmsd_A": "" if not rmsds else f"{min(rmsds):.6f}",
                            "frames_one_cluster": sum(1 for s in shapes if s[0] == 1),
                            "first_frame_components": "" if not shapes or shapes[0][0] is None else shapes[0][0],
                            "crest_status": status, "crest_wall_s": round(wall, 3),
                        })
                frames_handle.flush()
                summary_handle.flush()
                print(f"run {run} {source} frames={len(frames)} status={status} "
                      f"elapsed={time.time() - started:.0f}s", flush=True)
            except Exception as error:
                print(f"run {run} {source} failed: {type(error).__name__}: {error}", flush=True)
            finally:
                shutil.rmtree(work, ignore_errors=True)
    frames_handle.close()
    summary_handle.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
