# Copyright (c) anneal developers
"""Measure the landfold survey table.

LJ38 uses the how-to book: 2000 minima, seed 38, the fcc and icosahedral
fixtures. LJ75 and LJ98 use the same pattern, seed equal to the size, and
the committed minimum fixtures. Embeddings use seed 0. Trustworthiness
and continuity are Venna and Kaski at ten neighbors. Kruskal stress-1 and
the sketch-map stress use the scale of the map that minimizes each one.
The sketch-map switch is the landfold switch at the sigma that run prints.
Runtimes are the median wall time of three runs.

The reference row is lab-cosmo/sketchmap at c2871db187f4a60377e7727af781da6c6c3426cb.
``data/landfold_survey.csv`` is the table. The survey page is rewritten
from it.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
from sklearn.manifold import MDS, TSNE, Isomap
from sklearn.metrics import pairwise_distances

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = ROOT / "build" / "landfold-survey"
OUT = ROOT / "data" / "landfold_survey.csv"
PAGE = ROOT / "docs" / "orgmode" / "explanation" / "landfold-survey.org"
BOOKS = (
    ("lj38", "38", "38", ["tests/fixtures/lj38_fcc.xyz", "tests/fixtures/lj38_ico.xyz"]),
    ("lj75", "75", "75", ["tests/fixtures/lj75_ico.xyz", "tests/fixtures/lj75_marks.xyz"]),
    ("lj98", "98", "98", ["tests/fixtures/lj98_gm.xyz"]),
)
COUNT = "2000"
NEIGHBORS = 10
REPEATS = 3


def cargo_target() -> Path:
    meta = json.loads(
        subprocess.check_output(
            ["rustup", "run", "stable", "cargo", "metadata", "--format-version", "1", "--no-deps"],
            cwd=ROOT,
        )
    )
    return Path(meta["target_directory"]) / "release"


def ensure_book(target: Path, name: str, size: str, seed: str, fixtures: list[str]) -> Path:
    path = SCRATCH / f"{name}.min"
    if path.is_file() and sum(1 for line in path.read_text().splitlines() if line.strip()) >= int(COUNT):
        return path
    SCRATCH.mkdir(parents=True, exist_ok=True)
    cmd = [str(target / "examples" / "landfold_lj_book"), size, COUNT, seed, str(path), *fixtures]
    subprocess.check_call(cmd, cwd=ROOT)
    return path


def anneal(target: Path, args: list[str]) -> tuple[str, str, float]:
    started = time.perf_counter()
    done = subprocess.run(
        [str(target / "anneal"), "landfold", *args],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return done.stdout, done.stderr, time.perf_counter() - started


def load_descriptors(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",")


def load_map(path: Path) -> np.ndarray:
    rows = np.loadtxt(path, delimiter=",", skiprows=1)
    return rows[:, 2:4]


def median_time(fn) -> float:
    samples = [fn() for _ in range(REPEATS)]
    samples.sort()
    return samples[REPEATS // 2]


def upper(matrix: np.ndarray) -> np.ndarray:
    return matrix[np.triu_indices(matrix.shape[0], k=1)]


def trustworthiness(high: np.ndarray, low: np.ndarray, k: int = NEIGHBORS) -> float:
    """Venna and Kaski. Ranks are 1 for the nearest other point.

    ``high`` supplies the ranks. ``low`` supplies the k neighbors being checked.
    """
    n = high.shape[0]
    high = high.copy()
    low = low.copy()
    np.fill_diagonal(high, np.inf)
    np.fill_diagonal(low, np.inf)
    order = np.argsort(high, axis=1, kind="mergesort")[:, :-1]
    high_rank = np.empty((n, n), dtype=int)
    np.put_along_axis(high_rank, order, np.arange(1, n), axis=1)
    low_neighbors = np.argsort(low, axis=1, kind="mergesort")[:, :k]
    penalty = 0
    for i in range(n):
        for j in low_neighbors[i]:
            rank = int(high_rank[i, j])
            if rank > k:
                penalty += rank - k
    scale = n * k * (2 * n - 3 * k - 1)
    return 1.0 - 2.0 * penalty / scale


def switch(distance: np.ndarray, sigma: float, a: float = 2.0, b: float = 4.0) -> np.ndarray:
    ratio = np.maximum(distance, 0.0) / sigma
    lifted = (2.0 ** (a / b) - 1.0) * ratio**a
    return 1.0 - (1.0 + lifted) ** (-b / a)


def kruskal(high: np.ndarray, low: np.ndarray) -> float:
    scale = np.dot(high, low) / np.dot(low, low)
    residual = high - scale * low
    return float(np.sqrt(np.dot(residual, residual) / np.dot(high, high)))


def sketch_stress(high: np.ndarray, low: np.ndarray, sigma: float) -> float:
    from scipy.optimize import minimize_scalar

    switched = switch(high, sigma)

    def loss(scale: float) -> float:
        gap = switched - switch(low * scale, sigma)
        return float(np.dot(gap, gap) / gap.size)

    found = minimize_scalar(loss, bounds=(1.0e-4, 1.0e3), method="bounded")
    return float(found.fun)


def pair_metrics(descriptors: np.ndarray, xy: np.ndarray, sigma: float) -> dict[str, float]:
    high = pairwise_distances(descriptors, metric="manhattan")
    low = pairwise_distances(xy, metric="euclidean")
    high_u = upper(high)
    low_u = upper(low)
    return {
        "T": trustworthiness(high, low),
        "C": trustworthiness(low, high),
        "kruskal": kruskal(high_u, low_u),
        "sketch": sketch_stress(high_u, low_u, sigma),
    }


def sigma_of(stderr: str) -> float:
    for line in stderr.splitlines():
        if line.startswith("landfold:") and "sigma" in line:
            parts = line.split()
            return float(parts[parts.index("sigma") + 1].rstrip(","))
    raise SystemExit(f"no sigma in {stderr}")


def embed_sklearn(name: str, descriptors: np.ndarray, distance: np.ndarray) -> np.ndarray:
    if name == "sklearn_mds":
        model = MDS(
            n_components=2,
            metric="precomputed",
            n_init=1,
            init="random",
            random_state=0,
            normalized_stress=False,
        )
        return model.fit_transform(distance)
    if name == "sklearn_mds_switch":
        return embed_sklearn("sklearn_mds", descriptors, distance)
    if name == "sklearn_isomap":
        return Isomap(n_neighbors=NEIGHBORS, n_components=2, metric="manhattan").fit_transform(
            descriptors
        )
    if name == "sklearn_tsne":
        return TSNE(
            n_components=2,
            perplexity=30,
            metric="manhattan",
            method="exact",
            init="random",
            random_state=0,
        ).fit_transform(descriptors)
    raise KeyError(name)


def embed_others(name: str, descriptors: np.ndarray) -> np.ndarray:
    if name == "opentsne":
        from openTSNE import TSNE as OpenTSNE

        return np.asarray(
            OpenTSNE(
                perplexity=30,
                metric="manhattan",
                n_jobs=8,
                random_state=0,
                negative_gradient_method="bh",
            ).fit(descriptors)
        )
    if name == "umap":
        import umap

        return umap.UMAP(
            n_components=2,
            metric="manhattan",
            n_neighbors=NEIGHBORS,
            random_state=0,
            n_jobs=1,
            compatibility_layout=True,
        ).fit_transform(descriptors)
    if name == "phate":
        import phate

        return phate.PHATE(
            n_components=2,
            knn=NEIGHBORS,
            knn_dist="cityblock",
            mds_dist="cityblock",
            n_jobs=8,
            random_state=0,
            verbose=0,
        ).fit_transform(descriptors)
    raise KeyError(name)


def write_csv(rows: list[dict[str, str]]) -> None:
    fields = ["book", "method", "T", "C", "kruskal", "sketch", "runtime_s", "umap_learn"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    for path in (OUT, SCRATCH / "survey.csv"):
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)


def cell(rows: list[dict[str, str]], book: str, method: str) -> dict[str, str]:
    for row in rows:
        if row["book"] == book and row["method"] == method:
            return row
    raise KeyError((book, method))


def difference_table(rows: list[dict[str, str]]) -> str:
    methods = []
    for row in rows:
        if row["method"] != "anneal_landfold" and row["method"] not in methods:
            methods.append(row["method"])
    books = []
    for row in rows:
        if row["book"] not in books:
            books.append(row["book"])
    lines = [
        "Differences are landfold minus that method.",
        "Runtime ratios are that method divided by landfold.",
        "Each range is the smallest and largest value across the books.",
        "",
        "| method | T | C | sketch-map | runtime ratio |",
        "|--------+---+---+------------+---------------|",
    ]
    for method in methods:
        if method == "anneal_landfold_book":
            continue
        bags = {key: [] for key in ("T", "C", "sketch", "runtime_s")}
        for book in books:
            ours = cell(rows, book, "anneal_landfold")
            theirs = cell(rows, book, method)
            bags["T"].append(float(ours["T"]) - float(theirs["T"]))
            bags["C"].append(float(ours["C"]) - float(theirs["C"]))
            bags["sketch"].append(float(ours["sketch"]) - float(theirs["sketch"]))
            bags["runtime_s"].append(float(theirs["runtime_s"]) / float(ours["runtime_s"]))
        lines.append(
            "| {method} | {t} | {c} | {s} | {r} |".format(
                method=method,
                t=span(bags["T"]),
                c=span(bags["C"]),
                s=span(bags["sketch"]),
                r=span(bags["runtime_s"]),
            )
        )
    return "\n".join(lines) + "\n"


def span(values: list[float]) -> str:
    lo, hi = min(values), max(values)
    return f"{lo:.4f} to {hi:.4f}"


def org_table(rows: list[dict[str, str]]) -> str:
    lines = [
        "| book | method | T | C | Kruskal | sketch-map | runtime (s) |",
        "|------+--------+---+---+---------+------------+-------------|",
    ]
    for row in rows:
        lines.append(
            "| {book} | {method} | {T} | {C} | {kruskal} | {sketch} | {runtime_s} |".format(**row)
        )
    return "\n".join(lines) + "\n"


def rewrite(rows: list[dict[str, str]]) -> None:
    page = PAGE.read_text()
    versions = (
        "The versions measured here are scikit-learn {sk}, openTSNE {ot}, "
        "umap-learn {um} and PHATE {ph}.\n".format(
            sk=version("scikit-learn"),
            ot=version("opentsne"),
            um=version("umap-learn"),
            ph=version("phate"),
        )
    )
    old_versions = next(
        line
        for line in page.splitlines(keepends=True)
        if line.startswith("The versions are ") or line.startswith("The versions measured here are ")
    )
    page = page.replace(old_versions, versions, 1)
    page = page.replace(
        "Runtimes are asv medians of three runs on one machine with eight cores.\n",
        "Runtimes are the median wall time of three runs.\n",
    )
    page = page.replace(
        "Settings are in =landfold/embed.py= of the anneal_repro workflow, with seed 0 throughout.\n",
        "Settings and the run are =scripts/landfold_survey.py=, with seed 0 throughout.\n",
    )
    new_reference = (
        "The reference row is lab-cosmo/sketchmap at c2871db187f4a60377e7727af781da6c6c3426cb.\n"
        "It uses the landfold landmarks, sigma and switch exponents 2, 4.\n"
        "Landmark weights are Euclidean Voronoi populations to power 1, normalized, as =dimlandmark -w= writes them.\n"
        "The fit follows the stages in =utils/sketch-map.sh=, and other points are placed with =dimproj=.\n"
        "The table is =data/landfold_survey.csv=, written by =scripts/landfold_survey.py=.\n"
        "The =umap_learn= column is the umap-learn version of the run.\n"
    )
    anchors = (
        "The reference sketch-map run takes the landfold landmarks, sigma and switch exponents.\n"
        "Its landmark weights are Voronoi populations at power 1, as =dimlandmark -w= writes them.\n"
        "It follows the stages of its own =sketch-map.sh= script.\n",
        "The reference sketch-map program is not run here, so the table has no reference row.\n"
        "The table is =data/landfold_survey.csv=, written by =scripts/landfold_survey.py=.\n",
    )
    replaced = False
    for old_reference in anchors:
        if old_reference in page:
            page = page.replace(old_reference, new_reference, 1)
            replaced = True
            break
    if not replaced and "lab-cosmo/sketchmap at c2871db" not in page:
        raise SystemExit("the reference-sketch-map paragraph is not the one this script rewrites")
    start = page.index("| book | method | T |")
    end = page.index("* References")
    generated = org_table(rows) + "\n" + difference_table(rows) + "\n"
    page = page[:start] + generated + page[end:]
    PAGE.write_text(page)


def record(book: str, method: str, metrics: dict[str, float] | None, runtime: float) -> dict[str, str]:
    blank = {"T": "", "C": "", "kruskal": "", "sketch": ""}
    if metrics is not None:
        blank = {
            "T": f"{metrics['T']:.4f}",
            "C": f"{metrics['C']:.4f}",
            "kruskal": f"{metrics['kruskal']:.4f}",
            "sketch": f"{metrics['sketch']:.5f}",
        }
    return {
        "book": book,
        "method": method,
        "runtime_s": f"{runtime:.2f}",
        "umap_learn": version("umap-learn"),
        **blank,
    }


def main() -> None:
    ensure_sketchmap()
    target = cargo_target()
    rows: list[dict[str, str]] = []
    for name, size, seed, fixtures in BOOKS:
        book = ensure_book(target, name, size, seed, fixtures)
        desc = SCRATCH / f"{name}.desc.csv"
        mapped = SCRATCH / f"{name}.landfold.csv"
        _out, err, _wall = anneal(
            target,
            ["--seed", "0", "--threads", "8", str(book), "--out", str(mapped), "--dump-descriptors", str(desc)],
        )
        sigma = sigma_of(err)
        descriptors = load_descriptors(desc)
        xy = load_map(mapped)
        metrics = pair_metrics(descriptors, xy, sigma)

        def time_descriptors(threads: str) -> float:
            _o, _e, wall = anneal(
                target,
                ["--seed", "0", "--threads", threads, "--descriptors", str(desc), "--out", str(SCRATCH / "timed.csv")],
            )
            return wall

        def time_book() -> float:
            _o, _e, wall = anneal(
                target,
                ["--seed", "0", "--threads", "8", str(book), "--out", str(SCRATCH / "timed-book.csv")],
            )
            return wall

        rows.append(record(name, "anneal_landfold", metrics, median_time(lambda: time_descriptors("8"))))
        rows.append(record(name, "anneal_landfold_serial", metrics, median_time(lambda: time_descriptors("1"))))
        rows.append(record(name, "anneal_landfold_book", None, median_time(time_book)))
        distance = pairwise_distances(descriptors, metric="manhattan")
        switched = switch(distance, sigma)
        np.fill_diagonal(switched, 0.0)
        for method in ("sklearn_mds", "sklearn_mds_switch", "sklearn_isomap", "sklearn_tsne", "opentsne", "umap", "phate"):
            source = switched if method == "sklearn_mds_switch" else distance

            def fit(method: str = method, source: np.ndarray = source) -> np.ndarray:
                if method.startswith("sklearn"):
                    return embed_sklearn(method, descriptors, source)
                return embed_others(method, descriptors)

            embedding = fit()
            runtime = median_time(lambda method=method, source=source: time_call(method, descriptors, source))
            rows.append(record(name, method, pair_metrics(descriptors, embedding, sigma), runtime))
            print(name, method, rows[-1]["runtime_s"], flush=True)
    write_csv(rows)
    rewrite(rows)


def time_call(method: str, descriptors: np.ndarray, source: np.ndarray) -> float:
    started = time.perf_counter()
    if method.startswith("sklearn"):
        embed_sklearn(method, descriptors, source)
    else:
        embed_others(method, descriptors)
    return time.perf_counter() - started


def voronoi_weights(points: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """Euclidean Voronoi counts to power 1, normalized, as dimlandmark -w does."""
    counts = np.zeros(len(landmarks))
    for start in range(0, len(points), 250):
        block = points[start : start + 250]
        distance = np.linalg.norm(block[:, None, :] - landmarks[None, :, :], axis=2)
        counts += np.bincount(distance.argmin(axis=1), minlength=len(landmarks))
    return counts / counts.sum()


def write_rows(path: Path, rows: np.ndarray, weights: np.ndarray | None = None) -> None:
    lines = []
    for i, row in enumerate(rows):
        body = " ".join(f"{value:.10g}" for value in row)
        if weights is not None:
            body += f" {weights[i]:.10g}"
        lines.append(body)
    path.write_text("\n".join(lines) + "\n")


def coords_of(text: str) -> np.ndarray:
    rows = []
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            # dimred prints x y. dimproj prints x y, then the point error and a distance.
            rows.append((float(parts[0]), float(parts[1])))
        except ValueError:
            continue
    return np.array(rows)


SKETCH_URL = "https://github.com/lab-cosmo/sketchmap.git"
SKETCH_COMMIT = "c2871db187f4a60377e7727af781da6c6c3426cb"
SKETCH_ROOT = ROOT / "build" / "sketchmap"


def ensure_sketchmap() -> None:
    """Clone the pinned sketch-map commit into build/ and compile it."""
    binary = SKETCH_ROOT / "bin" / "dimred"
    if binary.is_file() and (SKETCH_ROOT / "bin" / "dimproj").is_file():
        return
    SKETCH_ROOT.parent.mkdir(parents=True, exist_ok=True)
    if not (SKETCH_ROOT / ".git").is_dir():
        subprocess.check_call(["git", "clone", SKETCH_URL, str(SKETCH_ROOT)])
    subprocess.check_call(["git", "checkout", SKETCH_COMMIT], cwd=SKETCH_ROOT)
    header = SKETCH_ROOT / "libs" / "tbdefs.hpp"
    text = header.read_text()
    marker = "operator<<(std::ostream& os, const std::valarray<double>& data)"
    if marker not in text:
        insert = (
            "namespace std {\n"
            "inline std::ostream& operator<<(std::ostream& os, const std::valarray<double>& data)\n"
            "{\n"
            "    for (std::size_t i = 0; i < data.size(); ++i) os << (i ? \" \" : \"\") << data[i];\n"
            "    return os;\n"
            "}\n"
            "}\n"
        )
        needle = "/************************\n  MPI SAFE STREAMS\n*************************/\n"
        if needle not in text:
            raise SystemExit("sketch-map tbdefs.hpp has no place for the valarray stream operator")
        header.write_text(text.replace(needle, needle + insert, 1))
    (SKETCH_ROOT / "make.in").write_text(
        "CXX=g++\n"
        "CXXFLAGS:=-O2 -g -I. -I../libs\n"
        "MAKEDEPEND=g++ -M $(CXXFLAGS)\n"
        "LDFLAGS:=-L. -L../libs -ltoolbox -llapack -lblas\n"
    )
    subprocess.check_call(["make", "-j", str(os.cpu_count() or 1)], cwd=SKETCH_ROOT)
    if not binary.is_file():
        raise SystemExit("sketch-map did not produce bin/dimred")


def dimred(argv: list[str], stdin: str, cwd: Path) -> str:
    binary = ROOT / "build" / "sketchmap" / "bin" / "dimred"
    done = subprocess.run(
        [str(binary), *argv],
        input=stdin,
        text=True,
        cwd=cwd,
        capture_output=True,
    )
    if done.returncode != 0:
        raise RuntimeError(done.stderr[-2000:])
    return done.stdout


def sketchmap_embed(descriptors: np.ndarray, landmark_index: np.ndarray, sigma: float, scratch: Path) -> np.ndarray:
    scratch.mkdir(parents=True, exist_ok=True)
    landmarks = descriptors[landmark_index]
    weights = voronoi_weights(descriptors, landmarks)
    high = scratch / "landmarks.hd"
    write_rows(high, landmarks, weights)
    high_text = high.read_text()
    dimension = str(descriptors.shape[1])
    first = dimred(["-D", dimension, "-d", "2", "-w", "-center", "-preopt", "100"], high_text, scratch)
    start = coords_of(first)
    (scratch / "init.txt").write_text("".join(f"{x:.10g} {y:.10g}\n" for x, y in start))
    switch = f"{sigma},2,4"
    second = dimred(
        [
            "-D", dimension, "-d", "2", "-w", "-center", "-preopt", "100",
            "-fun-hd", switch, "-fun-ld", switch, "-init", "init.txt",
        ],
        high_text,
        scratch,
    )
    current = coords_of(second)
    (scratch / "init.txt").write_text("".join(f"{x:.10g} {y:.10g}\n" for x, y in current))
    radius = float(np.linalg.norm(current, axis=1).max() * 1.2)
    for _ in range(3):
        fitted = dimred(
            [
                "-D", dimension, "-d", "2", "-w", "-center", "-preopt", "50",
                "-grid", f"{radius},21,201", "-fun-hd", switch, "-fun-ld", switch,
                "-init", "init.txt", "-gopt", "3", "-imix", "0.5",
            ],
            high_text,
            scratch,
        )
        current = coords_of(fitted)
        (scratch / "init.txt").write_text("".join(f"{x:.10g} {y:.10g}\n" for x, y in current))
        radius = float(np.linalg.norm(current, axis=1).max() * 1.2)
    final = dimred(
        [
            "-D", dimension, "-d", "2", "-w", "-center", "-preopt", "100",
            "-grid", f"{radius},21,201", "-fun-hd", switch, "-fun-ld", switch,
            "-init", "init.txt", "-gopt", "10",
        ],
        high_text,
        scratch,
    )
    current = coords_of(final)
    low = scratch / "landmarks.ld"
    low.write_text("".join(f"{x:.10g} {y:.10g}\n" for x, y in current))
    project = ROOT / "build" / "sketchmap" / "bin" / "dimproj"
    plain = scratch / "all.hd"
    write_rows(plain, descriptors)
    done = subprocess.run(
        [
            str(project), "-D", dimension, "-d", "2",
            "-P", str(high), "-p", str(low), "-w",
            "-fun-hd", switch, "-fun-ld", switch,
            "-grid", f"{radius},21,201", "-cgmin", "10",
        ],
        input=plain.read_text(),
        text=True,
        cwd=scratch,
        capture_output=True,
    )
    if done.returncode != 0:
        raise RuntimeError(done.stderr[-2000:])
    mapped = coords_of(done.stdout)
    if len(mapped) != len(descriptors):
        raise RuntimeError(f"dimproj returned {len(mapped)} rows, expected {len(descriptors)}")
    return mapped


def add_sketchmap() -> None:
    ensure_sketchmap()
    if version("umap-learn") != "0.6.0":
        raise SystemExit(f"umap-learn is {version('umap-learn')}, the survey cites 0.6.0")
    rows = list(csv.DictReader(OUT.open()))
    for row in rows:
        row["umap_learn"] = version("umap-learn")
    rows = [row for row in rows if row["method"] != "reference_sketchmap"]
    target = cargo_target()
    for name, size, seed, fixtures in BOOKS:
        book = ensure_book(target, name, size, seed, fixtures)
        mapped = SCRATCH / f"{name}.landfold.csv"
        desc = SCRATCH / f"{name}.desc.csv"
        _out, err, _wall = anneal(
            target,
            ["--seed", "0", "--threads", "8", str(book), "--out", str(mapped), "--dump-descriptors", str(desc)],
        )
        sigma = sigma_of(err)
        table = np.loadtxt(mapped, delimiter=",", skiprows=1)
        descriptors = load_descriptors(desc)
        landmark_index = np.flatnonzero(table[:, 4] > 0.5)
        print(name, "landmarks", len(landmark_index), "sigma", sigma, flush=True)
        folder = SCRATCH / "sketchmap" / name

        def once(folder: Path = folder, descriptors: np.ndarray = descriptors, landmark_index: np.ndarray = landmark_index, sigma: float = sigma):
            started = time.perf_counter()
            mapped = sketchmap_embed(descriptors, landmark_index, sigma, folder)
            return time.perf_counter() - started, mapped

        timed = [once() for _ in range(REPEATS)]
        timed.sort(key=lambda item: item[0])
        runtime, embedding = timed[REPEATS // 2]
        rows.append(record(name, "reference_sketchmap", pair_metrics(descriptors, embedding, sigma), runtime))
        print(name, "reference_sketchmap", rows[-1]["runtime_s"], flush=True)
    order = {method: index for index, method in enumerate(
        ["anneal_landfold", "anneal_landfold_serial", "anneal_landfold_book", "reference_sketchmap",
         "sklearn_mds", "sklearn_mds_switch", "sklearn_isomap", "sklearn_tsne", "opentsne", "umap", "phate"]
    )}
    rows.sort(key=lambda row: (row["book"], order.get(row["method"], 99)))
    write_csv(rows)
    rewrite(rows)


if __name__ == "__main__":
    if "--sketchmap" in sys.argv:
        add_sketchmap()
    else:
        main()
