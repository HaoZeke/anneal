"""anneal.landfold folds descriptor rows and minima books onto the plane."""

from pathlib import Path

import numpy as np
import pytest

anneal = pytest.importorskip("anneal")

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures"


def _planar(n, seed):
    rng = np.random.default_rng(seed)
    centres = np.array([[0.0, 0.0], [4.0, 0.0], [0.0, 3.0], [4.0, 3.0]])
    xy = centres[np.arange(n) % 4] + rng.normal(scale=0.4, size=(n, 2))
    frame, _ = np.linalg.qr(rng.normal(size=(6, 2)))
    return xy, xy @ frame.T


def _procrustes(x, y):
    x = x - x.mean(axis=0)
    y = y - y.mean(axis=0)
    u, _, vt = np.linalg.svd(y.T @ x)
    return np.linalg.norm(x - y @ (u @ vt)) / np.linalg.norm(x)


def test_planar_layout_is_recovered_up_to_rotation():
    xy, rows = _planar(120, 1)
    out = anneal.landfold(rows, landmarks=60, metric="l2")
    assert out["xy"].shape == (120, 2)
    assert len(out["landmarks"]) == 60
    assert out["spectrum"][1] > 0
    assert _procrustes(xy, out["xy"]) < 1e-4


def test_a_seed_fixes_the_map_at_any_thread_count():
    _, rows = _planar(90, 2)
    one = anneal.landfold(rows, landmarks=30, seed=5, threads=1)
    four = anneal.landfold(rows, landmarks=30, seed=5, threads=4)
    np.testing.assert_array_equal(one["xy"], four["xy"])
    np.testing.assert_array_equal(one["landmarks"], four["landmarks"])


def test_projection_matches_the_fit_for_non_landmarks():
    _, rows = _planar(90, 3)
    out = anneal.landfold(rows, landmarks=30)
    rest = np.setdiff1d(np.arange(90), out["landmarks"])
    np.testing.assert_array_equal(out["map"].project(rows[rest]), out["xy"][rest])
    assert out["map"].landmark_xy.shape == (30, 2)
    assert out["map"].weights.shape == (30,)


def test_bad_options_raise_value_error():
    _, rows = _planar(20, 4)
    with pytest.raises(ValueError):
        anneal.landfold(rows, metric="cosine")
    with pytest.raises(ValueError):
        anneal.landfold(rows, high=(1.0, 4.0))
    with pytest.raises(ValueError):
        anneal.landfold(rows[0])


def _xyz_coordinates(path):
    lines = path.read_text().splitlines()
    return np.array(
        [[float(t) for t in line.split()[1:4]] for line in lines[2:] if line.strip()]
    )


def test_a_book_of_both_lj38_funnels_folds_apart(tmp_path):
    rng = np.random.default_rng(0)
    lines = []
    for name in ("lj38_fcc.xyz", "lj38_ico.xyz"):
        x = _xyz_coordinates(FIXTURES / name)
        for k in range(6):
            y = x + (0.0 if k == 0 else rng.normal(scale=0.02, size=x.shape))
            lines.append(" ".join(["0.0", *(f"{v:.8f}" for v in y.ravel())]))
    book = tmp_path / "lj38.min"
    book.write_text("\n".join(lines) + "\n")

    out = anneal.landfold_book(book, landmarks=12)
    assert out["xy"].shape == (12, 2)
    assert out["rows"].shape[0] == 12
    assert out["energy"].shape == (12,)
    assert out["community"].dtype == np.int64
    fcc, ico = out["xy"][:6], out["xy"][6:]
    gap = np.linalg.norm(fcc.mean(axis=0) - ico.mean(axis=0))
    spread = max(
        np.linalg.norm(fcc - fcc.mean(axis=0), axis=1).max(),
        np.linalg.norm(ico - ico.mean(axis=0), axis=1).max(),
    )
    assert gap > spread
