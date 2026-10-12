"""The ChemFit bridges name all of the portfolio's arms or none of them.

The bridges run the portfolio of :func:`anneal.global_optimize` without a
gradient. A docstring of the bridges, or of the examples that drive them,
that names arms of the portfolio names every arm of that run and no other;
the ones that describe the default portfolio name none and point to
:func:`anneal.global_optimize`, whose documentation names them. ``ARMS``
gives the words that name each arm the portfolio defines, and is held to the
names of ``ArmKind`` in ``src/methods/portfolio.rs``.
"""

import ast
import re
from pathlib import Path

import numpy as np
import pytest

anneal = pytest.importorskip("anneal")

ROOT = Path(__file__).resolve().parents[1]
BRIDGES = (
    "python/anneal/chemfit.py",
    "examples/chemfit_anneal.py",
    "examples/chemfit_positions.py",
)
POINTER = ":func:`anneal.global_optimize`"

ARMS = {
    "am_sa": r"(?i:adaptive[- ]Metropolis)",
    "cma": r"\bCMA\b",
    "de": r"(?i:differential[- ]evolution)",
    "dmc_pop": r"\bDMC\b|(?i:population-controlled|diffusion)",
    "explore": r"(?i:\brestart)",
    "gle": r"\bGLE\b|(?i:Langevin)",
    "gsa": r"\bGSA\b|(?i:generalized simulated annealing)",
    "hmc": r"\bHMC\b|(?i:Hamiltonian)",
    "hop": r"(?i:basin[- ]hopping|\bhops?\b)",
    "metad": r"(?i:metadynamics)",
    "pt": r"(?i:parallel[- ]tempering|replica[- ]exchange)",
    "qn": r"\bBFGS\b|(?i:quasi-Newton)",
    "reduced": r"(?i:active[- ]subspace)",
    "shift": r"(?i:\belite)",
    "surrogate": r"(?i:surrogate)",
    "tps": r"(?i:transition[- ]path)",
    "tr_poll": r"(?i:trust[- ]region)",
    "variant": r"(?i:Bayesian[- ]pilot)",
}


def named(text: str) -> set[str]:
    """The arms that ``text`` names, read across its line breaks."""
    text = re.sub(r"(\w)- (\w)", r"\1-\2", " ".join(text.split()))
    return {arm for arm, words in ARMS.items() if re.search(words, text)}


def docstrings() -> dict[str, str]:
    """Every docstring of the bridges and the examples, by file and name."""
    found = {}
    for path in BRIDGES:
        tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                text = ast.get_docstring(node)
                if text:
                    found[f"{path}:{getattr(node, 'name', '')}"] = text
    return found


@pytest.fixture(scope="module")
def run() -> set[str]:
    """The arms of the portfolio run as the bridges run it."""
    result = anneal.global_optimize(
        lambda x: float(np.sum((x - 0.3) ** 2)),
        np.full(3, -1.0),
        np.full(3, 1.0),
        64,
        seed=0,
        x0=np.zeros(3),
    )
    return set(result["arm_pulls"])


def test_the_words_name_each_arm_the_portfolio_defines():
    source = (ROOT / "src" / "methods" / "portfolio.rs").read_text(encoding="utf-8")
    assert set(re.findall(r'ArmKind::\w+ => "(\w+)"', source)) == set(ARMS)


def test_the_portfolio_documentation_names_every_arm_the_bridges_run(run):
    assert run < set(ARMS)
    assert run <= named(anneal.global_optimize.__doc__)


def test_a_partial_list_is_not_every_arm_of_the_run(run):
    listed = "QMC restarts, basin hopping, differential evolution, GSA, parallel tempering"
    assert named(listed) == {"explore", "hop", "de", "gsa", "pt"}
    assert named(listed) != run


def test_a_bridge_docstring_names_every_arm_of_the_run_or_none(run):
    partial = {}
    for where, text in docstrings().items():
        arms = named(text)
        if arms and arms != run:
            partial[where] = sorted(arms)
    assert partial == {}


@pytest.mark.parametrize(
    "where",
    [
        "python/anneal/chemfit.py:fit_anneal",
        "python/anneal/chemfit.py:fit_chemfit",
        "examples/chemfit_anneal.py:run_anneal",
    ],
)
def test_the_default_portfolio_points_to_global_optimize(where):
    assert POINTER in docstrings()[where]
