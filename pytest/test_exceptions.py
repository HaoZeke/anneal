# Copyright (c) anneal developers
"""Errors raised for rejected input are typed and stay ValueError."""

import re
from pathlib import Path

import numpy as np
import pytest

import anneal
import anneal._core as core

ROOT = Path(__file__).resolve().parent.parent

HIERARCHY = {
    "AnnealError": ValueError,
    "InvalidArgumentError": "AnnealError",
    "BoundsError": "InvalidArgumentError",
    "ConfigError": "AnnealError",
    "RemovedConfigKeyError": "ConfigError",
    "NoFiniteStartError": "AnnealError",
    "FeatureUnavailableError": "AnnealError",
}


def f(x):
    return float(np.sum(np.asarray(x) ** 2))


def g(x):
    return 2.0 * np.asarray(x)


@pytest.mark.parametrize("name", sorted(HIERARCHY))
def test_the_class_sits_where_the_docs_put_it(name):
    cls = getattr(anneal, name)
    parent = HIERARCHY[name]
    parent = getattr(anneal, parent) if isinstance(parent, str) else parent
    assert cls.__bases__ == (parent,)
    assert issubclass(cls, ValueError)
    assert cls.__module__ == "anneal"
    assert getattr(core, name) is cls
    assert name in anneal.__all__


def test_every_class_has_a_docstring():
    for name in HIERARCHY:
        assert getattr(anneal, name).__doc__


def test_bad_core_arguments_raise_invalid_argument():
    with pytest.raises(anneal.InvalidArgumentError, match="at least 2"):
        anneal.cluster_search(f, g, n=1, budget=10)
    with pytest.raises(anneal.InvalidArgumentError, match="replicas"):
        anneal.ensemble_optimize(f, np.zeros(2), np.ones(2), budget=10, replicas=0)


def test_bad_wrapper_arguments_raise_invalid_argument():
    with pytest.raises(anneal.InvalidArgumentError, match="n_epochs"):
        anneal.run(f, np.zeros(1), np.ones(1), anneal.Boltzmann(), n_epochs=-1)
    with pytest.raises(anneal.InvalidArgumentError, match="max_evals"):
        anneal.run(f, np.zeros(1), np.ones(1), anneal.Boltzmann(), max_evals=0)


@pytest.mark.parametrize(
    "low, high, message",
    [
        (np.zeros(2), np.ones(1), "same length"),
        (np.zeros(1), np.array([np.nan]), "finite"),
        (np.ones(1), np.zeros(1), "strictly less"),
    ],
)
def test_bad_boxes_raise_bounds_error(low, high, message):
    with pytest.raises(anneal.BoundsError, match=message) as info:
        anneal.run(f, low, high, anneal.Boltzmann())
    assert isinstance(info.value, anneal.InvalidArgumentError)


@pytest.mark.parametrize(
    "check, text, key",
    [
        (anneal.Config.validate_json, '{"a": {"reseed_hops": 1}}', "a.reseed_hops"),
        (anneal.Config.validate_toml, "prior_motif = 1", "prior_motif"),
    ],
)
def test_a_removed_key_raises_removed_config_key_error(check, text, key):
    with pytest.raises(anneal.RemovedConfigKeyError) as info:
        check(text)
    assert info.value.key == key
    assert key in str(info.value)
    assert isinstance(info.value, anneal.ConfigError)


@pytest.mark.parametrize(
    "check, text, key",
    [
        (anneal.Config.validate_json, "{", "<json>"),
        (anneal.Config.validate_toml, "= =", "<toml>"),
    ],
)
def test_unreadable_text_raises_config_error_but_not_the_removed_key_class(check, text, key):
    with pytest.raises(anneal.ConfigError) as info:
        check(text)
    assert not isinstance(info.value, anneal.RemovedConfigKeyError)
    assert info.value.key == key


def test_a_clean_payload_is_accepted():
    anneal.Config.validate_json('{"prior_soap": true}')
    anneal.Config.validate_toml("prior_soap = true")


def test_no_finite_start_carries_the_attempt_count():
    with pytest.raises(anneal.NoFiniteStartError, match="no finite start") as info:
        anneal.cluster_search(lambda x: float("nan"), g, n=4, budget=100, seed=1)
    assert isinstance(info.value.attempts, int)
    assert info.value.attempts >= 1
    assert str(info.value.attempts) in str(info.value)


def test_a_missing_build_feature_raises_feature_unavailable():
    try:
        anneal.cluster_search(f, g, n=4, budget=100, ras=True)
    except ValueError as err:
        if "feature" in str(err):
            assert isinstance(err, anneal.FeatureUnavailableError)
        else:
            raise


def test_old_except_clauses_keep_working():
    with pytest.raises(ValueError):
        anneal.cluster_search(f, g, n=1, budget=10)
    with pytest.raises(ValueError):
        anneal.Config.validate_json("{")


def test_source_raises_no_bare_value_error():
    # A new `PyValueError` in the bindings would skip the typed classes.
    offenders = []
    for path in sorted((ROOT / "src").glob("python*")) + sorted((ROOT / "src" / "python").glob("*.rs")):
        if not path.is_file():
            continue
        for number, line in enumerate(path.read_text().splitlines(), 1):
            code = line.split("//", 1)[0]
            if "PyValueError" in code and path.name != "errors.rs":
                offenders.append(f"{path.relative_to(ROOT)}:{number}")
    assert offenders == []


def test_python_wrappers_raise_no_bare_value_error():
    offenders = []
    for name in ("__init__.py", "device.py", "_landfold.py"):
        text = (ROOT / "python" / "anneal" / name).read_text()
        for match in re.finditer(r"raise ValueError", text):
            offenders.append(f"{name}:{text[: match.start()].count(chr(10)) + 1}")
    assert offenders == []
