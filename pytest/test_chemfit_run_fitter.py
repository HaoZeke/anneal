"""``run_fitter``: the options it passes on to ``fit_anneal``, and no others."""

import numpy as np
import pytest

pytest.importorskip("anneal")

from anneal.chemfit import run_fitter  # noqa: E402
from chemfit_doubles import ReleasedFitter  # noqa: E402


def _fitter():
    return ReleasedFitter(
        {"x": np.array([0.2, -0.1]), "eps": 1.0},
        {"x": (-1.0, 1.0), "eps": (0.5, 2.0)},
    )


@pytest.mark.parametrize(
    "keyword, hint",
    [
        ("n_epochs", "on to fit_anneal"),
        ("bogus", "on to fit_anneal"),
        ("tell_every", "tell_every is an option of fit_chemfit"),
        ("default_span", "default_span is an option of fit_chemfit"),
        ("t_init", "pass t_init in preset_kwargs"),
    ],
)
def test_run_fitter_names_itself_for_a_keyword_it_does_not_pass_on(keyword, hint):
    fitter = _fitter()
    with pytest.raises(TypeError) as caught:
        run_fitter(fitter, 60, **{keyword: 5})
    message = str(caught.value)
    assert message.startswith(
        f"run_fitter() got an unexpected keyword argument {keyword!r}; it passes only "
        "x0, low, high, bound_span, steps_per_epoch and preset_kwargs on to fit_anneal"
    )
    assert hint in message
    assert "_fit_anneal" not in message
    assert fitter.calls == []


def test_a_fitter_with_its_own_fit_anneal_gets_every_keyword():
    class OwnFitAnneal:
        def fit_anneal(self, **kwargs):
            return kwargs

    got = run_fitter(OwnFitAnneal(), 60, method="sa", tell_every=5, n_epochs=3)
    assert got == {
        "budget": 60,
        "method": "sa",
        "preset": None,
        "seed": 42,
        "tell_every": 5,
        "n_epochs": 3,
    }
