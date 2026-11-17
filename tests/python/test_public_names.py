"""``rgpot.__all__`` lists exactly the public names the package exports."""

import types

import rgpot


def _exported(name, value):
    if name.startswith("_") or isinstance(value, types.ModuleType):
        return False
    if value is None or isinstance(value, (bool, str)):
        return True
    return getattr(value, "__module__", "").startswith("rgpot")


def test_all_lists_every_public_name():
    public = {n for n, v in vars(rgpot).items() if _exported(n, v)}
    missing = sorted(public - set(rgpot.__all__))
    assert missing == []


def test_all_names_resolve():
    unknown = sorted(n for n in rgpot.__all__ if n not in dir(rgpot) and n != "ase")
    assert unknown == []
