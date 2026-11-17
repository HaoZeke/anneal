#!/usr/bin/env python3
"""Contract tests for recursive ExprParams / PotSpec and PotentialConfig.expr."""

from __future__ import annotations

from pathlib import Path

import capnp

SCRIPT_DIR = Path(__file__).resolve().parent
SCHEMA_PATH = SCRIPT_DIR / "../CppCore/rgpot/rpc/Potentials.capnp"
pot_capnp = capnp.load(str(SCHEMA_PATH))


def _union_names(schema) -> set[str]:
    fields = getattr(schema, "union_fields", ())
    names = set()
    for field in fields:
        name = getattr(field, "name", field)
        names.add(str(name))
    return names


def test_schema_has_recursive_expr_tree() -> None:
    assert hasattr(pot_capnp, "ExprTerm")
    assert hasattr(pot_capnp, "ExprParams")
    assert hasattr(pot_capnp, "PotSpec")
    assert not hasattr(pot_capnp, "SumPot")


def test_potential_config_expr_is_upstream_arm_16() -> None:
    schema = pot_capnp.PotentialConfig.schema
    names = _union_names(schema)
    assert "expr" in names
    assert "sum" not in names
    assert "list" not in names
    assert {"none", "nwchem", "cpmd", "metatomic", "lammps"} <= names
    expr = schema.fields["expr"].proto
    assert expr.ordinal.explicit == 16
    assert expr.slot.type.struct.typeId == pot_capnp.ExprParams.schema.node.id


def test_expr_params_names_half_lj_plus_d3() -> None:
    spec = pot_capnp.PotSpec.new_message()
    expr = spec.init("expr")
    expr.expression = "0.5*lj + d3"
    terms = expr.init("terms", 2)
    terms[0].name = "lj"
    terms[0].pot.none = None
    terms[1].name = "d3"
    terms[1].pot.none = None
    assert spec.which() == "expr"
    assert expr.expression == "0.5*lj + d3"
    assert [t.name for t in expr.terms] == ["lj", "d3"]
    assert terms[0].pot.which() == "none"
    assert terms[1].pot.which() == "none"
    data = spec.to_bytes()
    with pot_capnp.PotSpec.from_bytes(data) as restored:
        assert restored.expr.expression == "0.5*lj + d3"
        assert [t.name for t in restored.expr.terms] == ["lj", "d3"]
        assert restored.expr.terms[0].pot.which() == "none"


def test_expr_params_recurses_through_potspec() -> None:
    outer = pot_capnp.PotSpec.new_message()
    expr = outer.init("expr")
    expr.expression = "0.5*inner + d3"
    terms = expr.init("terms", 2)
    terms[0].name = "inner"
    inner = terms[0].pot.init("expr")
    inner.expression = "lj"
    inner_terms = inner.init("terms", 1)
    inner_terms[0].name = "lj"
    inner_terms[0].pot.none = None
    terms[1].name = "d3"
    terms[1].pot.none = None
    data = outer.to_bytes()
    with pot_capnp.PotSpec.from_bytes(data) as restored:
        assert restored.expr.expression == "0.5*inner + d3"
        child = restored.expr.terms[0]
        assert child.name == "inner"
        assert child.pot.which() == "expr"
        assert child.pot.expr.expression == "lj"
        assert child.pot.expr.terms[0].name == "lj"


if __name__ == "__main__":
    test_schema_has_recursive_expr_tree()
    test_potential_config_expr_is_upstream_arm_16()
    test_expr_params_names_half_lj_plus_d3()
    test_expr_params_recurses_through_potspec()
    print("test_expr_params: all ok")
