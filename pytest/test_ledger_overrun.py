# Copyright (c) anneal developers
"""A Ledger reports every settled call, including the part past its budget."""

from anneal import Ledger


def test_charged_includes_what_settled_past_the_budget():
    led = Ledger(30)
    assert led.charge_many(20)
    assert (led.spent, led.settled_beyond_cap, led.charged) == (20, 0, 20)

    assert not led.charge_many(25)
    assert led.spent == 30, "spent never passes the budget"
    assert led.settled_beyond_cap == 15
    assert led.charged == led.spent + led.settled_beyond_cap == 45
    assert led.remaining == 0


def test_a_child_overrun_reaches_the_parent_charge():
    parent = Ledger(30)
    child = parent.slice(10)
    assert child.charge_many(10)
    assert not child.charge_many(40)
    assert (child.spent, child.settled_beyond_cap, child.charged) == (10, 40, 50)

    assert not parent.settle_child(child)
    assert parent.spent == 30
    assert parent.settled_beyond_cap == 20
    assert parent.charged == 50
    assert parent.charged == parent.spent + parent.settled_beyond_cap


def test_an_overrun_passes_through_two_levels():
    top = Ledger(100)
    middle = top.slice(20)
    inner = middle.slice(10)
    assert inner.charge_many(10)
    assert not inner.charge_many(7)
    assert middle.settle_child(inner)
    assert middle.charged == 17
    assert top.settle_child(middle)
    assert top.spent == 17
    assert top.charged == 17
    assert top.settled_beyond_cap == 0

    tight = Ledger(12)
    assert not tight.settle_child(middle)
    assert (tight.spent, tight.settled_beyond_cap, tight.charged) == (12, 5, 17)


def test_priced_units_are_summed_over_children():
    parent = Ledger(100)
    first = parent.slice(40)
    assert first.charge_frac(2.5)
    assert first.charge_many(5)
    assert first.priced == 2
    assert parent.settle_child(first)

    second = parent.slice(40)
    assert second.charge_frac(3.0)
    assert parent.settle_child(second)

    assert parent.priced == 5
    assert parent.spent == 10
    assert parent.charged == 10


def test_a_child_that_priced_keeps_it_when_the_parent_overruns():
    parent = Ledger(5)
    child = parent.slice(10)
    assert child.charge_frac(4.0)
    assert child.charge_many(4)
    assert not parent.settle_child(child)
    assert (parent.spent, parent.settled_beyond_cap, parent.charged) == (5, 3, 8)
    assert parent.priced == 4
