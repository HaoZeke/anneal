# Copyright (c) anneal developers
"""Verification for ledger_charge_invariant.org — integer Ledger contract."""
from __future__ import annotations


class Ledger:
    """Mirror of src/methods/cluster_hopping.rs Ledger charge API."""

    def __init__(self, budget: int) -> None:
        self.budget = budget
        self.spent = 0
        self.settled_beyond_cap = 0
        self.best = float("inf")
        self._stopped = False

    def stop(self) -> None:
        self._stopped = True

    def stopped(self) -> bool:
        return self._stopped

    def charge(self) -> bool:
        if self.spent >= self.budget or self.stopped():
            return False
        self.spent += 1
        return True

    def charge_many(self, n: int) -> bool:
        # A stop closes new work: remaining() is 0. It does not void
        # evaluations already performed. The debt added is n capped by
        # the budget still unpaid, which ignores the stop. Settled calls
        # that do not fit are kept in settled_beyond_cap, so spent stays
        # within the budget (I1) and no completed evaluation is dropped.
        open_room = self.remaining()
        owed_room = max(self.budget - self.spent, 0)
        paid = min(n, owed_room)
        self.spent += paid
        self.settled_beyond_cap += n - paid
        return n <= open_room

    def charged(self) -> int:
        # The count a benchmark reads: every settled call.
        return self.spent + self.settled_beyond_cap

    def settle_child(self, child: "Ledger", offset: int = 0) -> bool:
        # The child's own overrun is settled with its spend, so an
        # evaluation past a sub-ledger's cap still reaches the parent.
        return self.charge_many(max(child.spent - offset, 0) + child.settled_beyond_cap)

    def remaining(self) -> int:
        if self.stopped():
            return 0
        return max(self.budget - self.spent, 0)

    def record(self, value: float) -> None:
        if value < self.best:
            self.best = value


def i1_partition(led: Ledger) -> bool:
    return led.spent + led.remaining() == led.budget


def test_i1_through_charges() -> bool:
    for B in range(0, 12):
        led = Ledger(B)
        if not i1_partition(led):
            return False
        while led.charge():
            if not i1_partition(led):
                return False
        if led.remaining() != 0 or led.charge():
            return False
        if not i1_partition(led):
            return False
    return True


def test_i4_partial_many() -> bool:
    led = Ledger(10)
    for _ in range(7):
        led.charge()
    # remaining 3
    ok = led.charge_many(5) is False
    ok &= led.spent == 10
    ok &= led.remaining() == 0
    ok &= i1_partition(led)
    return ok


def test_i4_exact_many() -> bool:
    led = Ledger(10)
    ok = led.charge_many(4) is True
    ok &= led.spent == 4
    ok &= led.charge_many(6) is True
    ok &= led.spent == 10
    return ok and i1_partition(led)


def test_i5_record_no_charge() -> bool:
    led = Ledger(5)
    led.charge()
    s = led.spent
    led.record(-1.0)
    return led.spent == s and led.best == -1.0


def test_i3_hard_stop() -> bool:
    led = Ledger(2)
    assert led.charge() and led.charge()
    assert led.charge() is False
    s = led.spent
    assert led.charge() is False
    return led.spent == s == 2


def test_stopped_charge_many_pays_budget_room() -> bool:
    led = Ledger(10)
    for _ in range(4):
        assert led.charge()
    led.stop()
    # The stop refuses a new single charge and reports no open room,
    # while the unpaid budget is still 6.
    ok = led.charge() is False
    ok &= led.remaining() == 0
    ok &= led.spent == 4
    # Three of that room are owed. The stop does not void them, and
    # the call is not open work, so the return is false.
    ok &= led.charge_many(3) is False
    ok &= led.spent == 7
    ok &= led.remaining() == 0
    # The rest of the budget, then nothing past it.
    ok &= led.charge_many(100) is False
    ok &= led.spent == 10
    ok &= led.remaining() == 0
    ok &= i1_partition(led)
    return ok


def test_overrun_is_recorded_apart() -> bool:
    led = Ledger(3)
    ok = led.charge_many(3) is True
    ok &= led.charge_many(2) is False
    # spent holds at the budget; the two that did not fit are kept.
    ok &= led.spent == 3 and led.settled_beyond_cap == 2
    ok &= led.spent + led.settled_beyond_cap == 5
    ok &= i1_partition(led)
    # A stop does not void settled calls either.
    stopped = Ledger(10)
    stopped.charge_many(2)
    stopped.stop()
    ok &= stopped.charge_many(4) is False
    ok &= stopped.spent == 6 and stopped.settled_beyond_cap == 0
    ok &= stopped.charge_many(100) is False
    ok &= stopped.spent == 10 and stopped.settled_beyond_cap == 96
    ok &= stopped.spent + stopped.settled_beyond_cap == 2 + 4 + 100
    ok &= i1_partition(stopped)
    return ok


def test_every_settled_call_is_on_one_field() -> bool:
    for budget in range(0, 8):
        for batches in ([1], [3, 4], [5, 5, 5], [0, 2, 9], [4, 1, 1, 1]):
            for stop_at in range(0, len(batches) + 1):
                led = Ledger(budget)
                total = 0
                for index, n in enumerate(batches):
                    if index == stop_at:
                        led.stop()
                    led.charge_many(n)
                    total += n
                    if led.spent > led.budget:
                        return False
                    if not led.stopped() and not i1_partition(led):
                        return False
                if led.spent + led.settled_beyond_cap != total:
                    return False
    return True


def test_child_overrun_reaches_the_parent() -> bool:
    for parent_budget in range(0, 9):
        for child_budget in range(1, 6):
            for batches in ([2], [5, 4], [1, 1, 9], [0, 7]):
                parent = Ledger(parent_budget)
                child = Ledger(child_budget)
                for n in batches:
                    child.charge_many(n)
                parent.settle_child(child)
                if parent.charged() != sum(batches):
                    return False
                if parent.spent > parent.budget or child.spent > child.budget:
                    return False
                grand = Ledger(parent_budget + 3)
                grand.settle_child(parent)
                if grand.charged() != sum(batches):
                    return False
    return True


def all_checks() -> list[tuple[str, bool]]:
    return [
        ("I1 through charge to exhaustion", test_i1_through_charges()),
        ("I4 partial charge_many", test_i4_partial_many()),
        ("I4 exact charge_many", test_i4_exact_many()),
        ("I5 record does not charge", test_i5_record_no_charge()),
        ("I3 hard stop", test_i3_hard_stop()),
        ("stopped charge_many pays budget room", test_stopped_charge_many_pays_budget_room()),
        ("overrun is recorded apart", test_overrun_is_recorded_apart()),
        ("every settled call is on one field", test_every_settled_call_is_on_one_field()),
        ("a child's overrun reaches the parent", test_child_overrun_reaches_the_parent()),
    ]


WITNESS = all(v for _, v in all_checks())


def main() -> int:
    print("Ledger charge invariants — verification")
    print()
    ok = True
    for name, v in all_checks():
        print(f"  {name}: {v}")
        ok = ok and bool(v)
    print("LEDGER_INVARIANT_OK" if ok else "LEDGER_INVARIANT_FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
