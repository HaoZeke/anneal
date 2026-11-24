# Copyright (c) anneal developers
"""Verification for ledger_charge_invariant.org — integer Ledger contract."""
from __future__ import annotations


class Ledger:
    """Mirror of src/methods/cluster_hopping.rs Ledger charge API."""

    def __init__(self, budget: int) -> None:
        self.budget = budget
        self.spent = 0
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
        # the budget still unpaid, which ignores the stop.
        open_room = self.remaining()
        owed_room = max(self.budget - self.spent, 0)
        self.spent += min(n, owed_room)
        return n <= open_room

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


def all_checks() -> list[tuple[str, bool]]:
    return [
        ("I1 through charge to exhaustion", test_i1_through_charges()),
        ("I4 partial charge_many", test_i4_partial_many()),
        ("I4 exact charge_many", test_i4_exact_many()),
        ("I5 record does not charge", test_i5_record_no_charge()),
        ("I3 hard stop", test_i3_hard_stop()),
        ("stopped charge_many pays budget room", test_stopped_charge_many_pays_budget_room()),
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
