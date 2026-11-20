``Ledger::hold_tail`` returns a ``TailHold`` for the calls it hid, and
``Ledger::release_tail`` takes that hold back and restores only its calls.
Dropping a hold restores whatever it still keeps. A release leaves the
hold empty, so the later drop does not restore those calls again. A hold
cannot be copied, a second release of it restores nothing, and a hold
from another ledger restores nothing, so one caller cannot release
another caller's reserve.
