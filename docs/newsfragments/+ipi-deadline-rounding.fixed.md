`IPIPot` no longer fails with "timed out" just before its deadline.
The wait it passed to `poll()` was the time left rounded down to whole milliseconds, so `poll()` could return while time remained; it is now rounded up.
