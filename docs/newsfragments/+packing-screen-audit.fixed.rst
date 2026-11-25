The generic packing probe's screen can no longer refuse every candidate
for good. Its credit for the descent a screening pass leaves undone is
the upper decile of the walk's measured descents, an estimate, so a
candidate whose full relaxation would beat the bar could be refused,
and the probe's refusals had none of the audit escape the walk's own
screen has. The probe now screens with a copy of the walk's
``ScreenDescent``: every ``ScreenDescent::AUDIT``-th refusal is relaxed
in full and kept if it beats the bar, and every full relaxation feeds
the copy's window. The audit's calls are charged like every other call,
and the walk's own refusal count and window are left alone. On
Lennard-Jones 26, 38 and 75, Morse 38 at rho 6 and Morse 55 at rho 14
(``h2h_seed``), seeds 2000 to 2005, the audit did not fire and added no
calls; Lennard-Jones 75 seed 2001 spends 148 more probe calls, from the
window the probe now feeds, and hits at 3329394 instead of 3329246.
