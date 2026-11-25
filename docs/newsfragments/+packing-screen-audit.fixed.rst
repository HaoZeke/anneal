The generic packing probe's screen no longer drops a refused candidate.
Its credit for the descent a screening pass leaves undone is the upper
decile of the walk's measured descents, an estimate, so a candidate
whose full relaxation would beat the bar could be refused, and the
catalogue is offered once, so a refused candidate was never seen again.
A refusal now defers the candidate, once per distinct placement, and
the catalogue is offered a second time one ``Config::packing_deferral``
after the first. That offer relaxes every deferred candidate in full,
the one whose screening pass came closest to the bar that refused it
first, and holds the lowest that beats the walk's best and the packing
already held. Its calls are charged like every other call. A budget
that ends during the audit, which has half the walk's remaining budget,
leaves the candidate it cut short and the farther ones deferred, and
the walk offers them again one ``Config::packing_deferral`` later until
each has been relaxed in full or the walk ends. The probe screens with
a copy of the walk's ``ScreenDescent``, whose window every full
relaxation feeds, and the walk's own count and window are left alone.
A walk that runs past the audit with refusals waiting spends those
calls, so its later hops can differ from the walk without the audit.
