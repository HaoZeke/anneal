The LJ98 stop is the Cambridge Td global minimum, -543.665361 (Leary and
Doye, Phys. Rev. E 60, R6320, 1999). ``Ledger::stop_below`` refuses a halt
limit equal to the Cs icosahedral shelf -543.642957 (Deaven, Chem. Phys.
Lett. 256, 195, 1996) or the LJ75 icosahedral shelf -396.282249. When the
published minimum for that size is known, a limit above the minimum plus
the harness window of 0.001 warns and is kept. The optbench driver applies
the same check in ``load_problem``. ASE and AIRSS keep their LJ75 and LJ38
stops. A recorded energy at or below the published minimum plus the
tolerance counts as a hit.
