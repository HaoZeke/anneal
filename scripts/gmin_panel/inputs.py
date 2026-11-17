# Copyright (c) anneal developers
"""Per-size GMIN data for the paper panel.

The old panel pasted the LJ38 teaching file (TEMPERATURE 1.0, STEP 0.40
0.51, NEWRESTART 40, no symmetrisation) onto every system. Each block
below is the published search for that system. STEPS is long so the
shared call budget is the stop. TARGET is omitted because the harness
stops on H2H_TARGET at the same tolerance. DUMPINT, MONITOR and
TRACKDATA only schedule output, so they are left out. A RANSEED line is
appended by the runner.

Sources, one system at a time:

- LJ38. OptBench, Lennard-Jones 38, entry gmin-csm, Jacob Stevenson and
  David Wales, 6 August 2014, input gmin-csm.tgz. Mean 6.421e3 force
  calls, against 2.271e4 for gmin-symmetrise and 2.660e5 for gmin-nosym
  (https://optbench.org/globalopt.html). Oakley, Johnston and Wales,
  Phys. Chem. Chem. Phys. 15, 3965 (2013), Table 1, also has CSM ahead
  of core orbits and of plain basin-hopping on LJ38.
- LJ55. Oakley, Johnston and Wales, ibid., section 4.1: core-orbit
  symmetrisation is the better of the two basin-hopping methods they
  report, and the optimal NEWRESTART is around 60. No OptBench file is
  deposited for LJ55. Temperature, step, quench and the SYMMETRISE
  tolerances are the deposited core-orbit file for LJ38 (OptBench
  gmin-symmetrise, Stevenson and Wales, 6 August 2014). Only NEWRESTART
  is Oakley's LJ55 value.
- LJ75. OptBench, Lennard-Jones 75, entry gmin, David Wales, 9 January
  2014, input gmin.tgz. Core-orbit symmetrisation, NEWRESTART 4,
  TEMPERATURE 3.5. Oakley, Johnston and Wales, ibid., put the optimum
  near 5.
- LJ98. OptBench, Lennard-Jones 98, entry gmin-symmetrise, David Wales,
  3 October 2022, input gmin-symmetrise.tgz. Mean 2.495e5 energy
  evaluations, against 5.989e7 for gmin-nosym.
- LJ102, LJ103, LJ104. No OptBench file and no Oakley table. The
  core-orbit tolerances are the ones Oakley, Johnston and Wales, ibid.,
  section 5, call transferable, and the nearest published multi-funnel
  file is the LJ98 symmetrise deposit, used unchanged.
- LJ150. OptBench, Lennard-Jones 150, entry gmin-symmetrise, David
  Wales, 3 October 2022. Mean 1.276e5 energy evaluations, against
  6.960e5 for gmin-nosym.
- Au55. The corrected Gupta block, not retuned. RADIUS 11.829521 is
  GMIN's io1.f90 nearest-neighbour container at r0 2.884. STEP 1.027741
  0.51 and AVOID 10.277408 20 F are the LJ38 teaching distances times
  r0/2**(1/6); F keeps the taboo list and does not reseed. TEMPERATURE
  0.1 eV is the Wales group MGupta example. The Au-Au row is the Darby
  row used by scripts/metal_h2h/enter_gold.sh.
- Morse. The MORSE keyword selects GMIN's Morse pair and sets rho
  (GMIN documentation). The pair is in the same reduced units as
  Lennard-Jones. Doye and Wales, J. Chem. Soc., Faraday Trans. 93, 4233
  (1997), search Morse clusters by that basin-hopping. rho = 6 uses the
  Lennard-Jones block of the same size, or, at 80 atoms, Oakley's LJ74
  NEWRESTART of about 75 on the LJ55 core-orbit block, because LJ74 is
  the published "typical" cluster of this size and no M80 file is
  deposited. rho = 14 uses the LJ75 OptBench block: short-range Morse is
  the multi-funnel case, and LJ75 is the published multi-funnel deposit.
"""

from __future__ import annotations

# OptBench gmin-csm, the best published LJ38 entry.
LJ38 = """\
TIGHTCONV 1.0D-5
CENTRE
EDIFF 0.003
SORT
SAVE 1
MAXIT 1000 1000
STEPS 90000000 1.0
TEMPERATURE 3.0
NEWRESTART 30
AVOID 1.25 18
STEP 0.43 0.52
SLOPPYCONV 0.002
PERMDIST 0.005
SYMMETRISECSM 1 CI 1.0D-6 30 1 50
"""

# OptBench LJ38 gmin-symmetrise, with NEWRESTART 60 from Oakley for LJ55.
LJ55 = """\
TIGHTCONV 1.0D-5
CENTRE
EDIFF 0.003
SORT
SAVE 1
MAXIT 1000 1000
STEPS 90000000 1.0
TEMPERATURE 3.2
NEWRESTART 60
AVOID 1.25 20
STEP 0.43 0.52
SLOPPYCONV 0.002
PERMDIST 0.005
SYMMETRISE 1 0.04 0.03 0.09 0.09 0.4 18 0.4 1.7
"""

# OptBench LJ75 gmin.
LJ75 = """\
TIGHTCONV 1.0D-5
CENTRE
EDIFF 0.003
SORT
SAVE 1
MAXIT 1000 1000
STEPS 90000000 1.0
TEMPERATURE 3.5
NEWRESTART 4
AVOID 3.8 12
STEP 0.43 0.51
SLOPPYCONV 0.007
PERMDIST 0.0175
DGUESS 0.1
UPDATES 4
MAXBFGS 0.40
SYMMETRISE 1 0.01 0.05 0.3 0.012 0.6 4 0.27 2.4
"""

# OptBench LJ98 gmin-symmetrise. Also the block for LJ102, LJ103 and LJ104.
LJ98 = """\
TIGHTCONV 1.0D-5
CENTRE
EDIFF 0.003
SORT
SAVE 1
MAXIT 1000 1000
STEPS 90000000 1.0
FIXBOTH
PERMDIST 0.01
TEMPERATURE 1.3
NEWRESTART 6
STEP 0.46 0.47
SLOPPYCONV 0.007
SYMMETRISE 1 0.01 0.05 0.3 0.1 0.6 3 0.3 2.1
"""

# OptBench LJ150 gmin-symmetrise.
LJ150 = """\
TIGHTCONV 1.0D-5
CENTRE
EDIFF 0.003
SORT
SAVE 1
MAXIT 1000 1000
STEPS 90000000 1.0
FIXBOTH
PERMDIST 0.01
TEMPERATURE 1.0
NEWRESTART 7
STEP 0.415 0.45
SLOPPYCONV 0.007
SYMMETRISE 1 0.01 0.05 0.3 0.1 0.6 5 0.3 2.1
"""

# LJ55 core-orbit block with Oakley's LJ74 NEWRESTART of about 75.
M80_LONG = """\
TIGHTCONV 1.0D-5
CENTRE
EDIFF 0.003
SORT
SAVE 1
MAXIT 1000 1000
STEPS 90000000 1.0
TEMPERATURE 3.2
NEWRESTART 75
AVOID 1.25 20
STEP 0.43 0.52
SLOPPYCONV 0.002
PERMDIST 0.005
SYMMETRISE 1 0.04 0.03 0.09 0.09 0.4 18 0.4 1.7
"""

# Corrected Gupta block from the Au55 review. The quench and NEWRESTART 40
# are the LJ38 teaching file. The container is GMIN's io1.f90 nearest-
# neighbour radius at the Darby r0, the step and the avoid distance are
# the teaching file's sigma values times r0/2**(1/6), and AVOID's F keeps
# the taboo list without the random reseed. Temperature is 0.1 eV from
# the Wales group MGupta example. Do not retune this block.
def _gupta_radius(n: int = 55, r0: float = 2.884) -> float:
    return (2.0 + (3.0 * n / 17.77153175) ** (1.0 / 3.0)) * r0


def _gupta_scale(r0: float = 2.884) -> float:
    return r0 / 2.0 ** (1.0 / 6.0)


AU55 = f"""\
SLOPPYCONV 5.0D-3
TIGHTCONV 1.0D-3
EDIFF 0.003
MAXIT 600 500
SAVE 2
CENTRE
SORT
RADIUS {_gupta_radius():.6f}
TEMPERATURE 0.1
STEPS 90000000 1.0
STEP {_gupta_scale() * 0.40:.6f} 0.51
NEWRESTART 40
AVOID {_gupta_scale() * 4.0:.6f} 20 F
"""

# system -> (body, morse rho or 0). Au55 is filled by the runner.
BLOCKS = {
    "lj38": LJ38,
    "lj55": LJ55,
    "lj75": LJ75,
    "lj98": LJ98,
    "lj102": LJ98,
    "lj103": LJ98,
    "lj104": LJ98,
    "lj150": LJ150,
    "m55r6": LJ55,
    "m55r14": LJ75,
    "m80r6": M80_LONG,
    "m80r14": LJ75,
}

MORSE_RHO = {
    "m55r6": 6.0,
    "m55r14": 14.0,
    "m80r6": 6.0,
    "m80r14": 14.0,
}

# Citation key printed by the runner and the README.
CITATION = {
    "lj38": "OptBench LJ38 gmin-csm, Stevenson and Wales, 6 Aug 2014; Oakley, Johnston and Wales, PCCP 15, 3965 (2013), Table 1",
    "lj55": "Oakley, Johnston and Wales, PCCP 15, 3965 (2013), section 4.1, NEWRESTART about 60; tolerances from OptBench LJ38 gmin-symmetrise, Stevenson and Wales, 6 Aug 2014",
    "lj75": "OptBench LJ75 gmin, Wales, 9 Jan 2014; Oakley, Johnston and Wales, PCCP 15, 3965 (2013), NEWRESTART about 5",
    "lj98": "OptBench LJ98 gmin-symmetrise, Wales, 3 Oct 2022",
    "lj102": "OptBench LJ98 gmin-symmetrise, Wales, 3 Oct 2022; no deposit for this size, Oakley, Johnston and Wales, PCCP 15, 3965 (2013), section 5, transferable core-orbit tolerances",
    "lj103": "OptBench LJ98 gmin-symmetrise, Wales, 3 Oct 2022; no deposit for this size, Oakley, Johnston and Wales, PCCP 15, 3965 (2013), section 5, transferable core-orbit tolerances",
    "lj104": "OptBench LJ98 gmin-symmetrise, Wales, 3 Oct 2022; no deposit for this size, Oakley, Johnston and Wales, PCCP 15, 3965 (2013), section 5, transferable core-orbit tolerances",
    "lj150": "OptBench LJ150 gmin-symmetrise, Wales, 3 Oct 2022",
    "au55": "GMIN io1.f90 nearest-neighbour container, RADIUS 11.829521 A at r0 2.884 A; STEP 1.027741 0.51 and AVOID 10.277408 20 F, the LJ38 teaching distances times r0/2**(1/6); TEMPERATURE 0.1 eV from https://www-wales.ch.cam.ac.uk/examples/GMIN/MGupta.tar.bz2; Au-Au row of scripts/metal_h2h/enter_gold.sh",
    "m55r6": "GMIN MORSE keyword; LJ55 block, because rho = 6 is the Lennard-Jones well in the same reduced units (Doye and Wales, Faraday Trans. 93, 4233 (1997))",
    "m55r14": "GMIN MORSE keyword; OptBench LJ75 gmin block, the published multi-funnel core-orbit file, for short-range Morse",
    "m80r6": "GMIN MORSE keyword; LJ55 core-orbit block with NEWRESTART 75, Oakley's optimum for the typical cluster LJ74 (PCCP 15, 3965 (2013), section 4.1)",
    "m80r14": "GMIN MORSE keyword; OptBench LJ75 gmin block, the published multi-funnel core-orbit file, for short-range Morse",
}


def body(system: str) -> str:
    if system not in BLOCKS and system != "au55":
        raise KeyError(system)
    if system == "au55":
        return AU55
    text = BLOCKS[system]
    if system in MORSE_RHO:
        text += f"MORSE {MORSE_RHO[system]:.1f}\n"
    return text
