# Standalone minima hopping against anneal

Package `minimahopping` 1.1.3 revision `ff7e32bbd9a9b235f5f2a1dec11b311bf874300d`.
ASE 3.29.0. sqnm revision `4241692173cccb42a9ae05974bfd62285ec2cd7b`.
GitLab archive sha256 `dd103f13c316a68acbad1c5b3605dfeb2cb7ccb3f46f01641ed556e97c78a8f7`.
sdist sha256 `a13b444510a9e480669bf59fc15ac8dc7de03ae6b70ea5990f1f4a542874db20`.
wheel sha256 `ce156752b89dfed82fc81422168713f5f81c01cbc0445a47a83b3dfdad6dac35`.

The project name minimahopping is not on PyPI. The same HTTP 404 is returned for minima-hopping, minima_hopping, ase-mh, ase_mh, and python-minima-hopping. The pin is the GitLab tag 1.1.3.

Release 1.1.3 calls MaxwellBoltzmannDistribution with communicator='serial'. That keyword is accepted through ASE 3.28.0 and removed in ASE 3.29.0. This harness pins ASE 3.29.0 and rewrites the keyword to comm=DummyMPI(), the replacement ASE 3.28 applies for communicator='serial'.

Base `5eb79cdcb0a986984dc7b487e5a3cc93224246f9` tree `7a2ba49b37241a8fe42b2c45c226374de6d92274`.
Rows revision `d1def53122d913304825c490d516b4adad705ab7` tree `53dc6574b0af3183a48a9f3226c4e945df8b9b29`.
Harness revision `34bd3b2215e056a45c02633669ae99e842c0d7ed` tree `9c065caac5f36a4ab4a1e264bb95586d41597166`.
Harness revision `37987b38d5c557794a999cff0b47f4bf728f37c6` tree `b9574174d997e43e9d4323f05051fc3acd21daf5`.

Locked stack:

- anneal-core 0.12.0 
- readcon-core 0.14.8 registry+https://github.com/rust-lang/crates.io-index
- rgmin 0.5.0 registry+https://github.com/rust-lang/crates.io-index
- rgpot-core 3.5.1 registry+https://github.com/rust-lang/crates.io-index
- rgpot-pair 3.7.0 git+https://github.com/OmniPotentRPC/rgpot?rev=8efdd246322d1f32bc1bcbd7e5dfdd7a4237960e#8efdd246322d1f32bc1bcbd7e5dfdd7a4237960e

A hit is a quenched energy at or below the Cambridge minimum plus 0.001.
Medians and quartiles are Hyndman-Fan type 7.
Call medians and wall times are the stored values, including a capped seed at the calls and wall it reached.
The Mann-Whitney test on calls ranks every miss, including stop=error, as positive infinity.
The Mann-Whitney test on wall uses the stored wall.
Fisher p is the two-sided exact test on hit counts.
Mann-Whitney p is the two-sided normal approximation with the tie and continuity corrections, anneal as the left sample.
Bonferroni uses m=4, the number of complete head-to-head cells, separately for each test.
The same-binary Lennard-Jones 38 pair is the noise band and is not in that family.

## Head to head

### au38

anneal: n=12 hits=9 calls median 292454 IQR 114694 wall median 5.02915 IQR 2.05075 inside median 3.68178 outside median 1.35049 outside per call median 4.711e-06.
mhpkg: n=12 hits=0 calls median 400000 IQR 0 wall median 286.712 IQR 5.50435 inside median 5.59831 outside median 281.098 outside per call median 0.000703556.
Fisher p 0.000336519 adjusted 0.00134608. Mann-Whitney calls U 18 p 0.000383392 adjusted 0.00153357. Mann-Whitney wall U 11 p 0.000477682 adjusted 0.00191073.
Arm order counts: slot 0 anneal 6, slot 0 mhpkg 6, slot 1 anneal 6, slot 1 mhpkg 6.

### lj38

anneal: n=12 hits=9 calls median 111061 IQR 187804 wall median 0.560302 IQR 1.01314 inside median 0.261034 outside median 0.294295 outside per call median 2.5675e-06.
mhpkg: n=12 hits=10 calls median 211152 IQR 138977 wall median 178.012 IQR 119.705 inside median 0.673687 outside median 177.339 outside per call median 0.000798413.
Fisher p 1 adjusted 1. Mann-Whitney calls U 49 p 0.191987 adjusted 0.767948. Mann-Whitney wall U 0 p 3.65846e-05 adjusted 0.000146338.
Arm order counts: slot 0 anneal 6, slot 0 mhpkg 6, slot 1 anneal 6, slot 1 mhpkg 6.

### m80r14

anneal: n=12 hits=7 calls median 2.37423e+06 IQR 2.08602e+06 wall median 83.1821 IQR 73.5629 inside median 63.084 outside median 20.0981 outside per call median 8.9425e-06.
mhpkg: n=12 hits=0 calls median 4e+06 IQR 0 wall median 5240.77 IQR 261.091 inside median 101.278 outside median 5140.56 outside per call median 0.00128514.
Fisher p 0.00457666 adjusted 0.0183066. Mann-Whitney calls U 30 p 0.00285559 adjusted 0.0114223. Mann-Whitney wall U 0 p 3.65846e-05 adjusted 0.000146338.
Arm order counts: slot 0 anneal 6, slot 0 mhpkg 6, slot 1 anneal 6, slot 1 mhpkg 6.

### m80r6

anneal: n=12 hits=12 calls median 56621 IQR 53175.5 wall median 2.28602 IQR 2.07811 inside median 1.70698 outside median 0.579041 outside per call median 1.1174e-05.
mhpkg: n=12 hits=12 calls median 238030 IQR 91900.8 wall median 330.648 IQR 136.699 inside median 7.49785 outside median 323.15 outside per call median 0.00132728.
Fisher p 1 adjusted 1. Mann-Whitney calls U 24 p 0.00609895 adjusted 0.0243958. Mann-Whitney wall U 0 p 3.65846e-05 adjusted 0.000146338.
Arm order counts: slot 0 anneal 6, slot 0 mhpkg 6, slot 1 anneal 6, slot 1 mhpkg 6.

## Same binary, Lennard-Jones 38

lj38 aa0: n=12 hits=9 calls median 111061 IQR 187804 wall median 0.518378 IQR 0.87205.
lj38 aa1: n=12 hits=9 calls median 111061 IQR 187804 wall median 0.519957 IQR 0.879685.
Paired aa1 minus aa0 calls median 0 IQR 0 max abs 0.
Paired aa1 minus aa0 wall median 0.0015785 IQR 0.00251925 max abs 0.007633.

## Rigid TIP4P

The minimahopping package moves Cartesian ASE atoms. Its documented constraint support is ASE FixAtoms, with an explicit warning that other ASE constraints are untested. biomode is imported by the softening step and never called, and it does not integrate a rigid body. Rigid TIP4P in this repository is a 6-coordinate angle-axis chart per molecule (GminTip4p). There is no supported rigid-molecule interface, so the cell is not run.

## Shard

Au38 package seeds 2003 and 2006 stopped because the geometry optimizer raised `LinAlgError: Eigenvalues did not converge`. Those rows have `stop=error` and `hit=false`, at 294781 and 1769 calls. The other ten Au38 package seeds used the 400000-call budget. The medians above use the stored calls.

Au38 seeds 2003 through 2011 were measured with harness `37987b38d5c557794a999cff0b47f4bf728f37c6`. Every other row uses harness `34bd3b2215e056a45c02633669ae99e842c0d7ed`. The session id is `fde2aa71b65f4eeb`.

`git format-patch 5eb79cdcb0a986984dc7b487e5a3cc93224246f9..34bd3b2215e056a45c02633669ae99e842c0d7ed` has sha256 `a542ced35eea1cabff90ac65be84a5d75844ee820cd8574f54221ce8db87de54`.

Call medians above are the charged counts. `calls_to_hit` is the first quenched record at the threshold and can be smaller than the charged count.

## Anneal arm across the two harness revisions

`git diff --name-only 34bd3b2215e056a45c02633669ae99e842c0d7ed 37987b38d5c557794a999cff0b47f4bf728f37c6` lists `scripts/ase_panel/mhpkg.py` and `bench/ase-mhpkg/README.md`. The Python diff is the handler in `run_mhpkg` that records an optimizer abort, and the README paragraph that describes it. `run_anneal` and the example sources are not in that diff, so `37987b38d5c557794a999cff0b47f4bf728f37c6` does not change anything the anneal arm executes. The Au38 anneal seeds were not rerun.
