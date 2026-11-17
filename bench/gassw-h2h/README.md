# GA-SSW feasibility

Checked 2026-11-17 for Liu, Liu, and Shang, “Accelerated Global
Potential Energy Landscape Discovery: A Hybrid Genetic Algorithm and
Stochastic Surface Walking Approach”, JCTC 2026,
<https://doi.org/10.1021/acs.jctc.6c01078>.

## Result

The paper-specific GA-SSW implementation is obtainable without an author
request: the paper's Data Availability Statement links a precompiled binary
archive and an examples archive. It is not public source and it has no stated
software license. The requested head-to-head was not run because the released
program has no controllable random seed. Labelling twelve nondeterministic
runs as seeds 2000--2011 would not be the specified seeded comparison, and
adding seed control would require changing the released implementation for
which no source or modification license is supplied.

No Snakemake rule, ASV benchmark, summarizer, `h2h.json` entry, adapter, or
benchmark row was added.

## Paper and Supporting Information

The authors' publication list has separate PDF and SI links for item 244:

- publication list: <https://zpliu.fudan.edu.cn/publication/list.htm>
- paper PDF: <http://www.lasphub.com/publication/244.pdf>
- Supporting Information: <http://www.lasphub.com/publication/244-SI.pdf>

The main paper's Data Availability Statement says:

> The precompiled GA-SSW package is available for download at
> http://www.lasphub.com/supportings/GA-SSW_program.tgz. Accompanying
> example cases demonstrating global minima searches across diverse systems
> and the transition pathway identification for the LJ38 cluster can be found
> at http://www.lasphub.com/supportings/GA-SSW_examples.tgz.

The SI contains method details, test parameters, figures, and evaluation-count
statistics. It contains no source-code archive, repository URL, or software
license.

## Directly downloadable artifacts

Both paper links returned the archives directly, without authentication:

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| <http://www.lasphub.com/supportings/GA-SSW_program.tgz> | 46,973,457 | `7b1ec89c0d8209a54272ee5b42c64be0326b657d5b7ddcd0483113d9332bb660` |
| <http://www.lasphub.com/supportings/GA-SSW_examples.tgz> | 116,148,327 | `2d75b039a02f15dc86d84b038773c9c755a5f2c67ea31608301952da956a33c9` |

`GA-SSW_program.tgz` contains:

- `sgn.jar`, the precompiled Java GA-SSW/SGN program;
- `lasp`, a precompiled dynamically linked x86-64 Linux executable;
- `check_env.sh`; and
- `README.md`.

The bundled README says the supplied LASP executable expires on 2027-05-01.
It requires Java 9 or newer and Intel MPI. The binary links against
`libmpi.so.12` and `libmpifort.so.12`. The unmodified LJ75 example reached
LASP evaluation successfully on this VM with Java 21 and Intel MPI 2021.19.

Neither archive contains `LICENSE`, `COPYING`, `NOTICE`, a copyright-license
grant, or source for GA-SSW. Consequently the paper-specific binaries are
publicly downloadable but no software license is stated; ordinary copyright
defaults apply. They must not be redistributed in this repository.

## LASP access and licensing

The official LASP introduction
<http://laspmol.lasphub.com/Intro_eng.asp> lists two distributions:

- an academic version “with expiry date and limited NN library access”; and
- a professional version “to purchase”, by contacting
  `laspmaster@lasphub.com`.

Its registration page is <http://laspmol.lasphub.com/reg.asp>.

The LASP 3.7 review
<https://pmc.ncbi.nlm.nih.gov/articles/PMC11672538/> states that LASP is
copyrighted and commercially available. The paper-specific archive avoids an
author request for this test binary, but it does not add a source or
modification license.

## Package and repository searches

No GA-SSW source distribution was found:

- GitHub exact DOI search:
  <https://github.com/search?q=%2210.1021%2Facs.jctc.6c01078%22&type=code>
- GitHub exact title search:
  <https://github.com/search?q=%22Accelerated+Global+Potential+Energy+Landscape+Discovery%22&type=code>
- GitHub repository search:
  <https://github.com/search?q=%22GA-SSW%22+LASP&type=repositories>
- Zenodo exact DOI search:
  <https://zenodo.org/search?q=%2210.1021%2Facs.jctc.6c01078%22>
- Zenodo API query, which returned zero records:
  <https://zenodo.org/api/records/?q=%2210.1021%2Facs.jctc.6c01078%22&size=20>

PyPI has no package for this optimizer:

- <https://pypi.org/project/lasp/> is the unrelated Linear Attention Sequence
  Parallelism package.
- <https://pypi.org/project/lasp-ase/> is an ASE calculator wrapper, not LASP
  or GA-SSW. Its source is MIT-licensed at
  <https://github.com/renpj/lasp_ase>, but its installation instructions say
  that the LASP executable and potentials must be downloaded separately and
  require a license.

Two GitHub repositories surfaced in broader LASP/global-optimization searches,
but neither is the paper's GA-SSW implementation:

- <https://github.com/QuantumMisaka/LASP_autotrain_lib> contains LASP
  training-workflow helpers.
- <https://github.com/fil-end/global_optimization_lib> is a separate global
  optimization helper library.

## Why the specified benchmark is not executable as released

The documented `configure.non` fields include GA population and iteration
counts, SSW step counts, temperature, parallelism, and the LASP path. There is
no random-seed field. Inspection of `configure.ConfigureEntries` in the
released `sgn.jar` confirms that no seed setting is parsed. Application
bytecode constructs `java.util.Random` with its no-argument constructor and
also calls `Math.random()`. Therefore the program cannot run the required
seeds 2000--2011 reproducibly through its public interface.

The user guide also documents only `CalSoft=lasp`. The supplied LJ example
uses LASP's own `potential lj`; it does not call rgpot. LASP has an external
potential mode, but the only supplied external script is the
`IfSSWGaussian=true` Gaussian adapter. Making GA-SSW use rgpot would require a
new adapter and call-accounting harness. That work cannot solve the missing
seed control without modifying the closed GA-SSW implementation.

For these reasons, the obtainable binary does not support the requested real,
seeded, same-rgpot head-to-head as released.
