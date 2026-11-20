# Referee status for the JCTC manuscript

The manuscript, the committed records, and `scripts/paper_audit.py` are the
sources of truth for the status below.

## Closed review items

- The method, quench, move library, acceptance rule, call accounting, and
  structure-prior control are defined in the manuscript and Supporting
  Information.
- The GMIN, ASE, GOFEE, Sutton--Chen, and TIP4P comparisons state their
  protocols and keep their losses and ties in the tables.
- The manuscript has no rerun tags, rerun comments, unresolved markers, or
  future-work claims.
- The reproducibility archive is identified by its concept DOI, and the DOI
  macro is checked against `CITATION.cff`.
- Generated tables name their current committed sources. In particular, the
  TIP4P table is generated from `bench/tip4p-h2h/rows-h2h-r2.csv` and
  `bench/tip4p-h2h/wall-r2.csv`.
- The landfold figure is a vector redraw, and the withdrawn CREST figures and
  claims remain absent.
- The paper audit recomputes the reported values from committed records and
  rejects rewritten `rows.csv` lines.
- The ASE arms use their shipped constructor defaults on reduced-unit
  surfaces and use the recorded wall caps.
- The committed benchmark panels span several recorded code revisions.
- The depth-posterior comparison does not resolve an advantage over uniform
  selection on the recorded seeds.
- The TIP4P table uses the 120 committed r2 per-seed rows and the generated
  exact wall-time summary.
- Some recorded revisions are unreachable and remain classified in
  `bench/unreachable-revisions.csv`.

## Verification

The required Rust, Python, paper-audit, TeX, rendered-text, and whitespace
gates are rerun for the delivered series. Current gate results belong in the
handoff report rather than in this file, so they cannot become stale here.
