These fixtures are copied from `origin/bench/blj-gmin` so the binary-LJ CCD
unit test runs on a clean checkout without the benchmark branch.

Provenance:
- source branch: `origin/bench/blj-gmin`
- source commit: `8de2a95c3ff3eb2b173b88b4a8cb48c144cbfe19`
- source path: `bench/blj-gmin/ccd/`
- copied files: `blj105n38`, `blj105n55`, `blj115n38`, `blj115n55`,
  `blj130n38`, `blj130n55`, `blj130n98`, `blj130n98-original`,
  `sources.csv`

To refresh from the same source:

```bash
for f in blj105n38 blj105n55 blj115n38 blj115n55 blj130n38 blj130n55 blj130n98 blj130n98-original sources.csv; do
  git show origin/bench/blj-gmin:bench/blj-gmin/ccd/$f > tests/fixtures/blj-ccd/$f
done
```
