# LJ55 driver

The AGOX arm is `python3 scripts/lj_board/run_arm.py agox 55 <seed>`.
The anneal arm is `lj55_board_arm <seed> <budget>`.

AGOX version 3.11.1.
Wheel `agox-3.11.1-cp312-cp312-linux_x86_64.whl`.
sha256 `7f6a4939d981e1826c8ca4243279d9271c70313e1785b7873d1ae91f1ce8ab47`.

## LJ98

The current arm is `python3 scripts/lj_board/run_arm.py agox 98 <seed>`.
The documented arm is `python3 scripts/lj_board/run_arm.py agox-documented 98 <seed>`.
Both arms pin AGOX 3.11.1, source `0a5136605161aaff2610584cd93ae6e9564fda46`,
wheel sha256 `f8b4f49b0e5a053c1b69205633c8b19523cd188b4001a12f093aaaa7f6250457`.
The documented arm keeps the stock box-force adjustment.
Seeds 6400-6405, budget 4,000,000, host bench, reference -543.665361.
Current AGOX hit 0 of 6, median wall 7031.920068 s.
Documented AGOX hit 0 of 6, median wall 8399.304438 s.
Anneal at 30a4ea8801236aba5175eea07b07e2424a056bd7 hit 2 of 6, median wall 81.334000 s.
Type-7 median wall ratio, current AGOX over anneal: 87.498181.
Type-7 median wall ratio, anneal over current AGOX: 0.011434.
Type-7 median wall ratio, documented AGOX over anneal: 103.653286.
Type-7 median wall ratio, anneal over documented AGOX: 0.009648.
The rows are in `bench/agox-lj98`.
